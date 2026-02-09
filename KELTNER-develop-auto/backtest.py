# impulse_backtest_signal_v4_gap_filter.py
#
# V4 CHANGES:
# 1. Gap Correction: 시가(Open) 보정 로직 적용 (현실적 체결)
# 2. Gap Filter: 시가와 지정가의 괴리가 0.3% 이상이면 진입 포기 (리스크 관리)
# 3. Advanced Stats: Profit Factor, Sharpe, Sortino, MDD Duration 등 펀드 지표 추가
# 4. Added: Avg R per Trade, Avg Win R, Avg Loss R (R값 상세 분석)

import os
import glob
import json
import shutil
import csv
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# signal_engine_v2.py가 같은 폴더에 있어야 합니다.
#from signal_engine_modified import SignalConfig, compute_latest_signals
from signal_engine import SignalConfig, compute_latest_signals

# ============================================================
# 1. 백테스트용 설정
# ============================================================
class BTConfig:
    DATA_DIR = "../data50"          # CSV 폴더
    TIMEFRAME = "15m"            # 파일명에서 구분용

    INITIAL_BALANCE  = 1000000
    ACCOUNT_RISK_PCT = 0.022     # 0.5% risk per trade

    WINDOW_SIZE = 500            # 항상 500봉 윈도우

    # 포지션 개수 제한
    MAX_LONG_POS   = 0
    MAX_SHORT_POS  = 6
    MAX_TOTAL_POS  = 12

    # 갭(Gap) 제한 설정 (0.3%)
    # 시가가 지정가보다 0.3% 이상 벌어져서 시작하면 진입 포기
    MAX_GAP_PCT    = 0.003

    # 수수료 / 슬리피지 (프로젝트 공통 룰)
    FEE_MAKER   = 0.0002   # 0.02% (익절 지정가)
    FEE_TAKER   = 0.0004   # 0.04% (시장가 진입/손절)
    
    # 슬리피지: 시장가 진입 시 불리한 가격 체결 반영
    SL_SLIPPAGE    = 0.0004   # 0.04%
    ENTRY_SLIPPAGE = 0.0003   # 0.03%
    
    PENDING_TTL_BARS = int(SignalConfig.FLAG_MAX_BARS)  # pending 만료(15m bars)

    # 손절/익절 동시 터치 시 우선순위 (보수적 : SL 우선)
    SL_FIRST = True


# ============================================================
# 2. CSV 로더 (BTC / 알트 분리)
# ============================================================
def load_data_grouped(data_dir: str, timeframe: str):
    """
    CSV 파일을 읽어서 타임프레임별로 그룹화하여 반환.
    return: { '15m': {'btc': btc_df, 'alts': {sym: df, ...}} }
    """
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        print(f"❌ '{data_dir}' 폴더에 CSV 파일이 없습니다.")
        return {}

    grouped_data = {}
    print(f"📂 데이터 로딩 중... ({len(csv_files)}개 파일)")

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        name_no_ext = os.path.splitext(filename)[0]
        parts = name_no_ext.split("_")

        if len(parts) < 3:
            continue

        symbol_raw = parts[0]     # INJ
        quote_raw  = parts[1]     # USDT
        tf_name    = parts[2]     # 15m

        if tf_name != timeframe:
            continue

        symbol = f"{symbol_raw}/{quote_raw}"

        if tf_name not in grouped_data:
            grouped_data[tf_name] = {"btc": None, "alts": {}}

        try:
            df = pd.read_csv(file_path)
            df.columns = [c.lower().strip() for c in df.columns]

            required = ["open_time", "open", "high", "low", "close", "volume"]
            if not all(col in df.columns for col in required):
                if "timestamp" in df.columns:
                    df.rename(columns={"timestamp": "open_time"}, inplace=True)
                else:
                    continue

            if pd.api.types.is_numeric_dtype(df["open_time"]):
                df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
            else:
                df["timestamp"] = pd.to_datetime(df["open_time"])

            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)

            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df.dropna(subset=["open", "high", "low", "close", "volume"], inplace=True)

            if "BTC" in symbol_raw.upper():
                grouped_data[tf_name]["btc"] = df
                print(f"✅ [{tf_name}] BTC 로드: {symbol}")
            else:
                grouped_data[tf_name]["alts"][symbol] = df

        except Exception as e:
            print(f"❌ {filename} 로딩 실패: {e}")

    return grouped_data


# ============================================================
# 3. 백테스트 엔진
# ============================================================
class BacktestEngine:
    def __init__(self, initial_balance: float):
        self.balance = initial_balance
        self.equity  = initial_balance
        self.initial_balance = initial_balance

        # 포지션 / 대기 주문
        self.positions = {}      # {sym: {side, qty, entry, sl, tp}}
        self.pending   = {}      # {sym: {side, qty, lim, sl, tp, created_t}}

        # 기록
        self.trade_history = []
        self.equity_curve  = []
        self.benchmark_curve = [] # (time, btc_price)

    # -----------------------------------------
    # 메인 루프
    # -----------------------------------------
    def run(self, btc_df_full: pd.DataFrame, alt_dict_full: dict[str, pd.DataFrame], signal_config=SignalConfig):
        # [기간 설정 로직]
        limit = 100000
        window = BTConfig.WINDOW_SIZE

        if limit > 0:
            print(f"✂️ 최근 {limit}개 캔들만 테스트합니다. (데이터 자르는 중...)")
            required_len = limit + window
            if len(btc_df_full) > required_len:
                btc_df_full = btc_df_full.tail(required_len)
            
            start_time = btc_df_full.index[0]
            new_alts = {}
            for sym, df in alt_dict_full.items():
                sliced_df = df[df.index >= start_time]
                if not sliced_df.empty:
                    new_alts[sym] = sliced_df
            alt_dict_full = new_alts

        idx = btc_df_full.index
        n   = len(idx)
        w   = BTConfig.WINDOW_SIZE

        total_steps = n - (w - 1)
        if total_steps <= 0:
            print("❌ 데이터 길이가 WINDOW_SIZE 보다 짧아서 백테스트 불가")
            return

        print(f"🚀 백테스트 시작 (총 {total_steps} 스텝, V4 Gap Filter + Detailed Stats)")

        for step, t in enumerate(range(w - 1, n), start=1):
            current_time = idx[t]

            if step == 1 or step % 500 == 0 or step == total_steps:
                pct = step / total_steps * 100
                print(f"🔄 진행률: {step}/{total_steps} ({pct:5.1f}%) - 현재 시각: {current_time}")

            # 1) 포지션 관리
            self.check_positions(current_time, alt_dict_full)
            # 2) 만료된 Pending 제거
            self.expire_pending_orders(t)
            # 3) Pending 진입 체크
            self.check_pending_orders(current_time, alt_dict_full)

            # 4) 시그널 생성
            btc_window = btc_df_full.iloc[t - w + 1 : t + 1]
            raw_data_dict = {}
            for sym, df_full in alt_dict_full.items():
                df_window = df_full[df_full.index <= current_time].tail(w)
                min_len = max(SignalConfig.WARMUP_PERIOD, SignalConfig.EMA_PERIOD * 2)
                if len(df_window) < min_len:
                    continue
                raw_data_dict[sym] = df_window

            if raw_data_dict:
                signals = compute_latest_signals(btc_window, raw_data_dict, signal_config)
                self.process_new_signals(current_time, signals, t)

            # 5) Equity 업데이트
            self.update_equity(current_time, alt_dict_full)
            self.update_benchmark(current_time, btc_df_full.loc[current_time, "close"])

        print("✅ 백테스트 완료")

    # -----------------------------------------
    # 포지션 SL / TP 체크
    # -----------------------------------------
    def check_positions(self, timestamp, alt_dict_full):
        closed = []
        for sym, pos in list(self.positions.items()):
            df = alt_dict_full.get(sym)
            if df is None or timestamp not in df.index:
                continue

            row  = df.loc[timestamp]
            high = row["high"]
            low  = row["low"]

            side  = pos["side"]
            sl    = pos["sl"]
            tp    = pos["tp"]
            entry = pos["entry"]
            qty   = pos["qty"]

            if side == 1:  # Long
                hit_sl = low <= sl
                hit_tp = high >= tp
            else:          # Short
                hit_sl = high >= sl
                hit_tp = low  <= tp

            exit_reason = None
            exit_price  = None
            fee         = 0.0

            if hit_sl and hit_tp:
                if BTConfig.SL_FIRST: hit_tp = False
                else: hit_sl = False

            if hit_sl:
                if side == 1: exit_price = sl * (1 - BTConfig.SL_SLIPPAGE)
                else: exit_price = sl * (1 + BTConfig.SL_SLIPPAGE)
                fee = exit_price * qty * BTConfig.FEE_TAKER
                exit_reason = "SL"
            elif hit_tp:
                exit_price = tp
                fee = exit_price * qty * BTConfig.FEE_MAKER
                exit_reason = "TP"

            if exit_reason is not None:
                if side == 1: pnl = (exit_price - entry) * qty
                else: pnl = (entry - exit_price) * qty
                
                final_pnl = pnl - fee
                self.balance += final_pnl

                risk_amt = abs(entry - sl) * qty
                if risk_amt == 0: risk_amt = 1e-9

                self.trade_history.append({
                    "time": timestamp,
                    "symbol": sym,
                    "side": "Long" if side == 1 else "Short",
                    "result": exit_reason,
                    "pnl": final_pnl,
                    "entry": entry,
                    "exit": exit_price,
                    "risk": risk_amt
                })
                closed.append(sym)

        for sym in closed:
            self.positions.pop(sym, None)

    # -----------------------------------------
    # pending 주문 → 돌파로 체결 (Gap Filter)
    # -----------------------------------------
    def check_pending_orders(self, timestamp, alt_dict_full):
        filled = []
        for sym, order in list(self.pending.items()):
            df = alt_dict_full.get(sym)
            if df is None or timestamp not in df.index:
                continue

            row = df.loc[timestamp]
            high = row["high"]
            low  = row["low"]
            open_p = row["open"]

            side = order["side"]
            lim  = order["lim"]
            sl   = order["sl"]
            tp   = order["tp"]
            qty  = order["qty"]

            is_filled = False
            base_price = None

            if side == 1: # Long
                if high >= lim:
                    base_price = max(open_p, lim)
                    gap_pct = (base_price - lim) / lim
                    if gap_pct > BTConfig.MAX_GAP_PCT: continue
                    is_filled = True
                    fill_price = base_price * (1.0 + BTConfig.ENTRY_SLIPPAGE)
            else: # Short
                if low <= lim:
                    base_price = min(open_p, lim)
                    gap_pct = (lim - base_price) / lim
                    if gap_pct > BTConfig.MAX_GAP_PCT: continue
                    is_filled = True
                    fill_price = base_price * (1.0 - BTConfig.ENTRY_SLIPPAGE)

            if is_filled:
                self.positions[sym] = {
                    "side": side, "qty": qty, "entry": fill_price, "sl": sl, "tp": tp
                }
                cost = fill_price * qty
                fee  = cost * BTConfig.FEE_TAKER
                self.balance -= fee
                filled.append(sym)

        for sym in filled:
            self.pending.pop(sym, None)

    # -----------------------------------------
    # pending 만료 처리
    # -----------------------------------------
    def expire_pending_orders(self, current_bar_index: int):
        expired = []
        for sym, order in list(self.pending.items()):
            created_t = order.get("created_t")
            if created_t is None: continue
            age = current_bar_index - int(created_t)
            if age >= int(BTConfig.PENDING_TTL_BARS):
                expired.append(sym)
        for sym in expired:
            self.pending.pop(sym, None)

    # -----------------------------------------
    # 새 시그널 처리
    # -----------------------------------------
    def process_new_signals(self, timestamp, signals: dict, bar_index: int):
        active_longs  = sum(1 for p in self.positions.values() if p["side"] == 1)
        active_shorts = sum(1 for p in self.positions.values() if p["side"] == -1)
        pending_longs  = sum(1 for o in self.pending.values() if o["side"] == 1)
        pending_shorts = sum(1 for o in self.pending.values() if o["side"] == -1)

        curr_longs  = active_longs + pending_longs
        curr_shorts = active_shorts + pending_shorts

        for sym, sig in signals.items():
            act = sig["act"]
            lim = sig["lim"]
            sl  = sig["sl"]
            tp  = sig["tp"]

            if act == 0 or np.isnan(lim) or np.isnan(sl) or np.isnan(tp): continue
            if sig["timestamp"] != timestamp: continue
            if sym in self.positions or sym in self.pending: continue

            if act == 1:
                if curr_longs >= BTConfig.MAX_LONG_POS: continue
            else:
                if curr_shorts >= BTConfig.MAX_SHORT_POS: continue
            
            if curr_longs + curr_shorts >= BTConfig.MAX_TOTAL_POS: break

            risk_per_trade = self.equity * BTConfig.ACCOUNT_RISK_PCT
            price_dist     = abs(lim - sl)
            if price_dist <= 0: continue
            qty = risk_per_trade / price_dist

            self.pending[sym] = {
                "side": int(act), "qty": qty, "lim": lim, "sl": sl, "tp": tp, "created_t": int(bar_index)
            }
            if act == 1: curr_longs += 1
            else: curr_shorts += 1

            if curr_longs + curr_shorts >= BTConfig.MAX_TOTAL_POS: break

    # -----------------------------------------
    # Equity 업데이트
    # -----------------------------------------
    def update_equity(self, timestamp, alt_dict_full):
        unrealized = 0.0
        for sym, pos in self.positions.items():
            df = alt_dict_full.get(sym)
            if df is None or timestamp not in df.index: continue
            curr_price = df.loc[timestamp]["close"]
            entry = pos["entry"]
            qty = pos["qty"]
            if pos["side"] == 1: pnl = (curr_price - entry) * qty
            else: pnl = (entry - curr_price) * qty
            unrealized += pnl

        self.equity = self.balance + unrealized
        self.equity = self.balance + unrealized
        self.equity_curve.append({"time": timestamp, "equity": self.equity})

    def update_benchmark(self, timestamp, btc_price):
        self.benchmark_curve.append({"time": timestamp, "price": btc_price})

    # =========================================================
    # 5. [수정됨] 성과 분석 (Total / Long / Short 분리)
    # =========================================================
    def get_summary_by_side(self):
        """
        Total, Long, Short 각각에 대한 성과 지표를 계산하여 딕셔너리로 반환
        """
        if not self.trade_history:
            return None

        # 1. 전체 데이터프레임 생성
        df_all = pd.DataFrame(self.trade_history)
        if "risk" not in df_all.columns:
            df_all["risk"] = 1.0 # fallback
        df_all["r_val"] = df_all.apply(lambda x: x["pnl"] / x["risk"] if x["risk"] > 0 else 0, axis=1)

        # 2. Side별 분리
        df_long  = df_all[df_all["side"] == "Long"].copy()
        df_short = df_all[df_all["side"] == "Short"].copy()

        # 3. 전체 Equity Curve (Total용 Real Curve)
        eq_curve_df = pd.DataFrame(self.equity_curve)
        eq_curve_df["time"] = pd.to_datetime(eq_curve_df["time"])
        eq_curve_df.set_index("time", inplace=True)
        # 일별 데이터로 리샘플링 (Total)
        daily_equity_total = eq_curve_df["equity"].resample("1D").last().dropna()

        # 4. Long/Short 전용 가상 Equity Curve 생성 (Realized PnL 기준)
        #    (Total Balance에서 해당 Side의 PnL 누적분만 반영)
        def make_side_equity(trades_df):
            if trades_df.empty:
                return pd.Series(dtype=float)
            
            # 시간순 정렬
            trades_df["time"] = pd.to_datetime(trades_df["time"])
            temp = trades_df.set_index("time").sort_index()
            
            # 일별 PnL 합계
            daily_pnl = temp["pnl"].resample("1D").sum().fillna(0)
            
            # 전체 기간에 대해 인덱스 맞추기 (빈 날짜는 0)
            full_idx = daily_equity_total.index
            daily_pnl = daily_pnl.reindex(full_idx, fill_value=0)
            
            # 초기 자본 + 누적 PnL
            return self.initial_balance + daily_pnl.cumsum()

        daily_equity_long  = make_side_equity(df_long)
        daily_equity_short = make_side_equity(df_short)

        # 5. 지표 계산 헬퍼 함수
        def calculate_metrics(trades_df, daily_eq_series):
            stats = {
                "Count": 0, "WinRate": 0.0, "PF": 0.0, 
                "AvgR": 0.0, "AvgWinR": 0.0, "AvgLossR": 0.0,
                "NetProfit": 0.0, "MDD": 0.0, "Sharpe": 0.0, "Sortino": 0.0
            }
            
            if trades_df.empty:
                return stats

            # --- Trade Metrics ---
            wins = trades_df[trades_df["pnl"] > 0]
            losses = trades_df[trades_df["pnl"] <= 0]
            
            stats["Count"] = len(trades_df)
            stats["WinRate"] = len(wins) / len(trades_df) * 100
            
            gross_win = wins["pnl"].sum()
            gross_loss = abs(losses["pnl"].sum())
            stats["NetProfit"] = trades_df["pnl"].sum()
            stats["PF"] = gross_win / gross_loss if gross_loss > 0 else 99.99
            
            stats["AvgR"] = trades_df["r_val"].mean()
            stats["AvgWinR"] = wins["r_val"].mean() if not wins.empty else 0
            stats["AvgLossR"] = losses["r_val"].mean() if not losses.empty else 0

            # --- Time Series Metrics (MDD, Sharpe) ---
            if daily_eq_series is None or len(daily_eq_series) < 2:
                return stats

            # MDD
            peak = daily_eq_series.cummax()
            dd = (daily_eq_series - peak) / peak
            stats["MDD"] = dd.min() * 100
            
            # Sharpe / Sortino
            daily_ret = daily_eq_series.pct_change().dropna()
            if len(daily_ret) > 1 and daily_ret.std() > 0:
                mean_ret = daily_ret.mean()
                std_ret  = daily_ret.std()
                annual_factor = np.sqrt(365)
                stats["Sharpe"] = (mean_ret / std_ret) * annual_factor
                
                downside = daily_ret[daily_ret < 0]
                if len(downside) > 0 and downside.std() > 0:
                    stats["Sortino"] = (mean_ret / downside.std()) * annual_factor

            return stats

        # 6. 각각 계산 수행
        res_total = calculate_metrics(df_all, daily_equity_total)
        res_long  = calculate_metrics(df_long, daily_equity_long)
        res_short = calculate_metrics(df_short, daily_equity_short)

        return {"Total": res_total, "Long": res_long, "Short": res_short}

    def print_summary(self, tf_name):
        results = self.get_summary_by_side()
        
        print(f"\n{'='*75}")
        print(f"📊 DETAILED BACKTEST REPORT [{tf_name}] (Initial: ${self.initial_balance:,.0f})")
        print(f"{'='*75}")

        if not results:
            print("❌ 거래 기록이 없습니다.")
            return

        # 헤더 출력
        # 항목명 | Total | Long | Short
        headers = ["Metric", "Total", "Long", "Short"]
        row_fmt = "{:<18} | {:>15} | {:>15} | {:>15}"
        
        print(row_fmt.format(*headers))
        print("-" * 75)

        # 표시할 키와 라벨 매핑
        metrics_order = [
            ("Count", "Trades"),
            ("WinRate", "Win Rate (%)"),
            ("NetProfit", "Net Profit ($)"),
            ("PF", "Profit Factor"),
            ("AvgR", "Avg R"),
            ("AvgWinR", "Avg Win R"),
            ("AvgLossR", "Avg Loss R"),
            ("MDD", "MDD (%)"),
            ("Sharpe", "Sharpe Ratio"),
            ("Sortino", "Sortino Ratio")
        ]

        for key, label in metrics_order:
            vals = []
            for category in ["Total", "Long", "Short"]:
                val = results[category][key]
                # 포맷팅
                if key in ["Count"]:
                    s = f"{val}"
                elif key in ["NetProfit"]:
                    s = f"{val:,.0f}"
                elif key in ["WinRate", "MDD"]:
                    s = f"{val:.2f}%"
                elif key in ["AvgR", "AvgWinR", "AvgLossR"]:
                    s = f"{val:.2f}R"
                else: # PF, Sharpe, Sortino
                    s = f"{val:.2f}"
                vals.append(s)
            
            print(row_fmt.format(label, *vals))
        
        print(f"{'='*75}")
        print(f"💰 Final Balance: ${self.balance:,.0f} (ROI: {(self.balance/self.initial_balance - 1)*100:.2f}%)")
        print(f"{'='*75}")

    def print_monthly_and_daily_returns(self):
        if not self.equity_curve: return

        eq_df = pd.DataFrame(self.equity_curve).copy()
        eq_df["time"] = pd.to_datetime(eq_df["time"])
        eq_df.set_index("time", inplace=True)
        
        daily_eq = eq_df["equity"].resample("1D").last().dropna()
        if len(daily_eq) < 2: return

        daily_ret = daily_eq.pct_change().dropna()
        avg_daily_ret_pct = float(daily_ret.mean() * 100.0)
        avg_daily_pnl_won = float(daily_eq.diff().dropna().mean())

        period = daily_eq.index.to_period("M")
        monthly = daily_eq.groupby(period).agg(first="first", last="last")
        monthly["pnl_won"] = monthly["last"] - monthly["first"]
        monthly["return_pct"] = (monthly["last"] / monthly["first"] - 1.0) * 100.0

        print(f"\n[📆 Monthly Returns]")
        print(f"{'-'*40}")
        for p, row in monthly.iterrows():
            print(f"{str(p)} | {row['return_pct']:6.2f}% | ${row['pnl_won']:,.0f}")
        
        print(f"{'-'*40}")
        print(f"📅 Avg Daily Return: {avg_daily_ret_pct:.3f}%")
        print(f"💵 Avg Daily PnL   : ${avg_daily_pnl_won:,.0f}")


def plot_backtest_result(engine, filename="backtest_plot.png"):
    """
    BTC 가격 변화와 Equity Curve를 시각화하여 저장
    """
    if not engine.equity_curve :
        return None

    df_eq = pd.DataFrame(engine.equity_curve)
    df_eq["time"] = pd.to_datetime(df_eq["time"])
    df_eq.set_index("time", inplace=True)
    
    if hasattr(engine, 'benchmark_curve') and engine.benchmark_curve:
        df_bench = pd.DataFrame(engine.benchmark_curve)
        df_bench["time"] = pd.to_datetime(df_bench["time"])
        df_bench.set_index("time", inplace=True)
    else:
        df_bench = None

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Axis 1: Equity
    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Equity ($)', color=color)
    ax1.plot(df_eq.index, df_eq['equity'], color=color, label='Strategy Equity')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Axis 2: Benchmark (BTC)
    if df_bench is not None:
        ax2 = ax1.twinx()  
        color = 'tab:orange'
        ax2.set_ylabel('BTC Price ($)', color=color)
        ax2.plot(df_bench.index, df_bench['price'], color=color, linestyle='--', alpha=0.7, label='BTC Price')
        ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f"Backtest Result: Equity vs BTC (Final: ${engine.balance:,.0f})")
    fig.tight_layout()
    
    try:
        plt.savefig(filename)
        plt.close()
        return filename
    except Exception as e:
        print(f"❌ Plot Saving Failed: {e}")
        plt.close()
        return None

# ============================================================
# 5. 결과 기록 (History Logging)
# ============================================================
def save_experiment_result(results, config_cls, main_df_cols=None, engine=None):
    if not results:
        return

    # 1. 저장 디렉토리 설정
    history_dir = "history"
    plots_dir = os.path.join(history_dir, "plots")
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 2. 로직 파일 백업 (Strategy Snapshot)
    logic_source = "signal_engine_modified.py"
    logic_backup_name = f"signal_logic_{timestamp_str}.py"
    logic_backup_path = os.path.join(history_dir, logic_backup_name)
    
    if os.path.exists(logic_source):
        shutil.copy(logic_source, logic_backup_path)
    else:
        logic_backup_name = "Not Found"

    # 3. 플롯 저장 (추가됨)
    plot_filename = f"plot_{timestamp_str}.png"
    plot_path = os.path.join(plots_dir, plot_filename)
    if engine:
        plot_backtest_result(engine, filename=plot_path)
    else:
        plot_filename = "No Chart"

    # 4. 주요 파라미터 추출
    params = {k: v for k, v in vars(config_cls).items() if k.isupper() and not k.startswith("_")}
    params_json = json.dumps(params, ensure_ascii=False)

    # 5. CSV 저장 데이터 구성
    total_res = results.get("Total", {})
    
    record = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "MDD (%)": f"{total_res.get('MDD', 0):.2f}",
        "Trade Count": total_res.get("Count", 0),
        "Win Rate (%)": f"{total_res.get('WinRate', 0):.2f}",
        "Avg R": f"{total_res.get('AvgR', 0):.4f}",
        "Net Profit ($)": f"{total_res.get('NetProfit', 0):.2f}",
        "Profit Factor": f"{total_res.get('PF', 0):.2f}",
        "Sharpe": f"{total_res.get('Sharpe', 0):.4f}",
        "Logic File": logic_backup_name,
        "Chart Image": plot_filename, # New Column
        "Parameters": params_json
    }

    csv_file = "backtest_history.csv"
    file_exists = os.path.isfile(csv_file)
    fieldnames = list(record.keys())
    
    try:
        with open(csv_file, mode='a', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)
        print(f"💾 [History] 결과가 저장되었습니다 (Chart: {plot_filename})")
    except Exception as e:
        print(f"❌ 결과 저장 실패: {e}")

# ============================================================
# 4. 실행부
# ============================================================
if __name__ == "__main__":
    grouped = load_data_grouped(BTConfig.DATA_DIR, BTConfig.TIMEFRAME)

    if not grouped or BTConfig.TIMEFRAME not in grouped:
        print("❌ 실행 불가: 데이터가 없습니다.")
    else:
        data    = grouped[BTConfig.TIMEFRAME]
        btc_df  = data["btc"]
        alt_dict= data["alts"]

        if btc_df is None or not alt_dict:
            print("❌ 실행 불가: BTC 또는 알트 데이터 없음")
        else:
            engine = BacktestEngine(BTConfig.INITIAL_BALANCE)
            engine.run(btc_df, alt_dict)
            engine.print_summary(BTConfig.TIMEFRAME)
            engine.print_monthly_and_daily_returns()
            
            # 결과 기록 (Results + Logic Backup + Chart)
            results = engine.get_summary_by_side()
            save_experiment_result(results, SignalConfig, engine=engine)