"""
indicator_perf_viz.py (SAFE VERSION: no look-ahead in feature analysis)

목표
- backtest.py의 백테스트 로직(진입/청산/수수료/슬리피지/제한/loop 순서)을 "그대로" 사용하면서
- 추가로 "지표값(특징) ↔ 트레이드 성과"의 상관관계/예측력을 분석해서 PNG + CSV로 저장

중요(미래 참조 편향 방지)
- 분석용 feature는 "체결 봉"의 종가를 쓰지 않습니다.
- 기본은 "signal_time(주문 생성 시점 = 시그널 계산 시점, 즉 해당 봉 close 이후)"의 feature만 사용합니다.
  → signal_engine이 종가(close) 기반으로 신호를 만들기 때문에, signal_time feature가 가장 일관되고 안전합니다.

사용법
  python indicator_perf_viz.py

산출물
  analysis_outputs/YYYYMMDD_HHMMSS/
    - trades_with_features.csv
    - feature_scores.csv
    - feature_scores_top20.png
    - binned_<feature>.csv/.png (top5)
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

# ✅ 기존 코드 import (백테스트 로직은 그대로 재사용)
import backtest as bt
import signal_engine as se


# ============================================================
# 0) 유틸
# ============================================================
def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _safe_corr(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    try:
        s = pd.Series(x)
        t = pd.Series(y)
        v = s.corr(t, method=kind)
        return float(v) if pd.notna(v) else 0.0
    except Exception:
        return 0.0


# ============================================================
# 1) 분석용 feature 계산 (signal_engine 지표 기반)
# ============================================================
def compute_features_from_window(df_window: pd.DataFrame, side: int) -> Dict[str, float]:
    """
    df_window: signal_time까지의 WINDOW_SIZE 캔들(완성된 봉들)
    side: 1(long), -1(short)
    """
    curr = se.add_indicators(df_window, se.SignalConfig)
    if len(curr) == 0:
        return {}

    row = curr.iloc[-1]
    close = float(row.get("close", np.nan))
    atr = float(row.get("atr", np.nan))
    adx = float(row.get("adx", np.nan))
    ema_trend = float(row.get("ema_trend", np.nan))
    kc_mid = float(row.get("kc_mid", np.nan))
    kc_upper = float(row.get("kc_upper", np.nan))
    kc_lower = float(row.get("kc_lower", np.nan))
    vol = float(row.get("volume", np.nan))

    feat: Dict[str, float] = {}
    feat["feat_close"] = close
    feat["feat_atr"] = atr
    feat["feat_adx"] = adx
    feat["feat_volume"] = vol

    feat["feat_atr_pct"] = (atr / close) if (close and close > 0) else np.nan
    feat["feat_ema_dist_pct"] = ((close - ema_trend) / close) if (close and close > 0) else np.nan

    kc_bw = (kc_upper - kc_lower)
    feat["feat_kc_bw"] = kc_bw
    feat["feat_kc_bw_pct"] = (kc_bw / close) if (close and close > 0) else np.nan
    feat["feat_kc_pos"] = ((close - kc_mid) / kc_bw) if (kc_bw and kc_bw != 0) else np.nan

    # 방향 고려 "돌파 강도"
    if side == 1:
        feat["feat_breakout_px"] = (close - kc_upper)
        feat["feat_breakout_atr"] = (close - kc_upper) / atr if (atr and atr > 0) else np.nan
    else:
        feat["feat_breakout_px"] = (kc_lower - close)
        feat["feat_breakout_atr"] = (kc_lower - close) / atr if (atr and atr > 0) else np.nan

    # 볼륨 ratio (20)
    v_ma = curr["volume"].rolling(20).mean().iloc[-1] if "volume" in curr.columns else np.nan
    feat["feat_vol_ma20"] = float(v_ma) if pd.notna(v_ma) else np.nan
    feat["feat_vol_ratio20"] = (vol / v_ma) if (pd.notna(v_ma) and v_ma > 0) else np.nan

    return feat


# ============================================================
# 2) “백테스트는 동일하게” + 분석 데이터만 수집하는 엔진
#    - run()의 순서/로직은 backtest.BacktestEngine.run과 동일한 구조로 복제
#    - 차이: signal_time feature 저장 + trade_history에 signal/entry time을 추가로 기록
# ============================================================
class EnhancedBacktestEngine(bt.BacktestEngine):
    def __init__(self, initial_balance: float):
        super().__init__(initial_balance)

        # (symbol, signal_time) -> feature dict
        self.signal_features: Dict[Tuple[str, pd.Timestamp], Dict[str, float]] = {}

        # pending 주문 생성 시각: sym -> signal_time
        self._pending_signal_time: Dict[str, pd.Timestamp] = {}

    # --- backtest.py의 process_new_signals 그대로 + signal_time 기록 ---
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

            if act == 0 or np.isnan(lim) or np.isnan(sl) or np.isnan(tp): 
                continue
            if sig["timestamp"] != timestamp:
                continue
            if sym in self.positions or sym in self.pending:
                continue

            if act == 1:
                if curr_longs >= bt.BTConfig.MAX_LONG_POS: 
                    continue
            else:
                if curr_shorts >= bt.BTConfig.MAX_SHORT_POS: 
                    continue

            if curr_longs + curr_shorts >= bt.BTConfig.MAX_TOTAL_POS: 
                break

            risk_per_trade = self.equity * bt.BTConfig.ACCOUNT_RISK_PCT
            price_dist     = abs(lim - sl)
            if price_dist <= 0: 
                continue
            qty = risk_per_trade / price_dist

            self.pending[sym] = {
                "side": int(act), "qty": qty, "lim": lim, "sl": sl, "tp": tp, "created_t": int(bar_index)
            }
            # ✅ signal_time 기록
            self._pending_signal_time[sym] = pd.Timestamp(timestamp)

            if act == 1: 
                curr_longs += 1
            else: 
                curr_shorts += 1

            if curr_longs + curr_shorts >= bt.BTConfig.MAX_TOTAL_POS: 
                break

    # --- backtest.py의 check_pending_orders 그대로 + signal/entry time 추가 기록 ---
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

            if side == 1:  # Long
                if high >= lim:
                    base_price = max(open_p, lim)
                    gap_pct = (base_price - lim) / lim
                    if gap_pct > bt.BTConfig.MAX_GAP_PCT:
                        continue
                    is_filled = True
                    fill_price = base_price * (1.0 + bt.BTConfig.ENTRY_SLIPPAGE)
            else:  # Short
                if low <= lim:
                    base_price = min(open_p, lim)
                    gap_pct = (lim - base_price) / lim
                    if gap_pct > bt.BTConfig.MAX_GAP_PCT:
                        continue
                    is_filled = True
                    fill_price = base_price * (1.0 - bt.BTConfig.ENTRY_SLIPPAGE)

            if is_filled:
                sig_time = self._pending_signal_time.get(sym)
                self.positions[sym] = {
                    "side": side, "qty": qty, "entry": fill_price, "sl": sl, "tp": tp,
                    # ✅ 분석용 메타
                    "entry_time": pd.Timestamp(timestamp),
                    "signal_time": sig_time,
                }
                cost = fill_price * qty
                fee  = cost * bt.BTConfig.FEE_TAKER
                self.balance -= fee
                filled.append(sym)

        for sym in filled:
            self.pending.pop(sym, None)
            self._pending_signal_time.pop(sym, None)

    # --- backtest.py의 check_positions 그대로 + signal/entry time을 trade_history에 기록 ---
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
                if bt.BTConfig.SL_FIRST:
                    hit_tp = False
                else:
                    hit_sl = False

            if hit_sl:
                if side == 1:
                    exit_price = sl * (1 - bt.BTConfig.SL_SLIPPAGE)
                else:
                    exit_price = sl * (1 + bt.BTConfig.SL_SLIPPAGE)
                fee = exit_price * qty * bt.BTConfig.FEE_TAKER
                exit_reason = "SL"
            elif hit_tp:
                exit_price = tp
                fee = exit_price * qty * bt.BTConfig.FEE_MAKER
                exit_reason = "TP"

            if exit_reason is not None:
                pnl = (exit_price - entry) * qty if side == 1 else (entry - exit_price) * qty
                final_pnl = pnl - fee
                self.balance += final_pnl

                risk_amt = abs(entry - sl) * qty
                if risk_amt == 0:
                    risk_amt = 1e-9

                self.trade_history.append({
                    "time": pd.Timestamp(timestamp),  # exit time (기존)
                    "symbol": sym,
                    "side": "Long" if side == 1 else "Short",
                    "result": exit_reason,
                    "pnl": final_pnl,
                    "entry": entry,
                    "exit": exit_price,
                    "risk": risk_amt,
                    # ✅ 추가
                    "entry_time": pos.get("entry_time"),
                    "signal_time": pos.get("signal_time"),
                })
                closed.append(sym)

        for sym in closed:
            self.positions.pop(sym, None)

    # --- 핵심: run()을 원본과 동일한 순서로 복제 + signal_time feature 저장만 추가 ---
    def run(self, btc_df_full: pd.DataFrame, alt_dict_full: dict[str, pd.DataFrame]):
        limit = 30000
        window = bt.BTConfig.WINDOW_SIZE

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
        w   = bt.BTConfig.WINDOW_SIZE

        total_steps = n - (w - 1)
        if total_steps <= 0:
            print("❌ 데이터 길이가 WINDOW_SIZE 보다 짧아서 백테스트 불가")
            return

        print(f"🚀 백테스트 시작 (총 {total_steps} 스텝, V4 Gap Filter + Detailed Stats + Feature Logging)")

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

            # 4) 시그널 생성 (원본과 동일)
            btc_window = btc_df_full.iloc[t - w + 1 : t + 1]
            raw_data_dict = {}
            min_len = max(se.SignalConfig.WARMUP_PERIOD, se.SignalConfig.TREND_EMA_PERIOD * 2)  # 안전
            for sym, df_full in alt_dict_full.items():
                df_window = df_full[df_full.index <= current_time].tail(w)
                if len(df_window) < min_len:
                    continue
                raw_data_dict[sym] = df_window

            if raw_data_dict:
                signals = se.compute_latest_signals(btc_window, raw_data_dict, se.SignalConfig)

                # ✅ 핵심: signal_time feature 저장 (look-ahead 없음)
                for sym, sig in signals.items():
                    act = sig.get("act", 0.0)
                    if act == 0:
                        continue
                    if sig.get("timestamp") != current_time:
                        continue
                    dfw = raw_data_dict.get(sym)
                    if dfw is None:
                        continue
                    side = 1 if act > 0 else -1
                    feat = compute_features_from_window(dfw, side=side)
                    if feat:
                        self.signal_features[(sym, pd.Timestamp(current_time))] = feat

                self.process_new_signals(current_time, signals, t)

            # 5) Equity 업데이트
            self.update_equity(current_time, alt_dict_full)


# ============================================================
# 3) 데이터셋/스코어링/플롯
# ============================================================
def build_trade_dataset(engine: EnhancedBacktestEngine) -> pd.DataFrame:
    if not engine.trade_history:
        return pd.DataFrame()

    trades = pd.DataFrame(engine.trade_history).copy()
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    trades["signal_time"] = pd.to_datetime(trades["signal_time"])
    trades["exit_time"] = pd.to_datetime(trades["time"])
    trades.drop(columns=["time"], inplace=True)

    # r, win
    trades["r_val"] = trades.apply(lambda x: (x["pnl"] / x["risk"]) if x["risk"] > 0 else 0.0, axis=1)
    trades["win"] = (trades["pnl"] > 0).astype(int)

    # =========================================================
    # ✅ NEW: 시간 기반 feature(킬존/세션 분석용)
    # - entry_time: 실제 포지션이 열린 시각(체결 시각)
    # - signal_time: 신호가 생성되어 pending이 만들어진 시각
    #
    # 주의: 단순 hour/dow 같은 "원-핫/순서형" 특성은 모델에 따라 왜곡될 수 있어
    #       함께 제공하는 sin/cos(주기 인코딩)도 같이 쓰는 것을 권장.
    # =========================================================
    def _add_time_features(df: pd.DataFrame, ts_col: str, prefix: str) -> None:
        if ts_col not in df.columns:
            return
        s = pd.to_datetime(df[ts_col], errors="coerce")
        # 기본 파생
        df[f"{prefix}_hour"] = s.dt.hour.astype("float")
        df[f"{prefix}_minute"] = s.dt.minute.astype("float")
        df[f"{prefix}_dow"] = s.dt.dayofweek.astype("float")  # Mon=0
        df[f"{prefix}_is_weekend"] = (s.dt.dayofweek >= 5).astype(int)

        # minute-of-day (0..1439)
        mod = (s.dt.hour * 60 + s.dt.minute).astype("float")
        df[f"{prefix}_minute_of_day"] = mod

        # 주기 인코딩 (hour, minute-of-day)
        # - hour: 24시간 주기
        # - minute_of_day: 1440분 주기
        two_pi = 2.0 * np.pi

        h = s.dt.hour.astype("float")
        df[f"{prefix}_hour_sin"] = np.sin(two_pi * (h / 24.0))
        df[f"{prefix}_hour_cos"] = np.cos(two_pi * (h / 24.0))

        df[f"{prefix}_mod_sin"] = np.sin(two_pi * (mod / 1440.0))
        df[f"{prefix}_mod_cos"] = np.cos(two_pi * (mod / 1440.0))

    # feature 이름은 분석 파이프라인과 맞추기 위해 'feat_'로 시작
    _add_time_features(trades, "entry_time", "feat_entry")
    _add_time_features(trades, "signal_time", "feat_signal")

    # signal_features join (핵심)
    feat_rows = []
    for (sym, st), feat in engine.signal_features.items():
        row = {"symbol": sym, "signal_time": pd.Timestamp(st)}
        row.update(feat)
        feat_rows.append(row)
    feats = pd.DataFrame(feat_rows)

    if feats.empty:
        return trades

    out = trades.merge(feats, on=["symbol", "signal_time"], how="left")
    return out


def score_features(df: pd.DataFrame, feature_cols: List[str], out_dir: str) -> pd.DataFrame:
    y_r = df["r_val"].to_numpy(dtype=float)
    y_w = df["win"].to_numpy(dtype=int)

    rows = []
    for col in feature_cols:
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

        m = np.isfinite(x) & np.isfinite(y_r)
        if m.sum() < 50:
            continue

        xr = x[m]
        yr = y_r[m]
        yw = y_w[m]

        pear = _safe_corr(xr, yr, "pearson")
        spear = _safe_corr(xr, yr, "spearman")

        auc = np.nan
        try:
            if len(np.unique(yw)) == 2 and np.nanstd(xr) > 1e-12:
                auc = roc_auc_score(yw, xr)
        except Exception:
            auc = np.nan

        auc_ts = np.nan
        try:
            if len(np.unique(yw)) == 2 and len(xr) >= 200:
                tss = TimeSeriesSplit(n_splits=5)
                aucs = []
                for _, te_idx in tss.split(xr):
                    y_te = yw[te_idx]
                    x_te = xr[te_idx]
                    if len(np.unique(y_te)) < 2:
                        continue
                    aucs.append(roc_auc_score(y_te, x_te))
                if aucs:
                    auc_ts = float(np.mean(aucs))
        except Exception:
            auc_ts = np.nan

        # --- NEW: top/bottom decile win rate ---
        top_decile_win = np.nan
        bottom_decile_win = np.nan
        top_decile_n = 0
        bottom_decile_n = 0
        try:
            q10 = float(np.nanquantile(xr, 0.10))
            q90 = float(np.nanquantile(xr, 0.90))
            top_mask = xr >= q90
            bot_mask = xr <= q10
            top_decile_n = int(top_mask.sum())
            bottom_decile_n = int(bot_mask.sum())
            if top_decile_n >= 20:
                top_decile_win = float(np.mean(yw[top_mask]))
            if bottom_decile_n >= 20:
                bottom_decile_win = float(np.mean(yw[bot_mask]))
        except Exception:
            pass

        rows.append({
            "feature": col,
         
            "n": int(m.sum()),
            "pearson_r": float(pear),
            "spearman_r": float(spear),
            "abs_spearman": float(abs(spear)),
            "auc_win": float(auc) if pd.notna(auc) else np.nan,
            "auc_ts_win": float(auc_ts) if pd.notna(auc_ts) else np.nan,
            "top_decile_n": top_decile_n,
            "top_decile_win_rate": top_decile_win,
            "bottom_decile_n": bottom_decile_n,
            "bottom_decile_win_rate": bottom_decile_win,
        })

    scores = pd.DataFrame(rows).sort_values("abs_spearman", ascending=False)

    # Mutual Information (공통 유효 행만)
    if not scores.empty:
        X = df[[c for c in feature_cols if c in df.columns]].copy()
        X = X.apply(pd.to_numeric, errors="coerce")
        mask = np.isfinite(df["r_val"].to_numpy(dtype=float))
        mask &= np.isfinite(X.to_numpy(dtype=float)).all(axis=1)

        if mask.sum() >= 200:
            Xv = X.loc[mask].to_numpy(dtype=float)
            yv_r = df.loc[mask, "r_val"].to_numpy(dtype=float)
            yv_w = df.loc[mask, "win"].to_numpy(dtype=int)

            try:
                mi_r = mutual_info_regression(Xv, yv_r, random_state=42)
                mi_w = mutual_info_classif(Xv, yv_w, random_state=42)
                mi_map_r = dict(zip(X.columns, mi_r))
                mi_map_w = dict(zip(X.columns, mi_w))

                scores["mi_reg_rval"] = scores["feature"].map(mi_map_r)
                scores["mi_clf_win"] = scores["feature"].map(mi_map_w)
            except Exception:
                scores["mi_reg_rval"] = np.nan
                scores["mi_clf_win"] = np.nan

    scores.to_csv(os.path.join(out_dir, "feature_scores.csv"), index=False)
    return scores


def plot_top_features(scores: pd.DataFrame, out_dir: str, topk: int = 20):
    if scores.empty:
        return
    top = scores.head(topk).copy().sort_values("abs_spearman", ascending=True)
    plt.figure(figsize=(10, 7))
    plt.barh(top["feature"], top["abs_spearman"])
    plt.title(f"Top {topk} | abs(Spearman) vs r_val (signal_time features)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"feature_scores_top{topk}.png"), dpi=160)
    plt.close()


def plot_binned_relationship(df: pd.DataFrame, feature: str, out_dir: str, bins: int = 10):
    if feature not in df.columns:
        return
    x = pd.to_numeric(df[feature], errors="coerce")
    tmp = pd.DataFrame({"x": x, "r_val": df["r_val"], "win": df["win"]}).dropna()
    if len(tmp) < 200:
        return

    tmp["bin"] = pd.qcut(tmp["x"], q=bins, duplicates="drop")
    g = tmp.groupby("bin", observed=True).agg(
        n=("r_val", "size"),
        x_min=("x", "min"),
        x_max=("x", "max"),
        mean_r=("r_val", "mean"),
        winrate=("win", "mean"),
    ).reset_index(drop=True)

    g.to_csv(os.path.join(out_dir, f"binned_{feature}.csv"), index=False)

    xs = np.arange(len(g))
    plt.figure(figsize=(10, 5))
    plt.plot(xs, g["mean_r"].to_numpy(), marker="o", label="mean r_val")
    plt.plot(xs, g["winrate"].to_numpy(), marker="o", label="winrate")
    plt.title(f"Binned relationship (signal_time): {feature}")
    plt.xticks(xs, [f"{i+1}" for i in xs])
    plt.xlabel("quantile bin (1..N)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"binned_{feature}.png"), dpi=160)
    plt.close()


def export_binned_all_features(
    df: pd.DataFrame,
    features: List[str],
    out_dir: str,
    bins: int = 10,
    max_features: int | None = None,
) -> pd.DataFrame:
    """
    모든 feature에 대해 분위수(bin)별 성과를 계산해
    - 개별 CSV/PNG: binned_<feature>.csv/.png
    - 통합 CSV: binned_all_features.csv
    를 저장합니다.

    columns (per bin):
      feature, bin_idx(1..N), n, x_min, x_max, mean_r, winrate
    """
    rows = []
    feats = features if max_features is None else features[:max_features]

    for f in feats:
        if f not in df.columns:
            continue
        x = pd.to_numeric(df[f], errors="coerce")
        tmp = pd.DataFrame({"x": x, "r_val": df["r_val"], "win": df["win"]}).dropna()
        if len(tmp) < 200:
            continue

        tmp["bin"] = pd.qcut(tmp["x"], q=bins, duplicates="drop")
        g = tmp.groupby("bin", observed=True).agg(
            n=("r_val", "size"),
            x_min=("x", "min"),
            x_max=("x", "max"),
            mean_r=("r_val", "mean"),
            winrate=("win", "mean"),
        ).reset_index(drop=True)

        # bin index 부여
        g.insert(0, "bin_idx", np.arange(1, len(g) + 1))
        g.insert(0, "feature", f)

        # 통합 rows
        rows.append(g)

        # 개별 저장(사용자가 바로 눈으로 보기 편하게)
        g.to_csv(os.path.join(out_dir, f"binned_{f}.csv"), index=False)

        xs = np.arange(len(g))
        plt.figure(figsize=(10, 5))
        plt.plot(xs, g["mean_r"].to_numpy(), marker="o", label="mean r_val")
        plt.plot(xs, g["winrate"].to_numpy(), marker="o", label="winrate")
        plt.title(f"Binned relationship (signal_time): {f}")
        plt.xticks(xs, [f"{i+1}" for i in xs])
        plt.xlabel("quantile bin (1..N)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"binned_{f}.png"), dpi=160)
        plt.close()

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(os.path.join(out_dir, "binned_all_features.csv"), index=False)
    return out


# ============================================================
# 4) 실행부
# ============================================================
def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = _ensure_dir(os.path.join("analysis_outputs",ts))

    grouped = bt.load_data_grouped(bt.BTConfig.DATA_DIR, bt.BTConfig.TIMEFRAME)
    if not grouped or bt.BTConfig.TIMEFRAME not in grouped:
        print("❌ 실행 불가: 데이터가 없습니다.")
        return

    data = grouped[bt.BTConfig.TIMEFRAME]
    btc_df = data["btc"]
    alt_dict = data["alts"]

    if btc_df is None or not alt_dict:
        print("❌ 실행 불가: BTC 또는 알트 데이터 없음")
        return

    engine = EnhancedBacktestEngine(bt.BTConfig.INITIAL_BALANCE)
    engine.run(btc_df, alt_dict)  # ✅ 동일한 루프 순서 + feature logging만 추가

    engine.print_summary(bt.BTConfig.TIMEFRAME)
    engine.print_monthly_and_daily_returns()

    df = build_trade_dataset(engine)
    if df.empty:
        print("❌ 거래 기록이 없거나 특징 추출 실패")
        return

    df.to_csv(os.path.join(out_dir, "trades_with_features.csv"), index=False)

    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    scores = score_features(df, feature_cols, out_dir)

    plot_top_features(scores, out_dir, topk=20)

    # ✅ NEW: 모든 지표(feature) 구간별(분위수) 성과 저장
    # - 파일이 많아지는 것이 부담이면 max_features를 20~50 등으로 줄이면 됩니다.
    ordered_feats = scores["feature"].tolist() if not scores.empty else feature_cols
    export_binned_all_features(df, ordered_feats, out_dir, bins=10, max_features=None)

    print(f"\n✅ 저장 완료: {out_dir}")
    print(" - trades_with_features.csv")
    print(" - feature_scores.csv")
    print(" - feature_scores_top20.png")
    print(" - binned_<feature>.csv / .png (top5)")


if __name__ == "__main__":
    main()
