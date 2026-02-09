# live_impulse_bot_v5_fixed.py
#
# V5.1 FIXES (2025-12-19):
# 1. Scheduling Fix: Updates exactly every 15 mins (Absolute time).
# 2. Real Entry Tracking: Parses execution response to record actual fill price.

import os
import time
import hmac
import json
import math
import hashlib
import logging
import threading
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING, InvalidOperation, getcontext

import requests
import pandas as pd
from websocket import WebSocketApp

# signal_engine은 기존 파일 그대로 사용
from signal_engine import SignalConfig, compute_latest_signals

getcontext().prec = 28

# ============================================================
# 0. 로깅 설정
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("LiveImpulseBot")


# ============================================================
# 1. 설정 (LiveConfig)
# ============================================================
class LiveConfig:
    API_KEY    = os.getenv("binance_api")
    API_SECRET = os.getenv("binance_secret")

    BASE_URL   = "https://fapi.binance.com"
    WS_URL     = "wss://fstream.binance.com/stream?streams=!bookTicker@arr"

    SYMBOLS = [
        "APTUSDT", "ADAUSDT", "ARBUSDT", "ATOMUSDT", "TIAUSDT",
        "NEARUSDT", "IMXUSDT", "AVAXUSDT", "ARKMUSDT", "FILUSDT",
        "JUPUSDT", "XLMUSDT", "1000SATSUSDT", "RUNEUSDT", "STRKUSDT",
        "BNBUSDT", "OPUSDT", "LINKUSDT", "TAOUSDT", "SOLUSDT",
        "DOGEUSDT", "ZROUSDT", "NOTUSDT", "FETUSDT", "DOTUSDT",
        "WIFUSDT", "ETCUSDT", "WLDUSDT", "PENDLEUSDT", "SEIUSDT",
        "MEMEUSDT", "BOMEUSDT", "LTCUSDT", "UNIUSDT", "SUIUSDT",
        "INJUSDT", "BCHUSDT", "ORDIUSDT", "STXUSDT", "XRPUSDT",
        "TRXUSDT", "PYTHUSDT", "BTCUSDT"
    ]
    BTC_SYMBOL = "BTCUSDT"

    TIMEFRAME   = "15m"
    KLINE_LIMIT = 500

    # Risk Management
    ACCOUNT_RISK_PCT = 0.02  # 0.5% risk per trade
    MAX_LONG_POS   = 0
    MAX_SHORT_POS  = 6
    MAX_TOTAL_POS  = 6

    LEVERAGE = 5

    SIGNAL_DELAY_SEC = 5
    TRIGGER_INTERVAL_SEC = 1.5 

    HTTP_TIMEOUT = 10
    RECV_WINDOW = 10000
    TIME_SYNC_EVERY_SEC = 60 * 10

    VERIFY_PROTECTION_EACH_CYCLE = True


# ============================================================
# 2. Binance Futures REST 클라이언트
# ============================================================
class BinanceFuturesClient:
    def __init__(self, api_key: str, api_secret: str, base_url: str):
        self.api_key = api_key
        self.api_secret = api_secret.encode()
        self.base_url = base_url

        self.time_offset_ms: int = 0
        self._last_time_sync_ts: float = 0.0
        self.symbol_filters: Dict[str, Dict[str, Decimal]] = {}
        
        self._load_exchange_info()
        self._sync_time(force=True)

    def _sync_time(self, force: bool = False):
        now = time.time()
        if not force and (now - self._last_time_sync_ts) < LiveConfig.TIME_SYNC_EVERY_SEC:
            return
        try:
            data = self._request("GET", "/fapi/v1/time", signed=False)
            server_ms = int(data["serverTime"])
            local_ms = int(time.time() * 1000)
            self.time_offset_ms = server_ms - local_ms
            self._last_time_sync_ts = now
            log.info(f"🕒 서버 시간 동기화 완료 (offset={self.time_offset_ms}ms)")
        except Exception as e:
            log.warning(f"⚠️ 서버 시간 동기화 실패: {e}")

    def _sign_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = "&".join([f"{k}={v}" for k, v in params.items()])
        signature = hmac.new(self.api_secret, query.encode(), hashlib.sha256).hexdigest()
        params["signature"] = signature
        return params

    def _request(self, method: str, path: str, params=None, signed=False):
        if params is None: params = {}
        headers = {"X-MBX-APIKEY": self.api_key}
        url = self.base_url + path

        for attempt in range(3):
            try:
                if signed:
                    self._sync_time(force=False)
                    params["timestamp"] = int(time.time() * 1000) + int(self.time_offset_ms)
                    params.setdefault("recvWindow", LiveConfig.RECV_WINDOW)
                    params = self._sign_params(params)

                if method == "GET":
                    resp = requests.get(url, params=params, headers=headers, timeout=LiveConfig.HTTP_TIMEOUT)
                elif method == "POST":
                    resp = requests.post(url, params=params, headers=headers, timeout=LiveConfig.HTTP_TIMEOUT)
                elif method == "DELETE":
                    resp = requests.delete(url, params=params, headers=headers, timeout=LiveConfig.HTTP_TIMEOUT)
                else:
                    raise ValueError("Unsupported HTTP method")

                if resp.status_code == 429:
                    log.warning("⚠️ 429 Too Many Requests - 잠시 대기")
                    time.sleep(1.0)
                    continue

                if resp.status_code >= 400:
                    txt = resp.text or ""
                    if '"code":-1021' in txt or "recvWindow" in txt:
                        self._sync_time(force=True)
                    raise Exception(f"HTTP {resp.status_code}: {txt}")

                return resp.json()

            except Exception as e:
                log.error(f"❌ API 요청 에러 ({attempt+1}/3): {e}")
                time.sleep(1.0)

        raise Exception(f"❌ API 요청 최종 실패: {path}")

    def _load_exchange_info(self):
        try:
            data = self._request("GET", "/fapi/v1/exchangeInfo", signed=False)
            symbols = data.get("symbols", [])
            for s in symbols:
                sym = s.get("symbol")
                if not sym: continue
                info = {"price_tick": Decimal("0"), "qty_step": Decimal("0"), "min_qty": Decimal("0")}
                for f in s.get("filters", []):
                    if f["filterType"] == "PRICE_FILTER":
                        info["price_tick"] = Decimal(str(f.get("tickSize", "0")))
                    elif f["filterType"] == "LOT_SIZE":
                        info["qty_step"] = Decimal(str(f.get("stepSize", "0")))
                        info["min_qty"] = Decimal(str(f.get("minQty", "0")))
                self.symbol_filters[sym] = info
            log.info(f"✅ exchangeInfo 로딩 완료 ({len(self.symbol_filters)}개)")
        except Exception as e:
            log.error(f"❌ exchangeInfo 로딩 실패: {e}")

    def _quantize_to_step(self, value: Decimal, step: Decimal, mode: str) -> Decimal:
        if step <= 0: return value
        q = value / step
        if mode == "down": q_int = q.to_integral_value(rounding=ROUND_FLOOR)
        elif mode == "up": q_int = q.to_integral_value(rounding=ROUND_CEILING)
        else: raise ValueError("mode error")
        return q_int * step

    def _fmt_decimal(self, d: Decimal) -> str:
        s = format(d, "f")
        if "." in s: s = s.rstrip("0").rstrip(".")
        return s if s else "0"

    def round_price(self, symbol: str, price: float | Decimal, mode: str = "down") -> str:
        info = self.symbol_filters.get(symbol)
        try: p = Decimal(str(price))
        except InvalidOperation: return "0"
        if not info or info["price_tick"] <= 0:
            return self._fmt_decimal(p.quantize(Decimal("0.0000000001"), rounding=ROUND_FLOOR))
        return self._fmt_decimal(self._quantize_to_step(p, info["price_tick"], mode))

    def round_qty(self, symbol: str, qty: float | Decimal) -> str:
        info = self.symbol_filters.get(symbol)
        try: q = Decimal(str(qty))
        except InvalidOperation: return "0"
        if not info or info["qty_step"] <= 0:
            return self._fmt_decimal(q.quantize(Decimal("0.00000001"), rounding=ROUND_FLOOR))
        q2 = self._quantize_to_step(q, info["qty_step"], "down")
        if q2 < info["min_qty"]: return "0"
        return self._fmt_decimal(q2)

    def get_klines(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        data = self._request("GET", "/fapi/v1/klines", params, signed=False)
        cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "qav", "num_trades", "taker_base", "taker_quote", "ignore"]
        df = pd.DataFrame(data, columns=cols)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df.set_index("open_time", inplace=True)
        for c in ["open", "high", "low", "close", "volume", "close_time"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        
        # 진행 중인 봉 제거
        try:
            now_ms = int(time.time() * 1000) + int(self.time_offset_ms)
            if len(df) > 0:
                last_close = df["close_time"].iloc[-1]
                if pd.notna(last_close) and int(last_close) > (now_ms - int(LiveConfig.SIGNAL_DELAY_SEC * 1000)):
                    df = df.iloc[:-1]
        except: pass
        
        df.dropna(subset=["open", "high", "low", "close", "volume"], inplace=True)
        return df

    def get_usdt_equity(self) -> float:
        data = self._request("GET", "/fapi/v2/balance", signed=True)
        for asset in data:
            if asset.get("asset") == "USDT":
                return float(asset.get("balance", 0.0))
        return 0.0

    def set_leverage(self, symbol: str, leverage: int):
        return self._request("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage}, signed=True)

    def get_position_amt(self, symbol: str) -> float:
        data = self._request("GET", "/fapi/v2/positionRisk", signed=True)
        for p in data:
            if p.get("symbol") == symbol:
                return float(p.get("positionAmt", 0.0))
        return 0.0

    def place_market_order(self, symbol: str, side: str, quantity: float, reduce_only: bool = False):
        qty_s = self.round_qty(symbol, quantity)
        params = {"symbol": symbol, "side": side, "type": "MARKET", "quantity": qty_s, "newOrderRespType": "RESULT"}
        if reduce_only: params["reduceOnly"] = "true"
        return self._request("POST", "/fapi/v1/order", params, signed=True)

    def place_limit_tp(self, symbol: str, side: str, quantity: float, price: float):
        qty_s = self.round_qty(symbol, quantity)
        price_s = self.round_price(symbol, price, mode="down")
        params = {"symbol": symbol, "side": side, "type": "LIMIT", "timeInForce": "GTC", "quantity": qty_s, "price": price_s, "reduceOnly": "true"}
        return self._request("POST", "/fapi/v1/order", params, signed=True)

    def place_stop_loss_market_algo(self, symbol: str, side: str, quantity: float, trigger_price: float):
        qty_s = self.round_qty(symbol, quantity)
        mode = "up" if side == "SELL" else "down"
        trig_s = self.round_price(symbol, trigger_price, mode=mode)
        params = {
            "algoType": "CONDITIONAL", "symbol": symbol, "side": side, "type": "STOP_MARKET",
            "quantity": qty_s, "triggerPrice": trig_s, "workingType": "MARK_PRICE",
            "priceProtect": "TRUE", "reduceOnly": "true"
        }
        return self._request("POST", "/fapi/v1/algoOrder", params, signed=True)

    def get_open_orders(self, symbol: str):
        return self._request("GET", "/fapi/v1/openOrders", {"symbol": symbol}, signed=True)

    def get_open_algo_orders(self, symbol: str):
        return self._request("GET", "/fapi/v1/openAlgoOrders", {"symbol": symbol}, signed=True)

    def cancel_all_open_orders(self, symbol: str):
        self._request("DELETE", "/fapi/v1/algoOpenOrders", {"symbol": symbol}, signed=True)
        return self._request("DELETE", "/fapi/v1/allOpenOrders", {"symbol": symbol}, signed=True)
    
    # [추가됨] 주문 응답에서 평단가 추출
    def get_avg_price_from_response(self, order_resp: dict) -> float:
        if "avgPrice" in order_resp:
            return float(order_resp["avgPrice"])
        return 0.0


# ============================================================
# 3. 데이터 클래스
# ============================================================
@dataclass
class PendingSignal:
    symbol: str
    side: int
    lim: float
    sl: float
    tp: float
    qty: float
    timestamp: pd.Timestamp

@dataclass
class LivePosition:
    symbol: str
    side: int
    qty: float
    entry: float      # 실제 체결가 (Real Entry)
    sl: float
    tp: float
    sl_algo_id: Optional[int] = None
    tp_order_id: Optional[int] = None


# ============================================================
# 4. 실전 봇 본체
# ============================================================
class LiveImpulseBot:
    def __init__(self, client: BinanceFuturesClient):
        self.client = client
        self.last_price: Dict[str, float] = {}
        self.best_bid: Dict[str, float] = {}
        self.best_ask: Dict[str, float] = {}
        self.ws_event_ms: Dict[str, int] = {}
        self.pending_signals: Dict[str, PendingSignal] = {}
        self.positions: Dict[str, LivePosition] = {}
        self.stop_event = threading.Event()
        self.ws_thread: Optional[threading.Thread] = None

    def _on_ws_message(self, ws, message):
        try:
            data = json.loads(message)
            arr = data.get("data", [])
            for item in arr:
                sym = item.get("s")
                if not sym: continue
                if "b" in item and "a" in item:
                    bid, ask = float(item.get("b", 0)), float(item.get("a", 0))
                    if bid > 0: self.best_bid[sym] = bid
                    if ask > 0: self.best_ask[sym] = ask
                    if bid > 0 and ask > 0: self.last_price[sym] = (bid + ask) / 2.0
                    
                    ev = item.get("E")
                    if ev: self.ws_event_ms[sym] = int(ev)
        except Exception: pass

    def start_ws(self):
        def run():
            ws = WebSocketApp(LiveConfig.WS_URL, on_message=self._on_ws_message)
            while not self.stop_event.is_set():
                try: ws.run_forever(ping_interval=30, ping_timeout=10)
                except: time.sleep(5)
        self.ws_thread = threading.Thread(target=run, daemon=True)
        self.ws_thread.start()

    def verify_protection_orders(self):
        for sym, pos in list(self.positions.items()):
            try:
                open_algo = self.client.get_open_algo_orders(sym)
                open_lim = self.client.get_open_orders(sym)
                has_sl = any(o.get("orderType") == "STOP_MARKET" for o in open_algo)
                has_tp = any(o.get("type") == "LIMIT" for o in open_lim)

                if not has_sl:
                    exit_side = "SELL" if pos.side == 1 else "BUY"
                    slr = self.client.place_stop_loss_market_algo(sym, exit_side, pos.qty, pos.sl)
                    pos.sl_algo_id = int(slr.get("algoId", 0))
                    log.warning(f"🛡️ SL 재설정: {sym}")

                if not has_tp:
                    exit_side = "SELL" if pos.side == 1 else "BUY"
                    tpr = self.client.place_limit_tp(sym, exit_side, pos.qty, pos.tp)
                    pos.tp_order_id = int(tpr.get("orderId", 0))
                    log.warning(f"🛡️ TP 재설정: {sym}")
            except Exception as e:
                log.warning(f"⚠️ 보호주문 확인 실패({sym}): {e}")

    def sync_positions_and_cleanup(self):
        for sym in list(self.positions.keys()):
            try:
                amt = self.client.get_position_amt(sym)
                if abs(amt) < 1e-12:
                    self.client.cancel_all_open_orders(sym)
                    self.positions.pop(sym, None)
                    log.info(f"🧹 {sym} 포지션 종료 확인 및 정리")
            except Exception as e:
                log.error(f"⚠️ 포지션 싱크 에러({sym}): {e}")

    def update_signals(self):
        log.info("📥 시그널 업데이트 중...")
        try:
            btc_df = self.client.get_klines(LiveConfig.BTC_SYMBOL, LiveConfig.TIMEFRAME)
            raw_data = {}
            for sym in LiveConfig.SYMBOLS:
                df = self.client.get_klines(sym, LiveConfig.TIMEFRAME)
                raw_data[f"{sym[:-4]}/USDT"] = df
            
            signals = compute_latest_signals(btc_df, raw_data, SignalConfig)
            
            if len(btc_df) > 0:
                last_ts = btc_df.index[-1]
                for sym, pend in list(self.pending_signals.items()):
                    if sym in self.positions: continue
                    age = (last_ts - pend.timestamp).total_seconds() // 900
                    if age >= SignalConfig.FLAG_MAX_BARS:
                        self.pending_signals.pop(sym, None)

            equity = self.client.get_usdt_equity()
            
            curr_longs = sum(1 for p in self.positions.values() if p.side == 1) + \
                         sum(1 for p in self.pending_signals.values() if p.side == 1)
            curr_shorts = sum(1 for p in self.positions.values() if p.side == -1) + \
                          sum(1 for p in self.pending_signals.values() if p.side == -1)

            for pair, sig in signals.items():
                act, lim, sl, tp = sig["act"], sig["lim"], sig["sl"], sig["tp"]
                if act == 0 or math.isnan(lim): continue
                
                sym = pair.split("/")[0] + "USDT"
                if sym in self.positions or sym in self.pending_signals: continue

                if act == 1 and curr_longs >= LiveConfig.MAX_LONG_POS: continue
                if act == -1 and curr_shorts >= LiveConfig.MAX_SHORT_POS: continue
                if (curr_longs + curr_shorts) >= LiveConfig.MAX_TOTAL_POS: break

                dist = abs(lim - sl)
                if dist <= 0: continue
                qty = (equity * LiveConfig.ACCOUNT_RISK_PCT) / dist
                qty_s = self.client.round_qty(sym, qty)
                if qty_s == "0": continue

                self.pending_signals[sym] = PendingSignal(
                    sym, int(act), float(lim), float(sl), float(tp), float(qty_s), sig["timestamp"]
                )
                if act == 1: curr_longs += 1
                else: curr_shorts += 1

            self.log_pending_signals()
            
            if LiveConfig.VERIFY_PROTECTION_EACH_CYCLE:
                self.sync_positions_and_cleanup()
                self.verify_protection_orders()

        except Exception as e:
            log.error(f"❌ 시그널 업데이트 대실패: {e}")

    def log_pending_signals(self):
        if not self.pending_signals:
            log.info("📌 펜딩 시그널 없음")
            return
        log.info(f"📌 펜딩 시그널 {len(self.pending_signals)}개 대기 중 (가격 확인 중...)")

    def check_triggers_and_enter(self):
        to_remove = []
        fallback_prices = {}

        for sym, pending in list(self.pending_signals.items()):
            ask = self.best_ask.get(sym)
            bid = self.best_bid.get(sym)
            last = self.last_price.get(sym)

            if last is None:
                if sym not in fallback_prices:
                    try:
                        t = self.client._request("GET", "/fapi/v1/ticker/bookTicker", {"symbol": sym}, signed=False)
                        b_p, a_p = float(t["bidPrice"]), float(t["askPrice"])
                        fallback_prices[sym] = (b_p, a_p)
                        self.best_bid[sym] = b_p
                        self.best_ask[sym] = a_p
                        self.last_price[sym] = (b_p + a_p) / 2
                        log.info(f"⚠️ {sym} 가격 수신(REST API) 성공: {b_p} / {a_p}")
                    except Exception as e:
                        log.error(f"❌ {sym} REST 가격 조회 실패: {e}")
                        continue
                
                if sym in fallback_prices:
                    bid, ask = fallback_prices[sym]
                    last = (bid + ask) / 2

            if last is None:
                continue

            if pending.side == 1:
                price = ask if ask is not None else last
            else:
                price = bid if bid is not None else last

            triggered = False
            if pending.side == 1 and price >= pending.lim:
                log.info(f"🟢 [TRIGGER] {sym} LONG 돌파! (Now:{price} >= Lim:{pending.lim})")
                triggered = True
            elif pending.side == -1 and price <= pending.lim:
                log.info(f"🔴 [TRIGGER] {sym} SHORT 돌파! (Now:{price} <= Lim:{pending.lim})")
                triggered = True

            if triggered:
                self.enter_position(pending)
                to_remove.append(sym)

        for sym in to_remove:
            self.pending_signals.pop(sym, None)

    # [수정됨] 실제 체결가 추적 로직 적용
    def enter_position(self, pending: PendingSignal):
        sym, side, qty, sl, tp = pending.symbol, pending.side, pending.qty, pending.sl, pending.tp
        entry_side = "BUY" if side == 1 else "SELL"
        exit_side = "SELL" if side == 1 else "BUY"

        try:
            # 1. 시장가 진입
            resp = self.client.place_market_order(sym, entry_side, qty)
            
            # 2. 실제 체결가(Real Entry Price) 파싱
            real_entry = self.client.get_avg_price_from_response(resp)
            
            # 응답에서 가격을 못 찾았다면 Position Risk 조회로 2차 확인
            if real_entry <= 0:
                time.sleep(1.0) # 체결 데이터 동기화 대기
                try:
                    positions = self.client._request("GET", "/fapi/v2/positionRisk", {"symbol": sym}, signed=True)
                    for p in positions:
                        if p["symbol"] == sym:
                            real_entry = float(p.get("entryPrice", 0.0))
                            break
                except Exception as e:
                    log.error(f"⚠️ {sym} 체결가 조회 2차 실패: {e}")
                    real_entry = pending.lim # 최후의 수단
            
            if real_entry > 0:
                slippage = (real_entry - pending.lim) if side == 1 else (pending.lim - real_entry)
                slippage_pct = (slippage / pending.lim) * 100
                log.info(f"🚀 {sym} 진입 성공 | 예상: {pending.lim} -> 실제: {real_entry} (슬리피지: {slippage_pct:.3f}%)")
            else:
                log.warning(f"🚀 {sym} 진입 성공했으나 체결가 확인 불가. 예상가({pending.lim})로 기록.")
                real_entry = pending.lim

            # 3. 보호 주문 (SL/TP)
            slr = self.client.place_stop_loss_market_algo(sym, exit_side, qty, sl)
            sl_id = slr.get("algoId")
            
            tpr = self.client.place_limit_tp(sym, exit_side, qty, tp)
            tp_id = tpr.get("orderId")

            # 4. 포지션 객체 생성 (entry에 real_entry 저장)
            self.positions[sym] = LivePosition(
                sym, side, qty, 
                entry=real_entry,
                sl=sl, tp=tp,
                sl_algo_id=int(sl_id) if sl_id else None,
                tp_order_id=int(tp_id) if tp_id else None
            )
            log.info(f"✅ {sym} 포지션 등록 완료 (RealEntry: {real_entry}, SL:{sl}, TP:{tp})")

        except Exception as e:
            log.error(f"❌ 진입/보호주문 실패({sym}): {e}")
            try: 
                # 비상 청산
                amt = self.client.get_position_amt(sym)
                if abs(amt) > 0:
                    log.warning(f"⚠️ {sym} 보호주문 실패로 인한 비상 청산 시도")
                    self.client.place_market_order(sym, exit_side, qty, reduce_only=True)
            except: 
                pass

    def run(self):
        if not LiveConfig.API_KEY:
            log.error("API Key Missing")
            return

        self.start_ws()
        log.info("🚀 LiveImpulseBot V5.1 (Fixed Scheduler + Real Entry Tracking) Started")

        for sym in LiveConfig.SYMBOLS:
            try: self.client.set_leverage(sym, LiveConfig.LEVERAGE)
            except: pass

        # 시작 시점 기준, 가장 가까운 미래의 15분 경계 시간 계산
        now = time.time()
        next_target_time = (math.floor(now / 900) + 1) * 900

        while True:
            wake_up_time = next_target_time + LiveConfig.SIGNAL_DELAY_SEC
            sleep_sec = wake_up_time - time.time()
            
            if sleep_sec > 0:
                t_str = time.strftime('%H:%M:%S', time.localtime(next_target_time))
                log.info(f"⏳ 다음 15m 봉 마감({t_str})까지 {sleep_sec:.1f}초 대기...")
                time.sleep(sleep_sec)

            try:
                self.update_signals()
            except Exception as e:
                log.error(f"❌ 시그널 업데이트 실패: {e}")

            next_target_time += 900  

            while time.time() < next_target_time:
                try:
                    self.check_triggers_and_enter()
                except Exception as e:
                    log.error(f"Main Loop Error: {e}")
                
                time.sleep(LiveConfig.TRIGGER_INTERVAL_SEC)

if __name__ == "__main__":
    cli = BinanceFuturesClient(LiveConfig.API_KEY, LiveConfig.API_SECRET, LiveConfig.BASE_URL)
    bot = LiveImpulseBot(cli)
    bot.run()