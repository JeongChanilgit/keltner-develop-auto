# signal_engine.py
#
# Strategy: Regime-Filtered Keltner Burst (Tuned defaults)
# - I/O 포맷 유지: compute_latest_signals(btc_df, raw_data_dict, cfg=SignalConfig) -> {sym: {timestamp, act, lim, sl, tp, score}}
#
# 핵심:
#   1) Trend Regime: EMA200 위(bull) / 아래(bear)
#   2) Entry: Keltner upper/lower "band break"
#   3) Risk: SL=4*ATR, TP=8*ATR (RR=1:2)
#   4) Data-driven gates (기본 ON):
#       - ADX band: 22.7 ~ 41.2
#       - Volume ratio gate: vol_ratio20 <= 0.75
#       - EMA distance band: |ema_dist_pct| in [0.014, 0.028] (롱/숏 대칭)
#       - ATR% band: 0.0042 ~ 0.0112
#
# 주의:
#   - "돌파 판정"은 현재(i) 밴드가 아니라 이전(i-1) 밴드를 기준으로 비교(자기참조 완화)
#   - 데이터 컬럼: open, high, low, close, volume(선택) 필요

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


# ============================================================
# 1. 설정 (SignalConfig)
# ============================================================
class SignalConfig:
    # --- Trend Regime Filter ---
    TREND_EMA_PERIOD = 200

    # --- Keltner Channel Settings ---
    KC_PERIOD = 20
    KC_MULT = 1.5

    # --- ATR / Risk Management ---
    ATR_PERIOD = 14

    # RR = 1:2
    SL_ATR_MULT = 10
    TP_ATR_MULT = 5

    # --- ADX (Trend Strength) ---
    ADX_PERIOD = 14

    # Hard floor (avoid extreme chop). Keep loose.
    ADX_THRESHOLD = 3

    # Sweet-spot band (data-driven)
    USE_ADX_BAND = True
    ADX_BAND_MIN = 22.7
    ADX_BAND_MAX = 76  # upper bound to avoid overextended "too hot" regimes

    # Multi-range sweet spots (from binned_all_features.csv)
    USE_ADX_RANGES = False
    ADX_RANGES = (
        (8.063027, 15.457592),
        (20.127496, 43.826505),
    )

    # --- Volume Regime Gate (data-driven) ---
    VOL_MA_PERIOD = 20
    USE_VOL_RATIO_GATE = False
    VOL_RATIO_MAX = 0.75  # tune 0.69~0.84 depending on trade count

    # Multi-range volume regime (from binned_all_features.csv)
    # vol_ratio20 = volume / SMA(volume, 20)
    USE_VOL_RATIO_RANGES = True
    VOL_RATIO_RANGES = (
        (0.140231, 1.097824),
        (1.349656, 3.089243),
    )

    # --- EMA(20) distance band (data-driven) ---
    # ema_dist_pct = (close - ema20) / ema20
    # apply symmetric band on abs(distance); still enforce sign by direction
    USE_EMA_DIST_BAND = True
    EMA_DIST_MIN = 0.004407
    EMA_DIST_MAX = 0.031028

    # --- ATR% band (data-driven) ---
    # atr_pct = atr / close
    USE_ATR_PCT_BAND = False
    ATR_PCT_MIN = 0.0042
    ATR_PCT_MAX = 0.0112

    # Multi-range ATR% sweet spots (from binned_all_features.csv)
    USE_ATR_PCT_RANGES = True
    ATR_PCT_RANGES = (
        (0.005319, 0.009421),
        (0.011115, 0.013974),
    )

    # --- System ---
    # --- Time-of-day filter (data-driven) ---
    # NOTE: uses df.index hour as-is (no timezone conversion).
    # If your index is UTC but you want KST, convert index beforehand in data loader.
    USE_HOUR_FILTER = True
    # From binned_all_features.csv: best mean_r in 03~08, worst around 22~23.
    ALLOWED_HOURS = (3, 4, 5, 6, 7, 8)

    # --- Absolute volume gate (data-driven) ---
    # From binned_all_features.csv: very low volume bin performed poorly.
    USE_MIN_VOLUME = True
    MIN_VOLUME = 200_000  # approximate cut to remove the lowest-volume bin

    # --- Keltner bandwidth% band (data-driven) ---
    # kc_bw_pct = (kc_upper - kc_lower) / kc_mid
    USE_KC_BW_PCT_BAND = True
    KC_BW_PCT_MIN = 0.0172
    KC_BW_PCT_MAX = 0.0337

    # --- ATR% band (updated from analysis) ---
    # Avoid "too hot" volatility regimes (atr_pct >= ~0.0115 was negative).
    USE_ATR_PCT_BAND = True
    ATR_PCT_MIN = 0.0053
    ATR_PCT_MAX = 0.01

    # Disable old multi-range defaults (kept for compatibility).
    USE_ATR_PCT_RANGES = False

    # --- EMA distance 'too close' filter ---
    # From binned_all_features.csv: ema_dist_pct near 0 performed poorly.
    USE_EMA_NEAR_ZERO_CUT = True
    EMA_NEAR_ZERO_ABS_MAX = 0.0035
    WARMUP_PERIOD = 210
    FLAG_MAX_BARS = 3
    COOLDOWN_BARS = 6

    # Breakout strength band (distance beyond Keltner band, in ATR units)
    # breakout_atr = abs(close - band) / ATR
    USE_BREAKOUT_ATR_BAND = True
    BREAKOUT_ATR_MIN = 0.016841
    BREAKOUT_ATR_MAX = 0.281181

    # --- Auto-optimizer best params (2026-02-07 01:40:30) ---
    # NOTE:
    # - VOL_RATIO_MAX is only used when USE_VOL_RATIO_GATE=True and USE_VOL_RATIO_RANGES=False.
    # - ADX_THRESHOLD acts as a hard floor; if USE_ADX_BAND is enabled, the band may dominate.
    SL_ATR_MULT = 6.07729487695417
    TP_ATR_MULT = 8.94115441033344
    KC_PERIOD = 14
    KC_MULT = 1.6812415935078813
    ADX_THRESHOLD = 29
    VOL_RATIO_MAX = 2.0063736986428276
    EMA_DIST_MIN = 0.009492014285138375
    EMA_DIST_MAX = 0.0258262057340366
    ATR_PCT_MIN = 0.0036210538032121205
    ATR_PCT_MAX = 0.014449279597732368

    # --- Compatibility (미사용 호환 변수) ---
    HMA_PERIOD, RSI_PERIOD, RSI_PIVOT, CROSS_LOOKBACK_BARS = 0, 0, 0, 0
    VOL_MULT, USE_TR_FILTER, TR_ATR_MULT = 0, 0, 0
    SWING_LOOKBACK, EMA_PERIOD, EMA_TREND_PERIOD = 0, 0, 0
    KER_PERIOD, KER_THRESHOLD, BREAKOUT_BARS = 0, 0, 0
    ST_PERIOD, ST_MULTIPLIER, KC_PERIOD_COMPAT, KC_MULTIPLIER_COMPAT = 0, 0, 0, 0
    CHOP_PERIOD, CHOP_THRESHOLD, CMF_PERIOD, CMF_THRESHOLD = 0, 0, 0, 0
    L_IMP, L_DEPTH, L_RR, S_IMP, S_DEPTH, S_RR = 0, 0, 0, 0, 0, 0


def _in_any_range(x: float, ranges: tuple[tuple[float, float], ...]) -> bool:
    for lo, hi in ranges:
        if lo <= x <= hi:
            return True
    return False

# ============================================================
# 2. Indicators
# ============================================================
def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr = _true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = _true_range(high, low, close)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100.0 * pd.Series(plus_dm, index=high.index).ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100.0 * pd.Series(minus_dm, index=high.index).ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)

    dx = 100.0 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx


def _keltner_channel(close: pd.Series, high: pd.Series, low: pd.Series, period: int, mult: float):
    mid = _ema(close, period)
    atr = _atr(high, low, close, period)  # common choice
    upper = mid + mult * atr
    lower = mid - mult * atr
    return mid, upper, lower


def add_indicators(df: pd.DataFrame, cfg: SignalConfig) -> pd.DataFrame:
    out = df.copy()

    # required columns
    for c in ["open", "high", "low", "close"]:
        if c not in out.columns:
            raise ValueError(f"Missing column: {c}")

    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)

    # Trend regime EMA
    out["ema_trend"] = _ema(close, cfg.TREND_EMA_PERIOD)

    # ATR
    out["atr"] = _atr(high, low, close, cfg.ATR_PERIOD)
    out["atr_pct"] = out["atr"] / close.replace(0, np.nan)

    # Keltner
    kc_mid, kc_upper, kc_lower = _keltner_channel(close, high, low, cfg.KC_PERIOD, cfg.KC_MULT)
    out["kc_mid"] = kc_mid
    out["kc_upper"] = kc_upper
    out["kc_lower"] = kc_lower
    # Keltner bandwidth (%)
    out["kc_bw_pct"] = (out["kc_upper"] - out["kc_lower"]) / out["kc_mid"].replace(0, np.nan)

    # EMA distance vs EMA20 (=kc_mid)
    out["ema_dist_pct"] = (close - kc_mid) / kc_mid.replace(0, np.nan)

    # ADX
    out["adx"] = _adx(high, low, close, cfg.ADX_PERIOD)

    # Volume ratio (optional)
    if "volume" in out.columns:
        vol = out["volume"].astype(float).replace(0, np.nan)
        vol_ma = vol.rolling(cfg.VOL_MA_PERIOD).mean()
        out["vol_ratio20"] = vol / vol_ma
    else:
        out["vol_ratio20"] = np.nan

    return out


# ============================================================
# 3. Signal logic
# ============================================================
def compute_latest_signals(
    btc_df: pd.DataFrame,
    raw_data_dict: dict[str, pd.DataFrame],
    cfg: SignalConfig = SignalConfig,
) -> dict[str, dict]:
    """Compute last-bar signal per symbol."""
    signals: dict[str, dict] = {}

    for sym, df in raw_data_dict.items():
        curr_df = add_indicators(df, cfg)
        n = len(curr_df)
        last_ts = curr_df.index[-1] if n > 0 else None

        if n < cfg.WARMUP_PERIOD + 2:
            signals[sym] = {
                "timestamp": last_ts,
                "act": 0.0,
                "lim": np.nan,
                "sl": np.nan,
                "tp": np.nan,
                "score": 0.0,
            }
            continue

        closes = curr_df["close"].to_numpy(dtype=float)
        ema_trend = curr_df["ema_trend"].to_numpy(dtype=float)
        kc_upper = curr_df["kc_upper"].to_numpy(dtype=float)
        kc_lower = curr_df["kc_lower"].to_numpy(dtype=float)
        atr = curr_df["atr"].to_numpy(dtype=float)
        atr_pct = curr_df["atr_pct"].to_numpy(dtype=float)
        adx = curr_df["adx"].to_numpy(dtype=float)
        ema_dist_pct = curr_df["ema_dist_pct"].to_numpy(dtype=float)
        vol_ratio20 = curr_df["vol_ratio20"].to_numpy(dtype=float)
        kc_bw_pct = curr_df["kc_bw_pct"].to_numpy(dtype=float)
        volume = curr_df["volume"].to_numpy(dtype=float) if "volume" in curr_df.columns else np.full(n, np.nan, dtype=float)

        act = np.zeros(n, dtype=float)
        lim = np.full(n, np.nan, dtype=float)
        sl = np.full(n, np.nan, dtype=float)
        tp = np.full(n, np.nan, dtype=float)

        last_entry_idx = -10**9

        # iterate bars
        for i in range(cfg.WARMUP_PERIOD + 1, n):
            # cooldown
            if (i - last_entry_idx) < cfg.COOLDOWN_BARS:
                continue

            # -------- Regime gates --------
            # 1) ADX floor
            if adx[i] < cfg.ADX_THRESHOLD:
                continue

            # 2) ADX sweet-spot band / ranges (data-driven)
            if getattr(cfg, 'USE_ADX_RANGES', False):
                if not _in_any_range(float(adx[i]), cfg.ADX_RANGES):
                    continue
            elif cfg.USE_ADX_BAND and not (cfg.ADX_BAND_MIN <= adx[i] <= cfg.ADX_BAND_MAX):
                continue

            # 3) Volume ratio gate / ranges (data-driven)
            if getattr(cfg, 'USE_VOL_RATIO_RANGES', False):
                if np.isnan(vol_ratio20[i]) or not _in_any_range(float(vol_ratio20[i]), cfg.VOL_RATIO_RANGES):
                    continue
            elif cfg.USE_VOL_RATIO_GATE:
                if not np.isnan(vol_ratio20[i]) and vol_ratio20[i] > cfg.VOL_RATIO_MAX:
                    continue

            # 4) ATR% band / ranges (data-driven)
            if getattr(cfg, 'USE_ATR_PCT_RANGES', False):
                if np.isnan(atr_pct[i]) or not _in_any_range(float(atr_pct[i]), cfg.ATR_PCT_RANGES):
                    continue
            elif cfg.USE_ATR_PCT_BAND:
                if np.isnan(atr_pct[i]) or not (cfg.ATR_PCT_MIN <= atr_pct[i] <= cfg.ATR_PCT_MAX):
                    continue

            # 5) Time-of-day gate (data-driven)
            if getattr(cfg, "USE_HOUR_FILTER", False):
                ts = curr_df.index[i]
                hr = getattr(ts, "hour", None)
                if hr is None or int(hr) not in getattr(cfg, "ALLOWED_HOURS", ()):
                    continue

            # 6) Absolute volume floor (data-driven)
            if getattr(cfg, "USE_MIN_VOLUME", False):
                if np.isnan(volume[i]) or float(volume[i]) < float(getattr(cfg, "MIN_VOLUME", 0.0)):
                    continue

            # 7) Keltner bandwidth% band (data-driven)
            if getattr(cfg, "USE_KC_BW_PCT_BAND", False):
                if np.isnan(kc_bw_pct[i]) or not (cfg.KC_BW_PCT_MIN <= float(kc_bw_pct[i]) <= cfg.KC_BW_PCT_MAX):
                    continue

            current_close = closes[i]
            prev_close = closes[i - 1]
            current_atr = atr[i]

            # -------- Directional regime --------
            bull = current_close > ema_trend[i]
            bear = current_close < ema_trend[i]

            # -------- EMA distance band (data-driven) --------
            if cfg.USE_EMA_DIST_BAND and not np.isnan(ema_dist_pct[i]):
                d = float(abs(ema_dist_pct[i]))
                # 0) reject 'too close to EMA' regimes (near-zero was negative)
                if getattr(cfg, "USE_EMA_NEAR_ZERO_CUT", False) and d <= float(getattr(cfg, "EMA_NEAR_ZERO_ABS_MAX", 0.0)):
                    bull = False
                    bear = False
                # 1) magnitude must be inside the learned band
                if not (cfg.EMA_DIST_MIN <= d <= cfg.EMA_DIST_MAX):
                    bull = False
                    bear = False
                else:
                    # 2) keep sign requirement by direction
                    if bull and ema_dist_pct[i] <= 0:
                        bull = False
                    if bear and ema_dist_pct[i] >= 0:
                        bear = False

            # -------- Entry triggers (Keltner band breaks) --------
            # NOTE: use previous band (i-1) to reduce self-referential bias
            # Long: prev was <= upper, now breaks above upper
            if bull and prev_close <= kc_upper[i - 1] and current_close > kc_upper[i - 1]:
                # breakout strength in ATR units (using previous band)
                if getattr(cfg, 'USE_BREAKOUT_ATR_BAND', False) and current_atr > 0:
                    brk = float((current_close - kc_upper[i - 1]) / current_atr)
                    if not (cfg.BREAKOUT_ATR_MIN <= brk <= cfg.BREAKOUT_ATR_MAX):
                        continue

                entry = current_close
                risk = current_atr * float(cfg.SL_ATR_MULT)
                reward = current_atr * float(cfg.TP_ATR_MULT)

                act[i] = 1.0
                lim[i] = entry
                sl[i] = entry - risk
                tp[i] = entry + reward
                last_entry_idx = i
                continue

            # Short: prev was >= lower, now breaks below lower
            if bear and prev_close >= kc_lower[i - 1] and current_close < kc_lower[i - 1]:
                # breakout strength in ATR units (using previous band)
                if getattr(cfg, 'USE_BREAKOUT_ATR_BAND', False) and current_atr > 0:
                    brk = float((kc_lower[i - 1] - current_close) / current_atr)
                    if not (cfg.BREAKOUT_ATR_MIN <= brk <= cfg.BREAKOUT_ATR_MAX):
                        continue

                entry = current_close
                risk = current_atr * float(cfg.SL_ATR_MULT)
                reward = current_atr * float(cfg.TP_ATR_MULT)

                act[i] = -1.0
                lim[i] = entry
                sl[i] = entry + risk
                tp[i] = entry - reward
                last_entry_idx = i
                continue

        last_idx = n - 1
        last_act = float(act[last_idx])
        last_lim = float(lim[last_idx]) if not np.isnan(lim[last_idx]) else np.nan
        last_sl = float(sl[last_idx]) if not np.isnan(sl[last_idx]) else np.nan
        last_tp = float(tp[last_idx]) if not np.isnan(tp[last_idx]) else np.nan

        # score (lightweight heuristics; 0 if no signal)
        score = 0.0
        if last_act != 0.0:
            # bonus: ADX closeness to allowed regime center
            if getattr(cfg, 'USE_ADX_RANGES', False):
                mids = [0.5 * (lo + hi) for lo, hi in cfg.ADX_RANGES]
                nearest = min(mids, key=lambda m: abs(float(adx[last_idx]) - m))
                score += float(max(0.0, 100.0 - abs(float(adx[last_idx]) - nearest) * 3.0))
            elif cfg.USE_ADX_BAND:
                band_mid = 0.5 * (cfg.ADX_BAND_MIN + cfg.ADX_BAND_MAX)
                score += float(max(0.0, 100.0 - abs(adx[last_idx] - band_mid) * 3.0))

            # bonus: closer to EMA sweet spot (distance band center)
            if cfg.USE_EMA_DIST_BAND and not np.isnan(ema_dist_pct[last_idx]):
                d = abs(ema_dist_pct[last_idx])
                band_mid = 0.5 * (cfg.EMA_DIST_MIN + cfg.EMA_DIST_MAX)
                score += float(max(0.0, 50.0 - abs(d - band_mid) * 2000.0))

            # bonus: breakout strength (distance beyond band)
            if last_act == 1.0 and not np.isnan(kc_upper[last_idx]):
                score += float(max(0.0, (closes[last_idx] - kc_upper[last_idx]) / kc_upper[last_idx] * 1000.0))
            if last_act == -1.0 and not np.isnan(kc_lower[last_idx]):
                score += float(max(0.0, (kc_lower[last_idx] - closes[last_idx]) / kc_lower[last_idx] * 1000.0))

        signals[sym] = {
            "timestamp": last_ts,
            "act": last_act,
            "lim": last_lim,
            "sl": last_sl,
            "tp": last_tp,
            "score": float(score),
        }

    return signals
