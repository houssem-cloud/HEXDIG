"""
Technical indicators — implemented to match the exact input values used on the
Pocket Option chart (see screenshots):

    Bollinger Bands      Period 20, Deviation 2   (and a second band, Deviation 3)
    Stochastic           %K 6, %D 3, Smoothing 3, MA type SMA
    CCI                  Period 25
    ADX                  ADX Smoothing 1, DI Length 5
    Heikin Ashi candles  (chart type used as trend filter)

Everything is pure pandas/numpy — no TA-Lib dependency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


# ---------------------------------------------------------------- Bollinger
def bollinger(close: pd.Series, period: int = 20, deviation: float = 2.0):
    """Bollinger Bands: Period 20, Deviation 2 (and 3 for the stretch band)."""
    mid = sma(close, period)
    std = close.rolling(period).std(ddof=0)          # population std, like MT5/Pocket
    upper = mid + deviation * std
    lower = mid - deviation * std
    return mid, upper, lower


# --------------------------------------------------------------- Stochastic
def stochastic(df: pd.DataFrame, k_period: int = 6, d_period: int = 3,
               smoothing: int = 3) -> tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator — Pocket Option inputs:
        %K period 6, %D period 3, Smoothing 3, Moving Average SMA (slow stochastic)
    """
    hh = df["high"].rolling(k_period).max()
    ll = df["low"].rolling(k_period).min()
    raw = 100.0 * (df["close"] - ll) / (hh - ll)
    k = sma(raw, smoothing)              # smoothed %K
    d = sma(k, d_period)                 # %D = SMA of %K
    return k, d


# ---------------------------------------------------------------------- CCI
def cci(df: pd.DataFrame, period: int = 25) -> pd.Series:
    """Commodity Channel Index — Period 25."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    ma = sma(tp, period)
    md = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


# ---------------------------------------------------------------------- ADX
def adx(df: pd.DataFrame, di_length: int = 5, adx_smoothing: int = 1):
    """
    Average Directional Index — Pocket Option inputs: Smoothing 1, DI Length 5.
    Returns (adx, plus_di, minus_di).
    """
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)

    atr_ = wilder_smooth(tr, di_length)
    plus_di = 100.0 * wilder_smooth(pd.Series(plus_dm, index=df.index), di_length) / atr_
    minus_di = 100.0 * wilder_smooth(pd.Series(minus_dm, index=df.index), di_length) / atr_

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_ = wilder_smooth(dx, adx_smoothing)
    return adx_, plus_di, minus_di


def wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (used by ADX/ATR RMA)."""
    return series.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


# ---------------------------------------------------------------------- ATR
def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return wilder_smooth(tr, period)


# ------------------------------------------------------------- Heikin Ashi
def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Heikin Ashi candles — the chart type selected in the screenshots."""
    import numpy as np
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)

    ha_close = (o + h + l + c) / 4.0
    ha_open = np.empty_like(ha_close)
    ha_open[0] = (o[0] + ha_close[0]) / 2.0
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0

    ha_high = np.maximum.reduce([h, ha_open, ha_close])
    ha_low = np.minimum.reduce([l, ha_open, ha_close])

    return pd.DataFrame({"open": ha_open, "high": ha_high,
                         "low": ha_low, "close": ha_close}, index=df.index)


# ---------------------------------------------------------------- EMA trend
def ema_trend(close: pd.Series, fast: int = 9, slow: int = 21):
    return ema(close, fast), ema(close, slow)
