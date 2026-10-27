"""
Market data provider.

* STOCK pairs  — real 1-minute OHLC fetched from Yahoo Finance (EURUSD=X style
  symbols, mirrors the spot price Pocket Option shows for non-OTC quotes).
* OTC pairs    — Pocket Option OTC weekend feeds are private; OTC series is
  generated deterministically (seeded random walk anchored to the underlying
  pair's latest spot) so every user of the bot sees the *same* OTC candles in
  the same minute, and signals are reproducible.

If Yahoo is unreachable, STOCK falls back to the synthetic feed so the bot
keeps working (a warning is attached to output candles).
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
import pandas as pd

SYMBOL_MAP = {
    "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "USDJPY=X",
    "AUD/USD": "AUDUSD=X", "USD/CAD": "USDCAD=X", "USD/CHF": "USDCHF=X",
    "GBP/JPY": "GBPJPY=X", "EUR/JPY": "EURJPY=X", "AUD/CAD": "AUDCAD=X",
    "EUR/CAD": "EURCAD=X", "GBP/CAD": "GBPCAD=X", "CAD/CHF": "CADCHF=X",
    "EUR/GBP": "EURGBP=X", "AUD/JPY": "AUDJPY=X", "NZD/USD": "NZDUSD=X",
    "CHF/JPY": "CHFJPY=X", "USD/BRL": "USDBRL=X", "EUR/CHF": "EURCHF=X",
    "GBP/AUD": "GBPAUD=X", "NZD/JPY": "NZDJPY=X",
}

# Typical per-minute volatility (fractional) used to synthesize OTC feeds
BASE_PIP_VOL = {
    "JPY": 0.00016,   # JPY pairs quote ~150
    "DEFAULT": 0.00006,
}

_CACHE: dict[tuple, tuple[float, pd.DataFrame | None]] = {}
CACHE_TTL = 25.0  # seconds — a bit less than one 1-minute bar

# Feed health telemetry (surfaced by the /status command)
_FEED_HEALTH: dict[str, object] = {
    "last_live_ts": None,      # unix ts of last successful LIVE fetch
    "last_live_pair": "",
    "last_live_bars": 0,
    "fallback_today": 0,       # how many fetches degraded to synthetic
    "last_error": "",
}


def feed_health() -> dict[str, object]:
    """Snapshot for the /status command."""
    h = dict(_FEED_HEALTH)
    if _FEED_HEALTH["last_live_ts"]:
        h["live_candle_age"] = round(time.time() - _FEED_HEALTH["last_live_ts"], 1)
    else:
        h["live_candle_age"] = None
    return h


def _mark_live(pair: str, df: pd.DataFrame):
    _FEED_HEALTH["last_live_ts"] = time.time()
    _FEED_HEALTH["last_live_pair"] = pair
    _FEED_HEALTH["last_live_bars"] = len(df)
    _FEED_HEALTH["last_error"] = ""


def _mark_fallback(err: str):
    _FEED_HEALTH["fallback_today"] = int(_FEED_HEALTH["fallback_today"]) + 1
    _FEED_HEALTH["last_error"] = err[:200]


def has_internet(timeout: float = 5.0) -> bool:
    """Quick outbound reachability probe (this is what matters — not SSID)."""
    import urllib.request
    try:
        urllib.request.urlopen("https://query1.finance.yahoo.com/v8/finance/chart/EURUSD=X?interval=1m&range=1d", timeout=timeout)
        return True
    except Exception:
        try:
            urllib.request.urlopen("https://api.telegram.org", timeout=timeout)
            return True
        except Exception:
            return False


def _synthetic(pair: str, quote_type: str, bars: int = 240) -> pd.DataFrame:
    """
    Deterministic per-minute OTC series. Re-seeded every 15 minutes from
    hash(pair) so candles stay consistent within the bucket for all users.
    """
    bucket = int(time.time() // 900)
    digest = hashlib.sha256(f"{pair}|{quote_type}|{bucket}".encode()).digest()
    seed = int.from_bytes(digest[:8], "little")
    rng = np.random.default_rng(seed)

    anchor = SYMBOL_MAP.get(pair)
    base = 1.0
    if anchor:  # try to anchor the random walk to the real spot price
        cached = _CACHE.get((pair, "STOCK"))
        if cached and cached[0] > time.time() - 600 and cached[1] is not None:
            base = float(cached[1]["close"].iloc[-1])

    vol = BASE_PIP_VOL["JPY"] if "JPY" in pair else BASE_PIP_VOL["DEFAULT"]

    # add mild trend + mean-reversion so the series is indicator-friendly
    trend = rng.normal(0, vol * 0.15, size=bars).cumsum()
    noise = rng.normal(0, vol, size=bars)
    walk = np.cumsum(noise) + 0.4 * trend
    close = base * (1.0 + walk)
    open_ = np.concatenate([[close[0]], close[:-1]])

    spread = np.abs(rng.normal(0, vol * 0.6, size=bars))
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread

    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("min"),
                        periods=bars, freq="1min")
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": np.ones(bars),
    }, index=idx)
    return df


def _yahoo(pair: str) -> pd.DataFrame | None:
    import yfinance as yf
    sym = SYMBOL_MAP.get(pair)
    if not sym:
        return None
    try:
        t = yf.Ticker(sym)
        df = t.history(period="1d", interval="1m")
        if df is None or df.empty:
            return None
        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        df = df.dropna().tail(240)
        return df
    except Exception:
        return None


async def get_candles(pair: str, quote_type: str) -> tuple[pd.DataFrame | None, str]:
    """
    Returns (df, source_note). quote_type: "STOCK" or "OTC".
    """
    key = (pair, quote_type)
    cached = _CACHE.get(key)
    if cached and cached[0] > time.time() - CACHE_TTL:
        note = "" if quote_type == "STOCK" else "OTC (synthetic feed)"
        return cached[1], note

    if quote_type == "STOCK":
        import asyncio
        df = await asyncio.to_thread(_yahoo, pair)
        if df is not None and len(df) >= 60:
            _CACHE[key] = (time.time(), df)
            _mark_live(pair, df)
            return df, "Live spot feed (Yahoo Finance M1)"
        # PO ssid feed also covers spot pairs when connected
        from engine import pocketoption_ws as po
        pdf = po.candles(pair)
        if pdf is not None:
            return pdf, "Pocket Option live feed (ssid websocket)"
        # fall through to synthetic backup
        _mark_fallback(f"Yahoo M1 fetch failed for {pair}")
        df = _synthetic(pair, quote_type)
        _CACHE[key] = (time.time(), df)
        return df, "⚠️ Live feed unavailable — backup synthetic feed"

    # OTC: try the live Pocket Option ssid feed first (that's the whole point)
    from engine import pocketoption_ws as po
    pdf = po.candles(pair)
    if pdf is not None:
        return pdf, "Pocket Option live OTC (ssid authenticated)"

    df = _synthetic(pair, quote_type)
    _CACHE[key] = (time.time(), df)
    return df, "⚠️ SSID feed not connected — OTC is synthetic"

    df = _synthetic(pair, quote_type)
    _CACHE[key] = (time.time(), df)
    return df, "OTC (synthetic feed)"
