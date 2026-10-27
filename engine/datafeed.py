"""
Market data provider.

* STOCK pairs  — real 1-minute OHLC from Yahoo Finance (EURUSD=X style symbols,
  mirrors the spot price Pocket Option shows for non-OTC quotes). Yahoo can go
  stale when the market is closed, so freshness is checked; a stale series is
  never presented as a live one.
* OTC pairs    — Pocket Option's own OTC series via the ssid feed, and only while
  the broker actually offers that asset for trading (active list + payout). If
  the broker feed is unavailable, the bot REFUSES the signal: OTC prices are
  never synthesized, because a signal on invented prices can lose real money.

Freshness rules:
  * a bar set whose last candle is older than STALE_SECONDS is not "live";
  * STOCK then tries the broker's spot feed before giving up;
  * if the market is genuinely closed, get_candles returns (None, explanation)
    instead of inventing a signal — the Telegram layer shows that explanation.
"""
from __future__ import annotations

import asyncio
import hashlib
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

SYMBOL_MAP = {
    "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "USDJPY=X",
    "AUD/USD": "AUDUSD=X", "USD/CAD": "USDCAD=X", "USD/CHF": "USDCHF=X",
    "GBP/JPY": "GBPJPY=X", "EUR/JPY": "EURJPY=X", "AUD/CAD": "AUDCAD=X",
    "EUR/CAD": "EURCAD=X", "GBP/CAD": "GBPCAD=X", "CAD/CHF": "CADCHF=X",
    "EUR/GBP": "EURGBP=X", "AUD/JPY": "AUDJPY=X", "NZD/USD": "NZDUSD=X",
    "CHF/JPY": "CHFJPY=X", "USD/BRL": "USDBRL=X", "EUR/CHF": "EURCHF=X",
    "GBP/AUD": "GBPAUD=X", "NZD/JPY": "NZDJPY=X", "EUR/AUD": "EURAUD=X",
    "EUR/NZD": "EURNZD=X", "GBP/CHF": "GBPCHF=X", "USD/SGD": "USDSGD=X",
}

# Typical per-minute volatility (fractional) used to synthesize fallback feeds
BASE_PIP_VOL = {
    "JPY": 0.00016,   # JPY pairs quote ~150
    "DEFAULT": 0.00006,
}

# Realistic starting points so a synthetic series is at least in the right
# universe (a USD/JPY series near 1.0 was a display bug).
BASE_PRICE = {
    "EUR/USD": 1.09, "GBP/USD": 1.27, "AUD/USD": 0.66, "NZD/USD": 0.61,
    "USD/CAD": 1.36, "USD/CHF": 0.88, "USD/BRL": 5.40, "EUR/GBP": 0.85,
    "EUR/CHF": 0.95, "EUR/CAD": 1.48, "EUR/AUD": 1.65, "EUR/NZD": 1.79,
    "GBP/CAD": 1.73, "GBP/AUD": 1.93, "GBP/CHF": 1.12, "AUD/CAD": 0.90,
    "CAD/CHF": 0.65, "AUD/JPY": 102.0, "NZD/JPY": 95.0, "CHF/JPY": 175.0,
    "EUR/JPY": 168.0, "GBP/JPY": 196.0, "USD/SGD": 1.34, "USD/JPY": 155.0,
}

_CACHE: dict[tuple, tuple[float, pd.DataFrame, str]] = {}
CACHE_TTL = 25.0        # seconds — a bit less than one 1-minute bar
STALE_SECONDS = 300.0   # stock spot data older than this = market closed
DELAY_WARN = 150.0      # broker candle delay worth warning the user about

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


def bar_age(df: pd.DataFrame | None) -> float | None:
    """Seconds since the last candle's open, or None if the frame is unusable."""
    if df is None or df.empty:
        return None
    try:
        last = pd.Timestamp(df.index[-1])
        if last.tzinfo is None:
            last = last.tz_localize("UTC")
        return float((pd.Timestamp.now(tz="UTC") - last.tz_convert("UTC")).total_seconds())
    except Exception:
        return None


def has_internet(timeout: float = 5.0) -> bool:
    """Quick outbound reachability probe (this is what matters — not SSID)."""
    probes = (
        "https://query1.finance.yahoo.com/v8/finance/chart/EURUSD=X?interval=1m&range=1d",
        "https://api.telegram.org",
    )
    for url in probes:
        try:
            urllib.request.urlopen(url, timeout=timeout)
            return True
        except urllib.error.HTTPError:
            return True          # server answered -> the network is fine
        except Exception:
            continue
    return False


def _synthetic(pair: str, quote_type: str, bars: int = 240) -> pd.DataFrame:
    """
    Deterministic per-minute fallback series. Re-seeded every 15 minutes from
    hash(pair) so candles stay consistent within the bucket for all users.
    """
    bucket = int(time.time() // 900)
    digest = hashlib.sha256(f"{pair}|{quote_type}|{bucket}".encode()).digest()
    seed = int.from_bytes(digest[:8], "little")
    rng = np.random.default_rng(seed)

    base = float(BASE_PRICE.get(pair, 1.10))
    for qtype in ("STOCK", "OTC"):          # anchor to the latest real close if any
        cached = _CACHE.get((pair, qtype))
        if (cached and len(cached) > 2 and "synthetic" not in cached[2].lower()
                and cached[0] > time.time() - 900 and cached[1] is not None):
            try:
                base = float(cached[1]["close"].iloc[-1])
                break
            except Exception:
                pass

    vol = BASE_PIP_VOL["JPY"] if "JPY" in pair else BASE_PIP_VOL["DEFAULT"]

    # mild trend + noise so the series is indicator-friendly
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
    return pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": np.ones(bars),
    }, index=idx)


def _yahoo(pair: str) -> pd.DataFrame | None:
    import yfinance as yf
    sym = SYMBOL_MAP.get(pair)
    if not sym:
        return None
    try:
        df = yf.Ticker(sym).history(period="1d", interval="1m")
        if df is None or df.empty:
            return None
        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        df = df.dropna()
        # Keep short-but-real frames: right after the open there are fewer than
        # 60 bars, and that must be reported as "not enough history yet" rather
        # than silently replaced by synthetic prices.
        if df.empty or len(df) < 2:
            return None
        df = df[~df.index.duplicated(keep="last")].sort_index()
        # one timezone for every source — comparisons and logs stay sane
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_convert("UTC")
        else:
            df.index = df.index.tz_localize("UTC")
        return df.tail(240)
    except Exception:
        return None


def _store(key: tuple, df: pd.DataFrame, note: str) -> pd.DataFrame:
    _CACHE[key] = (time.time(), df, note)
    return df


async def get_candles(pair: str, quote_type: str) -> tuple[pd.DataFrame | None, str]:
    """
    Returns (df, source_note). quote_type: "STOCK" or "OTC".
    df is None only when there is no honest data to analyze (e.g. market closed);
    the note then explains why and is shown to the user.
    """
    key = (pair, quote_type)
    cached = _CACHE.get(key)
    if cached and cached[0] > time.time() - CACHE_TTL:
        return cached[1], cached[2]

    from engine import po_feed

    if quote_type == "STOCK":
        df = await asyncio.to_thread(_yahoo, pair)
        if df is not None:
            age = bar_age(df)
            if age is not None and age <= STALE_SECONDS and len(df) >= 60:
                note = "Live spot feed (Yahoo Finance M1)"
                _mark_live(pair, df)
                return _store(key, df, note), note

        # Yahoo is empty or stale -> the broker's own spot feed may still be live
        pdf = await asyncio.to_thread(po_feed.candles, pair, False)
        if pdf is not None:
            po_age = bar_age(pdf)
            if po_age is not None and po_age <= STALE_SECONDS and len(pdf) >= 60:
                note = "Pocket Option live spot feed (ssid)"
                _mark_live(pair, pdf)
                return _store(key, pdf, note), note

        # Diagnose from whichever source we actually have (a stale PO spot frame
        # proves the market is closed just as well as a stale Yahoo one).
        known = df if df is not None else pdf
        if known is not None:
            age = bar_age(known) or 0.0
            if age <= STALE_SECONDS and len(known) < 60:
                # Fresh data, but not enough of it: this is the first minutes of
                # a session (or a newly listed pair). Never substitute synthetic
                # prices here — say so and let the user retry shortly.
                _mark_fallback(f"only {len(known)} fresh M1 bars for {pair}")
                return None, (f"Not enough 1-minute history yet for {pair} "
                              f"({len(known)} bars). The market may have just opened — "
                              "try again in a few minutes.")
            # we have data, but the market is closed — do not fabricate a signal
            _mark_fallback(f"stale spot data for {pair} ({age / 60:.0f} min old)")
            hours = age / 3600.0
            when = f"{hours:.1f}h" if hours >= 1 else f"{age / 60:.0f} min"
            return None, (f"Market closed — Stock pairs are not trading right now "
                          f"(last candle {when} old). Try OTC instead.")

        _mark_fallback(f"Yahoo M1 fetch failed for {pair}")
        note = "⚠️ Live feed unreachable — backup synthetic feed (not a real price)"
        return _store(key, _synthetic(pair, quote_type), note), note

    # OTC: real, currently tradeable broker feed only. A signal on invented or
    # non-tradeable prices is worse than no signal, so nothing here is synthesized.
    # Check the active list first: a rotated-out pair is refused instantly instead
    # of paying for a 9-20 s live-fetch attempt that cannot succeed.
    ok, why = await asyncio.to_thread(po_feed.asset_status, pair, True)
    if not ok:
        _mark_fallback(why)
        return None, (f"{why}. Pocket Option rotates OTC assets — "
                      "pick another OTC pair or try again later.")

    pdf = await asyncio.to_thread(po_feed.candles, pair, True)
    if pdf is not None:
        ok, why = await asyncio.to_thread(po_feed.asset_status, pair, True)
        if not ok:                      # deactivated between the two checks
            _mark_fallback(why)
            return None, (f"{why}. Pocket Option rotates OTC assets — "
                          "pick another OTC pair or try again later.")
        pay = po_feed.payout_for(pair)
        pay_line = f" | payout {pay:.0f}%" if pay else " | payout n/a"
        age = bar_age(pdf)
        age_line = (f" | newest bar {age / 60:.0f} min old"
                    if age is not None and age > DELAY_WARN else "")
        _mark_live(pair, pdf)
        return pdf, f"Pocket Option live OTC (ssid){pay_line}{age_line}"

    issue = po_feed.last_issue()
    if po_feed.is_ready():
        _mark_fallback(f"OTC feed unavailable for {pair}: {issue}")
        note = (f"Pocket Option is not sending live OTC prices for {pair} right now"
                + (f" ({issue})" if issue else "")
                + ". No signal is better than a guess — pick another OTC pair.")
    else:
        _mark_fallback("PO ssid feed not connected")
        note = ("The Pocket Option ssid feed is not connected, so real OTC prices "
                "are unavailable. Stock pairs still work; reconnect the ssid to "
                "trade OTC.")
    return None, note
