"""
Pocket Option live feed (candles + payouts) via BinaryOptionsToolsV2.

Same proven engine as the user's previous bot, hardened for signal generation:

    po_feed.candles("EUR/USD")             -> M1 OHLC DataFrame, freshest bar
    po_feed.candles("EUR/USD", otc=False)  -> spot series
    po_feed.payouts()                      -> {"EURUSD-OTC": 92, ...}
    po_feed.status()                       -> /status dashboard data

Freshness: `get_candles()` returns CLOSED history that the broker refreshes in
chunks, so it can lag 1-10 minutes. A short-lived client, however, can call
`get_candles_live()` for any asset and get the current forming bar (the library
binds a long-lived client's stream to the first asset it requests, which is why
the shared client can only use history). So candles() uses a temporary client
for live bars and falls back to the shared client's history.

`get_ticks()` is NOT the candle series (tested: unrelated price path, tens of
pips apart), so it is deliberately not used.

 ssid priority:  PO_SSID env  >  pocket_ssid.json (project root, git-ignored)
 Save it with:   python -m engine.po_feed --save '42["auth",{...}]'

 The library is synchronous -> network calls run under a reentrant lock with an
 acquire timeout, so one wedged call can never block the whole bot forever.
 Callers from asyncio should use asyncio.to_thread().
"""
from __future__ import annotations

import json
import os
import threading
import time

import pandas as pd

_SSID_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "pocket_ssid.json")
_LOCK = threading.RLock()   # reentrant: candles() holds it and calls _ensure_client()
_LOCK_TIMEOUT = 45.0        # a wedged socket must not freeze every later request
_TTL = 30.0                 # candle cache seconds
_HARD_STALE = 900           # beyond this the feed is considered broken
_CLIENT = {"api": None, "ssid": ""}
_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}
_LAST_ISSUE = ""            # why the last candles() call returned None

# A short-lived client is the only way to get FRESH live candles for an arbitrary
# asset (the library binds a long-lived client's live stream to its first asset).
# One temporary client at a time: they share the session and the network budget.
_FRESH_GUARD = threading.Lock()
# The library's constructor waits for the broker's asset list, which takes
# 2.5-13 s. Plus ~6 s for the live fetch: budget 22 s so a slow handshake still
# returns FRESH bars instead of dropping to the stale history fallback.
_FRESH_TIMEOUT = 22.0
_FRESH_ISSUE = ""           # why the last _fetch_fresh() attempt returned None

# Signal-engine pair -> PO symbols. OTC quotes use the *_otc symbol ONLY (its price
# series differs from spot); STOCK quotes use the spot symbol. OTC pairs rotate out
# of PO's active list temporarily — that pair then falls back (clearly labelled).
_ASSETS = {
    "EUR/USD": {"otc": "EURUSD_otc", "spot": "EURUSD"},
    "USD/JPY": {"otc": "USDJPY_otc", "spot": "USDJPY"},
    "GBP/USD": {"otc": "GBPUSD_otc", "spot": "GBPUSD"},
    "AUD/USD": {"otc": "AUDUSD_otc", "spot": "AUDUSD"},
    "USD/CAD": {"otc": "USDCAD_otc", "spot": "USDCAD"},
    "USD/CHF": {"otc": "USDCHF_otc", "spot": "USDCHF"},
    "GBP/JPY": {"otc": "GBPJPY_otc", "spot": "GBPJPY"},
    "EUR/JPY": {"otc": "EURJPY_otc", "spot": "EURJPY"},
    "AUD/CAD": {"otc": "AUDCAD_otc", "spot": "AUDCAD"},
    "EUR/CAD": {"otc": "EURCAD_otc", "spot": "EURCAD"},
    "GBP/CAD": {"otc": "GBPCAD_otc", "spot": "GBPCAD"},
    "CAD/CHF": {"otc": "CADCHF_otc", "spot": "CADCHF"},
    "EUR/GBP": {"otc": "EURGBP_otc", "spot": "EURGBP"},
    "AUD/JPY": {"otc": "AUDJPY_otc", "spot": "AUDJPY"},
    "NZD/USD": {"otc": "NZDUSD_otc", "spot": "NZDUSD"},
    "CHF/JPY": {"otc": "CHFJPY_otc", "spot": "CHFJPY"},
    "EUR/CHF": {"otc": "EURCHF_otc", "spot": "EURCHF"},
    "GBP/AUD": {"otc": "GBPAUD_otc", "spot": "GBPAUD"},
    "NZD/JPY": {"otc": "NZDJPY_otc", "spot": "NZDJPY"},
    "USD/BRL": {"otc": "USDBRL_otc", "spot": "USDBRL"},
    "EUR/AUD": {"otc": "EURAUD_otc", "spot": "EURAUD"},
    "USD/SGD": {"otc": "USDSGD_otc", "spot": "USDSGD"},
    "GBP/CHF": {"otc": "GBPCHF_otc", "spot": "GBPCHF"},
    "EUR/NZD": {"otc": "EURNZD_otc", "spot": "EURNZD"},
}


# ------------------------------------------------------------------- ssid io
def load_ssid() -> str:
    env = os.getenv("PO_SSID", "").strip()
    if env:
        return env.strip().strip('"').strip("'")
    try:
        with open(_SSID_FILE) as f:
            return str(json.load(f).get("ssid", "")).strip()
    except Exception:
        return ""


def save_ssid(ssid: str) -> bool:
    ssid = (ssid or "").strip().strip('"').strip("'")
    if len(ssid) < 20:
        print('SSID looks too short. Copy the FULL 42["auth",{...}] string.')
        return False
    try:
        with open(_SSID_FILE, "w") as f:
            json.dump({"ssid": ssid}, f)
        try:
            os.chmod(_SSID_FILE, 0o600)
        except Exception:
            pass
        print(f"Saved SSID to {_SSID_FILE}")
        return True
    except Exception as e:
        print("Save failed:", e)
        return False


# -------------------------------------------------------------------- client
def _connected(api) -> bool:
    """True only if the library reports a live connection (cheap, no network)."""
    try:
        return bool(api.is_connected())
    except Exception:
        return False


def _reset_client():
    api = _CLIENT.get("api")
    _CLIENT["api"] = None
    _CLIENT["ssid"] = ""
    if api is not None:
        try:
            api.disconnect()
        except Exception:
            pass


def _ensure_client():
    """Connected PocketOption client, or None. Reconnects a dead client."""
    ssid = load_ssid()
    if not ssid:
        return None
    api = _CLIENT.get("api")
    if api is not None and _CLIENT.get("ssid") == ssid and _connected(api):
        return api
    with _LOCK:
        ssid = load_ssid()                      # re-read under the lock
        if not ssid:
            return None
        api = _CLIENT.get("api")
        if api is not None and _CLIENT.get("ssid") == ssid and _connected(api):
            return api
        _reset_client()
        try:
            from BinaryOptionsToolsV2 import PocketOption
        except Exception:
            return None
        try:
            api = PocketOption(ssid)
            try:
                api.connect()
            except Exception:
                pass
            try:
                api.wait_for_assets(timeout=30)
            except Exception:
                pass
            _CLIENT["api"] = api
            _CLIENT["ssid"] = ssid
            return api
        except Exception:
            return None


# ------------------------------------------------------------------- candles
def is_ready() -> bool:
    """Cheap check: is the PO client connected right now? No network call."""
    return _connected(_CLIENT.get("api"))


def last_issue() -> str:
    """Human-readable reason the previous candles() call failed ('' if it worked)."""
    return _LAST_ISSUE


def _bar_age(df: pd.DataFrame) -> float | None:
    """Seconds since the last candle's open time, or None if unusable."""
    try:
        last = pd.Timestamp(df.index[-1])
        if last.tzinfo is None:
            last = last.tz_localize("UTC")
        return float((pd.Timestamp.now(tz="UTC") - last.tz_convert("UTC")).total_seconds())
    except Exception:
        return None


def _rows_to_frame(rows) -> pd.DataFrame | None:
    """PO candle dicts -> clean, sorted, de-duplicated M1 OHLC frame."""
    if not rows or len(rows) < 60:
        return None
    try:
        idx = pd.to_datetime([c["time"] for c in rows], unit="s", utc=True)
        df = pd.DataFrame({
            "open": [float(c["open"]) for c in rows],
            "high": [float(c["high"]) for c in rows],
            "low": [float(c["low"]) for c in rows],
            "close": [float(c["close"]) for c in rows],
            "volume": [1.0] * len(rows),
        }, index=idx)
    except Exception:
        return None
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna()
    # Drop impossible bars (high below the body / low above it). A corrupt row
    # would otherwise skew ATR, Bollinger width and every pattern that uses wicks.
    body_hi = df[["open", "close"]].max(axis=1)
    body_lo = df[["open", "close"]].min(axis=1)
    sane = ((df[["open", "high", "low", "close"]] > 0).all(axis=1)
            & (df["high"] >= body_hi)
            & (df["low"] <= body_lo)
            & (df["high"] >= df["low"]))
    df = df[sane]
    if len(df) < 60:
        return None
    return df.tail(240)


def _fetch_live(api, sym: str, bars: int):
    """
    Historical closed candles (the same series the Pocket Option chart draws).

    Notes from testing this library build:
      * get_ticks() is NOT this series (it returned an unrelated price path),
        so it cannot be used to extend candles;
      * get_candles_live() / subscribe_symbol() are fresh and correct, but a
        long-lived client's stream binds to the first asset of the session, so
        only one asset can use them — see _fetch_fresh() for the workaround.
    """
    return _rows_to_frame(_fetch_history(api, sym, bars))


def _fetch_history(api, sym: str, bars: int):
    """Closed history fallback (can lag a few minutes)."""
    for want in (bars, 120, 60):
        try:
            data = api.get_candles(sym, 60, want)
        except Exception:
            continue
        if data and len(data) >= 60:
            return data
    return None


def _fetch_fresh(ssid: str, sym: str, bars: int) -> pd.DataFrame | None:
    """
    Fresh bars via a short-lived client: closed history + the forming candle.

    Why this exists: the broker's get_candles() history is refreshed in chunks
    and can be 1-10 minutes behind, which is useless for a 1-minute signal. A
    brand-new client can call get_candles_live() for ANY asset (the stream
    binding only poisons reused clients) and returns a current forming bar.

    Cost is ~9 s, so it runs on a worker thread with a hard join deadline. If
    it overruns, the caller falls back to the shared client's history; the
    worker keeps the guard until it finishes so requests cannot pile up.
    """
    if not ssid:
        _FRESH_ISSUE = "no ssid configured"
        return None
    if not _FRESH_GUARD.acquire(blocking=False):
        _FRESH_ISSUE = "another live fetch is still running"
        return None                     # another fresh fetch is already running

    box: dict[str, object] = {}

    def work():
        api = None
        it = None
        try:
            from BinaryOptionsToolsV2 import PocketOption
            # The constructor already waits for the asset list; an extra
            # wait_for_assets() here only added latency.
            api = PocketOption(ssid)
            try:
                api.connect()
            except Exception:
                pass
            it = api.get_candles_live(sym, 60, max(1.0, bars / 60.0), bars)
            closed, forming = next(it)
            rows = list(closed or [])
            if (isinstance(forming, dict) and forming.get("time")
                    and (not rows or int(forming["time"]) > int(rows[-1].get("time", 0)))):
                rows.append(forming)
            box["rows"] = rows
        except Exception as e:
            box["err"] = f"{type(e).__name__}: {e}"
        finally:
            # Close the async generator first: that cancels its tick reader
            # task, so shutdown() does not tear down a channel under it (which
            # printed "Error: Core(ChannelReceiver(Closed))" and leaked a task).
            if it is not None:
                try:
                    import asyncio
                    asyncio.run_coroutine_threadsafe(
                        it.async_gen.aclose(), it.loop).result(timeout=5)
                except Exception:
                    pass
            if api is not None:
                try:
                    api.shutdown()
                except Exception:
                    pass
            _FRESH_GUARD.release()

    th = threading.Thread(target=work, name=f"po-fresh-{sym}", daemon=True)
    th.start()
    th.join(_FRESH_TIMEOUT)
    if th.is_alive():
        _FRESH_ISSUE = f"live fetch timed out after {_FRESH_TIMEOUT:.0f}s"
        return None                     # worker still owns the guard

    rows = box.get("rows")
    if not rows:
        _FRESH_ISSUE = str(box.get("err") or "no bars returned")
        return None
    _FRESH_ISSUE = ""
    return _rows_to_frame(rows)


def _cached_fresh(key: str) -> pd.DataFrame | None:
    cached = _CACHE.get(key)
    if not cached:
        return None
    df = cached[1]
    age = _bar_age(df)
    if age is not None and age <= _HARD_STALE:
        return df
    return None


def candles(pair: str, otc: bool = True, bars: int = 240) -> pd.DataFrame | None:
    """M1 OHLC DataFrame from the broker's candle series, or None. Never raises."""
    global _LAST_ISSUE
    _LAST_ISSUE = ""
    key = f"{pair}|{'otc' if otc else 'spot'}"
    cached = _CACHE.get(key)
    if cached and time.time() - cached[0] < _TTL:
        return cached[1]

    spec = _ASSETS.get(pair)
    if spec is None:
        base = pair.replace("/", "")
        spec = {"otc": base + "_otc", "spot": base}
    sym = spec["otc"] if otc else spec["spot"]

    # 1) Fresh path: short-lived client (does not touch the shared client, so no
    #    lock is held and payouts/status keep answering).
    ssid = load_ssid()
    df = _fetch_fresh(ssid, sym, bars) if ssid else None
    if df is not None:
        age = _bar_age(df)
        if age is not None and age <= _HARD_STALE:
            _CACHE[key] = (time.time(), df)
            return df
        _LAST_ISSUE = f"live feed delayed {age / 60:.0f} min"
        df = None
    elif _FRESH_ISSUE:
        _LAST_ISSUE = _FRESH_ISSUE

    # 2) Fallback: the shared client's closed history (may lag a few minutes).
    if not _LOCK.acquire(timeout=_LOCK_TIMEOUT):
        _LAST_ISSUE = _LAST_ISSUE or "feed busy"
        return _cached_fresh(key)

    hist = None
    try:
        api = _ensure_client()
        if api is None:
            _LAST_ISSUE = _LAST_ISSUE or "ssid not connected"
            return _cached_fresh(key)
        hist = _fetch_live(api, sym, bars)                 # broker candle series
    except Exception as e:
        _LAST_ISSUE = f"feed error ({type(e).__name__})"
        hist = None
    finally:
        _LOCK.release()

    if hist is None:
        _LAST_ISSUE = _LAST_ISSUE or f"{sym} not in PO's active list"
        return _cached_fresh(key)

    age = _bar_age(hist)
    if age is None or age > _HARD_STALE:
        _LAST_ISSUE = (f"feed delayed {age / 60:.0f} min" if age is not None
                       else "no usable bars")
        return _cached_fresh(key)          # never present a dead feed as "now"

    _LAST_ISSUE = ""
    _CACHE[key] = (time.time(), hist)
    return hist


# ------------------------------------------------------------------- payouts
_PAYOUT_ISSUE = ""          # why the last bulk payout read was empty/failed


def _as_pct(val) -> float | None:
    """92 / 92.0 / "92%" / 0.92 -> 92.0; anything unusable -> None."""
    try:
        if isinstance(val, str):
            val = val.strip().rstrip("%").strip()
        f = float(val)
    except Exception:
        return None
    if f <= 0:
        return None
    # library returns whole percentages (92); tolerate fractions (0.92)
    return f if f >= 2.0 else f * 100


def _normalize_payouts(raw) -> dict[str, float]:
    """Raw library payout payload -> {"EURUSD-OTC": 92.0, ...}."""
    out: dict[str, float] = {}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        if isinstance(v, dict):
            v = v.get("payout", v.get("payment"))
        f = _as_pct(v)
        if f is not None:
            out[str(k).upper().replace("_OTC", "-OTC")] = f
    return out


def _asset_payouts(api) -> dict[str, float]:
    """
    Payouts taken from the broker's active-asset list.

    This is the fallback for builds/sessions where `api.payout()` returns an
    empty map even though the asset list (and therefore the payout field) is
    loaded - seen on the AWS server. It is still the broker's own number for
    that exact symbol, never a guess.
    """
    try:
        active = api.active_assets() or []
    except Exception:
        return {}
    if isinstance(active, dict):                 # some builds key by symbol
        active = list(active.values())
    return _normalize_payouts({a.get("symbol"): a.get("payout")
                               for a in active if isinstance(a, dict)})


def _bulk_payouts(api) -> dict[str, float]:
    """`api.payout()` map, or {} (never raises). Records why it was empty."""
    global _PAYOUT_ISSUE
    try:
        raw = api.payout()
    except Exception as e:
        _PAYOUT_ISSUE = f"payout() raised {type(e).__name__}: {e}"
        return {}
    pay = _normalize_payouts(raw)
    if not pay:
        shape = type(raw).__name__
        size = len(raw) if hasattr(raw, "__len__") else "n/a"
        _PAYOUT_ISSUE = f"payout() returned {shape} size={size}"
    else:
        _PAYOUT_ISSUE = ""
    return pay


def payouts() -> dict[str, float]:
    """{"EURUSD-OTC": 92, ...} percentages, or {} if unavailable. Never raises."""
    api = _CLIENT.get("api")
    if api is None or not _connected(api):
        with _LOCK:
            api = _ensure_client()
    if api is None:
        return {}
    pay = _bulk_payouts(api)
    # Fill gaps from the asset list: some library builds return an empty bulk
    # map while the active assets (loaded at connect) carry the same payouts.
    for k, v in _asset_payouts(api).items():
        pay.setdefault(k, v)
    return pay


def payout_for(pair: str, otc: bool = True) -> float | None:
    """
    Payout for THIS pair only; never another asset's number.

    Uses the already-connected client and never constructs one: building a
    PocketOption client can take 10+ seconds, and a signal must not wait for a
    cosmetic percentage. Boot-time warm() establishes the client; /status can
    rebuild it when needed.

    Three sources, all the broker's own number for this exact symbol:
      1. `payout(symbol)` - direct per-asset lookup (cheapest, most precise);
      2. the bulk `payout()` map;
      3. the `payout` field of the active-asset list (works on builds where
         the bulk map comes back empty).
    """
    try:
        api = _CLIENT.get("api")
        if api is None or not _connected(api):
            return None
        global _PAYOUT_ISSUE
        sym = symbol_for(pair, otc)
        base = pair.replace("/", "").upper()
        key = base + ("-OTC" if otc else "")

        # 1) direct per-asset lookup
        try:
            one = api.payout(sym)
        except TypeError:
            one = None                  # older stub without the asset argument
        except Exception:
            one = None
        if one is not None:
            f = _as_pct(one)
            if f is not None:
                _PAYOUT_ISSUE = ""
                return f

        # 2) bulk map
        pay = _bulk_payouts(api)
        val = pay.get(key) or (pay.get(base) if otc else None)
        if val is not None:
            return val

        # 3) active-asset list
        pay = _asset_payouts(api)
        val = pay.get(key) or (pay.get(base) if otc else None)
        if val is not None:
            _PAYOUT_ISSUE = "bulk payout empty; used active-asset payout"
        return val
    except Exception:
        return None


def payout_issue() -> str:
    """Diagnostic for /status: why payouts could not be read ('' if fine)."""
    return _PAYOUT_ISSUE


def symbol_for(pair: str, otc: bool = True) -> str:
    """Signal-engine pair -> Pocket Option symbol."""
    spec = _ASSETS.get(pair)
    if spec is None:
        base = pair.replace("/", "")
        spec = {"otc": base + "_otc", "spot": base}
    return spec["otc"] if otc else spec["spot"]


def asset_status(pair: str, otc: bool = True) -> tuple[bool, str]:
    """
    (tradeable now?, human explanation).

    Pocket Option serves candle history for assets that are NOT in its active
    list (e.g. USD/BRL OTC out of session). Such a chart looks alive but no trade
    can be placed — a signal on it is useless. Requiring the symbol to be in the
    active list prevents that.

    Payout is deliberately NOT part of this test: it can be missing for a moment
    while the asset is still tradeable. The signal footer shows "payout n/a" in
    that case instead of refusing a valid setup.

    If the active list cannot be read (transient API gap / client not connected),
    we do not block: the caller may still have real candles.
    """
    sym = symbol_for(pair, otc)
    api = _CLIENT.get("api")
    if api is None or not _connected(api):
        return True, ""
    try:
        active = {str(a.get("symbol", "")).lower() for a in (api.active_assets() or [])}
    except Exception:
        return True, ""
    if not active:
        return True, ""
    if sym.lower() not in active:
        kind = "OTC" if otc else ""
        return False, (f"{pair} {kind} is not in Pocket Option's active asset list "
                       "right now").replace("  ", " ")
    return True, ""


# -------------------------------------------------------------------- status
def status() -> dict:
    st = {"library": os.getenv("PO_SSID") and "env" or
          ("file" if os.path.exists(_SSID_FILE) else "none"),
          "connected": False, "assets": 0, "otc_assets": 0,
          "sample_payout": None, "payout_error": "", "error": ""}
    if not _LOCK.acquire(timeout=30):
        st["error"] = "feed busy (a previous request is still running)"
        return st
    try:
        api = _ensure_client()
    finally:
        _LOCK.release()
    if api is None:
        st["error"] = "no ssid / library unavailable"
        return st
    try:
        assets = api.active_assets() or []
        st["assets"] = len(assets)
        st["otc_assets"] = sum(1 for a in assets if "otc" in str(a.get("symbol", "")).lower())
        st["connected"] = _connected(api)
        st["sample_payout"] = payout_for("EUR/USD")
        st["payout_error"] = payout_issue()
        if not st["connected"]:
            st["error"] = "socket disconnected"
    except Exception as e:
        _reset_client()
        st["error"] = f"{type(e).__name__}: {e}"
        st["connected"] = False
    return st


def warm() -> bool:
    """Best-effort warm-up used at boot; returns True if the client is ready."""
    with _LOCK:
        api = _ensure_client()
    if api is None:
        return False
    try:
        return bool(api.active_assets()) and _connected(api)
    except Exception:
        return False


def stop():
    with _LOCK:
        _reset_client()


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "--save":
        save_ssid(sys.argv[2])
    elif len(sys.argv) >= 2 and sys.argv[1] == "--show":
        st = status()
        print(st)
        pay = payouts()
        top = sorted(pay.items(), key=lambda kv: -kv[1])[:15]
        print("top payouts:", top)
    elif len(sys.argv) >= 3 and sys.argv[1] == "--candles":
        df = candles(sys.argv[2])
        if df is None:
            print("no candles (check ssid)")
        else:
            print(f"{sys.argv[2]}: {len(df)} bars | last close "
                  f"{df['close'].iloc[-1]:.5f} @ {df.index[-1]} | age "
                  f"{_bar_age(df):.0f}s")
    else:
        print("usage: python -m engine.po_feed --save 'SSID' | --show | --candles EURUSD")
