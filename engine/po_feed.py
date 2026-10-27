"""
Pocket Option live feed (snapshots + payouts) via BinaryOptionsToolsV2.

Same proven engine as the user's previous bot (Desktop/Pocket bot/pocket_payout.py),
adapted to serve the signal engine:

    po_feed.candles("EUR/USD")  ->  M1 OHLC DataFrame (240 bars, cached 30 s)
    po_feed.candles("EUR/USD", otc=True)  ->  OTC series preferred
    po_feed.payouts()           ->  {ASSET: payout%}
    po_feed.status()            ->  /status dashboard data

 ssid priority:  PO_SSID env  >  pocket_ssid.json (project root, git-ignored)
 The sid file can be created either by copying the old bot's file or:
     python -m engine.po_feed --save "42[\"auth\",{...}]"

 The library is synchronous -> every network call runs under a thread lock;
 callers from asyncio should use asyncio.to_thread().
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
_TTL = 30.0            # candle cache seconds — fresh M1 bar granularity
_CLIENT = {"api": None, "ssid": ""}
_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}

# Signal-engine pair -> PO asset symbols (OTC first for the OTC quote type)
_ASSETS = {
    "EUR/USD": (["EURUSD_otc", "EURUSD"], "EURUSD_otc"),
    "USD/JPY": (["USDJPY_otc", "USDJPY"], "USDJPY_otc"),
    "GBP/USD": (["GBPUSD_otc", "GBPUSD"], "GBPUSD_otc"),
    "AUD/USD": (["AUDUSD_otc", "AUDUSD"], "AUDUSD_otc"),
    "USD/CAD": (["USDCAD_otc", "USDCAD"], "USDCAD_otc"),
    "USD/CHF": (["USDCHF_otc", "USDCHF"], "USDCHF_otc"),
    "GBP/JPY": (["GBPJPY_otc", "GBPJPY"], "GBPJPY_otc"),
    "EUR/JPY": (["EURJPY_otc", "EURJPY"], "EURJPY_otc"),
    "AUD/CAD": (["AUDCAD_otc", "AUDCAD"], "AUDCAD_otc"),
    "EUR/CAD": (["EURCAD_otc", "EURCAD"], "EURCAD_otc"),
    "GBP/CAD": (["GBPCAD_otc", "GBPCAD"], "GBPCAD_otc"),
    "CAD/CHF": (["CADCHF_otc", "CADCHF"], "CADCHF_otc"),
    "EUR/GBP": (["EURGBP_otc", "EURGBP"], "EURGBP_otc"),
    "AUD/JPY": (["AUDJPY_otc", "AUDJPY"], "AUDJPY_otc"),
    "NZD/USD": (["NZDUSD_otc", "NZDUSD"], "NZDUSD_otc"),
    "CHF/JPY": (["CHFJPY_otc", "CHFJPY"], "CHFJPY_otc"),
    "EUR/CHF": (["EURCHF_otc", "EURCHF"], "EURCHF_otc"),
    "GBP/AUD": (["GBPAUD_otc", "GBPAUD"], "GBPAUD_otc"),
    "NZD/JPY": (["NZDJPY_otc", "NZDJPY"], "NZDJPY_otc"),
    "USD/BRL": (["USDBRL_otc", "USDBRL"], "USDBRL_otc"),
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


def _reset_client():
    api = _CLIENT.get("api")
    _CLIENT["api"] = None
    if api is not None:
        try:
            api.disconnect()
        except Exception:
            pass


def _ensure_client():
    """Connected PocketOption client, or None. Mirrors the proven old-bot logic."""
    ssid = load_ssid()
    if not ssid:
        return None
    api = _CLIENT.get("api")
    if api is not None and _CLIENT.get("ssid") == ssid:
        return api
    with _LOCK:
        ssid = load_ssid()                      # re-read under the lock
        if not ssid:
            return None
        api = _CLIENT.get("api")
        if api is not None and _CLIENT.get("ssid") == ssid:
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
def candles(pair: str, otc: bool = True, bars: int = 240) -> pd.DataFrame | None:
    """M1 OHLC DataFrame via PO, or None. Never raises, cached _TTL seconds."""
    key = f"{pair}|{'otc' if otc else 'spot'}"
    now = time.time()
    cached = _CACHE.get(key)
    if cached and now - cached[0] < _TTL:
        return cached[1]

    alts = _ASSETS.get(pair)
    if alts is None:
        base = pair.replace("/", "")
        alts = ([base + "_otc", base] if otc else [base])

    with _LOCK:
        api = _ensure_client()
        if api is None:
            return None
        rows = None
        for sym in alts:                       # try OTC first, then spot variant
            try:
                data = api.get_candles(sym, 60, bars)
            except Exception:
                continue
            if data and len(data) >= 60:
                rows = (sym, data)
                break
    if rows is None:
        return None
    sym, data = rows
    try:
        idx = pd.to_datetime([c["time"] for c in data], unit="s", utc=True)
        df = pd.DataFrame({
            "open": [float(c["open"]) for c in data],
            "high": [float(c["high"]) for c in data],
            "low": [float(c["low"]) for c in data],
            "close": [float(c["close"]) for c in data],
            "volume": [1.0] * len(data),
        }, index=idx).dropna().tail(240)
    except Exception:
        return None
    if len(df) >= 60:
        _CACHE[key] = (now, df)
        return df
    return None


# ------------------------------------------------------------------- payouts
def payouts() -> dict[str, float]:
    """{EURUSD_OTC: 92} normalized percentages, or {} if unavailable. Never raises."""
    with _LOCK:
        api = _ensure_client()
    if api is None:
        return {}
    out: dict[str, float] = {}
    try:
        raw = api.payout()
    except Exception:
        _reset_client()
        return {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                val = v.get("payout", v.get("payment")) if isinstance(v, dict) else v
                if val is not None and float(val) > 0:
                    f = float(val)
                    out[str(k).upper().replace("_OTC", "-OTC")] = f if f > 1.5 else f * 100
            except Exception:
                continue
    return out


def payout_for(pair: str, otc: bool = True) -> float | None:
    try:
        base = pair.replace("/", "").upper()
        return payouts().get(base + "-OTC" if otc else base)
    except Exception:
        return None


# -------------------------------------------------------------------- status
def status() -> dict:
    st = {"library": os.getenv("PO_SSID") and "env" or
          ("file" if os.path.exists(_SSID_FILE) else "none"),
          "connected": False, "assets": 0, "otc_assets": 0,
          "sample_payout": None, "error": ""}
    with _LOCK:
        api = _ensure_client()
    if api is None:
        st["error"] = "no ssid / library unavailable"
        return st
    try:
        assets = api.active_assets() or []
        st["assets"] = len(assets)
        st["otc_assets"] = sum(1 for a in assets if "otc" in str(a.get("symbol", "")).lower())
        st["connected"] = True
        st["sample_payout"] = payout_for("EUR/USD")
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
        return len(api.active_assets() or []) > 0
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
                  f"{df['close'].iloc[-1]:.5f} @ {df.index[-1]}")
    else:
        print("usage: python -m engine.po_feed --save 'SSID' | --show | --candles EURUSD")
