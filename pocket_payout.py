"""
Pocket Option LIVE data (read-only): per-asset payout % AND real M1 candles,
using your saved SSID.

This powers the POCKET signal bot with REAL prices (no more random-walk feed)
and REAL payouts. Fully optional: if SSID/library is missing or fails, the bot
falls back to its demo feed and manual payout.

Setup (PC only, local PowerShell - NEVER paste SSID in Telegram):
  1. Save SSID:   python pocket_payout.py --save "FULL_SSID"
  2. Show data:   python pocket_payout.py --show
  3. Candles:     python pocket_payout.py --candles EURUSD

Engine: BinaryOptionsToolsV2 (unofficial, read-only). No trading.
"""

import collections
import json
import os
import sys
import time
import threading
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

SSID_FILE = os.path.join(os.path.dirname(__file__), "pocket_ssid.json")

# ---- request throttling (idea ported from Hummingbot's AsyncThrottler) --------
# Hummingbot guards every exchange call with a shared rate budget so a burst of
# requests can't get the account rate-limited/banned. We do the same with a
# small sliding-window limiter: generous enough to never slow normal use, but a
# hard ceiling when the scanner fans out over every pair.
_RATE_LIMITS = {                 # key -> (max calls, window seconds)
    "candles": (40, 60.0),
    "payout": (12, 60.0),
}
_HITS = {}
_HITS_LOCK = threading.Lock()


def _throttle(key, block=True):
    """Reserve a slot for `key`. True if reserved; False only when non-blocking
    and the window is full. Never raises."""
    limit, window = _RATE_LIMITS.get(key, (60, 60.0))
    while True:
        with _HITS_LOCK:
            now = time.time()
            dq = _HITS.setdefault(key, collections.deque())
            while dq and now - dq[0] > window:
                dq.popleft()
            if len(dq) < limit:
                dq.append(now)
                return True
            wait = dq[0] + window - now
        if not block:
            return False
        time.sleep(max(0.05, min(wait, 1.0)))


def _retry(fn, tries=3, base=0.4):
    """Call fn() up to `tries` times with exponential backoff (Hummingbot's
    async_retry, synchronous). Returns the result or None. Never raises."""
    for k in range(max(1, tries)):
        try:
            return fn()
        except Exception:
            if k == tries - 1:
                return None
            time.sleep(base * (2 ** k))
    return None

_LOCK = threading.Lock()
_TTL = 120            # payout cache seconds
_CLOSES_TTL = 55      # candle cache seconds
_PAY = {"t": 0, "data": {}}
_CLOSES = {}          # norm asset -> (time, [closes])
_OHLC = {}            # norm asset -> (time, [{o,h,l,c}, ...])
_CLIENT = {"api": None, "ssid": "", "map": {}}
LAST_FETCH = {}       # diagnostics: last large candle request


def load_ssid():
    if not os.path.exists(SSID_FILE):
        return ""
    try:
        with open(SSID_FILE) as f:
            return str(json.load(f).get("ssid", "")).strip()
    except Exception:
        return ""


def save_ssid(ssid):
    ssid = (ssid or "").strip().strip('"').strip("'")
    if len(ssid) < 20:
        print('SSID looks too short. Copy the FULL 42["auth",{...}] string.')
        return False
    try:
        with open(SSID_FILE, "w") as f:
            json.dump({"ssid": ssid}, f)
        try:
            os.chmod(SSID_FILE, 0o600)
        except Exception:
            pass
        print("Saved SSID (local pocket_ssid.json only).")
        return True
    except Exception as e:
        print("Save failed:", e)
        return False


def _norm(sym):
    """Normalize Pocket asset names to EURUSD / EURUSD-OTC style."""
    s = str(sym or "").upper().strip()
    s = s.replace("_OTC", "-OTC").replace(" OTC", "-OTC")
    s = s.replace("/", "").replace(" ", "")
    return s


def _reset_client():
    api = _CLIENT.get("api")
    _CLIENT["api"] = None
    _CLIENT["map"] = {}
    if api is not None:
        try:
            api.disconnect()
        except Exception:
            pass


def _ensure_client():
    """Return a connected BinaryOptionsToolsV2 client, or None."""
    ssid = load_ssid()
    if not ssid:
        return None
    api = _CLIENT.get("api")
    if api is not None and _CLIENT.get("ssid") == ssid:
        return api
    _reset_client()
    try:
        from BinaryOptionsToolsV2 import PocketOption
    except Exception as e:
        print(f"[pocket] library missing ({e}) - live features disabled")
        return None
    try:
        api = PocketOption(ssid)
        try:
            api.connect()
        except Exception as e:
            print(f"[pocket] connect() warning: {e}")
        try:
            api.wait_for_assets(timeout=30)
        except Exception:
            pass
        _CLIENT["api"] = api
        _CLIENT["ssid"] = ssid
        _CLIENT["map"] = {}
        try:
            m = {}
            for a in api.active_assets():
                sym = a.get("symbol")
                if sym:
                    m[_norm(sym)] = sym
            _CLIENT["map"] = m
            print(f"[pocket] connected - {len(m)} assets mapped"
                  + ("" if m else " (asset list empty - SSID may be stale)"))
        except Exception as e:
            print(f"[pocket] asset list failed: {e} - SSID may be expired")
        return api
    except Exception:
        return None


def feed_symbol(asset):
    """Map a normalized asset (EURUSD-OTC) to the feed symbol (EURUSD_otc)."""
    key = _norm(asset)
    sym = (_CLIENT.get("map") or {}).get(key)
    if sym:
        return sym
    if key.endswith("-OTC"):
        return key[:-4] + "_otc"
    return key


def refresh_payouts(force=False):
    """Return {ASSET: payout%}. Empty dict if unavailable. Never raises."""
    now = time.time()
    if not force and _PAY["data"] and now - _PAY["t"] < _TTL:
        return dict(_PAY["data"])
    with _LOCK:
        api = _ensure_client()
        if api is None:
            return dict(_PAY["data"])
        out = {}
        try:
            _throttle("payout")
            raw = _retry(api.payout)
            if isinstance(raw, dict):
                for k, v in raw.items():
                    try:
                        val = v.get("payout", v.get("payment")) if isinstance(v, dict) else v
                        if val is not None and float(val) > 0:
                            f = float(val)
                            out[_norm(k)] = f if f > 1.5 else f * 100
                    except Exception:
                        continue
        except Exception:
            _reset_client()
            return dict(_PAY["data"])
        if out:
            _PAY["t"] = now
            _PAY["data"] = out
        return dict(out)


def get_payout(asset, default=None):
    data = refresh_payouts()
    v = data.get(_norm(asset))
    return v if v is not None else default


def _fetch_candles(api, sym, period, count, block=True):
    """Historical candles, version-proof.

    Uses the legacy `get_candles` (fast, non-blocking) while it exists, and
    transparently falls back to the non-deprecated live backfill
    (`get_candles_live`) once the library removes it. Never leaves a hang: the
    live path is only tried after the legacy call fails.

    `block=False` returns [] instead of sleeping when the per-minute candle
    budget is full - used by low-priority background work so interactive scans
    always get the budget first."""
    count = int(count)
    if not _throttle("candles", block=block):
        return []
    legacy = getattr(api, "get_candles", None)
    partial = []
    if legacy is not None:
        try:
            # the SDK deprecates get_candles in favour of get_candles_live,
            # which is a much heavier streaming API; keep using the fast path
            # while it exists and silence the per-call warning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                data = list(legacy(sym, int(period), count) or [])
            # small requests are fine with the legacy window; large ones must
            # prove they actually returned the extra history
            if data and (len(data) >= count or count <= 260):
                if count >= 500:
                    LAST_FETCH.update({"sym": sym, "period": int(period),
                                       "want": count, "legacy": len(data),
                                       "live": 0})
                    print(f"[candles] {sym} {period}s: want {count}, "
                          f"legacy {len(data)} (ok)")
                return data
            partial = data
        except Exception:
            pass
    if not _throttle("candles", block=block):
        return partial
    try:
        hours = max(1.0, (int(period) * count) / 3600.0 * 1.25)
        it = api.get_candles_live(sym, int(period), hours=hours, max_rows=count)
        if count < 500:
            closed, _forming = next(it)
            try:
                it.stop()
            except Exception:
                pass
            return list(closed or []) or partial
        # large request: the stream may yield history in several batches, so
        # drain them (bounded) instead of keeping only the first one
        rows, batches = [], 0
        first = [None]

        def _pull():
            nonlocal batches
            try:
                for closed, forming in it:
                    if first[0] is None:
                        first[0] = (closed, forming)
                    rows.extend(list(closed or []))
                    batches += 1
                    if len(rows) >= count:
                        break
            except Exception:
                pass

        th = threading.Thread(target=_pull, daemon=True)
        th.start()
        th.join(timeout=5.0)
        try:
            it.stop()
        except Exception:
            pass
        th.join(timeout=1.0)
        if not rows and first[0]:
            rows = list(first[0][0] or [])
        LAST_FETCH.update({"sym": sym, "period": int(period),
                           "want": count, "legacy": len(partial),
                           "live": len(rows), "batches": batches})
        print(f"[candles] {sym} {period}s: want {count}, "
              f"legacy {len(partial)}, live {len(rows)} "
              f"({batches} batches)")
        return list(rows) or partial
    except Exception as e:
        if count >= 500:
            LAST_FETCH.update({"sym": sym, "period": int(period),
                               "want": count, "legacy": len(partial),
                               "live": -1, "batches": 0})
            print(f"[candles] {sym} {period}s: want {count}, "
                  f"legacy {len(partial)}, live failed ({e})")
        return partial


def get_closes(asset, period=60, offset=200, ttl=_CLOSES_TTL):
    """Real closes for an asset+timeframe, or None if unavailable. Never raises."""
    key = f"{_norm(asset)}#{int(period)}"
    now = time.time()
    cached = _CLOSES.get(key)
    if cached and now - cached[0] < ttl:
        return list(cached[1])
    with _LOCK:
        api = _ensure_client()
        if api is None:
            return list(cached[1]) if cached else None
        sym = feed_symbol(_norm(asset))
        try:
            data = _fetch_candles(api, sym, period, offset)
        except Exception:
            _reset_client()
            return list(cached[1]) if cached else None
        if not data:
            return list(cached[1]) if cached else None
        try:
            closes = [float(c.get("close")) for c in data if c.get("close") is not None]
        except Exception:
            return list(cached[1]) if cached else None
        if len(closes) < 30:
            return list(cached[1]) if cached else None
        if not cached or len(closes) >= len(cached[1]):
            _CLOSES[key] = (now, closes)
        return list(closes)


def peek_closes(asset, period=60, max_age=_CLOSES_TTL + 25):
    """Return cached closes WITHOUT any network call, or None. Never raises."""
    key = f"{_norm(asset)}#{int(period)}"
    c = _CLOSES.get(key)
    if c and time.time() - c[0] < max_age:
        return list(c[1])
    return None


def get_ohlcv(asset, period=60, offset=220, ttl=_CLOSES_TTL):
    """Real OHLC candles for pattern detection, or None. Never raises."""
    key = f"{_norm(asset)}#{int(period)}"
    now = time.time()
    cached = _OHLC.get(key)
    if cached and now - cached[0] < ttl:
        return [dict(r) for r in cached[1]]
    with _LOCK:
        api = _ensure_client()
        if api is None:
            return [dict(r) for r in cached[1]] if cached else None
        sym = feed_symbol(_norm(asset))
        try:
            data = _fetch_candles(api, sym, period, offset)
        except Exception:
            _reset_client()
            return [dict(r) for r in cached[1]] if cached else None
        if not data:
            return [dict(r) for r in cached[1]] if cached else None
        rows = []
        for c in data:
            try:
                cl = float(c["close"])
            except Exception:
                continue
            def _f(key, default):
                try:
                    return float(c.get(key, default))
                except Exception:
                    return default
            rows.append({"o": _f("open", cl), "h": _f("high", cl),
                         "l": _f("low", cl), "c": cl})
        if len(rows) < 30:
            return [dict(r) for r in cached[1]] if cached else None
        if not cached or len(rows) >= len(cached[1]):
            _OHLC[key] = (now, rows)
        return [dict(r) for r in rows]


def peek_ohlcv(asset, period=60, max_age=_CLOSES_TTL + 25):
    """Cached OHLC only, no network. Never raises."""
    key = f"{_norm(asset)}#{int(period)}"
    c = _OHLC.get(key)
    if c and time.time() - c[0] < max_age:
        return [dict(r) for r in c[1]]
    return None


# ===== live streaming feed (real-time candles + the forming candle) =====
_LIVE = {}        # key -> (updated_ts, closed_candles, forming_candle)
_FEEDS = {}       # key -> {"stop": Event, "thread": Thread, "last_used": ts}
_LIVE_LOCK = threading.Lock()
_MAX_FEEDS = 3
_LIVE_MAX_AGE = 25.0   # live data older than this counts as stale


def _feed_worker(key, asset, period, max_rows, stop):
    """Background thread: stream gap-free candles (closed + forming) on every tick."""
    try:
        with _LOCK:
            api = _ensure_client()
        if api is None:
            return
        sym = feed_symbol(_norm(asset))
        hours = max(1.0, (period * max_rows) / 3600.0 * 1.25)
        it = api.get_candles_live(sym, int(period), hours=hours, max_rows=int(max_rows))
    except Exception:
        return
    while not stop.is_set():
        try:
            closed, forming = next(it)
        except Exception:
            break
        now = time.time()
        with _LIVE_LOCK:
            _LIVE[key] = (now, list(closed or []), dict(forming) if forming else None)
            if key in _FEEDS:
                _FEEDS[key]["last_used"] = now


def start_live(asset, period=60, max_rows=220, wait=10.0):
    """Start (or keep) the real-time candle stream for an asset+timeframe.
    Returns True once live data is flowing (waits up to `wait` seconds)."""
    with _LOCK:
        api = _ensure_client()
    if api is None:
        return False
    key = f"{_norm(asset)}#{int(period)}"
    now = time.time()
    with _LIVE_LOCK:
        feed = _FEEDS.get(key)
        if feed and feed["thread"].is_alive():
            feed["last_used"] = now
        else:
            while len(_FEEDS) >= _MAX_FEEDS:
                oldest = min(_FEEDS, key=lambda k: _FEEDS[k]["last_used"])
                _FEEDS[oldest]["stop"].set()
                _FEEDS.pop(oldest)
            stop_ev = threading.Event()
            th = threading.Thread(target=_feed_worker,
                                  args=(key, asset, int(period), max_rows, stop_ev),
                                  daemon=True)
            _FEEDS[key] = {"stop": stop_ev, "thread": th, "last_used": now}
            th.start()
    deadline = time.time() + max(0.0, wait)
    while True:
        with _LIVE_LOCK:
            row = _LIVE.get(key)
        if row and time.time() - row[0] < _LIVE_MAX_AGE:
            return True
        if time.time() >= deadline:
            return False
        time.sleep(0.25)


def stop_live(asset=None, period=None):
    """Stop one feed (asset+period) or all feeds. Never raises."""
    with _LIVE_LOCK:
        keys = list(_FEEDS) if asset is None else [f"{_norm(asset)}#{int(period or 60)}"]
        for key in keys:
            feed = _FEEDS.pop(key, None)
            if feed:
                feed["stop"].set()
        if asset is None:
            _LIVE.clear()


def _live_row(asset, period, max_age=_LIVE_MAX_AGE):
    key = f"{_norm(asset)}#{int(period)}"
    with _LIVE_LOCK:
        row = _LIVE.get(key)
    if not row or time.time() - row[0] > max_age:
        return None
    return row


def live_closes(asset, period=60, max_age=_LIVE_MAX_AGE, include_forming=True):
    """Closes from the live stream; with include_forming=False the last bar is
    the most recent CLOSED candle (what signal evaluation should use).
    None if no fresh stream."""
    row = _live_row(asset, period, max_age)
    if not row:
        return None
    _, closed, forming = row
    closes = []
    for c in closed:
        try:
            closes.append(float(c["close"]))
        except Exception:
            continue
    if forming and include_forming:
        try:
            closes.append(float(forming["close"]))
        except Exception:
            pass
    return closes if len(closes) >= 30 else None


def live_ohlcv(asset, period=60, max_age=_LIVE_MAX_AGE, include_forming=True):
    """OHLC from the live stream; with include_forming=False the last row is
    the most recent CLOSED candle. None if no fresh stream."""
    row = _live_row(asset, period, max_age)
    if not row:
        return None
    _, closed, forming = row
    rows = []
    for c in list(closed) + ([forming] if forming and include_forming else []):
        try:
            rows.append({"o": float(c["open"]), "h": float(c["high"]),
                         "l": float(c["low"]), "c": float(c["close"])})
        except Exception:
            continue
    return rows if len(rows) >= 30 else None


def feed_status(asset, period=60):
    """('live', age_seconds) | ('snapshot', age_seconds) | ('none', None)"""
    row = _live_row(asset, period)
    if row:
        return "live", time.time() - row[0]
    cached = _CLOSES.get(f"{_norm(asset)}#{int(period)}")
    if cached and time.time() - cached[0] < _CLOSES_TTL + 200:
        return "snapshot", time.time() - cached[0]
    return "none", None


_CAT = {"t": 0.0, "data": []}

def asset_catalog():
    """[{norm, symbol, payout, otc, type, name}] from the live asset list. Never raises."""
    now = time.time()
    if _CAT["data"] and now - _CAT["t"] < 300:
        return list(_CAT["data"])
    with _LOCK:
        api = _ensure_client()
        if api is None:
            return list(_CAT["data"])
        try:
            rows = api.active_assets() or []
        except Exception:
            _reset_client()
            return list(_CAT["data"])
    out = []
    for a in rows:
        try:
            sym = a.get("symbol")
            if not sym:
                continue
            out.append({"norm": _norm(sym), "symbol": sym,
                        "payout": float(a.get("payout") or 0),
                        "otc": bool(a.get("is_otc")),
                        "type": str(a.get("asset_type") or ""),
                        "name": str(a.get("name") or sym)})
        except Exception:
            continue
    if out:
        _CAT["t"] = now
        _CAT["data"] = out
    return list(out)


def account_trades(max_tries=5):
    """Recent closed deals from the linked account (read-only). Never raises.
    Returns [{id, asset, entry, exit, profit, win, refund, demo, open_ts, close_ts, tf}]."""
    api = _ensure_client()
    if api is None:
        return []
    deals = {}
    for _ in range(max_tries):
        try:
            deals = api.closed_deals() or {}
        except Exception:
            deals = {}
        if deals:
            break
        time.sleep(1.0)
    out = []
    items = deals.items() if isinstance(deals, dict) else list(enumerate(deals))
    for did, d in items:
        try:
            if not isinstance(d, dict):
                continue
            entry = float(d.get("openPrice") or 0)
            exit_ = float(d.get("closePrice") or 0)
            profit = float(d.get("profit") or 0)
            ots = int(d.get("openTimestamp") or 0)
            cts = int(d.get("closeTimestamp") or 0)
            out.append({
                "id": str(d.get("id") or did),
                "asset": _norm(d.get("asset")),
                "entry": entry, "exit": exit_,
                "profit": profit,
                "win": profit > 0,
                "refund": profit == 0,
                "demo": bool(d.get("isDemo")),
                "open_ts": ots,
                "close_ts": cts,
                "tf": max(5, (cts - ots)) if cts > ots else 60,
            })
        except Exception:
            continue
    out.sort(key=lambda x: x["open_ts"])
    return out


def place_trade(asset, direction, amount, tf):
    """Place a CALL/PUT on the linked account. READ-WRITE - real money on live
    accounts. Never raises. Returns (ok, message).

    Hard-disabled unless the ALLOW_TRADING env var is explicitly set to "1", so
    a display-only bot can never place an order by accident."""
    if os.getenv("ALLOW_TRADING") != "1":
        return False, ("Trading is disabled. This is a display-only bot; set "
                       "ALLOW_TRADING=1 only if you really intend to trade.")
    # separate guard: refuse REAL-money accounts unless explicitly unlocked
    if os.getenv("ALLOW_REAL") != "1":
        demo = None
        try:
            api_probe = _ensure_client()
            if api_probe is not None:
                demo = bool(api_probe.is_demo())
        except Exception:
            demo = None
        if demo is False:
            return False, ("Linked account is a REAL account. Order refused. "
                           "Set ALLOW_REAL=1 only if you truly want real "
                           "money orders.")
    api = _ensure_client()
    if api is None:
        return False, "Account not linked (save SSID first)."
    sym = feed_symbol(_norm(asset))
    try:
        amt = float(amount)
        secs = int(tf)
    except Exception:
        return False, "Invalid amount or timeframe."
    if amt <= 0 or secs <= 0:
        return False, "Invalid amount or timeframe."
    try:
        if str(direction).upper() == "CALL":
            tid, details = api.buy(sym, amt, secs)
        else:
            tid, details = api.sell(sym, amt, secs)
        return True, str(tid)
    except Exception as e:
        return False, str(e)


def account_balance():
    api = _ensure_client()
    if api is None:
        return None
    try:
        return float(api.balance())
    except Exception:
        return None


def account_is_demo():
    api = _ensure_client()
    if api is None:
        return None
    try:
        return bool(api.is_demo())
    except Exception:
        return None


def reset_all():
    stop_live()
    with _LOCK:
        _reset_client()


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--save":
        save_ssid(sys.argv[2])
    elif len(sys.argv) >= 2 and sys.argv[1] == "--show":
        print("SSID saved:", "yes" if load_ssid() else "no")
        data = refresh_payouts(force=True)
        if not data:
            print("No payouts read. Check SSID format / library.")
        else:
            for k in sorted(data):
                print(f"{k}: {data[k]:.0f}%")
    elif len(sys.argv) >= 3 and sys.argv[1] == "--candles":
        asset = sys.argv[2]
        closes = get_closes(asset)
        if not closes:
            print(f"No candles read for {asset} (want e.g. EURUSD or EURUSD-OTC).")
        else:
            print(f"{asset}: {len(closes)} closes, last 5: {[round(c,5) for c in closes[-5:]]}")
    else:
        print("Use (local PowerShell only):")
        print('  python pocket_payout.py --save "FULL_SSID"')
        print("  python pocket_payout.py --show")
        print("  python pocket_payout.py --candles EURUSD")
