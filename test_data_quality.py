"""
Regression tests for the data-quality bugs found in the stock/OTC audit.

Run:  python test_data_quality.py
"""
import asyncio
import time

import numpy as np
import pandas as pd

from engine import datafeed as d
from engine import po_feed
from engine.patterns import recognize
from engine.signals import analyze


def frame(closes, *, tz="UTC", end_offset_min=0):
    n = len(closes)
    closes = np.asarray(closes, dtype=float)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    idx = pd.date_range(
        end=pd.Timestamp.now(tz=tz).floor("min") - pd.Timedelta(minutes=end_offset_min),
        periods=n, freq="1min")
    return pd.DataFrame({"open": opens, "high": np.maximum(opens, closes) + 1e-5,
                         "low": np.minimum(opens, closes) - 1e-5,
                         "close": closes, "volume": np.ones(n)}, index=idx)


# 1) tweezer patterns must not fire on unrelated candles ----------------------
def candles(rows, tz="UTC", end_offset_min=0):
    """rows: list of (open, high, low, close)."""
    idx = pd.date_range(
        end=pd.Timestamp.now(tz=tz).floor("min") - pd.Timedelta(minutes=end_offset_min),
        periods=len(rows), freq="1min")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"],
                        index=idx).assign(volume=1.0)


rng0 = np.random.default_rng(3)
noise = 1.1000 + np.cumsum(rng0.normal(0, 2e-5, 80))
prev = [(p, p + 1e-5, p - 1e-5, p) for p in noise]


def add_last(prev_low, last_low):
    return candles(prev + [
        (1.1005, 1.1010, prev_low, 1.1000),      # bearish
        (1.1000, 1.1012, last_low, 1.1010),      # bullish
    ])


names = [p.name for p in recognize(add_last(prev_low=1.0995, last_low=1.0985))]
assert "Tweezer Bottom" not in names, names
print("tweezer: far-apart lows rejected        OK")

names = [p.name for p in recognize(add_last(prev_low=1.0995, last_low=1.0995))]
assert "Tweezer Bottom" in names, names
print("tweezer: equal lows accepted            OK")


# 2) synthetic fallback must use realistic price levels -----------------------
jpy = d._synthetic("USD/JPY", "OTC")
assert 100 < float(jpy["close"].iloc[-1]) < 220, float(jpy["close"].iloc[-1])
eur = d._synthetic("EUR/USD", "OTC")
assert 0.8 < float(eur["close"].iloc[-1]) < 1.5
print("synthetic: realistic anchors           OK")


# 3) stale data must never be labelled live -----------------------------------
stale = frame([1.10] * 70, end_offset_min=600)
assert d.bar_age(stale) is not None and d.bar_age(stale) > d.STALE_SECONDS
assert d.bar_age(frame([1.10] * 70)) < 60
print("freshness helper                       OK")


# 4) market closed -> refusal, not a fabricated signal ------------------------
orig_yahoo, orig_po = d._yahoo, po_feed.candles
try:
    d._yahoo = lambda pair: stale
    po_feed.candles = lambda pair, otc=True, bars=240: None
    df, note = asyncio.run(d.get_candles("EUR/USD", "STOCK"))
    assert df is None and "Market closed" in note, (df, note)
    print("stock: closed market refuses           OK")

    # broker spot also stale -> still "closed", never "live spot"
    d._CACHE.clear()
    d._yahoo = lambda pair: stale
    po_feed.candles = lambda pair, otc=True, bars=240: stale
    df, note = asyncio.run(d.get_candles("USD/JPY", "STOCK"))
    assert df is None and "Market closed" in note, (df, note)
    print("stock: stale PO spot refuses           OK")

    # fresh but fewer than 60 bars (market just opened) -> refuse, not synthetic
    d._CACHE.clear()
    d._yahoo = lambda pair: frame([1.1000] * 10)
    df, note = asyncio.run(d.get_candles("EUR/USD", "STOCK"))
    assert df is None and "Not enough 1-minute history" in note, (df, note)
    print("stock: short fresh history refuses     OK")

    # network failure (no data at all) still falls back, clearly labelled
    d._CACHE.clear()
    d._yahoo = lambda pair: None
    po_feed.candles = lambda pair, otc=True, bars=240: None
    df, note = asyncio.run(d.get_candles("GBP/USD", "STOCK"))
    assert df is not None and "synthetic" in note.lower(), note
    print("stock: network failure -> labelled synthetic OK")
finally:
    d._yahoo, po_feed.candles = orig_yahoo, orig_po
    d._CACHE.clear()


# 5) cached notes must describe the data they carry ---------------------------
orig_yahoo = d._yahoo
try:
    live = frame([1.10 + i * 1e-5 for i in range(80)])
    d._yahoo = lambda pair: live
    d._CACHE.clear()
    _, note1 = asyncio.run(d.get_candles("EUR/USD", "STOCK"))
    _, note2 = asyncio.run(d.get_candles("EUR/USD", "STOCK"))   # cached path
    assert note1 == note2 and "Yahoo" in note2, (note1, note2)
    print("cache: note survives caching           OK")
finally:
    d._yahoo = orig_yahoo
    d._CACHE.clear()


# 6) payout lookup must not borrow another asset's number ---------------------
class _FakeAPI:
    def __init__(self, mapping, active=None):
        self.mapping = mapping
        self.active = active or []

    def is_connected(self):
        return True

    def payout(self, asset=None):
        if asset is not None:
            return self.mapping.get(asset) if isinstance(self.mapping, dict) else None
        return self.mapping

    def active_assets(self):
        return self.active


orig_api = po_feed._CLIENT.get("api")
try:
    po_feed._CLIENT["api"] = _FakeAPI({"GBPJPY_otc": 92, "EURUSD_otc": 78})
    assert po_feed.payout_for("EUR/USD") == 78.0
    po_feed._CLIENT["api"] = _FakeAPI({"GBPJPY_otc": 92})
    assert po_feed.payout_for("EUR/USD") is None      # never a random asset
    print("payout: exact asset only               OK")

    # Server shape: the bulk map comes back empty, but the active-asset list
    # (loaded at connect) still carries the broker's payout for each symbol.
    active = [{"symbol": "EURUSD_otc", "payout": 83, "is_active": True},
              {"symbol": "GBPJPY_otc", "payout": 79, "is_active": True}]
    po_feed._CLIENT["api"] = _FakeAPI({}, active)
    assert po_feed.payout_for("EUR/USD") == 83.0
    assert po_feed.payout_for("GBP/JPY") == 79.0
    assert po_feed.payout_for("USD/CHF") is None
    assert po_feed.payouts().get("EURUSD-OTC") == 83.0
    print("payout: asset-list fallback            OK")

    # Library 0.2.16 made payout(asset) mandatory: payout() raises TypeError.
    class _NewSignatureAPI(_FakeAPI):
        def payout(self, asset):                  # no default -> no-arg call raises
            return self.mapping.get(asset)

    po_feed._CLIENT["api"] = _NewSignatureAPI({}, active)
    assert po_feed.payout_for("EUR/USD") == 83.0
    assert po_feed.payouts().get("EURUSD-OTC") == 83.0
    print("payout: 0.2.16 signature handled       OK")

    # Library builds differ in how a percentage is encoded; all mean the same.
    norm = po_feed._normalize_payouts({"EURUSD_otc": "92%", "GBPUSD_otc": 0.88,
                                       "USDJPY_otc": 0, "AUDUSD_otc": None})
    assert norm == {"EURUSD-OTC": 92.0, "GBPUSD-OTC": 88.0}, norm
    print("payout: percent encodings               OK")
finally:
    po_feed._CLIENT["api"] = orig_api


# 7) candle rows are de-duplicated / sorted ----------------------------------
rows = [{"time": 1000 + 60 * i, "open": 1.1, "high": 1.11, "low": 1.09,
         "close": 1.10} for i in range(70)]
rows.append(dict(rows[-1], close=1.105))          # duplicate newest bar
rows.append({"time": 940, "open": 1.1, "high": 1.11, "low": 1.09, "close": 1.1})
df = po_feed._rows_to_frame(rows)
assert df is not None and df.index.is_monotonic_increasing
assert not df.index.duplicated().any()
assert abs(float(df["close"].iloc[-1]) - 1.105) < 1e-12
print("candles: dedupe + sort                 OK")

# corrupt rows (high below close) must be dropped, not analysed
bad = [dict(r) for r in rows]
bad[10] = dict(bad[10], high=1.0, low=0.5, close=1.10, open=1.10)
df_bad = po_feed._rows_to_frame(bad)
assert df_bad is not None and len(df_bad) == len(df) - 1, (len(df_bad), len(df))
print("candles: impossible bars dropped       OK")


# 7b) candles() prefers the fresh live path, and refuses a dead feed ----------
fresh = frame([1.1000 + i * 1e-5 for i in range(80)])
dead = frame([1.1000] * 80, end_offset_min=60)     # an hour old -> hard stale
orig = (po_feed._fetch_fresh, po_feed._ensure_client, po_feed.load_ssid)
try:
    po_feed.load_ssid = lambda: "fake-ssid"
    po_feed._ensure_client = lambda: None          # no shared client, no network
    po_feed._CACHE.clear()
    po_feed._fetch_fresh = lambda ssid, sym, bars: fresh
    got = po_feed.candles("EUR/USD", True)
    assert got is not None and len(got) == len(fresh), got
    assert abs(float(got["close"].iloc[-1]) - float(fresh["close"].iloc[-1])) < 1e-12

    po_feed._CACHE.clear()
    po_feed._fetch_fresh = lambda ssid, sym, bars: dead
    assert po_feed.candles("EUR/USD", True) is None
    assert "delayed" in po_feed.last_issue(), po_feed.last_issue()
    print("candles: fresh preferred, dead rejected OK")
finally:
    (po_feed._fetch_fresh, po_feed._ensure_client, po_feed.load_ssid) = orig
    po_feed._CACHE.clear()



# 8) OTC must refuse instead of inventing prices -----------------------------
orig_c, orig_r, orig_i, orig_as = (po_feed.candles, po_feed.is_ready,
                                   po_feed.last_issue, po_feed.asset_status)
try:
    d._CACHE.clear()
    po_feed.candles = lambda pair, otc=True, bars=240: None
    po_feed.is_ready = lambda: True
    po_feed.last_issue = lambda: "USDXXX_otc not in PO's active list"
    df, note = asyncio.run(d.get_candles("AUD/CAD", "OTC"))
    assert df is None, "OTC feed failure must not produce synthetic candles"
    assert "not sending live OTC" in note and "guess" in note, note
    print("otc: feed failure refuses               OK")

    # history exists but the asset is not tradeable -> refuse too
    bars = frame([1.1000 + i * 1e-5 for i in range(80)])
    d._CACHE.clear()
    po_feed.candles = lambda pair, otc=True, b=240: bars
    po_feed.asset_status = lambda pair, otc=True: (
        False, f"{pair} OTC is not in Pocket Option's active asset list right now")
    df, note = asyncio.run(d.get_candles("USD/BRL", "OTC"))
    assert df is None and "active asset list" in note, (df, note)
    print("otc: non-tradeable asset refuses        OK")
finally:
    (po_feed.candles, po_feed.is_ready,
     po_feed.last_issue, po_feed.asset_status) = orig_c, orig_r, orig_i, orig_as
    d._CACHE.clear()


# 9) balanced confluence must not produce a SELL -----------------------------
# A flat series gives no directional votes at all -> no signal.
flat = frame([1.1000] * 80)
assert analyze(flat, "1 minute") is None
print("signals: no edge -> no signal          OK")


# 9) no crashes across random markets ----------------------------------------
rng = np.random.default_rng(5)
for s in range(30):
    p = 1.1 * (1 + np.cumsum(rng.normal(0, 6e-5, 120)))
    sig = analyze(frame(list(p)), "1 minute")
    if sig is not None:
        assert 55 <= sig.accuracy <= 97 and sig.direction in ("BUY", "SELL")
print("signals: 30 random markets             OK")

print("\nDATA QUALITY TESTS: ALL OK")
