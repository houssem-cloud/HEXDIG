"""
Pocket Option LIVE feed over the broker's own WebSocket (socket.io), authenticated
with the user's `ssid` session string — this is what finally provides REAL OTC candles.

How the ssid works:
    While logged into the Pocket Option web platform, the browser opens a
    socket.io websocket. The first outgoing message starts with
        42["auth", {"session": "a:4:{...}", "isDemo": ..., "uid": ..., "platform": 2}]
    The full `42["auth", {...}]` string (or at least the dict inside) is what
    unofficial clients call the "ssid". Feed it to this module and it reconnects
    with the same session, subscribes to asset streams, and rebuilds M1 candles
    from live ticks.

Honest engineering notes:
    * PO does not publish this protocol. Event names/endpoints are reverse-
      engineered from the web client and DO change — every incoming frame is
      logged (DEBUG) so a changing protocol can be re-tuned in minutes, and any
      unparseable frame never crashes the bot.
    * A broker may flag automation (ToS). Use the DEMO ssid (isDemo: 1) first.
    * If PO is unreachable, engine.datafeed falls back exactly as before.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone

log = logging.getLogger("po-ws")

# ----------------------------------------------------------- configuration
# socket.io websocket endpoints (the web client rotates these; add more as PO changes)
ENDPOINTS = [
    "wss://api-l.po.market/socket.io/?EIO=4&transport=websocket",
    "wss://api-eu.po.market/socket.io/?EIO=4&transport=websocket",
    "wss://dc1-fdyp73x-wss.po.market/socket.io/?EIO=4&transport=websocket",
]

# Engine pair -> Pocket Option asset names (OTC first, fallback spot)
ASSET_MAP = {
    "EUR/USD": ["EURUSD_otc", "EURUSD"],
    "USD/JPY": ["USDJPY_otc", "USDJPY"],
    "GBP/USD": ["GBPUSD_otc", "GBPUSD"],
    "AUD/USD": ["AUDUSD_otc", "AUDUSD"],
    "USD/CAD": ["USDCAD_otc", "USDCAD"],
    "USD/CHF": ["USDCHF_otc", "USDCHF"],
    "GBP/JPY": ["GBPJPY_otc", "GBPJPY"],
    "EUR/JPY": ["EURJPY_otc", "EURJPY"],
    "AUD/CAD": ["AUDCAD_otc", "AUDCAD"],
    "EUR/CAD": ["EURCAD_otc", "EURCAD"],
    "GBP/CAD": ["GBPCAD_otc", "GBPCAD"],
    "CAD/CHF": ["CADCHF_otc", "CADCHF"],
    "EUR/GBP": ["EURGBP_otc", "EURGBP"],
    "AUD/JPY": ["AUDJPY_otc", "AUDJPY"],
    "NZD/USD": ["NZDUSD_otc", "NZDUSD"],
    "CHF/JPY": ["CHFJPY_otc", "CHFJPY"],
}
REVERSE_MAP = {a: p for p, alts in ASSET_MAP.items() for a in alts}

MAX_BARS = 600                      # keep ~10h of M1 bars per asset
FRESH_ALLOWANCE = 180               # seconds a candle set may lag to be "live"


def normalize_ssid(raw: str) -> str:
    """Accept either the full 42["auth",{...}] frame or just the dict JSON."""
    raw = raw.strip()
    if raw.startswith("42["):
        return raw
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("ssid must be the auth frame or a JSON object")
    return json.dumps(["auth", obj])


def auth_frame_to_session(raw: str) -> tuple[str, str]:
    """Returns (session_string, demo_flag) from whatever ssid format was given."""
    if raw.startswith("42["):
        payload = json.loads(raw[2:])[1]
    else:
        payload = json.loads(raw)
    return payload.get("session", ""), payload.get("isDemo", 1)


def frame_for(asset: str) -> str:
    """Subscription frame — same shape the web client sends."""
    return '42["subfor","%s"]' % asset


class POFeed:
    def __init__(self, ssid_raw: str):
        self.ssid = normalize_ssid(ssid_raw)
        self.session, self.demo = auth_frame_to_session(ssid_raw)  # best effort
        self._bars: dict[str, dict[int, dict]] = {}                 # asset -> minute -> bar
        self.connected = False
        self.endpoint = ""
        self.ticks = 0
        self.last_tick_ts: float | None = None
        self._task: asyncio.Task | None = None
        self._stop = False

    # ------------------------------------------------------------ lifecycle
    async def run(self):
        attempt = 0
        while not self._stop:
            try:
                await self._session_once()
            except Exception as e:
                log.warning("PO feed dropped: %s — reconnecting", e)
            self.connected = False
            attempt += 1
            await asyncio.sleep(min(30, 2 ** min(attempt, 5)))

    def stop(self):
        self._stop = True
        if self._task:
            self._task.cancel()

    async def _session_once(self):
        import websockets
        last_err = None
        for url in ENDPOINTS:
            try:
                log.info("PO feed: connecting %s (demo=%s)", url.split('?')[0], self.demo)
                async with websockets.connect(url, open_timeout=10,
                                              additional_headers={"Origin": "https://pocketoption.com"}) as ws:
                    self.endpoint = url
                    # socket.io handshake v4: "40", then authenticate with the ssid frame
                    await ws.send("40")
                    auth = self.ssid if self.ssid.startswith("42") else "42" + self.ssid
                    await ws.send(auth)
                    self.connected = True
                    log.info("PO feed: authenticated, subscribing to %d assets", len(ASSET_MAP))
                    for asset in set(a for alts in ASSET_MAP.values() for a in alts):
                        await ws.send('42["subfor","%s"]' % asset)
                    async for raw in ws:
                        self._handle_frame(raw, ws)
            except Exception as e:
                last_err = e
                log.debug("endpoint failed: %s — %s", url, e)
        if last_err:
            raise last_err
        raise RuntimeError("all PO endpoints failed")

    # ------------------------------------------------------------ framing
    def _handle_frame(self, raw: str, ws):
        try:
            if raw.startswith("2"):                     # socket.io PING
                asyncio.create_task(ws.send("3"))
                return
            if not raw.startswith("42"):                # 40/451 frames logged for tuning
                log.debug("PO frame: %.120s", raw)
                return
            data = json.loads(raw[2:])
            if not isinstance(data, list) or len(data) < 2:
                return
            event, payload = data[0], data[1]
            log.debug("PO event: %s (%.80s)", event, str(payload))
            if event in ("updateStream", "stream", "updateTick"):
                self._ingest_ticks(payload)
            elif event in ("loadHistoryPeriod", "successloadHistoryPeriod"):
                self._ingest_history(payload)
            else:
                log.info("PO event (unmapped, logged for tuning): %s", event)
        except Exception:
            log.exception("frame parse error: %.120s", raw)

    # ------------------------------------------------------------ ingestion
    def _ingest_ticks(self, payload):
        """
        Accepts whatever tick shape PO uses:
            [[ts_ms, price], ...] | {asset:.., price:.., time:..} | {asset: [...]}
        """
        def tick(obj):
            if isinstance(obj, (list, tuple)) and len(obj) >= 2:
                ts, price = float(obj[0]) / 1000.0 if float(obj[0]) > 1e10 else float(obj[0]), float(obj[1])
                return None, ts, price
            if isinstance(obj, dict):
                ts = obj.get("ts") or obj.get("time") or obj.get("timestamp")
                price = obj.get("price") or obj.get("quote")
                asset = obj.get("asset") or obj.get("symbol") or obj.get("pair")
                if ts and price:
                    ts = float(ts) / 1000.0 if float(ts) > 1e10 else float(ts)
                    return asset, ts, float(price)
            return None, None, None

        for item in (payload if isinstance(payload, list) else [payload]):
            asset, ts, price = tick(item)
            if ts and price:
                self._add_tick(asset, ts, price)

    def _ingest_history(self, payload):
        log.debug("history payload: %.200s", str(payload))  # tune when encountered

    def _add_tick(self, asset: str, ts: float, price: float):
        self.ticks += 1
        self.last_tick_ts = time.time()
        minute = int(ts // 60)
        asset = asset or "?"           # some stream events omit the asset — logged above
        b = self._bars.setdefault(asset, {})
        bar = b.get(minute)
        if bar is None:
            b[minute] = {"open": price, "high": price, "low": price, "close": price}
        else:
            bar["high"] = max(bar["high"], price)
            bar["low"] = min(bar["low"], price)
            bar["close"] = price
        for a in list(self._bars):                       # cap memory
            if len(self._bars[a]) > MAX_BARS:
                self._bars[a] = dict(sorted(self._bars[a].items())[-MAX_BARS:])

    # ------------------------------------------------------------ consumers
    def candles(self, pair: str):
        """M1 DataFrame for a pair (tries _otc first), or None if not enough data."""
        import pandas as pd
        if not self.connected:
            return None
        for asset in ASSET_MAP.get(pair, [pair.replace('/', '') + "_otc"]):
            b = self._bars.get(asset)
            if not b or len(b) < 60:
                continue
            newest = max(b.keys())
            if time.time() - newest * 60 > FRESH_ALLOWANCE:
                continue                                  # stale
            idx = sorted(b.keys())
            df = pd.DataFrame(
                {
                    "open": [b[m]["open"] for m in idx],
                    "high": [b[m]["high"] for m in idx],
                    "low": [b[m]["low"] for m in idx],
                    "close": [b[m]["close"] for m in idx],
                    "volume": [1.0] * len(idx),
                },
                index=[datetime.fromtimestamp(m * 60, tz=timezone.utc) for m in idx],
            )
            return df
        return None

    def status(self) -> dict:
        age = None
        if self.last_tick_ts:
            age = round(time.time() - self.last_tick_ts, 1)
        return {
            "connected": self.connected,
            "endpoint": self.endpoint or "-",
            "demo": self.demo,
            "ticks": self.ticks,
            "tick_age_s": age,
            "assets_with_bars": sum(1 for b in self._bars.values() if len(b) >= 60),
        }


_singleton: POFeed | None = None


async def start(ssid_raw: str) -> POFeed:
    global _singleton
    if _singleton:
        return _singleton
    _singleton = POFeed(ssid_raw)
    _singleton._task = asyncio.create_task(_singleton.run())
    return _singleton


def stop():
    if _singleton:
        _singleton.stop()


def candles(pair: str):
    return _singleton.candles(pair) if _singleton else None


def status() -> dict | None:
    return _singleton.status() if _singleton else None
