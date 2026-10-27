# Pocket Option Signal Bot (Telegram)

A Telegram bot that mirrors the reference "signal assistant" flow and analyses **M1 (1-minute)**
market data with the exact indicator setup from the Pocket Option chart screenshots:

| Indicator | Inputs used |
|---|---|
| Bollinger Bands | Period 20, Deviation **2** *(second band: Deviation **3**)* |
| Stochastic Oscillator | %K = **6**, %D = **3**, Smoothing = **3**, MA = **SMA** |
| CCI | Period **25** |
| ADX | ADX Smoothing **1**, DI Length **5** |
| Chart type | **Heikin Ashi** candles |

Plus: candlestick-pattern recognition (engulfing, hammer / shooting star, marubozu,
morning / evening star, doji, tweezers), volatility awareness (ATR regime, BB squeeze)
and short-term trend detection (EMA 9/21, Heikin Ashi streak, DI direction).

## Telegram flow (same as reference)
`/start` → **Welcome** → **Select Quote (Stock / OTC)** → **Select Asset** (3 pages of pairs)
→ **Select Time** (1 min … 4 hours) → **"Get a signal"** → **Analyzing…** progress bar
→ 🟢 **Higher** / 🔴 **Lower** with **Accuracy %**, confluence breakdown, then New signal / Exit to Menu.

## Setup

```bash
pip install -r requirements.txt
copy .env.example .env     # then paste your BotFather token into .env
python main.py
```

1. Open Telegram → **@BotFather** → `/newbot` → copy the token.
2. Paste it into `.env` as `BOT_TOKEN=...`.
3. Optional: restrict access with `ALLOWED_IDS=123456789,987654321` (your Telegram ID from @userinfobot).

## How the signal is generated (`engine/signals.py`)

Each component votes Up/Down with a weight; final direction = sum of votes, and the
**Accuracy % is an internal confidence score** derived from:

- % of confluence across the indicator set (agreement ratio),
- dominance of the winning side,
- volatility regime (low ATR or ADX < 15 reduces confidence; BB squeeze / spike add warnings).

🔴 **Accuracy here is a heuristic agreement score — it is NOT a measured win-rate and does
not guarantee outcomes.** Binary options are high-risk; this tool is for analysis/education.

## Data feeds (`engine/datafeed.py`, `engine/po_feed.py`)

- **OTC pairs** — live Pocket Option prices through the `ssid` session
  (`engine/po_feed.py`, BinaryOptionsToolsV2), using the broker's own M1 candle
  series — the same series the Pocket Option chart draws. Bars are fetched with a
  short-lived client (`get_candles_live()`), which returns the closed history plus
  the **current forming candle**, so the newest bar is usually 5–60 seconds old.
- **Stock pairs** — real 1-minute OHLC from Yahoo Finance spot symbols (`EURUSD=X`, …).
  If Yahoo is empty or stale, the broker's spot feed is tried.
- **OTC is never faked.** If the broker feed is down, or the pair is currently
  rotated out of Pocket Option's tradeable list (e.g. EUR/CAD OTC some hours), the
  bot refuses the signal and explains why. A signal on invented or non-tradeable
  prices can lose real money, so there is no synthetic OTC fallback.
- **Freshness is enforced**: a warning appears when the newest bar exceeds 2.5
  minutes, and a feed older than 15 minutes is refused instead of analysed.
- **Stock market closed** — the bot says so rather than inventing a signal. The only
  synthetic series left is the clearly-labelled Stock fallback used when no live
  source is reachable at all (total network failure).
- Payout percentages come from the broker and are shown for the requested asset only.
  OTC assets must be both active and have a payout, otherwise they are refused.

Set the session up once:

```bash
python -m engine.po_feed --save '42["auth",{...}]'   # or put PO_SSID in .env
python -m engine.po_feed --show                      # assets + top payouts
python -m engine.po_feed --candles EUR/USD           # freshness check
```

## Tests

```bash
python smoke_test.py          # offline engine smoke test
python test_data_quality.py   # regression tests for every feed/signal bug found in audit
python test_po_feed.py        # offline protocol/candle-builder tests
python test_live_check.py     # live: all 24 pairs, freshness, refusals, latency
```

`test_live_check.py` needs a valid `pocket_ssid.json` (or `PO_SSID`) and takes a few
minutes because it requests every OTC pair through the real broker.

## Files

```
main.py                 Telegram bot + UI flow + signal formatting
engine/indicators.py    BB / Stoch / CCI / ADX / ATR / Heikin Ashi / EMA
engine/patterns.py      Candlestick pattern recognition
engine/signals.py       Confluence engine -> Higher/Lower + confidence
engine/po_feed.py       Pocket Option live feed (ssid) + payouts + tradeability
engine/datafeed.py      source routing, freshness checks, refusals
smoke_test.py           Offline test of the engine (no token needed)
```

## Extending

- To rate-limit or gate access, edit `ALLOWED_IDS` / `deliver_signal`.
- To add pairs, extend `ASSET_PAGES` in `main.py` and `SYMBOL_MAP` in `engine/datafeed.py`
  (plus `_ASSETS` in `engine/po_feed.py` for the OTC symbol mapping).
- The engine is feed-agnostic: any source that returns an M1 OHLC DataFrame
  (`open/high/low/close/volume`, UTC index) plugs into `engine/datafeed.py`.
