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

## Data feeds (`engine/datafeed.py`)

- **Stock pairs** — real 1-minute OHLC from Yahoo Finance spot symbols (`EURUSD=X`, …).
- **OTC pairs** — Pocket Option's OTC feed is private; OTC series is a **deterministic
  synthetic feed** (seeded per pair per 15-minute bucket, anchored to spot) so every user
  sees the same candles; signals are reproducible but *not* Pocket Options's live OTC prices.
- If Yahoo is unreachable, Stock falls back to the synthetic feed and the bot notes this.

## Files

```
main.py                 Telegram bot + UI flow + signal formatting
engine/indicators.py    BB / Stoch / CCI / ADX / ATR / Heikin Ashi / EMA
engine/patterns.py      Candlestick pattern recognition
engine/signals.py       Confluence engine -> Higher/Lower + confidence
engine/datafeed.py      Yahoo spot feed + deterministic OTC generator
smoke_test.py           Offline test of the engine (no token needed)
```

## Extending

- To rate-limit or gate access, edit `ALLOWED_IDS` / `deliver_signal`.
- To add pairs, extend `ASSET_PAGES` in `main.py` and `SYMBOL_MAP` in `engine/datafeed.py`.
- To wire a real Pocket Option account feed later, replace `get_candles` with your own
  websocket/iframe source; the engine is feed-agnostic.
