"""
Spike-style Pocket Option SIGNAL bot - Market -> Pair -> Time -> Signal.

Flow:  /start  ->  Start  ->  Select a market (Stock/OTC)
       ->  Select a currency pair  ->  Select trading time
       ->  Signal Generation...  ->  BUY/SELL or NO TRADE card

One signal engine: a confluence reversal rule - price tags Bollinger 20/2
(with Bollinger 20/3 as the "extreme" guard), Stochastic 6/3/3 confirms the
oversold/overbought hook, CCI 25 must be beyond +/-100 and turning, and
ADX (smoothing 1, DI 5) must be below 25 (never fade a runaway trend). The setup is replayed
causally over the pair's own history and its out-of-sample hit-rate is
compared to the payout break-even. A BUY/SELL card is shown ONLY when that edge
clears break-even; otherwise the bot says NO TRADE.

Display-only by default. The card's "Accept signal" button can place a real
order only after `/trading on` (persisted in pocket_settings.json) and with an
SSID saved (see pocket_payout.py); otherwise it prints a disabled notice. The
ALLOW_TRADING=1 env var is still honoured for one-off launches.

Run: pip install pandas numpy requests
     set TELEGRAM_TOKEN (BotFather token) and TELEGRAM_CHAT_ID env vars
     python pocket_signal_bot.py
     (/trading on to enable the Accept-order button, or export ALLOW_TRADING=1)
"""

import time
import json
import os
import math
import html
import threading
import requests
import datetime
import pandas as pd
import numpy as np
from collections import deque

# ===== CONFIG =====
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "PASTE_CHAT_ID_NUMBER")

PERIOD = 20
STD_MULT = 2.0        # first Bollinger set: period 20, deviation 2
STD_MULT_OUTER = 3.0  # second Bollinger set: period 20, deviation 3
COOLDOWN_SEC = 60
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "pocket_settings.json")

RSI_PERIOD = 14
# Target is exactly ONE bar of the selected timeframe, matching a binary
# option bought now that expires `sel_time` seconds later.
HORIZON = 1
SQUEEZE_PCT = 40

# ---- chart-setup indicators (replace the old EMA/MACD/RSI vote) ----
STOCH_K = 6           # Stochastic %K period (picture: 6)
STOCH_D = 3           # Stochastic %D period   (picture: 3)
STOCH_SMOOTH = 3      # %K smoothing period, SMA (picture: 3)
STOCH_OS = 20.0       # oversold / overbought bands
STOCH_OB = 80.0
CCI_PERIOD = 25
CCI_OS = -100.0
CCI_OB = 100.0
ADX_SMOOTHING = 1     # TradingView "ADX Smoothing" (picture: 1 -> raw DX)
ADX_DI_LEN = 5        # TradingView "DI Length"      (picture: 5)
ADX_TREND = 25.0      # no reversal setups while the move is this strong
WARMUP_BARS = max(PERIOD + 10, CCI_PERIOD + 10,
                  ADX_DI_LEN * 4 + ADX_SMOOTHING + 10,
                  STOCH_K + STOCH_SMOOTH + STOCH_D + 10)

# ---- daily risk limits (overridable in pocket_settings.json) ----
DAILY_MAX_LOSS = 10.0     # units; Accept refuses once today's realized P/L <= -this
DAILY_MAX_TRADES = 30     # hard cap of accepted orders per calendar day
MAX_STAKE_BALANCE_PCT = 5.0  # refuse a stake larger than this % of the balance

# ---- honest edge gate (tunable via pocket_settings.json) ----
EDGE_MARGIN = 2.0      # measured hit-rate must beat break-even by this many %
MIN_BACKTEST_N = 100   # minimum resolved setups before the edge is trusted
MIN_PAYOUT = 60.0      # assets paying less than this are never signalled
WILSON_Z = float(os.environ.get("WILSON_Z", "1.96"))  # 1.96=95%, 1.28=80%
GATE_MODE = os.environ.get("GATE_MODE", "strict").strip().lower()
if GATE_MODE not in ("strict", "point"):
    GATE_MODE = "strict"
EDGE_REPORT_MAX_AGE = 7 * 86400   # ignore validated reports older than 7 days
TRADE_AMOUNT = 1.0     # stake for an accepted signal (only used with ALLOW_TRADING=1)
SCAN_HISTORY = int(os.environ.get("SCAN_HISTORY", "900"))  # deep-check bars
SCAN_DEEP_N = int(os.environ.get("SCAN_DEEP_N", "4"))      # candidates deep-checked
SCAN_HISTORY = max(300, min(SCAN_HISTORY, 5000))

# Signal + message-id -> context, so "Accept signal" can place the order the card
# showed. Only the owner's chats reach here, and orders still need ALLOW_TRADING=1.
PENDING_SIGNALS = {}


def remember_signal(mid, res, pair, period, price):
    if not mid:
        return
    PENDING_SIGNALS[mid] = {"pair": pair, "period": int(period),
                            "signal": res.get("signal"), "price": price,
                            "ts": time.time()}
    if len(PENDING_SIGNALS) > 200:
        oldest = sorted(PENDING_SIGNALS, key=lambda k: PENDING_SIGNALS[k]["ts"])
        for k in oldest[:100]:
            PENDING_SIGNALS.pop(k, None)

# Single engine: the chart-setup confluence rule, gated by its measured
# out-of-sample hit-rate vs. the payout break-even. The previous six "models"
# were cosmetic weight variants with no measured edge, so they were removed.
ENGINE_WEIGHTS = (1,)  # unused by the confluence rule; kept for signature compat
MODEL_PARAMS = {
    "engine": {"w": ENGINE_WEIGHTS,
               "label": "BB 20/2 + 20/3 · Stoch 6/3/3 · CCI 25 · ADX (1,5)"},
}
MODELS = ["engine"]
MODEL_NAMES = {"engine": "Engine"}

TIMES = [(5, "5 seconds"), (15, "15 seconds"), (30, "30 seconds"), (60, "1 minutes"),
         (120, "2 minutes"), (180, "3 minutes"), (300, "5 minutes"), (600, "10 minutes")]

try:
    import pocket_payout
except Exception:
    pocket_payout = None

DATA_SRC = "demo (random)"
CURRENT_MODEL = "engine"

DEFAULT_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
                 "EURGBP", "EURJPY", "GBPJPY", "EURAUD", "EURCHF", "EURCAD", "GBPCHF",
                 "GBPCAD", "GBPAUD", "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "NZDCAD",
                 "NZDCHF", "NZDJPY", "CADCHF", "CADJPY", "CHFJPY", "EURNZD", "USDSGD",
                 "USDHKD", "USDCNH"]

# ===== concurrent-safe state files =====
# Several threads (settler, account sync, Telegram loop) load-modify-save the
# JSON state files. All cycles are serialized here and every save is atomic
# (write temp file, then rename) so a crash can no longer half-write a file.
_STATE_LOCK = threading.RLock()


def _atomic_write(path, text):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


# ===== settings persistence =====
def load_settings():
    d = {"last_pair": "EURUSD-OTC", "last_time": 60, "auto": False,
         "model": "engine", "market": "all"}
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE) as f:
                d.update(json.load(f))
        except Exception:
            pass
    if d.get("model") not in MODEL_PARAMS:
        d["model"] = "engine"
    if d.get("market") not in ("stock", "otc", "all"):
        d["market"] = "all"
    gm = str(d.get("gate_mode") or os.getenv("GATE_MODE", "strict")).lower()
    d["gate_mode"] = gm if gm in ("strict", "point") else "strict"
    if "allow_trading" not in d:
        d["allow_trading"] = os.getenv("ALLOW_TRADING") == "1"
    else:
        d["allow_trading"] = bool(d["allow_trading"])
    return d


def trading_on():
    return os.getenv("ALLOW_TRADING") == "1"

def save_settings(s):
    try:
        with _STATE_LOCK:
            cur = {}
            if os.path.exists(SETTINGS_FILE):
                try:
                    with open(SETTINGS_FILE) as f:
                        cur = json.load(f)
                except Exception:
                    cur = {}
            cur.update(s)
            _atomic_write(SETTINGS_FILE, json.dumps(cur))
    except Exception as e:
        print("Settings save failed:", e)

# ===== strategy =====
def _indicator_stack(closes, ohlc=None):
    """Precompute every series the decision needs.

    Indicators (chart setup): Bollinger (20, 2) + (20, 3), Stochastic 6/3/3
    (SMA-smoothed %K), CCI 25, ADX (smoothing 1, DI length 5).

    When real OHLC is supplied, Bollinger bands, the Stochastic ranges, CCI's
    typical price and +DI/-DI use actual high/low. Otherwise they fall back to
    a close-only approximation (high = low = close).
    """
    s = pd.Series([float(x) for x in closes])
    n = len(s)
    if ohlc and len(ohlc) >= n:
        rows = ohlc[-n:]
        high = pd.Series([float(r["h"]) for r in rows], index=s.index)
        low = pd.Series([float(r["l"]) for r in rows], index=s.index)
    else:
        high = s.copy()
        low = s.copy()

    ema9 = s.ewm(span=9, adjust=False).mean()
    ema21 = s.ewm(span=21, adjust=False).mean()
    macd = ema9 - ema21
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / RSI_PERIOD, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / RSI_PERIOD, adjust=False).mean()
    rsi = 100.0 - (100.0 / (1.0 + gain / loss.replace(0, 1e-12)))
    mom3 = s - s.shift(3)
    sma = s.rolling(PERIOD).mean()
    std = s.rolling(PERIOD).std(ddof=0)   # TradingView uses population stdev
    width = (2.0 * STD_MULT) * std
    wpc = (width / sma * 100).rolling(50).rank(pct=True) * 100

    # Bollinger 20/2 and 20/3 (the two sets in the chart setup)
    lower2 = sma - STD_MULT * std
    upper2 = sma + STD_MULT * std
    lower3 = sma - STD_MULT_OUTER * std
    upper3 = sma + STD_MULT_OUTER * std

    # True Range + Wilder ATR (feeds +DI/-DI/ADX)
    prev_close = s.shift(1)
    tr = pd.concat([(high - low),
                    (high - prev_close).abs(),
                    (prev_close - low).abs()], axis=1).max(axis=1)
    # TradingView ADX inputs: "DI Length" smooths TR and +DM/-DM,
    # "ADX Smoothing" smooths DX (period 1 -> raw DX, no smoothing)
    alpha_di = 1.0 / ADX_DI_LEN
    atr = tr.ewm(alpha=alpha_di, adjust=False).mean()
    # Directional Movement (Wilder): +DI, -DI, ADX
    up = high.diff()
    dn = -low.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=s.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=s.index)
    natr = atr.replace(0, 1e-12)
    plus_di = 100.0 * plus_dm.ewm(alpha=alpha_di, adjust=False).mean() / natr
    minus_di = 100.0 * minus_dm.ewm(alpha=alpha_di, adjust=False).mean() / natr
    dx = ((plus_di - minus_di).abs()
          / (plus_di + minus_di).replace(0, 1e-12)) * 100.0
    alpha_adx = 1.0 / max(1, ADX_SMOOTHING)
    adx = dx.ewm(alpha=alpha_adx, adjust=False).mean()

    # Stochastic (%K period 6, %D period 3, %K smoothing 3 via SMA)
    low_k = low.rolling(STOCH_K).min()
    high_k = high.rolling(STOCH_K).max()
    rng_k = (high_k - low_k).replace(0, 1e-12)
    k_raw = 100.0 * (s - low_k) / rng_k
    stoch_k = k_raw.rolling(STOCH_SMOOTH).mean()   # smoothed %K as displayed
    stoch_d = stoch_k.rolling(STOCH_D).mean()

    # CCI 25 (typical price uses real high/low when available)
    tp = (high + low + s) / 3.0
    tp_ma = tp.rolling(CCI_PERIOD).mean()
    mad = tp.rolling(CCI_PERIOD).apply(
        lambda x: float(np.mean(np.abs(x - x.mean()))), raw=True)
    cci = (tp - tp_ma) / (0.015 * mad.replace(0, 1e-12))

    return {"s": s, "ema9": ema9, "ema21": ema21, "hist": hist, "rsi": rsi,
            "mom3": mom3, "wpc": wpc, "sma": sma, "std": std,
            "high": high, "low": low,
            "lower2": lower2, "upper2": upper2,
            "lower3": lower3, "upper3": upper3,
            "stoch_k": stoch_k, "stoch_d": stoch_d, "cci": cci,
            "adx": adx, "plus_di": plus_di, "minus_di": minus_di, "atr": atr}


def _votes(stk, i, w):
    """Confluence reversal rule (the chart setup) - or None when no setup.

    Measured as variant "D" (best sample/fidelity balance, see notes):
      - the price TAGGED the lower Bollinger 20/2 band within the last 3 bars
        (wick or close), and has NOT closed beyond the 20/3 band now;
      - Stochastic %K(6/3) hooked: close zone K <= 30, K >= previous K
        (up-hook), and %K touched <= 20 within the last 3 bars (extreme);
      - CCI(25) <= -100 on this or the previous bar (extreme), with the
        up-tick owned by the stochastic hook;
      - ADX (smoothing 1, DI 5) < 25 - ranges only, never fade a runaway.
    BUY fires at the hook bar, settling one bar later (HORIZON=1).
    SELL (PUT): exact mirror at the upper bands.
    """
    _ = w  # confluence rule has no tunable vote weights
    def val(key, off=0):
        v = stk[key].iloc[i - off]
        return None if pd.isna(v) else float(v)

    c = val("s")
    low2, up2 = val("lower2"), val("upper2")
    low3, up3 = val("lower3"), val("upper3")
    k_now, k_prev = val("stoch_k"), val("stoch_k", 1)
    cci_now, cci_prev = val("cci"), val("cci", 1)
    adx = val("adx")
    low_w3 = [val("low", off) for off in (0, 1, 2)]
    high_w3 = [val("high", off) for off in (0, 1, 2)]
    if any(v is None for v in (c, low2, up2, low3, up3, k_now, k_prev,
                               cci_now, cci_prev, adx)):
        return None
    if any(v is None for v in low_w3 + high_w3):
        return None
    k_min3 = min(val("stoch_k", off) for off in (0, 1, 2))
    k_max3 = max(val("stoch_k", off) for off in (0, 1, 2))
    if None in (k_min3, k_max3):
        return None

    # --- CALL: lower-band bounce with oversold confirmation ---
    touch_low = min(low_w3) <= low2                 # tagged band recently
    not_knife = c >= low3                           # no close beyond 20/3 now
    stoch_call = (k_now <= 30.0 and k_now >= k_prev  # hooking up from below
                  and k_min3 <= STOCH_OS)            # extreme seen recently
    cci_call = cci_now <= CCI_OS or cci_prev <= CCI_OS
    if touch_low and not_knife and stoch_call and cci_call and adx < ADX_TREND:
        return "CALL"

    # --- PUT: upper-band fade with overbought confirmation ---
    touch_up = max(high_w3) >= up2                  # tagged band recently
    not_rocket = c <= up3                           # no close beyond 20/3 now
    stoch_put = (k_now >= 70.0 and k_now <= k_prev  # hooking down from above
                 and k_max3 >= STOCH_OB)            # extreme seen recently
    cci_put = cci_now >= CCI_OB or cci_prev >= CCI_OB
    if touch_up and not_rocket and stoch_put and cci_put and adx < ADX_TREND:
        return "PUT"
    return None


def regime_at(stk, i):
    v = stk["wpc"].iloc[i]
    if pd.isna(v):
        return "normal"
    v = float(v)
    if v <= SQUEEZE_PCT:
        return "squeeze"
    if v >= 75:
        return "wide"
    return "normal"


def decide_at(stk, i, params):
    if i < WARMUP_BARS:
        return None
    for key in ("stoch_k", "stoch_d", "cci", "adx", "lower2", "lower3"):
        if pd.isna(stk[key].iloc[i]):
            return None
    return _votes(stk, i, params["w"])


def pair_payout(pair):
    """Current payout % for an asset, or None. Never raises."""
    if pocket_payout is None:
        return None
    try:
        return pocket_payout.get_payout(pair)
    except Exception:
        return None


def payout_break_even(payout_pct):
    """Win-rate (%) required to break even at a given payout %."""
    try:
        p = float(payout_pct)
    except Exception:
        return None
    if p <= 0:
        return None
    return 100.0 / (1.0 + p / 100.0)


def _wilson_lower(wins, n, z=1.96):
    """Lower bound of the 95% Wilson interval for a win rate, in %."""
    if not n:
        return 0.0
    p = wins / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return max(0.0, (centre - margin) / denom) * 100.0


# ===== validated edge report (from backtest.py --save) =====
EDGE_REPORT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "edge_report.json")
_EDGE_REPORT = {"mtime": None, "generated": None, "data": {}}


def load_edge_report():
    """Validated per-pair stats written by backtest.py --save (cached by mtime)."""
    try:
        m = os.path.getmtime(EDGE_REPORT_FILE)
    except OSError:
        _EDGE_REPORT["data"] = {}
        _EDGE_REPORT["mtime"] = None
        return {}
    if _EDGE_REPORT["data"] and _EDGE_REPORT.get("mtime") == m:
        return _EDGE_REPORT["data"]
    gen = None
    try:
        with open(EDGE_REPORT_FILE) as f:
            d = json.load(f)
        data = d.get("pairs", d) if isinstance(d, dict) else {}
        if isinstance(d, dict):
            gen = d.get("generated")
    except Exception:
        data = {}
    try:
        gen_ts = time.mktime(time.strptime(str(gen), "%Y-%m-%d %H:%M:%S"))
    except Exception:
        gen_ts = m
    _EDGE_REPORT["mtime"] = m
    _EDGE_REPORT["generated"] = gen_ts
    _EDGE_REPORT["data"] = data or {}
    return _EDGE_REPORT["data"]


def edge_from_report(pair, period):
    """Validated stats for this pair+timeframe, or None. Stale reports are
    ignored so an old backtest can never permanently veto live signals."""
    data = load_edge_report()
    if not data:
        return None
    rep = data.get(f"{pair}#{int(period)}") or data.get(pair)
    if not rep:
        return None
    gen = _EDGE_REPORT.get("generated")
    if gen and time.time() - float(gen) > EDGE_REPORT_MAX_AGE:
        return None
    rep = dict(rep)
    if gen:
        rep["_age_days"] = round((time.time() - float(gen)) / 86400.0, 1)
    return rep


def _fold_edge(stk, params, folds=4):
    """Replay the rule causally and score it per fold.

    Returns the *most recent* fold's stats as the gate input when that fold has
    enough samples (so a decayed edge is caught), otherwise the whole sample.
    `accs` lists every fold's hit-rate for a stability check."""
    s = stk["s"]
    start = WARMUP_BARS + 1
    wins = []
    for i in range(start, len(s) - HORIZON):
        sig = decide_at(stk, i, params)
        if not sig:
            continue
        c = float(s.iloc[i])
        nxt = float(s.iloc[i + HORIZON])
        wins.append((nxt > c) if sig == "CALL" else (nxt < c))
    total = len(wins)
    all_wins = sum(wins)
    all_acc = (all_wins / total * 100.0) if total else 0.0
    empty = {"wins": 0, "total": 0, "acc": 0.0, "accs": [],
             "all_wins": 0, "all_total": 0, "all_acc": 0.0, "fold_stats": []}
    if not total:
        return empty
    k = total // folds if folds >= 2 else 0
    accs = []
    fold_stats = []
    if k >= 10:
        for f in range(folds):
            lo = f * k
            hi = (f + 1) * k if f < folds - 1 else total
            seg = wins[lo:hi]
            w = sum(seg)
            t = len(seg)
            acc = (w / t * 100.0) if t else 0.0
            accs.append(round(acc, 1))
            fold_stats.append({"wins": w, "total": t, "acc": acc})
        last = wins[-k:]
        if len(last) >= 30:
            lw = sum(last)
            lt = len(last)
            return {"wins": lw, "total": lt,
                    "acc": lw / lt * 100.0, "accs": accs,
                    "all_wins": all_wins, "all_total": total,
                    "all_acc": all_acc, "fold_stats": fold_stats}
    return {"wins": all_wins, "total": total, "acc": all_acc, "accs": accs,
            "all_wins": all_wins, "all_total": total, "all_acc": all_acc,
            "fold_stats": fold_stats}


def analyze(closes, model=None, news_symbol=None, pair=None, period=None,
            ohlc=None, z_mult=1.0):
    """Evaluate the current bar for a tradeable setup.

    `z_mult` widens the Wilson lower-bound to account for the fact that the
    pair was picked as the best of ~N candidates (multiple-comparison /
    winner's-curse correction when scanning many pairs)."""
    model = model or CURRENT_MODEL
    params = MODEL_PARAMS.get(model, MODEL_PARAMS["engine"])
    pair = pair or sel_pair_default()
    period = int(period or 60)
    out = {"signal": "WAIT", "note": "", "rsi": None,
           "stoch": None, "cci": None, "adx": None,
           "up": None, "lo": None, "wide": False, "depth": 0.0, "model": model,
           "setup": False, "regime": "normal", "horizon": HORIZON,
           "full_acc": 0.0, "full_wins": 0, "full_total": 0, "fold_accs": [],
           "fold_stats": [],
           "confidence": 0.0, "tradeable": False, "payout": None,
           "break_even": None, "wilson": 0.0, "gate_note": "",
           "gate_reason": "", "validated": False, "validated_stats": None,
           "report_age": None, "z_mult": z_mult}
    need = WARMUP_BARS + 1
    if not closes or len(closes) < need:
        out["note"] = "Not enough candle history for this timeframe. Pick a shorter time."
        return out
    ohlc_data = ohlc if ohlc is not None else (fetch_ohlc(pair, period)
                                               if pocket_payout else None)
    stk = _indicator_stack(closes, ohlc_data)
    s = stk["s"]

    fold = _fold_edge(stk, params)
    all_wins = fold.get("all_wins", fold["wins"])
    all_total = fold.get("all_total", fold["total"])
    all_acc = fold.get("all_acc", fold["acc"])
    f_wins, f_total = all_wins, all_total
    out["full_wins"], out["full_total"] = all_wins, all_total
    out["full_acc"] = all_acc
    out["fold_accs"] = fold["accs"]
    out["fold_stats"] = fold.get("fold_stats", [])

    # ---- honest edge gate: does the measured hit-rate beat break-even? ----
    payout = pair_payout(pair)
    out["payout"] = payout
    be = payout_break_even(payout)

    rep = edge_from_report(pair, period)
    if be is None and rep is not None:
        # no live payout - fall back to the payout the backtest validated with
        be = float(rep.get("break_even") or 0.0) or None
        if out["payout"] is None and rep.get("payout"):
            out["payout"] = rep.get("payout")
    out["break_even"] = be
    out["wilson"] = _wilson_lower(f_wins, f_total, WILSON_Z * z_mult)

    report_ok = False
    report_note = ""
    r_bets = r_exp = r_wr = r_wil = 0.0
    if be is not None and (out["payout"] or 0) >= MIN_PAYOUT and rep is not None:
        out["validated_stats"] = rep
        out["report_age"] = rep.get("_age_days")
        r_bets = int(rep.get("bets", 0))
        r_exp = float(rep.get("expectancy", 0.0))
        r_wr = float(rep.get("win_rate", 0.0))
        r_wil = float(rep.get("wilson", 0.0))
        folds = rep.get("folds") or []
        last = folds[-1] if folds else None
        lw = float(last.get("wilson", 0.0)) if last else 0.0
        lexp = float(last.get("expectancy", 0.0)) if last else 0.0
        lbets = int(last.get("bets", 0)) if last else 0
        age = (f" ({out['report_age']:.0f}d old)" if out["report_age"] else "")
        if r_bets < MIN_BACKTEST_N:
            report_note = (f"Validated{age} on only {r_bets} bets "
                           f"(need {MIN_BACKTEST_N}).")
        elif r_exp <= 0:
            report_note = (f"Validated{age} expectancy {r_exp:+.3f}/bet "
                           "is not positive.")
        elif r_wil < be:
            report_note = (f"Validated{age} win {r_wr:.0f}% (Wilson "
                           f"{r_wil:.0f}%) below break-even {be:.0f}%.")
        elif last and lbets >= 20 and (lexp <= 0 or lw < be):
            report_note = "Recent validated fold decayed."
        else:
            report_ok = True

    if be is None:
        out["gate_reason"] = "payout"
        out["gate_note"] = "Payout unavailable - cannot judge the edge."
    elif (out["payout"] or 0) < MIN_PAYOUT:
        out["gate_reason"] = "payout"
        out["gate_note"] = (f"Payout {out['payout']:.0f}% is below the "
                            f"{MIN_PAYOUT:.0f}% minimum.")
    elif report_ok:
        out["validated"] = True
        out["tradeable"] = True
        out["gate_reason"] = "ok"
        out["full_acc"] = r_wr
        out["full_wins"] = int(rep.get("wins", 0))
        out["full_total"] = r_bets
        out["wilson"] = r_wil
        out["gate_note"] = (f"Validated {r_wr:.0f}% win, {r_exp:+.3f}/bet, "
                            f"SQN {rep.get('sqn', 0)} over {r_bets} bets.")
    else:
        # the live replay is the decider; a negative or stale report can no
        # longer veto a pair forever - it only adds a note when replay passes
        if f_total < MIN_BACKTEST_N:
            out["gate_reason"] = "sample"
            out["gate_note"] = (f"Only {f_total} resolved setups - need "
                                f"{MIN_BACKTEST_N} to trust the edge.")
        elif out["full_acc"] < be + EDGE_MARGIN:
            out["gate_reason"] = "edge"
            out["gate_note"] = (f"Edge {out['full_acc']:.0f}% is not above "
                                f"break-even {be:.0f}% (need +{EDGE_MARGIN:.0f}).")
        elif GATE_MODE != "point" and out["wilson"] < be:
            out["gate_reason"] = "wilson"
            out["gate_note"] = (f"Edge {out['full_acc']:.0f}% looks fine but its "
                                f"statistical floor ({out['wilson']:.0f}%) is "
                                f"below break-even {be:.0f}% - too thin.")
        elif out["fold_accs"] and out["fold_accs"][-1] < be:
            out["gate_reason"] = "fold"
            out["gate_note"] = (f"Most recent fold {out['fold_accs'][-1]:.0f}% "
                                f"is below break-even {be:.0f}% - edge not stable.")
        else:
            out["tradeable"] = True
            out["gate_reason"] = "ok"
            if GATE_MODE == "point":
                out["gate_note"] = (f"Edge {out['full_acc']:.0f}% clears "
                                    f"break-even {be:.0f}% +{EDGE_MARGIN:.0f} "
                                    f"over {f_total} setups [point mode].")
            else:
                out["gate_note"] = (f"Edge {out['full_acc']:.0f}% (floor "
                                    f"{out['wilson']:.0f}%) clears break-even "
                                    f"{be:.0f}% over {f_total} setups.")
            if report_note:
                out["gate_note"] += f" ({report_note} Live replay passes now.)"

    i = len(s) - 1
    out["rsi"] = float(stk["rsi"].iloc[i]) if not pd.isna(stk["rsi"].iloc[i]) else None
    out["stoch"] = float(stk["stoch_k"].iloc[i]) if not pd.isna(stk["stoch_k"].iloc[i]) else None
    out["cci"] = float(stk["cci"].iloc[i]) if not pd.isna(stk["cci"].iloc[i]) else None
    out["adx"] = float(stk["adx"].iloc[i]) if not pd.isna(stk["adx"].iloc[i]) else None
    sma, std = stk["sma"], stk["std"]
    up = sma + STD_MULT * std
    lo = sma - STD_MULT * std
    out["up"] = float(up.iloc[i]) if not pd.isna(up.iloc[i]) else None
    out["lo"] = float(lo.iloc[i]) if not pd.isna(lo.iloc[i]) else None
    out["regime"] = regime_at(stk, i)
    out["wide"] = out["regime"] in ("wide",)
    out["note"] = ("Quiet market - the regime with the best measured edge."
                   if out["regime"] == "squeeze"
                   else "Normal volatility - accuracy near a coin flip.")

    out["confidence"] = out["full_acc"]
    vote = decide_at(stk, i, params)
    out["setup"] = bool(vote)
    out["signal"] = "NO TRADE"
    if out["tradeable"]:
        if vote:
            out["signal"] = vote
        else:
            out["gate_note"] = (
                (out["gate_note"] + " Edge is validated, but the BB 2/3 + "
                 "Stoch + CCI confluence has no setup on the current bar.")
                .strip())
    return out


# ===== REAL ML: models learn from candles + settled signal outcomes =====
ML_DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_data.json")
ML_MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_models.pkl")
ML_STALE_SEC = 900          # background refresh interval
ML_MAX_SAMPLES = 3000       # feedback samples kept per pair
ML_MIN_SAMPLES = 100        # minimum samples before a model trains

_ml_lock = threading.RLock()
_ml_models = {}        # "pair#model" -> fitted estimator
_ml_meta = {}          # "pair#model" -> {"acc", "n", "trained_at"}
_ml_pending_train = set()
_ml_learned_since = 0


def _ml_data_load():
    try:
        with open(ML_DATA_FILE) as f:
            d = json.load(f)
        if isinstance(d, dict) and isinstance(d.get("pairs"), dict):
            return d
    except Exception:
        pass
    return {"pairs": {}}


def _ml_data_save(d):
    try:
        with _STATE_LOCK:
            _atomic_write(ML_DATA_FILE, json.dumps(d))
    except Exception as e:
        print("ML data save failed:", e)


def _ml_save():
    try:
        import io
        import pickle
        buf = io.BytesIO()
        pickle.dump({"models": _ml_models, "meta": _ml_meta}, buf)
        with _STATE_LOCK:
            _atomic_write_binary(ML_MODEL_FILE, buf.getvalue())
    except Exception as e:
        print("ML model save failed:", e)


def _atomic_write_binary(path, data):
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _ml_load():
    try:
        import pickle
        if os.path.exists(ML_MODEL_FILE):
            with open(ML_MODEL_FILE, "rb") as f:
                d = pickle.load(f)
            _ml_models.update(d.get("models") or {})
            _ml_meta.update(d.get("meta") or {})
            print(f"ML: loaded {len(_ml_models)} trained models")
    except Exception as e:
        print("ML model load failed:", e)


def _feat_matrix(closes):
    """16 real features per bar: returns, RSI, MACD, bands, trend, flow."""
    try:
        vals = [float(x) for x in closes]
        if len(vals) < 70:
            return [], []
        s = pd.Series(vals)
        n = len(s)
        delta = s.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        rsi = (100 - 100 / (1 + gain / (loss + 1e-12)) - 50) / 50
        ema9 = s.ewm(span=9, adjust=False).mean()
        ema21 = s.ewm(span=21, adjust=False).mean()
        macd = ema9 - ema21
        hist = macd - macd.ewm(span=9, adjust=False).mean()
        sma20 = s.rolling(20).mean()
        std20 = s.rolling(20).std()
        std5 = s.rolling(5).std()
        sma50 = s.rolling(50).mean()
        hh = s.rolling(14).max()
        ll = s.rolling(14).min()
        pb = ((s - (sma20 - 2 * std20)) / (4 * std20 + 1e-12) - 0.5) * 2
        stoch = ((s - ll) / (hh - ll + 1e-12) - 0.5) * 2
        up10 = (delta > 0).rolling(10).mean() * 2 - 1
        up3 = (delta > 0).rolling(3).mean() * 2 - 1
        ret1 = s.pct_change() * 1000
        ret3 = s.pct_change(3) * 1000
        ret5 = s.pct_change(5) * 1000
        ret10 = s.pct_change(10) * 1000
        X, idx = [], []
        for i in range(60, n):
            if pd.isna(pb.iloc[i]) or pd.isna(sma50.iloc[i]) or pd.isna(std5.iloc[i]):
                continue
            c = vals[i]
            X.append([
                float(ret1.iloc[i]), float(ret3.iloc[i]),
                float(ret5.iloc[i]), float(ret10.iloc[i]),
                float(rsi.iloc[i]),
                float(hist.iloc[i] / c * 1000),
                float(macd.iloc[i] / c * 1000),
                float(pb.iloc[i]),
                float(std20.iloc[i] / c * 1000),
                float((sma20.iloc[i] - sma20.iloc[i - 5]) / c * 1000),
                float((c - sma20.iloc[i]) / c * 1000),
                float((c - sma50.iloc[i]) / c * 1000),
                float(stoch.iloc[i]),
                float(up10.iloc[i]),
                float(up3.iloc[i]),
                float(std5.iloc[i] / (std20.iloc[i] + 1e-12)),
            ])
            idx.append(i)
        return X, idx
    except Exception as e:
        print("feat_matrix failed:", e)
        return [], []


def _feat_last(closes):
    X, _ = _feat_matrix(closes)
    return X[-1] if X else None


def _ml_dataset(pair, closes):
    """Backtest labels from candles + real labels from settled outcomes."""
    X, idx = _feat_matrix(closes)
    n = len(closes)
    keep = [(x, i) for x, i in zip(X, idx) if i + HORIZON < n]
    xs = [k[0] for k in keep]
    ys = [1 if float(closes[k[1] + HORIZON]) > float(closes[k[1]]) else 0
          for k in keep]
    try:
        fb = _ml_data_load()["pairs"].get(pair, {})
        for vec, lab in zip(fb.get("X", []), fb.get("y", [])):
            if isinstance(vec, (list, tuple)) and len(vec) == 16:
                xs.append([float(v) for v in vec])
                ys.append(1 if int(lab) else 0)
    except Exception:
        pass
    return xs, ys


def _ml_estimator(key=None):
    """One regularised classifier, tuned for small noisy FX samples."""
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(n_estimators=120, max_depth=3,
                                      learning_rate=0.05, subsample=0.8,
                                      min_samples_leaf=20, random_state=7)


def walk_forward_acc(closes, key=None, folds=4):
    """Rolling-origin out-of-sample accuracy (%), the honest way to score a
    classifier: train on the past, test on the next unseen slice, roll forward.
    Returns (accuracy_pct, n_test) or (None, 0) when there is too little data."""
    try:
        X, y = _ml_dataset("__wf__", closes)
        n = len(X)
        step = n // (folds + 1)
        if n < 120 or step < 20:
            return None, 0
        from sklearn.metrics import accuracy_score
        accs, tested = [], 0
        for f in range(1, folds + 1):
            tr_end = step * f
            te_end = min(n, step * (f + 1))
            if te_end - tr_end < 5:
                continue
            Xtr, ytr = X[:tr_end], y[:tr_end]
            Xte, yte = X[tr_end:te_end], y[tr_end:te_end]
            if len(set(ytr)) < 2:
                continue
            clf = _ml_estimator(key)
            clf.fit(Xtr, ytr)
            accs.append(accuracy_score(yte, clf.predict(Xte)))
            tested += len(yte)
        if not accs:
            return None, 0
        return sum(accs) / len(accs) * 100.0, tested
    except Exception as e:
        print("walk_forward_acc failed:", e)
        return None, 0


def ml_train(pair, closes, keys=None, force=False):
    """Fit/refresh models for a pair; honest temporal holdout accuracy."""
    global _ml_learned_since
    if not pair or not closes or len(closes) < 80:
        return False
    with _ml_lock:
        try:
            X, y = _ml_dataset(pair, closes)
            if len(X) < ML_MIN_SAMPLES or len(set(y)) < 2:
                return False
            cut = int(len(X) * 0.8)
            Xtr, ytr = X[:cut], y[:cut]
            Xte, yte = X[cut:], y[cut:]
            if len(set(ytr)) < 2:
                Xtr, ytr = X, y
                Xte, yte = X[-max(1, len(X) // 5):], y[-max(1, len(y) // 5):]
            trained = False
            for key in (keys or MODELS):
                ck = f"{pair}#{key}"
                meta = _ml_meta.get(ck) or {}
                if (not force and key in _ml_models
                        and time.time() - float(meta.get("trained_at", 0)) < ML_STALE_SEC):
                    continue
                try:
                    clf = _ml_estimator(key)
                    clf.fit(Xtr, ytr)
                    from sklearn.metrics import accuracy_score
                    acc = 0.0
                    oos_n = len(yte)
                    if Xte and len(set(yte)) >= 1:
                        acc = float(accuracy_score(yte, clf.predict(Xte))) * 100.0
                    wf_acc, wf_n = walk_forward_acc(closes, key)
                    if wf_acc is not None:
                        acc, oos_n = wf_acc, wf_n
                    _ml_models[ck] = clf
                    _ml_meta[ck] = {"acc": round(acc, 1), "n": len(X),
                                    "oos_n": oos_n,
                                    "trained_at": time.time()}
                    trained = True
                except Exception as e:
                    print(f"ML train {key} failed:", e)
            if trained:
                _ml_save()
                _ml_learned_since = 0
            return trained
        except Exception as e:
            print("ml_train failed:", e)
            return False


def ml_train_async(pair, tf=None, force=False):
    """Retrain in the background so signals never wait on fitting."""
    if not pair:
        return
    if not force:
        now = time.time()
        if any(k.startswith(pair + "#")
               and now - float(v.get("trained_at", 0)) < ML_STALE_SEC
               for k, v in _ml_meta.items()):
            return
    if pair in _ml_pending_train:
        return
    _ml_pending_train.add(pair)

    def worker(p=pair, per=tf, fr=force):
        try:
            period = per or int(globals().get("sel_time", 60))
            closes, _ = get_series(p, period, fetch=True)
            if closes:
                ml_train(p, closes, force=fr)
        except Exception as e:
            print("ML async train failed:", e)
        finally:
            _ml_pending_train.discard(p)

    threading.Thread(target=worker, daemon=True).start()


def _ml_learn(pair, features, direction, win):
    """Feedback: a settled signal outcome becomes training data."""
    global _ml_learned_since
    if not features or len(features) != 16:
        return
    if direction == "CALL":
        label = 1 if win else 0
    elif direction == "PUT":
        label = 0 if win else 1
    else:
        return
    with _ml_lock:
        try:
            d = _ml_data_load()
            pr = d["pairs"].setdefault(pair, {"X": [], "y": []})
            pr["X"].append([round(float(v), 5) for v in features])
            pr["y"].append(int(label))
            if len(pr["y"]) > ML_MAX_SAMPLES:
                pr["X"] = pr["X"][-ML_MAX_SAMPLES:]
                pr["y"] = pr["y"][-ML_MAX_SAMPLES:]
            _ml_data_save(d)
            _ml_learned_since += 1
            print(f"ML feedback {pair} {direction} -> "
                  f"{'up' if label else 'down'} "
                  f"(+{_ml_learned_since} since last train)")
        except Exception as e:
            print("ML learn failed:", e)
    if _ml_learned_since >= 20:
        ml_train_async(pair, force=True)


def model_perf(key, last=50):
    """Real measured win rate of a model from settled signals."""
    try:
        st = [x for x in track_load()["settled"]
              if x.get("model") == key][-last:]
        if not st:
            return None, 0
        w = sum(1 for x in st if x.get("win"))
        return w / len(st) * 100.0, len(st)
    except Exception:
        return None, 0


def gate_score(res, model=None):
    """Displayed confidence: the rule's measured (or validated) hit-rate (%) -
    the same number the edge gate uses. No synthetic uplift."""
    return max(0.0, min(100.0, float(res.get("full_acc", 0.0))))


# ===== live signal tracking =====
TRACK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "signal_track.json")
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "pending_results.json")


def results_load():
    with _STATE_LOCK:
        try:
            with open(RESULTS_FILE) as f:
                d = json.load(f)
            return d if isinstance(d, list) else []
        except Exception:
            return []


def results_save(items):
    with _STATE_LOCK:
        try:
            _atomic_write(RESULTS_FILE, json.dumps(items[-100:]))
        except Exception as e:
            print("Results save failed:", e)


def note_placed(chat, ticket, pair, direction, period, entry=None):
    """Remember an accepted order so its settlement can be announced."""
    with _STATE_LOCK:
        cutoff = time.time() - 86400
        items = [p for p in results_load()
                 if float(p.get("placed_ts") or 0) >= cutoff]
        items.append({
            "chat": str(chat), "ticket": str(ticket or ""),
            "pair": pair, "dir": direction, "period": int(period),
            "entry": entry, "placed_ts": time.time(),
        })
        results_save(items)

def track_load():
    with _STATE_LOCK:
        try:
            with open(TRACK_FILE) as f:
                d = json.load(f)
            if isinstance(d, dict):
                d.setdefault("pending", [])
                d.setdefault("settled", [])
                return d
        except Exception:
            pass
        return {"pending": [], "settled": []}

def track_save(d):
    with _STATE_LOCK:
        try:
            _atomic_write(TRACK_FILE, json.dumps(d))
        except Exception as e:
            print("Track save failed:", e)

def track_add(pair, tf, direction, entry, model, regime, features=None):
    tfi = max(1, int(tf))
    try:
        if pocket_payout is not None:
            cl = pocket_payout.live_closes(pair, tfi)
            if cl and len(cl) >= 70:
                if entry is None:
                    entry = float(cl[-1])
                features = features or _feat_last(cl)
    except Exception:
        pass
    try:
        entry = float(entry)
    except Exception:
        return
    with _STATE_LOCK:
        d = track_load()
        d["pending"].append({
            "id": int(time.time() * 1000),
            "pair": pair, "tf": tfi, "dir": direction,
            "entry": entry, "model": model, "regime": regime,
            "features": features,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "expire_at": time.time() + HORIZON * tfi,
        })
        track_save(d)

def _settle_price(pair, tfi, expire_ts):
    """Close of the first CLOSED candle at/after the expiry moment.
    None = candle not closed yet (caller retries later)."""
    tfi = max(1, int(tfi))
    try:
        t_close = -(-int(expire_ts) // tfi) * tfi   # first bar boundary >= expiry
        now = time.time()
        if now < t_close + 1:
            return None
        target = t_close - tfi                       # candle whose close we need
        raw = None
        if pocket_payout is not None:
            api = pocket_payout._ensure_client()
            if api is not None:
                sym = pocket_payout.feed_symbol(pocket_payout._norm(pair))
                raw = pocket_payout._fetch_candles(api, sym, tfi, 60,
                                                   block=False) or []
        if raw:
            for row in raw:
                try:
                    if int(row.get("time") or 0) == target:
                        return float(row["close"])
                except Exception:
                    continue
            closed = [r for r in raw
                      if int(r.get("time") or 0) + tfi <= now]
            if closed:
                try:
                    return float(closed[-1]["close"])
                except Exception:
                    pass
        if pocket_payout is not None:
            cl = pocket_payout.get_closes(pair, tfi, 60)
            if cl:
                return float(cl[-1])
        cl2, _ = get_series(pair, tfi, fetch=True)
        return float(cl2[-1]) if cl2 else None
    except Exception:
        return None

def track_settle():
    with _STATE_LOCK:
        d = track_load()
        if not d["pending"]:
            return
        now = time.time()
        still = []
        changed = False
        for p in d["pending"]:
            if now < p.get("expire_at", 0) or now < float(p.get("check_at", 0)):
                still.append(p)
                continue
            pair = p.get("pair", sel_pair_default())
            tfi = int(p.get("tf", 60))
            px = _settle_price(pair, tfi, float(p.get("expire_at", 0)))
            if px is None:
                p["check_at"] = now + 20
                still.append(p)
                continue
            entry = float(p["entry"])
            changed = True
            if px == entry:
                print(f"Settled {p['dir']} {pair} entry={entry} exit={px} -> REFUND (tie)")
                continue
            win = (px > entry) if p["dir"] == "CALL" else (px < entry)
            d["settled"].append({
                "pair": p["pair"], "tf": p["tf"], "dir": p["dir"],
                "entry": entry, "exit": px, "win": bool(win),
                "model": p.get("model"), "regime": p.get("regime"),
                "closed": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            try:
                _ml_learn(pair, p.get("features"), p.get("dir"), bool(win))
            except Exception as e:
                print("ML feedback skipped:", e)
            print(f"Settled {p['dir']} {pair} entry={entry} exit={px} "
                  f"-> {'WIN' if win else 'LOSS'}")
        d["settled"] = d["settled"][-200:]
        d["pending"] = still
        if changed:
            track_save(d)

def track_summary(last=30):
    st = track_load()["settled"][-last:]
    if not st:
        return 0, 0, None, 0
    w = sum(1 for x in st if x.get("win"))
    n = len(st)
    return w, n - w, w / n * 100.0, n

def track_text():
    w, l, pct, n = track_summary()
    if not n:
        return "building (0 signals settled yet)"
    return f"{w}W / {l}L \u00b7 {pct:.0f}% (last {n})"


# ===== daily risk stop (accepted orders honour it) =====
RISK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "risk_state.json")


def risk_state_load():
    try:
        with open(RISK_FILE) as f:
            d = json.load(f)
        if isinstance(d, dict):
            return d
    except Exception:
        pass
    return {"date": time.strftime("%Y-%m-%d"), "profit": 0.0, "n": 0}


def risk_state_save(d):
    with _STATE_LOCK:
        _atomic_write(RISK_FILE, json.dumps(d))


def risk_reset_if_stale(d):
    if d.get("date") != time.strftime("%Y-%m-%d"):
        return {"date": time.strftime("%Y-%m-%d"), "profit": 0.0, "n": 0}
    return d


def risk_note_outcome(profit):
    """Record one settled order's P/L against today's total."""
    with _STATE_LOCK:
        d = risk_reset_if_stale(risk_state_load())
        d["profit"] = round(float(d.get("profit") or 0.0) + float(profit or 0.0), 4)
        d["n"] = int(d.get("n") or 0) + 1
        risk_state_save(d)
        return d


def risk_daily_block():
    """True when today's realized loss or order count hits the daily cap."""
    d = risk_reset_if_stale(risk_state_load())
    if float(d.get("profit") or 0.0) <= -abs(DAILY_MAX_LOSS):
        return True
    if int(d.get("n") or 0) >= DAILY_MAX_TRADES:
        return True
    return False


# ===== real account sync (linked via pocket_ssid.json, read-only) =====
ACCT_SYNC_GAP = 45.0
ACCT_SYNC_MAX_DEALS = 12      # bound work per sync so the loop never stalls
ACCT_FEATURE_ASSETS = 4       # only build ML features for this many assets


def _acct_raw(asset, open_ts):
    """Candles covering a past timestamp with >=80 bars of pre-history."""
    try:
        if pocket_payout is None:
            return []
        api = pocket_payout._ensure_client()
        if api is None:
            return []
        sym = pocket_payout.feed_symbol(pocket_payout._norm(asset))
        for period in (60, 300, 900, 1800):
            try:
                raw = pocket_payout._fetch_candles(api, sym, period, 144,
                                                   block=False) or []
            except Exception:
                raw = []
            if not raw:
                continue
            try:
                oldest = int(raw[0].get("time") or 0)
            except Exception:
                return raw
            bars_before = (open_ts - oldest) / float(period)
            if bars_before >= 80 or period == 1800:
                return raw
        return []
    except Exception:
        return []


def _feat_from_candles(raw, open_ts):
    """16-feature snapshot at (or just after) a past timestamp, from real candles.

    Only candles that CLOSED strictly before `open_ts` are used - the candle
    whose timestamp equals open_ts closes after the bet was placed, so
    including it would leak the trade window into the training features."""
    try:
        closes = []
        for row in raw:
            try:
                ts = int(row.get("time") or 0)
            except Exception:
                ts = 0
            if ts and ts >= int(open_ts):
                continue          # candle covering open_ts closes in the future
            try:
                closes.append(float(row["close"]))
            except Exception:
                continue
        if len(closes) < 70:
            return None
        return _feat_last(closes)
    except Exception:
        return None


def account_perf(last=50):
    """Win rate from real linked-account trades."""
    try:
        st = [x for x in track_load()["settled"]
              if x.get("source") == "account"][-last:]
        if not st:
            return 0, 0, None, 0
        w = sum(1 for x in st if x.get("win"))
        return w, len(st) - w, w / len(st) * 100.0, len(st)
    except Exception:
        return 0, 0, None, 0


def _notify_result(t):
    """If a closed account deal matches an accepted (Accept-button) order,
    announce its outcome in the chat that placed it."""
    items = results_load()
    if not items or pocket_payout is None:
        return
    tid = str(t.get("id") or "")
    match = next((p for p in items
                  if p.get("ticket") and str(p["ticket"]) == tid), None)
    if match is None:
        asset = t.get("asset") or ""
        for p in items:
            try:
                if pocket_payout._norm(p.get("pair", "")) != asset:
                    continue
                period = int(p.get("period") or 60)
                gap = abs(float(t.get("open_ts") or 0)
                          - float(p.get("placed_ts") or 0))
            except Exception:
                continue
            if gap <= period * 2 + 60:
                match = p
                break
    if match is None:
        return
    try:
        period = int(match.get("period") or 60)
        placed = float(match.get("placed_ts") or 0)
    except Exception:
        period, placed = 60, 0.0
    close_ts = int(t.get("close_ts") or 0)
    # ignore a deal listed before the bet could have settled (still open)
    if placed and (time.time() < placed + period - 2
                   or (close_ts and close_ts < placed - 5)):
        return
    try:
        items.remove(match)
        results_save(items)
    except ValueError:
        pass
    refund = bool(t.get("refund"))
    win = bool(t.get("win"))
    emoji = "\u21a9\ufe0f" if refund else ("\U0001f7e2" if win else "\U0001f534")
    verdict = "REFUND (tie)" if refund else ("WIN" if win else "LOSS")
    profit = float(t.get("profit") or 0.0)
    bal = None
    try:
        bal = pocket_payout.account_balance()
    except Exception:
        pass
    extra = f"\nBalance: {bal:.2f}" if bal is not None else ""
    send(f"{emoji} <b>Trade settled \u2014 {verdict}</b>\n"
         f"{match.get('dir')} {html.escape(fmt_pair(match.get('pair')))} \u00b7 "
         f"{tf_label(match.get('period'))}\nProfit: {profit:+.2f}{extra}",
         chat_id=match.get("chat"), html_on=True)
    try:
        risk_note_outcome(profit)
    except Exception as e:
        print("risk note failed:", e)
    print(f"[result] {match.get('pair')} {verdict} profit={profit:+.2f}")


def sync_account_trades():
    """Pull real closed trades from Pocket Option and learn from them."""
    if pocket_payout is None:
        return
    try:
        trades = pocket_payout.account_trades()
    except Exception:
        return
    if not trades:
        return
    try:
        d = _ml_data_load()
    except Exception:
        return
    seen = set(str(x) for x in (d.get("seen_deals") or []))
    acct_settled = []
    changed = False
    new_ts = {}
    for t in trades:
        if len(new_seen) >= ACCT_SYNC_MAX_DEALS:
            break
        tid = str(t["id"])
        if tid in seen:
            continue
        new_seen.append(tid)
        if not t.get("refund") and t.get("entry"):
            a = t["asset"]
            new_ts[a] = min(new_ts.get(a, 10 ** 12), t["open_ts"])
    raw_cache = {}
    for a, ts0 in list(new_ts.items())[:ACCT_FEATURE_ASSETS]:
        raw_cache[a] = _acct_raw(a, ts0)
    for t in trades:
        tid = str(t["id"])
        if tid not in new_seen:
            continue
        changed = True
        try:
            _notify_result(t)
        except Exception as e:
            print("result notify failed:", e)
        if t.get("refund") or not t.get("entry"):
            continue
        went_up = t["exit"] > t["entry"]
        if t["win"]:
            direction = "CALL" if went_up else "PUT"
        else:
            direction = "PUT" if went_up else "CALL"
        feats = _feat_from_candles(raw_cache.get(t["asset"]) or [], t["open_ts"])
        _ml_learn(t["asset"], feats, direction, bool(t["win"]))
        acct_settled.append({
            "pair": t["asset"], "tf": t["tf"], "dir": direction,
            "entry": t["entry"], "exit": t["exit"], "win": bool(t["win"]),
            "model": None, "regime": "account", "source": "account",
            "closed": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        tag = "demo" if t.get("demo") else "real"
        print(f"Account {t['asset']} {direction} -> "
              f"{'WIN' if t['win'] else 'LOSS'} ({tag}, profit {t['profit']})")
    if changed:
        # re-read so feedback written by _ml_learn() is not overwritten
        with _ml_lock:
            d = _ml_data_load()
            seen = set(str(x) for x in (d.get("seen_deals") or []))
            seen.update(new_seen)
            d["seen_deals"] = sorted(seen)[-500:]
            _ml_data_save(d)
        with _STATE_LOCK:
            tr = track_load()
            tr["settled"].extend(acct_settled)
            tr["settled"] = tr["settled"][-200:]
            track_save(tr)

def sel_pair_default():
    return "EURUSD-OTC"

# ===== real-time market read =====
def market_report(pair, period_sec, fetch=True):
    if pocket_payout is None:
        return None
    ohlc = None
    try:
        # closed bars only - display may lag a second, decisions must not
        ohlc = (pocket_payout.live_ohlcv(pair, period_sec, include_forming=False)
                if fetch else None)
        if not ohlc:
            ohlc = (pocket_payout.get_ohlcv(pair, period_sec, 220) if fetch
                    else pocket_payout.peek_ohlcv(pair, period_sec))
    except Exception:
        return None
    if not ohlc or len(ohlc) < 60:
        return None

    closes = [r["c"] for r in ohlc]
    s = pd.Series(closes)
    i = len(s) - 1

    ema9 = s.ewm(span=9, adjust=False).mean()
    ema21 = s.ewm(span=21, adjust=False).mean()
    if ema9.iloc[i] > ema21.iloc[i] and ema9.iloc[i] >= ema9.iloc[i - 3]:
        trend, tnote = "UP", "EMA9 above EMA21 and rising"
    elif ema9.iloc[i] < ema21.iloc[i] and ema9.iloc[i] <= ema9.iloc[i - 3]:
        trend, tnote = "DOWN", "EMA9 below EMA21 and falling"
    else:
        trend, tnote = "SIDEWAYS", "EMAs flat or tangled"

    sma = s.rolling(PERIOD).mean()
    std = s.rolling(PERIOD).std()
    up = sma + STD_MULT * std
    lo = sma - STD_MULT * std
    width = up - lo
    hist = width.iloc[max(0, i - 100):i + 1].dropna()
    vol_pct = float((hist < width.iloc[i]).mean() * 100) if len(hist) and not pd.isna(width.iloc[i]) else 50.0
    if vol_pct >= 75:
        vol = "expanding (breakout conditions)"
    elif vol_pct <= 25:
        vol = "squeeze (compression - big move often follows)"
    else:
        vol = "normal"

    delta = s.diff()
    gain = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
    loss = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
    rsi = 100.0 - (100.0 / (1.0 + gain / loss))
    r_now = float(rsi.iloc[i]) if not pd.isna(rsi.iloc[i]) else 50.0
    r_prev = float(rsi.iloc[i - 3]) if not pd.isna(rsi.iloc[i - 3]) else r_now
    rsi_dir = "rising" if r_now > r_prev + 1.5 else ("falling" if r_now < r_prev - 1.5 else "flat")

    pos = None
    if not pd.isna(up.iloc[i]) and not pd.isna(lo.iloc[i]) and up.iloc[i] > lo.iloc[i]:
        pos = float((s.iloc[i] - lo.iloc[i]) / (up.iloc[i] - lo.iloc[i]) * 100.0)
    pos = max(0.0, min(100.0, pos)) if pos is not None else 50.0
    pos_note = ("upper band zone" if pos >= 70 else
                "lower band zone" if pos <= 30 else "middle of the range")

    pats = []
    last, prev = ohlc[-1], ohlc[-2]
    body = abs(last["c"] - last["o"])
    rng = max(last["h"] - last["l"], 1e-12)
    if body / rng < 0.10:
        pats.append("indecision candle (doji)")
    if min(last["o"], last["c"]) - last["l"] > 2 * max(body, 1e-12):
        pats.append("bullish rejection wick")
    if last["h"] - max(last["o"], last["c"]) > 2 * max(body, 1e-12):
        pats.append("bearish rejection wick")
    if prev["c"] < prev["o"] and last["c"] > last["o"] and last["c"] >= prev["o"] and last["o"] <= prev["c"]:
        pats.append("bullish engulfing")
    if prev["c"] > prev["o"] and last["c"] < last["o"] and last["c"] <= prev["o"] and last["o"] >= prev["c"]:
        pats.append("bearish engulfing")
    trio = [1 if ohlc[-k]["c"] > ohlc[-k]["o"] else -1 for k in (1, 2, 3)]
    if trio == [1, 1, 1]:
        pats.append("3 up-closes in a row")
    elif trio == [-1, -1, -1]:
        pats.append("3 down-closes in a row")
    if not pats:
        pats.append("no clear pattern on the last candle")

    setup = analyze(closes, pair=pair, period=period_sec)
    return {"ok": True, "pair": pair, "period": period_sec, "trend": trend,
            "trend_note": tnote, "vol": vol, "vol_pct": vol_pct, "rsi": r_now,
            "rsi_dir": rsi_dir, "pos": pos, "pos_note": pos_note, "patterns": pats,
            "setup": setup, "stamp": time.strftime("%H:%M:%S"),
            "feed": feed_note(pair, period_sec), "closes": closes}


def _ordinal(n):
    n = int(round(n))
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def watch_text(rep, pair, period_sec):
    st = rep["setup"]
    if st["signal"] in ("CALL", "PUT"):
        word = "BUY / Higher" if st["signal"] == "CALL" else "SELL / Lower"
        dot = "\U0001f7e2" if st["signal"] == "CALL" else "\U0001f534"
        edge = st.get("full_acc", 0.0)
        be = st.get("break_even") or 0.0
        tag = "EDGE" if st.get("tradeable") else "signal"
        hist = (f"{edge:.0f}% edge vs {be:.0f}% break-even "
                f"({st.get('horizon', HORIZON)}-bar target)")
        status = f"{dot} <b>{tag}: {word}</b> ({hist})"
    else:
        reason = st.get("gate_note") or st.get("note") or "not enough candle history yet"
        status = f"\U0001f6ab <b>NO TRADE</b> \u2014 {html.escape(reason, quote=False)}"
    lines = [
        f"\U0001f4ca <b>{html.escape(fmt_pair(pair))}</b> \u00b7 {tf_label(period_sec)} \u2014 live",
        "",
        f"\U0001f4c8 Trend: <b>{rep['trend']}</b> ({rep['trend_note']})",
        f"\U0001f32a\ufe0f Volatility: <b>{rep['vol']}</b> ({_ordinal(rep['vol_pct'])} percentile)",
        f"\U0001f4aa Momentum: RSI <b>{rep['rsi']:.1f}</b> {rep['rsi_dir']}",
        f"\U0001f4cd Price position: {rep.get('pos', 0):.0f}% through the bands ({rep.get('pos_note', 'middle of range')})",
        f"\U0001f56f\ufe0f Patterns: {html.escape(', '.join(rep['patterns']))}",
        f"\U0001f4e1 Data: {rep.get('feed', 'unknown')}",
        "",
        f"\U0001f3af Status: {status}",
        "",
        f"\U0001f552 updated {rep['stamp']}",
    ]
    return "\n".join(lines)

# ===== price series =====
def demo_series(key, period_sec, n=220):
    base = 150.0 if "JPY" in key else 1.0850
    vol = base * 0.0008 * np.sqrt(period_sec / 60.0)
    return list(np.cumsum(np.random.normal(0, vol, n)) + base)

def get_series(pair, period_sec, fetch=True, count=220):
    global DATA_SRC
    if pocket_payout is not None:
        try:
            # closed bars only for signal evaluation - the forming candle is
            # still changing and must never drive a decision
            closes = (pocket_payout.live_closes(pair, period_sec,
                                                include_forming=False)
                      if fetch and count <= 220 else None)
            if closes and len(closes) >= PERIOD + RSI_PERIOD + 5:
                DATA_SRC = "live stream"
                return closes, ""
            closes = (pocket_payout.get_closes(pair, period_sec, count) if fetch
                      else pocket_payout.peek_closes(pair, period_sec))
            if closes and len(closes) >= PERIOD + RSI_PERIOD + 5:
                DATA_SRC = "SSID snapshot"
                return closes, ""
        except Exception:
            pass
    DATA_SRC = "demo (random)"
    return demo_series(pair, period_sec), "SSID unavailable - demo feed."


def feed_note(pair, period_sec):
    if pocket_payout is None:
        return "demo feed"
    try:
        kind, age = pocket_payout.feed_status(pair, period_sec)
    except Exception:
        return "unknown"
    if kind == "live":
        return "live stream (now)" if (age or 0) < 1 else f"live stream ({age:.1f}s)"
    if kind == "snapshot":
        return f"snapshot ({age:.0f}s old)"
    return "unavailable"


def warm_feed(pair, period_sec, wait=10.0):
    if pocket_payout is None:
        return
    threading.Thread(target=pocket_payout.start_live,
                     args=(pair, int(period_sec)),
                     kwargs={"wait": wait}, daemon=True).start()

def fetch_ohlc(pair, period_sec, count=220, ttl=None):
    if pocket_payout is None:
        return None
    try:
        ohlc = (pocket_payout.live_ohlcv(pair, period_sec, include_forming=False)
                if count <= 220 else None)
        if not ohlc:
            extra = {} if ttl is None else {"ttl": ttl}
            ohlc = pocket_payout.get_ohlcv(pair, period_sec, count, **extra)
        return ohlc
    except Exception:
        return None

# ===== Telegram =====
def api_call(method, payload):
    try:
        r = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/{method}",
                          json=payload, timeout=15)
        return r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    except Exception as e:
        print(f"{method} failed:", e)
        return {}

def send(text, chat_id=None, keyboard=None, html_on=False):
    target = chat_id or CHAT_ID
    if not TELEGRAM_TOKEN or TELEGRAM_TOKEN.startswith("PASTE"):
        print(f"[DRY] {text}")
        return None
    if isinstance(target, str) and target.startswith("PASTE"):
        print(f"[WAIT FOR /start] {text}")
        return None
    payload = {"chat_id": target, "text": text}
    if html_on:
        payload["parse_mode"] = "HTML"
    if keyboard:
        payload["reply_markup"] = {"inline_keyboard": keyboard}
    res = api_call("sendMessage", payload)
    if not res.get("ok"):
        print("sendMessage failed:", res.get("description") or res)
        return None
    return str(res.get("result", {}).get("message_id"))

_IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_spike_img")

def _load_font(size):
    from PIL import ImageFont
    for path in (r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\segoeuib.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                 "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
                 r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()

def header_image(pair, signal, closes):
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return None
    try:
        os.makedirs(_IMG_DIR, exist_ok=True)
        W, H = 800, 300
        img = Image.new("RGB", (W, H), (13, 17, 23))
        d = ImageDraw.Draw(img)
        buy = signal == "CALL"
        color = (57, 255, 20) if buy else (255, 60, 47)
        dim = (30, 90, 45) if buy else (100, 35, 30)
        vals = [float(x) for x in (closes or [])[-80:]]
        if len(vals) >= 2:
            lo, hi = min(vals), max(vals)
            rng = (hi - lo) or 1e-12
            n = len(vals)
            pts = [(16 + (W - 32) * k / (n - 1),
                    (H - 26) - (v - lo) / rng * (H * 0.26)) for k, v in enumerate(vals)]
            d.line(pts, fill=dim, width=6, joint="curve")
            d.line(pts[-max(2, n // 3):], fill=color, width=6, joint="curve")
        d.text((24, 14), fmt_pair(pair), fill=(150, 160, 170), font=_load_font(30))
        word = "BUY" if buy else "SELL"
        d.text((24, 52), word, fill=color, font=_load_font(160))
        d.text((40 + _load_font(160).getlength(word), 70),
               "\u25b2" if buy else "\u25bc", fill=color, font=_load_font(80))
        path = os.path.join(_IMG_DIR, f"hdr_{int(time.time() * 1000)}.png")
        img.save(path)
        return path
    except Exception as e:
        print("header_image failed:", e)
        return None

def send_photo(path, caption, chat_id=None, keyboard=None, html_on=False):
    target = chat_id or CHAT_ID
    if not TELEGRAM_TOKEN or TELEGRAM_TOKEN.startswith("PASTE"):
        print(f"[DRY] (photo) {caption[:80]}")
        return None
    if isinstance(target, str) and target.startswith("PASTE"):
        print(f"[WAIT FOR /start] (photo) {caption[:80]}")
        return None
    data = {"chat_id": str(target)}
    if html_on:
        data["parse_mode"] = "HTML"
    if caption:
        data["caption"] = caption[:1024]
    if keyboard:
        data["reply_markup"] = json.dumps({"inline_keyboard": keyboard})
    try:
        with open(path, "rb") as f:
            r = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
                              data=data, files={"photo": f}, timeout=30)
        res = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        if not res.get("ok"):
            print("sendPhoto failed:", res.get("description") or res)
            return None
        return str(res["result"]["message_id"])
    except Exception as e:
        print("sendPhoto failed:", e)
        return None

def edit(text, chat_id, message_id, html_on=False):
    payload = {"chat_id": chat_id, "message_id": message_id, "text": text}
    if html_on:
        payload["parse_mode"] = "HTML"
    return api_call("editMessageText", payload)

def answer(cb_id, text=None):
    payload = {"callback_query_id": cb_id}
    if text:
        payload["text"] = text
    api_call("answerCallbackQuery", payload)

def get_updates(offset=None):
    try:
        r = requests.get(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates",
                         params={"offset": offset, "timeout": 30}, timeout=40)
        return r.json().get("result", [])
    except Exception as e:
        print("getUpdates failed:", e)
        return []

# ===== assets =====
def fmt_pair(norm):
    base = norm[:-4] if norm.endswith("-OTC") else norm
    name = base[:3] + "/" + base[3:] if (len(base) == 6 and base.isalpha()) else base
    if norm.endswith("-OTC"):
        name += " OTC"
    return name

_PAIR_CACHE = {"t": 0.0, "data": []}

def get_pairs():
    now = time.time()
    if _PAIR_CACHE["data"] and now - _PAIR_CACHE["t"] < 300:
        return list(_PAIR_CACHE["data"])
    pairs = []
    if pocket_payout is not None:
        try:
            rows = pocket_payout.asset_catalog()
            cur = [r for r in rows if r["type"] == "currency" and r["payout"] > 0]
            cur.sort(key=lambda r: (-r["payout"], r["norm"]))
            pairs = [(r["norm"], r["payout"]) for r in cur]
        except Exception:
            pairs = []
    if not pairs:
        pairs = [(p, 0) for p in DEFAULT_PAIRS]
    _PAIR_CACHE["t"] = now
    _PAIR_CACHE["data"] = pairs
    return list(pairs)

def market_pairs(market):
    pairs = get_pairs()
    if market == "otc":
        return [(p, pay) for p, pay in pairs if p.endswith("-OTC")]
    if market == "all":
        return list(pairs)
    return [(p, pay) for p, pay in pairs if not p.endswith("-OTC")]

def market_label(market):
    return {"otc": "OTC", "all": "BOTH (OTC + Stock)"}.get(market, "Stock")

def tf_label(sec):
    for s, lab in TIMES:
        if s == sec:
            return lab
    return "1 minutes"

def now_hm():
    return time.strftime("%I:%M %p").lstrip("0")

# ===== Spike screens =====
def screen_head(title, step):
    dots = "".join(("\U0001f7e6" if i < step else "\u2b1b") for i in range(4))
    return f"\U0001f7e2 Spike\n\n<b>{title}</b>\n\n{dots}\n"

def kb_menu():
    return [[{"text": "Start", "callback_data": "ui:market"}],
            [{"text": "\U0001f50e Find a trade", "callback_data": "scan:run"},
             {"text": "\U0001f501 Auto-scan", "callback_data": "scan:toggle"}]]

def kb_market():
    return [[{"text": "Stock", "callback_data": "mkt:stock"},
             {"text": "OTC", "callback_data": "mkt:otc"}],
            [{"text": "\U0001f310 Both (OTC + Stock)", "callback_data": "mkt:all"}],
            [{"text": "Back to Menu", "callback_data": "ui:menu"}]]

def kb_pairs(page, pairs, per=8):
    pages = max(1, (len(pairs) + per - 1) // per)
    page = max(0, min(page, pages - 1))
    rows = []
    chunk = pairs[page * per:(page + 1) * per]
    for i in range(0, len(chunk), 2):
        rows.append([{"text": fmt_pair(p), "callback_data": f"pr:{p}"} for p, _ in chunk[i:i + 2]])
    nav = [{"text": "\u25c0", "callback_data": f"pg:{max(0, page-1)}"},
           {"text": f"{page + 1}/{pages}", "callback_data": f"pg:{page}"},
           {"text": "\u25b6", "callback_data": f"pg:{min(pages - 1, page + 1)}"}]
    rows.append(nav)
    rows.append([{"text": "Back to Market", "callback_data": "ui:market"}])
    return rows

def kb_times():
    rows = [[{"text": lab, "callback_data": f"tm:{sec}"} for sec, lab in TIMES[i:i + 3]]
            for i in range(0, 6, 3)]
    rows.append([{"text": lab, "callback_data": f"tm:{sec}"} for sec, lab in TIMES[6:]])
    rows.append([{"text": "Back", "callback_data": "ui:pairback"}])
    return rows

def kb_signal_ctx(pair, signal, period_sec, ts=None):
    """Signal card keyboard whose Accept button carries its own context, so it
    still works after a bot restart (no in-memory state to lose)."""
    ts = int(ts if ts is not None else time.time())
    cb = f"acc:{pair}:{signal}:{int(period_sec)}:{ts}"
    if len(cb.encode("utf-8")) > 64:
        cb = f"acc:{pair}:{signal}:{int(period_sec)}"
    return [[{"text": "\u2705 Accept signal", "callback_data": cb}],
            [{"text": "\U0001f504 Another signal", "callback_data": "sig:again"},
             {"text": "\U0001f50e Find a trade", "callback_data": "scan:run"}],
            [{"text": "Back to Menu", "callback_data": "ui:menu"}]]


def kb_signal():
    """Navigation-only keyboard for cards that carry no actionable signal."""
    return [[{"text": "\U0001f504 Another signal", "callback_data": "sig:again"},
             {"text": "\U0001f50e Find a trade", "callback_data": "scan:run"}],
            [{"text": "Back to Menu", "callback_data": "ui:menu"}]]

# ===== signal card =====
def sparkline(vals):
    if not vals:
        return ""
    blocks = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1e-12
    return "".join(blocks[min(7, int((v - lo) / rng * 7.999))] for v in vals)

def tech_metrics(closes, ohlc, res):
    """Card readouts from the SAME stack the engine decides with:
    Bollinger position/volatility, Stochastic, CCI, ADX."""
    out = {"vol": "Normal", "vol_pct": 50.0, "rsi_label": "Neutral",
           "macd": "Neutral", "ma": "Ranging", "sentiment": "Sideways pressure",
           "price": None, "conditions": "Neutral"}
    if not closes or len(closes) < WARMUP_BARS:
        return out
    stk = _indicator_stack([float(x) for x in closes], ohlc)
    i = len(stk["s"]) - 1

    def fval(key):
        v = stk[key].iloc[i]
        return None if pd.isna(v) else float(v)

    price = fval("s")
    c_now, lo_now = fval("lower2"), fval("upper2")
    width = c_now and lo_now and (lo_now - c_now)
    hist = (stk["upper2"] - stk["lower2"]).iloc[max(0, i - 100):i + 1].dropna()
    if width and len(hist):
        out["vol_pct"] = float((hist < float(hist.iloc[-1])).mean() * 100)
        out["vol"] = ("High" if out["vol_pct"] >= 75 else
                      "Increased" if out["vol_pct"] >= 50 else
                      "Normal" if out["vol_pct"] >= 25 else "Low")

    k = fval("stoch_k")
    d = fval("stoch_d")
    if k is not None:
        zone = ("Oversold" if k <= STOCH_OS else
                "Overbought" if k >= STOCH_OB else "Neutral")
        dd = f"/D {d:.0f}" if d is not None else ""
        out["rsi_label"] = f"{zone} (K {k:.0f}{dd})"

    cci = fval("cci")
    if cci is not None:
        lbl = ("Oversold" if cci <= CCI_OS else
               "Overbought" if cci >= CCI_OB else "Neutral")
        out["macd"] = f"{lbl} ({cci:+.0f})"

    adx = fval("adx")
    if adx is not None:
        out["ma"] = (f"ADX {adx:.0f} - "
                     + ("ranging (rebounds favoured)" if adx < 20 else
                        "trending (no fades)" if adx >= ADX_TREND else
                        "trend building"))

    e9, e21 = fval("ema9"), fval("ema21")
    if e9 is not None and e21 is not None:
        out["sentiment"] = ("Upward pressure" if e9 > e21 else
                            "Downward pressure" if e9 < e21 else "Sideways pressure")

    out["price"] = price
    out["conditions"] = ("Optimal" if res.get("setup") else
                         "Favorable" if out["vol_pct"] >= 30 else "Neutral")
    return out

def card_caption(res, m, conf, pair, tf_txt):
    buy = res["signal"] == "CALL"
    word = "BUY" if buy else "SELL"
    arrow = "\u25b2" if buy else "\u25bc"
    mkt = "OTC" if pair.endswith("-OTC") else "Stock"
    dec = 3 if "JPY" in pair else 5
    def price(v):
        return f"{v:.{dec}f}" if v is not None else "n/a"
    regime = res.get("regime", "normal")
    mkey = str(globals().get("sel_model") or "engine")
    learn_parts = []
    mmeta = _ml_meta.get(f"{pair}#{mkey}") or {}
    if mmeta.get("acc"):
        oos = int(mmeta.get("oos_n") or 0)
        learn_parts.append(f"{mmeta['acc']:.0f}% walk-forward"
                           + (f" on {oos} test bars" if oos else ""))
    learn_line = (f"\U0001f9e0 {MODEL_NAMES.get(mkey, mkey)}: "
                  f"{' \u00b7 '.join(learn_parts)}\n\n" if learn_parts else "")
    win_lines = []
    aw, al, apct, an = account_perf()
    if an >= 1:
        win_lines.append(f"\U0001f3c6 Account: <b>{aw}W / {al}L</b> "
                         f"(<b>{apct:.0f}%</b> win \u00b7 real trades)")
    lpct, ln = model_perf(mkey)
    if ln >= 1:
        w = int(round(lpct / 100.0 * ln))
        win_lines.append(f"\U0001f4c8 Signals: <b>{w}W / {ln - w}L</b> "
                         f"(<b>{lpct:.0f}%</b> \u00b7 last {ln})")
    win_line = ("\n".join(win_lines) + "\n\n") if win_lines else ""
    return (
        f"<b>{html.escape(fmt_pair(pair))}</b> | {tf_txt} | {word} {arrow}\n\n"
        f"\U0001fab6 Market info:\n"
        f"\u2022 Volatility: <b>{m['vol']}</b>\n"
        f"\u2022 Sentiment: <b>{m['sentiment']}</b>\n\n"
        f"\U0001f9ed Technical overview:\n"
        f"\u2022 Current price: <b>{price(m['price'])} {mkt}</b>\n"
        f"\u2022 Stochastic: <b>{m['rsi_label']}</b>\n"
        f"\u2022 CCI: <b>{m['macd']}</b>\n"
        f"\u2022 ADX: <b>{m['ma']}</b>\n\n"
        f"\U0001f4e6 Signal edge:\n"
        f"\u2022 Edge: <b>{res.get('full_acc', 0):.0f}%</b> vs break-even "
        f"<b>{(res.get('break_even') or 0):.0f}%</b> "
        f"(payout {(res.get('payout') or 0):.0f}%) "
        f"\u00b7 {res.get('full_total', 0)} setups\n"
        f"{_fold_line(res)}"
        f"{_validated_line(res)}"
        f"\u2022 Market conditions: <b>{m['conditions']}</b>\n"
        f"\u2022 Regime: <b>{regime}</b>\n\n"
        f"{learn_line}"
        f"{win_line}"
        f"{now_hm()}"
    )

def _fold_line(res):
    """Per-fold hit-rate so edge stability is visible, or ''."""
    ac = res.get("fold_accs") or []
    if len(ac) < 2:
        return ""
    return ("\u2022 Folds: "
            + " \u00b7 ".join(f"{a:.0f}%" for a in ac) + "\n")


def _validated_line(res):
    """One-line summary of a validated backtest report, or ''."""
    vs = res.get("validated_stats")
    if not vs:
        return ""
    pf = vs.get("profit_factor")
    pf_txt = f" \u00b7 PF {pf:.2f}" if isinstance(pf, (int, float)) else ""
    return (f"\U0001f9ea Backtest: <b>{vs.get('win_rate', 0):.0f}%</b> win \u00b7 "
            f"exp {vs.get('expectancy', 0):+.3f}/bet \u00b7 "
            f"SQN {vs.get('sqn', 0)} \u00b7 DD {vs.get('max_drawdown_pct', 0):.0f}% "
            f"({vs.get('bets', 0)} bets{pf_txt})\n")


def no_trade_text(res, m, pair, tf_txt):
    edge = res.get("full_acc", 0.0)
    be = res.get("break_even")
    payout = res.get("payout")
    n = res.get("full_total", 0)
    reason = res.get("gate_note") or "No statistical edge right now."
    src = "Validated edge" if res.get("validated") else "Measured edge"
    lines = [
        f"\U0001f6ab <b>NO TRADE</b> \u2014 {html.escape(fmt_pair(pair))} | {tf_txt}",
        "",
        "This setup does not clear the break-even bar, so no BUY/SELL is issued.",
        "",
        f"\u2022 {src}: <b>{edge:.0f}%</b>"
        + (f" over {n} setups" if n else ""),
        f"\u2022 Break-even needed: <b>{be:.0f}%</b>" if be
        else "\u2022 Break-even: n/a (no payout)",
        f"\u2022 Payout: <b>{payout:.0f}%</b>" if payout
        else "\u2022 Payout: n/a",
    ]
    if res.get("validated_stats"):
        lines.append(_validated_line(res).rstrip("\n"))
    lines += [
        f"\u2022 Reason: {html.escape(reason)}",
        "",
        now_hm(),
    ]
    return "\n".join(lines)


def scan_pairs(period_sec, market="otc", max_pairs=30):
    """Run the edge gate over every pair in a market, ranked best-first.

    Returns a list of dicts (one per usable pair) sorted so any tradeable pair
    comes first, then by how far its measured edge sits above break-even. Pairs
    whose payout is already below MIN_PAYOUT are skipped before any fetch."""
    out = []
    t0 = time.time()
    SCAN_TIMING["started"] = t0
    SCAN_TIMING["running"] = True
    try:
        pairs = market_pairs(market)
    except Exception:
        pairs = []
    # multiple-comparison correction: this pair beat ~N others, so its
    # measured edge must clear a wider Wilson floor before we trust it
    n_cand = max(1, len(pairs))
    z_mult = min(2.5, math.sqrt(2.0 * math.log(max(2, n_cand))))

    def _one(pair, pay):
        if pay and pay < MIN_PAYOUT:
            return None
        try:
            ohlc = fetch_ohlc(pair, period_sec)
            if ohlc and len(ohlc) >= PERIOD + RSI_PERIOD + 6:
                closes = [r["c"] for r in ohlc]
            else:
                closes, note = get_series(pair, period_sec, fetch=True)
                if note or not closes:
                    return None
                ohlc = None
            res = analyze(closes, pair=pair, period=period_sec, ohlc=ohlc,
                          z_mult=z_mult)
            be = res.get("break_even")
            edge = res.get("full_acc", 0.0)
            return {
                "pair": pair, "signal": res.get("signal"), "edge": edge,
                "wilson": res.get("wilson", 0.0), "break_even": be,
                "payout": res.get("payout"), "total": res.get("full_total", 0),
                "tradeable": bool(res.get("tradeable")),
                "validated": bool(res.get("validated")),
                "reason": res.get("gate_reason", ""),
                "report_age": res.get("report_age"),
                "note": res.get("gate_note", ""),
                "margin": (edge - be) if be else -99.0,
                "res": res, "closes": closes,
            }
        except Exception:
            return None

    if market == "all":
        otc = [x for x in pairs if x[0].endswith("-OTC")]
        reg = [x for x in pairs if not x[0].endswith("-OTC")]
        half = max(1, max_pairs // 2)
        print(f"[scan] all: {len(otc)} OTC / {len(reg)} regular available")
        for pair, pay in otc[:half] + reg[:half]:
            r = _one(pair, pay)
            if r:
                out.append(r)
        # one side is often closed: backfill its empty slots from the other
        otc_ok = sum(1 for r in out if r["pair"].endswith("-OTC"))
        spare = []
        if len(out) - otc_ok < half:
            spare += otc[half:]
        if otc_ok < half:
            spare += reg[half:]
        for pair, pay in spare:
            if len(out) >= max_pairs:
                break
            r = _one(pair, pay)
            if r:
                out.append(r)
    else:
        for pair, pay in pairs[:max_pairs]:
            r = _one(pair, pay)
            if r:
                out.append(r)
    out.sort(key=lambda x: (x["tradeable"], x["margin"]), reverse=True)

    def _deepen(r):
        """Re-score a top candidate on far more history (default 900 bars).

        A 220-bar replay gives a very wide Wilson interval, so a genuine 55%
        edge can never clear break-even. More bars make the floor honest
        without loosening it."""
        pair, per = r["pair"], period_sec
        broad_n = len(r.get("closes") or [])
        diag = {}

        def mark(got, kept):
            DEEP_INFO[pair] = {"got": got, "want": SCAN_HISTORY,
                               "kept": kept,
                               "legacy": diag.get("legacy"),
                               "live": diag.get("live"),
                               "batches": diag.get("batches")}

        try:
            ohlc = fetch_ohlc(pair, per, SCAN_HISTORY, ttl=0)
            try:
                diag = dict(getattr(pocket_payout, "LAST_FETCH", {}) or {})
            except Exception:
                diag = {}
            closes = [x["c"] for x in ohlc] if ohlc else []
            if not closes or len(closes) < PERIOD + RSI_PERIOD + 6:
                mark(len(closes), broad_n)
                print(f"[deep] {pair}: no extra history "
                      f"(asked {SCAN_HISTORY} bars)")
                return r
            if len(closes) <= broad_n:
                # never let a shorter fetch weaken the estimate
                mark(len(closes), broad_n)
                print(f"[deep] {pair}: got {len(closes)} <= broad {broad_n} "
                      "bars, keeping broad")
                return r
            res = analyze(closes, pair=pair, period=per, ohlc=ohlc,
                          z_mult=z_mult)
        except Exception as e:
            mark(0, broad_n)
            print(f"[deep] {pair}: failed ({e})")
            return r
        be = res.get("break_even")
        edge = res.get("full_acc", 0.0)
        r.update({
            "signal": res.get("signal"), "edge": edge,
            "wilson": res.get("wilson", 0.0), "break_even": be,
            "payout": res.get("payout"), "total": res.get("full_total", 0),
            "tradeable": bool(res.get("tradeable")),
            "validated": bool(res.get("validated")),
            "reason": res.get("gate_reason", ""),
            "report_age": res.get("report_age"),
            "note": res.get("gate_note", ""),
            "margin": (edge - be) if be else -99.0,
            "res": res, "closes": closes, "history": len(closes),
        })
        mark(len(closes), len(closes))
        print(f"[deep] {pair}: {len(closes)} bars -> {edge:.0f}% vs "
              f"{be or 0:.0f}% "
              f"({'TRADEABLE' if r['tradeable'] else r['reason']})")
        return r

    for r in out[:max(1, SCAN_DEEP_N)]:
        _deepen(r)
    out.sort(key=lambda x: (x["tradeable"], x["margin"]), reverse=True)
    SCAN_TIMING["took"] = time.time() - t0
    SCAN_TIMING["running"] = False
    return out


REASON_LABELS = {"report": "negative report", "sample": "too few setups",
                 "edge": "edge too small", "wilson": "thin floor",
                 "fold": "recent fold weak", "payout": "payout/sample"}

DEEP_INFO = {}   # pair -> diagnostics for the last deep re-check
SCAN_TIMING = {"started": 0.0, "took": 0.0, "running": False}


def market_counts(results):
    otc = sum(1 for r in results if str(r.get("pair", "")).endswith("-OTC"))
    return otc, len(results) - otc


def reason_counts(results):
    c = {}
    for r in results:
        if r.get("tradeable"):
            continue
        k = r.get("reason") or "other"
        c[k] = c.get(k, 0) + 1
    return c


def reason_text(results):
    c = reason_counts(results)
    return ", ".join(f"{REASON_LABELS.get(k, k)} {v}"
                     for k, v in sorted(c.items(), key=lambda x: -x[1]))


def scan_text(results, period_sec, market):
    tf = tf_label(period_sec)
    head = f"\U0001f50e <b>Scan {market_label(market)}</b> \u00b7 {tf}\n"
    if not results:
        return head + "\nNo pairs with usable live data right now."
    checked = f"{len(results)} pairs checked"
    if market == "all":
        o, s = market_counts(results)
        checked += f" ({o} OTC / {s} stock)"
    lines = [head, f"{checked} \u2014 none clears break-even.", ""]
    blk = reason_text(results)
    if blk:
        lines.append(f"\U0001f6ab Blocked by: {blk}")
        lines.append("")
    for r in results[:5]:
        be = r.get("break_even") or 0.0
        pay = r.get("payout") or 0.0
        tag = "validated" if r.get("validated") else "replay"
        lines.append(f"\u2022 {html.escape(fmt_pair(r['pair']))}: "
                     f"<b>{r['edge']:.0f}%</b> vs {be:.0f}% "
                     f"(payout {pay:.0f}%, {tag})")
    if results[0].get("note"):
        lines += ["", f"Why the best was refused: {html.escape(results[0]['note'])}"]
    lines += ["", "Closest is still short \u2014 nothing worth risking money on.",
              "", now_hm()]
    return "\n".join(lines)


def autoscan_card_text(res, m, pair, tf_txt):
    buy = res["signal"] == "CALL"
    dot = "\U0001f7e2" if buy else "\U0001f534"
    return (f"{dot} <b>AUTO {html.escape('BUY' if buy else 'SELL')}</b> "
            f"\u2014 {html.escape(fmt_pair(pair))} | {tf_txt}\n\n"
            f"{card_caption(res, m, gate_score(res), pair, tf_txt)}")


def send_card(chat, res, m, conf, closes, pair=None, tf_txt=None):
    pair = pair or sel_pair
    tf_txt = tf_txt or tf_label(sel_time)
    cap = card_caption(res, m, conf, pair, tf_txt)
    path = header_image(pair, res["signal"], closes)
    if path:
        try:
            mid = send_photo(path, cap, chat_id=chat,
                             keyboard=kb_signal_ctx(pair, res["signal"], sel_time),
                             html_on=True)
        finally:
            try:
                os.remove(path)
            except Exception:
                pass
        if mid:
            remember_signal(mid, res, pair, sel_time, m.get("price"))
            return mid
    buy = res["signal"] == "CALL"
    dot = "\U0001f7e2" if buy else "\U0001f534"
    mid = send(f"{dot} <b>{'BUY' if buy else 'SELL'}</b> - "
               f"{html.escape(fmt_pair(pair))} | {tf_txt}\n\n{cap}",
               chat_id=chat,
               keyboard=kb_signal_ctx(pair, res["signal"], sel_time), html_on=True)
    remember_signal(mid, res, pair, sel_time, m.get("price"))
    return mid

GEN_STEPS = ["Volatility scan", "Stochastic", "CCI", "ADX Trend", "Signal Accuracy"]
GEN_PCT = [15, 35, 55, 75, 100]

def gen_text(pct, values):
    filled = max(0, min(5, (pct + 19) // 20))
    bar = "\u25a0" * filled + "\u25a1" * (5 - filled)
    lines = [f"Signal Generation...\n[{bar}] {pct}%", ""]
    for idx, name in enumerate(GEN_STEPS):
        v = values[idx] if idx < len(values) else "\u2026"
        prefix = "\u2514\u2500" if idx == len(GEN_STEPS) - 1 else "\u251c\u2500"
        lines.append(f"{prefix} {name}: {v}")
    return "\n".join(lines)

HELP_TEXT = (
    "/start - Main menu\n"
    "/scan - Scan every pair for a real edge now\n"
    "/auto - Toggle continuous auto-scan alerts\n"
    "/amount N - Set the stake for accepted signals\n"
    "/trading on|off - Enable/disable real orders (persists across restarts)\n"
    "/gate strict|point - Strict Wilson floor vs point-edge mode\n"
    "/status - Is the bot alive, last scan result, feed, balance\n"
    "/stop - Stop all signals\n"
    "Strategy engine: Bollinger 20/2 + 20/3, Stochastic 6/3 (SMA smooth 3), "
    "CCI 25, ADX (smoothing 1, DI 5) - band-tag reversal setups only while "
    "ADX < 25, gated by the "
    "setup's measured out-of-sample hit-rate vs payout break-even.\n"
    "Flow: Start -> market (Stock/OTC/Both) -> pair -> time -> signal\n"
    "Tap Find a trade to scan all pairs and get pinged only when one clears "
    "break-even.\n"
    "Accept signal places a real order only when trading is enabled with "
    "/trading on (and your SSID is saved). Otherwise it is display-only.\n"
    "Daily risk stop: Accept pauses after the configured realized loss "
    "or order count for the day.\n"
    "Accepted orders send a WIN/LOSS/REFUND message with profit and balance "
    "once the account deal settles."
)

if __name__ == "__main__":
    if not TELEGRAM_TOKEN:
        print("TELEGRAM_TOKEN is not set. Set the env var to your BotFather token,")
        print('e.g. in PowerShell:  $env:TELEGRAM_TOKEN="<new token>"  then run again.')
        raise SystemExit(1)

    import signal

    def _clean_exit(signum, frame):
        print("\nStopped.")
        raise SystemExit(0)

    try:
        signal.signal(signal.SIGINT, _clean_exit)
        signal.signal(signal.SIGTERM, _clean_exit)
    except Exception:
        pass

    print("Bot starting (SPIKE UI, live feed, on-demand signals)...")
    S = load_settings()
    sel_model = str(S.get("model") or "engine")
    sel_market = str(S.get("market") or "all")
    sel_pair = str(S.get("last_pair") or "EURUSD-OTC")
    sel_time = int(S.get("last_time") or 60)
    auto_on = False
    CURRENT_MODEL = sel_model
    print(f"Market: {market_label(sel_market)} | {tf_label(sel_time)} | {sel_pair}")
    if S.get("allow_trading"):
        os.environ["ALLOW_TRADING"] = "1"
    else:
        os.environ.pop("ALLOW_TRADING", None)
    print("Trading: " + ("ENABLED (Accept places real orders)"
                         if trading_on() else "display-only"))
    GATE_MODE = S["gate_mode"]
    save_settings({"gate_mode": GATE_MODE})
    print("Gate: " + ("strict (Wilson floor)"
                      if GATE_MODE == "strict" else "point (+margin)"))
    EDGE_MARGIN = float(S.get("edge_margin", EDGE_MARGIN))
    MIN_BACKTEST_N = max(100, int(S.get("min_backtest_n", MIN_BACKTEST_N)))
    MIN_PAYOUT = float(S.get("min_payout", MIN_PAYOUT))
    TRADE_AMOUNT = max(0.01, float(S.get("trade_amount", TRADE_AMOUNT)))
    DAILY_MAX_LOSS = max(1.0, float(S.get("daily_max_loss", DAILY_MAX_LOSS)))
    DAILY_MAX_TRADES = max(1, int(S.get("daily_max_trades", DAILY_MAX_TRADES)))
    MAX_STAKE_BALANCE_PCT = max(0.5,
                                float(S.get("max_stake_pct",
                                            MAX_STAKE_BALANCE_PCT)))
    print(f"Risk: Accept stops after {DAILY_MAX_LOSS:g} units realized loss "
          f"or {DAILY_MAX_TRADES} orders today; stake <= "
          f"{MAX_STAKE_BALANCE_PCT:.0f}% of balance.")
    update_offset = None
    # fixed owner: explicit env id wins, else the TELEGRAM_CHAT_ID, else
    # whoever sends /start first (last resort, as before)
    owner_id = (str(os.getenv("TELEGRAM_OWNER_ID") or "").strip()
                or (None if str(CHAT_ID).startswith("PASTE") else str(CHAT_ID)))
    last_state = {}
    auto_tick = 0.0
    acct_tick = 0.0
    settle_tick = 0.0
    acct_busy = threading.Lock()
    settle_busy = threading.Lock()
    last_page = 0
    warm_feed(sel_pair, sel_time, wait=20.0)
    _ml_load()
    ml_train_async(sel_pair, sel_time)

    WATCH_REFRESH = 15
    WATCH_ALERT_GAP = 120
    watch = {"on": False, "chat": None, "msg": None, "t": 0.0, "alert": 0.0,
             "last_sig": None}

    STATUS = {"started": time.time(), "last_scan": 0.0, "scan_n": 0,
              "scan_otc": 0, "scan_stock": 0, "best": None, "last_card": 0.0,
              "blocked": ""}

    AUTOSCAN_GAP = 300
    AUTOSCAN_COOLDOWN = 900
    AUTOSCAN_MAX_PAIRS = 16
    scan_lock = threading.Lock()
    scan_state = {"alerted": {}}
    autoscan = {"on": False, "chat": None, "t": 0.0}

    def _emit_signal(chat, pair, period_sec, res, closes):
        m = tech_metrics(closes, fetch_ohlc(pair, period_sec), res)
        mid = send(autoscan_card_text(res, m, pair, tf_label(period_sec)), chat_id=chat,
                   keyboard=kb_signal_ctx(pair, res["signal"], period_sec),
                   html_on=True)
        STATUS["last_card"] = time.time()
        remember_signal(mid, res, pair, period_sec, m.get("price"))
        track_add(pair, period_sec, res["signal"], m.get("price"), sel_model,
                  res.get("regime"), features=_feat_last(closes))

    def run_scan(chat):
        mid = send(f"\U0001f50e Scanning {market_label(sel_market)} pairs for a real edge\u2026",
                   chat_id=chat, html_on=True)

        def worker():
            nonlocal mid
            if not scan_lock.acquire(timeout=300):
                send("A scan is stuck \u2014 try again in a moment.", chat_id=chat)
                return
            try:
                results = scan_pairs(sel_time, sel_market, AUTOSCAN_MAX_PAIRS)
                STATUS["last_scan"] = time.time()
                STATUS["scan_n"] = len(results)
                STATUS["scan_otc"], STATUS["scan_stock"] = market_counts(results)
                STATUS["blocked"] = reason_text(results)
                STATUS["best"] = results[0] if results else None
                best = next((r for r in results if r["tradeable"]), None)
                if best:
                    STATUS["last_card"] = time.time()
                    res, closes = best["res"], best["closes"]
                    m = tech_metrics(closes, fetch_ohlc(best["pair"], sel_time), res)
                    cap = autoscan_card_text(res, m, best["pair"], tf_label(sel_time))
                    txt = (f"\u2728 <b>Edge found</b> \u2014 cleared break-even "
                           f"({best['edge']:.0f}% vs {best['break_even']:.0f}%).\n\n{cap}")
                    if mid:
                        edit(txt, chat, mid, html_on=True)
                        api_call("editMessageReplyMarkup", {"chat_id": chat,
                                 "message_id": mid,
                                 "reply_markup": {"inline_keyboard":
                                                  kb_signal_ctx(best["pair"],
                                                                res["signal"],
                                                                sel_time)}})
                    else:
                        mid = send(txt, chat_id=chat,
                                   keyboard=kb_signal_ctx(best["pair"],
                                                          res["signal"], sel_time),
                                   html_on=True)
                    remember_signal(mid, res, best["pair"], sel_time, m.get("price"))
                    track_add(best["pair"], sel_time, res["signal"], m.get("price"),
                              sel_model, res.get("regime"), features=_feat_last(closes))
                else:
                    txt = scan_text(results, sel_time, sel_market)
                    if mid:
                        edit(txt, chat, mid, html_on=True)
                    else:
                        send(txt, chat_id=chat, html_on=True)
            except Exception as e:
                msg = f"\u26a0\ufe0f Scan failed: {html.escape(str(e))}"
                if mid:
                    edit(msg, chat, mid, html_on=True)
                else:
                    send(msg, chat_id=chat)
            finally:
                scan_lock.release()

        threading.Thread(target=worker, daemon=True).start()

    def accept_signal(chat, pair, signal, period, ts=0):
        if signal not in ("CALL", "PUT") or not pair:
            send("This button carries no signal \u2014 ask for a fresh one.",
                 chat_id=chat)
            return
        try:
            period = int(period)
        except Exception:
            period = int(sel_time)
        if ts:
            age = time.time() - int(ts)
            # stale signal = the setup bar is long gone; allow only a small
            # fraction of the expiry as acceptance delay
            limit = max(10, min(60, int(period) // 2))
            if age > limit:
                send(f"\u231b <b>Signal expired</b> \u2014 it is {int(age)}s old, "
                     f"past the {limit}s window for a "
                     f"{tf_label(period)} setup. Ask for a fresh one.",
                     chat_id=chat, html_on=True)
                return
        if pocket_payout is None:
            send("Live trading is unavailable on this host (library not loaded).",
                 chat_id=chat)
            return
        if not trading_on():
            send("\U0001f512 <b>Accept is disabled.</b> Send "
                 "/trading on to allow real orders.\n\nSignal: "
                 f"{signal} {html.escape(fmt_pair(pair))} \u00b7 "
                 f"{tf_label(period)}", chat_id=chat, html_on=True)
            return
        if risk_daily_block():
            send("\U0001f6d1 <b>Daily risk stop reached</b> \u2014 today's "
                 "realized loss limit or order cap is hit. No more orders "
                 "until tomorrow. Send /status to review.", chat_id=chat,
                 html_on=True)
            return
        amount = TRADE_AMOUNT
        bal0 = None
        try:
            bal0 = pocket_payout.account_balance()
        except Exception:
            bal0 = None
        if bal0 is not None and amount > bal0 * MAX_STAKE_BALANCE_PCT / 100.0:
            send(f"\u26a0 <b>Stake too large</b>: {amount:g} is more than "
                 f"{MAX_STAKE_BALANCE_PCT:.0f}% of balance {bal0:.2f}. "
                 f"Use /amount to lower it.", chat_id=chat, html_on=True)
            return
        ok, msg = pocket_payout.place_trade(pair, signal, amount, period)
        if ok:
            try:
                note_placed(chat, msg, pair, signal, period)
            except Exception as e:
                print("note_placed failed:", e)
            bal = None
            try:
                bal = pocket_payout.account_balance()
            except Exception:
                pass
            extra = f"\nBalance: {bal:.2f}" if bal is not None else ""
            send(f"\u2705 <b>Order placed</b>\n{signal} "
                 f"{html.escape(fmt_pair(pair))} \u00b7 {tf_label(period)} "
                 f"\u00b7 stake {amount:g}\nTicket: {html.escape(str(msg))}{extra}",
                 chat_id=chat, html_on=True)
        else:
            send(f"\u274c <b>Order failed</b>: {html.escape(str(msg))}",
                 chat_id=chat, html_on=True)

    def autoscan_tick():
        chat = autoscan["chat"]

        def worker():
            if not scan_lock.acquire(blocking=False):
                return
            try:
                results = scan_pairs(sel_time, sel_market, AUTOSCAN_MAX_PAIRS)
                STATUS["last_scan"] = time.time()
                STATUS["scan_n"] = len(results)
                STATUS["scan_otc"], STATUS["scan_stock"] = market_counts(results)
                STATUS["blocked"] = reason_text(results)
                STATUS["best"] = results[0] if results else None
                if results:
                    top = results[0]
                    blocked = ("" if top["tradeable"]
                               else f" \u00b7 blocked: {STATUS['blocked']}")
                    hist = (f" [deep {top['history']} bars]"
                            if top.get("history") else "")
                    print(f"[autoscan] {market_label(sel_market)}: "
                          f"{len(results)} pairs "
                          f"({STATUS['scan_otc']} OTC / "
                          f"{STATUS['scan_stock']} stock), best "
                          f"{top['pair']} {top['edge']:.0f}% vs "
                          f"{top['break_even'] or 0:.0f}%{hist} "
                          f"({'TRADEABLE' if top['tradeable'] else 'no edge'})"
                          f"{blocked}")
                else:
                    print("[autoscan] no usable pairs")
                now = time.time()
                for r in results:
                    if not r["tradeable"]:
                        continue
                    if now - scan_state["alerted"].get(r["pair"], 0) < AUTOSCAN_COOLDOWN:
                        continue
                    scan_state["alerted"][r["pair"]] = now
                    _emit_signal(chat, r["pair"], sel_time, r["res"], r["closes"])
                    print(f"[autoscan] alert {r['signal']} {r['pair']} "
                          f"{r['edge']:.0f}% vs {r['break_even']:.0f}%")
            except Exception as e:
                print("autoscan failed:", e)
            finally:
                scan_lock.release()

        threading.Thread(target=worker, daemon=True).start()

    def start_watch(chat):
        watch["on"] = True
        watch["chat"] = chat
        watch["t"] = 0.0
        watch["alert"] = 0.0
        watch["last_sig"] = None
        rep = market_report(sel_pair, sel_time, fetch=True) if pocket_payout else None
        if rep is None:
            watch["on"] = False
            send("\U0001f4ca Live analysis is unavailable for this pair right now.",
                 chat_id=chat)
            return
        mid = send(watch_text(rep, sel_pair, sel_time), chat_id=chat,
                   keyboard=[[{"text": "\U0001f6d1 Stop watch", "callback_data": "w:stop"}],
                             [{"text": "Back to Menu", "callback_data": "ui:menu"}]],
                   html_on=True)
        watch["msg"] = mid

    def stop_watch(chat, note="\U0001f4ca Live analysis stopped."):
        watch["on"] = False
        if watch["msg"] and chat:
            edit(note, chat, watch["msg"])
        watch["msg"] = None

    def persist():
        save_settings({"last_pair": sel_pair, "last_time": sel_time, "auto": auto_on,
                       "model": sel_model, "market": sel_market,
                       "edge_margin": EDGE_MARGIN, "min_backtest_n": MIN_BACKTEST_N,
                       "min_payout": MIN_PAYOUT, "trade_amount": TRADE_AMOUNT,
                       "daily_max_loss": DAILY_MAX_LOSS,
                       "daily_max_trades": DAILY_MAX_TRADES,
                       "max_stake_pct": MAX_STAKE_BALANCE_PCT,
                       "allow_trading": trading_on(), "gate_mode": GATE_MODE})

    def send_menu(chat):
        send(f"{screen_head('Spike', 1)}\nSignal engine for Pocket Option.\n"
             f"Real live market data. No orders are placed.",
             chat_id=chat, keyboard=kb_menu(), html_on=True)

    def send_market_screen(chat, edit_mid=None):
        text = f"{screen_head('Market', 1)}\nSelect a market:  <i>{now_hm()}</i>"
        if edit_mid:
            edit(text, chat, edit_mid, html_on=True)
            api_call("editMessageReplyMarkup", {"chat_id": chat, "message_id": edit_mid, "reply_markup": {"inline_keyboard": kb_market()}})
        else:
            send(text, chat_id=chat, keyboard=kb_market(), html_on=True)

    def send_pair_screen(chat, page=0, edit_mid=None):
        pairs = market_pairs(sel_market)
        text = f"{screen_head('Pair', 2)}\nSelect a currency pair:  <i>{now_hm()}</i>"
        if not pairs:
            text = f"{screen_head('Pair', 2)}\nNo {market_label(sel_market)} currency pairs are online right now. Try another market."
            keyboard = kb_market()
        else:
            pages = max(1, (len(pairs) + 7) // 8)
            page = max(0, min(page, pages - 1))
            global last_page
            last_page = page
            keyboard = kb_pairs(page, pairs)
        if edit_mid:
            edit(text, chat, edit_mid, html_on=True)
            api_call("editMessageReplyMarkup", {"chat_id": chat, "message_id": edit_mid, "reply_markup": {"inline_keyboard": keyboard}})
        else:
            send(text, chat_id=chat, keyboard=keyboard, html_on=True)

    def send_time_screen(chat, edit_mid=None):
        text = f"{screen_head('Time', 3)}\nSelect trading time for {html.escape(fmt_pair(sel_pair))}:"
        if edit_mid:
            edit(text, chat, edit_mid, html_on=True)
            api_call("editMessageReplyMarkup", {"chat_id": chat, "message_id": edit_mid, "reply_markup": {"inline_keyboard": kb_times()}})
        else:
            send(text, chat_id=chat, keyboard=kb_times(), html_on=True)

    def send_result(res, chat, m, conf, closes):
        pair_txt = fmt_pair(sel_pair)
        tf_txt = tf_label(sel_time)
        if res.get("demo"):
            send(f"\u26a0\ufe0f Live Pocket market data is unavailable right now for "
                 f"{html.escape(pair_txt)} ({tf_txt}).\n\n"
                 f"A real BUY/SELL card needs live candles - tap the signal button "
                 f"again in a minute and it will appear.",
                 chat_id=chat, keyboard=kb_signal(), html_on=True)
            return False
        if res["signal"] == "NO TRADE":
            send(no_trade_text(res, m, sel_pair, tf_txt), chat_id=chat,
                 keyboard=kb_signal(), html_on=True)
            return False
        if res["signal"] not in ("CALL", "PUT"):
            send(f"\U0001f504 Still loading candle history for "
                 f"{html.escape(pair_txt)} ({tf_txt}). Tap the signal button again in "
                 f"a few seconds - a signal is always given once the data is ready.",
                 chat_id=chat, keyboard=kb_signal(), html_on=True)
            return False
        return send_card(chat, res, m, conf, closes)

    def run_generation(chat):
        global CURRENT_MODEL
        CURRENT_MODEL = sel_model
        mid = send(gen_text(0, []), chat_id=chat)
        values = []

        def reveal(pct):
            if mid:
                edit(gen_text(pct, values), chat, mid)
            time.sleep(0.4)

        closes, note = get_series(sel_pair, sel_time, fetch=True)
        if pocket_payout is not None and DATA_SRC != "live stream":
            if pocket_payout.start_live(sel_pair, sel_time, wait=8.0):
                closes, note = get_series(sel_pair, sel_time, fetch=True)
        res = analyze(closes, sel_model, news_symbol=sel_pair.replace("-OTC", ""),
                      pair=sel_pair, period=sel_time)
        res["demo"] = bool(note)
        ohlc = fetch_ohlc(sel_pair, sel_time)
        m = tech_metrics(closes, ohlc, res)
        conf = gate_score(res, sel_model)

        params = MODEL_PARAMS[sel_model]
        reveal_vals = [
            f"{m['vol']} \u26a1",
            m["rsi_label"],
            m["macd"],
            m["ma"],
            f"{res.get('full_acc', 0):.0f}% ({res.get('full_total', 0)} past signals)",
        ]
        for idx, val in enumerate(reveal_vals):
            values.append(val)
            reveal(GEN_PCT[idx])
        time.sleep(0.4)
        ok = send_result(res, chat, m, conf, closes)
        if ok:
            STATUS["last_card"] = time.time()
        if ok and res.get("signal") in ("CALL", "PUT"):
            track_add(sel_pair, sel_time, res["signal"], m.get("price"),
                      sel_model, res.get("regime"),
                      features=_feat_last(closes))
            ml_train_async(sel_pair, sel_time)
        return ok

    _processed_cbs = set()

    def handle_update(upd):
        global owner_id, sel_model, sel_market, sel_pair, sel_time, auto_on, last_page
        global TRADE_AMOUNT, GATE_MODE
        if "callback_query" in upd:
            cb = upd["callback_query"]
            cb_id = cb.get("id", "")
            data = cb.get("data", "")
            chat = str(cb.get("message", {}).get("chat", {}).get("id", ""))
            if cb_id in _processed_cbs:
                answer(cb_id)
                return
            _processed_cbs.add(cb_id)
            if len(_processed_cbs) > 500:
                _processed_cbs.clear()
            if owner_id and chat and chat != owner_id:
                answer(cb_id)
                return
            answer(cb_id)
            if data == "ui:menu":
                send_menu(chat)
            elif data in ("ui:model", "ui:market"):
                mid = cb.get("message", {}).get("message_id")
                send_market_screen(chat, edit_mid=mid)
            elif data.startswith("mkt:"):
                sel_market = data.split(":", 1)[1]
                persist()
                if autoscan["on"]:
                    send(f"\U0001f310 Market set to <b>{market_label(sel_market)}</b> "
                         f"\u2014 auto-scan now covers it.", chat_id=chat,
                         html_on=True)
                mid = cb.get("message", {}).get("message_id")
                send_pair_screen(chat, 0, edit_mid=mid)
            elif data.startswith("pg:"):
                try:
                    page = int(data.split(":")[1])
                except Exception:
                    page = 0
                mid = cb.get("message", {}).get("message_id")
                send_pair_screen(chat, page, edit_mid=mid)
            elif data.startswith("pr:"):
                sel_pair = data.split(":", 1)[1]
                persist()
                warm_feed(sel_pair, sel_time)
                ml_train_async(sel_pair, sel_time)
                mid = cb.get("message", {}).get("message_id")
                send_time_screen(chat, edit_mid=mid)
            elif data == "ui:pairback":
                mid = cb.get("message", {}).get("message_id")
                send_pair_screen(chat, last_page, edit_mid=mid)
            elif data.startswith("tm:"):
                try:
                    sel_time = int(data.split(":")[1])
                except Exception:
                    sel_time = 60
                persist()
                warm_feed(sel_pair, sel_time)
                run_generation(chat)
            elif data == "sig:again":
                run_generation(chat)
            elif data.startswith("acc:"):
                parts = data.split(":")
                if len(parts) >= 4:
                    ts = int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else 0
                    accept_signal(chat, parts[1], parts[2], parts[3], ts)
                else:
                    send("This signal button is invalid \u2014 request a fresh one.",
                         chat_id=chat)
            elif data == "sig:accept":
                mid = cb.get("message", {}).get("message_id")
                ctx = PENDING_SIGNALS.get(mid)
                if ctx:
                    accept_signal(chat, ctx["pair"], ctx["signal"],
                                  ctx["period"], ctx.get("ts", 0))
                else:
                    send("This signal has expired \u2014 ask for a fresh one.",
                         chat_id=chat)
            elif data == "scan:run":
                run_scan(chat)
            elif data == "scan:toggle":
                autoscan["on"] = not autoscan["on"]
                autoscan["chat"] = chat
                autoscan["t"] = 0.0
                if autoscan["on"]:
                    send(f"\U0001f501 <b>Auto-scan ON</b> \u2014 {market_label(sel_market)}, "
                         f"{tf_label(sel_time)}. I'll check every "
                         f"{AUTOSCAN_GAP}s and message you only when a pair clears "
                         f"break-even.", chat_id=chat, html_on=True)
                else:
                    send("\U0001f501 Auto-scan OFF.", chat_id=chat)
            elif data == "w:stop":
                stop_watch(chat)
            return

        msg = upd.get("message", {})
        chat = str(msg.get("chat", {}).get("id", ""))
        raw = msg.get("text", "").strip()
        text = raw.lower()
        if not chat or not text:
            return
        if owner_id is None and text.startswith("/start"):
            owner_id = chat
            send_menu(chat)
            return
        if owner_id and chat != owner_id:
            send("Not authorized.", chat_id=chat)
            return
        cmd = text.split()[0]

        if cmd in ("/start", "start"):
            send_menu(chat)
        elif cmd in ("/stop", "stop"):
            auto_on = False
            autoscan["on"] = False
            persist()
            if watch["on"]:
                stop_watch(chat)
            send("All signals stopped.", chat_id=chat)
        elif cmd in ("/scan", "scan"):
            run_scan(chat)
        elif cmd in ("/amount", "amount", "/stake", "stake"):
            parts = raw.split()
            if len(parts) >= 2:
                try:
                    TRADE_AMOUNT = max(0.01, float(parts[1]))
                    persist()
                    send(f"Stake set to {TRADE_AMOUNT:g}.", chat_id=chat)
                except Exception:
                    send("Usage: /amount 1", chat_id=chat)
            else:
                send(f"Stake is {TRADE_AMOUNT:g}. Usage: /amount 1", chat_id=chat)
        elif cmd in ("/trading", "trading"):
            parts = raw.split()
            arg = parts[1].lower() if len(parts) >= 2 else ""
            if arg in ("on", "1", "yes", "enable", "enabled"):
                os.environ["ALLOW_TRADING"] = "1"
                persist()
                send("\U0001f513 <b>Trading ENABLED</b> \u2014 Accept now places "
                     "real orders on the linked account (demo or live), stake "
                     f"{TRADE_AMOUNT:g}. It stays on across restarts. Send "
                     "/trading off to disable.", chat_id=chat, html_on=True)
            elif arg in ("off", "0", "no", "disable", "disabled"):
                os.environ.pop("ALLOW_TRADING", None)
                persist()
                send("\U0001f512 <b>Trading disabled</b> \u2014 the bot is "
                     "display-only again.", chat_id=chat, html_on=True)
            else:
                send(f"Trading is "
                     f"{'ENABLED' if trading_on() else 'disabled'}. "
                     "Usage: /trading on | /trading off", chat_id=chat,
                     html_on=True)
        elif cmd in ("/gate", "gate"):
            parts = raw.split()
            arg = parts[1].lower() if len(parts) >= 2 else ""
            if arg in ("strict", "wilson"):
                GATE_MODE = "strict"
                persist()
                send("\U0001f6e1 <b>Gate: strict</b> \u2014 a pair must clear "
                     "break-even even on the 95% Wilson floor. Honest, but "
                     "rarely fires with short history.", chat_id=chat,
                     html_on=True)
            elif arg in ("point", "pragmatic"):
                GATE_MODE = "point"
                persist()
                send("\u26a1 <b>Gate: point</b> \u2014 fires when the measured "
                     "edge clears break-even + margin (no floor). Lower "
                     "confidence; keep it on demo.", chat_id=chat, html_on=True)
            else:
                send(f"Gate is <b>{GATE_MODE}</b>. "
                     "Usage: /gate strict | /gate point", chat_id=chat,
                     html_on=True)
        elif cmd in ("/auto", "auto"):
            autoscan["on"] = not autoscan["on"]
            autoscan["chat"] = chat
            autoscan["t"] = 0.0
            send(f"\U0001f501 Auto-scan {'ON' if autoscan['on'] else 'OFF'}.",
                 chat_id=chat)
        elif cmd in ("/status", "status"):
            up = int(time.time() - STATUS["started"])
            h, rem = divmod(up, 3600)
            mnt = rem // 60
            lines = ["\U0001f4e1 <b>Status</b>",
                     f"Uptime: {h}h {mnt}m",
                     f"Market: {market_label(sel_market)} \u00b7 {tf_label(sel_time)} "
                     f"\u00b7 {html.escape(fmt_pair(sel_pair))}",
                     f"Auto-scan: {'ON' if autoscan['on'] else 'off'}",
                     f"Watch: {'ON' if watch['on'] else 'off'}",
                      "Trading: " + ("ENABLED (real orders)"
                                     if trading_on()
                                     else "display-only (Accept is disabled)"),
                      "Gate: " + ("strict (Wilson floor)"
                                  if GATE_MODE == "strict"
                                  else "point (+margin, no floor)")]
            if STATUS["last_scan"]:
                ago = int(time.time() - STATUS["last_scan"])
                b = STATUS["best"]
                split = ""
                if sel_market == "all":
                    split = (f" ({STATUS.get('scan_otc', 0)} OTC / "
                             f"{STATUS.get('scan_stock', 0)} stock)")
                if b:
                    bars = (f" \u00b7 {b['history']} bars"
                            if b.get("history") else "")
                    took = (f" \u00b7 took {SCAN_TIMING['took']:.0f}s"
                            if SCAN_TIMING.get("took") else "")
                    lines.append(f"Last scan: {ago}s ago \u00b7 {STATUS['scan_n']} "
                                 f"pairs{split} \u00b7 best "
                                 f"{html.escape(fmt_pair(b['pair']))} "
                                 f"{b['edge']:.0f}% vs "
                                 f"{(b['break_even'] or 0):.0f}%{bars}{took}")
                    if SCAN_TIMING.get("running"):
                        lines.append(
                            "Scan: running "
                            f"({int(time.time() - SCAN_TIMING['started'])}s)")
                    if b.get("note"):
                        lines.append(f"Why: {html.escape(b['note'])}")
                    if STATUS.get("blocked"):
                        lines.append(f"Blocked: {html.escape(STATUS['blocked'])}")
                    di = DEEP_INFO.get(b["pair"])
                    if di:
                        src = (f" (legacy {di.get('legacy')} / "
                               f"live {di.get('live')} / "
                               f"{di.get('batches')} batches)")
                        lines.append(
                            f"Deep: got {di['got']}/{di['want']} bars"
                            f"{src} \u00b7 kept {di['kept']}")
                else:
                    lines.append(f"Last scan: {ago}s ago \u00b7 no usable pairs "
                                 "(check payout/SSID)")
            else:
                lines.append("Last scan: none yet \u2014 tap Find a trade")
            if pocket_payout is not None:
                try:
                    st, age = pocket_payout.feed_status(sel_pair, sel_time)
                    lines.append("Feed: " + st +
                                 (f" ({age:.0f}s)" if age else ""))
                except Exception:
                    pass
                try:
                    bal = pocket_payout.account_balance()
                    if bal is not None:
                        lines.append(f"Balance: {bal:.2f}")
                except Exception:
                    pass
            send("\n".join(lines), chat_id=chat, html_on=True)
        elif cmd in ("/help", "help"):
            send(HELP_TEXT, chat_id=chat)

    import queue as _queue
    update_q = _queue.Queue()

    def _poll_worker():
        global update_offset
        while True:
            try:
                updates = get_updates(update_offset)
            except Exception:
                updates = []
            for upd in updates:
                update_offset = upd["update_id"] + 1
                update_q.put(upd)
            if not updates:
                time.sleep(0.2)

    threading.Thread(target=_poll_worker, daemon=True).start()

    while True:
        while True:
            try:
                upd = update_q.get_nowait()
            except _queue.Empty:
                break
            try:
                handle_update(upd)
            except Exception as e:
                import traceback
                traceback.print_exc()
                try:
                    send(f"\u26a0 {type(e).__name__}: {e}",
                         chat_id=upd.get("message", {}).get("chat", {})
                         .get("id"))
                except Exception:
                    pass

        if watch["on"] and time.time() - watch["t"] >= WATCH_REFRESH:
            watch["t"] = time.time()
            rep = market_report(sel_pair, sel_time, fetch=True)
            if rep is None:
                stop_watch(watch["chat"], "\U0001f4ca Live analysis stopped - "
                                           "market data unavailable.")
            else:
                if watch["msg"]:
                    edit(watch_text(rep, sel_pair, sel_time), watch["chat"],
                         watch["msg"], html_on=True)
                st = rep["setup"]
                flipped = watch.get("last_sig") not in (None, st["signal"])
                if (st["signal"] in ("CALL", "PUT")
                        and (st.get("setup") or flipped)
                        and time.time() - watch["alert"] >= WATCH_ALERT_GAP):
                    watch["alert"] = time.time()
                    watch["last_sig"] = st["signal"]
                    ohlc = fetch_ohlc(sel_pair, sel_time)
                    closes_w = rep.get("closes") or [r["c"] for r in (ohlc or [])]
                    m = tech_metrics(closes_w, ohlc, st)
                    send_card(watch["chat"], st, m, gate_score(st), closes_w)
                    track_add(sel_pair, sel_time, st["signal"], m.get("price"),
                              globals().get("sel_model", "engine"),
                              st.get("regime"), features=_feat_last(closes_w))
                    print(f"Watch alert {st['signal']} {sel_pair}")

        if autoscan["on"] and autoscan["chat"] and time.time() - autoscan["t"] >= AUTOSCAN_GAP:
            autoscan["t"] = time.time()
            autoscan_tick()

        if auto_on and owner_id and time.time() - auto_tick >= 60:
            auto_tick = time.time()
            if pocket_payout is not None and pocket_payout.live_closes(sel_pair, sel_time) is None:
                warm_feed(sel_pair, sel_time)
            closes, note = get_series(sel_pair, sel_time, fetch=True)
            if note or not closes:
                continue
            res = analyze(closes, sel_model, pair=sel_pair, period=sel_time)
            if res["signal"] in ("CALL", "PUT") and res.get("setup"):
                key = f"{sel_pair}#{sel_time}#{sel_model}"
                if time.time() - last_state.get(key, 0) >= COOLDOWN_SEC:
                    last_state[key] = time.time()
                    ohlc = fetch_ohlc(sel_pair, sel_time)
                    m = tech_metrics(closes, ohlc, res)
                    send_card(owner_id, res, m, gate_score(res), closes)
                    track_add(sel_pair, sel_time, res["signal"], m.get("price"),
                              sel_model, res.get("regime"),
                              features=_feat_last(closes))
                    print(f"Auto {res['signal']} {sel_pair} {tf_label(sel_time)}")

        if time.time() - acct_tick >= ACCT_SYNC_GAP:
            acct_tick = time.time()
            if acct_busy.acquire(blocking=False):
                def _sync_worker():
                    try:
                        sync_account_trades()
                    except Exception as e:
                        print("account sync failed:", e)
                    finally:
                        acct_busy.release()
                threading.Thread(target=_sync_worker, daemon=True).start()

        if time.time() - settle_tick >= 15:
            settle_tick = time.time()
            if settle_busy.acquire(blocking=False):
                def _settle_worker():
                    try:
                        track_settle()
                    except Exception:
                        pass
                    finally:
                        settle_busy.release()
                threading.Thread(target=_settle_worker, daemon=True).start()

        time.sleep(1)
