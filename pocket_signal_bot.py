"""
Spike-style Pocket Option SIGNAL bot - Market -> Pair -> Time -> Signal.

Flow:  /start  ->  Start  ->  Select a market (Stock/OTC)
       ->  Select a currency pair  ->  Select trading time
       ->  Signal Generation...  ->  BUY/SELL or NO TRADE card

One signal engine: a transparent rule vote confirmed by a single classifier
scored with rolling walk-forward validation. Data is the live tick stream via
SSID (pocket_payout.py); the last bar is the CURRENT price. The decision rule
is replayed over the pair's own history and its out-of-sample hit-rate is
compared to the payout break-even. A BUY/SELL card is shown ONLY when that edge
clears break-even; otherwise the bot says NO TRADE.

Display-only by default. The card's "Accept signal" button can place a real
order, but ONLY when the bot is started with ALLOW_TRADING=1 and an SSID is
saved (see pocket_payout.py); otherwise it just prints a disabled notice.

Run: pip install pandas numpy requests
     set TELEGRAM_TOKEN (BotFather token) and TELEGRAM_CHAT_ID env vars
     python pocket_signal_bot.py
     (add ALLOW_TRADING=1 to enable the Accept-order button)
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
STD_MULT = 2.0
COOLDOWN_SEC = 60
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "pocket_settings.json")

RSI_PERIOD = 14
# Target is exactly ONE bar of the selected timeframe, matching a binary
# option bought now that expires `sel_time` seconds later.
HORIZON = 1
SQUEEZE_PCT = 40

# ---- honest edge gate (tunable via pocket_settings.json) ----
EDGE_MARGIN = 2.0      # measured hit-rate must beat break-even by this many %
MIN_BACKTEST_N = 100   # minimum resolved setups before the edge is trusted
MIN_PAYOUT = 60.0      # assets paying less than this are never signalled
TRADE_AMOUNT = 1.0     # stake for an accepted signal (only used with ALLOW_TRADING=1)

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

# Single engine: one transparent rule vote, gated by that same rule's measured
# out-of-sample hit-rate vs. the payout break-even. The previous six "models"
# were cosmetic weight variants with no measured edge, so they were removed.
ENGINE_WEIGHTS = (2, 2, 1, 1)
MODEL_PARAMS = {
    "engine": {"w": ENGINE_WEIGHTS, "label": "Rule vote + validated edge gate"},
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

# ===== settings persistence =====
def load_settings():
    d = {"last_pair": "EURUSD-OTC", "last_time": 60, "auto": False,
         "model": "engine", "market": "otc"}
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE) as f:
                d.update(json.load(f))
        except Exception:
            pass
    if d.get("model") not in MODEL_PARAMS:
        d["model"] = "engine"
    if d.get("market") not in ("stock", "otc"):
        d["market"] = "otc"
    return d

def save_settings(s):
    try:
        cur = {}
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE) as f:
                    cur = json.load(f)
            except Exception:
                cur = {}
        cur.update(s)
        with open(SETTINGS_FILE, "w") as f:
            json.dump(cur, f)
    except Exception as e:
        print("Settings save failed:", e)

# ===== strategy =====
def _indicator_stack(closes, ohlc=None):
    """Precompute every series the decision needs.

    When real OHLC is supplied, True Range / Wilder ATR / +DI / -DI / ADX and
    the stochastic/Williams ranges use actual high/low (the same formulas as
    backtrader's atr.py and directionalmove.py). Otherwise it falls back to a
    close-only approximation.
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
    std = s.rolling(PERIOD).std()
    width = (2.0 * STD_MULT) * std
    wpc = (width / sma * 100).rolling(50).rank(pct=True) * 100

    # True Range + Wilder ATR (backtrader TrueRange / ATR)
    prev_close = s.shift(1)
    tr = pd.concat([(high - low),
                    (high - prev_close).abs(),
                    (prev_close - low).abs()], axis=1).max(axis=1)
    alpha = 1.0 / 14.0
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    # Directional Movement (backtrader DirectionalMove): +DI, -DI, ADX
    up = high.diff()
    dn = -low.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=s.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=s.index)
    natr = atr.replace(0, 1e-12)
    plus_di = 100.0 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / natr
    minus_di = 100.0 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / natr
    dx = ((plus_di - minus_di).abs()
          / (plus_di + minus_di).replace(0, 1e-12)) * 100.0
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    # Stochastic / Williams %R / pivots use the true 14-bar high-low range
    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    rng14 = (high_14 - low_14).replace(0, 1e-12)
    stoch_k = 100 * (s - low_14) / rng14
    stoch_d = stoch_k.rolling(3).mean()
    will_r = -100 * (high_14 - s) / rng14
    pivot = (high_14 + low_14 + s) / 3
    r1 = 2 * pivot - low_14
    s1 = 2 * pivot - high_14
    return {"s": s, "ema9": ema9, "ema21": ema21, "hist": hist, "rsi": rsi,
            "mom3": mom3, "wpc": wpc, "sma": sma, "std": std,
            "adx": adx, "plus_di": plus_di, "minus_di": minus_di, "atr": atr,
            "stoch_k": stoch_k, "stoch_d": stoch_d,
            "will_r": will_r, "pivot": pivot, "r1": r1, "s1": s1}


def _votes(stk, i, w):
    """Weighted majority vote of predictors. Always CALL or PUT."""
    e = 1 if float(stk["ema9"].iloc[i]) >= float(stk["ema21"].iloc[i]) else -1
    m = 1 if float(stk["hist"].iloc[i]) >= 0 else -1
    r = stk["rsi"].iloc[i]
    r = 1 if (50.0 if pd.isna(r) else float(r)) >= 50 else -1
    o = 1 if float(stk["mom3"].iloc[i]) >= 0 else -1
    adx_val = float(stk["adx"].iloc[i]) if not pd.isna(stk["adx"].iloc[i]) else 0
    w_adx = 1.0 if adx_val >= 20 else 0.5
    stoch_k = float(stk["stoch_k"].iloc[i]) if not pd.isna(stk["stoch_k"].iloc[i]) else 50
    w_stoch = 1.0 if 20 <= stoch_k <= 80 else 0.7
    will_r = float(stk["will_r"].iloc[i]) if not pd.isna(stk["will_r"].iloc[i]) else -50
    w_will = 1.0 if will_r < -20 else 0.8
    sc = (w[0] * w_adx * e + w[1] * w_stoch * m + w[2] * w_will * r + w[3] * e) / 4.0
    if sc > 0:
        return "CALL"
    if sc < 0:
        return "PUT"
    return "CALL" if e > 0 else "PUT"


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
    if i < PERIOD + RSI_PERIOD:
        return None
    if pd.isna(stk["ema21"].iloc[i]) or pd.isna(stk["mom3"].iloc[i]):
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
_EDGE_REPORT = {"mtime": None, "data": {}}


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
    try:
        with open(EDGE_REPORT_FILE) as f:
            d = json.load(f)
        data = d.get("pairs", d) if isinstance(d, dict) else {}
    except Exception:
        data = {}
    _EDGE_REPORT["mtime"] = m
    _EDGE_REPORT["data"] = data or {}
    return _EDGE_REPORT["data"]


def edge_from_report(pair, period):
    """Validated stats for this pair+timeframe, or None."""
    data = load_edge_report()
    if not data:
        return None
    return data.get(f"{pair}#{int(period)}") or data.get(pair)


def _fold_edge(stk, params, folds=4):
    """Replay the rule causally and score it per fold.

    Returns the *most recent* fold's stats as the gate input when that fold has
    enough samples (so a decayed edge is caught), otherwise the whole sample.
    `accs` lists every fold's hit-rate for a stability check."""
    s = stk["s"]
    start = PERIOD + RSI_PERIOD + 4
    wins = []
    for i in range(start, len(s) - HORIZON):
        sig = decide_at(stk, i, params)
        if not sig:
            continue
        c = float(s.iloc[i])
        nxt = float(s.iloc[i + HORIZON])
        wins.append((nxt > c) if sig == "CALL" else (nxt < c))
    total = len(wins)
    if not total:
        return {"wins": 0, "total": 0, "acc": 0.0, "accs": []}
    k = total // folds if folds >= 2 else 0
    accs = []
    if k >= 10:
        for f in range(folds):
            lo = f * k
            hi = (f + 1) * k if f < folds - 1 else total
            seg = wins[lo:hi]
            accs.append(round(sum(seg) / len(seg) * 100.0, 1) if seg else 0.0)
        last = wins[-k:]
        if len(last) >= 30:
            return {"wins": sum(last), "total": len(last),
                    "acc": sum(last) / len(last) * 100.0, "accs": accs}
    return {"wins": sum(wins), "total": total,
            "acc": sum(wins) / total * 100.0, "accs": accs}


def analyze(closes, model=None, news_symbol=None, pair=None, period=None):
    model = model or CURRENT_MODEL
    params = MODEL_PARAMS.get(model, MODEL_PARAMS["engine"])
    pair = pair or sel_pair_default()
    period = int(period or 60)
    out = {"signal": "WAIT", "note": "", "rsi": None,
           "up": None, "lo": None, "wide": False, "depth": 0.0, "model": model,
           "setup": False, "regime": "normal", "horizon": HORIZON,
           "full_acc": 0.0, "full_wins": 0, "full_total": 0, "fold_accs": [],
           "confidence": 0.0, "tradeable": False, "payout": None,
           "break_even": None, "wilson": 0.0, "gate_note": "",
           "validated": False, "validated_stats": None}
    need = PERIOD + RSI_PERIOD + 6
    if not closes or len(closes) < need:
        out["note"] = "Not enough candle history for this timeframe. Pick a shorter time."
        return out
    ohlc_data = fetch_ohlc(pair, period) if pocket_payout else None
    stk = _indicator_stack(closes, ohlc_data)
    s = stk["s"]

    fold = _fold_edge(stk, params)
    f_wins, f_total = fold["wins"], fold["total"]
    out["full_wins"], out["full_total"] = f_wins, f_total
    out["full_acc"] = fold["acc"]
    out["fold_accs"] = fold["accs"]

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
    out["wilson"] = _wilson_lower(f_wins, f_total)

    if be is None:
        out["gate_note"] = "Payout unavailable - cannot judge the edge."
    elif (out["payout"] or 0) < MIN_PAYOUT:
        out["gate_note"] = (f"Payout {out['payout']:.0f}% is below the "
                            f"{MIN_PAYOUT:.0f}% minimum.")
    elif rep is not None:
        # a validated backtest report exists - trust it over the live replay
        out["validated"] = True
        out["validated_stats"] = rep
        r_bets = int(rep.get("bets", 0))
        r_exp = float(rep.get("expectancy", 0.0))
        r_wr = float(rep.get("win_rate", 0.0))
        r_wil = float(rep.get("wilson", 0.0))
        out["full_acc"] = r_wr
        out["full_wins"] = int(rep.get("wins", 0))
        out["full_total"] = r_bets
        out["wilson"] = r_wil
        folds = rep.get("folds") or []
        last = folds[-1] if folds else None
        lw = float(last.get("wilson", 0.0)) if last else 0.0
        lexp = float(last.get("expectancy", 0.0)) if last else 0.0
        lbets = int(last.get("bets", 0)) if last else 0
        if r_bets < MIN_BACKTEST_N:
            out["gate_note"] = (f"Validated on only {r_bets} bets - need "
                                f"{MIN_BACKTEST_N}.")
        elif r_exp <= 0:
            out["gate_note"] = (f"Validated expectancy {r_exp:+.3f}/bet is not "
                                f"positive.")
        elif r_wil < be:
            out["gate_note"] = (f"Validated win {r_wr:.0f}% (Wilson {r_wil:.0f}%) "
                                f"below break-even {be:.0f}%.")
        elif last and lbets >= 20 and (lexp <= 0 or lw < be):
            out["gate_note"] = (f"Recent fold decayed: {last.get('win_rate', 0):.0f}% "
                                f"win, {lexp:+.3f}/bet - edge not stable.")
        else:
            out["tradeable"] = True
            out["gate_note"] = (f"Validated {r_wr:.0f}% win, {r_exp:+.3f}/bet, "
                                f"SQN {rep.get('sqn', 0)} over {r_bets} bets.")
    elif f_total < MIN_BACKTEST_N:
        out["gate_note"] = (f"Only {f_total} resolved setups - need "
                            f"{MIN_BACKTEST_N} to trust the edge.")
    elif out["full_acc"] < be + EDGE_MARGIN:
        out["gate_note"] = (f"Edge {out['full_acc']:.0f}% is not above break-even "
                            f"{be:.0f}% (need +{EDGE_MARGIN:.0f}).")
    elif out["wilson"] < be:
        out["gate_note"] = (f"Edge {out['full_acc']:.0f}% looks fine but its "
                            f"95% floor ({out['wilson']:.0f}%) is below break-even "
                            f"{be:.0f}% - too thin to trust.")
    elif out["fold_accs"] and out["fold_accs"][-1] < be:
        out["gate_note"] = (f"Most recent fold {out['fold_accs'][-1]:.0f}% is below "
                            f"break-even {be:.0f}% - edge not stable.")
    else:
        out["tradeable"] = True
        out["gate_note"] = (f"Edge {out['full_acc']:.0f}% (floor {out['wilson']:.0f}%) "
                            f"clears break-even {be:.0f}% over {f_total} setups.")

    i = len(s) - 1
    out["rsi"] = float(stk["rsi"].iloc[i]) if not pd.isna(stk["rsi"].iloc[i]) else None
    sma, std = stk["sma"], stk["std"]
    up = sma + STD_MULT * std
    lo = sma - STD_MULT * std
    out["up"] = float(up.iloc[i]) if not pd.isna(up.iloc[i]) else None
    out["lo"] = float(lo.iloc[i]) if not pd.isna(lo.iloc[i]) else None
    out["regime"] = regime_at(stk, i)
    out["wide"] = out["regime"] in ("wide",)
    out["setup"] = out["regime"] == "squeeze"
    out["note"] = ("Quiet market - the regime with the best measured edge."
                   if out["setup"] else "Normal volatility - accuracy near a coin flip.")

    out["confidence"] = out["full_acc"]
    vote = decide_at(stk, i, params) or "WAIT"
    out["signal"] = vote if out["tradeable"] else "NO TRADE"
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
        with open(ML_DATA_FILE, "w") as f:
            json.dump(d, f)
    except Exception as e:
        print("ML data save failed:", e)


def _ml_save():
    try:
        import pickle
        with open(ML_MODEL_FILE, "wb") as f:
            pickle.dump({"models": _ml_models, "meta": _ml_meta}, f)
    except Exception as e:
        print("ML model save failed:", e)


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
        print(f"ML feedback {pair} {direction} -> {'up' if label else 'down'} "
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

def track_load():
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
    try:
        with open(TRACK_FILE, "w") as f:
            json.dump(d, f)
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
                raw = pocket_payout._fetch_candles(api, sym, tfi, 60) or []
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


# ===== real account sync (linked via pocket_ssid.json, read-only) =====
ACCT_SYNC_GAP = 45.0


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
                raw = pocket_payout._fetch_candles(api, sym, period, 144) or []
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
    """16-feature snapshot at (or just after) a past timestamp, from real candles."""
    try:
        closes, cut = [], None
        for row in raw:
            try:
                closes.append(float(row["close"]))
            except Exception:
                continue
            ts = int(row.get("time") or 0)
            if cut is None and ts and ts >= open_ts:
                cut = len(closes) - 1
        if cut is None:
            cut = len(closes) - 1
        if cut < 70:
            return None
        return _feat_last(closes[:cut + 1])
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
    track = track_load()
    new_seen = []
    changed = False
    new_ts = {}
    for t in trades:
        tid = str(t["id"])
        if tid in seen:
            continue
        new_seen.append(tid)
        if not t.get("refund") and t.get("entry"):
            a = t["asset"]
            new_ts[a] = min(new_ts.get(a, 10 ** 12), t["open_ts"])
    raw_cache = {a: _acct_raw(a, ts0) for a, ts0 in new_ts.items()}
    for t in trades:
        tid = str(t["id"])
        if tid not in new_seen:
            continue
        changed = True
        if t.get("refund") or not t.get("entry"):
            continue
        went_up = t["exit"] > t["entry"]
        if t["win"]:
            direction = "CALL" if went_up else "PUT"
        else:
            direction = "PUT" if went_up else "CALL"
        feats = _feat_from_candles(raw_cache.get(t["asset"]) or [], t["open_ts"])
        _ml_learn(t["asset"], feats, direction, bool(t["win"]))
        track["settled"].append({
            "pair": t["asset"], "tf": t["tf"], "dir": direction,
            "entry": t["entry"], "exit": t["exit"], "win": bool(t["win"]),
            "model": None, "regime": "account", "source": "account",
            "closed": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        tag = "demo" if t.get("demo") else "real"
        print(f"Account {t['asset']} {direction} -> "
              f"{'WIN' if t['win'] else 'LOSS'} ({tag}, profit {t['profit']})")
    if changed:
        seen.update(new_seen)
        d["seen_deals"] = sorted(seen)[-500:]
        _ml_data_save(d)
        track["settled"] = track["settled"][-200:]
        track_save(track)

def sel_pair_default():
    return "EURUSD-OTC"

# ===== real-time market read =====
def market_report(pair, period_sec, fetch=True):
    if pocket_payout is None:
        return None
    ohlc = None
    try:
        ohlc = pocket_payout.live_ohlcv(pair, period_sec) if fetch else None
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

def get_series(pair, period_sec, fetch=True):
    global DATA_SRC
    if pocket_payout is not None:
        try:
            closes = pocket_payout.live_closes(pair, period_sec) if fetch else None
            if closes and len(closes) >= PERIOD + RSI_PERIOD + 5:
                DATA_SRC = "live stream"
                return closes, ""
            closes = (pocket_payout.get_closes(pair, period_sec, 220) if fetch
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

def fetch_ohlc(pair, period_sec):
    if pocket_payout is None:
        return None
    try:
        ohlc = pocket_payout.live_ohlcv(pair, period_sec)
        if not ohlc:
            ohlc = pocket_payout.get_ohlcv(pair, period_sec, 220)
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
    return [(p, pay) for p, pay in pairs if not p.endswith("-OTC")]

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

def kb_signal():
    return [[{"text": "\u2705 Accept signal", "callback_data": "sig:accept"}],
            [{"text": "\U0001f504 Another signal", "callback_data": "sig:again"},
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
    out = {"vol": "Normal", "vol_pct": 50.0, "rsi_label": "Neutral", "macd": "Neutral",
           "ma": "Sideways", "sentiment": "Sideways pressure", "price": None,
           "conditions": "Favorable"}
    if not closes or len(closes) < 30:
        return out
    s = pd.Series([float(x) for x in closes])
    i = len(s) - 1

    sma = s.rolling(PERIOD).mean()
    std = s.rolling(PERIOD).std()
    up = sma + STD_MULT * std
    lo = sma - STD_MULT * std
    width = up - lo
    hist = width.iloc[max(0, i - 100):i + 1].dropna()
    if len(hist) and not pd.isna(width.iloc[i]):
        out["vol_pct"] = float((hist < width.iloc[i]).mean() * 100)
    out["vol"] = ("High" if out["vol_pct"] >= 75 else
                  "Increased" if out["vol_pct"] >= 50 else
                  "Normal" if out["vol_pct"] >= 25 else "Low")

    r = res.get("rsi")
    if r is not None:
        out["rsi_label"] = ("Oversold" if r <= 30 else "Low" if r < 40 else
                            "Neutral" if r < 60 else "High" if r < 70 else "Overbought")

    ema12 = s.ewm(span=12, adjust=False).mean()
    ema26 = s.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sigl = macd.ewm(span=9, adjust=False).mean()
    histm = macd - sigl
    h_now, h_prev = float(histm.iloc[i]), float(histm.iloc[i - 1])
    if h_prev <= 0 < h_now:
        out["macd"] = "Bullish crossover"
    elif h_prev >= 0 > h_now:
        out["macd"] = "Bearish crossover"
    elif h_now > 0:
        out["macd"] = "Buying pressure"
    elif h_now < 0:
        out["macd"] = "Selling pressure"

    ema9 = s.ewm(span=9, adjust=False).mean()
    ema21 = s.ewm(span=21, adjust=False).mean()
    slope = (float(ema21.iloc[i]) - float(ema21.iloc[max(0, i - 5)])) / max(abs(float(ema21.iloc[i])), 1e-12) * 100
    out["ma"] = ("Upward trend" if slope > 0.01 else
                 "Downward trend" if slope < -0.01 else "Sideways")
    if float(ema9.iloc[i]) > float(ema21.iloc[i]):
        out["sentiment"] = "Upward pressure"
    elif float(ema9.iloc[i]) < float(ema21.iloc[i]):
        out["sentiment"] = "Downward pressure"

    out["price"] = float(s.iloc[i])

    out["conditions"] = ("Optimal" if (res.get("wide") and out["vol_pct"] >= 60) else
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
        f"\u2022 RSI: <b>{m['rsi_label']}</b>\n"
        f"\u2022 MACD: <b>{m['macd']}</b>\n"
        f"\u2022 Moving Average: <b>{m['ma']}</b>\n\n"
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
    try:
        pairs = market_pairs(market)
    except Exception:
        pairs = []
    for pair, pay in pairs[:max_pairs]:
        if pay and pay < MIN_PAYOUT:
            continue
        try:
            closes, note = get_series(pair, period_sec, fetch=True)
            if note or not closes:
                continue
            res = analyze(closes, pair=pair, period=period_sec)
            be = res.get("break_even")
            edge = res.get("full_acc", 0.0)
            out.append({
                "pair": pair, "signal": res.get("signal"), "edge": edge,
                "wilson": res.get("wilson", 0.0), "break_even": be,
                "payout": res.get("payout"), "total": res.get("full_total", 0),
                "tradeable": bool(res.get("tradeable")),
                "validated": bool(res.get("validated")),
                "note": res.get("gate_note", ""),
                "margin": (edge - be) if be else -99.0,
                "res": res, "closes": closes,
            })
        except Exception:
            continue
    out.sort(key=lambda x: (x["tradeable"], x["margin"]), reverse=True)
    return out


def scan_text(results, period_sec, market):
    tf = tf_label(period_sec)
    head = f"\U0001f50e <b>Scan {market.upper()}</b> \u00b7 {tf}\n"
    if not results:
        return head + "\nNo pairs with usable live data right now."
    lines = [head,
             f"{len(results)} pairs checked \u2014 none clears break-even.", ""]
    for r in results[:5]:
        be = r.get("break_even") or 0.0
        pay = r.get("payout") or 0.0
        tag = "validated" if r.get("validated") else "replay"
        lines.append(f"\u2022 {html.escape(fmt_pair(r['pair']))}: "
                     f"<b>{r['edge']:.0f}%</b> vs {be:.0f}% "
                     f"(payout {pay:.0f}%, {tag})")
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
            mid = send_photo(path, cap, chat_id=chat, keyboard=kb_signal(), html_on=True)
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
               chat_id=chat, keyboard=kb_signal(), html_on=True)
    remember_signal(mid, res, pair, sel_time, m.get("price"))
    return mid

GEN_STEPS = ["Volatility scan", "RSI Analysis", "MACD Status", "Moving Average",
             "Signal Accuracy"]
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
    "/stop - Stop all signals\n"
    "Flow: Start -> market (Stock/OTC) -> pair -> time -> signal\n"
    "Tap Find a trade to scan all pairs and get pinged only when one clears "
    "break-even.\n"
    "Accept signal places a real order ONLY if the bot was started with "
    "ALLOW_TRADING=1 (and your SSID is saved). Otherwise it is display-only."
)

if __name__ == "__main__":
    if not TELEGRAM_TOKEN:
        print("TELEGRAM_TOKEN is not set. Set the env var to your BotFather token,")
        print('e.g. in PowerShell:  $env:TELEGRAM_TOKEN="<new token>"  then run again.')
        raise SystemExit(1)
    print("Bot starting (SPIKE UI, live feed, on-demand signals)...")
    S = load_settings()
    sel_model = str(S.get("model") or "engine")
    sel_market = str(S.get("market") or "otc")
    sel_pair = str(S.get("last_pair") or "EURUSD-OTC")
    sel_time = int(S.get("last_time") or 60)
    auto_on = False
    CURRENT_MODEL = sel_model
    EDGE_MARGIN = float(S.get("edge_margin", EDGE_MARGIN))
    MIN_BACKTEST_N = max(100, int(S.get("min_backtest_n", MIN_BACKTEST_N)))
    MIN_PAYOUT = float(S.get("min_payout", MIN_PAYOUT))
    TRADE_AMOUNT = max(0.01, float(S.get("trade_amount", TRADE_AMOUNT)))
    update_offset = None
    owner_id = None if str(CHAT_ID).startswith("PASTE") else str(CHAT_ID)
    last_state = {}
    auto_tick = 0.0
    acct_tick = 0.0
    last_page = 0
    warm_feed(sel_pair, sel_time, wait=20.0)
    _ml_load()
    ml_train_async(sel_pair, sel_time)

    WATCH_REFRESH = 15
    WATCH_ALERT_GAP = 120
    watch = {"on": False, "chat": None, "msg": None, "t": 0.0, "alert": 0.0,
             "last_sig": None}

    AUTOSCAN_GAP = 120
    AUTOSCAN_COOLDOWN = 900
    AUTOSCAN_MAX_PAIRS = 24
    scan_state = {"running": False, "alerted": {}}
    autoscan = {"on": False, "chat": None, "t": 0.0}

    def _emit_signal(chat, pair, period_sec, res, closes):
        m = tech_metrics(closes, fetch_ohlc(pair, period_sec), res)
        mid = send(autoscan_card_text(res, m, pair, tf_label(period_sec)), chat_id=chat,
                   keyboard=kb_signal(), html_on=True)
        remember_signal(mid, res, pair, period_sec, m.get("price"))
        track_add(pair, period_sec, res["signal"], m.get("price"), sel_model,
                  res.get("regime"), features=_feat_last(closes))

    def run_scan(chat):
        if scan_state["running"]:
            send("A scan is already running \u2014 hang on a moment.", chat_id=chat)
            return
        scan_state["running"] = True
        mid = send(f"\U0001f50e Scanning {sel_market.upper()} pairs for a real edge\u2026",
                   chat_id=chat, html_on=True)

        def worker():
            try:
                results = scan_pairs(sel_time, sel_market, AUTOSCAN_MAX_PAIRS)
                best = next((r for r in results if r["tradeable"]), None)
                if best:
                    res, closes = best["res"], best["closes"]
                    m = tech_metrics(closes, fetch_ohlc(best["pair"], sel_time), res)
                    cap = autoscan_card_text(res, m, best["pair"], tf_label(sel_time))
                    txt = (f"\u2728 <b>Edge found</b> \u2014 cleared break-even "
                           f"({best['edge']:.0f}% vs {best['break_even']:.0f}%).\n\n{cap}")
                    if mid:
                        edit(txt, chat, mid, html_on=True)
                        api_call("editMessageReplyMarkup", {"chat_id": chat,
                                 "message_id": mid,
                                 "reply_markup": {"inline_keyboard": kb_signal()}})
                    else:
                        mid = send(txt, chat_id=chat, keyboard=kb_signal(), html_on=True)
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
                scan_state["running"] = False

        threading.Thread(target=worker, daemon=True).start()

    def accept_signal(chat, ctx, mid):
        if not ctx:
            send("This signal has expired \u2014 ask for a fresh one.", chat_id=chat)
            return
        if pocket_payout is None:
            send("Live trading is unavailable on this host (library not loaded).",
                 chat_id=chat)
            return
        if os.getenv("ALLOW_TRADING") != "1":
            send("\U0001f512 <b>Accept is disabled.</b> Restart the bot with "
                 "ALLOW_TRADING=1 to allow real orders.\n\nSignal: "
                 f"{ctx['signal']} {html.escape(fmt_pair(ctx['pair']))} \u00b7 "
                 f"{tf_label(ctx['period'])}", chat_id=chat, html_on=True)
            return
        amount = TRADE_AMOUNT
        ok, msg = pocket_payout.place_trade(ctx["pair"], ctx["signal"], amount,
                                            ctx["period"])
        if ok:
            PENDING_SIGNALS.pop(mid, None)
            bal = None
            try:
                bal = pocket_payout.account_balance()
            except Exception:
                pass
            extra = f"\nBalance: {bal:.2f}" if bal is not None else ""
            send(f"\u2705 <b>Order placed</b>\n{ctx['signal']} "
                 f"{html.escape(fmt_pair(ctx['pair']))} \u00b7 {tf_label(ctx['period'])} "
                 f"\u00b7 stake {amount:g}\nTicket: {html.escape(str(msg))}{extra}",
                 chat_id=chat, html_on=True)
        else:
            send(f"\u274c <b>Order failed</b>: {html.escape(str(msg))}",
                 chat_id=chat, html_on=True)

    def autoscan_tick():
        if scan_state["running"]:
            return
        scan_state["running"] = True
        chat = autoscan["chat"]

        def worker():
            try:
                results = scan_pairs(sel_time, sel_market, AUTOSCAN_MAX_PAIRS)
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
                scan_state["running"] = False

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
                       "min_payout": MIN_PAYOUT, "trade_amount": TRADE_AMOUNT})

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
            text = f"{screen_head('Pair', 2)}\nNo {sel_market.upper()} currency pairs are online right now. Try the other market."
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
            (f"{m['rsi_label']} ({res['rsi']:.1f})" if res.get("rsi") is not None
             else m["rsi_label"]),
            m["macd"],
            m["ma"],
            f"{res.get('full_acc', 0):.0f}% ({res.get('full_total', 0)} past signals)",
        ]
        for idx, val in enumerate(reveal_vals):
            values.append(val)
            reveal(GEN_PCT[idx])
        time.sleep(0.4)
        ok = send_result(res, chat, m, conf, closes)
        if ok and res.get("signal") in ("CALL", "PUT"):
            track_add(sel_pair, sel_time, res["signal"], m.get("price"),
                      sel_model, res.get("regime"),
                      features=_feat_last(closes))
            ml_train_async(sel_pair, sel_time)
        return ok

    _processed_cbs = set()

    def handle_update(upd):
        global owner_id, sel_model, sel_market, sel_pair, sel_time, auto_on, last_page
        global TRADE_AMOUNT
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
            elif data == "sig:accept":
                mid = cb.get("message", {}).get("message_id")
                accept_signal(chat, PENDING_SIGNALS.get(mid), mid)
            elif data == "scan:run":
                run_scan(chat)
            elif data == "scan:toggle":
                autoscan["on"] = not autoscan["on"]
                autoscan["chat"] = chat
                autoscan["t"] = 0.0
                if autoscan["on"]:
                    send(f"\U0001f501 <b>Auto-scan ON</b> \u2014 {sel_market.upper()}, "
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
        elif cmd in ("/auto", "auto"):
            autoscan["on"] = not autoscan["on"]
            autoscan["chat"] = chat
            autoscan["t"] = 0.0
            send(f"\U0001f501 Auto-scan {'ON' if autoscan['on'] else 'OFF'}.",
                 chat_id=chat)
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
            except Exception:
                import traceback
                traceback.print_exc()

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
            try:
                sync_account_trades()
            except Exception as e:
                print("account sync failed:", e)

        try:
            track_settle()
        except Exception:
            pass

        time.sleep(1)
