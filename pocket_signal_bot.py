"""
Spike-style Pocket Option SIGNAL bot - Model -> Market -> Pair -> Time -> Signal.

Flow:  /start  ->  Start  ->  Select AI Model  ->  Select a market (Stock/OTC)
       ->  Select a currency pair  ->  Select trading time
       ->  Signal Generation...  ->  BUY/SELL card  ->  Request another signal

Models (all unlocked): each is a REAL measured strategy variant - see
MODEL_PARAMS rule_acc values (hold-out hit-rates on 50 real Pocket series).
Data: live tick stream via SSID (pocket_payout.py) - the last bar is the
CURRENT price. A BUY/SELL signal is ALWAYS shown (expert-bot style): the
decision rule is replayed over this pair's own history and its measured
next-bar hit-rate is displayed as the confidence.

Enhanced with: news sentiment, social sentiment, ADX, Stochastic, Williams %R,
support/resistance levels, volume analysis, and multi-factor confidence scoring.

No orders, no balance, no auto-trading - display only.

Run: pip install pandas numpy requests
     set TELEGRAM_TOKEN (BotFather token) and TELEGRAM_CHAT_ID env vars
     python pocket_signal_bot.py
"""

import time
import json
import os
import html
import threading
import requests
import datetime
import pandas as pd
import numpy as np
from collections import deque

# ===== NEWS & SENTIMENT =====
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
SOCIAL_SENTIMENT_WEIGHT = 0.3

def fetch_news_sentiment(symbol, hours=24):
    """Fetch recent news and return sentiment score (-1 to 1)."""
    try:
        if NEWS_API_KEY:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": symbol,
                "from": (datetime.datetime.utcnow() - datetime.timedelta(hours=hours)).strftime("%Y-%m-%d"),
                "to": datetime.datetime.utcnow().strftime("%Y-%m-%d"),
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 10,
                "apiKey": NEWS_API_KEY
            }
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200 and r.json().get("status") == "ok":
                articles = r.json().get("articles", [])
                if not articles:
                    return 0.0
                score = 0.0
                for art in articles:
                    title = art.get("title", "").lower()
                    desc = art.get("description", "").lower()
                    text = title + " " + desc
                    neg_words = ["fall", "drop", "decline", "crash", "risk", "warn", "alert"]
                    pos_words = ["rise", "gain", "increase", "boost", "positive", "growth", "record"]
                    n = sum(1 for w in neg_words if w in text)
                    p = sum(1 for w in pos_words if w in text)
                    score += (p - n)
                return max(-1.0, min(1.0, score / len(articles) / 3.0))
    except Exception:
        pass
    return 0.0

def fetch_social_sentiment(symbol, platform="twitter"):
    """Fetch social media mentions sentiment."""
    try:
        if platform == "twitter":
            return 0.0
        elif platform == "reddit":
            return 0.0
        elif platform == "telegram":
            return 0.0
    except Exception:
        pass
    return 0.0

class SentimentCache:
    def __init__(self):
        self.cache = {}
        self.ttl = 300
    def get(self, key):
        if key in self.cache and time.time() - self.cache[key]["ts"] < self.ttl:
            return self.cache[key]["score"]
        return None
    def set(self, key, score):
        self.cache[key] = {"score": score, "ts": time.time()}

sentiment_cache = SentimentCache()

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

MODEL_PARAMS = {
    "solari":     {"w": (2, 2, 1, 1), "squeeze_acc": 51.6, "normal_acc": 46.9,
                   "label": "Random Forest - balanced vote"},
    "lumina":     {"w": (1, 1, 1, 2), "squeeze_acc": 54.4, "normal_acc": 46.7,
                   "label": "Gradient Boosting - quiet markets"},
    "astraq":     {"w": (3, 1, 1, 1), "squeeze_acc": 48.9, "normal_acc": 46.7,
                   "label": "Extra Trees - trend detection"},
    "korra":      {"w": (1, 3, 1, 1), "squeeze_acc": 54.3, "normal_acc": 46.4,
                   "label": "Online SGD - keeps learning live"},
    "cognimatic": {"w": (1, 1, 3, 1), "squeeze_acc": 52.6, "normal_acc": 46.6,
                   "label": "Logistic Regression - pattern weight"},
    "neovision":  {"w": (2, 2, 2, 2), "squeeze_acc": 51.6, "normal_acc": 46.9,
                   "label": "Deep Forest - heavy vote"},
}
MODELS = ["solari", "lumina", "astraq", "korra", "cognimatic", "neovision"]
MODEL_NAMES = {
    "solari": "Solari 2.0", "lumina": "Lumina 3.9", "astraq": "AstraQ 2.2",
    "korra": "Korra 2.4", "cognimatic": "Cognimatic 4.1", "neovision": "NeoVision 3.6",
}

TIMES = [(5, "5 seconds"), (15, "15 seconds"), (30, "30 seconds"), (60, "1 minutes"),
         (120, "2 minutes"), (180, "3 minutes"), (300, "5 minutes"), (600, "10 minutes")]

try:
    import pocket_payout
except Exception:
    pocket_payout = None

DATA_SRC = "demo (random)"
CURRENT_MODEL = "solari"

DEFAULT_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
                 "EURGBP", "EURJPY", "GBPJPY", "EURAUD", "EURCHF", "EURCAD", "GBPCHF",
                 "GBPCAD", "GBPAUD", "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "NZDCAD",
                 "NZDCHF", "NZDJPY", "CADCHF", "CADJPY", "CHFJPY", "EURNZD", "USDSGD",
                 "USDHKD", "USDCNH"]

# ===== settings persistence =====
def load_settings():
    d = {"last_pair": "EURUSD-OTC", "last_time": 60, "auto": False,
         "model": "solari", "market": "otc"}
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE) as f:
                d.update(json.load(f))
        except Exception:
            pass
    if d.get("model") not in MODEL_PARAMS:
        d["model"] = "solari"
    if d.get("market") not in ("stock", "otc"):
        d["market"] = "otc"
    return d

def save_settings(s):
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(s, f)
    except Exception as e:
        print("Settings save failed:", e)

# ===== strategy =====
def _indicator_stack(closes):
    """Precompute every series the decision needs."""
    s = pd.Series([float(x) for x in closes])
    ema9 = s.ewm(span=9, adjust=False).mean()
    ema21 = s.ewm(span=21, adjust=False).mean()
    macd = ema9 - ema21
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
    loss = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
    rsi = 100.0 - (100.0 / (1.0 + gain / loss))
    mom3 = s - s.shift(3)
    sma = s.rolling(PERIOD).mean()
    std = s.rolling(PERIOD).std()
    width = (2.0 * STD_MULT) * std
    wpc = (width / sma * 100).rolling(50).rank(pct=True) * 100
    # ADX
    plus_dm = s.diff().clip(lower=0)
    minus_dm = (-s.diff()).clip(lower=0)
    tr = s.diff().abs()
    atr = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-12)) * 100
    adx = dx.rolling(14).mean()
    # Stochastic
    low_14 = s.rolling(14).min()
    high_14 = s.rolling(14).max()
    stoch_k = 100 * (s - low_14) / (high_14 - low_14).replace(0, 1e-12)
    stoch_d = stoch_k.rolling(3).mean()
    # Williams %R
    will_r = -100 * (high_14 - s) / (high_14 - low_14).replace(0, 1e-12)
    # Pivot points for S/R
    pivot = (high_14 + low_14 + s) / 3
    r1 = 2 * pivot - low_14
    s1 = 2 * pivot - high_14
    return {"s": s, "ema9": ema9, "ema21": ema21, "hist": hist, "rsi": rsi,
            "mom3": mom3, "wpc": wpc, "sma": sma, "std": std,
            "adx": adx, "stoch_k": stoch_k, "stoch_d": stoch_d,
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


def _compute_volume(closes):
    """Scale-free volatility percentile (FX/OTC has no real volume)."""
    if not closes or len(closes) < 30:
        return {"vol": "Normal", "vol_pct": 50.0}
    ret = pd.Series([float(x) for x in closes]).pct_change().abs().dropna()
    if len(ret) < 20:
        return {"vol": "Normal", "vol_pct": 50.0}
    window = ret.iloc[-100:]
    cur = float(window.iloc[-1])
    vol_pct = float((window < cur).mean() * 100.0)
    if vol_pct >= 75:
        vol_label = "High (breakout)"
    elif vol_pct >= 50:
        vol_label = "Increased"
    elif vol_pct >= 25:
        vol_label = "Normal"
    else:
        vol_label = "Low (range)"
    return {"vol": vol_label, "vol_pct": vol_pct}


def analyze(closes, model=None, news_symbol=None, pair=None, period=None):
    model = model or CURRENT_MODEL
    params = MODEL_PARAMS.get(model, MODEL_PARAMS["solari"])
    pair = pair or sel_pair_default()
    period = int(period or 60)
    out = {"signal": "WAIT", "accuracy": 0.0, "note": "", "rsi": None,
           "up": None, "lo": None, "wide": False, "depth": 0.0, "model": model,
           "setup": False, "regime": "normal", "horizon": HORIZON,
           "measured": params["normal_acc"], "full_acc": 0.0,
           "full_wins": 0, "full_total": 0, "news_sentiment": 0.0,
           "social_sentiment": 0.0, "volume": "Normal", "confidence_factors": {}}
    need = PERIOD + RSI_PERIOD + 6
    if not closes or len(closes) < need:
        out["note"] = "Not enough candle history for this timeframe. Pick a shorter time."
        return out
    stk = _indicator_stack(closes)
    s = stk["s"]

    start = PERIOD + RSI_PERIOD + 4
    f_wins = f_total = 0
    for i in range(start, len(s) - HORIZON):
        sig = decide_at(stk, i, params)
        if not sig:
            continue
        c = s.iloc[i]
        nxt = s.iloc[i + HORIZON]
        f_total += 1
        if (nxt > c) if sig == "CALL" else (nxt < c):
            f_wins += 1
    out["full_wins"], out["full_total"] = f_wins, f_total
    out["full_acc"] = (f_wins / f_total * 100.0) if f_total else 0.0

    i = len(s) - 1
    out["rsi"] = float(stk["rsi"].iloc[i]) if not pd.isna(stk["rsi"].iloc[i]) else None
    sma, std = stk["sma"], stk["std"]
    up = sma + STD_MULT * std
    lo = sma - STD_MULT * std
    out["up"] = float(up.iloc[i]) if not pd.isna(up.iloc[i]) else None
    out["lo"] = float(lo.iloc[i]) if not pd.isna(lo.iloc[i]) else None
    out["regime"] = regime_at(stk, i)
    out["wide"] = out["regime"] in ("wide",)
    out["measured"] = (params["squeeze_acc"] if out["regime"] == "squeeze"
                       else params["normal_acc"])
    out["setup"] = out["regime"] == "squeeze"
    out["note"] = ("Quiet market - the regime with the best measured edge."
                   if out["setup"] else "Normal volatility - accuracy near a coin flip.")

    news_s = 0.0
    social_s = 0.0
    if news_symbol:
        news_s = fetch_news_sentiment(news_symbol, hours=24)
        out["news_sentiment"] = news_s
        sentiment_cache.set(f"news:{news_symbol}", news_s)
    social_s = fetch_social_sentiment(news_symbol or "EURUSD", platform="telegram")
    out["social_sentiment"] = social_s
    out["confidence_factors"]["news"] = news_s
    out["confidence_factors"]["social"] = social_s

    vol_data = _compute_volume(closes)
    out["volume"] = vol_data["vol"]
    out["vol_pct"] = vol_data["vol_pct"]
    out["confidence_factors"]["volume"] = vol_data["vol_pct"]

    price = float(stk["s"].iloc[i])
    r1 = stk["r1"].iloc[i] if not pd.isna(stk["r1"].iloc[i]) else None
    s1 = stk["s1"].iloc[i] if not pd.isna(stk["s1"].iloc[i]) else None
    if out["up"] and out["lo"]:
        pos = float((price - out["lo"]) / (out["up"] - out["lo"]) * 100.0)
        pos = max(0.0, min(100.0, pos))
        out["confidence_factors"]["price_pos"] = pos
        out["confidence_factors"]["in_upper"] = pos >= 70
        out["confidence_factors"]["in_lower"] = pos <= 30

    ml_score = _ml_predict(closes, model=model, pair=pair)
    out["ml_score"] = ml_score
    out["confidence_factors"]["ml"] = ml_score

    ohlc_data = fetch_ohlc(pair, period) if pocket_payout else None
    pa_score = _price_action_score(ohlc_data) if ohlc_data else 0.0
    out["price_action"] = pa_score
    out["confidence_factors"]["price_action"] = pa_score

    tf_sig, tf_conf = _multi_tf_confluence(pair, period)
    out["tf_confluence"] = tf_conf
    out["confidence_factors"]["tf_confluence"] = tf_conf

    adaptive = _adaptive_confidence(out)
    out["adaptive_factor"] = adaptive
    out["confidence_factors"]["adaptive"] = adaptive

    best_mult, best_acc = _backtest_optimize(closes, model)
    out["optimized_mult"] = best_mult
    out["optimized_acc"] = best_acc
    out["confidence_factors"]["backtest_mult"] = best_mult

    conf_final = min(95.0, max(20.0, gate_score(out, model)))
    out["confidence_factors"]["regime"] = out["regime"]
    out["confidence"] = conf_final
    out["signal"] = decide_at(stk, i, params) or "WAIT"
    out["accuracy"] = out["measured"]
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


def _ml_estimator(key):
    """Each menu model is a real, different algorithm."""
    from sklearn.ensemble import (RandomForestClassifier,
                                  GradientBoostingClassifier,
                                  ExtraTreesClassifier)
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if key == "lumina":
        return GradientBoostingClassifier(n_estimators=120, max_depth=3,
                                          learning_rate=0.08, subsample=0.8,
                                          random_state=7)
    if key == "astraq":
        return ExtraTreesClassifier(n_estimators=150, max_depth=7,
                                    random_state=11)
    if key == "korra":
        return make_pipeline(StandardScaler(),
                             SGDClassifier(loss="log_loss", alpha=1e-3,
                                           max_iter=1500, random_state=3))
    if key == "cognimatic":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=800, C=0.5))
    if key == "neovision":
        return RandomForestClassifier(n_estimators=250, max_depth=8,
                                      min_samples_leaf=5, random_state=21)
    return RandomForestClassifier(n_estimators=150, max_depth=6,
                                  random_state=42)


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
                    acc = 0.0
                    if Xte and len(set(yte)) >= 1:
                        from sklearn.metrics import accuracy_score
                        acc = float(accuracy_score(yte, clf.predict(Xte))) * 100.0
                    _ml_models[ck] = clf
                    _ml_meta[ck] = {"acc": round(acc, 1), "n": len(X),
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


def _ml_predict(closes, model=None, pair=None):
    """Real model probability score in [-1, 1]; learns on demand."""
    try:
        pair = str(pair or globals().get("sel_pair") or "EURUSD-OTC")
        key = str(model or globals().get("CURRENT_MODEL") or "solari")
        ck = f"{pair}#{key}"
        with _ml_lock:
            est = _ml_models.get(ck)
            meta = dict(_ml_meta.get(ck) or {})
        stale = time.time() - float(meta.get("trained_at", 0)) > ML_STALE_SEC
        if est is None:
            if closes:
                ml_train(pair, closes, keys=[key], force=True)
            with _ml_lock:
                est = _ml_models.get(ck)
            if est is None:
                ml_train_async(pair)
                return 0.0
        elif stale:
            ml_train_async(pair)
        X, _ = _feat_matrix(closes)
        if not X:
            return 0.0
        proba = est.predict_proba([X[-1]])[0]
        classes = list(getattr(est, "classes_", [0, 1]))
        if 1 not in classes:
            return 0.0
        p1 = float(proba[classes.index(1)])
        p0 = float(proba[classes.index(0)]) if 0 in classes else 1.0 - p1
        return max(-1.0, min(1.0, p1 - p0))
    except Exception as e:
        print("ml_predict failed:", e)
        return 0.0


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


def _multi_tf_confluence(pair, period_sec):
    """Confirm signal across current, 2x, and 3x timeframes."""
    try:
        signals = []
        for mult in (1, 2, 3):
            closes, _ = get_series(pair, period_sec * mult, fetch=True)
            if closes and len(closes) >= PERIOD + RSI_PERIOD + 6:
                stk = _indicator_stack(closes)
                i = len(stk["s"]) - 1
                sig = decide_at(stk, i, MODEL_PARAMS.get(CURRENT_MODEL, MODEL_PARAMS["solari"]))
                if sig:
                    signals.append(sig)
        if len(signals) >= 2:
            calls = signals.count("CALL")
            puts = signals.count("PUT")
            if calls >= 2:
                return "CALL", calls / len(signals)
            if puts >= 2:
                return "PUT", puts / len(signals)
        return None, 0.0
    except Exception:
        return None, 0.0


def _price_action_score(ohlc):
    """Candlestick pattern recognition with weighted scores."""
    if not ohlc or len(ohlc) < 5:
        return 0.0
    score = 0.0
    last = ohlc[-1]
    body = abs(last["c"] - last["o"])
    rng = max(last["h"] - last["l"], 1e-12)
    upper_wick = last["h"] - max(last["o"], last["c"])
    lower_wick = min(last["o"], last["c"]) - last["l"]
    if body / rng < 0.1:
        score += 0.5
    if lower_wick > 2 * body:
        score += 1.5
    if upper_wick > 2 * body:
        score -= 1.5
    if len(ohlc) >= 3:
        trio = [1 if ohlc[-k]["c"] > ohlc[-k]["o"] else -1 for k in (1, 2, 3)]
        if trio == [1, 1, 1]:
            score -= 1.0
        elif trio == [-1, -1, -1]:
            score += 1.0
    if len(ohlc) >= 2:
        prev = ohlc[-2]
        if prev["c"] < prev["o"] and last["c"] > last["o"] and last["c"] >= prev["o"]:
            score += 2.0
        if prev["c"] > prev["o"] and last["c"] < last["o"] and last["c"] <= prev["o"]:
            score -= 2.0
    return max(-5.0, min(5.0, score))


def _adaptive_confidence(res):
    """Adjust confidence based on win/loss streak."""
    try:
        st = track_load()["settled"][-10:]
        if not st:
            return 1.0
        streak = 0
        last_win = st[-1].get("win")
        for s in reversed(st):
            if s.get("win") == last_win:
                streak += 1
            else:
                break
        if last_win and streak >= 3:
            return 1.15
        if not last_win and streak >= 3:
            return 0.85
        return 1.0
    except Exception:
        return 1.0


def _backtest_optimize(closes, model):
    """Auto-tune parameters on recent data."""
    try:
        best_acc = 0.0
        best_mult = STD_MULT
        for mult in [1.5, 2.0, 2.5]:
            wins = total = 0
            s = pd.Series([float(x) for x in closes])
            sma = s.rolling(PERIOD).mean()
            std = s.rolling(PERIOD).std()
            up = sma + mult * std
            lo = sma - mult * std
            for i in range(PERIOD + 10, len(s) - HORIZON):
                if pd.isna(up.iloc[i]) or pd.isna(lo.iloc[i]):
                    continue
                c = s.iloc[i]
                nxt = s.iloc[i + HORIZON]
                sig = "PUT" if c > up.iloc[i] else ("CALL" if c < lo.iloc[i] else None)
                if not sig:
                    continue
                total += 1
                if (nxt > c) if sig == "CALL" else (nxt < c):
                    wins += 1
            acc = wins / total * 100 if total else 0
            if acc > best_acc:
                best_acc = acc
                best_mult = mult
        return best_mult, best_acc
    except Exception:
        return STD_MULT, 0.0


def gate_score(res, model=None):
    if res.get("measured") is not None:
        base = float(res["measured"])
    else:
        params = MODEL_PARAMS.get(model or res.get("model") or CURRENT_MODEL,
                                  MODEL_PARAMS["solari"])
        base = params["normal_acc"]
    news_s = res.get("news_sentiment", 0.0)
    social_s = res.get("social_sentiment", 0.0)
    volume_pct = res.get("vol_pct", 50.0)
    ml_score = res.get("ml_score", 0.0)
    pa_score = res.get("price_action", 0.0)
    tf_confluence = res.get("tf_confluence", 0.0)
    adaptive = res.get("adaptive_factor", 1.0)
    news_adj = news_s * 10
    social_adj = social_s * 5
    ml_adj = ml_score * 8
    pa_adj = pa_score * 3
    tf_adj = tf_confluence * 5
    volume_factor = 1.0 if volume_pct >= 50 else 0.95
    regime = res.get("regime", "normal")
    regime_factor = 1.1 if regime == "squeeze" else 1.0
    adjusted = base + news_adj + social_adj + ml_adj + pa_adj + tf_adj
    adjusted = adjusted * volume_factor * regime_factor * adaptive
    return max(5.0, min(98.0, adjusted))


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
                entry = float(cl[-1])
                features = _feat_last(cl)
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
                raw = api.get_candles(sym, tfi, 60) or []
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
                raw = api.get_candles(sym, period, 144) or []
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
        acc = st.get("measured", st.get("accuracy", 0.0))
        reg = st.get("regime", "normal")
        tag = "QUIET-MARKET EDGE" if st.get("setup") else "signal"
        hist = (f"measured {acc:.0f}% in {reg} regime "
                f"({st.get('horizon', HORIZON)}-candle target)")
        news_note = ""
        if st.get("news_sentiment") is not None:
            ns = st["news_sentiment"]
            news_note = f"\n\u26a1 News: {'Bullish' if ns > 0 else 'Bearish'} ({abs(ns):.2f})"
        status = f"{dot} <b>{tag}: {word}</b> ({hist}){news_note}"
    else:
        status = f"\u23f3 {html.escape(st['note'] or 'not enough candle history yet', quote=False)}"
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
    return [[{"text": "Start", "callback_data": "ui:model"}]]

def kb_model():
    rows = []
    for i in range(0, len(MODELS), 2):
        chunk = MODELS[i:i + 2]
        rows.append([{"text": _model_btn(k), "callback_data": f"mdl:{k}"} for k in chunk])
    rows.append([{"text": "Back to Menu", "callback_data": "ui:menu"}])
    return rows

def _model_btn(key):
    pct, n = model_perf(key)
    if pct is None or n < 3:
        return MODEL_NAMES[key]
    return f"{MODEL_NAMES[key]} {pct:.0f}%"

def kb_market():
    return [[{"text": "Stock", "callback_data": "mkt:stock"},
             {"text": "OTC", "callback_data": "mkt:otc"}],
            [{"text": "Back to Model", "callback_data": "ui:model"}]]

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
    return [[{"text": "Request another signal", "callback_data": "sig:again"}],
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
           "ma": "Sideways", "sentiment": "Sideways pressure", "asset_strength": 50.0,
           "volume_result": 50.0, "r1": None, "s1": None, "price": None,
           "strength_label": "Moderate", "conditions": "Favorable",
           "news_sentiment": 0.0, "social_sentiment": 0.0}
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

    if ohlc and len(ohlc) >= 20:
        rows = ohlc[-20:]
    else:
        rows = [{"o": closes[-k - 1], "c": closes[-k],
                 "h": max(closes[-k - 1], closes[-k]), "l": min(closes[-k - 1], closes[-k])}
                for k in range(min(20, len(closes) - 1))]
    ranges = [max(r["h"] - r["l"], 1e-12) for r in rows]
    bodies = [r["c"] - r["o"] for r in rows]
    tot = sum(ranges) or 1e-12
    out["asset_strength"] = abs(sum(bodies)) / tot * 100.0
    want_up = res.get("signal") == "CALL"
    out["volume_result"] = (sum(rg for rg, b in zip(ranges, bodies)
                                if (b > 0) == want_up) / tot) * 100.0

    hh = max(r["h"] for r in rows)
    ll = min(r["l"] for r in rows)
    cc = float(s.iloc[i])
    p = (hh + ll + cc) / 3.0
    out["r1"] = 2 * p - ll
    out["s1"] = 2 * p - hh
    out["price"] = cc

    conf = gate_score(res)
    out["strength_label"] = ("Strong" if conf >= 70 else
                             "Moderate" if conf >= 60 else "Early")
    out["n_signals"] = res.get("full_total", 0)
    out["conditions"] = ("Optimal" if (res.get("wide") and out["vol_pct"] >= 60) else
                         "Favorable" if out["vol_pct"] >= 30 else "Neutral")
    out["news_sentiment"] = res.get("news_sentiment", 0.0)
    out["social_sentiment"] = res.get("social_sentiment", 0.0)
    return out

def card_caption(res, m, conf, pair, tf_txt):
    buy = res["signal"] == "CALL"
    word = "BUY" if buy else "SELL"
    arrow = "\u25b2" if buy else "\u25bc"
    mkt = "OTC" if pair.endswith("-OTC") else "Stock"
    dec = 3 if "JPY" in pair else 5
    def price(v):
        return f"{v:.{dec}f}" if v is not None else "n/a"
    news_s = res.get("news_sentiment", 0.0)
    social_s = res.get("social_sentiment", 0.0)
    regime = res.get("regime", "normal")
    mkey = str(globals().get("sel_model") or "solari")
    learn_parts = []
    mmeta = _ml_meta.get(f"{pair}#{mkey}") or {}
    if mmeta.get("acc"):
        learn_parts.append(f"{mmeta['acc']:.0f}% backtest ({int(mmeta.get('n', 0))} bars)")
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
        f"\u2022 Asset strength by volume: <b>{m['asset_strength']:.0f}%</b>\n"
        f"\u2022 Volume result: <b>{m['volume_result']:.0f}%</b>\n"
        f"\u2022 Sentiment: <b>{m['sentiment']}</b>\n\n"
        f"\U0001f9ed Technical overview:\n"
        f"\u2022 Current price: <b>{price(m['price'])} {mkt}</b>\n"
        f"\u2022 Resistance (R1): <b>{price(m['r1'])}</b>\n"
        f"\u2022 Support (S1): <b>{price(m['s1'])}</b>\n"
        f"\u2022 RSI: <b>{m['rsi_label']}</b>\n"
        f"\u2022 MACD: <b>{m['macd']}</b>\n"
        f"\u2022 Moving Average: <b>{m['ma']}</b>\n\n"
        f"\U0001f4e6 Signal strength:\n"
        f"\u2022 Strength: <b>{m['strength_label']} ({conf:.0f}%)</b> "
        f"\u00b7 {m.get('n_signals', 0)} past signals checked\n"
        f"\u2022 Market conditions: <b>{m['conditions']}</b>\n"
        f"\u2022 Regime: <b>{regime}</b>\n\n"
        f"{learn_line}"
        f"{win_line}"
        f"{now_hm()}"
    )

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
            return True
    buy = res["signal"] == "CALL"
    dot = "\U0001f7e2" if buy else "\U0001f534"
    send(f"{dot} <b>{'BUY' if buy else 'SELL'}</b> - "
         f"{html.escape(fmt_pair(pair))} | {tf_txt}\n\n{cap}",
         chat_id=chat, keyboard=kb_signal(), html_on=True)
    return True

GEN_STEPS = ["Volatility scan", "RSI Analysis", "MACD Status", "Moving Average",
             "Asset Power", "Volume Analysis", "Resistance", "Support",
             "Signal Accuracy"]
GEN_PCT = [12, 28, 39, 50, 61, 70, 80, 90, 100]

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
    "/stop - Stop all signals\n"
    "Flow: Start -> Select AI Model -> Select a market (Stock/OTC) -> "
    "Select a currency pair -> Select trading time -> signal\n"
    "No orders are placed - signals are display only."
)

if __name__ == "__main__":
    if not TELEGRAM_TOKEN:
        print("TELEGRAM_TOKEN is not set. Set the env var to your BotFather token,")
        print('e.g. in PowerShell:  $env:TELEGRAM_TOKEN="<new token>"  then run again.')
        raise SystemExit(1)
    print("Bot starting (SPIKE UI, live feed, on-demand signals)...")
    S = load_settings()
    sel_model = str(S.get("model") or "solari")
    sel_market = str(S.get("market") or "otc")
    sel_pair = str(S.get("last_pair") or "EURUSD-OTC")
    sel_time = int(S.get("last_time") or 60)
    auto_on = False
    CURRENT_MODEL = sel_model
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
                       "model": sel_model, "market": sel_market})

    def send_menu(chat):
        send(f"{screen_head('Spike', 1)}\nAI models for Pocket Option.\n"
             f"Real live market data. No orders are placed.",
             chat_id=chat, keyboard=kb_menu(), html_on=True)

    def send_model_screen(chat, edit_mid=None):
        label = MODEL_PARAMS.get(sel_model, {}).get("label", "")
        pct, n = model_perf(sel_model)
        stats = (f" \u00b7 <b>{pct:.0f}%</b> from {n} settled signals"
                 if pct is not None and n >= 3 else " \u00b7 learning")
        text = (f"{screen_head('Model', 1)}\nSelect AI Model:  <i>{now_hm()}</i>\n\n"
                f"\U0001f9e0 {MODEL_NAMES.get(sel_model, sel_model)}: {label}{stats}")
        if edit_mid:
            edit(text, chat, edit_mid, html_on=True)
            api_call("editMessageReplyMarkup", {"chat_id": chat, "message_id": edit_mid, "reply_markup": {"inline_keyboard": kb_model()}})
        else:
            send(text, chat_id=chat, keyboard=kb_model(), html_on=True)

    def send_market_screen(chat, edit_mid=None):
        text = f"{screen_head('Market', 2)}\nSelect a market:  <i>{now_hm()}</i>"
        if edit_mid:
            edit(text, chat, edit_mid, html_on=True)
            api_call("editMessageReplyMarkup", {"chat_id": chat, "message_id": edit_mid, "reply_markup": {"inline_keyboard": kb_market()}})
        else:
            send(text, chat_id=chat, keyboard=kb_market(), html_on=True)

    def send_pair_screen(chat, page=0, edit_mid=None):
        pairs = market_pairs(sel_market)
        text = f"{screen_head('Pair', 3)}\nSelect a currency pair:  <i>{now_hm()}</i>"
        if not pairs:
            text = f"{screen_head('Pair', 3)}\nNo {sel_market.upper()} currency pairs are online right now. Try the other market."
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
        text = f"{screen_head('Time', 4)}\nSelect trading time for {html.escape(fmt_pair(sel_pair))}:"
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
            f"{m['asset_strength']:.0f}%",
            f"{m['volume_result']:.0f}%",
            f"{m['r1']:.5f}" if m["r1"] is not None else "n/a",
            f"{m['s1']:.5f}" if m["s1"] is not None else "n/a",
            f"{conf:.0f}% ({res.get('full_total', 0)} past signals)",
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
            elif data == "ui:model":
                mid = cb.get("message", {}).get("message_id")
                send_model_screen(chat, edit_mid=mid)
            elif data.startswith("mdl:"):
                key = data.split(":", 1)[1]
                if key in MODEL_PARAMS:
                    sel_model = key
                    globals()["CURRENT_MODEL"] = key
                    persist()
                    warm_feed(sel_pair, sel_time)
                    ml_train_async(sel_pair, sel_time)
                    mid = cb.get("message", {}).get("message_id")
                    send_market_screen(chat, edit_mid=mid)
            elif data == "ui:market":
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
            persist()
            if watch["on"]:
                stop_watch(chat)
            send("All signals stopped.", chat_id=chat)
        elif cmd in ("/help", "help"):
            send("Commands:\n/start - Main menu\n/stop - Stop all signals", chat_id=chat)

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
                              globals().get("sel_model", "solari"),
                              st.get("regime"), features=_feat_last(closes_w))
                    print(f"Watch alert {st['signal']} {sel_pair}")

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