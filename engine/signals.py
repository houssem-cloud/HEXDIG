"""
Signal engine — combines all indicator votes into one Higher / Lower decision
with a confidence ("accuracy") score, exactly like the reference bot UI.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from engine import indicators as ind
from engine.patterns import recognize


@dataclass
class Signal:
    direction: str          # "BUY" or "SELL"
    label: str              # "Higher" or "Lower"
    accuracy: int           # 55..97 displayed as accuracy %
    expiry_line: str        # e.g. "1 minute"
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    regime: str = ""

    def summary(self) -> str:
        arrow = "🔺" if self.direction == "BUY" else "🔻"
        emoji = "🟢" if self.direction == "BUY" else "🔴"
        lines = [
            f"{arrow} <b>{self.label}</b>",
            "",
            f"{emoji} <b>- {self.direction}</b>",
            f"🎯 Accuracy: {self.accuracy}%",
            "⚠️  Use this signal immediately for best results.",
        ]
        return "\n".join(lines)


def analyze(df: pd.DataFrame, expiry_line: str = "1 minute") -> Signal | None:
    """
    df: 1-minute OHLCV DataFrame (>= 60 bars).
    Weights mirror a 1-minute momentum-reversion hybrid:
      * BB(20,2) + BB(20,3) stretch  -> reversion entries
      * Stochastic 6/3/3 cross       -> fast timing
      * CCI 25                       -> momentum confirmation
      * ADX(1,5) + DI                -> trend gate & direction
      * EMA 9/21 + Heikin Ashi       -> short trend alignment
      * candle patterns              -> micro reversal confirmation
    """
    if df is None or len(df) < 60 or df["close"].isna().all():
        return None

    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    price = float(c.iloc[-1])

    mid2, up2, lo2 = ind.bollinger(c, 20, 2.0)
    _,    up3, lo3 = ind.bollinger(c, 20, 3.0)
    k, d = ind.stochastic(df, 6, 3, 3)
    cci_ = ind.cci(df, 25)
    adx_, pdi, mdi = ind.adx(df, 5, 1)
    atr_ = ind.atr(df, 14)
    ema9, ema21 = ind.ema_trend(c, 9, 21)
    ha = ind.heikin_ashi(df)
    patterns = recognize(df)

    votes: list[tuple[str, int, float]] = []     # (name, dir, weight)
    reasons: list[str] = []
    warnings: list[str] = []

    # ---------------- Bollinger stretch (reversion) -----------------------
    if price <= float(lo2.iloc[-1]):
        votes.append(("bb", +1, 2.0)); reasons.append("Price rejected lower Bollinger Band (20, 2)")
    elif price >= float(up2.iloc[-1]):
        votes.append(("bb", -1, 2.0)); reasons.append("Price rejected upper Bollinger Band (20, 2)")

    if price <= float(lo3.iloc[-1]):
        votes.append(("bb3", +1, 1.0)); reasons.append("Stretched beyond BB (20, 3) — volatility extreme")
    elif price >= float(up3.iloc[-1]):
        votes.append(("bb3", -1, 1.0)); reasons.append("Stretched beyond BB (20, 3) — volatility extreme")

    # Band-width squeeze -> breakout watch
    bw = (up2 - lo2) / mid2
    if float(bw.iloc[-1]) < float(bw.rolling(50).mean().iloc[-1]) * 0.6:
        warnings.append("Squeeze: volatility compressed — expect fast move")

    # ---------------- Stochastic 6/3/3 ------------------------------------
    k_now, d_now = float(k.iloc[-1]), float(d.iloc[-1])
    k_prev, d_prev = float(k.iloc[-2]), float(d.iloc[-2])
    if k_prev <= d_prev and k_now > d_now and k_now < 40:
        votes.append(("stoch", +1, 1.8)); reasons.append(f"Stochastic(6,3,3) bullish cross at {k_now:.0f}")
    elif k_prev >= d_prev and k_now < d_now and k_now > 60:
        votes.append(("stoch", -1, 1.8)); reasons.append(f"Stochastic(6,3,3) bearish cross at {k_now:.0f}")
    elif k_now < 15:
        votes.append(("stoch", +1, 0.8)); reasons.append("Stochastic deeply oversold (<15)")
    elif k_now > 85:
        votes.append(("stoch", -1, 0.8)); reasons.append("Stochastic deeply overbought (>85)")

    # ---------------- CCI 25 ----------------------------------------------
    c_now, c_prev = float(cci_.iloc[-1]), float(cci_.iloc[-2])
    if c_now < -100 and c_now > c_prev:
        votes.append(("cci", +1, 1.4)); reasons.append(f"CCI(25) turning up from oversold ({c_now:.0f})")
    elif c_now > 100 and c_now < c_prev:
        votes.append(("cci", -1, 1.4)); reasons.append(f"CCI(25) turning down from overbought ({c_now:.0f})")
    elif c_now > 0 and c_now > c_prev:
        votes.append(("cci", +1, 0.5))
    elif c_now < 0 and c_now < c_prev:
        votes.append(("cci", -1, 0.5))

    # ---------------- ADX(1,5) trend gate ----------------------------------
    a_now = float(adx_.iloc[-1])
    di_dir = 1 if float(pdi.iloc[-1]) > float(mdi.iloc[-1]) else -1
    if not pd.isna(a_now) and a_now >= 22:
        votes.append(("adx", di_dir, 1.2))
        reasons.append(f"ADX(1,5)={a_now:.0f} — trend active, DI {'+' if di_dir > 0 else '-'} side")
    elif not pd.isna(a_now) and a_now < 15:
        warnings.append(f"ADX={a_now:.0f}: flat/choppy market — lower reliability")

    # ---------------- EMA 9/21 short trend ---------------------------------
    if float(ema9.iloc[-1]) > float(ema21.iloc[-1]) and price > float(ema9.iloc[-1]):
        votes.append(("ema", +1, 1.0)); reasons.append("Short-term uptrend: EMA9 > EMA21, price above EMA9")
    elif float(ema9.iloc[-1]) < float(ema21.iloc[-1]) and price < float(ema9.iloc[-1]):
        votes.append(("ema", -1, 1.0)); reasons.append("Short-term downtrend: EMA9 < EMA21, price below EMA9")

    # ---------------- Heikin Ashi ------------------------------------------
    ha_bull = ha["close"].iloc[-1] > ha["open"].iloc[-1]
    ha_bull_prev = ha["close"].iloc[-2] > ha["open"].iloc[-2]
    if ha_bull and ha_bull_prev and not (ha["open"].iloc[-1] - ha["close"].iloc[-1] == 0):
        votes.append(("ha", +1, 0.7)); reasons.append("Heikin Ashi: consecutive bullish candles")
    elif (not ha_bull) and (not ha_bull_prev):
        votes.append(("ha", -1, 0.7)); reasons.append("Heikin Ashi: consecutive bearish candles")

    # ---------------- Candlestick patterns ---------------------------------
    for p in patterns:
        if p.direction != 0:
            votes.append(("pat", p.direction, p.weight))
            reasons.append(f"Pattern: {p.note}")
        elif p.name == "Doji":
            warnings.append("Doji on last candle — indecision")

    # ---------------- Aggregation ------------------------------------------
    if not votes:
        return None

    bull = sum(w for _, dr, w in votes if dr > 0)
    bear = sum(w for _, dr, w in votes if dr < 0)
    total = bull + bear
    if total <= 0:
        return None

    score = bull - bear
    direction = "BUY" if score > 0 else "SELL"
    dominant = max(bull, bear)
    agreement = dominant / total                  # 0.5 .. 1.0
    strength = abs(score) / max(total, 1e-9)

    # Volatility regime: ATR vs its own average
    a_now_atr = float(atr_.iloc[-1])
    a_avg = float(atr_.rolling(50).mean().iloc[-1])
    vol_ratio = a_now_atr / a_avg if a_avg > 0 else 1.0
    if vol_ratio > 2.2:
        warnings.append("Volatility spike — candles are large, spread risk")
        conf_penalty = 4
    elif vol_ratio < 0.6:
        warnings.append("Low volatility — signals are weaker")
        conf_penalty = 8
    else:
        conf_penalty = 0

    choppy = (not pd.isna(a_now)) and a_now < 15

    accuracy = 55 + int(min(28, strength * 30) + (agreement - 0.5) * 22 - conf_penalty - (6 if choppy else 0))
    accuracy = max(55, min(97, accuracy))

    return Signal(
        direction=direction,
        label="Higher" if direction == "BUY" else "Lower",
        accuracy=accuracy,
        expiry_line=expiry_line,
        reasons=reasons[:6],
        warnings=warnings,
        regime=f"ADX {a_now:.0f} | ATRx {vol_ratio:.1f}",
    )
