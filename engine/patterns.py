"""
Candlestick pattern recognition on raw M1 candles.

Detects reversal / continuation formations on the most recent candles and
returns bullish/bearish votes for the signal engine.
"""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class Pattern:
    name: str
    direction: int          # +1 bullish, -1 bearish, 0 neutral
    weight: float           # strength 0..1.5
    note: str


def _body(o, c): return abs(c - o)
def _range(h, l): return max(h - l, 1e-12)


def recognize(df: pd.DataFrame) -> list[Pattern]:
    """Evaluate patterns on the last 3 candles of the 1-minute series."""
    if len(df) < 5:
        return []
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()

    out: list[Pattern] = []
    i = len(df) - 1

    body = _body(o[i], c[i])
    rng = _range(h[i], l[i])
    upper_wick = h[i] - max(o[i], c[i])
    lower_wick = min(o[i], c[i]) - l[i]
    prev_body = _body(o[i - 1], c[i - 1])

    # --- Engulfing -------------------------------------------------------
    if (c[i] > o[i] and c[i - 1] < o[i - 1]
            and c[i] >= o[i - 1] and o[i] <= c[i - 1] and body > prev_body):
        out.append(Pattern("Bullish Engulfing", +1, 1.4, "Bullish engulfing candle"))

    if (c[i] < o[i] and c[i - 1] > o[i - 1]
            and o[i] >= c[i - 1] and c[i] <= o[i - 1] and body > prev_body):
        out.append(Pattern("Bearish Engulfing", -1, 1.4, "Bearish engulfing candle"))

    # --- Hammer / Shooting star (pin bars) --------------------------------
    if lower_wick > 2.0 * body and upper_wick < body * 0.8 and body < rng * 0.45:
        out.append(Pattern("Hammer", +1, 1.2, "Hammer / lower pin bar"))

    if upper_wick > 2.0 * body and lower_wick < body * 0.8 and body < rng * 0.45:
        out.append(Pattern("Shooting Star", -1, 1.2, "Shooting star / upper pin bar"))

    # --- Doji (indecision — used as a veto hint) ---------------------------
    if body < rng * 0.1:
        out.append(Pattern("Doji", 0, 0.4, "Doji — indecision candle"))

    # --- Marubozu ----------------------------------------------------------
    if c[i] > o[i] and upper_wick < rng * 0.08 and lower_wick < rng * 0.08:
        out.append(Pattern("Bullish Marubozu", +1, 1.0, "Bullish marubozu"))

    if c[i] < o[i] and upper_wick < rng * 0.08 and lower_wick < rng * 0.08:
        out.append(Pattern("Bearish Marubozu", -1, 1.0, "Bearish marubozu"))

    # --- Morning / Evening star -------------------------------------------
    if len(df) >= 3:
        o1, c1 = o[i - 2], c[i - 2]
        o2, c2 = o[i - 1], c[i - 1]
        if c1 < o1 and _body(o2, c2) < _body(o1, c1) * 0.5 and c[i] > o[i] and c[i] > (o1 + c1) / 2:
            out.append(Pattern("Morning Star", +1, 1.5, "Morning star reversal"))

        if c1 > o1 and _body(o2, c2) < _body(o1, c1) * 0.5 and c[i] < o[i] and c[i] < (o1 + c1) / 2:
            out.append(Pattern("Evening Star", -1, 1.5, "Evening star reversal"))

    # --- Tweezer tops / bottoms --------------------------------------------
    # Equal highs/lows must be genuinely close (about half a pip), not "within
    # one candle range" — the previous tolerance made this fire on almost every
    # candle and skewed the confluence score.
    tol = max(abs(c[i]) * 5e-5, 1e-9)
    if (abs(l[i] - l[i - 1]) <= tol and c[i] > o[i] and c[i - 1] < o[i - 1]):
        out.append(Pattern("Tweezer Bottom", +1, 0.8, "Tweezer bottom"))

    if (abs(h[i] - h[i - 1]) <= tol and c[i] < o[i] and c[i - 1] > o[i - 1]):
        out.append(Pattern("Tweezer Top", -1, 0.8, "Tweezer top"))

    return out
