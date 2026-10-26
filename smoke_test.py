"""Offline smoke test for the confluence engine (no Telegram token needed)."""
import numpy as np
import pandas as pd

from engine.signals import analyze
from engine.datafeed import _synthetic


def random_market(seed=7, bars=240, drift=+0.00003):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.00006, bars)
    close = 1.10 * (1 + np.cumsum(noise) + np.linspace(0, drift * bars, bars))
    open_ = np.concatenate([[close[0]], close[:-1]])
    wick = np.abs(rng.normal(0, 0.00004, bars))
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("min"), periods=bars, freq="1min")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": np.ones(bars)}, index=idx)


def main():
    df = random_market(seed=11, drift=-0.00004)
    sig = analyze(df, "1 minute")
    assert sig is not None, "engine produced no signal"
    print("Direction:", sig.direction, "| Label:", sig.label,
          "| Accuracy:", f"{sig.accuracy}%")
    print("Regime:", sig.regime)
    for r in sig.reasons:
        print("  -", r)
    assert 55 <= sig.accuracy <= 97
    assert sig.direction in ("BUY", "SELL")

    # OTC synthetic feed must be reproducible (same minute -> same candles)
    a, b = _synthetic("EUR/USD", "OTC"), _synthetic("EUR/USD", "OTC")
    assert np.allclose(a["close"], b["close"])
    print("OTC feed deterministic: OK")

    # 3 random markets must yield valid signals or None, never crash
    for s in range(5):
        analyze(random_market(seed=100 + s, drift=np.random.default_rng(s).normal(0, 5e-5)))
    print("Engine smoke test: ALL OK")


if __name__ == "__main__":
    main()
