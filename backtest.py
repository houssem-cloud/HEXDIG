"""
Offline binary-option backtester for pocket_signal_bot.

Runs the SAME decision rule the live bot uses over historical candles and
scores it as binary-option bets (fixed stake, fixed expiry, fixed payout),
instead of a plain hit-rate. It drives the bar loop with backtrader
(Cerebro + PandasData) and a custom binary analyzer.

Metrics reported per pair:
  bets, wins/losses/ties, win-rate, Wilson 95% lower bound, break-even,
  expectancy per bet, profit factor, max drawdown, longest streaks, SQN.

Usage:
  # live candles through your saved SSID
  python backtest.py EURUSD-OTC --period 60 --bars 800

  # several pairs, save a report the bot reads as its edge gate
  python backtest.py EURUSD-OTC GBPUSD-OTC --period 60 --save edge_report.json

  The saved file is keyed "PAIR#PERIOD" and is picked up automatically by
  pocket_signal_bot.analyze(); when present it REPLACES the live replay gate
  (expectancy + Wilson bound vs. break-even). Delete it to fall back.

  # from a CSV (columns: datetime,open,high,low,close)
  python backtest.py --csv mydata.csv --payout 88

Notes:
  - Display/reporting only. Nothing is traded.
  - Requires your SSID saved (python pocket_payout.py --save "...") unless
    you pass --csv.
"""

import os
import sys
import json
import math
import time
import argparse
import collections
import collections.abc

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# --- backtrader on modern Python (3.10+ / 3.15): restore symbols it expects ---
for _name in ("Iterable", "Mapping", "MutableMapping", "Sequence",
              "MutableSequence", "Callable", "Iterator", "Hashable"):
    if not hasattr(collections, _name) and hasattr(collections.abc, _name):
        setattr(collections, _name, getattr(collections.abc, _name))
if not hasattr(time, "clock"):
    time.clock = time.perf_counter

# --- backtrader: prefer an installed copy, else the sibling folder ---
try:
    import backtrader as bt
except Exception:
    _bt = os.path.join(os.path.dirname(HERE), "backtrader-master")
    if os.path.isdir(_bt):
        sys.path.insert(0, _bt)
    import backtrader as bt

import pocket_signal_bot as bot


# ============================================================ data

def _frame_from_rows(rows, period):
    if not rows or len(rows) < 80:
        return None
    idx = pd.date_range(end=pd.Timestamp.now().floor("min"),
                        periods=len(rows), freq=f"{max(1, int(period))}s")
    df = pd.DataFrame({
        "open":   [float(r["o"]) for r in rows],
        "high":   [float(r["h"]) for r in rows],
        "low":    [float(r["l"]) for r in rows],
        "close":  [float(r["c"]) for r in rows],
        "volume": 100.0,
    }, index=idx)
    ohlc = [{"o": float(r["o"]), "h": float(r["h"]),
             "l": float(r["l"]), "c": float(r["c"])} for r in rows]
    return df, ohlc


def load_live(pair, period, bars):
    if bot.pocket_payout is None:
        return None, None
    try:
        rows = bot.pocket_payout.get_ohlcv(pair, period, bars)
    except Exception as e:
        print(f"  data error {pair}: {e}")
        return None, None
    if not rows:
        return None, None
    # the snapshot may include the currently-forming candle - drop it
    if len(rows) > 80:
        rows = rows[:-1]
    return _frame_from_rows(rows, period)


def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    ren = {}
    for c in df.columns:
        if c in ("date", "time", "datetime", "timestamp", "index"):
            ren[c] = "datetime"
    df = df.rename(columns=ren)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.date_range(end=pd.Timestamp.now().floor("min"),
                                 periods=len(df), freq="60s")
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise SystemExit(f"CSV missing '{col}' column")
    if "volume" not in df.columns:
        df["volume"] = 100.0
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    ohlc = [{"o": r.open, "h": r.high, "l": r.low, "c": r.close}
            for r in df.itertuples()]
    return df, ohlc


# ============================================================ signals

def build_signals(closes, ohlc, model="engine"):
    """Replay the live rule causally over the whole series."""
    stk = bot._indicator_stack(closes, ohlc)
    params = bot.MODEL_PARAMS.get(model, bot.MODEL_PARAMS["engine"])
    n = len(closes)
    sigs = [None] * n
    for i in range(n):
        sigs[i] = bot.decide_at(stk, i, params)
    return sigs


# ============================================================ engine

class BinaryStrategy(bt.Strategy):
    """Each signal opens a fixed-expiry binary bet; settle HORIZON bars later."""
    params = dict(signals=None, payout=0.85, horizon=1)

    def __init__(self):
        self.sig = self.p.signals or []
        self.open = None
        self.bets = []

    def prenext(self):
        self.next()

    def next(self):
        i = len(self) - 1
        if self.open is not None and i >= self.open["expiry"]:
            self._settle()
        if self.open is None and 0 <= i < len(self.sig) and self.sig[i] in ("CALL", "PUT"):
            self.open = {"dir": self.sig[i],
                         "entry": float(self.data.close[0]),
                         "expiry": i + int(self.p.horizon)}

    def _settle(self):
        exit_ = float(self.data.close[0])
        entry = self.open["entry"]
        d = self.open["dir"]
        if exit_ == entry:
            ret = 0.0                       # tie -> refund
        else:
            win = (exit_ > entry) if d == "CALL" else (exit_ < entry)
            ret = self.p.payout if win else -1.0
        self.bets.append({"dir": d, "entry": entry, "exit": exit_, "ret": ret})
        self.open = None


class BinaryAnalyzer(bt.Analyzer):
    """Binary-option statistics over the recorded bets."""
    def stop(self):
        self.rets["bets"] = list(self.strategy.bets)


# ============================================================ stats

def _streaks(rets):
    best_w = best_l = cur_w = cur_l = 0
    for r in rets:
        if r > 0:
            cur_w += 1; cur_l = 0
        elif r < 0:
            cur_l += 1; cur_w = 0
        else:
            continue
        best_w = max(best_w, cur_w)
        best_l = max(best_l, cur_l)
    return best_w, best_l


def summarize(bets, payout_pct, start_bank=100.0, stake=1.0):
    n = len(bets)
    out = {"bets": 0, "wins": 0, "losses": 0, "ties": 0, "win_rate": 0.0,
           "wilson": 0.0, "break_even": bot.payout_break_even(payout_pct) or 0.0,
           "expectancy": 0.0, "profit_factor": 0.0, "max_drawdown_units": 0.0,
           "max_drawdown_pct": 0.0, "best_win_streak": 0, "worst_loss_streak": 0,
           "sqn": 0.0, "sortino": 0.0, "net_units": 0.0, "payout": payout_pct}
    if not n:
        return out
    rets = [b["ret"] for b in bets]
    wins = sum(1 for r in rets if r > 0)
    losses = sum(1 for r in rets if r < 0)
    ties = n - wins - losses
    decided = wins + losses
    wr = (wins / decided * 100.0) if decided else 0.0
    p = payout_pct / 100.0
    exp = (wr / 100.0) * p - (1 - wr / 100.0)     # per 1-unit stake
    gross_win = sum(r for r in rets if r > 0)
    gross_loss = -sum(r for r in rets if r < 0)
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

    # equity curve with fixed fractional stake for a realistic drawdown
    eq = start_bank
    peak = start_bank
    maxdd_pct = 0.0
    for r in rets:
        eq += stake * r
        peak = max(peak, eq)
        if peak > 0:
            maxdd_pct = max(maxdd_pct, (peak - eq) / peak * 100.0)

    # drawdown in units (flat 1u stake)
    eq_u = peak_u = maxdd_u = 0.0
    for r in rets:
        eq_u += r
        peak_u = max(peak_u, eq_u)
        maxdd_u = max(maxdd_u, peak_u - eq_u)

    mean = sum(rets) / n
    sd = (sum((r - mean) ** 2 for r in rets) / n) ** 0.5
    sqn = (mean / sd * math.sqrt(n)) if sd > 0 else 0.0
    # downside deviation -> Sortino (metric style borrowed from Hummingbot's
    # backtesting results, which reports risk-adjusted return, not just win %)
    downside = [r for r in rets if r < 0]
    dsd = (sum(r * r for r in downside) / len(downside)) ** 0.5 if downside else 0.0
    sortino = (mean / dsd * math.sqrt(n)) if dsd > 0 else 0.0
    bw, bl = _streaks(rets)

    out.update({
        "bets": n, "wins": wins, "losses": losses, "ties": ties,
        "win_rate": round(wr, 1), "wilson": round(bot._wilson_lower(wins, decided), 1),
        "expectancy": round(exp, 3),
        "profit_factor": (round(pf, 2) if pf != float("inf") else None),
        "max_drawdown_units": round(maxdd_u, 2),
        "max_drawdown_pct": round(maxdd_pct, 2),
        "best_win_streak": bw, "worst_loss_streak": bl,
        "sqn": round(sqn, 2), "sortino": round(sortino, 2),
        "net_units": round(sum(rets), 2),
    })
    return out


def verdict(s):
    """Mirror the live bot's edge gate so the CLI never contradicts it."""
    if s["bets"] < 30:
        return "too few bets"
    if s["win_rate"] >= 95.0:
        return "SUSPICIOUS: win-rate >= 95% reads like a broken/synthetic feed"
    if s["expectancy"] <= 0:
        return "no edge"
    if s["wilson"] < s["break_even"]:
        return "uncertain (Wilson below break-even)"
    return "POSITIVE"


def report(pair, s, period):
    be = s["break_even"]
    edge = s["win_rate"] - be
    verdict_txt = verdict(s)
    pf = "inf" if s["profit_factor"] is None else f"{s['profit_factor']:.2f}"
    print(f"\n=== {pair}  ({int(period)}s bars) ===")
    print(f"  bets {s['bets']}  (W {s['wins']} / L {s['losses']} / tie {s['ties']})")
    print(f"  win-rate {s['win_rate']:.1f}%   Wilson95 {s['wilson']:.1f}%   "
          f"break-even {be:.1f}%   (payout {s['payout']:.0f}%)")
    print(f"  expectancy/bet {s['expectancy']:+.3f}u   profit factor {pf}   "
          f"net {s['net_units']:+.1f}u   -> {verdict_txt}")
    print(f"  max drawdown {s['max_drawdown_units']:.1f}u "
          f"({s['max_drawdown_pct']:.1f}%)   streaks W{s['best_win_streak']} "
          f"L{s['worst_loss_streak']}   SQN {s['sqn']}   Sortino {s['sortino']}")


# ============================================================ main

def main():
    ap = argparse.ArgumentParser(description="Binary-option backtest for pocket_signal_bot")
    ap.add_argument("pairs", nargs="*", default=["EURUSD-OTC"])
    ap.add_argument("--period", type=int, default=60, help="bar size in seconds")
    ap.add_argument("--bars", type=int, default=800, help="candles to fetch")
    ap.add_argument("--payout", type=float, default=None, help="payout %% (default: live)")
    ap.add_argument("--horizon", type=int, default=bot.HORIZON, help="bars to expiry")
    ap.add_argument("--folds", type=int, default=3,
                    help="split bets into N chronological folds to test stability")
    ap.add_argument("--csv", default=None, help="use a CSV instead of live data")
    ap.add_argument("--save", default=None, help="write the report to a JSON file")
    args = ap.parse_args()

    results = {}
    if args.csv:
        df, ohlc = load_csv(args.csv)
        pair = os.path.basename(args.csv)
        payout = args.payout if args.payout else 85.0
        _run_one(pair, df, ohlc, payout, args, results)
    else:
        if not args.pairs:
            raise SystemExit("give at least one pair, or --csv")
        for pair in args.pairs:
            df, ohlc = load_live(pair, args.period, args.bars)
            if df is None:
                print(f"\n=== {pair} ===\n  no candles (SSID not saved or pair offline)")
                continue
            payout = args.payout
            if payout is None and bot.pocket_payout is not None:
                try:
                    payout = bot.pocket_payout.get_payout(pair)
                except Exception:
                    payout = None
            if not payout:
                payout = 85.0
                print(f"  ({pair}: payout unknown, assuming 85%)")
            _run_one(pair, df, ohlc, payout, args, results)

    if args.save and results:
        payload = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "pairs": results}
        with open(args.save, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nsaved report -> {args.save}  ({len(results)} pair/timeframe)")


def _run_one(pair, df, ohlc, payout, args, results):
    closes = list(df["close"].values)
    signals = build_signals(closes, ohlc, model="engine")
    n_sig = sum(1 for s in signals if s in ("CALL", "PUT"))

    cerebro = bt.Cerebro()
    cerebro.adddata(bt.feeds.PandasData(dataname=df))
    cerebro.addstrategy(BinaryStrategy, signals=signals,
                        payout=payout / 100.0, horizon=args.horizon)
    cerebro.addanalyzer(BinaryAnalyzer, _name="bin")
    strat = cerebro.run()[0]
    bets = strat.analyzers.bin.rets.get("bets", [])

    s = summarize(bets, payout)
    report(pair, s, args.period)
    print(f"  ({n_sig} rule signals over {len(closes)} bars)")

    # split the bets chronologically so edge stability is visible, and so the
    # live bot can refuse an edge that only existed in an older regime
    folds = []
    nf = max(1, int(getattr(args, "folds", 3)))
    if nf >= 2 and len(bets) >= nf * 20:
        step = len(bets) // nf
        for f in range(nf):
            lo = f * step
            hi = (f + 1) * step if f < nf - 1 else len(bets)
            fs = summarize(bets[lo:hi], payout)
            folds.append({k: fs[k] for k in
                          ("bets", "wins", "win_rate", "wilson",
                           "expectancy", "sqn", "break_even")})
        print("  folds: " + "  ".join(
            f"[{x['win_rate']:.0f}%/{x['bets']}b]" for x in folds))

    results[f"{pair}#{int(args.period)}"] = {"pair": pair,
                                             "period": int(args.period),
                                             "folds": folds, **s}


if __name__ == "__main__":
    main()
    # BinaryOptionsToolsV2 keeps native websocket threads alive after the
    # report is written; interpreter shutdown can segfault (core dumped)
    # even on a successful run. The report is already on disk, so exit
    # hard and clean.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
