"""Core invariants for pocket_signal_bot: edge math, gate logic, safety.

Run:  python -m unittest discover -s tests
"""

import os
import sys
import json
import tempfile
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# work whether this file sits at the repo root or inside a tests/ folder
for _d in (_HERE, os.path.dirname(_HERE)):
    if os.path.exists(os.path.join(_d, "pocket_signal_bot.py")):
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break

import pocket_signal_bot as bot


def synth(n=400, seed=1, drift=0.0, vol=0.0007):
    rng = np.random.default_rng(seed)
    return list(np.cumsum(rng.normal(drift, vol, n)) + 1.10)


class TestEdgeMath(unittest.TestCase):
    def test_break_even(self):
        self.assertAlmostEqual(bot.payout_break_even(100.0), 50.0, places=6)
        self.assertAlmostEqual(bot.payout_break_even(50.0), 100 / 1.5, places=6)
        self.assertIsNone(bot.payout_break_even(0))
        self.assertIsNone(bot.payout_break_even(None))

    def test_wilson_lower_bounds(self):
        self.assertEqual(bot._wilson_lower(0, 0), 0.0)
        # all wins -> lower bound below 100 but high
        self.assertLess(bot._wilson_lower(100, 100), 100.0)
        self.assertGreater(bot._wilson_lower(100, 100), 90.0)
        # more evidence tightens the interval toward the point estimate
        small = 50.0 - bot._wilson_lower(50, 100)
        big = 50.0 - bot._wilson_lower(500, 1000)
        self.assertLess(big, small)


class TestFoldEdge(unittest.TestCase):
    def test_fold_edge_shape(self):
        closes = synth(300)
        stk = bot._indicator_stack(closes)
        got = bot._fold_edge(stk, bot.MODEL_PARAMS["engine"])
        self.assertIn("wins", got)
        self.assertIn("total", got)
        self.assertIn("acc", got)
        self.assertGreaterEqual(got["wins"], 0)
        self.assertLessEqual(got["wins"], got["total"])
        if got["total"]:
            self.assertAlmostEqual(got["acc"],
                                   got["wins"] / got["total"] * 100.0, places=6)


class TestGate(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._orig = bot.EDGE_REPORT_FILE
        self._orig_payout = bot.pair_payout
        bot.pair_payout = lambda pair: 95.0          # be ~ 51.28
        bot._EDGE_REPORT["data"] = {}
        bot._EDGE_REPORT["mtime"] = None

    def tearDown(self):
        bot.EDGE_REPORT_FILE = self._orig
        bot.pair_payout = self._orig_payout
        bot._EDGE_REPORT["data"] = {}
        bot._EDGE_REPORT["mtime"] = None

    def _write_report(self, rec):
        path = os.path.join(self._tmp, "edge_report.json")
        with open(path, "w") as f:
            json.dump({"generated": "test", "pairs": {"EURUSD-OTC#60": rec}}, f)
        bot.EDGE_REPORT_FILE = path
        bot._EDGE_REPORT["mtime"] = None
        bot._EDGE_REPORT["data"] = {}

    def test_no_report_is_not_tradeable(self):
        bot.EDGE_REPORT_FILE = os.path.join(self._tmp, "missing.json")
        res = bot.analyze(synth(300), pair="EURUSD-OTC", period=60)
        self.assertFalse(res["tradeable"])
        self.assertEqual(res["signal"], "NO TRADE")

    def test_strong_report_is_tradeable(self):
        self._write_report({"bets": 500, "wins": 350, "win_rate": 70.0,
                            "wilson": 66.0, "expectancy": 0.2, "sqn": 2.0,
                            "break_even": 51.3, "payout": 95.0,
                            "max_drawdown_pct": 10.0, "profit_factor": 1.6,
                            "folds": []})
        res = bot.analyze(synth(300), pair="EURUSD-OTC", period=60)
        self.assertTrue(res["validated"])
        self.assertTrue(res["tradeable"])
        self.assertEqual(res["signal"] in ("CALL", "PUT"), True)

    def test_decayed_fold_blocks_trade(self):
        self._write_report({"bets": 500, "wins": 350, "win_rate": 70.0,
                            "wilson": 66.0, "expectancy": 0.2, "sqn": 2.0,
                            "break_even": 51.3, "payout": 95.0,
                            "max_drawdown_pct": 10.0, "profit_factor": 1.6,
                            "folds": [{"bets": 150, "win_rate": 45.0,
                                       "wilson": 38.0, "expectancy": -0.05,
                                       "sqn": -1.0, "break_even": 51.3}]})
        res = bot.analyze(synth(300), pair="EURUSD-OTC", period=60)
        self.assertFalse(res["tradeable"])
        self.assertEqual(res["signal"], "NO TRADE")
        self.assertIn("decayed", res["gate_note"])


class TestFetchCandles(unittest.TestCase):
    def test_prefers_legacy_get_candles(self):
        import pocket_payout
        seen = {}

        class Api:
            def get_candles(self, sym, period, count):
                seen["legacy"] = True
                return [{"close": 1.0}]

            def get_candles_live(self, *a, **k):
                raise AssertionError("live path must not run when legacy works")

        out = pocket_payout._fetch_candles(Api(), "EURUSD", 60, 10)
        self.assertTrue(seen.get("legacy"))
        self.assertEqual(out, [{"close": 1.0}])

    def test_falls_back_to_live_when_legacy_gone(self):
        import pocket_payout

        class Iterator:
            def __init__(self):
                self.calls = 0

            def __next__(self):
                self.calls += 1
                if self.calls > 1:
                    raise StopIteration
                return ([{"close": 2.0}], None)

            def stop(self):
                pass

        class Api:
            def get_candles(self, *a, **k):
                raise RuntimeError("removed in this version")

            def get_candles_live(self, *a, **k):
                return Iterator()

        out = pocket_payout._fetch_candles(Api(), "EURUSD", 60, 10)
        self.assertEqual(out, [{"close": 2.0}])


class TestScanner(unittest.TestCase):
    def setUp(self):
        self._mp = bot.market_pairs
        self._gs = bot.get_series
        self._an = bot.analyze
        self._min = bot.MIN_PAYOUT
        bot.MIN_PAYOUT = 60.0

    def tearDown(self):
        bot.market_pairs = self._mp
        bot.get_series = self._gs
        bot.analyze = self._an
        bot.MIN_PAYOUT = self._min

    def _wire(self, tradeable_pairs):
        bot.market_pairs = lambda market: [("AAA-OTC", 95.0),
                                           ("BBB-OTC", 50.0),
                                           ("CCC-OTC", 95.0)]
        bot.get_series = lambda pair, period_sec, fetch=True: ([1.0, 1.1, 1.2], "")

        def fake_analyze(closes, model=None, news_symbol=None, pair=None, period=None):
            tr = pair in tradeable_pairs
            return {"signal": "CALL" if tr else "NO TRADE",
                    "full_acc": 60.0 if tr else 45.0, "full_total": 200,
                    "break_even": 52.0, "payout": 95.0, "wilson": 55.0,
                    "tradeable": tr, "validated": tr, "gate_note": ""}

        bot.analyze = fake_analyze

    def test_scan_filters_low_payout_and_sorts(self):
        self._wire({"AAA-OTC"})
        res = bot.scan_pairs(60, "otc")
        self.assertEqual([r["pair"] for r in res], ["AAA-OTC", "CCC-OTC"])
        self.assertTrue(res[0]["tradeable"])
        self.assertFalse(res[1]["tradeable"])
        self.assertEqual(res[0]["edge"], 60.0)

    def test_scan_text_lists_candidates(self):
        self._wire({"AAA-OTC"})
        txt = bot.scan_text(bot.scan_pairs(60, "otc"), 60, "otc")
        self.assertIn("AAA", txt)
        self.assertIn("pairs checked", txt)

    def test_scan_text_empty(self):
        self.assertIn("No pairs", bot.scan_text([], 60, "otc"))


class TestSafety(unittest.TestCase):
    def test_place_trade_disabled_by_default(self):
        import pocket_payout
        os.environ.pop("ALLOW_TRADING", None)
        ok, msg = pocket_payout.place_trade("EURUSD", "CALL", 1, 60)
        self.assertFalse(ok)
        self.assertIn("disabled", msg.lower())


if __name__ == "__main__":
    unittest.main()
