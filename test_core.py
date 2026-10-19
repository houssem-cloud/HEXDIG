"""Core invariants for pocket_signal_bot: edge math, gate logic, safety.

Run:  python -m unittest discover -s tests
"""

import os
import sys
import json
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

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


class TestSafety(unittest.TestCase):
    def test_place_trade_disabled_by_default(self):
        import pocket_payout
        os.environ.pop("ALLOW_TRADING", None)
        ok, msg = pocket_payout.place_trade("EURUSD", "CALL", 1, 60)
        self.assertFalse(ok)
        self.assertIn("disabled", msg.lower())


if __name__ == "__main__":
    unittest.main()
