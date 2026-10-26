"""Core invariants for pocket_signal_bot: edge math, gate logic, safety.

Run:  python -m unittest discover -s tests
"""

import os
import sys
import json
import time
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
        # the gate needs the whole-sample size, not just the recent fold
        self.assertGreaterEqual(got["all_total"], got["total"])
        self.assertGreaterEqual(got["all_wins"], 0)
        self.assertLessEqual(got["all_wins"], got["all_total"])
        if got["all_total"]:
            self.assertAlmostEqual(got["all_acc"],
                                   got["all_wins"] / got["all_total"] * 100.0,
                                   places=6)


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
        # confluence rule: a validated edge only becomes CALL/PUT when the
        # setup itself exists on the current bar, else an honest NO TRADE
        self.assertIn(res["signal"], ("CALL", "PUT", "NO TRADE"))

    def test_tradeable_without_current_setup_is_no_trade(self):
        self._write_report({"bets": 500, "wins": 350, "win_rate": 70.0,
                            "wilson": 66.0, "expectancy": 0.2, "sqn": 2.0,
                            "break_even": 51.3, "payout": 95.0,
                            "max_drawdown_pct": 10.0, "profit_factor": 1.6,
                            "folds": []})
        res = bot.analyze(synth(300), pair="EURUSD-OTC", period=60)
        self.assertTrue(res["tradeable"])
        # if there is a vote the signal matches it; otherwise NO TRADE
        if res["setup"]:
            self.assertIn(res["signal"], ("CALL", "PUT"))
        else:
            self.assertEqual(res["signal"], "NO TRADE")
            self.assertIn("confluence", res["gate_note"])

    def test_decayed_report_does_not_veto(self):
        # a negative/decayed report is informational; the live replay decides
        self._write_report({"bets": 500, "wins": 350, "win_rate": 70.0,
                            "wilson": 66.0, "expectancy": 0.2, "sqn": 2.0,
                            "break_even": 51.3, "payout": 95.0,
                            "max_drawdown_pct": 10.0, "profit_factor": 1.6,
                            "folds": [{"bets": 150, "win_rate": 45.0,
                                       "wilson": 38.0, "expectancy": -0.05,
                                       "sqn": -1.0, "break_even": 51.3}]})
        res = bot.analyze(synth(300), pair="EURUSD-OTC", period=60)
        self.assertFalse(res["validated"])
        if res["tradeable"]:
            self.assertIn(res["signal"], ("CALL", "PUT"))
        else:
            self.assertEqual(res["signal"], "NO TRADE")
        self.assertIn(res["gate_reason"],
                      ("edge", "wilson", "fold", "sample", "ok"))


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


class TestReplayGate(unittest.TestCase):
    """No report present -> the live replay gate must use the Wilson floor."""

    def setUp(self):
        self._fe = bot._fold_edge
        self._pp = bot.pair_payout
        self._rep = bot.EDGE_REPORT_FILE
        bot.pair_payout = lambda pair: 90.0          # break-even ~52.63
        bot.EDGE_REPORT_FILE = os.path.join(tempfile.gettempdir(),
                                            "no_such_edge.json")
        bot._EDGE_REPORT["data"] = {}
        bot._EDGE_REPORT["mtime"] = None

    def tearDown(self):
        bot._fold_edge = self._fe
        bot.pair_payout = self._pp
        bot.EDGE_REPORT_FILE = self._rep
        bot._EDGE_REPORT["data"] = {}
        bot._EDGE_REPORT["mtime"] = None

    def _fold(self, wins, total, accs):
        bot._fold_edge = lambda stk, params, folds=4: {
            "wins": wins, "total": total,
            "acc": wins / total * 100.0, "accs": accs}

    def test_thin_floor_blocks(self):
        self._fold(62, 100, [62, 62, 62, 62])   # 62% acc but floor ~52% < be 52.63
        res = bot.analyze(synth(300), pair="ZARUSD-OTC", period=60)
        self.assertFalse(res["tradeable"])
        self.assertIn("floor", res["gate_note"])

    def test_below_min_accuracy_blocks(self):
        self._fold(55, 100, [55, 55, 55, 55])   # 55% < MIN_ACCURACY 60
        res = bot.analyze(synth(300), pair="ZARUSD-OTC", period=60)
        self.assertFalse(res["tradeable"])
        self.assertEqual(res["gate_reason"], "accuracy")
        self.assertIn("minimum", res["gate_note"])

    def test_strong_floor_trades(self):
        self._fold(70, 100, [70, 70, 70, 70])
        res = bot.analyze(synth(300), pair="ZARUSD-OTC", period=60)
        self.assertTrue(res["tradeable"])

    def test_decayed_recent_fold_blocks(self):
        self._fold(70, 100, [70, 70, 70, 40])   # last fold collapsed
        res = bot.analyze(synth(300), pair="ZARUSD-OTC", period=60)
        self.assertFalse(res["tradeable"])
        self.assertIn("fold", res["gate_note"].lower())

    def test_point_mode_skips_wilson_floor(self):
        orig = bot.GATE_MODE
        bot.GATE_MODE = "point"
        try:
            self._fold(68, 110, [68, 68, 68, 68])   # 61.8% acc, thin floor
            res = bot.analyze(synth(300), pair="ZARUSD-OTC", period=60)
            self.assertTrue(res["tradeable"])
            self.assertIn("point mode", res["gate_note"])
        finally:
            bot.GATE_MODE = orig


class TestScanner(unittest.TestCase):
    def setUp(self):
        self._mp = bot.market_pairs
        self._gs = bot.get_series
        self._an = bot.analyze
        self._fo = bot.fetch_ohlc
        self._min = bot.MIN_PAYOUT
        bot.MIN_PAYOUT = 60.0

    def tearDown(self):
        bot.market_pairs = self._mp
        bot.get_series = self._gs
        bot.analyze = self._an
        bot.fetch_ohlc = self._fo
        bot.MIN_PAYOUT = self._min

    def _wire(self, tradeable_pairs):
        bot.market_pairs = lambda market: [("AAA-OTC", 95.0),
                                           ("BBB-OTC", 50.0),
                                           ("CCC-OTC", 95.0)]
        bot.fetch_ohlc = lambda pair, period_sec, count=220, ttl=None: None
        bot.get_series = lambda pair, period_sec, fetch=True, count=220: ([1.0, 1.1, 1.2], "")

        def fake_analyze(closes, model=None, news_symbol=None, pair=None,
                         period=None, ohlc=None, z_mult=1.0):
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

    def test_scan_all_balances_markets(self):
        self._wire(set())
        pairs = ([(f"O{i}-OTC", 95.0) for i in range(4)]
                 + [(f"S{i}USD", 95.0) for i in range(4)])
        bot.market_pairs = lambda market: pairs
        got = [r["pair"] for r in bot.scan_pairs(60, "all", max_pairs=4)]
        self.assertEqual(len(got), 4)
        self.assertTrue(any(p.endswith("-OTC") for p in got))
        self.assertTrue(any(not p.endswith("-OTC") for p in got))

    def test_deep_check_rescores_with_more_history(self):
        # a 220-bar replay can be too thin; the best candidate is re-scored on
        # SCAN_HISTORY bars before a card is ever sent
        bot.market_pairs = lambda market: [("AAA-OTC", 95.0)]
        bars = [{"o": 1.0, "h": 1.1, "l": 0.9, "c": 1.0 + i * 0.001}
                for i in range(900)]
        seen = []

        def fake_ohlc(pair, period_sec, count=220, ttl=None):
            return bars if count >= 600 else bars[:220]

        def fake_analyze(closes, model=None, news_symbol=None, pair=None,
                         period=None, ohlc=None, z_mult=1.0):
            seen.append(len(closes))
            tr = len(closes) >= 600
            return {"signal": "CALL" if tr else "NO TRADE",
                    "full_acc": 60.0 if tr else 50.0, "full_total": len(closes),
                    "break_even": 52.0, "payout": 95.0, "wilson": 55.0,
                    "tradeable": tr, "validated": False,
                    "gate_reason": "ok" if tr else "edge", "gate_note": ""}

        bot.fetch_ohlc = fake_ohlc
        bot.analyze = fake_analyze
        res = bot.scan_pairs(60, "otc")
        self.assertTrue(res[0]["tradeable"])
        self.assertEqual(res[0]["history"], 900)
        self.assertIn(900, seen)

    def test_deep_check_never_downgrades_sample(self):
        bot.market_pairs = lambda market: [("AAA-OTC", 95.0)]
        broad = [{"o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0} for _ in range(220)]
        bot.fetch_ohlc = lambda pair, period_sec, count=220, ttl=None: (
            broad if count <= 220 else broad[:150])

        def fake_analyze(closes, model=None, news_symbol=None, pair=None,
                         period=None, ohlc=None, z_mult=1.0):
            return {"signal": "CALL", "full_acc": 60.0,
                    "full_total": len(closes), "break_even": 52.0,
                    "payout": 95.0, "wilson": 55.0, "tradeable": True,
                    "validated": False, "gate_reason": "ok", "gate_note": ""}

        bot.analyze = fake_analyze
        res = bot.scan_pairs(60, "otc")
        self.assertTrue(res[0]["tradeable"])
        self.assertNotIn("history", res[0])
        self.assertEqual(bot.DEEP_INFO["AAA-OTC"]["kept"], 220)
        self.assertEqual(bot.DEEP_INFO["AAA-OTC"]["got"], 150)


class TestMarketAll(unittest.TestCase):
    def test_all_and_labels(self):
        orig = bot.get_pairs
        bot.get_pairs = lambda: [("EURUSD-OTC", 90.0), ("EURUSD", 84.0)]
        try:
            self.assertEqual(len(bot.market_pairs("all")), 2)
            self.assertEqual(len(bot.market_pairs("otc")), 1)
            self.assertEqual(len(bot.market_pairs("stock")), 1)
        finally:
            bot.get_pairs = orig
        self.assertIn("BOTH", bot.market_label("all"))
        self.assertEqual(bot.market_label("otc"), "OTC")
        self.assertEqual(bot.market_label("stock"), "Stock")
        self.assertEqual(bot.market_counts([{"pair": "EURUSD-OTC"},
                                            {"pair": "GBPUSD"},
                                            {"pair": "AUDCAD-OTC"}]), (2, 1))

    def test_settings_legacy_all_migrates_to_otc(self):
        tmp = tempfile.mkdtemp()
        orig = bot.SETTINGS_FILE
        bot.SETTINGS_FILE = os.path.join(tmp, "pocket_settings.json")
        try:
            self.assertEqual(bot.load_settings()["market"], "otc")
            with open(bot.SETTINGS_FILE, "w") as f:
                json.dump({"market": "all"}, f)   # legacy "Both" -> migrates
            self.assertEqual(bot.load_settings()["market"], "otc")
            with open(bot.SETTINGS_FILE, "w") as f:
                json.dump({"market": "otc"}, f)
            self.assertEqual(bot.load_settings()["market"], "otc")
        finally:
            bot.SETTINGS_FILE = orig

    def test_settings_gate_mode_default_and_normalize(self):
        tmp = tempfile.mkdtemp()
        orig = bot.SETTINGS_FILE
        orig_env = os.environ.pop("GATE_MODE", None)
        bot.SETTINGS_FILE = os.path.join(tmp, "pocket_settings.json")
        try:
            self.assertEqual(bot.load_settings()["gate_mode"], "strict")
            with open(bot.SETTINGS_FILE, "w") as f:
                json.dump({"gate_mode": "point"}, f)
            self.assertEqual(bot.load_settings()["gate_mode"], "point")
            with open(bot.SETTINGS_FILE, "w") as f:
                json.dump({"gate_mode": "bogus"}, f)
            self.assertEqual(bot.load_settings()["gate_mode"], "strict")
            os.environ["GATE_MODE"] = "point"
            with open(bot.SETTINGS_FILE, "w") as f:
                json.dump({}, f)
            self.assertEqual(bot.load_settings()["gate_mode"], "point")
        finally:
            bot.SETTINGS_FILE = orig
            os.environ.pop("GATE_MODE", None)
            if orig_env is not None:
                os.environ["GATE_MODE"] = orig_env

    def test_scan_text_empty(self):
        self.assertIn("No pairs", bot.scan_text([], 60, "otc"))


class TestAcceptRegistry(unittest.TestCase):
    def test_remember_signal_stores_context(self):
        bot.PENDING_SIGNALS.clear()
        bot.remember_signal(7, {"signal": "CALL"}, "EURUSD-OTC", 60, 1.105)
        ctx = bot.PENDING_SIGNALS[7]
        self.assertEqual(ctx["signal"], "CALL")
        self.assertEqual(ctx["pair"], "EURUSD-OTC")
        self.assertEqual(ctx["period"], 60)
        bot.PENDING_SIGNALS.clear()

    def test_remember_signal_ignores_empty_mid(self):
        bot.PENDING_SIGNALS.clear()
        bot.remember_signal(0, {"signal": "PUT"}, "EURUSD-OTC", 60, 1.1)
        self.assertEqual(bot.PENDING_SIGNALS, {})

    def test_registry_is_bounded(self):
        bot.PENDING_SIGNALS.clear()
        for i in range(1, 260):
            bot.remember_signal(i, {"signal": "CALL"}, "EURUSD-OTC", 60, 1.0)
        self.assertLessEqual(len(bot.PENDING_SIGNALS), 200)
        bot.PENDING_SIGNALS.clear()


class TestAcceptButton(unittest.TestCase):
    def test_payload_fits_telegram_limit_and_parses(self):
        kb = bot.kb_signal_ctx("BCHNZD-OTC", "SELL", 60, 1700000000)
        cb = kb[0][0]["callback_data"]
        self.assertLessEqual(len(cb.encode("utf-8")), 64)
        self.assertTrue(cb.startswith("acc:"))
        parts = cb.split(":")
        self.assertEqual(parts[1], "BCHNZD-OTC")
        self.assertEqual(parts[2], "SELL")
        self.assertEqual(parts[3], "60")
        self.assertEqual(parts[4], "1700000000")

    def test_long_pair_drops_timestamp_to_fit(self):
        pair = "VERYLONGPAIRNAME-OTC"
        cb = bot.kb_signal_ctx(pair, "CALL", 60, 1700000000)[0][0]["callback_data"]
        self.assertLessEqual(len(cb.encode("utf-8")), 64)


class TestThrottleRetry(unittest.TestCase):
    def test_throttle_nonblocking_window(self):
        import pocket_payout as pp
        pp._RATE_LIMITS["__test__"] = (2, 60.0)
        pp._HITS.pop("__test__", None)
        self.assertTrue(pp._throttle("__test__", block=False))
        self.assertTrue(pp._throttle("__test__", block=False))
        self.assertFalse(pp._throttle("__test__", block=False))
        pp._HITS.pop("__test__", None)
        pp._RATE_LIMITS.pop("__test__", None)

    def test_fetch_candles_nonblocking_empty_when_full(self):
        import pocket_payout as pp
        orig = pp._RATE_LIMITS.get("candles")
        try:
            pp._RATE_LIMITS["candles"] = (1, 60.0)
            pp._HITS["candles"] = pp.collections.deque([time.time()])
            self.assertEqual(
                pp._fetch_candles(None, "EURUSD", 60, 10, block=False), [])
        finally:
            if orig:
                pp._RATE_LIMITS["candles"] = orig
            pp._HITS.pop("candles", None)

    def test_retry_returns_value_and_swallows(self):
        import pocket_payout as pp
        self.assertEqual(pp._retry(lambda: 42, tries=2, base=0), 42)

        def boom():
            raise RuntimeError("x")
        self.assertIsNone(pp._retry(boom, tries=2, base=0))


class TestResultNotify(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._orig_results = bot.RESULTS_FILE
        self._orig_pp = bot.pocket_payout
        self._orig_send = bot.send
        bot.RESULTS_FILE = os.path.join(self._tmp, "pending_results.json")
        self.sent = []

        def fake_send(text, chat_id=None, keyboard=None, html_on=False):
            self.sent.append((chat_id, text))

        class FakePP:
            @staticmethod
            def _norm(p):
                return p.replace("/", "").upper()

            @staticmethod
            def account_balance():
                return 1000.0

        bot.send = fake_send
        bot.pocket_payout = FakePP

    def tearDown(self):
        bot.RESULTS_FILE = self._orig_results
        bot.pocket_payout = self._orig_pp
        bot.send = self._orig_send

    def _backdate(self, secs=120):
        items = bot.results_load()
        for p in items:
            p["placed_ts"] = time.time() - secs
        bot.results_save(items)

    def test_ticket_match_notifies_once(self):
        bot.note_placed(42, "abc123", "EURUSD-OTC", "CALL", 60)
        self.assertEqual(len(bot.results_load()), 1)
        self._backdate()
        deal = {"id": "abc123", "asset": "EURUSD-OTC", "win": True,
                "refund": False, "profit": 0.92, "open_ts": 1000,
                "close_ts": int(time.time()), "tf": 60}
        bot._notify_result(deal)
        self.assertEqual(bot.results_load(), [])
        self.assertEqual(len(self.sent), 1)
        chat, text = self.sent[0]
        self.assertEqual(chat, "42")
        self.assertIn("WIN", text)
        self.assertIn("+0.92", text)
        self.assertIn("Balance", text)
        bot._notify_result(deal)
        self.assertEqual(len(self.sent), 1)

    def test_fallback_pair_time_and_refund(self):
        bot.note_placed(7, "", "AUDCAD-OTC", "PUT", 60)
        self._backdate()
        placed = bot.results_load()[0]["placed_ts"]
        deal = {"id": "unrelated", "asset": "AUDCAD-OTC", "win": False,
                "refund": True, "profit": 0.0, "open_ts": placed + 3,
                "close_ts": int(time.time()), "tf": 60}
        bot._notify_result(deal)
        self.assertEqual(bot.results_load(), [])
        self.assertIn("REFUND", self.sent[0][1])

    def test_unrelated_deal_ignored(self):
        bot.note_placed(9, "t1", "GBPUSD-OTC", "CALL", 60)
        bot._notify_result({"id": "t2", "asset": "USDJPY-OTC",
                            "win": True, "refund": False, "profit": 0.9,
                            "open_ts": 0, "tf": 60})
        self.assertEqual(len(bot.results_load()), 1)
        self.assertEqual(self.sent, [])


class TestReportAge(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.orig = bot.EDGE_REPORT_FILE
        bot.EDGE_REPORT_FILE = os.path.join(self.tmp, "edge_report.json")
        bot._EDGE_REPORT.update({"mtime": None, "generated": None, "data": {}})

    def tearDown(self):
        bot.EDGE_REPORT_FILE = self.orig
        bot._EDGE_REPORT.update({"mtime": None, "generated": None, "data": {}})

    def _write(self, days_old):
        gen = time.strftime("%Y-%m-%d %H:%M:%S",
                            time.localtime(time.time() - days_old * 86400))
        with open(bot.EDGE_REPORT_FILE, "w") as f:
            json.dump({"generated": gen,
                       "pairs": {"EURUSD#60": {"bets": 200,
                                              "expectancy": 0.02}}}, f)

    def test_stale_report_ignored(self):
        self._write(10)
        self.assertIsNone(bot.edge_from_report("EURUSD", 60))

    def test_fresh_report_used(self):
        self._write(1)
        rep = bot.edge_from_report("EURUSD", 60)
        self.assertIsNotNone(rep)
        self.assertLess(rep["_age_days"], 2)

    def test_reason_text(self):
        txt = bot.reason_text([{"tradeable": False, "reason": "wilson"},
                               {"tradeable": False, "reason": "wilson"},
                               {"tradeable": False, "reason": "edge"},
                               {"tradeable": True, "reason": "ok"}])
        self.assertIn("thin floor 2", txt)
        self.assertIn("edge too small 1", txt)


class TestTradingToggle(unittest.TestCase):
    def test_env_toggle(self):
        orig = os.environ.get("ALLOW_TRADING")
        try:
            os.environ.pop("ALLOW_TRADING", None)
            self.assertFalse(bot.trading_on())
            os.environ["ALLOW_TRADING"] = "1"
            self.assertTrue(bot.trading_on())
        finally:
            if orig is None:
                os.environ.pop("ALLOW_TRADING", None)
            else:
                os.environ["ALLOW_TRADING"] = orig

    def test_settings_persist_allow_trading(self):
        tmp = tempfile.mkdtemp()
        orig_file = bot.SETTINGS_FILE
        orig_env = os.environ.pop("ALLOW_TRADING", None)
        bot.SETTINGS_FILE = os.path.join(tmp, "pocket_settings.json")
        try:
            self.assertFalse(bot.load_settings()["allow_trading"])
            with open(bot.SETTINGS_FILE, "w") as f:
                json.dump({"allow_trading": True}, f)
            self.assertTrue(bot.load_settings()["allow_trading"])
            os.environ["ALLOW_TRADING"] = "1"
            with open(bot.SETTINGS_FILE, "w") as f:
                json.dump({}, f)
            self.assertTrue(bot.load_settings()["allow_trading"])
        finally:
            bot.SETTINGS_FILE = orig_file
            if orig_env is None:
                os.environ.pop("ALLOW_TRADING", None)
            else:
                os.environ["ALLOW_TRADING"] = orig_env


class TestSafety(unittest.TestCase):
    def test_place_trade_disabled_by_default(self):
        import pocket_payout
        os.environ.pop("ALLOW_TRADING", None)
        ok, msg = pocket_payout.place_trade("EURUSD", "CALL", 1, 60)
        self.assertFalse(ok)
        self.assertIn("disabled", msg.lower())


class TestRuleSetup(unittest.TestCase):
    """The confluence rule fires exactly on a confirmed setup, else None."""

    @staticmethod
    def _series(vals, n):
        import pandas as pd
        base = pd.Series([1.0] * n)
        out = {key: pd.Series([1.0] * n)
               for key in ("s", "high", "low", "lower2", "upper2", "lower3",
                           "upper3", "stoch_k", "stoch_d", "cci", "adx")}
        out.update({k: pd.Series(v) for k, v in vals.items()})
        return out

    def test_flat_market_has_no_setup(self):
        i = bot.WARMUP_BARS + 4
        series = self._series({}, i + 1)
        self.assertIsNone(bot.decide_at(series, i,
                                        bot.MODEL_PARAMS["engine"]))

    def _call_series(self, adx=10.0):
        i = bot.WARMUP_BARS + 4
        n = i + 1
        return self._series({
            "lower2": [1.0] * (n - 1) + [1.0005],  # band tagged by price
            "upper2": [1.2] * n, "upper3": [1.5] * n,
            "lower3": [1.0] * (n - 1) + [0.995],   # not a falling knife
            "stoch_k": [5.0] * (n - 1) + [12.0],   # oversold, hooking up
            "stoch_d": [7.0] * n,
            "cci": [-150.0] * (n - 1) + [-120.0],  # <= -100, turning up
            "adx": [adx] * n,                       # ranging vs runaway trend
            "low": [1.0] * (n - 1) + [0.9995],     # wick pierces the band
        }, n), i

    def _put_series(self, adx=10.0):
        i = bot.WARMUP_BARS + 4
        n = i + 1
        return self._series({
            "upper2": [1.0] * (n - 1) + [0.9995],  # band tagged by price
            "lower2": [0.8] * n, "lower3": [0.5] * n,
            "upper3": [1.0] * (n - 1) + [1.005],   # not a rocket close
            "stoch_k": [95.0] * (n - 1) + [88.0],  # overbought, hooking down
            "stoch_d": [90.0] * n,
            "cci": [150.0] * (n - 1) + [120.0],    # >= 100, turning down
            "adx": [adx] * n,
            "high": [1.0] * (n - 1) + [1.0005],    # wick pierces the band
        }, n), i

    def test_call_setup(self):
        series, i = self._call_series()
        self.assertEqual(bot.decide_at(series, i,
                                       bot.MODEL_PARAMS["engine"]), "CALL")

    def test_strong_trend_blocks_reversal(self):
        series, i = self._call_series(adx=40.0)
        self.assertIsNone(bot.decide_at(series, i,
                                        bot.MODEL_PARAMS["engine"]))

    def test_put_setup(self):
        series, i = self._put_series()
        self.assertEqual(bot.decide_at(series, i,
                                       bot.MODEL_PARAMS["engine"]), "PUT")


class TestRiskStop(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.orig = bot.RISK_FILE
        bot.RISK_FILE = os.path.join(self.tmp, "risk_state.json")
        self._loss, self._cap = bot.DAILY_MAX_LOSS, bot.DAILY_MAX_TRADES
        bot.DAILY_MAX_LOSS = 10.0
        bot.DAILY_MAX_TRADES = 3

    def tearDown(self):
        bot.RISK_FILE = self.orig
        bot.DAILY_MAX_LOSS, bot.DAILY_MAX_TRADES = self._loss, self._cap

    def test_fresh_day_not_blocked(self):
        self.assertFalse(bot.risk_daily_block())

    def test_loss_stop_blocks(self):
        bot.risk_note_outcome(-11.0)
        self.assertTrue(bot.risk_daily_block())

    def test_order_cap_blocks(self):
        for _ in range(bot.DAILY_MAX_TRADES):
            bot.risk_note_outcome(0.95)
        self.assertTrue(bot.risk_daily_block())

    def test_profit_keeps_gate_open(self):
        bot.risk_note_outcome(0.95)
        bot.risk_note_outcome(-0.5)
        self.assertFalse(bot.risk_daily_block())


class TestAccountSync(unittest.TestCase):
    """End-to-end coverage of sync_account_trades (regression: a dropped
    `new_seen = []` initializer raised NameError, only visible as a print)."""

    def setUp(self):
        tmp = tempfile.mkdtemp()
        self.origs = (bot.ML_DATA_FILE, bot.TRACK_FILE, bot.RESULTS_FILE,
                      bot.pocket_payout, bot.send)
        bot.ML_DATA_FILE = os.path.join(tmp, "ml_data.json")
        bot.TRACK_FILE = os.path.join(tmp, "track.json")
        bot.RESULTS_FILE = os.path.join(tmp, "results.json")
        bot.send = lambda *a, **k: None

        self.deal = {"id": "d1", "asset": "EURUSD-OTC", "entry": 1.1000,
                     "exit": 1.1050, "profit": 0.85, "win": True,
                     "refund": False, "demo": True,
                     "open_ts": int(time.time()) - 120,
                     "close_ts": int(time.time()) - 60, "tf": 60}

        class FakePP:
            @staticmethod
            def _norm(p):
                return str(p).upper().replace("/", "")

            @staticmethod
            def account_trades(max_tries=1):
                return [dict(self.deal)]

        bot.pocket_payout = FakePP
        # keep the sync from hitting any network path
        bot._acct_raw = lambda asset, ts0: []

    def tearDown(self):
        (bot.ML_DATA_FILE, bot.TRACK_FILE, bot.RESULTS_FILE,
         bot.pocket_payout, bot.send) = self.origs

    def test_sync_records_deal_without_error(self):
        try:
            bot.sync_account_trades()
        except Exception as e:
            self.fail(f"sync_account_trades raised: {e!r}")
        d = bot._ml_data_load()
        self.assertIn("d1", [str(x) for x in (d.get("seen_deals") or [])])
        tr = bot.track_load()
        accts = [x for x in tr["settled"] if x.get("source") == "account"]
        self.assertEqual(len(accts), 1)
        self.assertTrue(accts[0]["win"])

    def test_sync_is_idempotent(self):
        bot.sync_account_trades()
        bot.sync_account_trades()
        d = bot._ml_data_load()
        self.assertEqual([str(x) for x in (d.get("seen_deals") or [])],
                         ["d1"])
        accts = [x for x in bot.track_load()["settled"]
                 if x.get("source") == "account"]
        self.assertEqual(len(accts), 1)


if __name__ == "__main__":
    unittest.main()
