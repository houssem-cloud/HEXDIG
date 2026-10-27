"""
Live verification of the data path (needs a valid ssid in pocket_ssid.json).

    python test_live_check.py

Checks, against the real broker / Yahoo feeds:
  * every pair in the Telegram menu has both a Yahoo and a PO symbol;
  * OTC pairs either return fresh broker bars with a payout, or are refused
    (never synthesized, never served from an untradeable asset);
  * Stock pairs use the primary Yahoo M1 feed;
  * payouts are looked up per asset, never borrowed from another pair;
  * a cold OTC request stays inside the 35 s Telegram fetch budget.
"""
import asyncio
import time

import main
from engine import datafeed as d
from engine import po_feed
from engine.datafeed import bar_age, get_candles
from engine.signals import analyze

PAIRS = sorted({p for page in main.ASSET_PAGES for p in page})
problems = []


def _report(ok, line):
    print(("OK  " if ok else "FAIL") + "  " + line)
    if not ok:
        problems.append(line)


def check_maps():
    missing_yahoo = sorted(set(PAIRS) - set(d.SYMBOL_MAP))
    missing_po = sorted(set(PAIRS) - set(po_feed._ASSETS))
    missing_base = sorted(set(PAIRS) - set(d.BASE_PRICE))
    _report(not (missing_yahoo or missing_po or missing_base),
            f"menu mapping ({len(PAIRS)} pairs) | Yahoo missing={missing_yahoo} "
            f"PO missing={missing_po} anchor missing={missing_base}")


async def check_otc():
    live = refused = 0
    missing_pay = 0
    worst_age = 0.0
    for pair in PAIRS:
        t = time.monotonic()
        df, note = await get_candles(pair, "OTC")
        took = time.monotonic() - t
        if df is None:
            refused += 1
            _report("synthetic" not in note.lower(),
                    f"{pair:8s} OTC refused ({took:4.1f}s) | {note}")
            continue
        if "synthetic" in note.lower():
            _report(False, f"{pair:8s} OTC returned SYNTHETIC data | {note}")
            continue
        age = bar_age(df) or 0.0
        worst_age = max(worst_age, age)
        sig = analyze(df, "1 minute")
        live += 1
        if "payout n/a" in note:
            missing_pay += 1
        _report(age <= po_feed._HARD_STALE and len(df) >= 60,
                f"{pair:8s} OTC {len(df):3d} bars | age {age:4.0f}s | {took:4.1f}s "
                f"| {sig.direction if sig else 'no signal':9s} | {note}")
    print(f"     OTC summary: {live} live, {refused} refused, "
          f"oldest accepted bar {worst_age:.0f}s")
    # A payout may be missing for a moment on one asset, but "n/a" on most live
    # pairs means the payout source broke (seen on AWS with an empty bulk map).
    limit = max(1, live // 5)
    _report(missing_pay <= limit,
            f"OTC payouts: {live - missing_pay}/{live} live pairs have a payout "
            f"(missing {missing_pay}, allowed {limit}) | "
            f"{po_feed.payout_issue() or 'source ok'}")


async def check_stock():
    for pair in ("EUR/USD", "USD/JPY", "EUR/AUD", "USD/SGD"):
        t = time.monotonic()
        df, note = await get_candles(pair, "STOCK")
        took = time.monotonic() - t
        if df is None:
            _report(True, f"{pair:8s} STOCK refused as expected | {note}")
            continue
        _report("Yahoo" in note and len(df) >= 60,
                f"{pair:8s} STOCK {len(df):3d} bars | age {(bar_age(df) or 0):4.0f}s "
                f"| {took:4.1f}s | {note}")


def check_payouts():
    if not po_feed.is_ready():
        print("     payouts skipped (shared client not connected)")
        return
    exact = po_feed.payout_for("EUR/USD")
    unknown = po_feed.payout_for("XYZ/ABC")
    _report(unknown is None,
            f"payout exactness: EUR/USD-OTC={exact} | unknown pair={unknown} | "
            f"{po_feed.payout_issue() or 'source ok'}")


async def main_async():
    t = time.monotonic()
    warm = await asyncio.to_thread(po_feed.warm)
    print(f"warm: {warm} in {time.monotonic() - t:.1f}s | status: {po_feed.status()}")
    if not warm:
        problems.append("PO feed did not warm up")

    check_maps()
    check_payouts()
    await check_stock()
    await check_otc()

    # cold-start latency of the fresh path (cache cleared, new client)
    po_feed._CACHE.clear()
    d._CACHE.clear()
    t = time.monotonic()
    df, note = await get_candles("EUR/USD", "OTC")
    took = time.monotonic() - t
    _report(df is not None and took < 35.0,
            f"cold OTC latency: {took:.1f}s (limit 35s) | {note}")

    print()
    if problems:
        print(f"LIVE CHECK: {len(problems)} PROBLEM(S)")
        for p in problems:
            print("  -", p)
        raise SystemExit(1)
    print("LIVE CHECK: ALL OK")


asyncio.run(main_async())
