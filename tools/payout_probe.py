"""
Payout diagnostics for BinaryOptionsToolsV2.

    python tools/payout_probe.py            # uses pocket_ssid.json / PO_SSID
    python tools/payout_probe.py EURUSD_otc

Prints the raw shape of `api.payout()` and the payout field inside
`api.active_assets()`, so a server that reports "payout n/a" can be compared
with a machine where payouts work. Never prints the ssid.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import po_feed


def _brief(v, depth=0):
    if depth > 2:
        return type(v).__name__
    if isinstance(v, dict):
        items = list(v.items())[:8]
        return {str(k): _brief(x, depth + 1) for k, x in items}
    if isinstance(v, (list, tuple)):
        return [_brief(x, depth + 1) for x in v[:8]]
    return v


def main() -> int:
    sym = sys.argv[1] if len(sys.argv) > 1 else "EURUSD_otc"
    try:
        import BinaryOptionsToolsV2 as bot
        try:
            from importlib.metadata import version
            lib_ver = version("BinaryOptionsToolsV2")
        except Exception:
            lib_ver = getattr(bot, "__version__", "?")
    except Exception:
        lib_ver = "IMPORT FAILED"
    print(f"python     : {sys.version.split()[0]}")
    print(f"pandas     : {pd.__version__}")
    print(f"library    : {lib_ver}")

    t = time.monotonic()
    ok = po_feed.warm()
    print(f"warm       : {ok} in {time.monotonic() - t:.1f}s")
    api = po_feed._CLIENT.get("api")
    if api is None:
        print("no client - check ssid")
        return 1

    print(f"connected  : {po_feed._connected(api)}")

    # 1) bulk payout map
    t = time.monotonic()
    try:
        raw = api.payout()
        print(f"payout()   : {time.monotonic() - t:.1f}s type={type(raw).__name__} "
              f"len={len(raw) if hasattr(raw, '__len__') else 'n/a'}")
        print("  sample   :", _brief(raw))
        print("  normalized:", _brief(po_feed._normalize_payouts(raw)))
    except Exception as exc:
        print(f"payout()   : RAISED in {time.monotonic() - t:.1f}s "
              f"{type(exc).__name__}: {exc}")
        traceback.print_exc()

    # 2) per-asset payout
    t = time.monotonic()
    try:
        one = api.payout(sym)
        print(f"payout({sym}): {time.monotonic() - t:.1f}s -> {one!r} "
              f"({type(one).__name__})")
    except Exception as exc:
        print(f"payout({sym}): RAISED {type(exc).__name__}: {exc}")

    # 3) payout field from the active asset list (independent source)
    try:
        active = api.active_assets() or []
        match = [a for a in active if str(a.get("symbol", "")).lower() == sym.lower()]
        print(f"active     : {len(active)} assets")
        if match:
            a = match[0]
            print("  entry    :", json.dumps({k: a.get(k) for k in
                                              ("symbol", "payout", "is_active",
                                               "is_otc", "allowed_candles")},
                                             default=str))
        else:
            print(f"  {sym} not in active list")
    except Exception as exc:
        print(f"active     : RAISED {type(exc).__name__}: {exc}")

    print("payout_for :", po_feed.payout_for("EUR/USD"))
    print("status     :", po_feed.status())
    po_feed.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
