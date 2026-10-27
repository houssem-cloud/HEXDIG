"""
Pocket Option ssid probe v2 — handles socket.io binary-event frames.

Run on the AWS server:
    sudo /opt/pocketoption-signal-bot/venv/bin/python tools/po_probe.py "$(cat /tmp/ssid.txt)"
"""
import asyncio
import json
import sys

sys.path.insert(0, ".")

from engine.pocketoption_ws import ENDPOINTS, WS_HEADERS, normalize_ssid


def to_text(raw):
    if isinstance(raw, (bytes, bytearray)):
        t = raw.decode("utf-8", "ignore")
        return t, f"BINARY {len(raw)}B: {t[:150]}"
    return raw, raw[:150]


async def probe(ssid: str, frames_to_show: int = 12):
    import websockets
    auth = normalize_ssid(ssid)
    results = []
    for url in ENDPOINTS:
        host = url.split("/")[2].split("?")[0]
        print("=" * 70)
        print("ENDPOINT:", host)
        try:
            async with websockets.connect(url, open_timeout=10,
                                          additional_headers=WS_HEADERS) as ws:
                print("  TLS connect ......... OK")
                await ws.send("40")
                sent_auth = False
                streamed = 0
                outcome = None
                for i in range(frames_to_show):
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=6)
                    except asyncio.TimeoutError:
                        print("  frame: TIMEOUT")
                        break
                    text, shown = to_text(raw)
                    print(f"  frame {i}: {shown}")

                    t = text if isinstance(text, str) else str(text)
                    if isinstance(raw, bytes):
                        streamed += 1                       # binary payload arrived
                        continue
                    if t.startswith("2"):                   # engine.io ping
                        await ws.send("3")
                        continue
                    if t.startswith("41"):
                        print(f"\n  RESULT {host}: SESSION-REJECTED (41) — ssid expired/"
                              f"truncated or device/IP-bound")
                        outcome = "SESSION-REJECTED"
                        break
                    if not sent_auth and t.startswith(("0", "40")):
                        frame = auth if auth.startswith("42") else "42" + auth
                        await ws.send(frame)
                        sent_auth = True
                        print("  auth ................ sent")
                        continue
                    if '"successauth"' in t:
                        print("  auth confirmed by server")
                        continue
                    if '"_placeholder"' in t:
                        try:
                            ev = json.loads(t[4:].lstrip("-"))[0] if t.startswith("451-") else "?"
                        except Exception:
                            ev = "?"
                        print(f"  (binary event '{ev}' — REAL payload is the NEXT frame)")
                        continue
                    streamed += 1                          # any live text event counts

                if outcome is None:
                    if sent_auth and streamed >= 1:
                        outcome = "STREAMING"
                        print(f"\n  RESULT {host}: PASS — server accepted auth and streamed "
                              f"{streamed} data frame(s)")
                    elif sent_auth:
                        outcome = "CONNECTED-BUT-QUIET"
                        print(f"\n  RESULT {host}: auth ok but no data frames —")
                        print("           subscriptions may need a different event name")
                    else:
                        outcome = "NO-HANDSHAKE"
                results.append((host, outcome))
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {e}")
            results.append((host, f"FAIL:{type(e).__name__}"))

    print("=" * 70)
    print("SUMMARY")
    for host, r in results:
        print(f"  {host:<28} {r}")
    print("Where to go next:")
    print("  STREAMING            -> sudo systemctl restart po-bot; then /status")
    print("  SESSION-REJECTED     -> re-capture ssid from F12, paste it via a file,")
    print("                          or the session is bound to the capture device/IP")
    print("  CONNECTED-BUT-QUIET  -> paste the frame list so the tick event can be mapped")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python tools/po_probe.py '<full 42[\"auth\",{...}] ssid frame>'")
        sys.exit(1)
    asyncio.run(probe(sys.argv[1]))
