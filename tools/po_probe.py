"""
Pocket Option ssid connectivity probe — run this on the AWS server to see exactly
where the websocket handshake/auth fails:

    sudo /opt/pocketoption-signal-bot/venv/bin/python tools/po_probe.py "42[\"auth\",{...}]"
        (paste the full ssid frame as the single argument)

It tries every known endpoint, performs the socket.io handshake + auth, prints the
first raw frames the server sends back, and prints PASS/FAIL per endpoint.
No engine code is touched — pure diagnostics.
"""
import asyncio
import sys

sys.path.insert(0, ".")

from engine.pocketoption_ws import ENDPOINTS, normalize_ssid


async def probe(ssid: str, frames_to_show: int = 6):
    import websockets
    auth = normalize_ssid(ssid)
    results = []
    for url in ENDPOINTS:
        host = url.split("/")[2].split("?")[0]
        print("=" * 70)
        print("ENDPOINT:", host)
        try:
            async with websockets.connect(url, open_timeout=10,
                                          additional_headers={"Origin": "https://pocketoption.com"}) as ws:
                print("  TLS connect ......... OK")
                await ws.send("40")
                print("  socket.io handshake . sent 40")
                got_handshake = False
                for i in range(frames_to_show):
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=5)
                    except asyncio.TimeoutError:
                        print(f"  frame {i}: TIMEOUT")
                        break
                    head = raw if len(raw) <= 160 else raw[:160] + f"  (+{len(raw)} bytes)"
                    print(f"  frame {i}: {head}")
                    if raw.startswith(("40", "451")):
                        got_handshake = True
                        await ws.send("42" + auth if not auth.startswith("42") else auth)
                        print("  auth ................ sent")
                    if '"successauth"' in raw or '"session"' in raw:
                        print(f"\n  RESULT {host}: PASS — session accepted")
                        results.append((host, "PASS"))
                        break
                    if "error" in raw.lower() or "can't" in raw.lower():
                        print(f"\n  RESULT {host}: REJECTED — server error frame above")
                        results.append((host, "REJECTED"))
                        break
                else:
                    results.append((host, "NO-AUTH-REPLY"))
                if not got_handshake:
                    results.append((host, "NO-HANDSHAKE"))
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {e}")
            results.append((host, f"FAIL:{type(e).__name__}"))

    print("=" * 70)
    print("SUMMARY")
    for host, r in results:
        print(f"  {host:<28} {r}")
    print("Interpretation:")
    print("  FAIL:*  on every endpoint  -> AWS datacenter IP likely blocked by PO")
    print("           (try again from a home IP, or use a PO-allowed region)")
    print("  NO-AUTH-REPLY             -> handshake ok, event names may have")
    print("                              changed; send the frames above for tuning")
    print("  REJECTED                  -> ssid expired/malformed; re-capture from F12")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python tools/po_probe.py '<full 42[\"auth\",{...}] ssid frame>'")
        sys.exit(1)
    asyncio.run(probe(sys.argv[1]))
