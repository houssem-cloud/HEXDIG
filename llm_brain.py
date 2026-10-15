"""
Free-tier LLM trading brain for the signal bot.
Chains Groq (Llama 3.3 70B) -> Gemini Flash -> OpenRouter free models,
using each provider's free quota, then falls back silently.

Keys (never commit these):
  python3 llm_brain.py --set-key groq "GROQ_API_KEY"
  python3 llm_brain.py --set-key gemini "GEMINI_API_KEY"
  python3 llm_brain.py --set-key openrouter "OPENROUTER_API_KEY"
  python3 llm_brain.py --status
  python3 llm_brain.py --test
Or env vars: LLM_GROQ_KEY / LLM_GEMINI_KEY / LLM_OPENROUTER_KEY
Free keys: console.groq.com  ·  aistudio.google.com  ·  openrouter.ai
"""

import json
import os
import re
import time

import requests

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_config.json")
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_state.json")

PROVIDERS = [
    {"name": "groq", "env": "LLM_GROQ_KEY",
     "url": "https://api.groq.com/openai/v1/chat/completions",
     "model": "llama-3.3-70b-versatile", "daily_cap": 50},
    {"name": "gemini", "env": "LLM_GEMINI_KEY",
     "url": ("https://generativelanguage.googleapis.com/v1beta/models/"
             "gemini-2.0-flash:generateContent"),
     "model": "gemini-2.0-flash", "daily_cap": 100},
    {"name": "openrouter", "env": "LLM_OPENROUTER_KEY",
     "url": "https://openrouter.ai/api/v1/chat/completions",
     "model": "meta-llama/llama-3.3-70b-instruct:free", "daily_cap": 25},
]

DAILY_BUDGET = 100
MIN_GAP_SEC = 12

SYSTEM_PROMPT = (
    "You are a binary-options market analyst. You get compact market context "
    "and reply with ONLY compact JSON, no markdown:\n"
    '{"signal":"CALL|PUT|WAIT","confidence":0-100,"reason":"max 8 words"}\n'
    "CALL = price will be higher at expiry, PUT = lower, WAIT = no edge. "
    "Prefer WAIT when the picture is mixed. Strong confluence + clean setup "
    "justifies confidence above 70."
)


def _load_cfg():
    try:
        with open(CONFIG_FILE) as f:
            d = json.load(f)
            return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _save_cfg(cfg):
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(cfg, f)
        try:
            os.chmod(CONFIG_FILE, 0o600)
        except Exception:
            pass
        return True
    except Exception as e:
        print("llm config save failed:", e)
        return False


def set_key(provider, key):
    cfg = _load_cfg()
    cfg[f"{provider}_key"] = str(key).strip()
    return _save_cfg(cfg)


def _keys():
    cfg = _load_cfg()
    out = {}
    for p in PROVIDERS:
        k = (os.environ.get(p["env"]) or cfg.get(f"{p['name']}_key") or "").strip()
        if k:
            out[p["name"]] = k
    return out


def _state_load():
    today = time.strftime("%Y-%m-%d")
    try:
        with open(STATE_FILE) as f:
            d = json.load(f)
        if isinstance(d, dict) and d.get("date") == today:
            d.setdefault("by_provider", {})
            d.setdefault("last_call", {})
            return d
    except Exception:
        pass
    return {"date": today, "by_provider": {}, "last_call": {}, "total": 0}


def _state_save(d):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(d, f)
    except Exception:
        pass


def _build_prompt(ctx):
    c = ctx or {}
    closes = c.get("closes") or []
    parts = [
        f"pair={c.get('pair')} tf={c.get('tf')}s price={c.get('price')} "
        f"payout={c.get('payout')}",
        f"closes={closes}",
        f"rsi={c.get('rsi')} regime={c.get('regime')} vol={c.get('vol_pct')}",
        f"price_action={c.get('price_action')} tf_confluence={c.get('tf_confluence')}",
        f"my_signals_last10={c.get('signals_last10')} "
        f"my_account_recent={c.get('account_recent')}",
        "Reply with the JSON only.",
    ]
    return "\n".join(parts)


def _parse_json(text):
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    sig = str(d.get("signal", "")).upper().strip()
    if sig not in ("CALL", "PUT", "WAIT"):
        return None
    try:
        conf = max(0.0, min(100.0, float(d.get("confidence"))))
    except Exception:
        conf = 50.0
    return {"signal": sig, "confidence": conf,
            "reason": str(d.get("reason") or "")[:80]}


def _call_openai_style(url, key, model, prompt, timeout):
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"},
        json={"model": model, "temperature": 0.2, "max_tokens": 200,
              "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                           {"role": "user", "content": prompt}]},
        timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


def _call_gemini(url, key, prompt, timeout):
    r = requests.post(
        f"{url}?key={key}",
        json={"systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
              "contents": [{"role": "user", "parts": [{"text": prompt}]}],
              "generationConfig": {"temperature": 0.2, "maxOutputTokens": 200}},
        timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


def analyze(ctx):
    """Ask the free LLM chain for a CALL/PUT/WAIT decision.
    Returns {"signal","confidence","reason","provider"} or None."""
    keys = _keys()
    if not keys:
        return None
    st = _state_load()
    if st.get("total", 0) >= DAILY_BUDGET:
        return None
    prompt = _build_prompt(ctx)
    for p in PROVIDERS:
        key = keys.get(p["name"])
        if not key:
            continue
        used = int(st["by_provider"].get(p["name"], 0))
        if used >= p["daily_cap"]:
            continue
        last = float(st["last_call"].get(p["name"], 0))
        if time.time() - last < MIN_GAP_SEC:
            continue
        try:
            if p["name"] == "gemini":
                text = _call_gemini(p["url"], key, prompt, 25)
            else:
                text = _call_openai_style(p["url"], key, p["model"], prompt, 25)
            parsed = _parse_json(text)
            if not parsed:
                continue
            parsed["provider"] = p["name"]
            parsed["model"] = p["model"]
            st["by_provider"][p["name"]] = used + 1
            st["last_call"][p["name"]] = time.time()
            st["total"] = int(st.get("total", 0)) + 1
            _state_save(st)
            return parsed
        except Exception as e:
            print(f"LLM {p['name']} failed: {e}")
            continue
    return None


def status():
    keys = _keys()
    st = _state_load()
    return {
        "keys": {p["name"]: ("set" if p["name"] in keys else "missing")
                 for p in PROVIDERS},
        "used_today": st.get("total", 0),
        "by_provider": st.get("by_provider", {}),
        "budget": DAILY_BUDGET,
    }


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if len(args) >= 3 and args[0] == "--set-key":
        ok = set_key(args[1], args[2])
        print("saved." if ok else "FAILED.")
    elif args and args[0] == "--status":
        print(json.dumps(status(), indent=2))
    elif args and args[0] == "--test":
        demo = {"pair": "EURUSD-OTC", "tf": 60, "price": 1.13525, "payout": 92,
                "closes": [1.1350, 1.1351, 1.1352, 1.1353, 1.1352, 1.1354,
                           1.1355, 1.1353, 1.1352, 1.1354, 1.1356, 1.1355,
                           1.1357, 1.1358, 1.1356, 1.1355, 1.1357, 1.1359,
                           1.1360, 1.1358, 1.1357, 1.1359, 1.1361, 1.1360,
                           1.1362],
                "rsi": 58.0, "regime": "squeeze", "vol_pct": 40,
                "price_action": 1.5, "tf_confluence": 0.66,
                "signals_last10": "6W/4L", "account_recent": "52% (40)"}
        r = analyze(demo)
        print("RESULT:", json.dumps(r, indent=2) if r else
              "no provider answered (check keys / free quota)")
    else:
        print("Use: --set-key <groq|gemini|openrouter> <key> | --status | --test")
