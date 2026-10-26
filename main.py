"""
Pocket Option Signal Bot — Telegram front-end.

Flow mirrors the reference bot screenshots:
    Welcome -> Select Quote (Stock / OTC) -> Select Asset (3 pages)
            -> Select Time (expiry) -> Pair screen -> "Get a signal"
            -> "Analyzing..." progress -> Higher / Lower + Accuracy.

Strategy inputs match the Pocket Option chart screenshots:
    Bollinger 20/2 + 20/3, Stochastic 6/3/3 SMA, CCI 25, ADX(1,5),
    Heikin Ashi candles, evaluated on the M1 timeframe.
"""
from __future__ import annotations

import asyncio
import logging
import os

from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import (Application, CallbackQueryHandler, CommandHandler,
                          ContextTypes)

from engine.datafeed import get_candles
from engine.signals import analyze

load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("po-bot")

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
ALLOWED_IDS = {int(x) for x in os.getenv("ALLOWED_IDS", "").split(",") if x.strip().isdigit()}

# ---------------------------------------------------------------- playlists
ASSET_PAGES = [
    ["EUR/USD", "USD/JPY", "GBP/USD", "AUD/USD", "USD/CAD", "USD/CHF", "GBP/JPY", "EUR/JPY"],
    ["AUD/CAD", "EUR/CAD", "GBP/CAD", "CAD/CHF", "EUR/GBP", "AUD/JPY", "NZD/USD", "CHF/JPY"],
    ["EUR/CHF", "GBP/AUD", "NZD/JPY", "USD/BRL", "EUR/AUD", "USD/SGD", "GBP/CHF", "EUR/NZD"],
]
TIMES = [("1 min", "1 minute"), ("3 min", "3 minutes"), ("5 min", "5 minutes"),
         ("30 min", "30 minutes"), ("1 hour", "1 hour"), ("4 hours", "4 hours")]

WELCOME = (
    "👋 <b>Welcome!</b>\n\n"
    "Your personal assistant for quick trading signals is now "
    "<b>active</b> and ready to help you spot opportunities in the market.\n\n"
    "Press the button below to begin 🚀"
)

DISCLAIMER = "\n\n<i>Signals are algorithmic estimates — not financial advice.</i>"


def kb(rows: list[list[tuple[str, str]]]) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton(t, callback_data=d) for t, d in r] for r in rows])


# ------------------------------------------------------------------- access
def allowed(update: Update) -> bool:
    if not ALLOWED_IDS:
        return True
    u = update.effective_user
    return bool(u and u.id in ALLOWED_IDS)


# --------------------------------------------------------------- main menu
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    await update.effective_message.reply_text(
        WELCOME,
        parse_mode=ParseMode.HTML,
        reply_markup=kb([[("🚀 Begin", "quote")]]),
    )


async def cb_quote(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    await q.edit_message_text(
        "🈯 <b>Select Quote</b>\n\n"
        "Choose whether you want to trade on <b>Stock</b> or <b>OTC</b> markets:",
        parse_mode=ParseMode.HTML,
        reply_markup=kb([
            [("📈 Stock", "assets:STOCK:0"), ("🧪 OTC", "assets:OTC:0")],
            [("⬅️ Back", "home")],
        ]),
    )


# ------------------------------------------------------------- feed status
async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    from engine.datafeed import feed_health, has_internet

    online = await asyncio.to_thread(has_internet)
    hs = feed_health()
    age = hs.get("live_candle_age") if isinstance(hs, dict) else None

    conn = "🟢 Internet reachable" if online else "🔴 No outbound internet — signals fall back to synthetic"

    if age is None:
        feed_line = "⚪ No live fetch yet — run a Stock signal once"
    elif age < 120:
        feed_line = (f"🟢 Live feed healthy — last M1 pull {age:.0f}s ago "
                     f"({hs.get('last_live_pair', '?')}, {hs.get('last_live_bars', 0)} bars)")
    else:
        feed_line = f"🟡 Last live pull was {age / 60:.1f} min ago ({hs.get('last_live_pair', '?')})"

    fb = int(hs.get("fallback_today", 0) or 0)
    if fb:
        feed_line += f"\n🟡 Synthetic fallbacks used since start: {fb}"

    err = hs.get("last_error", "")
    await update.effective_message.reply_text(
        f"🩺 <b>Feed Status</b>\n\n"
        f"🌐 {conn}\n"
        f"📡 {feed_line}\n"
        + (f"⚠️ Last feed error: <code>{err}</code>\n" if err else "")
        + "\nℹ️ Your Wi-Fi/SSID name is not a data source — the bot reads the internet, "
          "so any connection works.\n\n"
          "📈 <b>Stock pairs</b>: real live 1-min spot candles\n"
          "🧪 <b>OTC pairs</b>: deterministic synthetic feed (Pocket Option's OTC "
          "prices are private and can't be read by any tool).",
        parse_mode=ParseMode.HTML,
    )


async def cb_home(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    await q.edit_message_text(WELCOME, parse_mode=ParseMode.HTML,
                              reply_markup=kb([[("🚀 Begin", "quote")]]))


async def cb_assets(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    _, quote, page_s = q.data.split(":")
    page = int(page_s)
    pairs = ASSET_PAGES[page]
    rows = [[(pairs[i + 1] + (" (OTC)" if quote == "OTC" else ""), f"asset:{quote}:{pairs[i + 1]}"),
             (pairs[i] + (" (OTC)" if quote == "OTC" else ""), f"asset:{quote}:{pairs[i]}")]
            for i in range(0, len(pairs), 2)]
    nav = []
    if page > 0:
        nav.append(("⬅️", f"assets:{quote}:{page - 1}"))
    nav.append((f"{page + 1} of {len(ASSET_PAGES)}", f"noop:{page}"))
    if page < len(ASSET_PAGES) - 1:
        nav.append(("➡️", f"assets:{quote}:{page + 1}"))
    rows.append(nav)
    rows.append([("⬅️ Back", "quote"), ("🚪 Exit to Menu", "home")])

    await q.edit_message_text(
        f"💠 <b>Select Asset</b>\n\nBot recommendation: <b>EUR/USD {quote.capitalize()}</b>\n"
        f"Or choose another currency pair below:",
        parse_mode=ParseMode.HTML,
        reply_markup=kb(rows),
    )


async def cb_asset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    _, quote, pair = q.data.split(":")
    await q.edit_message_text(
        f"💠 <b>Pair: {pair}</b>\n\n"
        "Select the time (signal expiry). Choose how long you want your signal to last:",
        parse_mode=ParseMode.HTML,
        reply_markup=kb([
            [(t, f"time:{quote}:{pair}:{label}") for t, label in (TIMES[0], TIMES[1])],
            [(t, f"time:{quote}:{pair}:{label}") for t, label in (TIMES[2], TIMES[3])],
            [(t, f"time:{quote}:{pair}:{label}") for t, label in (TIMES[4], TIMES[5])],
            [("⬅️ Back", f"assets:{quote}:0"), ("🚪 Exit to Menu", "home")],
        ]),
    )


async def cb_time(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    _, quote, pair, label = q.data.split(":")
    await q.edit_message_text(
        f"💱 <b>{pair}</b>{' 🧪' if quote == 'OTC' else ''}\n<b>{label}</b>\n\n"
        f"💙 Pair: {pair}\n⏱ Time: {label}\n\n"
        "Press the button below to get a signal",
        parse_mode=ParseMode.HTML,
        reply_markup=kb([
            [("📡 Get a signal", f"signal:{quote}:{pair}:{label}")],
            [("⬅️ Back", f"asset:{quote}:{pair}"), ("🚪 Exit to Menu", "home")],
        ]),
    )


# ------------------------------------------------------------ signal flow
PROG_STEPS = [10, 25, 45, 65, 80, 92, 100]
FETCH_TIMEOUT = 25  # seconds


async def cb_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    _, quote, pair, label = q.data.split(":", 3)

    # Start the data fetch NOW so it runs in parallel with the animation
    # (yfinance can take a few seconds — don't make the user wait after 100%).
    fetch = asyncio.create_task(get_candles(pair, quote))

    # "Analyzing the market" progress animation (like the screenshots)
    for step in PROG_STEPS:
        filled = step // 10
        bar = "■" * filled + "□" * (10 - filled)
        try:
            await q.edit_message_text(
                f"👩‍💻 <b>Analyzing…</b>\n\n"
                f"▫️ Analyzing the market  [{bar}] {step}%",
                parse_mode=ParseMode.HTML,
            )
        except Exception:
            pass  # rate-limit / "message not modified" — cosmetic only
        await asyncio.sleep(0.45)

    # Deliver inline — nothing can strand the message at 100%.
    await deliver_signal(context, q, quote, pair, label, fetch)


async def deliver_signal(context: ContextTypes.DEFAULT_TYPE, q, quote: str,
                         pair: str, label: str, fetch: asyncio.Task):
    try:
        try:
            df, note = await asyncio.wait_for(fetch, FETCH_TIMEOUT)
        except asyncio.TimeoutError:
            log.warning("data fetch timeout for %s/%s", quote, pair)
            df, note = None, "Data feed timeout"
        if df is not None and not df.empty:
            sig = analyze(df, expiry_line=label)
        else:
            sig = None
    except Exception as e:                                    # engine never kills the bot
        log.exception("signal failed for %s %s: %s", pair, quote, e)
        sig, note = None, "Analysis error"

    try:
        if sig is None:
            await q.edit_message_text(
                "🎯 <b>SIGNAL UNAVAILABLE</b>\n\n"
                "No trade setup detected right now — the market is unclear for this pair.\n"
                "Try again in a minute or pick another asset.",
                parse_mode=ParseMode.HTML,
                reply_markup=kb([[("🔁 Try again", f"signal:{quote}:{pair}:{label}")],
                                 [("⬅️ Back", f"time:{quote}:{pair}:{label}"),
                                   ("🚪 Exit to Menu", "home")]]),
            )
            return

        head = f"💱 {pair} ({label}) - {'BUY' if sig.direction == 'BUY' else 'SELL'}"
        breakdown = "\n".join(f"   • {r}" for r in sig.reasons)
        warn = ("\n\n⚠️ " + "\n⚠️ ".join(sig.warnings)) if sig.warnings else ""
        text = (
            f"{sig.summary()}\n\n"
            f"<b>{head}</b>\n"
            f"📊 Regime: {sig.regime}\n"
            f"🧩 <b>Confluence breakdown:</b>\n{breakdown}{warn}"
            + (f"\n\n<i>{note}</i>" if note else "")
        )
        markup = kb([[("🆕 New signal", f"signal:{quote}:{pair}:{label}")],
                     [("⬅️ Select another time", f"asset:{quote}:{pair}")],
                     [("🚪 Exit to Menu", "home")]])

        try:
            await q.edit_message_text(text + DISCLAIMER,
                                      parse_mode=ParseMode.HTML, reply_markup=markup)
        except Exception:
            # HTML rejected (e.g. format error / rate limit) → plain-text retry
            import re
            plain = re.sub(r"<[^>]+>", "", text + DISCLAIMER)
            await q.edit_message_text(plain, reply_markup=markup)
    except Exception as deliver_err:
        log.exception("deliver_signal failed: %s", deliver_err)
        try:
            await q.edit_message_text(
                "⚠️ Could not render the signal — press below.",
                reply_markup=kb([[("🔁 Try again", f"signal:{quote}:{pair}:{label}")],
                                 [("🚪 Exit to Menu", "home")]]),
            )
        except Exception:
            log.exception("final edit also failed")


async def cb_noop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.callback_query.answer(f"Page {update.callback_query.data}", show_alert=False)


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    log.error("Unhandled error in update", exc_info=context.error)
    try:
        if isinstance(update, Update) and update.callback_query is not None:
            await update.callback_query.answer("⚠️ Something went wrong — try again.", show_alert=True)
    except Exception:
        pass


def main():
    if not BOT_TOKEN or "paste" in BOT_TOKEN:
        raise SystemExit("Set BOT_TOKEN in .env first (get a token from @BotFather).")

    async def _boot_check(_app: Application):
        from engine.datafeed import has_internet
        ok = await asyncio.to_thread(has_internet)
        log.info("Startup connectivity check: %s",
                 "LIVE FEED REACHABLE ✅" if ok else "NO INTERNET — synthetic fallbacks will be used ❌")

    app = Application.builder().token(BOT_TOKEN).post_init(_boot_check).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CallbackQueryHandler(cb_home, pattern=r"^home$"))
    app.add_handler(CallbackQueryHandler(cb_quote, pattern=r"^quote$"))
    app.add_handler(CallbackQueryHandler(cb_assets, pattern=r"^assets:"))
    app.add_handler(CallbackQueryHandler(cb_asset, pattern=r"^asset:"))
    app.add_handler(CallbackQueryHandler(cb_time, pattern=r"^time:"))
    app.add_handler(CallbackQueryHandler(cb_signal, pattern=r"^signal:"))
    app.add_handler(CallbackQueryHandler(cb_noop, pattern=r"^noop:"))
    app.add_error_handler(on_error)

    log.info("Bot up. M1 confluence engine armed.")
    app.run_polling()


if __name__ == "__main__":
    main()
