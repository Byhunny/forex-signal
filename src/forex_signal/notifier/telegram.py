"""Telegram notifier — best-effort fire-and-forget.

Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from env. If either is missing,
all calls become no-ops so the rest of the bot keeps working unchanged.

We use urllib (stdlib only) to avoid an extra dependency, and a short timeout
so a slow Telegram API can never block the trading loop.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from threading import Thread

log = logging.getLogger(__name__)


def _send_sync(token: str, chat_id: str, text: str, timeout: float = 5.0) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": "true",
    }).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
            if resp.status != 200:
                log.warning("telegram %d: %s", resp.status, body[:200])
    except Exception as e:
        log.warning("telegram send failed: %s", e)


def notify(text: str) -> None:
    """Send a Telegram message in the background. No-op if creds missing."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return
    t = Thread(target=_send_sync, args=(token, chat_id, text), daemon=True)
    t.start()


def is_configured() -> bool:
    return bool(os.getenv("TELEGRAM_BOT_TOKEN", "").strip() and os.getenv("TELEGRAM_CHAT_ID", "").strip())
