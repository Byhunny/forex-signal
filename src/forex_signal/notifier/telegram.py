"""Telegram notifier — fire-and-forget messages + optional command listener.

Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from env. If either is missing,
all calls become no-ops so the rest of the bot keeps working unchanged.

We use urllib (stdlib only) to avoid an extra dependency.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.parse
import urllib.request
from threading import Thread
from typing import Callable

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


def _get_updates(token: str, offset: int, timeout: int = 25) -> list[dict]:
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    params = urllib.parse.urlencode({"offset": offset, "timeout": timeout, "allowed_updates": '["message"]'})
    try:
        with urllib.request.urlopen(f"{url}?{params}", timeout=timeout + 10) as resp:
            data = json.loads(resp.read())
            return data.get("result", [])
    except Exception as e:
        log.debug("telegram getUpdates: %s", e)
        return []


def _command_loop(token: str, chat_id: str, handlers: dict[str, Callable[[], str]]) -> None:
    """Long-poll for commands. Each handler returns text to send back."""
    offset = 0
    log.info("telegram command listener started — handlers: %s", list(handlers.keys()))
    while True:
        try:
            updates = _get_updates(token, offset)
            for u in updates:
                offset = u["update_id"] + 1
                msg = u.get("message") or {}
                if str(msg.get("chat", {}).get("id")) != chat_id:
                    continue  # only respond to authorized chat
                text = (msg.get("text") or "").strip().lower()
                # strip "@botname" if user typed /cmd@mybot
                cmd = text.split("@", 1)[0].split()[0] if text else ""
                handler = handlers.get(cmd)
                if not handler:
                    continue
                try:
                    response = handler()
                except Exception as e:
                    response = f"⚠️ `{cmd}` failed: {e}"
                _send_sync(token, chat_id, response)
        except Exception as e:
            log.warning("telegram command loop error: %s", e)
            time.sleep(5)


def start_command_listener(handlers: dict[str, Callable[[], str]]) -> None:
    """Spawn a daemon thread that listens for Telegram commands and replies via handlers."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        log.info("telegram command listener disabled (creds missing)")
        return
    t = Thread(target=_command_loop, args=(token, chat_id, handlers), daemon=True)
    t.start()
