"""
alerting.py

Master's thesis: Learning the Levers — Which Creator-Controllable
Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
from a Swiss German-Language Perspective
Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026

Purpose:
    Telegram push notifications for the scraper pipeline. Surfaces
    cookie expiry, authentication failures, scrape failures, and
    database errors. Falls back to log warnings if credentials are
    not configured so the pipeline never crashes because of alerting.

Inputs:
    .env (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)    credentials
    01_config/settings.py                          config loader

Outputs:
    Telegram messages to the operator's chat       push notifications
    Log records (fallback when Telegram disabled)  log output

Usage:
    from alerting import send_alert, alert_cookie_expiry
    send_alert("Pipeline started")
"""

import logging
import sys
import urllib.request
import urllib.parse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from settings_loader import load_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — reads from 01_config/settings.py via settings_loader
# ---------------------------------------------------------------------------
_BOT_TOKEN: str | None = None
_CHAT_ID: str | None = None
_INITIALISED = False


def _ensure_init() -> None:
    """Load Telegram credentials from settings (which reads .env)."""
    global _BOT_TOKEN, _CHAT_ID, _INITIALISED
    if _INITIALISED:
        return

    settings = load_settings()

    _BOT_TOKEN = getattr(settings, "TELEGRAM_BOT_TOKEN", "") or None
    _CHAT_ID = getattr(settings, "TELEGRAM_CHAT_ID", "") or None
    _INITIALISED = True

    if not _BOT_TOKEN or not _CHAT_ID:
        logger.info(
            "Telegram alerting disabled: TELEGRAM_BOT_TOKEN and/or "
            "TELEGRAM_CHAT_ID not set in .env. Alerts will be logged only."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def send_alert(message: str, silent: bool = False) -> bool:
    """
    Send a Telegram message. Falls back to logging if not configured.

    Args:
        message: Text to send (supports Telegram MarkdownV2 if escaped).
                 Plain text is safest.
        silent: If True, send without notification sound.

    Returns:
        True if the message was sent successfully, False otherwise.
    """
    _ensure_init()

    if not _BOT_TOKEN or not _CHAT_ID:
        logger.warning("[ALERT] %s", message)
        return False

    url = f"https://api.telegram.org/bot{_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": _CHAT_ID,
        "text": message,
        "disable_notification": silent,
    }

    try:
        data = urllib.parse.urlencode(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read())
            if body.get("ok"):
                logger.debug("Telegram alert sent successfully")
                return True
            else:
                logger.warning(
                    "Telegram API returned ok=false: %s",
                    body.get("description", "unknown error"),
                )
                return False

    except Exception as e:
        # Never let alerting crash the pipeline
        logger.warning(
            "Failed to send Telegram alert: %s. Message was: %s",
            e, message[:200],
        )
        return False


# ---------------------------------------------------------------------------
# Convenience helpers for common alert types
# ---------------------------------------------------------------------------

def alert_cookie_expiry(platform: str, warnings: list[str]) -> None:
    """Alert about expired or soon-to-expire cookies."""
    if not warnings:
        return

    lines = [f"COOKIE WARNING ({platform.upper()})"]
    for w in warnings:
        lines.append(f"  - {w}")
    lines.append("")
    lines.append("Action: re-export cookies from your Mac browser.")

    send_alert("\n".join(lines))


def alert_auth_failure(platform: str, account_key: str, detail: str = "") -> None:
    """Alert when a scraper detects an auth/login redirect."""
    msg = (
        f"AUTH FAILURE ({platform.upper()})\n"
        f"Account: {account_key}\n"
        f"Detail: {detail[:300]}\n\n"
        f"Action: re-export cookies for {account_key} from your Mac."
    )
    send_alert(msg)


def alert_scrape_failure(platform: str, account_key: str, error: str = "") -> None:
    """Alert when a platform scrape fails."""
    msg = (
        f"SCRAPE FAILURE ({platform.upper()})\n"
        f"Account: {account_key}\n"
        f"Error: {error[:300]}"
    )
    send_alert(msg)


def alert_db_failure(context: str, error: str = "", fallback_path: str = "") -> None:
    """Alert when a database write fails."""
    lines = [
        f"DB FAILURE",
        f"Context: {context}",
        f"Error: {error[:300]}",
    ]
    if fallback_path:
        lines.append(f"\nData saved to: {fallback_path}")
    else:
        lines.append(f"\nCheck fallback directory for saved JSON.")
    send_alert("\n".join(lines))


def alert_hourly_summary(
    hour: int,
    succeeded: int,
    failed: int,
    total_duration: float,
    failures: list[dict] | None = None,
) -> None:
    """
    Send a summary alert after an hourly run, but only if something failed.

    Successful runs are silent (no spam). Failures get immediate alerts.
    """
    if failed == 0:
        return

    lines = [
        f"HOURLY RUN SUMMARY (UTC {hour:02d}:00)",
        f"  OK: {succeeded}  |  FAILED: {failed}  |  Duration: {total_duration:.0f}s",
    ]

    if failures:
        lines.append("")
        for f in failures:
            lines.append(
                f"  FAIL: {f.get('platform', '?')} ({f.get('account_key', '?')})"
                f" -- {f.get('error', 'unknown')[:150]}"
            )

    send_alert("\n".join(lines))
