"""
fallback.py

Master's thesis: Learning the Levers — Which Creator-Controllable
Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
from a Swiss German-Language Perspective
Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026

Purpose:
    JSON fallback buffer for database write failures. When SQLite is
    unreachable (locked, disk full, corrupted), scrape results are
    written to the fallback directory as timestamped JSON envelopes
    and replayed on the next successful run.

Inputs:
    01_config/settings.py             FALLBACK_DIR location
    DATA_DIR/fallback/*.json          pending fallback envelopes

Outputs:
    DATA_DIR/fallback/*.json          new fallback envelopes
    DATA_DIR/fallback/*.json.done     archived after ingestion

Usage:
    from fallback import write_fallback, list_pending, read_fallback
    write_fallback(data, context="tiktok_fresh")
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from settings_loader import load_settings

logger = logging.getLogger(__name__)


def _get_fallback_dir() -> Path:
    """Return the fallback directory from settings (derived from DATA_DIR)."""
    settings = load_settings()
    return getattr(settings, "FALLBACK_DIR", settings.DATA_DIR / "fallback")


# ---------------------------------------------------------------------------
# Write fallback
# ---------------------------------------------------------------------------

def write_fallback(
    data: dict,
    context: str = "unknown",
    fallback_dir: Path | None = None,
) -> Path | None:
    """
    Write scrape data to a JSON fallback file.

    Args:
        data: Dictionary of scrape results to preserve.
        context: Label for the fallback file (e.g. "tiktok_fresh",
                 "revisit_t24_instagram").
        fallback_dir: Override fallback directory (for testing).

    Returns:
        Path to the written file, or None if write also failed.
    """
    out_dir = fallback_dir or _get_fallback_dir()

    try:
        out_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = f"{context}_{ts}.json"
        filepath = out_dir / filename

        # Wrap with metadata
        envelope = {
            "fallback_created_at": ts,
            "context": context,
            "data": data,
        }

        filepath.write_text(
            json.dumps(envelope, indent=2, default=str),
            encoding="utf-8",
        )

        logger.warning(
            "Fallback written: %s (%d bytes)",
            filepath, filepath.stat().st_size,
        )

        # Send Telegram alert so the operator knows DB is failing
        try:
            from alerting import alert_db_failure
            alert_db_failure(context, "DB write failed, data saved to fallback JSON",
                             fallback_path=str(filepath))
        except Exception:
            pass  # Never let alerting break the fallback path

        return filepath

    except Exception as e:
        logger.critical(
            "FALLBACK WRITE ALSO FAILED for context=%s: %s. "
            "Data may be lost. Error: %s",
            context, e, str(data)[:500],
        )
        return None


# ---------------------------------------------------------------------------
# List and ingest pending fallbacks
# ---------------------------------------------------------------------------

def list_pending(fallback_dir: Path | None = None) -> list[Path]:
    """Return all pending fallback JSON files, sorted oldest first."""
    out_dir = fallback_dir or _get_fallback_dir()

    if not out_dir.exists():
        return []

    files = sorted(out_dir.glob("*.json"))
    return files


def read_fallback(filepath: Path) -> dict | None:
    """
    Read a fallback JSON file.

    Returns:
        The parsed envelope dict, or None if reading failed.
    """
    try:
        return json.loads(filepath.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error("Failed to read fallback %s: %s", filepath, e)
        return None


def mark_ingested(filepath: Path) -> None:
    """
    Move a processed fallback file to a .done suffix.

    We rename rather than delete so there is an audit trail.
    """
    try:
        done_path = filepath.with_suffix(".json.done")
        filepath.rename(done_path)
        logger.info("Fallback ingested and archived: %s", done_path.name)
    except Exception as e:
        logger.warning(
            "Could not rename fallback %s: %s", filepath.name, e,
        )


def count_pending(fallback_dir: Path | None = None) -> int:
    """Return the number of pending (un-ingested) fallback files."""
    return len(list_pending(fallback_dir))


def cleanup_done(max_age_days: int = 7, fallback_dir: Path | None = None) -> int:
    """
    Delete .done fallback files older than max_age_days.

    Returns the number of files removed.
    """
    out_dir = fallback_dir or _get_fallback_dir()
    if not out_dir.exists():
        return 0

    import time
    cutoff = time.time() - (max_age_days * 86400)
    removed = 0

    for f in out_dir.glob("*.json.done"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
        except Exception as e:
            logger.warning("Could not remove old fallback %s: %s", f.name, e)

    if removed:
        logger.info("Cleaned up %d old .done fallback files", removed)
    return removed
