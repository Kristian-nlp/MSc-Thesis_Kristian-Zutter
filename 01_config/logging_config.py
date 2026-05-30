"""
logging_config.py

Master's thesis: Learning the Levers — Which Creator-Controllable
Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
from a Swiss German-Language Perspective
Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026

Purpose:
    Centralised logging configuration for the scraper pipeline.
    Every entry point calls setup_logging() once at startup; all
    modules using logging.getLogger(__name__) inherit this config.
    Provides per-platform daily log rotation and a snapshot-summary
    helper for greppable per-run summary lines.

Inputs:
    01_config/settings.py             LOG_DIR, LOG_FORMAT, LOG_LEVEL

Outputs:
    DATA_DIR/logs/<platform>.log      rotating daily logs (30-day retention)
    DATA_DIR/logs/<platform>.log.YYYY-MM-DD     rotated history

Usage:
    from logging_config import setup_logging, log_snapshot_summary
    setup_logging(platform="tiktok", level="INFO")
"""

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from settings_loader import load_settings

logger = logging.getLogger(__name__)


def setup_logging(
    platform: str = "scraper",
    level: str = "INFO",
    log_dir: Path | None = None,
) -> None:
    """
    Configure logging for a scraper run.

    Call this once at the start of every entry point (run_tiktok.py,
    cron jobs, etc.). All loggers created with getLogger(__name__)
    in any module will inherit this configuration.

    Args:
        platform: Name used in the log filename, e.g. "tiktok"
                  produces logs/tiktok.log.
        level: Log level for both console and file handlers
               ("DEBUG", "INFO", "WARNING", "ERROR").
        log_dir: Override log directory. Defaults to settings.LOG_DIR.
    """
    settings = load_settings()
    log_dir = log_dir or settings.LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    # Clear any existing handlers (prevents duplicate output if
    # setup_logging is called more than once, e.g. in tests)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    # --- Formatter ---
    formatter = logging.Formatter(
        fmt=settings.LOG_FORMAT,
        datefmt=settings.LOG_DATE_FORMAT,
    )

    # --- Console handler (INFO+, human-readable) ---
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, level.upper(), logging.INFO))
    console.setFormatter(formatter)
    root.addHandler(console)

    # --- File handler (DEBUG, daily rotation, 30 days retention) ---
    # Let TimedRotatingFileHandler manage the date suffix on rotation.
    # Base file: platform.log -> rotated to platform.log.2026-02-01, etc.
    log_file = log_dir / f"{platform}.log"

    file_handler = TimedRotatingFileHandler(
        filename=str(log_file),
        when="midnight",
        interval=1,
        backupCount=30,      # auto-delete logs older than 30 days
        encoding="utf-8",
        utc=True,
    )
    # File logs at INFO by default to limit disk usage on the VM.
    # Set LOG_LEVEL=DEBUG in .env for temporary diagnostic sessions.
    file_level = getattr(logging, level.upper(), logging.INFO)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Quiet down noisy third-party loggers
    logging.getLogger("selenium").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logger.debug(
        "Logging initialised: console=%s, file=%s",
        level,
        log_file,
    )


def log_snapshot_summary(
    snapshot_id: str,
    platform: str,
    account_type: str,
    success: bool,
    top_count: int,
    baseline_count: int,
    duration: float,
    error: str = "",
) -> None:
    """
    Log a single, scannable summary line for a completed snapshot.

    Produces the key metrics in one line: success/failure, posts
    captured, duration, error. Designed to be greppable for daily
    monitoring.

    Format:
        SNAPSHOT | tiktok | fresh | SUCCESS | top=20 base=50 | 34.2s
        SNAPSHOT | instagram | light_seeded | FAILURE | top=0 base=0 | 5.1s | TimeoutError

    Args:
        snapshot_id: Unique snapshot identifier.
        platform: Platform name.
        account_type: "fresh" or "light_seeded".
        success: Whether the scrape succeeded.
        top_count: Number of Top posts captured.
        baseline_count: Number of baseline posts captured.
        duration: Total scrape duration in seconds.
        error: Error message if the scrape failed.
    """
    status = "SUCCESS" if success else "FAILURE"
    total = top_count + baseline_count

    parts = [
        "SNAPSHOT",
        platform,
        account_type,
        status,
        f"top={top_count} base={baseline_count} total={total}",
        f"{duration:.1f}s",
    ]

    if error:
        parts.append(error[:200])  # truncate long errors

    summary = " | ".join(parts)

    if success:
        logging.getLogger("snapshot").info(summary)
    else:
        logging.getLogger("snapshot").error(summary)


def log_scrape_start(
    platform: str,
    account_type: str,
    account_key: str,
) -> None:
    """Log the start of a scrape run. Pairs with log_snapshot_summary."""
    logging.getLogger("snapshot").info(
        "SCRAPE START | %s | %s | %s",
        platform,
        account_type,
        account_key,
    )
