"""
revisit_db.py

Master's thesis: Learning the Levers — Which Creator-Controllable
Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
from a Swiss German-Language Perspective
Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026

Purpose:
    Database queries and writes for engagement velocity revisits.
    Provides query helpers for posts due for T+24h or T+72h
    revisits (with a +/- 2h tolerance window so cron drift does not
    cause misses), writes revisit counters to the counters table,
    and records revisit attempts in a dedicated revisit_log table.

Inputs:
    01_config/settings.py             DB_PATH (via db.get_connection)
    04_database/scraper.db            posts, captures, counters tables

Outputs:
    04_database/scraper.db            counters (T24/T72), revisit_log
                                      tables (rows appended)

Usage:
    from revisit_db import (get_posts_due_for_revisit, write_revisit_counters,
                            write_revisit_log)
    Imported by the scraper and feature-extraction orchestrators.
"""

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from db import get_connection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DB_PATH = Path(__file__).resolve().parent / "scraper.db"

# Tolerance windows (hours) around the target revisit time.
# T24 revisit: posts whose T0 was captured 22-26 hours ago.
# T72 revisit: posts whose T0 was captured 70-74 hours ago.
# This +/- 2h window prevents missed revisits if cron drifts.
REVISIT_WINDOWS = {
    "t24": {"target_hours": 24, "tolerance_hours": 2},
    "t72": {"target_hours": 72, "tolerance_hours": 2},
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PendingRevisit:
    """A post that is due for a revisit."""
    post_id: str
    platform: str
    permalink: str
    revisit_type: str           # "t24" or "t72"
    t0_captured_at: str         # ISO 8601 UTC of original T0 capture
    hours_since_t0: float       # how many hours since T0


@dataclass
class RevisitCounters:
    """
    Engagement counters collected during a revisit.

    NOTE on follower_count: Permalink pages typically do not display
    the author's follower count. This field will be NULL for T24/T72
    revisits. The T0 follower count (from discovery surface JSON) is
    available in the posts table and should be used as the velocity
    denominator in analysis.

    NOTE on views: TikTok permalink pages do not show a separate view
    counter. Views are only available at T0 from JSON interception.

    NOTE on shares: Instagram does not expose share counts publicly.
    Shares will be NULL for Instagram at all time points.
    """
    likes: int | None = None
    comments: int | None = None
    shares: int | None = None
    views: int | None = None
    follower_count: int | None = None


@dataclass
class RevisitResult:
    """Result of a single post revisit attempt."""
    post_id: str
    platform: str
    permalink: str
    revisit_type: str
    success: bool = False
    post_missing: bool = False  # True if post was deleted/unavailable
    counters: RevisitCounters = field(default_factory=RevisitCounters)
    error_message: str = ""
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Schema check: revisit_log table
# ---------------------------------------------------------------------------
# The revisit_log table is defined in schema.sql (single source of truth).
# get_connection() runs schema.sql on first use, creating all tables.
# This function is a safety net for cases where revisit_db.py is called
# before db.py has initialised the schema.

def ensure_revisit_log_table(db_path: Path | str | None = None) -> None:
    """Ensure the revisit_log table exists (created by schema.sql via get_connection)."""
    conn = get_connection(db_path)
    conn.close()


# ---------------------------------------------------------------------------
# Query: posts due for revisit
# ---------------------------------------------------------------------------

def get_posts_due_for_revisit(
    revisit_type: str,
    platform: str | None = None,
    db_path: Path | str | None = None,
) -> list[PendingRevisit]:
    """
    Find posts that are due for a T24 or T72 revisit.

    A post is due when:
        1. It has a T0 counter row (was successfully scraped initially)
        2. It does NOT yet have a counter row for this revisit_type
        3. Its T0 capture time falls within the tolerance window
        4. It has a non-null permalink (needed to navigate to the post)

    Args:
        revisit_type: "t24" or "t72"
        platform: Optional filter for a single platform.
        db_path: Override database path.

    Returns:
        List of PendingRevisit objects sorted by hours_since_t0 descending
        (oldest first, so we revisit the most time-critical posts first).
    """
    if revisit_type not in REVISIT_WINDOWS:
        raise ValueError(
            f"Invalid revisit_type: {revisit_type}. "
            f"Must be one of {list(REVISIT_WINDOWS.keys())}"
        )

    window = REVISIT_WINDOWS[revisit_type]
    min_hours = window["target_hours"] - window["tolerance_hours"]
    max_hours = window["target_hours"] + window["tolerance_hours"]

    conn = get_connection(db_path)

    try:
        query = """
            SELECT
                p.post_id,
                p.platform,
                p.permalink,
                ct0.captured_at_utc AS t0_captured_at,
                (julianday('now') - julianday(ct0.captured_at_utc)) * 24
                    AS hours_since_t0
            FROM counters ct0
            INNER JOIN posts p ON ct0.post_id = p.post_id
            WHERE ct0.revisit_type = 't0'
              AND p.permalink IS NOT NULL
              AND p.permalink != ''
              AND (julianday('now') - julianday(ct0.captured_at_utc)) * 24
                  BETWEEN ? AND ?
              AND NOT EXISTS (
                  SELECT 1 FROM counters cx
                  WHERE cx.post_id = ct0.post_id
                    AND cx.revisit_type = ?
              )
        """
        params: list = [min_hours, max_hours, revisit_type]

        if platform:
            query += "\n              AND p.platform = ?"
            params.append(platform)

        query += "\n            ORDER BY hours_since_t0 DESC"

        rows = conn.execute(query, params).fetchall()

        results = [
            PendingRevisit(
                post_id=row["post_id"],
                platform=row["platform"],
                permalink=row["permalink"],
                revisit_type=revisit_type,
                t0_captured_at=row["t0_captured_at"],
                hours_since_t0=round(row["hours_since_t0"], 2),
            )
            for row in rows
        ]

        logger.info(
            "Found %d posts due for %s revisit%s (window: %d-%dh)",
            len(results),
            revisit_type,
            f" on {platform}" if platform else "",
            min_hours,
            max_hours,
        )
        return results

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Write: revisit counters
# ---------------------------------------------------------------------------

def write_revisit_counters(
    result: RevisitResult,
    db_path: Path | str | None = None,
    conn: sqlite3.Connection | None = None,
) -> bool:
    """
    Write a single revisit counter row to the database.

    If the post was deleted/missing, inserts a row with NULL counters
    so we know the revisit was attempted.  The UNIQUE constraint on
    (post_id, revisit_type) prevents duplicate revisits.

    Args:
        result: RevisitResult from the scraper.
        db_path: Override database path.
        conn: Optional reusable connection. If provided, the caller
              is responsible for closing it. If None, a new connection
              is opened and closed per call.

    Returns:
        True if the write succeeded, False otherwise.
    """
    own_conn = conn is None
    if own_conn:
        conn = get_connection(db_path)

    try:
        captured_at = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        if result.post_missing:
            # Post was deleted/private -- insert NULLs to record the attempt
            conn.execute(
                """
                INSERT OR IGNORE INTO counters (
                    post_id, revisit_type, likes, comments, shares,
                    views, follower_count, captured_at_utc
                ) VALUES (?, ?, NULL, NULL, NULL, NULL, NULL, ?)
                """,
                (result.post_id, result.revisit_type, captured_at),
            )
            conn.commit()
            logger.info(
                "Recorded missing post for %s revisit: %s",
                result.revisit_type,
                result.post_id,
            )
            return True

        if not result.success:
            logger.warning(
                "Revisit failed for %s (%s): %s",
                result.post_id,
                result.revisit_type,
                result.error_message,
            )
            return False

        c = result.counters
        conn.execute(
            """
            INSERT OR IGNORE INTO counters (
                post_id, revisit_type, likes, comments, shares,
                views, follower_count, captured_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.post_id,
                result.revisit_type,
                c.likes,
                c.comments,
                c.shares,
                c.views,
                c.follower_count,
                captured_at,
            ),
        )
        conn.commit()

        logger.info(
            "Wrote %s counters for %s: likes=%s, comments=%s, shares=%s",
            result.revisit_type,
            result.post_id,
            c.likes,
            c.comments,
            c.shares,
        )
        return True

    except sqlite3.Error as e:
        logger.error(
            "Failed to write %s counters for %s: %s",
            result.revisit_type,
            result.post_id,
            e,
        )
        return False

    finally:
        if own_conn:
            conn.close()


# ---------------------------------------------------------------------------
# Write: revisit log (batch-level summary)
# ---------------------------------------------------------------------------

def write_revisit_log(
    revisit_type: str,
    platform: str,
    total_due: int,
    total_success: int,
    total_missing: int,
    total_failed: int,
    duration_sec: float,
    started_at_utc: str,
    error_message: str = "",
    db_path: Path | str | None = None,
) -> None:
    """
    Write a summary row to revisit_log for monitoring.

    One row per (revisit_type, platform) per run.
    """
    # Ensure the table exists
    ensure_revisit_log_table(db_path)

    finished_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            INSERT INTO revisit_log (
                revisit_type, platform,
                total_due, total_success, total_missing, total_failed,
                duration_sec, started_at_utc, finished_at_utc,
                error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                revisit_type,
                platform,
                total_due,
                total_success,
                total_missing,
                total_failed,
                round(duration_sec, 2),
                started_at_utc,
                finished_at,
                error_message or None,
            ),
        )
        conn.commit()
        logger.info(
            "Revisit log: %s/%s -- due=%d success=%d missing=%d failed=%d (%.1fs)",
            revisit_type,
            platform,
            total_due,
            total_success,
            total_missing,
            total_failed,
            duration_sec,
        )
    finally:
        conn.close()
