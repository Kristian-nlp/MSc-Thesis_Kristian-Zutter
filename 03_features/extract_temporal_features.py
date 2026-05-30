"""
extract_temporal_features.py

Master's thesis: Learning the Levers — Which Creator-Controllable
Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
from a Swiss German-Language Perspective
Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026

Purpose:
    Batch temporal-feature extraction. Reads posted_at_utc from the
    posts table and the earliest T0 capture timestamp from the
    counters table, converts to Europe/Zurich local time, and writes
    local hour, ISO weekday, weekend flag, and post-age-hours to
    features_temporal. Posts with missing or unparseable timestamps
    receive NULL rows so they are not retried. Stdlib only.

Inputs:
    01_config/settings.py             DB_PATH
    04_database/scraper.db            posts.posted_at_utc,
                                      counters table (T0 capture)

Outputs:
    04_database/scraper.db            features_temporal table

Usage:
    python 03_features/extract_temporal_features.py
    python 03_features/extract_temporal_features.py --dry-run
"""

import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

# Resolve numbered-directory imports (04_database/db.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "04_database"))

logger = logging.getLogger(__name__)


# ===================================================================
# Configuration
# ===================================================================

DB_PATH = Path(__file__).resolve().parent.parent / "04_database" / "scraper.db"
BATCH_SIZE = 500
SWISS_TZ = ZoneInfo("Europe/Zurich")


# ===================================================================
# Timestamp parsing helpers
# ===================================================================

# Common ISO 8601 formats encountered in scraped data.
# Ordered from most specific to least specific so the first
# successful parse wins.
_TIMESTAMP_FORMATS = [
    "%Y-%m-%dT%H:%M:%SZ",        # 2026-02-15T14:30:00Z
    "%Y-%m-%dT%H:%M:%S.%fZ",     # 2026-02-15T14:30:00.000Z
    "%Y-%m-%dT%H:%M:%S%z",       # 2026-02-15T14:30:00+00:00
    "%Y-%m-%dT%H:%M:%S.%f%z",    # 2026-02-15T14:30:00.000+00:00
    "%Y-%m-%d %H:%M:%S",         # 2026-02-15 14:30:00 (assumed UTC)
    "%Y-%m-%dT%H:%M:%S",         # 2026-02-15T14:30:00  (no Z, assumed UTC)
]


def parse_utc_timestamp(raw: str | None) -> datetime | None:
    """
    Parse a UTC timestamp string into a timezone-aware datetime.

    Tries several ISO 8601 variants.  If the parsed datetime is
    naive (no tzinfo), UTC is assumed.

    Args:
        raw: ISO 8601 timestamp string, or None.

    Returns:
        Timezone-aware datetime in UTC, or None if parsing fails.
    """
    if not raw or not raw.strip():
        return None

    raw = raw.strip()

    for fmt in _TIMESTAMP_FORMATS:
        try:
            dt = datetime.strptime(raw, fmt)
            # Ensure timezone-aware (assume UTC if naive)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue

    # Last resort: try fromisoformat (handles many edge cases in 3.11+)
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        pass

    logger.warning("Could not parse timestamp: %r", raw)
    return None


# ===================================================================
# Feature computation
# ===================================================================

def compute_local_hour(posted_utc: datetime) -> int:
    """Convert UTC posting time to Swiss local hour (0-23)."""
    local_dt = posted_utc.astimezone(SWISS_TZ)
    return local_dt.hour


def compute_weekday(posted_utc: datetime) -> int:
    """
    ISO weekday from Swiss local date: 0=Monday .. 6=Sunday.

    Uses .weekday() which already returns 0=Mon, 6=Sun.
    """
    local_dt = posted_utc.astimezone(SWISS_TZ)
    return local_dt.weekday()


def compute_is_weekend(weekday: int) -> int:
    """Return 1 if Saturday (5) or Sunday (6), else 0."""
    return 1 if weekday >= 5 else 0


def compute_post_age_hours(
    posted_utc: datetime,
    t0_captured_utc: datetime | None,
) -> float | None:
    """
    Hours between posting and first observation (T0 capture).

    Returns None if the T0 capture timestamp is unavailable.
    A negative value is theoretically impossible but is preserved
    (not clamped) to flag data-quality issues during analysis.

    Args:
        posted_utc: When the post was created (UTC).
        t0_captured_utc: When the scraper first captured the post (UTC).

    Returns:
        Age in hours (float), or None.
    """
    if t0_captured_utc is None:
        return None

    delta = t0_captured_utc - posted_utc
    return delta.total_seconds() / 3600.0


# ===================================================================
# Database queries
# ===================================================================

def get_unprocessed_posts(
    conn: sqlite3.Connection,
    limit: int | None = None,
) -> list[dict]:
    """
    Fetch posts that do not yet have a features_temporal row.

    Joins to the counters table to retrieve the T0 capture timestamp
    for post_age_hours computation. Falls back to the earliest
    snapshot capture time (via captures -> snapshots) when no T0
    counter exists yet.

    Args:
        conn: Open SQLite connection (row_factory = sqlite3.Row).
        limit: Max rows to fetch (None = all).

    Returns:
        List of dicts with post_id, posted_at_utc, t0_captured_at_utc.
    """
    sql = """
        SELECT
            p.post_id,
            p.posted_at_utc,
            COALESCE(
                c_t0.captured_at_utc,
                (SELECT MIN(s.captured_at_utc)
                 FROM captures cap
                 JOIN snapshots s ON cap.snapshot_id = s.snapshot_id
                 WHERE cap.post_id = p.post_id)
            ) AS t0_captured_at_utc
        FROM posts p
        LEFT JOIN features_temporal ft
            ON p.post_id = ft.post_id
        LEFT JOIN counters c_t0
            ON p.post_id = c_t0.post_id
            AND c_t0.revisit_type = 't0'
        WHERE ft.post_id IS NULL
        ORDER BY p.created_at ASC
    """
    params: list = []
    if limit:
        sql += "\n        LIMIT ?"
        params.append(int(limit))

    rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def write_features_batch(
    conn: sqlite3.Connection,
    features: list[dict],
) -> int:
    """
    Insert a batch of computed features into features_temporal.

    Uses INSERT OR IGNORE to handle duplicates safely if the
    script is interrupted and restarted.

    Args:
        conn: Open SQLite connection.
        features: List of dicts with keys matching features_temporal columns.

    Returns:
        Number of rows inserted.
    """
    if not features:
        return 0

    inserted = 0
    with conn:
        for feat in features:
            try:
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO features_temporal (
                        post_id, local_hour, weekday, is_weekend,
                        post_age_hours, processed_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        feat["post_id"],
                        feat["local_hour"],
                        feat["weekday"],
                        feat["is_weekend"],
                        feat["post_age_hours"],
                        feat["processed_at"],
                    ),
                )
                if cursor.rowcount > 0:
                    inserted += 1
            except sqlite3.Error as e:
                logger.error(
                    "Failed to insert temporal features for %s: %s",
                    feat["post_id"], e,
                )

    return inserted


# ===================================================================
# Per-post processing
# ===================================================================

def process_post(row: dict) -> dict:
    """
    Compute all temporal features for a single post.

    If posted_at_utc is NULL or unparseable, all feature fields are
    set to None.  The row is still inserted (with processed_at set)
    to mark it as processed.

    Args:
        row: Dict with post_id, posted_at_utc, t0_captured_at_utc.

    Returns:
        Dict ready for insertion into features_temporal.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    posted_utc = parse_utc_timestamp(row.get("posted_at_utc"))
    t0_utc = parse_utc_timestamp(row.get("t0_captured_at_utc"))

    if posted_utc is None:
        # Cannot derive any temporal features without a posting timestamp
        logger.debug(
            "post_id=%s: posted_at_utc is NULL or unparseable, "
            "inserting NULL row",
            row["post_id"],
        )
        return {
            "post_id": row["post_id"],
            "local_hour": None,
            "weekday": None,
            "is_weekend": None,
            "post_age_hours": None,
            "processed_at": now,
        }

    weekday = compute_weekday(posted_utc)

    return {
        "post_id": row["post_id"],
        "local_hour": compute_local_hour(posted_utc),
        "weekday": weekday,
        "is_weekend": compute_is_weekend(weekday),
        "post_age_hours": compute_post_age_hours(posted_utc, t0_utc),
        "processed_at": now,
    }


# ===================================================================
# Main pipeline
# ===================================================================

def run(
    db_path: Path | str | None = None,
    batch_size: int = BATCH_SIZE,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict:
    """
    Run the full temporal-feature extraction pipeline.

    Args:
        db_path: Path to the SQLite database. Defaults to /data/scraper.db.
        batch_size: Rows per DB transaction.
        limit: Max total posts to process (None = all pending).
        dry_run: If True, compute features but do not write to DB.

    Returns:
        Dict with summary stats: total, processed, null_timestamps, errors.
    """
    path = Path(db_path) if db_path else DB_PATH
    logger.info("Temporal feature extraction starting (db=%s)", path)

    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")

    stats = {
        "total": 0,
        "processed": 0,
        "null_timestamps": 0,
        "errors": 0,
    }

    try:
        # Fetch unprocessed posts
        posts = get_unprocessed_posts(conn, limit=limit)
        stats["total"] = len(posts)
        logger.info("Found %d unprocessed posts", len(posts))

        if not posts:
            logger.info("Nothing to do -- all posts already processed")
            return stats

        # Process in batches
        batch: list[dict] = []

        for i, row in enumerate(posts, 1):
            try:
                features = process_post(row)
                batch.append(features)

                if features["local_hour"] is None:
                    stats["null_timestamps"] += 1

            except Exception as e:
                stats["errors"] += 1
                logger.error(
                    "Error processing post %s: %s", row["post_id"], e,
                )
                continue

            # Flush batch
            if len(batch) >= batch_size:
                if not dry_run:
                    written = write_features_batch(conn, batch)
                    stats["processed"] += written
                else:
                    stats["processed"] += len(batch)
                logger.info(
                    "Progress: %d / %d posts (batch written: %d)",
                    i, len(posts), len(batch),
                )
                batch = []

        # Flush remaining
        if batch:
            if not dry_run:
                written = write_features_batch(conn, batch)
                stats["processed"] += written
            else:
                stats["processed"] += len(batch)

        logger.info(
            "Temporal feature extraction complete: "
            "total=%d, processed=%d, null_timestamps=%d, errors=%d",
            stats["total"],
            stats["processed"],
            stats["null_timestamps"],
            stats["errors"],
        )

    except Exception as e:
        logger.exception("Fatal error in temporal feature extraction: %s", e)
        raise

    finally:
        conn.close()

    return stats


# ===================================================================
# CLI entry point
# ===================================================================

def main():
    """
    Command-line entry point.

    Usage:
        python extract_temporal_features.py [--db PATH] [--batch N]
                                            [--limit N] [--dry-run]
                                            [--log-level LEVEL]
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Extract temporal features from post timestamps. "
            "Converts posted_at_utc to Swiss local time and computes "
            "local_hour, weekday, is_weekend, and post_age_hours."
        ),
    )
    parser.add_argument(
        "--db", type=str, default=str(DB_PATH),
        help=f"Path to SQLite database (default: {DB_PATH})",
    )
    parser.add_argument(
        "--batch", type=int, default=BATCH_SIZE,
        help=f"Batch size for DB writes (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max posts to process (default: all pending)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute features but do not write to the database",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Set up basic logging (standalone mode)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stats = run(
        db_path=args.db,
        batch_size=args.batch,
        limit=args.limit,
        dry_run=args.dry_run,
    )

    # Exit code: 0 if no errors, 1 if some errors occurred
    if stats["errors"] > 0:
        logger.warning(
            "Completed with %d errors out of %d posts",
            stats["errors"], stats["total"],
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
