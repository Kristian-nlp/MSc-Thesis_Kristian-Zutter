"""
extract_audio_features.py

Master's thesis: Learning the Levers — Which Creator-Controllable
Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
from a Swiss German-Language Perspective
Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026

Purpose:
    Batch audio-feature extraction. Reads audio metadata
    (audio_present, audio_id, audio_name) from the posts table,
    derives an audio_is_original flag from multilingual TikTok
    patterns, and computes a dataset-internal trending heuristic
    (sound appears across >= TRENDING_THRESHOLD distinct authors).
    Instagram and LinkedIn posts have no audio metadata and receive
    null/zero values.

Inputs:
    01_config/settings.py             DB_PATH
    04_database/scraper.db            posts.audio_present, audio_id,
                                      audio_name, author_id

Outputs:
    04_database/scraper.db            features_audio table

Usage:
    python 03_features/extract_audio_features.py
    python 03_features/extract_audio_features.py --recompute
    python 03_features/extract_audio_features.py --dry-run
"""

import argparse
import logging
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# Resolve numbered-directory imports (04_database/db.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "04_database"))

logger = logging.getLogger(__name__)


# ===================================================================
# Configuration
# ===================================================================

DB_PATH = Path(__file__).resolve().parent.parent / "04_database" / "scraper.db"

# How many posts to process per DB transaction
BATCH_SIZE = 500

# Minimum number of distinct authors sharing an audio_id for it
# to be flagged as "trending". Configurable via env var or CLI.
TRENDING_THRESHOLD = int(os.getenv("TRENDING_THRESHOLD", "2"))

# ---------------------------------------------------------------------------
# Original sound detection
# ---------------------------------------------------------------------------
# TikTok localises "original sound" into many languages. We match
# known patterns from the dataset plus common TikTok localisations.
# The pattern also matches "Originalton - <username>" variants.
_ORIGINAL_SOUND_PATTERNS = [
    r"original sound",       # English
    r"Originalton",          # German
    r"Original Sound",       # English (capitalised)
    r"originalljud",         # Swedish
    r"son original",         # French
    r"sonido original",      # Spanish
    r"som original",         # Portuguese
    r"suara asli",           # Indonesian / Malay
    r"orijinal ses",         # Turkish
    r"оригинальный звук",    # Russian
    r"الصوت الأصلي",         # Arabic
    r"πρωτότυπος ήχος",      # Greek
]

_ORIGINAL_SOUND_REGEX = re.compile(
    "|".join(f"(?:{p})" for p in _ORIGINAL_SOUND_PATTERNS),
    flags=re.IGNORECASE,
)


# ===================================================================
# Trending sound lookup
# ===================================================================

def build_trending_lookup(
    conn: sqlite3.Connection,
    threshold: int = TRENDING_THRESHOLD,
) -> set[str]:
    """
    Build a set of audio_ids that qualify as "trending".

    A sound is trending if it is used by >= `threshold` distinct
    authors (identified by author_hash) in the posts table. This
    captures the core signal: multiple creators independently
    choosing the same sound on a discovery surface implies the
    platform's algorithm is amplifying that sound.

    Args:
        conn: Open SQLite connection.
        threshold: Minimum distinct-author count to qualify.

    Returns:
        Set of audio_id strings that meet the threshold.
    """
    sql = """
        SELECT audio_id, COUNT(DISTINCT author_hash) AS author_count
        FROM posts
        WHERE audio_id IS NOT NULL
          AND audio_id != ''
          AND author_hash IS NOT NULL
        GROUP BY audio_id
        HAVING COUNT(DISTINCT author_hash) >= ?
    """
    rows = conn.execute(sql, (threshold,)).fetchall()
    trending = {row["audio_id"] for row in rows}

    logger.info(
        "Trending lookup: %d sounds meet threshold of %d distinct authors "
        "(out of %d total distinct sounds)",
        len(trending),
        threshold,
        _count_distinct_sounds(conn),
    )

    return trending


def _count_distinct_sounds(conn: sqlite3.Connection) -> int:
    """Count total distinct non-empty audio_ids in posts."""
    row = conn.execute(
        """
        SELECT COUNT(DISTINCT audio_id) AS n
        FROM posts
        WHERE audio_id IS NOT NULL AND audio_id != ''
        """
    ).fetchone()
    return row["n"] if row else 0


# ===================================================================
# Feature extraction
# ===================================================================

def detect_original_sound(audio_name: str | None) -> int | None:
    """
    Determine if the audio is the creator's original sound.

    TikTok labels original sounds with localised variants of
    "original sound" (e.g. "Originalton", "son original").
    Returns 1 if original, 0 if a licensed/shared sound, or
    None if no audio metadata is available.

    Args:
        audio_name: The sound title from the posts table.

    Returns:
        1 (original), 0 (not original), or None (no audio).
    """
    if not audio_name:
        return None
    return 1 if _ORIGINAL_SOUND_REGEX.search(audio_name) else 0


def process_post(
    row: dict,
    trending_ids: set[str],
) -> dict:
    """
    Compute audio features for a single post.

    Args:
        row: Dict with post_id, audio_present, audio_id, audio_name.
        trending_ids: Pre-computed set of trending audio_ids.

    Returns:
        Dict ready for insertion into features_audio.
    """
    audio_id = row.get("audio_id") or ""
    audio_name = row.get("audio_name") or ""
    audio_present = row.get("audio_present") or 0

    # Determine trending flag
    if audio_id and audio_id in trending_ids:
        is_trending = 1
    else:
        is_trending = 0

    # Determine original sound flag from audio_name
    audio_is_original = detect_original_sound(audio_name) if audio_present else None

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "post_id": row["post_id"],
        "audio_present": int(audio_present),
        "is_trending": is_trending,
        "audio_id": audio_id if audio_id else None,
        "audio_name": audio_name if audio_name else None,
        "audio_is_original": audio_is_original,
        "processed_at": now,
    }


# ===================================================================
# Database interaction
# ===================================================================

def get_unprocessed_posts(
    conn: sqlite3.Connection,
    limit: int | None = None,
) -> list[dict]:
    """
    Fetch posts that do not yet have a features_audio row.

    Args:
        conn: Open SQLite connection.
        limit: Max rows to fetch (None = all).

    Returns:
        List of dicts with post_id, audio_present, audio_id, audio_name.
    """
    sql = """
        SELECT p.post_id, p.audio_present, p.audio_id, p.audio_name
        FROM posts p
        LEFT JOIN features_audio fa ON p.post_id = fa.post_id
        WHERE fa.post_id IS NULL
        ORDER BY p.created_at ASC
    """
    params: list = []
    if limit:
        sql += " LIMIT ?"
        params.append(int(limit))

    rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def write_features_batch(
    conn: sqlite3.Connection,
    features: list[dict],
) -> int:
    """
    Insert a batch of computed features into features_audio.

    Uses INSERT OR IGNORE to be safe against duplicates if the
    script is interrupted and restarted.

    Args:
        conn: Open SQLite connection.
        features: List of dicts with keys matching features_audio columns.

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
                    INSERT OR IGNORE INTO features_audio (
                        post_id, audio_present, is_trending,
                        audio_id, audio_name, audio_is_original,
                        processed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        feat["post_id"],
                        feat["audio_present"],
                        feat["is_trending"],
                        feat["audio_id"],
                        feat["audio_name"],
                        feat["audio_is_original"],
                        feat["processed_at"],
                    ),
                )
                if cursor.rowcount > 0:
                    inserted += 1
            except sqlite3.Error as e:
                logger.error(
                    "Failed to insert features for %s: %s",
                    feat["post_id"], e,
                )

    return inserted


# ===================================================================
# Main pipeline
# ===================================================================

def run(
    db_path: Path | str | None = None,
    batch_size: int = BATCH_SIZE,
    limit: int | None = None,
    trending_threshold: int = TRENDING_THRESHOLD,
    recompute: bool = False,
    dry_run: bool = False,
) -> dict:
    """
    Run the full audio-feature extraction pipeline.

    Args:
        db_path: Path to the SQLite database. Defaults to /data/scraper.db.
        batch_size: Rows per DB transaction.
        limit: Max total posts to process (None = all pending).
        trending_threshold: Min distinct authors for trending flag.
        recompute: If True, delete all features_audio rows first and
                   reprocess everything (useful after threshold change).
        dry_run: If True, compute features but do not write to DB.

    Returns:
        Dict with summary stats: total, processed, errors,
        trending_sounds, trending_threshold.
    """
    path = Path(db_path) if db_path else DB_PATH
    logger.info("Audio feature extraction starting (db=%s)", path)

    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")

    stats = {
        "total": 0,
        "processed": 0,
        "errors": 0,
        "trending_sounds": 0,
        "trending_threshold": trending_threshold,
    }

    try:
        # Optionally wipe existing features for a full recompute
        if recompute and not dry_run:
            deleted = conn.execute("DELETE FROM features_audio").rowcount
            conn.commit()
            logger.info(
                "Recompute mode: deleted %d existing features_audio rows",
                deleted,
            )

        # Build the trending sound lookup (always from full posts table)
        trending_ids = build_trending_lookup(conn, trending_threshold)
        stats["trending_sounds"] = len(trending_ids)

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
                features = process_post(row, trending_ids)
                batch.append(features)
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
            "Audio feature extraction complete: "
            "total=%d, processed=%d, errors=%d, "
            "trending_sounds=%d (threshold=%d)",
            stats["total"],
            stats["processed"],
            stats["errors"],
            stats["trending_sounds"],
            stats["trending_threshold"],
        )

    except Exception as e:
        logger.exception("Fatal error in audio feature extraction: %s", e)
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
        python extract_audio_features.py [--db PATH] [--batch N]
                                         [--limit N] [--trending-threshold N]
                                         [--recompute] [--dry-run]
                                         [--log-level LEVEL]
    """
    parser = argparse.ArgumentParser(
        description=(
            "Extract audio features from scraped post metadata. "
            "Populates the features_audio table with audio_present, "
            "is_trending, and audio_id."
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
        "--trending-threshold", type=int, default=TRENDING_THRESHOLD,
        help=(
            f"Minimum distinct authors sharing an audio_id for it to be "
            f"flagged as trending (default: {TRENDING_THRESHOLD})"
        ),
    )
    parser.add_argument(
        "--recompute", action="store_true",
        help=(
            "Delete all existing features_audio rows and reprocess "
            "from scratch. Useful after changing --trending-threshold."
        ),
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

    # Set up basic logging (standalone mode, not using the scraper's
    # logging_config because this runs independently)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stats = run(
        db_path=args.db,
        batch_size=args.batch,
        limit=args.limit,
        trending_threshold=args.trending_threshold,
        recompute=args.recompute,
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
