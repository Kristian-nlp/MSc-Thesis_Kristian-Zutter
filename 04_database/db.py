"""
db.py

Master's thesis: Learning the Levers — Which Creator-Controllable
Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
from a Swiss German-Language Perspective
Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026

Purpose:
    Database persistence layer for the scraper. Takes a SnapshotResult
    from any platform scraper and writes it to SQLite in a single
    atomic transaction across the snapshots, posts, captures,
    counters, and scrape_log tables. Initialises the schema on first
    use, supports JSON fallback recovery, and computes T0 engagement
    counters in the same write.

Inputs:
    04_database/schema.sql            SQL schema (run on first use)
    04_database/scraper.db            target database
    DATA_DIR/fallback/*.json          recoverable fallback envelopes

Outputs:
    04_database/scraper.db            snapshots, posts, captures,
                                      counters, scrape_log rows
    04_database/fallback/*.json       fallback envelopes on DB error

Usage:
    from db import write_snapshot, recover_from_json, get_connection
    write_snapshot(snapshot_result)
    Imported by the scraper and feature-extraction orchestrators.
"""

import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

# Import scraper data classes
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "02_scraper" / "02_platforms"))
from base import CapturedPost, SnapshotResult

_SENTINEL = object()  # distinguishes "not passed" from explicit None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DB_PATH = Path(__file__).resolve().parent / "scraper.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"
TEMP_DIR = Path(__file__).resolve().parent / "fallback"  # repo-relative fallback dir


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

def get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """
    Open a connection to the SQLite database.

    On first use (empty database), runs schema.sql to create all tables.
    Enables WAL mode and foreign keys as specified in the schema.

    Args:
        db_path: Path to the database file. Defaults to repo-relative 04_database/scraper.db.

    Returns:
        sqlite3.Connection with row_factory set to sqlite3.Row.
    """
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")

    # Initialise schema if tables do not exist
    if not _tables_exist(conn):
        _init_schema(conn)

    return conn


def _tables_exist(conn: sqlite3.Connection) -> bool:
    """Check whether the core tables already exist."""
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='snapshots'"
    )
    return cursor.fetchone() is not None


def _init_schema(conn: sqlite3.Connection) -> None:
    """
    Run schema.sql to create all tables and indexes.

    Falls back to a minimal inline schema if schema.sql is not found,
    so the scraper can still run even if deployed without the SQL file.
    """
    if SCHEMA_PATH.exists():
        logger.info("Initialising database from %s", SCHEMA_PATH)
        sql = SCHEMA_PATH.read_text(encoding="utf-8")
        conn.executescript(sql)
        logger.info("Database schema created successfully")
    else:
        logger.warning(
            "schema.sql not found at %s. "
            "Please ensure the database is initialised before scraping.",
            SCHEMA_PATH,
        )
        raise FileNotFoundError(
            f"schema.sql not found at {SCHEMA_PATH}. "
            f"Run: sqlite3 {DB_PATH} < schema.sql"
        )


# ---------------------------------------------------------------------------
# Core write function
# ---------------------------------------------------------------------------

def write_snapshot(
    result: SnapshotResult,
    db_path: Path | str | None = None,
) -> bool:
    """
    Persist a complete SnapshotResult to SQLite.

    Wraps all data writes in a single transaction for atomicity.
    If the transaction fails, dumps the snapshot to a temp JSON file
    so it can be recovered later.

    The scrape_log entry is written after the main data write so that
    the snapshot FK exists. For failed scrapes, scrape_log is written
    with snapshot_id=NULL to avoid FK violations.

    Args:
        result: SnapshotResult from any platform scraper.
        db_path: Override database path (useful for testing).

    Returns:
        True if the write succeeded, False if it fell back to JSON.
    """
    # If the scrape itself failed, there is no data to write
    if not result.success:
        logger.info(
            "Snapshot %s failed at scrape time, nothing to persist",
            result.snapshot_id,
        )
        # Log the failed scrape with NULL snapshot_id (no FK reference)
        _write_scrape_log(result, db_path, snapshot_id_override=None)
        return True  # Not a DB failure, just no data

    all_posts = result.top_posts + result.baseline_posts
    if not all_posts:
        logger.warning(
            "Snapshot %s succeeded but has no posts to write",
            result.snapshot_id,
        )
        return True

    conn = None
    try:
        conn = get_connection(db_path)

        with conn:  # context manager handles COMMIT / ROLLBACK
            # 1. Insert snapshot
            _insert_snapshot(conn, result)

            # 2. Insert posts (deduplicated)
            for post in all_posts:
                _upsert_post(conn, post, result.platform)

            # 3. Insert captures (always new rows)
            for post in all_posts:
                _insert_capture(conn, result.snapshot_id, post, result.platform)

            # 4. Insert T0 counters
            for post in all_posts:
                _insert_t0_counters(conn, post, result.platform, result.captured_at_utc)

        logger.info(
            "Snapshot %s written: %d posts, %d captures, %d counters",
            result.snapshot_id,
            len(all_posts),
            len(all_posts),
            sum(1 for p in all_posts if _has_counters(p)),
        )
        # Snapshot row exists now, safe to write scrape_log with FK
        _write_scrape_log(result, db_path)
        return True

    except sqlite3.Error as e:
        logger.error(
            "SQLite write failed for snapshot %s: %s. "
            "Falling back to temp JSON.",
            result.snapshot_id,
            e,
        )
        _save_to_temp_json(result)
        # Snapshot row may not exist; use NULL snapshot_id
        _write_scrape_log(result, db_path, snapshot_id_override=None)
        return False

    except Exception as e:
        logger.exception(
            "Unexpected error writing snapshot %s: %s",
            result.snapshot_id,
            e,
        )
        _save_to_temp_json(result)
        _write_scrape_log(result, db_path, snapshot_id_override=None)
        return False

    finally:
        if conn:
            conn.close()


# ---------------------------------------------------------------------------
# Table-specific insert functions
# ---------------------------------------------------------------------------

def _insert_snapshot(conn: sqlite3.Connection, result: SnapshotResult) -> None:
    """Insert one row into the snapshots table."""
    conn.execute(
        """
        INSERT INTO snapshots (
            snapshot_id, platform, account_type, surface,
            captured_at_utc, timezone
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            result.snapshot_id,
            result.platform,
            result.account_type,
            result.surface,
            result.captured_at_utc,
            result.timezone,
        ),
    )


def _upsert_post(
    conn: sqlite3.Connection,
    post: CapturedPost,
    platform: str,
) -> None:
    """
    Insert a post if it does not already exist.

    Uses INSERT OR IGNORE on the UNIQUE(platform, permalink) constraint.
    The first time a post is seen, all metadata is stored. On subsequent
    appearances, the existing row is kept (captures still record the
    new observation).

    Author pseudonymisation: raw author_id and author_handle are
    hashed to SHA-256 at this point. The raw values are never stored
    in the database.
    """
    post_id = _make_post_id(platform, post.post_id)
    author_hash = _hash_author(post.author_id, post.author_handle)
    hashtags_raw = ",".join(post.hashtags) if post.hashtags else None

    conn.execute(
        """
        INSERT OR IGNORE INTO posts (
            post_id, platform, permalink, media_type, author_hash,
            posted_at_utc, follower_count, caption_raw, hashtags_raw,
            thumbnail_url, audio_present, audio_id, audio_name
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            post_id,
            platform,
            post.permalink or None,
            post.media_type or None,
            author_hash,
            post.posted_at_utc or None,
            post.followers,
            post.caption or None,
            hashtags_raw,
            post.thumbnail_url or None,
            1 if (post.audio_present if post.audio_present is not None else bool(post.audio_id)) else 0,
            post.audio_id or None,
            post.audio_name or None,
        ),
    )


def _insert_capture(
    conn: sqlite3.Connection,
    snapshot_id: str,
    post: CapturedPost,
    platform: str,
) -> None:
    """Insert one row into the captures table."""
    post_id = _make_post_id(platform, post.post_id)

    # join_key: the permalink or native post ID used to merge DOM + JSON
    join_key = post.permalink or post.post_id or None
    # source: which extraction path produced this record
    source = post.source or None  # "dom", "json", or "joined"

    conn.execute(
        """
        INSERT INTO captures (
            snapshot_id, post_id, rank_observed, is_top, join_key, source
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            snapshot_id,
            post_id,
            post.rank_observed,
            1 if post.is_top else 0,
            join_key,
            source,
        ),
    )


def _insert_t0_counters(
    conn: sqlite3.Connection,
    post: CapturedPost,
    platform: str,
    captured_at_utc: str | None = None,
) -> None:
    """
    Insert T0 engagement counters for a post.

    Uses INSERT ... ON CONFLICT DO UPDATE to merge non-null values.
    The first T0 observation establishes the row; subsequent
    observations fill in any previously-null counters without
    overwriting existing data.

    Only inserts if the post has at least one non-null counter.
    """
    if not _has_counters(post):
        return

    post_id = _make_post_id(platform, post.post_id)

    conn.execute(
        """
        INSERT INTO counters (
            post_id, revisit_type, likes, comments, shares, views,
            follower_count, captured_at_utc
        ) VALUES (?, 't0', ?, ?, ?, ?, ?, ?)
        ON CONFLICT (post_id, revisit_type) DO UPDATE SET
            likes          = COALESCE(counters.likes, excluded.likes),
            comments       = COALESCE(counters.comments, excluded.comments),
            shares         = COALESCE(counters.shares, excluded.shares),
            views          = COALESCE(counters.views, excluded.views),
            follower_count = COALESCE(counters.follower_count, excluded.follower_count)
        """,
        (
            post_id,
            post.likes,
            post.comments,
            post.shares,
            post.views,
            post.followers,
            captured_at_utc or datetime.now(timezone.utc).isoformat(),
        ),
    )


def _has_counters(post: CapturedPost) -> bool:
    """Check if a post has any engagement counters worth storing."""
    return any(v is not None for v in (post.likes, post.comments, post.shares, post.views))


# ---------------------------------------------------------------------------
# Scrape log (written independently of the main transaction)
# ---------------------------------------------------------------------------

def _write_scrape_log(
    result: SnapshotResult,
    db_path: Path | str | None = None,
    snapshot_id_override: str | None = _SENTINEL,
) -> None:
    """
    Write a scrape_log entry for monitoring.

    Called after the main data write (or after fallback) so the
    snapshot FK exists when referenced. For failed scrapes where
    no snapshot row was created, pass snapshot_id_override=None
    to insert NULL instead.
    """
    conn = None
    try:
        conn = get_connection(db_path)

        # Determine status
        total_posts = len(result.top_posts) + len(result.baseline_posts)
        if result.success and total_posts > 0:
            status = "success"
        elif result.success and total_posts == 0:
            status = "partial"
        else:
            status = "failure"

        # Resolve snapshot_id: use override if provided, else from result
        sid = result.snapshot_id if snapshot_id_override is _SENTINEL else snapshot_id_override

        # Calculate started_at from captured_at and duration
        started_at = result.captured_at_utc
        finished_at = datetime.now(timezone.utc).isoformat()

        conn.execute(
            """
            INSERT INTO scrape_log (
                snapshot_id, platform, account_type, status,
                posts_captured, error_message, duration_sec,
                started_at_utc, finished_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sid,
                result.platform,
                result.account_type,
                status,
                total_posts,
                result.error_message or None,
                result.duration_seconds,
                started_at,
                finished_at,
            ),
        )
        conn.commit()

    except Exception as e:
        logger.error("Failed to write scrape_log: %s", e)

    finally:
        if conn:
            conn.close()


# ---------------------------------------------------------------------------
# JSON fallback
# ---------------------------------------------------------------------------

def _save_to_temp_json(result: SnapshotResult) -> Path | None:
    """
    Dump a SnapshotResult to a temp JSON file when SQLite write fails.

    Files are saved with the snapshot_id and timestamp so they can
    be replayed later via recover_from_json().
    """
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"snapshot_{result.snapshot_id}_{timestamp}.json"
    filepath = TEMP_DIR / filename

    try:
        data = _snapshot_to_dict(result)
        filepath.write_text(
            json.dumps(data, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Snapshot saved to temp JSON: %s", filepath)
        return filepath

    except Exception as e:
        logger.error("Failed to save temp JSON: %s", e)
        return None


def _snapshot_to_dict(result: SnapshotResult) -> dict:
    """Convert a SnapshotResult to a JSON-serialisable dictionary."""
    return {
        "snapshot_id": result.snapshot_id,
        "platform": result.platform,
        "account_type": result.account_type,
        "surface": result.surface,
        "captured_at_utc": result.captured_at_utc,
        "timezone": result.timezone,
        "success": result.success,
        "error_message": result.error_message,
        "duration_seconds": result.duration_seconds,
        "top_posts": [_post_to_dict(p) for p in result.top_posts],
        "baseline_posts": [_post_to_dict(p) for p in result.baseline_posts],
    }


def _post_to_dict(post: CapturedPost) -> dict:
    """Convert a CapturedPost to a JSON-serialisable dictionary.

    Author fields are hashed before serialisation so that raw PII
    is never written to the fallback JSON files on disc (FADP compliance).
    raw_data is deliberately excluded: it may contain unhashed author
    identifiers from platform API responses.
    """
    return {
        "post_id": post.post_id,
        "permalink": post.permalink,
        "rank_observed": post.rank_observed,
        "is_top": post.is_top,
        "media_type": post.media_type,
        "caption": post.caption,
        "author_hash": _hash_author(post.author_id, post.author_handle),
        "likes": post.likes,
        "comments": post.comments,
        "shares": post.shares,
        "views": post.views,
        "followers": post.followers,
        "posted_at_utc": post.posted_at_utc,
        "hashtags": post.hashtags,
        "audio_present": post.audio_present,
        "audio_id": post.audio_id,
        "audio_name": post.audio_name,
        "audio_is_original": getattr(post, "audio_is_original", None),
        "thumbnail_url": post.thumbnail_url,
        "source": post.source,
    }


def recover_from_json(
    json_dir: Path | str | None = None,
    db_path: Path | str | None = None,
) -> dict:
    """
    Replay temp JSON files back into SQLite.

    Call this after fixing whatever caused the original write failure
    (e.g. disk full, database locked). Successfully replayed files
    are moved to a 'recovered/' subdirectory.

    Args:
        json_dir: Directory containing temp JSON files.
        db_path: Override database path.

    Returns:
        Dict with counts: {"replayed": N, "failed": N, "skipped": N}.
    """
    json_dir = Path(json_dir) if json_dir else TEMP_DIR
    recovered_dir = json_dir / "recovered"
    recovered_dir.mkdir(parents=True, exist_ok=True)

    stats = {"replayed": 0, "failed": 0, "skipped": 0}

    if not json_dir.exists():
        logger.info("No temp directory found at %s", json_dir)
        return stats

    json_files = sorted(json_dir.glob("snapshot_*.json"))
    if not json_files:
        logger.info("No temp JSON files to recover")
        return stats

    logger.info("Found %d temp JSON files to recover", len(json_files))

    for filepath in json_files:
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
            result = _dict_to_snapshot(data)

            if write_snapshot(result, db_path):
                filepath.rename(recovered_dir / filepath.name)
                stats["replayed"] += 1
                logger.info("Recovered: %s", filepath.name)
            else:
                stats["failed"] += 1
                logger.warning("Recovery failed again: %s", filepath.name)

        except Exception as e:
            stats["failed"] += 1
            logger.error("Error recovering %s: %s", filepath.name, e)

    logger.info(
        "Recovery complete: %d replayed, %d failed, %d skipped",
        stats["replayed"],
        stats["failed"],
        stats["skipped"],
    )
    return stats


def _dict_to_snapshot(data: dict) -> SnapshotResult:
    """Reconstruct a SnapshotResult from a JSON dictionary."""
    result = SnapshotResult(
        snapshot_id=data["snapshot_id"],
        platform=data["platform"],
        account_type=data["account_type"],
        surface=data["surface"],
        captured_at_utc=data["captured_at_utc"],
        timezone=data.get("timezone", "Europe/Zurich"),
        success=data.get("success", True),
        error_message=data.get("error_message", ""),
        duration_seconds=data.get("duration_seconds", 0.0),
    )

    result.top_posts = [_dict_to_post(p) for p in data.get("top_posts", [])]
    result.baseline_posts = [_dict_to_post(p) for p in data.get("baseline_posts", [])]

    return result


def _dict_to_post(data: dict) -> CapturedPost:
    """Reconstruct a CapturedPost from a JSON dictionary.

    Handles both old format (author_id/author_handle) and new format
    (author_hash) for backward compatibility with existing fallback files.
    """
    # New format stores author_hash; old format stores raw author_id/handle
    author_id = data.get("author_id", "")
    author_handle = data.get("author_handle", "")
    if not author_id and not author_handle and data.get("author_hash"):
        # Pre-hashed: put it in author_id so _upsert_post passes it through
        author_id = data["author_hash"]

    return CapturedPost(
        post_id=data.get("post_id", ""),
        permalink=data.get("permalink", ""),
        rank_observed=data.get("rank_observed", 0),
        is_top=data.get("is_top", False),
        media_type=data.get("media_type", ""),
        caption=data.get("caption", ""),
        author_id=author_id,
        author_handle=author_handle,
        likes=data.get("likes"),
        comments=data.get("comments"),
        shares=data.get("shares"),
        views=data.get("views"),
        followers=data.get("followers"),
        posted_at_utc=data.get("posted_at_utc", ""),
        hashtags=data.get("hashtags", []),
        audio_present=data.get("audio_present"),
        audio_id=data.get("audio_id", ""),
        audio_name=data.get("audio_name", ""),
        audio_is_original=data.get("audio_is_original"),
        thumbnail_url=data.get("thumbnail_url", ""),
        source=data.get("source", ""),
        raw_data=data.get("raw_data", {}),
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _make_post_id(platform: str, native_id: str) -> str:
    """
    Create a globally unique post_id by prefixing with platform.

    Examples:
        _make_post_id("tiktok", "7608611486151281942")
        -> "tiktok_7608611486151281942"

    This avoids collisions when posts from different platforms
    share numeric ID spaces.
    """
    if not native_id:
        # Fallback: generate a hash from timestamp to avoid empty IDs
        fallback = hashlib.sha256(
            datetime.now(timezone.utc).isoformat().encode()
        ).hexdigest()[:16]
        logger.warning("Empty native post_id, using fallback: %s", fallback)
        return f"{platform}_{fallback}"

    # If the native_id already has the platform prefix, don't double it
    if native_id.startswith(f"{platform}_"):
        return native_id

    return f"{platform}_{native_id}"


def _is_sha256_hex(value: str) -> bool:
    """Check if a string is already a SHA-256 hex digest (64 hex chars)."""
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _hash_author(author_id: str, author_handle: str) -> str | None:
    """
    Hash the author identifier for privacy (FADP compliance).

    Uses SHA-256. Prefers author_id (stable) over author_handle
    (can change). Returns None if neither is available.

    If the input is already a SHA-256 hex digest (64 hex chars),
    it is returned as-is to avoid double-hashing during JSON
    fallback recovery.

    Ref: schema.sql design notes - "author_hash: SHA-256 of the raw
         author/username at ingestion time (privacy by design)."
    """
    raw = author_id or author_handle
    if not raw:
        return None

    if _is_sha256_hex(raw):
        return raw

    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
