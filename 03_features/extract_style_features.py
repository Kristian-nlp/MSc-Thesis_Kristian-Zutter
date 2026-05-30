"""
extract_style_features.py

Master's thesis: Learning the Levers — Which Creator-Controllable
Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
from a Swiss German-Language Perspective
Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026

Purpose:
    Batch text-style feature extraction. Reads caption_raw from the
    posts table and computes writing-style features: sentence counts,
    punctuation density, capitalisation, URLs and mentions, and the
    German-aware Flesch reading ease via textstat. Writes results to
    features_style. Incremental: only posts without an existing row
    are processed.

Inputs:
    01_config/settings.py             DB_PATH
    04_database/scraper.db            posts.caption_raw
    NLTK punkt_tab tokenizer data     downloaded on first run

Outputs:
    04_database/scraper.db            features_style table

Usage:
    python 03_features/extract_style_features.py
    python 03_features/extract_style_features.py --dry-run
"""

import logging
import re
import sqlite3
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

# Resolve numbered-directory imports (04_database/db.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "04_database"))

# ---------------------------------------------------------------------------
# Optional imports with clear error messages
# ---------------------------------------------------------------------------
try:
    import textstat
except ImportError:
    sys.exit(
        "ERROR: 'textstat' library not installed.\n"
        "  Install with:  pip install textstat --break-system-packages"
    )

try:
    import nltk
    from nltk.tokenize import sent_tokenize
except ImportError:
    sys.exit(
        "ERROR: 'nltk' library not installed.\n"
        "  Install with:  pip install nltk --break-system-packages"
    )

logger = logging.getLogger(__name__)


# ===================================================================
# Configuration
# ===================================================================

DB_PATH = Path(__file__).resolve().parent.parent / "04_database" / "scraper.db"
BATCH_SIZE = 500

# Minimum caption length for Flesch reading ease to be meaningful
FLESCH_MIN_CHARS = 30

# Pre-compiled regexes
_URL_REGEX = re.compile(r"https?://\S+", re.IGNORECASE)
_MENTION_REGEX = re.compile(r"@\w+", re.UNICODE)
_ELLIPSIS_REGEX = re.compile(r"\.{3,}|…")

# All Unicode punctuation categories: Pc, Pd, Pe, Pf, Pi, Po, Ps
_PUNCT_CATEGORIES = {"Pc", "Pd", "Pe", "Pf", "Pi", "Po", "Ps"}


# ===================================================================
# NLTK data setup
# ===================================================================

_nltk_ready = False


def _ensure_nltk_data():
    """Download punkt_tab tokenizer data if not already present."""
    global _nltk_ready
    if _nltk_ready:
        return

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        logger.info("Downloading NLTK punkt_tab tokenizer data...")
        nltk.download("punkt_tab", quiet=True)

    _nltk_ready = True


# ===================================================================
# Feature extraction functions
# ===================================================================

def compute_sentence_count(caption: str | None) -> int:
    """Count sentences using NLTK punkt tokenizer (German-aware)."""
    if not caption or not caption.strip():
        return 0
    _ensure_nltk_data()
    sentences = sent_tokenize(caption, language="german")
    return len(sentences)


def compute_avg_sentence_len(caption: str | None) -> float | None:
    """Mean words per sentence. Returns None if no sentences."""
    if not caption or not caption.strip():
        return None
    _ensure_nltk_data()
    sentences = sent_tokenize(caption, language="german")
    if not sentences:
        return None
    word_counts = [len(s.split()) for s in sentences]
    return sum(word_counts) / len(word_counts)


def compute_exclamation_density(caption: str | None) -> float:
    """Exclamation marks per character. Returns 0.0 for empty."""
    if not caption or len(caption) == 0:
        return 0.0
    count = caption.count("!") + caption.count("❗") + caption.count("‼")
    return count / len(caption)


def compute_question_density(caption: str | None) -> float:
    """Question marks per character. Returns 0.0 for empty."""
    if not caption or len(caption) == 0:
        return 0.0
    count = caption.count("?") + caption.count("❓") + caption.count("⁉")
    return count / len(caption)


def compute_ellipsis_count(caption: str | None) -> int:
    """Count ellipsis patterns ('...' or '…')."""
    if not caption:
        return 0
    return len(_ELLIPSIS_REGEX.findall(caption))


def compute_caps_ratio(caption: str | None) -> float:
    """Fraction of alphabetic characters that are uppercase."""
    if not caption:
        return 0.0
    alpha_chars = [c for c in caption if c.isalpha()]
    if not alpha_chars:
        return 0.0
    upper_count = sum(1 for c in alpha_chars if c.isupper())
    return upper_count / len(alpha_chars)


def compute_caps_word_count(caption: str | None) -> int:
    """Count ALL-CAPS words (2+ alphabetic characters)."""
    if not caption:
        return 0
    words = caption.split()
    count = 0
    for word in words:
        # Strip non-alpha chars from edges for checking
        alpha_only = "".join(c for c in word if c.isalpha())
        if len(alpha_only) >= 2 and alpha_only.isupper():
            count += 1
    return count


def compute_line_break_count(caption: str | None) -> int:
    """Count newline characters."""
    if not caption:
        return 0
    return caption.count("\n")


def compute_url_count(caption: str | None) -> int:
    """Count URLs in the caption."""
    if not caption:
        return 0
    return len(_URL_REGEX.findall(caption))


def compute_mention_count(caption: str | None) -> int:
    """Count @mentions in the caption."""
    if not caption:
        return 0
    return len(_MENTION_REGEX.findall(caption))


def compute_punct_diversity(caption: str | None) -> float:
    """
    Ratio of distinct Unicode punctuation types used to total
    punctuation characters. Measures writing sophistication.

    Uses unicodedata.category() to handle all Unicode punctuation,
    not just ASCII string.punctuation.
    """
    if not caption:
        return 0.0
    punct_chars = [c for c in caption if unicodedata.category(c) in _PUNCT_CATEGORIES]
    if not punct_chars:
        return 0.0
    distinct = len(set(punct_chars))
    return distinct / len(punct_chars)


def compute_flesch_reading_ease(caption: str | None) -> float | None:
    """
    Flesch reading ease score. German-aware via textstat.set_lang().

    Returns None for captions shorter than FLESCH_MIN_CHARS, as the
    formula produces meaningless results on very short text.
    """
    if not caption or len(caption.strip()) < FLESCH_MIN_CHARS:
        return None
    # Detect language: use German if caption looks German-ish,
    # otherwise default to English. We rely on features_text.lang
    # being populated first, but for simplicity we use textstat's
    # default which works reasonably for both.
    textstat.set_lang("de")
    try:
        score = textstat.flesch_reading_ease(caption)
        return round(score, 2)
    except Exception:
        return None


# ===================================================================
# Database interaction
# ===================================================================

def get_unprocessed_posts(
    conn: sqlite3.Connection,
    limit: int | None = None,
) -> list[dict]:
    """
    Fetch posts that do not yet have a features_style row.

    Args:
        conn: Open SQLite connection.
        limit: Max rows to fetch (None = all).

    Returns:
        List of dicts with post_id, caption_raw.
    """
    sql = """
        SELECT p.post_id, p.caption_raw
        FROM posts p
        LEFT JOIN features_style fs ON p.post_id = fs.post_id
        WHERE fs.post_id IS NULL
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
    Insert a batch of computed features into features_style.

    Uses INSERT OR IGNORE to be safe against duplicates if the
    script is interrupted and restarted.

    Args:
        conn: Open SQLite connection.
        features: List of dicts with keys matching features_style columns.

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
                    INSERT OR IGNORE INTO features_style (
                        post_id, sentence_count, avg_sentence_len,
                        exclamation_density, question_density, ellipsis_count,
                        caps_ratio, caps_word_count, line_break_count,
                        url_count, mention_count, punct_diversity,
                        flesch_reading_ease, processed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        feat["post_id"],
                        feat["sentence_count"],
                        feat["avg_sentence_len"],
                        feat["exclamation_density"],
                        feat["question_density"],
                        feat["ellipsis_count"],
                        feat["caps_ratio"],
                        feat["caps_word_count"],
                        feat["line_break_count"],
                        feat["url_count"],
                        feat["mention_count"],
                        feat["punct_diversity"],
                        feat["flesch_reading_ease"],
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

def process_post(row: dict) -> dict:
    """
    Compute all style features for a single post.

    Args:
        row: Dict with post_id, caption_raw.

    Returns:
        Dict ready for insertion into features_style.
    """
    caption = row.get("caption_raw") or ""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "post_id": row["post_id"],
        "sentence_count": compute_sentence_count(caption),
        "avg_sentence_len": compute_avg_sentence_len(caption),
        "exclamation_density": compute_exclamation_density(caption),
        "question_density": compute_question_density(caption),
        "ellipsis_count": compute_ellipsis_count(caption),
        "caps_ratio": compute_caps_ratio(caption),
        "caps_word_count": compute_caps_word_count(caption),
        "line_break_count": compute_line_break_count(caption),
        "url_count": compute_url_count(caption),
        "mention_count": compute_mention_count(caption),
        "punct_diversity": compute_punct_diversity(caption),
        "flesch_reading_ease": compute_flesch_reading_ease(caption),
        "processed_at": now,
    }


def run(
    db_path: Path | str | None = None,
    batch_size: int = BATCH_SIZE,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict:
    """
    Run the full style-feature extraction pipeline.

    Args:
        db_path: Path to the SQLite database. Defaults to /data/scraper.db.
        batch_size: Rows per DB transaction.
        limit: Max total posts to process (None = all pending).
        dry_run: If True, compute features but do not write to DB.

    Returns:
        Dict with summary stats: total, processed, skipped, errors.
    """
    path = Path(db_path) if db_path else DB_PATH
    logger.info("Style feature extraction starting (db=%s)", path)

    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")

    stats = {"total": 0, "processed": 0, "errors": 0}

    try:
        # Ensure NLTK data is available before processing
        _ensure_nltk_data()

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
            except Exception as e:
                stats["errors"] += 1
                logger.error(
                    "Error processing post %s: %s", row["post_id"], e
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
            "Style feature extraction complete: "
            "total=%d, processed=%d, errors=%d",
            stats["total"], stats["processed"], stats["errors"],
        )

    except Exception as e:
        logger.exception("Fatal error in style feature extraction: %s", e)
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
        python extract_style_features.py [--db PATH] [--batch N]
                                         [--limit N] [--dry-run]
                                         [--log-level LEVEL]
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract text-style features from scraped post captions.",
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
