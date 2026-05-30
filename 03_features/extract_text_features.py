"""
extract_text_features.py

Master's thesis: Learning the Levers — Which Creator-Controllable
Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
from a Swiss German-Language Perspective
Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026

Purpose:
    Batch text-feature extraction. Reads caption_raw and hashtags_raw
    from the posts table and writes caption length, word count,
    hashtag count, emoji count, a bilingual CTA flag, and a fastText
    language code into the features_text table. Incremental: only
    processes posts that do not yet have a features_text row.

Inputs:
    01_config/settings.py             DB_PATH, DATA_DIR
    04_database/scraper.db            posts table (caption_raw,
                                      hashtags_raw)
    DATA_DIR/models/lid.176.ftz       fastText language ID model
                                      (downloaded on first run)

Outputs:
    04_database/scraper.db            features_text table (one row
                                      per post)

Usage:
    python 03_features/extract_text_features.py
    python 03_features/extract_text_features.py --dry-run
    python 03_features/extract_text_features.py --db /data/scraper.db
"""

import logging
import re
import sqlite3
import sys
import urllib.request
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
    import emoji
except ImportError:
    sys.exit(
        "ERROR: 'emoji' library not installed.\n"
        "  Install with:  pip install emoji --break-system-packages"
    )

try:
    import fasttext
except ImportError:
    sys.exit(
        "ERROR: 'fasttext' library not installed.\n"
        "  Install with:  pip install fasttext-wheel --break-system-packages\n"
        "  (or: pip install fasttext --break-system-packages)"
    )

# The fastText load_model() prints a deprecation warning to stderr.
# We suppress it only during model loading (see _get_langid_model)
# rather than globally monkey-patching eprint, which would hide
# genuine errors.
import contextlib
import io

logger = logging.getLogger(__name__)


# ===================================================================
# Configuration
# ===================================================================

DB_PATH = Path(__file__).resolve().parent.parent / "04_database" / "scraper.db"
MODEL_DIR = Path(__file__).resolve().parent.parent / "04_database" / "models"
LANGID_MODEL_URL = (
    "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
)
LANGID_MODEL_PATH = MODEL_DIR / "lid.176.ftz"

# How many posts to process per DB transaction
BATCH_SIZE = 500


# ===================================================================
# CTA patterns  (bilingual DE / EN)
# ===================================================================
# Each pattern is compiled as case-insensitive. The list covers
# common social-media calls to action in both German and English,
# as posts in the dataset may mix the two languages.
#
# Grouped for readability; flattened into a single compiled regex
# at module load time for performance.

_CTA_PATTERNS_RAW: list[str] = [
    # --- English ---
    r"link in bio",
    r"check.{0,5}link",
    r"tap.{0,5}link",
    r"click.{0,5}link",
    r"swipe up",
    r"comment below",
    r"drop a comment",
    r"leave a comment",
    r"tag a friend",
    r"tag someone",
    r"share this",
    r"share with",
    r"save this",
    r"save for later",
    r"follow (?:me|us|for)",
    r"subscribe",
    r"sign up",
    r"dm (?:me|us|for)",
    r"send (?:me|us) a",
    r"tell (?:me|us) in",
    r"let (?:me|us) know",
    r"what do you think",
    r"do you agree",
    r"double tap",
    r"like if",
    r"repost",
    r"turn on.{0,10}notification",
    r"shop (?:now|the|my|our)",
    r"buy now",
    r"order (?:now|here|today)",
    r"use (?:my |our )?code",
    r"grab yours",
    r"get yours",
    r"learn more",
    r"find out",
    r"read more",
    r"watch (?:the |my |our )?(?:full|entire|whole)",
    r"check.{0,5}out",
    r"don.t miss",
    r"limited (?:time|offer|edition)",
    r"giveaway",
    r"enter to win",
    r"try it",

    # --- German ---
    r"link in (?:der )?bio",
    r"jetzt (?:kaufen|bestellen|anmelden|sichern|entdecken)",
    r"klick.{0,5}(?:link|hier)",
    r"schreib.{0,10}(?:kommentar|mir|uns)",
    r"kommentier",
    r"markier.{0,5}(?:jemand|freund|eine)",
    r"teil.{0,5}(?:das|dies|es|mit)",
    r"speicher.{0,5}(?:dir|das|dies)",
    r"folg.{0,5}(?:mir|uns|f.r)",
    r"abonnier",
    r"schick.{0,5}(?:mir|uns)",
    r"sag.{0,10}(?:mir|uns|bescheid)",
    r"was (?:denkst|meinst|sagst) du",
    r"(?:doppel|zweimal).{0,5}tipp",
    r"(?:hier|jetzt) bestellen",
    r"hol.{0,5}dir",
    r"(?:mehr|weiter).{0,5}(?:erfahren|lesen)",
    r"schau.{0,10}(?:dir|mal|vorbei|rein)",
    r"verpass.{0,5}(?:nicht|es)",
    r"gewinnspiel",
    r"mitmachen",
    r"probier.{0,5}(?:es|mal)",
]

# Compile into a single alternation regex for speed
_CTA_REGEX = re.compile(
    "|".join(f"(?:{p})" for p in _CTA_PATTERNS_RAW),
    flags=re.IGNORECASE,
)

# Regex to extract hashtags from caption text (fallback)
_HASHTAG_REGEX = re.compile(r"#(\w+)", re.UNICODE)


# ===================================================================
# fastText language identification model
# ===================================================================

_langid_model = None  # lazy-loaded singleton


def _get_langid_model():
    """
    Load (and download if needed) the fastText lid.176.ftz model.

    The model is cached in DATA_DIR/models/ so the download only
    happens once. Subsequent calls return the cached model object.

    Returns:
        fasttext.FastText._FastText model instance.
    """
    global _langid_model
    if _langid_model is not None:
        return _langid_model

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not LANGID_MODEL_PATH.exists():
        logger.info(
            "Downloading fastText langid model to %s ...", LANGID_MODEL_PATH
        )
        urllib.request.urlretrieve(LANGID_MODEL_URL, str(LANGID_MODEL_PATH))
        logger.info("Download complete (%.1f MB)", LANGID_MODEL_PATH.stat().st_size / 1e6)

    # Suppress the one-time deprecation warning that fastText prints to stderr
    with contextlib.redirect_stderr(io.StringIO()):
        _langid_model = fasttext.load_model(str(LANGID_MODEL_PATH))
    logger.info("fastText langid model loaded")
    return _langid_model


# ===================================================================
# Feature extraction functions
# ===================================================================

def compute_caption_len(caption: str | None) -> int:
    """Character count of the caption. Returns 0 for empty/NULL."""
    if not caption:
        return 0
    return len(caption)


def compute_word_count(caption: str | None) -> int:
    """
    Token count via whitespace split.

    Simple and robust for short, multilingual social-media text.
    Strips URLs and hashtag '#' symbols before counting so that
    '#summer' is counted as one word ('summer').
    """
    if not caption:
        return 0
    # Remove URLs to avoid inflating count with long links
    text = re.sub(r"https?://\S+", "", caption)
    # Strip hashtag symbols so '#summer' becomes 'summer'
    text = text.replace("#", "")
    tokens = text.split()
    return len(tokens)


def compute_hashtag_count(
    hashtags_raw: str | None,
    caption: str | None,
) -> int:
    """
    Count hashtags, preferring the pre-extracted hashtags_raw column.

    Falls back to regex extraction from caption_raw if hashtags_raw
    is empty (e.g. if the scraper's DOM parser missed them but they
    exist in the caption text).

    Args:
        hashtags_raw: Comma-separated hashtag list from posts table.
        caption: Raw caption text.

    Returns:
        Integer hashtag count.
    """
    # Primary: use the pre-extracted column
    if hashtags_raw and hashtags_raw.strip():
        tags = [t.strip() for t in hashtags_raw.split(",") if t.strip()]
        if tags:
            return len(tags)

    # Fallback: extract from caption via regex
    if caption:
        matches = _HASHTAG_REGEX.findall(caption)
        return len(matches)

    return 0


def compute_emoji_count(caption: str | None) -> int:
    """
    Count emoji characters in the caption.

    Uses the emoji library which handles multi-codepoint emoji
    (flags, skin tones, ZWJ sequences) correctly.
    """
    if not caption:
        return 0
    return emoji.emoji_count(caption)


def compute_cta_flag(caption: str | None) -> int:
    """
    Detect whether the caption contains a call-to-action phrase.

    Returns 1 if any CTA pattern matches, 0 otherwise.
    Bilingual DE/EN pattern list -- see _CTA_PATTERNS_RAW.
    """
    if not caption:
        return 0
    return 1 if _CTA_REGEX.search(caption) else 0


def detect_language(caption: str | None) -> str | None:
    """
    Detect the primary language of the caption using fastText.

    Returns an ISO 639-1 code (e.g. 'de', 'en', 'fr') or None if
    the caption is too short or empty to classify reliably.

    fastText's lid model expects a single line of text, so we
    collapse newlines and strip whitespace.
    """
    if not caption or len(caption.strip()) < 5:
        return None

    model = _get_langid_model()

    # fastText expects single-line input
    clean = caption.replace("\n", " ").strip()

    predictions = model.predict(clean, k=1)
    # predictions = (('__label__de',), array([0.95]))
    label = predictions[0][0]  # '__label__de'
    lang_code = label.replace("__label__", "")

    return lang_code


# ===================================================================
# Database interaction
# ===================================================================

def get_unprocessed_posts(
    conn: sqlite3.Connection,
    limit: int | None = None,
) -> list[dict]:
    """
    Fetch posts that do not yet have a features_text row.

    Args:
        conn: Open SQLite connection.
        limit: Max rows to fetch (None = all).

    Returns:
        List of dicts with post_id, caption_raw, hashtags_raw.
    """
    sql = """
        SELECT p.post_id, p.caption_raw, p.hashtags_raw
        FROM posts p
        LEFT JOIN features_text ft ON p.post_id = ft.post_id
        WHERE ft.post_id IS NULL
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
    Insert a batch of computed features into features_text.

    Uses INSERT OR IGNORE to be safe against duplicates if the
    script is interrupted and restarted.

    Args:
        conn: Open SQLite connection.
        features: List of dicts with keys matching features_text columns.

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
                    INSERT OR IGNORE INTO features_text (
                        post_id, caption_len, word_count, hashtag_count,
                        emoji_count, cta_flag, lang, processed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        feat["post_id"],
                        feat["caption_len"],
                        feat["word_count"],
                        feat["hashtag_count"],
                        feat["emoji_count"],
                        feat["cta_flag"],
                        feat["lang"],
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
    Compute all text features for a single post.

    Args:
        row: Dict with post_id, caption_raw, hashtags_raw.

    Returns:
        Dict ready for insertion into features_text.
    """
    caption = row.get("caption_raw") or ""
    hashtags_raw = row.get("hashtags_raw") or ""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "post_id": row["post_id"],
        "caption_len": compute_caption_len(caption),
        "word_count": compute_word_count(caption),
        "hashtag_count": compute_hashtag_count(hashtags_raw, caption),
        "emoji_count": compute_emoji_count(caption),
        "cta_flag": compute_cta_flag(caption),
        "lang": detect_language(caption),
        "processed_at": now,
    }


def run(
    db_path: Path | str | None = None,
    batch_size: int = BATCH_SIZE,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict:
    """
    Run the full text-feature extraction pipeline.

    Args:
        db_path: Path to the SQLite database. Defaults to /data/scraper.db.
        batch_size: Rows per DB transaction.
        limit: Max total posts to process (None = all pending).
        dry_run: If True, compute features but do not write to DB.

    Returns:
        Dict with summary stats: total, processed, skipped, errors.
    """
    path = Path(db_path) if db_path else DB_PATH
    logger.info("Text feature extraction starting (db=%s)", path)

    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")

    stats = {"total": 0, "processed": 0, "errors": 0}

    try:
        # Fetch unprocessed posts
        posts = get_unprocessed_posts(conn, limit=limit)
        stats["total"] = len(posts)
        logger.info("Found %d unprocessed posts", len(posts))

        if not posts:
            logger.info("Nothing to do -- all posts already processed")
            return stats

        # Eagerly load the langid model before the loop
        _get_langid_model()

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
            "Text feature extraction complete: "
            "total=%d, processed=%d, errors=%d",
            stats["total"], stats["processed"], stats["errors"],
        )

    except Exception as e:
        logger.exception("Fatal error in text feature extraction: %s", e)
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
        python extract_text_features.py [--db PATH] [--batch N]
                                        [--limit N] [--dry-run]
                                        [--log-level LEVEL]
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract text features from scraped post captions.",
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
