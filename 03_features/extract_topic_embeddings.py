"""
extract_topic_embeddings.py

Master's thesis: Learning the Levers — Which Creator-Controllable
Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
from a Swiss German-Language Perspective
Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026

Purpose:
    Batch semantic-embedding pipeline. Reads caption_raw from the
    posts table, encodes each caption to a 384-dimensional dense
    vector with paraphrase-multilingual-MiniLM-L12-v2, and writes
    the result as a raw float32 BLOB into the topic_embedding column
    of features_text. Empty captions receive a zero vector so every
    row has a consistent-length embedding. CPU only.

Inputs:
    01_config/settings.py             DB_PATH
    04_database/scraper.db            posts.caption_raw and existing
                                      features_text rows (topic_embedding
                                      NULL — populated here)
    Hugging Face model cache          paraphrase-multilingual-MiniLM-L12-v2

Outputs:
    04_database/scraper.db            features_text.topic_embedding
                                      (BLOB, 384 * 4 = 1536 bytes)

Usage:
    python 03_features/extract_topic_embeddings.py
    python 03_features/extract_topic_embeddings.py --dry-run
    python 03_features/extract_topic_embeddings.py --batch-size 64
"""

import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# Resolve numbered-directory imports (04_database/db.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "04_database"))

import numpy as np

# ---------------------------------------------------------------------------
# Optional import with clear error message
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    sys.exit(
        "ERROR: 'sentence-transformers' library not installed.\n"
        "  Install with:  pip install sentence-transformers --break-system-packages"
    )

logger = logging.getLogger(__name__)


# ===================================================================
# Configuration
# ===================================================================

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384          # output dimension of this model
DB_PATH = Path(__file__).resolve().parent.parent / "04_database" / "scraper.db"

# How many captions to encode in one model.encode() call.
# Reduced from 256 to 64 to limit peak RAM on the e2-medium VM
# (2 vCPU, 4 GB). Throughput difference on CPU is minimal.
ENCODE_BATCH_SIZE = 64

# How many rows to commit to SQLite per transaction.
DB_BATCH_SIZE = 500

# Zero vector for empty captions (allocated once, reused)
_ZERO_VECTOR = np.zeros(EMBEDDING_DIM, dtype=np.float32)


# ===================================================================
# Model loading (lazy singleton)
# ===================================================================

_model = None


def _get_model() -> SentenceTransformer:
    """
    Load the sentence-transformers model (downloads on first run).

    The model is cached in ~/.cache/huggingface/ after the initial
    download (~90 MB). Subsequent calls return the cached object.

    Returns:
        SentenceTransformer model instance.
    """
    global _model
    if _model is not None:
        return _model

    logger.info("Loading sentence-transformers model: %s", MODEL_NAME)
    _model = SentenceTransformer(MODEL_NAME)
    logger.info(
        "Model loaded (embedding dim=%d, max_seq_length=%d)",
        _model.get_sentence_embedding_dimension(),
        _model.max_seq_length,
    )
    return _model


# ===================================================================
# Embedding helpers
# ===================================================================

def encode_captions(captions: list[str]) -> np.ndarray:
    """
    Encode a list of caption strings into a 2-D numpy array.

    Empty strings are handled by the model (they produce near-zero
    vectors), but we explicitly replace truly empty captions with
    zero vectors after encoding for consistency.

    Args:
        captions: List of N caption strings.

    Returns:
        np.ndarray of shape (N, 384), dtype float32.
    """
    model = _get_model()

    # sentence-transformers handles empty strings but logs warnings.
    # Replace empty/whitespace-only captions with a placeholder so the
    # model doesn't see truly empty input.  We overwrite them with
    # zero vectors afterwards.
    empty_mask = [not c or not c.strip() for c in captions]
    cleaned = [c if not empty else " " for c, empty in zip(captions, empty_mask)]

    embeddings = model.encode(
        cleaned,
        batch_size=ENCODE_BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,   # L2-normalise for cosine similarity
        convert_to_numpy=True,
    )

    # Overwrite empty-caption rows with the zero vector
    for i, is_empty in enumerate(empty_mask):
        if is_empty:
            embeddings[i] = _ZERO_VECTOR

    return embeddings.astype(np.float32)


def embedding_to_blob(vec: np.ndarray) -> bytes:
    """Convert a 1-D float32 numpy array to raw bytes for SQLite BLOB."""
    return vec.astype(np.float32).tobytes()


def blob_to_embedding(blob: bytes) -> np.ndarray:
    """
    Deserialise a BLOB back to a 1-D float32 numpy array.

    Utility for downstream code and the Parquet export pipeline.

    Args:
        blob: Raw bytes from SQLite (384 * 4 = 1536 bytes).

    Returns:
        np.ndarray of shape (384,), dtype float32.
    """
    return np.frombuffer(blob, dtype=np.float32)


# ===================================================================
# Database interaction
# ===================================================================

def get_pending_posts(
    conn: sqlite3.Connection,
    limit: int | None = None,
) -> list[dict]:
    """
    Fetch posts that have a features_text row but topic_embedding IS NULL.

    This means the text-feature stage has processed the post but
    the embedding has not yet been written. Ordering by created_at
    ensures deterministic processing order.

    Args:
        conn: Open SQLite connection.
        limit: Max rows to fetch (None = all).

    Returns:
        List of dicts with post_id and caption_raw.
    """
    sql = """
        SELECT ft.post_id, p.caption_raw
        FROM features_text ft
        INNER JOIN posts p ON ft.post_id = p.post_id
        WHERE ft.topic_embedding IS NULL
        ORDER BY p.created_at ASC
    """
    params: list = []
    if limit:
        sql += " LIMIT ?"
        params.append(int(limit))

    rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def write_embeddings_batch(
    conn: sqlite3.Connection,
    updates: list[tuple[bytes, str, str]],
) -> int:
    """
    Update topic_embedding and processed_at for a batch of posts.

    Uses UPDATE (not INSERT) because the row was created by the
    text-feature stage. Only updates topic_embedding and refreshes
    processed_at.

    Args:
        conn: Open SQLite connection.
        updates: List of (embedding_blob, processed_at, post_id) tuples.

    Returns:
        Number of rows successfully updated.
    """
    if not updates:
        return 0

    updated = 0
    with conn:
        for blob, ts, post_id in updates:
            try:
                cursor = conn.execute(
                    """
                    UPDATE features_text
                    SET topic_embedding = ?,
                        processed_at   = ?
                    WHERE post_id = ?
                    """,
                    (blob, ts, post_id),
                )
                if cursor.rowcount > 0:
                    updated += 1
            except sqlite3.Error as e:
                logger.error(
                    "Failed to update embedding for %s: %s",
                    post_id, e,
                )

    return updated


# ===================================================================
# Main pipeline
# ===================================================================

def run(
    db_path: Path | str | None = None,
    encode_batch: int = ENCODE_BATCH_SIZE,
    db_batch: int = DB_BATCH_SIZE,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict:
    """
    Run the full topic-embedding pipeline.

    1. Query all posts with topic_embedding IS NULL in features_text.
    2. Batch-encode captions with sentence-transformers.
    3. Write embedding BLOBs back to features_text.

    Args:
        db_path:       Path to SQLite database. Defaults to /data/scraper.db.
        encode_batch:  Captions per model.encode() call.
        db_batch:      Rows per DB transaction.
        limit:         Max posts to process (None = all pending).
        dry_run:       If True, encode but do not write to DB.

    Returns:
        Dict with summary stats: total, processed, errors.
    """
    path = Path(db_path) if db_path else DB_PATH
    logger.info("Topic embedding extraction starting (db=%s)", path)

    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")

    stats = {"total": 0, "processed": 0, "errors": 0}

    try:
        # 1. Fetch pending posts
        posts = get_pending_posts(conn, limit=limit)
        stats["total"] = len(posts)
        logger.info("Found %d posts pending embedding", len(posts))

        if not posts:
            logger.info("Nothing to do -- all embeddings already computed")
            return stats

        # 2. Eagerly load the model before the loop
        _get_model()

        # 3. Process in encoding batches
        for batch_start in range(0, len(posts), encode_batch):
            batch = posts[batch_start : batch_start + encode_batch]
            captions = [row.get("caption_raw") or "" for row in batch]
            post_ids = [row["post_id"] for row in batch]

            try:
                embeddings = encode_captions(captions)
            except Exception as e:
                logger.error(
                    "Encoding failed for batch starting at index %d: %s",
                    batch_start, e,
                )
                stats["errors"] += len(batch)
                continue

            # Build update tuples
            now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            updates = []
            for i, (emb, pid) in enumerate(zip(embeddings, post_ids)):
                try:
                    blob = embedding_to_blob(emb)
                    updates.append((blob, now, pid))
                except Exception as e:
                    logger.error(
                        "Serialisation failed for %s: %s", pid, e,
                    )
                    stats["errors"] += 1

            # Write to DB in sub-batches if needed
            if not dry_run:
                for db_start in range(0, len(updates), db_batch):
                    db_chunk = updates[db_start : db_start + db_batch]
                    written = write_embeddings_batch(conn, db_chunk)
                    stats["processed"] += written
            else:
                stats["processed"] += len(updates)

            logger.info(
                "Progress: %d / %d posts encoded",
                min(batch_start + encode_batch, len(posts)),
                len(posts),
            )

        logger.info(
            "Topic embedding extraction complete: "
            "total=%d, processed=%d, errors=%d",
            stats["total"], stats["processed"], stats["errors"],
        )

    except Exception as e:
        logger.exception("Fatal error in topic embedding extraction: %s", e)
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
        python extract_topic_embeddings.py [--db PATH] [--encode-batch N]
                                            [--db-batch N] [--limit N]
                                            [--dry-run] [--log-level LEVEL]
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Compute semantic topic embeddings for scraped post captions. "
            "Requires extract_text_features.py to have run first."
        ),
    )
    parser.add_argument(
        "--db", type=str, default=str(DB_PATH),
        help=f"Path to SQLite database (default: {DB_PATH})",
    )
    parser.add_argument(
        "--encode-batch", type=int, default=ENCODE_BATCH_SIZE,
        help=f"Captions per model.encode() call (default: {ENCODE_BATCH_SIZE})",
    )
    parser.add_argument(
        "--db-batch", type=int, default=DB_BATCH_SIZE,
        help=f"Rows per DB transaction (default: {DB_BATCH_SIZE})",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max posts to process (default: all pending)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Encode embeddings but do not write to the database",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Standalone logging (not the scraper's logging_config)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stats = run(
        db_path=args.db,
        encode_batch=args.encode_batch,
        db_batch=args.db_batch,
        limit=args.limit,
        dry_run=args.dry_run,
    )

    if stats["errors"] > 0:
        logger.warning(
            "Completed with %d errors out of %d posts",
            stats["errors"], stats["total"],
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
