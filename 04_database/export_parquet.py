"""
export_parquet.py

Master's thesis: Learning the Levers — Which Creator-Controllable
Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
from a Swiss German-Language Perspective
Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026

Purpose:
    Export model-ready data from SQLite to Parquet files, partitioned
    by platform and capture_date. Joins the snapshots, posts,
    captures, counters, and per-stage features tables; serialises the
    384-dimensional topic embedding as a list column; and emits a
    round-trip test option to validate column types.

Inputs:
    04_database/scraper.db            source database (snapshots,
                                      posts, captures, counters,
                                      features_text, features_topic,
                                      features_style, features_audio,
                                      features_visual, features_temporal)

Outputs:
    04_database/exports/platform=<P>/capture_date=<D>/part-0.parquet

Usage:
    python 04_database/export_parquet.py
    python 04_database/export_parquet.py --platform tiktok
    python 04_database/export_parquet.py --test
"""

import argparse
import logging
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from db import get_connection


# ---------------------------------------------------------------------------
# Topic embedding configuration
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384  # paraphrase-multilingual-MiniLM-L12-v2 output dimension


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DB_PATH = str(Path(__file__).resolve().parent / "scraper.db")
EXPORT_DIR = str(Path(__file__).resolve().parent / "exports")


# ---------------------------------------------------------------------------
# 1. Extract joined dataset from SQLite
# ---------------------------------------------------------------------------
def extract_from_sqlite(db_path: str, platform: str | None = None) -> pd.DataFrame:
    """
    Join core tables into a single model-ready DataFrame.

    Joins: snapshots + captures + posts + counters (T0) + all four
    feature tables (left joins, as features may not yet be populated).

    Matches the database schema defined in 04_database/schema.sql:
        platform, account_type, captured_at, snapshot_id, rank, ...
    """

    query = """
    SELECT
        -- Snapshot context
        s.snapshot_id,
        s.platform,
        s.account_type,
        s.surface,
        s.captured_at_utc,
        DATE(s.captured_at_utc)         AS capture_date,         -- UTC date (for partitioning)
        DATE(s.captured_at_utc, '+1 hour') AS capture_date_local, -- approx CET (analysis reference)

        -- Capture / visibility
        c.rank_observed,
        c.is_top,

        -- Post metadata
        p.post_id,
        p.permalink,
        p.media_type,
        p.author_hash,
        p.posted_at_utc,
        p.follower_count,
        p.caption_raw,
        p.hashtags_raw,
        p.thumbnail_url,
        p.audio_present                 AS audio_present_raw,
        p.audio_id                      AS audio_id_raw,
        p.audio_name,

        -- Counters at T0
        ct0.likes                       AS likes_t0,
        ct0.comments                    AS comments_t0,
        ct0.shares                      AS shares_t0,
        ct0.views                       AS views_t0,

        -- Counters at T24
        ct24.likes                      AS likes_t24,
        ct24.comments                   AS comments_t24,
        ct24.shares                     AS shares_t24,
        ct24.views                      AS views_t24,
        ct24.follower_count             AS follower_count_t24,

        -- Counters at T72
        ct72.likes                      AS likes_t72,
        ct72.comments                   AS comments_t72,
        ct72.shares                     AS shares_t72,
        ct72.views                      AS views_t72,
        ct72.follower_count             AS follower_count_t72,

        -- Text features (may be NULL if not yet processed)
        ft.caption_len,
        ft.word_count,
        ft.hashtag_count,
        ft.emoji_count,
        ft.cta_flag,
        ft.lang,
        ft.topic_embedding,                 -- raw BLOB, expanded after query

        -- Visual features (may be NULL if not yet processed)
        fv.brightness,
        fv.contrast,
        fv.colourfulness,
        fv.face_count,
        fv.face_flag,
        fv.ocr_text_len,

        -- Temporal features (may be NULL if not yet processed)
        ftp.local_hour,
        ftp.weekday,
        ftp.is_weekend,
        ftp.post_age_hours,

        -- Audio features (may be NULL if not yet processed)
        fa.audio_present                AS audio_present_feat,
        fa.is_trending                  AS audio_is_trending,
        fa.audio_name                   AS audio_name_feat,
        fa.audio_is_original,

        -- Style features (may be NULL if not yet processed)
        fst.sentence_count,
        fst.avg_sentence_len,
        fst.exclamation_density,
        fst.question_density,
        fst.ellipsis_count,
        fst.caps_ratio,
        fst.caps_word_count,
        fst.line_break_count,
        fst.url_count,
        fst.mention_count,
        fst.punct_diversity,
        fst.flesch_reading_ease

    FROM captures c
    INNER JOIN snapshots s      ON c.snapshot_id = s.snapshot_id
    INNER JOIN posts p          ON c.post_id     = p.post_id

    -- Counters: one left join per revisit window
    LEFT JOIN counters ct0      ON p.post_id = ct0.post_id  AND ct0.revisit_type = 't0'
    LEFT JOIN counters ct24     ON p.post_id = ct24.post_id AND ct24.revisit_type = 't24'
    LEFT JOIN counters ct72     ON p.post_id = ct72.post_id AND ct72.revisit_type = 't72'

    -- Feature tables (left joins: may not be populated yet)
    LEFT JOIN features_text ft  ON p.post_id = ft.post_id
    LEFT JOIN features_visual fv ON p.post_id = fv.post_id
    LEFT JOIN features_temporal ftp ON p.post_id = ftp.post_id
    LEFT JOIN features_audio fa ON p.post_id = fa.post_id
    LEFT JOIN features_style fst ON p.post_id = fst.post_id
    """

    params = []
    if platform:
        query += "\n    WHERE s.platform = ?"
        params.append(platform)

    query += "\n    ORDER BY s.captured_at_utc, c.rank_observed"

    conn = get_connection(db_path)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    logger.info("Extracted %d rows, %d columns.", len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# 1b. Expand topic_embedding BLOB into 384 float columns
# ---------------------------------------------------------------------------
def expand_topic_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand the topic_embedding BLOB column into 384 separate float columns
    named topic_dim_001 through topic_dim_384, then drop the raw BLOB.

    Posts without an embedding (NULL) get NaN across all 384 columns,
    which R and LightGBM handle natively as missing values.

    Args:
        df: DataFrame with a 'topic_embedding' column containing raw bytes
            or None.

    Returns:
        DataFrame with BLOB column replaced by topic_dim_001 ... topic_dim_384.
    """
    if "topic_embedding" not in df.columns:
        return df

    col_names = [f"topic_dim_{i:03d}" for i in range(1, EMBEDDING_DIM + 1)]

    def _blob_to_row(blob):
        if blob is None or (isinstance(blob, (bytes, bytearray)) and len(blob) == 0):
            return [np.nan] * EMBEDDING_DIM
        vec = np.frombuffer(blob, dtype=np.float32)
        if len(vec) != EMBEDDING_DIM:
            return [np.nan] * EMBEDDING_DIM
        # Zero vectors (empty captions) become NaN so R treats them as missing
        if np.all(vec == 0.0):
            return [np.nan] * EMBEDDING_DIM
        return vec.tolist()

    expanded = df["topic_embedding"].apply(_blob_to_row)
    emb_df = pd.DataFrame(expanded.tolist(), columns=col_names, index=df.index)

    df = pd.concat([df.drop(columns=["topic_embedding"]), emb_df], axis=1)

    n_with_embedding = emb_df[col_names[0]].notna().sum()
    logger.info(
        "Topic embeddings expanded: %d posts have vectors, %d are NaN.",
        n_with_embedding, len(df) - n_with_embedding,
    )

    return df


# ---------------------------------------------------------------------------
# 2. Write to Parquet, partitioned by platform and capture_date
# ---------------------------------------------------------------------------
def write_parquet(df: pd.DataFrame, export_dir: str) -> None:
    """
    Write DataFrame to Parquet, partitioned by platform and capture_date.
    Uses PyArrow for efficient columnar storage.
    """

    if df.empty:
        logger.warning("No data to export.")
        return

    table = pa.Table.from_pandas(df)

    pq.write_to_dataset(
        table,
        root_path=export_dir,
        partition_cols=["platform", "capture_date"],
        existing_data_behavior="overwrite_or_ignore",
    )

    # Report output size
    total_size = sum(
        f.stat().st_size for f in Path(export_dir).rglob("*.parquet")
    )
    logger.info("Parquet export complete: %s", export_dir)
    logger.info("Total size: %.1f MB", total_size / 1024 / 1024)


# ---------------------------------------------------------------------------
# 3. Round-trip test: Parquet back to Pandas
# ---------------------------------------------------------------------------
def test_round_trip(export_dir: str) -> None:
    """
    Read exported Parquet back into Pandas and verify integrity.
    Checklist task 20: test round-trip SQLite -> Parquet -> Pandas.
    """

    logger.info("--- Round-trip test ---")

    # Read back with PyArrow
    df_back = pq.read_table(export_dir).to_pandas()
    logger.info("Read back %d rows, %d columns from Parquet.", len(df_back), len(df_back.columns))

    # Basic checks
    assert len(df_back) > 0, "Round-trip failed: no rows read back."
    assert "platform" in df_back.columns, "Round-trip failed: 'platform' column missing."
    assert "post_id" in df_back.columns, "Round-trip failed: 'post_id' column missing."
    assert "rank_observed" in df_back.columns, "Round-trip failed: 'rank_observed' column missing."

    # Show summary
    logger.info("Platforms: %s", df_back['platform'].unique().tolist())
    logger.info("Date range: %s to %s", df_back['capture_date'].min(), df_back['capture_date'].max())
    logger.info("Top posts: %d | Baseline: %d", df_back['is_top'].sum(), (~df_back['is_top'].astype(bool)).sum())
    logger.info("Round-trip test PASSED.")


# ---------------------------------------------------------------------------
# Optional: DuckDB verification (alternative read path)
# ---------------------------------------------------------------------------
def test_duckdb_read(export_dir: str) -> None:
    """
    Verify Parquet files are readable via DuckDB as an alternative to Pandas.
    """
    try:
        import duckdb

        logger.info("--- DuckDB verification ---")
        result = duckdb.sql(
            f"SELECT platform, COUNT(*) AS n FROM read_parquet('{export_dir}/**/*.parquet', hive_partitioning=1) GROUP BY platform"
        ).fetchdf()
        logger.info("\n%s", result.to_string(index=False))
        logger.info("DuckDB verification PASSED.")
    except ImportError:
        logger.info("DuckDB not installed, skipping verification.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Export SQLite to Parquet")
    parser.add_argument("--db", default=DB_PATH, help="Path to SQLite database")
    parser.add_argument("--out", default=EXPORT_DIR, help="Parquet output directory")
    parser.add_argument("--platform", choices=["tiktok", "instagram", "linkedin"],
                        help="Export single platform only")
    parser.add_argument("--test", action="store_true", help="Run round-trip test only")
    args = parser.parse_args()

    if args.test:
        test_round_trip(args.out)
        test_duckdb_read(args.out)
        return

    if args.platform:
        # Single platform: load, expand, write
        df = extract_from_sqlite(args.db, args.platform)
        df = expand_topic_embeddings(df)
        write_parquet(df, args.out)
    else:
        # Process one platform at a time to stay within 4 GB VM RAM.
        # Loading all platforms + 384-dim embeddings at once risks OOM.
        for plat in ("tiktok", "instagram", "linkedin"):
            logger.info("--- Exporting platform: %s ---", plat)
            df = extract_from_sqlite(args.db, plat)
            df = expand_topic_embeddings(df)
            write_parquet(df, args.out)
            del df  # free memory before next platform

    test_round_trip(args.out)
    test_duckdb_read(args.out)


if __name__ == "__main__":
    main()
