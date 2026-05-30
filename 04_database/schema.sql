-- =============================================================================
-- schema.sql
--
-- Master's thesis: Learning the Levers — Which Creator-Controllable
-- Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
-- from a Swiss German-Language Perspective
-- Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026
--
-- Purpose:
--   SQLite schema for the scraper database. Loaded automatically by
--   04_database/db.py on first connection to an empty database.
--
-- Design notes
--   - Core tables (snapshots, posts, captures, counters, scrape_log) hold
--     raw scraped data written during hourly collection.
--   - Feature tables (features_text, features_style, features_visual,
--     features_temporal, features_audio) are populated by a decoupled
--     batch pipeline every 6-12 h (see 03_features/run_features_batch.py).
--   - author_hash: SHA-256 of the raw author/username at ingestion time
--     (privacy by design, FADP-aligned). Raw IDs are never stored.
--   - All timestamps stored as ISO 8601 UTC strings.
--   - Columns may be added via ALTER TABLE as scrapers reveal
--     platform-specific fields.
-- =============================================================================

PRAGMA journal_mode = WAL;          -- better concurrency for reader + writer
PRAGMA foreign_keys = ON;

-- =============================================================================
-- 1. SNAPSHOTS
--    One row per hourly scrape run per account per platform.
-- =============================================================================

CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id     TEXT PRIMARY KEY,          -- e.g. uuid4 or tiktok_fresh_20260201T080000Z
    platform        TEXT NOT NULL              -- 'tiktok' | 'instagram' | 'linkedin'
                    CHECK (platform IN ('tiktok', 'instagram', 'linkedin')),
    account_type    TEXT NOT NULL              -- 'fresh' | 'light_seeded'
                    CHECK (account_type IN ('fresh', 'light_seeded')),
    surface         TEXT NOT NULL              -- 'explore' | 'top_feed' | 'recent'
                    CHECK (surface IN ('explore', 'top_feed', 'recent')),
    captured_at_utc TEXT NOT NULL,             -- ISO 8601 UTC timestamp of snapshot
    timezone        TEXT NOT NULL DEFAULT 'Europe/Zurich',
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);


-- =============================================================================
-- 2. POSTS
--    One row per unique post encountered across all snapshots.
--    Deduplicated on (platform, permalink).
-- =============================================================================

CREATE TABLE IF NOT EXISTS posts (
    post_id         TEXT PRIMARY KEY,          -- uuid4 or platform-native ID hash
    platform        TEXT NOT NULL
                    CHECK (platform IN ('tiktok', 'instagram', 'linkedin')),
    permalink       TEXT,                      -- canonical URL for revisits
    media_type      TEXT,                      -- 'video' | 'image' | 'carousel' | 'text' | etc.
    author_hash     TEXT,                      -- SHA-256 of raw author identifier
    posted_at_utc   TEXT,                      -- post creation timestamp (from JSON metadata)
    follower_count  INTEGER,                   -- author follower count at first capture

    -- Raw caption / hashtag text stored here for later feature extraction
    caption_raw     TEXT,                      -- full caption text as scraped
    hashtags_raw    TEXT,                      -- comma-separated hashtag list

    -- Thumbnail / cover image URL for visual feature extraction
    thumbnail_url   TEXT,                      -- cover tile or video poster URL

    -- Raw audio metadata (from JSON) for later feature extraction
    audio_present   INTEGER,                   -- 0/1 flag
    audio_id        TEXT,                      -- platform audio/sound ID
    audio_name      TEXT,                      -- sound name if available

    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),

    UNIQUE (platform, permalink)
);


-- =============================================================================
-- 3. CAPTURES
--    One row per post per snapshot. Links a post to the snapshot in which
--    it was observed, recording its rank and Top/Baseline status.
-- =============================================================================

CREATE TABLE IF NOT EXISTS captures (
    capture_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id     TEXT NOT NULL REFERENCES snapshots(snapshot_id),
    post_id         TEXT NOT NULL REFERENCES posts(post_id),
    rank_observed   INTEGER,                   -- 1-20 for Top, 51-100 for baseline, etc.
    is_top          INTEGER NOT NULL           -- 1 = Top-20, 0 = Baseline
                    CHECK (is_top IN (0, 1)),
    join_key        TEXT,                      -- DOM-to-JSON join key (permalink or post ID)
    source          TEXT                       -- extraction path: 'dom', 'json', 'joined', or 'api'
                    CHECK (source IS NULL OR source IN ('dom', 'json', 'joined', 'api')),
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);


-- =============================================================================
-- 4. COUNTERS
--    Public engagement counters collected at T0, T0+24h, T0+72h.
--    One row per revisit per post.
-- =============================================================================

CREATE TABLE IF NOT EXISTS counters (
    counter_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id         TEXT NOT NULL REFERENCES posts(post_id),
    revisit_type    TEXT NOT NULL              -- 't0' | 't24' | 't72'
                    CHECK (revisit_type IN ('t0', 't24', 't72')),
    likes           INTEGER,
    comments        INTEGER,
    shares          INTEGER,
    views           INTEGER,                   -- if available (TikTok, LinkedIn)
    follower_count  INTEGER,                   -- re-captured at revisit for velocity calc
    captured_at_utc TEXT NOT NULL,             -- when this revisit was executed
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),

    UNIQUE (post_id, revisit_type)            -- one row per post per revisit window
);


-- =============================================================================
-- 5. SCRAPE_LOG
--    One row per scrape attempt. Tracks success/failure for monitoring
--    (checklist tasks 27, 43, 47).
-- =============================================================================

CREATE TABLE IF NOT EXISTS scrape_log (
    log_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id     TEXT REFERENCES snapshots(snapshot_id),
    platform        TEXT NOT NULL
                    CHECK (platform IN ('tiktok', 'instagram', 'linkedin')),
    account_type    TEXT NOT NULL
                    CHECK (account_type IN ('fresh', 'light_seeded')),
    status          TEXT NOT NULL              -- 'success' | 'partial' | 'failure'
                    CHECK (status IN ('success', 'partial', 'failure')),
    posts_captured  INTEGER DEFAULT 0,
    error_message   TEXT,
    duration_sec    REAL,                      -- wall-clock seconds for the scrape
    started_at_utc  TEXT NOT NULL,
    finished_at_utc TEXT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);


-- =============================================================================
-- 6. FEATURES_TEXT  (populated by 03_features/extract_text_features.py)
--    Derived from caption_raw and hashtags_raw in posts table.
-- =============================================================================

CREATE TABLE IF NOT EXISTS features_text (
    post_id         TEXT PRIMARY KEY REFERENCES posts(post_id),
    caption_len     INTEGER,                   -- character count
    word_count      INTEGER,                   -- token count
    hashtag_count   INTEGER,
    emoji_count     INTEGER,
    cta_flag        INTEGER,                   -- 0/1: call-to-action detected
    lang            TEXT,                      -- ISO 639-1 code via fastText langid
    topic_embedding BLOB,                      -- sentence-transformer vector (binary)
    processed_at    TEXT                        -- when feature extraction ran
);


-- =============================================================================
-- 6b. FEATURES_STYLE  (batch pipeline)
--     Derived from caption_raw in posts table. Writing-style features:
--     readability, punctuation, capitalization, formatting.
-- =============================================================================

CREATE TABLE IF NOT EXISTS features_style (
    post_id             TEXT PRIMARY KEY REFERENCES posts(post_id),
    sentence_count      INTEGER,                   -- number of sentences (NLTK punkt)
    avg_sentence_len    REAL,                      -- mean words per sentence
    exclamation_density REAL,                      -- exclamation marks per character
    question_density    REAL,                      -- question marks per character
    ellipsis_count      INTEGER,                   -- '...' or '…' count
    caps_ratio          REAL,                      -- fraction of alpha chars that are uppercase
    caps_word_count     INTEGER,                   -- ALL-CAPS words (2+ chars)
    line_break_count    INTEGER,                   -- newline count
    url_count           INTEGER,                   -- URL count
    mention_count       INTEGER,                   -- @mention count
    punct_diversity     REAL,                      -- distinct punct types / total punct chars
    flesch_reading_ease REAL,                      -- Flesch reading ease (German-aware)
    processed_at        TEXT                        -- when feature extraction ran
);


-- =============================================================================
-- 7. FEATURES_VISUAL  (populated by 03_features/extract_visual_features.py)
--    Derived from cover tile / thumbnail at capture time.
-- =============================================================================

CREATE TABLE IF NOT EXISTS features_visual (
    post_id         TEXT PRIMARY KEY REFERENCES posts(post_id),
    brightness      REAL,                      -- mean pixel brightness
    contrast        REAL,                      -- RMS contrast
    colourfulness   REAL,                      -- Hasler-Suesstrunk metric or similar
    face_count      INTEGER,                   -- MediaPipe face detection
    face_flag       INTEGER                    -- 0/1: any face present
                    CHECK (face_flag IN (0, 1)),
    ocr_text_len    INTEGER,                   -- pytesseract: character count of text on image
    processed_at    TEXT
);


-- =============================================================================
-- 8. FEATURES_TEMPORAL  (populated by 03_features/extract_temporal_features.py)
--    Derived from posted_at_utc in posts table.
-- =============================================================================

CREATE TABLE IF NOT EXISTS features_temporal (
    post_id         TEXT PRIMARY KEY REFERENCES posts(post_id),
    local_hour      INTEGER,                   -- 0-23, Swiss local time (CET/CEST)
    weekday         INTEGER,                   -- 0=Mon, 6=Sun (ISO weekday)
    is_weekend      INTEGER                    -- 0/1
                    CHECK (is_weekend IN (0, 1)),
    post_age_hours  REAL,                      -- hours between posted_at and first capture
    processed_at    TEXT
);


-- =============================================================================
-- 9. FEATURES_AUDIO  (populated by 03_features/extract_audio_features.py)
--    Derived from audio metadata in posts table.
-- =============================================================================

CREATE TABLE IF NOT EXISTS features_audio (
    post_id             TEXT PRIMARY KEY REFERENCES posts(post_id),
    audio_present       INTEGER                    -- 0/1 (denormalised from posts for modelling)
                        CHECK (audio_present IN (0, 1)),
    is_trending         INTEGER                    -- 0/1: trending sound flag
                        CHECK (is_trending IN (0, 1)),
    audio_id            TEXT,                      -- denormalised for convenience
    audio_name          TEXT,                      -- sound title for modelling
    audio_is_original   INTEGER                    -- 0/1: original sound vs licensed/shared
                        CHECK (audio_is_original IS NULL OR audio_is_original IN (0, 1)),
    processed_at        TEXT
);


-- =============================================================================
-- 10. REVISIT_LOG
--     One row per revisit batch per platform. Tracks T24/T72 revisit
--     execution for monitoring (checklist task 30).
-- =============================================================================

CREATE TABLE IF NOT EXISTS revisit_log (
    log_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    revisit_type    TEXT NOT NULL
                    CHECK (revisit_type IN ('t24', 't72')),
    platform        TEXT NOT NULL
                    CHECK (platform IN ('tiktok', 'instagram', 'linkedin')),
    total_due       INTEGER DEFAULT 0,
    total_success   INTEGER DEFAULT 0,
    total_missing   INTEGER DEFAULT 0,
    total_failed    INTEGER DEFAULT 0,
    duration_sec    REAL,
    started_at_utc  TEXT NOT NULL,
    finished_at_utc TEXT,
    error_message   TEXT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);


-- =============================================================================
-- INDEXES  (checklist task 18: post_id, snapshot_id, platform, captured_at)
-- =============================================================================

-- Posts
CREATE INDEX IF NOT EXISTS idx_posts_platform        ON posts(platform);
CREATE INDEX IF NOT EXISTS idx_posts_permalink        ON posts(platform, permalink);
CREATE INDEX IF NOT EXISTS idx_posts_posted_at        ON posts(posted_at_utc);

-- Captures
CREATE INDEX IF NOT EXISTS idx_captures_snapshot      ON captures(snapshot_id);
CREATE INDEX IF NOT EXISTS idx_captures_post          ON captures(post_id);
CREATE INDEX IF NOT EXISTS idx_captures_is_top        ON captures(is_top);

-- Snapshots
CREATE INDEX IF NOT EXISTS idx_snapshots_platform     ON snapshots(platform);
CREATE INDEX IF NOT EXISTS idx_snapshots_captured_at  ON snapshots(captured_at_utc);
CREATE INDEX IF NOT EXISTS idx_snapshots_account      ON snapshots(platform, account_type);

-- Counters
CREATE INDEX IF NOT EXISTS idx_counters_post          ON counters(post_id);
CREATE INDEX IF NOT EXISTS idx_counters_revisit       ON counters(post_id, revisit_type);
CREATE INDEX IF NOT EXISTS idx_counters_revisit_type  ON counters(revisit_type, captured_at_utc);

-- Scrape log
CREATE INDEX IF NOT EXISTS idx_scrape_log_platform    ON scrape_log(platform);
CREATE INDEX IF NOT EXISTS idx_scrape_log_status      ON scrape_log(status);
CREATE INDEX IF NOT EXISTS idx_scrape_log_started     ON scrape_log(started_at_utc);

-- Revisit log
CREATE INDEX IF NOT EXISTS idx_revisit_log_type       ON revisit_log(revisit_type);
CREATE INDEX IF NOT EXISTS idx_revisit_log_platform   ON revisit_log(platform);
CREATE INDEX IF NOT EXISTS idx_revisit_log_started    ON revisit_log(started_at_utc);


-- =============================================================================
-- DONE. Run with:  sqlite3 /data/scraper.db < schema.sql
-- =============================================================================
