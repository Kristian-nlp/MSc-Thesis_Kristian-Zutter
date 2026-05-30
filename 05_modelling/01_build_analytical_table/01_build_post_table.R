# =============================================================================
# 01_build_post_table.R
#
# Master's thesis: Learning the Levers — Which Creator-Controllable
# Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
# from a Swiss German-Language Perspective
# Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026
#
# Purpose:
#   Builds a single post-level analytical table from scraper.db. One row per
#   post (16,638 posts) with all features, outcomes, controls, and topic
#   embeddings expanded from BLOB to 384 float columns.
#
# Pipeline position:
#   Step 1 of the modelling pipeline. Depends on the database produced by
#   the Python scrapers; feeds 02a_pca_embeddings.R.
#
# Inputs:
#   04_database/scraper.db                                       SQLite raw database (read-only)
#
# Outputs:
#   05_modelling/01_build_analytical_table/data/df_post.parquet  Post-level analytical table
#
# Usage:
#   Rscript 05_modelling/01_build_analytical_table/01_build_post_table.R
# =============================================================================

rm(list = ls())

# -- Source shared config ------------------------------------------------------
# Adjust path if running from a different working directory
source(file.path("config", "packages.R"))

con <- dbConnect(SQLite(), DB_PATH)
cat("Connected to:", DB_PATH, "\n\n")


# =============================================================================
# SUB-STEP 1: Load posts + all five feature tables
# =============================================================================
cat("== SUB-STEP 1: Loading posts + feature tables ==\n")

df_post <- dbGetQuery(con, "
  SELECT
    p.post_id, p.platform, p.permalink, p.media_type, p.author_hash,
    p.posted_at_utc, p.follower_count,
    ft.caption_len, ft.word_count, ft.hashtag_count, ft.emoji_count,
    ft.cta_flag, ft.lang, ft.topic_embedding,
    fs.sentence_count, fs.avg_sentence_len,
    fs.exclamation_density, fs.question_density, fs.ellipsis_count,
    fs.caps_ratio, fs.caps_word_count, fs.line_break_count,
    fs.url_count, fs.mention_count, fs.punct_diversity,
    fs.flesch_reading_ease,
    fv.brightness, fv.contrast, fv.colourfulness,
    fv.face_count, fv.face_flag, fv.ocr_text_len,
    ftemp.local_hour, ftemp.weekday, ftemp.is_weekend, ftemp.post_age_hours,
    fa.audio_present, fa.is_trending, fa.audio_id, fa.audio_name,
    fa.audio_is_original
  FROM posts p
  LEFT JOIN features_text    ft    ON p.post_id = ft.post_id
  LEFT JOIN features_style   fs    ON p.post_id = fs.post_id
  LEFT JOIN features_visual  fv    ON p.post_id = fv.post_id
  LEFT JOIN features_temporal ftemp ON p.post_id = ftemp.post_id
  LEFT JOIN features_audio   fa    ON p.post_id = fa.post_id
")

cat(sprintf("  Posts loaded: %d rows, %d columns\n", nrow(df_post), ncol(df_post)))
cat(sprintf("  Platform breakdown: %s\n",
    paste(names(table(df_post$platform)), table(df_post$platform),
          sep = "=", collapse = ", ")))


# =============================================================================
# SUB-STEP 2: Compute outcomes from captures
# =============================================================================
cat("\n== SUB-STEP 2: Computing outcomes (ever_top, best_rank, n_captures) ==\n")

outcomes <- dbGetQuery(con, "
  SELECT
    post_id,
    MAX(is_top) AS ever_top,
    MIN(CASE WHEN is_top = 1 THEN rank_observed ELSE NULL END) AS best_rank,
    COUNT(*) AS n_captures
  FROM captures
  GROUP BY post_id
")

df_post <- df_post |> left_join(outcomes, by = "post_id")

cat(sprintf("  ever_top distribution:\n"))
print(table(df_post$platform, df_post$ever_top, dnn = c("platform", "ever_top")))
cat(sprintf("\n  best_rank summary (top posts only):\n"))
df_post |> filter(ever_top == 1) |>
  group_by(platform) |>
  summarise(n = n(), mean_rank = round(mean(best_rank, na.rm = TRUE), 1),
            median_rank = median(best_rank, na.rm = TRUE),
            .groups = "drop") |>
  print()


# =============================================================================
# SUB-STEP 3: Assign account_type (fresh / seeded / both)
# =============================================================================
cat("\n== SUB-STEP 3: Assigning account_type ==\n")

account_types <- dbGetQuery(con, "
  SELECT
    c.post_id,
    GROUP_CONCAT(DISTINCT s.account_type) AS observed_by
  FROM captures c
  JOIN snapshots s ON c.snapshot_id = s.snapshot_id
  GROUP BY c.post_id
")

# Derive three-level factor
account_types <- account_types |>
  mutate(
    account_type = case_when(
      grepl(",", observed_by)        ~ "both",
      observed_by == "fresh"         ~ "fresh",
      observed_by == "light_seeded"  ~ "seeded",
      TRUE                           ~ observed_by
    )
  ) |>
  select(post_id, account_type)

df_post <- df_post |> left_join(account_types, by = "post_id")

# --- DECISION DIAGNOSTIC: Account type overlap ---
cat("\n  >>> DECISION: Account type overlap <<<\n")
acct_diag <- df_post |>
  group_by(platform, account_type) |>
  summarise(n = n(), .groups = "drop") |>
  group_by(platform) |>
  mutate(pct = round(n / sum(n) * 100, 1)) |>
  ungroup()
print(as.data.frame(acct_diag))

cat("\n  'both' category per platform:\n")
acct_diag |> filter(account_type == "both") |> print()
cat("  Rule: if 'both' < 5% for a platform, collapse to two-level factor.\n")


# =============================================================================
# SUB-STEP 4: Extract counters (T0 / T24 / T72)
# =============================================================================
cat("\n== SUB-STEP 4: Extracting counters ==\n")

# T0 counters (primary: follower_count + engagement baseline)
counters_t0 <- dbGetQuery(con, "
  SELECT post_id,
         likes          AS likes_t0,
         comments       AS comments_t0,
         shares         AS shares_t0,
         views          AS views_t0,
         follower_count AS follower_count_t0
  FROM counters
  WHERE revisit_type = 't0'
")

# T24 counters
counters_t24 <- dbGetQuery(con, "
  SELECT post_id,
         likes    AS likes_t24,
         comments AS comments_t24,
         shares   AS shares_t24,
         views    AS views_t24
  FROM counters
  WHERE revisit_type = 't24'
")

# T72 counters
counters_t72 <- dbGetQuery(con, "
  SELECT post_id,
         likes    AS likes_t72,
         comments AS comments_t72,
         shares   AS shares_t72,
         views    AS views_t72
  FROM counters
  WHERE revisit_type = 't72'
")

df_post <- df_post |>
  left_join(counters_t0,  by = "post_id") |>
  left_join(counters_t24, by = "post_id") |>
  left_join(counters_t72, by = "post_id")

# --- DECISION DIAGNOSTIC: follower_count T0 coverage ---
cat("\n  >>> DECISION: follower_count T0 coverage <<<\n")
fc_diag <- df_post |>
  group_by(platform) |>
  summarise(
    n_posts         = n(),
    has_t0_row      = sum(!is.na(likes_t0)),
    fc_t0_non_null  = sum(!is.na(follower_count_t0)),
    fc_t0_null_pct  = round(sum(is.na(follower_count_t0)) / n() * 100, 1),
    fc_posts_table  = sum(!is.na(follower_count)),
    .groups = "drop"
  )
print(as.data.frame(fc_diag))
cat("  Note: follower_count_t0 is the primary source. posts.follower_count is fallback.\n")

# Use T0 counter follower_count as primary; fall back to posts.follower_count
df_post <- df_post |>
  mutate(
    follower_count_final = coalesce(follower_count_t0, follower_count)
  )

cat(sprintf("  follower_count_final null rate: %.1f%% (%d / %d)\n",
    sum(is.na(df_post$follower_count_final)) / nrow(df_post) * 100,
    sum(is.na(df_post$follower_count_final)), nrow(df_post)))


# =============================================================================
# SUB-STEP 5: Compute velocity
# =============================================================================
cat("\n== SUB-STEP 5: Computing velocity ==\n")

# Velocity is defined as the 24- / 72-hour log-growth in COMBINED engagement
# (likes + comments + shares). This aligns with the original proposal
# definition in master_context.txt (Visibility Proxies block) and
# supersedes the prior likes-only operationalisation.
#
# Decision: the thesis methodology decision log, Decision 38 (2026-04-17).
#
# Background: the likes-only formula was a compromise driven by Instagram
# data gaps (comments 87.8% null at T24, shares always null). With
# Instagram velocity now dropped from RQ1 reporting entirely (Decision
# 37 + amendment, 2026-04-17), that compromise is no longer binding.
# TikTok and LinkedIn have reliable likes, comments, and shares at both
# T24 and T72 per appendix_ig_velocity_diagnostic.csv, so combined
# engagement velocity can be computed directly.
#
# Column names velocity_24h and velocity_72h are deliberately retained;
# downstream scripts (03c, 04_eda, 05c, 08, 09, 10/table_01) treat the
# columns as opaque outcomes. See Decision 38 for audit-trail notes.
#
# Any coalesce to zero here is safe: if likes_tX, comments_tX, or
# shares_tX is NA for a particular revisit row the post had no data at
# that time point, and the resulting velocity NA will be dropped by the
# downstream filter(!is.na(velocity_...)) in 05c_gam_velocity.R. We do
# NOT coalesce NAs to zero here, which would silently turn "no data"
# into "zero engagement". Keeping NAs propagating is the correct
# behaviour.

df_post <- df_post |>
  mutate(
    velocity_24h = log(likes_t24 + comments_t24 + shares_t24 + 1) -
                   log(likes_t0  + comments_t0  + shares_t0  + 1),
    velocity_72h = log(likes_t72 + comments_t72 + shares_t72 + 1) -
                   log(likes_t0  + comments_t0  + shares_t0  + 1)
  )

vel_diag <- df_post |>
  group_by(platform) |>
  summarise(
    n_vel24     = sum(!is.na(velocity_24h)),
    pct_vel24   = round(sum(!is.na(velocity_24h)) / n() * 100, 1),
    mean_vel24  = round(mean(velocity_24h, na.rm = TRUE), 3),
    sd_vel24    = round(sd(velocity_24h, na.rm = TRUE), 3),
    n_vel72     = sum(!is.na(velocity_72h)),
    pct_vel72   = round(sum(!is.na(velocity_72h)) / n() * 100, 1),
    .groups = "drop"
  )
cat("  Velocity coverage:\n")
print(as.data.frame(vel_diag))


# =============================================================================
# SUB-STEP 6: Collapse audio_name to uses_named_audio
# =============================================================================
cat("\n== SUB-STEP 6: Creating uses_named_audio ==\n")

# Patterns matching "original sound" in multiple languages
# (from extract_audio_features.py _ORIGINAL_SOUND_PATTERNS).
# enc2utf8() guards against mixed source/locale encoding in Rscript;
# perl = TRUE makes the regex engine Unicode-aware (2026-04-17 fix).
original_sound_pattern <- enc2utf8(paste(
  "original sound", "Originalton", "Original Sound",
  "originalljud", "son original", "sonido original",
  "som original", "suara asli", "orijinal ses",
  "оригинальный звук", "الصوت الأصلي", "πρωτότυπος ήχος",
  sep = "|"
))

df_post <- df_post |>
  mutate(
    uses_named_audio = case_when(
      is.na(audio_name)                                       ~ NA_integer_,
      grepl(original_sound_pattern, enc2utf8(audio_name),
            ignore.case = TRUE, perl = TRUE)                  ~ 0L,
      TRUE                                                    ~ 1L
    )
  )

cat("  uses_named_audio distribution (TikTok only):\n")
df_post |> filter(platform == "tiktok") |>
  count(uses_named_audio) |> print()


# =============================================================================
# SUB-STEP 7: Expand topic embeddings (BLOB -> 384 float columns)
# =============================================================================
cat("\n== SUB-STEP 7: Expanding topic embeddings ==\n")

EMBEDDING_DIM <- 384L

expand_blob <- function(blob) {
  if (is.null(blob) || length(blob) == 0 || all(is.na(blob))) {
    return(rep(NA_real_, EMBEDDING_DIM))
  }
  # Handle list-wrapped blobs from RSQLite
  if (is.list(blob)) blob <- blob[[1]]
  vec <- readBin(blob, what = "double", n = EMBEDDING_DIM, size = 4, endian = "little")
  if (length(vec) != EMBEDDING_DIM) {
    return(rep(NA_real_, EMBEDDING_DIM))
  }
  # Check for zero vector (empty captions)
  if (all(vec == 0)) {
    return(rep(NA_real_, EMBEDDING_DIM))
  }
  return(vec)
}

cat("  Expanding BLOBs (this may take a minute)...\n")
emb_matrix <- do.call(rbind, lapply(df_post$topic_embedding, expand_blob))
colnames(emb_matrix) <- sprintf("topic_dim_%03d", 1:EMBEDDING_DIM)

# Check coverage
n_valid <- sum(!is.na(emb_matrix[, 1]))
cat(sprintf("  Valid embeddings: %d / %d (%.1f%%)\n",
    n_valid, nrow(df_post), n_valid / nrow(df_post) * 100))

# Bind to main dataframe and drop raw BLOB column
df_post <- df_post |>
  select(-topic_embedding) |>
  bind_cols(as_tibble(emb_matrix))


# =============================================================================
# SUB-STEP 8: Instagram temporal diagnostics (NO FILTER)
# =============================================================================
cat("\n== SUB-STEP 8: Instagram temporal diagnostics ==\n")

ig_before <- sum(df_post$platform == "instagram")
ig_temporal_null <- sum(df_post$platform == "instagram" & is.na(df_post$local_hour))

cat(sprintf("  Instagram total: %d | with temporal nulls: %d (%.1f%%)\n",
    ig_before, ig_temporal_null, ig_temporal_null / ig_before * 100))

# Show ever_top breakdown for null vs present temporal features
cat("\n  Instagram ever_top by temporal completeness:\n")
df_post |> filter(platform == "instagram") |>
  mutate(temporal = ifelse(is.na(local_hour), "null", "present")) |>
  count(temporal, ever_top) |> print()

# NOTE: Instagram temporal filter REMOVED (2026-04-04).
# Root cause: posted_at_utc is NULL for 1,956 Instagram posts, all captured
# via DOM-only path. These are overwhelmingly top-20 posts (1,919 of 1,956).
# Filtering would remove 85% of top posts — a scraper instrumentation artifact,
# not a content property. Instead, temporal features are excluded from the
# Instagram model entirely. See the thesis methodology decision log for full
# justification.
cat("  NOTE: No filter applied. Instagram model will exclude temporal features.\n")
cat(sprintf("  All %d Instagram posts retained.\n", ig_before))

# --- DECISION DIAGNOSTIC: Instagram velocity bias ---
cat("\n  >>> DECISION: Instagram velocity bias check <<<\n")
ig_data <- df_post |> filter(platform == "instagram")

ig_bias <- ig_data |>
  mutate(has_t24 = !is.na(velocity_24h)) |>
  group_by(has_t24) |>
  summarise(
    n              = n(),
    ever_top_rate  = round(mean(ever_top, na.rm = TRUE), 3),
    mean_caption   = round(mean(caption_len, na.rm = TRUE), 1),
    mean_hashtags  = round(mean(hashtag_count, na.rm = TRUE), 1),
    mean_follower  = round(mean(follower_count_final, na.rm = TRUE), 0),
    median_follower = round(median(follower_count_final, na.rm = TRUE), 0),
    .groups = "drop"
  )
cat("  Instagram T24 subset vs rest:\n")
print(as.data.frame(ig_bias))


# =============================================================================
# SUB-STEP 9: Clean up and write parquet
# =============================================================================
cat("\n== SUB-STEP 9: Final cleanup and write ==\n")

# Drop columns not needed for modelling
df_post <- df_post |>
  select(
    # Identifiers
    post_id, platform, permalink, media_type, author_hash,
    # Outcomes
    ever_top, best_rank, n_captures,
    # Velocity
    velocity_24h, velocity_72h,
    # Controls
    account_type,
    # Text features
    caption_len, word_count, hashtag_count, emoji_count, cta_flag, lang,
    # Style features
    sentence_count, avg_sentence_len, exclamation_density, question_density,
    ellipsis_count, caps_ratio, caps_word_count, line_break_count,
    url_count, mention_count, punct_diversity, flesch_reading_ease,
    # Visual features
    brightness, contrast, colourfulness, face_count, face_flag, ocr_text_len,
    # Temporal features
    local_hour, weekday, is_weekend, post_age_hours,
    # Audio features
    audio_present, is_trending, audio_is_original, uses_named_audio,
    # Metadata
    follower_count_final, posted_at_utc,
    # Counter baselines (for reference)
    likes_t0, comments_t0, shares_t0, views_t0,
    likes_t24, comments_t24, shares_t24, views_t24,
    likes_t72, comments_t72, shares_t72, views_t72,
    # Embeddings
    starts_with("topic_dim_")
  )

# Final summary
cat("\n  === FINAL SUMMARY ===\n")
cat(sprintf("  Dimensions: %d rows x %d columns\n", nrow(df_post), ncol(df_post)))
cat("\n  Platform counts:\n")
print(table(df_post$platform))
cat("\n  ever_top by platform:\n")
print(table(df_post$platform, df_post$ever_top, dnn = c("platform", "ever_top")))
cat("\n  account_type by platform:\n")
print(table(df_post$platform, df_post$account_type, dnn = c("platform", "account_type")))

# Write parquet
out_path <- file.path(DATA_DIR, "df_post.parquet")
write_parquet(df_post, out_path)
cat(sprintf("\n  Written to: %s\n", out_path))
cat(sprintf("  File size: %.1f MB\n", file.size(out_path) / 1e6))

# Disconnect
dbDisconnect(con)

cat("\n== Step 1 complete ==\n")
