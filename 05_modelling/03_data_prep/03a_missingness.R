# =============================================================================
# 03a_missingness.R
#
# Master's thesis: Learning the Levers — Which Creator-Controllable
# Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
# from a Swiss German-Language Perspective
# Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026
#
# Purpose:
#   Handles missing data via imputation, indicator variables, capping, and
#   structural transformations. Each strategy is matched to the missingness
#   mechanism (structural zero, scraper artefact, design constraint, etc.).
#
# Pipeline position:
#   Step 3a of the modelling pipeline. Depends on 02b_topic_clustering.R;
#   feeds 03b_collinearity.R.
#
# Inputs:
#   05_modelling/01_build_analytical_table/data/df_post.parquet  Post-level analytical table
#
# Outputs:
#   05_modelling/01_build_analytical_table/data/df_post.parquet  Updated in place with imputed values
#   05_modelling/03_data_prep/output/feature_decisions.csv       Audit log of every decision
#
# Usage:
#   Rscript 05_modelling/03_data_prep/03a_missingness.R
# =============================================================================

rm(list = ls())

# -- Source shared config ------------------------------------------------------
source(file.path("config", "packages.R"))

STEP_DIR <- file.path(BASE_DIR, "03_data_prep")
OUT_DIR  <- file.path(STEP_DIR, "output")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)


# -- Decision log helper -------------------------------------------------------
decisions <- tibble(
  feature   = character(),
  platform  = character(),
  action    = character(),
  rationale = character()
)

log_decision <- function(feature, platform, action, rationale) {
  decisions <<- bind_rows(decisions, tibble(
    feature   = feature,
    platform  = platform,
    action    = action,
    rationale = rationale
  ))
}


# =============================================================================
# 1. Load and verify
# =============================================================================
cat("== 1. Loading df_post.parquet ==\n")

df <- read_parquet(file.path(DATA_DIR, "df_post.parquet"))
cat(sprintf("  Loaded: %d rows x %d columns\n", nrow(df), ncol(df)))

stopifnot("Expected 16,638 rows" = nrow(df) == 16638)
stopifnot("topic_cluster column missing" = "topic_cluster" %in% names(df))
cat("  Assertions passed (16,638 rows, topic_cluster present).\n")


# =============================================================================
# 2. Pre-imputation null scan
# =============================================================================
cat("\n== 2. Pre-imputation null scan ==\n")

# Features to check (exclude identifiers, embeddings, counters)
check_features <- c(
  # Text
  "caption_len", "word_count", "hashtag_count", "emoji_count", "cta_flag", "lang",
  # Style
  "sentence_count", "avg_sentence_len", "exclamation_density", "question_density",
  "ellipsis_count", "caps_ratio", "caps_word_count", "line_break_count",
  "url_count", "mention_count", "punct_diversity", "flesch_reading_ease",
  # Visual
  "brightness", "contrast", "colourfulness", "face_count", "face_flag", "ocr_text_len",
  # Temporal
  "local_hour", "weekday", "is_weekend", "post_age_hours",
  # Audio
  "audio_present", "is_trending", "audio_is_original", "uses_named_audio",
  # Metadata
  "follower_count_final",
  # Topic
  "topic_cluster"
)

null_before <- df |>
  group_by(platform) |>
  summarise(
    across(all_of(check_features), ~ sum(is.na(.x))),
    .groups = "drop"
  ) |>
  pivot_longer(-platform, names_to = "feature", values_to = "n_null") |>
  pivot_wider(names_from = platform, values_from = n_null) |>
  filter(if_any(c(instagram, linkedin, tiktok), ~ .x > 0))

cat("  Features with any nulls (pre-imputation):\n")
print(as.data.frame(null_before), row.names = FALSE)


# =============================================================================
# 3. flesch_reading_ease: cap + indicator + impute
# =============================================================================
cat("\n== 3. flesch_reading_ease: cap [-50, 121] + indicator + impute ==\n")

# Cap before computing median (extreme values are formula artefacts)
n_below <- sum(df$flesch_reading_ease < -50, na.rm = TRUE)
n_above <- sum(df$flesch_reading_ease > 121, na.rm = TRUE)
cat(sprintf("  Values below -50: %d | above 121: %d\n", n_below, n_above))

df <- df |>
  mutate(
    flesch_reading_ease = pmin(pmax(flesch_reading_ease, -50), 121)
  )

# Create indicator BEFORE imputation
df <- df |>
  mutate(flesch_available = as.integer(!is.na(flesch_reading_ease)))

cat("  flesch_available distribution:\n")
print(table(df$platform, df$flesch_available, dnn = c("platform", "flesch_available")))

# Impute with platform-specific median
platform_medians_flesch <- df |>
  filter(!is.na(flesch_reading_ease)) |>
  group_by(platform) |>
  summarise(med = median(flesch_reading_ease), .groups = "drop")

cat("  Platform medians for flesch_reading_ease:\n")
print(as.data.frame(platform_medians_flesch))

df <- df |>
  left_join(platform_medians_flesch, by = "platform", suffix = c("", "_med")) |>
  mutate(
    flesch_reading_ease = coalesce(flesch_reading_ease, med)
  ) |>
  select(-med)

cat(sprintf("  Remaining NAs: %d\n", sum(is.na(df$flesch_reading_ease))))

log_decision("flesch_reading_ease", "all", "cap [-50, 121] + impute platform median + flesch_available indicator",
             "Structural missingness: short captions (<30 chars) cannot produce meaningful readability score. Values outside [-50, 121] are formula artefacts.")


# =============================================================================
# 4. avg_sentence_len: impute structural zeros
# =============================================================================
cat("\n== 4. avg_sentence_len: impute with 0 (structural zero) ==\n")

n_na <- sum(is.na(df$avg_sentence_len))
n_zero_sent <- sum(is.na(df$avg_sentence_len) & df$sentence_count == 0, na.rm = TRUE)
cat(sprintf("  NAs: %d | of which sentence_count == 0: %d\n", n_na, n_zero_sent))

if (n_na > 0 && n_zero_sent < n_na) {
  cat("  WARNING: Some avg_sentence_len NAs have sentence_count > 0. Investigate.\n")
}

df <- df |>
  mutate(avg_sentence_len = replace_na(avg_sentence_len, 0))

cat(sprintf("  Remaining NAs: %d\n", sum(is.na(df$avg_sentence_len))))

log_decision("avg_sentence_len", "all", "impute with 0",
             "Structural zero: posts with 0 sentences cannot have avg sentence length.")


# =============================================================================
# 5. follower_count_final: indicator + impute
# =============================================================================
cat("\n== 5. follower_count_final: indicator + impute ==\n")

df <- df |>
  mutate(follower_available = as.integer(!is.na(follower_count_final)))

cat("  follower_available distribution:\n")
print(table(df$platform, df$follower_available, dnn = c("platform", "follower_available")))

platform_medians_fc <- df |>
  filter(!is.na(follower_count_final)) |>
  group_by(platform) |>
  summarise(med = median(follower_count_final), .groups = "drop")

cat("  Platform medians for follower_count_final:\n")
print(as.data.frame(platform_medians_fc))

df <- df |>
  left_join(platform_medians_fc, by = "platform", suffix = c("", "_med")) |>
  mutate(
    follower_count_final = coalesce(follower_count_final, med)
  ) |>
  select(-med)

cat(sprintf("  Remaining NAs: %d\n", sum(is.na(df$follower_count_final))))

log_decision("follower_count_final", "all", "impute platform median + follower_available indicator",
             "Missingness may predict visibility (e.g. private profiles). Indicator preserves that signal.")


# =============================================================================
# 6. post_age_hours: impute TikTok NAs + log-transform
# =============================================================================
cat("\n== 6. post_age_hours: impute TikTok NAs + create log_post_age ==\n")

# Check NAs by platform
cat("  post_age_hours NAs by platform:\n")
print(table(df$platform, is.na(df$post_age_hours), dnn = c("platform", "is_na")))

# Impute TikTok NAs with TikTok median
tt_median_age <- median(df$post_age_hours[df$platform == "tiktok"], na.rm = TRUE)
cat(sprintf("  TikTok median post_age_hours: %.1f\n", tt_median_age))

df <- df |>
  mutate(
    post_age_hours = if_else(
      platform == "tiktok" & is.na(post_age_hours),
      tt_median_age,
      post_age_hours
    )
  )

# Instagram NAs left as-is (temporal features dropped from IG model in 03c)
cat(sprintf("  Instagram post_age_hours NAs left: %d (dropped in 03c)\n",
    sum(df$platform == "instagram" & is.na(df$post_age_hours))))

# Log-transform (non-NA only)
df <- df |>
  mutate(log_post_age = log(post_age_hours + 1))

cat(sprintf("  log_post_age range (non-NA): [%.2f, %.2f]\n",
    min(df$log_post_age, na.rm = TRUE),
    max(df$log_post_age, na.rm = TRUE)))

log_decision("post_age_hours", "tiktok", "impute 5 NAs with platform median",
             "Negligible missingness (0.1%). Platform median is safe.")
log_decision("post_age_hours", "instagram", "leave NAs (temporal features dropped from IG model)",
             "28.3% null due to DOM-only capture artifact. Temporal features excluded from IG model entirely.")
log_decision("log_post_age", "all", "log(post_age_hours + 1)",
             "Compresses extreme range (0-101,155h to 0-11.5). Stabilises variance for GAM.")


# =============================================================================
# 7. TikTok temporal: impute 5 posts
# =============================================================================
cat("\n== 7. TikTok temporal: impute 5 NAs ==\n")

tt_local_hour_med <- round(median(df$local_hour[df$platform == "tiktok"], na.rm = TRUE))
tt_weekday_med    <- round(median(df$weekday[df$platform == "tiktok"], na.rm = TRUE))
tt_is_weekend_med <- as.integer(tt_weekday_med >= 5)  # 5=Sat, 6=Sun

cat(sprintf("  TikTok medians: local_hour=%d, weekday=%d, is_weekend=%d\n",
    tt_local_hour_med, tt_weekday_med, tt_is_weekend_med))

df <- df |>
  mutate(
    local_hour = if_else(platform == "tiktok" & is.na(local_hour),
                         as.double(tt_local_hour_med), local_hour),
    weekday    = if_else(platform == "tiktok" & is.na(weekday),
                         as.double(tt_weekday_med), weekday),
    is_weekend = if_else(platform == "tiktok" & is.na(is_weekend),
                         as.double(tt_is_weekend_med), is_weekend)
  )

cat(sprintf("  TikTok temporal NAs remaining: local_hour=%d, weekday=%d, is_weekend=%d\n",
    sum(df$platform == "tiktok" & is.na(df$local_hour)),
    sum(df$platform == "tiktok" & is.na(df$weekday)),
    sum(df$platform == "tiktok" & is.na(df$is_weekend))))

log_decision("local_hour, weekday, is_weekend", "tiktok", "impute 5 NAs with platform median",
             "0.1% null (5 posts). Platform median rounded to integer.")


# =============================================================================
# 8. lang: replace NA with "unknown"
# =============================================================================
cat("\n== 8. lang: replace NAs with 'unknown' ==\n")

cat("  lang NAs by platform:\n")
print(table(df$platform, is.na(df$lang), dnn = c("platform", "is_na")))

df <- df |>
  mutate(lang = replace_na(lang, "unknown"))

cat(sprintf("  Remaining NAs: %d\n", sum(is.na(df$lang))))

log_decision("lang", "all", "replace NA with 'unknown'",
             "1-3% null. Too few to impute meaningfully; separate category preserves signal.")


# =============================================================================
# 9. Instagram visual features: impute 263 NAs
# =============================================================================
cat("\n== 9. Instagram visual: impute 263 NAs with platform median ==\n")

visual_continuous <- c("brightness", "contrast", "colourfulness", "face_count", "ocr_text_len")

# Compute Instagram medians for visual features
ig_visual_medians <- df |>
  filter(platform == "instagram", !is.na(brightness)) |>
  summarise(across(all_of(visual_continuous), median))

cat("  Instagram medians for visual features:\n")
print(as.data.frame(ig_visual_medians))

# Impute continuous visual features for Instagram
for (feat in visual_continuous) {
  med_val <- ig_visual_medians[[feat]]
  df <- df |>
    mutate(!!feat := if_else(
      platform == "instagram" & is.na(.data[[feat]]),
      med_val,
      .data[[feat]]
    ))
}

# face_flag: impute with mode (most common value among Instagram non-NA)
ig_face_flag_mode <- as.integer(names(sort(table(
  df$face_flag[df$platform == "instagram" & !is.na(df$face_flag)]
), decreasing = TRUE))[1])

cat(sprintf("  Instagram face_flag mode: %d\n", ig_face_flag_mode))

df <- df |>
  mutate(face_flag = if_else(
    platform == "instagram" & is.na(face_flag),
    as.double(ig_face_flag_mode),
    face_flag
  ))

cat(sprintf("  Instagram visual NAs remaining: %d\n",
    sum(df$platform == "instagram" & is.na(df$brightness))))

log_decision("brightness, contrast, colourfulness, face_count, ocr_text_len", "instagram",
             "impute 263 NAs with platform median",
             "3.8% null (263 posts). Below 5%, median imputation is defensible and preserves sample size.")
log_decision("face_flag", "instagram", "impute 263 NAs with platform mode",
             "Binary feature: mode imputation for 3.8% missing is conservative.")


# =============================================================================
# 10. topic_cluster: assign NAs to "unknown" level
# =============================================================================
cat("\n== 10. topic_cluster: assign 97 NAs to 'unknown' ==\n")

n_tc_na <- sum(is.na(df$topic_cluster))
cat(sprintf("  topic_cluster NAs: %d\n", n_tc_na))

# Add "unknown" as a new level and assign to NAs
df <- df |>
  mutate(
    topic_cluster = fct_expand(topic_cluster, "unknown"),
    topic_cluster = replace_na(topic_cluster, "unknown")
  )

cat(sprintf("  topic_cluster NAs remaining: %d\n", sum(is.na(df$topic_cluster))))
cat("  topic_cluster levels:\n")
print(table(df$topic_cluster))

log_decision("topic_cluster", "all", "assign 97 NAs to 'unknown' level",
             "97 posts with missing embeddings (88 IG, 9 LI). Separate category preserves signal.")


# =============================================================================
# 11. Post-imputation verification
# =============================================================================
cat("\n== 11. Post-imputation null scan ==\n")

null_after <- df |>
  group_by(platform) |>
  summarise(
    across(all_of(c(check_features, "flesch_available", "follower_available", "log_post_age")),
           ~ sum(is.na(.x))),
    .groups = "drop"
  ) |>
  pivot_longer(-platform, names_to = "feature", values_to = "n_null") |>
  pivot_wider(names_from = platform, values_from = n_null) |>
  filter(if_any(c(instagram, linkedin, tiktok), ~ .x > 0))

if (nrow(null_after) > 0) {
  cat("  Features with remaining nulls (expected: Instagram temporal + LinkedIn visual):\n")
  print(as.data.frame(null_after), row.names = FALSE)
} else {
  cat("  No remaining nulls.\n")
}

# Verify: only expected nulls remain
# Instagram temporal (local_hour, weekday, is_weekend, post_age_hours, log_post_age)
ig_temporal_nulls <- sum(df$platform == "instagram" & is.na(df$local_hour))
# LinkedIn visual (brightness, contrast, colourfulness, face_count, face_flag, ocr_text_len)
li_visual_nulls <- sum(df$platform == "linkedin" & is.na(df$brightness))

cat(sprintf("\n  Expected remaining nulls:\n"))
cat(sprintf("    Instagram temporal: %d posts (will be dropped in 03c)\n", ig_temporal_nulls))
cat(sprintf("    LinkedIn visual: %d posts (will be dropped in 03c)\n", li_visual_nulls))

# Check nothing unexpected
# Expected remaining nulls:
#   - Instagram temporal (1,956): dropped in 03c
#   - LinkedIn visual (1,170): dropped in 03c
#   - audio_is_original / uses_named_audio: TikTok-only (IG/LI 100% null by design,
#     TikTok 5 null = same 5 posts with missing audio metadata, negligible)
unexpected <- null_after |>
  filter(!feature %in% c("local_hour", "weekday", "is_weekend", "post_age_hours", "log_post_age",
                          "brightness", "contrast", "colourfulness", "face_count", "face_flag",
                          "ocr_text_len", "audio_is_original", "uses_named_audio"))

if (nrow(unexpected) > 0) {
  cat("\n  WARNING: Unexpected remaining nulls:\n")
  print(as.data.frame(unexpected), row.names = FALSE)
} else {
  cat("  No unexpected nulls. All imputation successful.\n")
}


# =============================================================================
# 12. Write outputs
# =============================================================================
cat("\n== 12. Writing outputs ==\n")

# Write feature decisions
write_csv(decisions, file.path(OUT_DIR, "feature_decisions.csv"))
cat(sprintf("  feature_decisions.csv: %d rows written to %s\n",
    nrow(decisions), file.path(OUT_DIR, "feature_decisions.csv")))

# Write updated parquet
out_path <- file.path(DATA_DIR, "df_post.parquet")
write_parquet(df, out_path)
cat(sprintf("  df_post.parquet: %d rows x %d columns\n", nrow(df), ncol(df)))
cat(sprintf("  File size: %.1f MB\n", file.size(out_path) / 1e6))
cat(sprintf("  New columns: flesch_available, follower_available, log_post_age\n"))


cat("\n== Step 3a complete ==\n")
