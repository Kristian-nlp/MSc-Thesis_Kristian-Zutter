# =============================================================================
# 03c_transformations.R
#
# Master's thesis: Learning the Levers — Which Creator-Controllable
# Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
# from a Swiss German-Language Perspective
# Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026
#
# Purpose:
#   Applies log-transforms, collapses rare factor levels, drops collinear and
#   platform-inappropriate features, z-standardises continuous predictors
#   within platform, and splits the post table into three platform-specific
#   analytical frames.
#
# Pipeline position:
#   Step 3c of the modelling pipeline. Depends on 03a_missingness.R and
#   03b_collinearity.R; feeds 03d_patch_audio.R and Step 4 onwards.
#
# Inputs:
#   05_modelling/01_build_analytical_table/data/df_post.parquet  Post-level analytical table
#   05_modelling/03_data_prep/output/collinear_drops.csv         Collinearity drop decisions
#   05_modelling/03_data_prep/output/feature_decisions.csv       Decision log so far
#
# Outputs:
#   05_modelling/03_data_prep/data/df_tt.parquet                 TikTok analytical frame
#   05_modelling/03_data_prep/data/df_ig.parquet                 Instagram analytical frame
#   05_modelling/03_data_prep/data/df_li.parquet                 LinkedIn analytical frame
#   05_modelling/03_data_prep/output/z_score_lookup.csv          Per-feature z-score parameters
#   05_modelling/03_data_prep/output/feature_decisions.csv       Updated decision log
#
# Usage:
#   Rscript 05_modelling/03_data_prep/03c_transformations.R
# =============================================================================

rm(list = ls())

# -- Source shared config ------------------------------------------------------
source(file.path("config", "packages.R"))

STEP_DIR  <- file.path(BASE_DIR, "03_data_prep")
OUT_DIR   <- file.path(STEP_DIR, "output")
DATA_OUT  <- file.path(STEP_DIR, "data")
dir.create(DATA_OUT, recursive = TRUE, showWarnings = FALSE)
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)


# =============================================================================
# 1. Load data and collinearity decisions
# =============================================================================
cat("== 1. Loading df_post.parquet and collinear_drops.csv ==\n")

df <- read_parquet(file.path(DATA_DIR, "df_post.parquet"))
cat(sprintf("  Loaded: %d rows x %d columns\n", nrow(df), ncol(df)))

# Verify 03a outputs
stopifnot("flesch_available missing (run 03a first)" = "flesch_available" %in% names(df))
stopifnot("log_post_age missing (run 03a first)" = "log_post_age" %in% names(df))

# Load collinearity drop decisions
drops_file <- file.path(OUT_DIR, "collinear_drops.csv")
stopifnot("collinear_drops.csv missing (run 03b first)" = file.exists(drops_file))

collinear_drops <- read_csv(drops_file, show_col_types = FALSE)
cat(sprintf("  Collinear drops: %s\n", paste(collinear_drops$drop, collapse = ", ")))

# Load existing decisions
decisions <- read_csv(file.path(OUT_DIR, "feature_decisions.csv"),
                      show_col_types = FALSE)
cat(sprintf("  Loaded %d existing decisions.\n", nrow(decisions)))


# =============================================================================
# 2. Log-transforms
# =============================================================================
cat("\n== 2. Log-transforms ==\n")

# log_post_age already created in 03a -- verify
stopifnot("log_post_age should exist from 03a" = "log_post_age" %in% names(df))
cat("  log_post_age: already exists from 03a.\n")

# Create log_follower
df <- df |>
  mutate(log_follower = log(follower_count_final + 1))

cat(sprintf("  log_follower range: [%.2f, %.2f]\n",
    min(df$log_follower, na.rm = TRUE),
    max(df$log_follower, na.rm = TRUE)))

# Create log_ocr_text_len
df <- df |>
  mutate(log_ocr_text_len = log(ocr_text_len + 1))

cat(sprintf("  log_ocr_text_len range (non-NA): [%.2f, %.2f]\n",
    min(df$log_ocr_text_len, na.rm = TRUE),
    max(df$log_ocr_text_len, na.rm = TRUE)))

decisions <- bind_rows(decisions, tibble(
  feature   = c("log_follower", "log_ocr_text_len"),
  platform  = c("all", "tiktok, instagram"),
  action    = c("log(follower_count_final + 1)", "log(ocr_text_len + 1)"),
  rationale = c("Right-skewed distribution. Log stabilises variance.",
                "Right-skewed distribution. Log stabilises variance.")
))


# =============================================================================
# 3. Collapse lang to top levels + 'other' + 'unknown'
# =============================================================================
cat("\n== 3. Collapsing lang ==\n")

platforms <- c("tiktok", "instagram", "linkedin")

for (plat in platforms) {
  plat_data <- df |> filter(platform == plat)
  lang_counts <- plat_data |> count(lang, sort = TRUE) |> mutate(pct = n / sum(n))

  # Keep levels covering >= 95% cumulatively, plus "unknown"
  lang_counts <- lang_counts |> mutate(cum_pct = cumsum(pct))
  top_langs <- lang_counts |> filter(cum_pct <= 0.95 | lag(cum_pct, default = 0) < 0.95) |> pull(lang)
  # Always keep "unknown" as its own level
  top_langs <- union(top_langs, "unknown")

  cat(sprintf("  %s: keeping %d lang levels (%s)\n", plat,
      length(top_langs), paste(top_langs, collapse = ", ")))

  # All others -> "other"
  df <- df |>
    mutate(lang = if_else(
      platform == plat & !lang %in% top_langs,
      "other",
      lang
    ))
}

# Convert to factor
df$lang <- factor(df$lang)
cat(sprintf("  Final lang levels: %s\n", paste(levels(df$lang), collapse = ", ")))

decisions <- bind_rows(decisions, tibble(
  feature   = "lang",
  platform  = "all",
  action    = "collapse to top levels (>= 95% cumulative) + 'other' + 'unknown'",
  rationale = "Reduces cardinality while preserving dominant language effects."
))


# =============================================================================
# 4. Collapse media_type: lump rare levels
# =============================================================================
cat("\n== 4. Collapsing media_type ==\n")

MIN_OBS <- 30

for (plat in platforms) {
  plat_data <- df |> filter(platform == plat)
  mt_counts <- plat_data |> count(media_type, sort = TRUE)

  rare_levels <- mt_counts |> filter(n < MIN_OBS) |> pull(media_type)

  if (length(rare_levels) > 0) {
    cat(sprintf("  %s: lumping %d rare levels (n < %d) into 'other_format': %s\n",
        plat, length(rare_levels), MIN_OBS, paste(rare_levels, collapse = ", ")))

    df <- df |>
      mutate(media_type = if_else(
        platform == plat & media_type %in% rare_levels,
        "other_format",
        media_type
      ))
  } else {
    cat(sprintf("  %s: no rare levels (all >= %d)\n", plat, MIN_OBS))
  }
}

# Convert to factor
df$media_type <- factor(df$media_type)
cat(sprintf("  Final media_type levels: %s\n", paste(levels(df$media_type), collapse = ", ")))

decisions <- bind_rows(decisions, tibble(
  feature   = "media_type",
  platform  = "all",
  action    = sprintf("lump levels with < %d obs per platform into 'other_format'", MIN_OBS),
  rationale = "Ensures stable estimation. GAM needs sufficient obs per factor level."
))


# =============================================================================
# 5. Drop collinear features
# =============================================================================
cat("\n== 5. Dropping collinear features ==\n")

drop_cols <- collinear_drops$drop
cat(sprintf("  Dropping: %s\n", paste(drop_cols, collapse = ", ")))

df <- df |> select(-all_of(drop_cols))
cat(sprintf("  Columns after drop: %d\n", ncol(df)))


# =============================================================================
# 6. Split by platform and drop platform-specific features
# =============================================================================
cat("\n== 6. Splitting by platform ==\n")

# --- Columns to drop per platform ---
# Instagram: no temporal, no TikTok-only audio
ig_drop <- c("local_hour", "weekday", "is_weekend", "post_age_hours", "log_post_age",
             "audio_is_original", "uses_named_audio")

# LinkedIn: no visual features, no TikTok-only audio
li_drop <- c("brightness", "contrast", "colourfulness", "face_flag",
             "ocr_text_len", "log_ocr_text_len",
             "audio_is_original", "uses_named_audio")

# Also drop raw values that have been log-transformed (keep log version only)
# follower_count_final -> log_follower (drop raw from all)
# post_age_hours -> log_post_age (drop raw from TikTok/LinkedIn; already dropped for IG)
# ocr_text_len -> log_ocr_text_len (drop raw from TikTok/IG; already dropped for LI)
raw_drop_all <- c("follower_count_final", "post_age_hours", "ocr_text_len")

# --- TikTok ---
df_tt <- df |>
  filter(platform == "tiktok") |>
  select(-all_of(raw_drop_all))

cat(sprintf("  df_tt: %d rows x %d columns\n", nrow(df_tt), ncol(df_tt)))

# --- Instagram ---
ig_all_drop <- c(ig_drop, raw_drop_all)
# Remove any that are already gone (e.g. post_age_hours is in both ig_drop and raw_drop_all)
ig_all_drop <- unique(ig_all_drop)
# Only drop columns that actually exist
ig_all_drop <- intersect(ig_all_drop, names(df))

df_ig <- df |>
  filter(platform == "instagram") |>
  select(-all_of(ig_all_drop))

cat(sprintf("  df_ig: %d rows x %d columns\n", nrow(df_ig), ncol(df_ig)))

# --- LinkedIn ---
li_all_drop <- c(li_drop, raw_drop_all)
li_all_drop <- unique(li_all_drop)
li_all_drop <- intersect(li_all_drop, names(df))

df_li <- df |>
  filter(platform == "linkedin") |>
  select(-all_of(li_all_drop))

# Collapse LinkedIn account_type: "both" (13 posts) -> "seeded"
n_both_li <- sum(df_li$account_type == "both", na.rm = TRUE)
cat(sprintf("  LinkedIn account_type 'both': %d posts -> reassigning to 'seeded'\n", n_both_li))

df_li <- df_li |>
  mutate(
    account_type = if_else(account_type == "both", "seeded", account_type),
    account_type = factor(account_type, levels = c("fresh", "seeded"))
  )

cat(sprintf("  df_li: %d rows x %d columns\n", nrow(df_li), ncol(df_li)))

decisions <- bind_rows(decisions, tibble(
  feature   = c("temporal features", "audio_is_original, uses_named_audio",
                 "visual features", "account_type"),
  platform  = c("instagram", "instagram", "linkedin", "linkedin"),
  action    = c("drop (local_hour, weekday, is_weekend, post_age_hours, log_post_age)",
                "drop (TikTok-only features)",
                "drop (brightness, contrast, colourfulness, face_flag, ocr_text_len, log_ocr_text_len)",
                "collapse 'both' (13 posts) -> 'seeded'"),
  rationale = c("28.3% null from DOM-only capture artifact. Conditioning on completeness = collider bias.",
                "audio_is_original and uses_named_audio only available for TikTok.",
                "100% null: LinkedIn posts have no thumbnail in scraper.",
                "Only 13 'both' posts -- too few for stable estimation as separate level.")
))


# =============================================================================
# 7. Z-standardise continuous features within platform
# =============================================================================
cat("\n== 7. Z-standardising continuous features ==\n")

# Define which features to z-standardise.
# Do NOT standardise: binary, factors, local_hour/weekday (raw for bs='cc'),
#                     outcomes, identifiers, counters.

# Binary features (never standardise)
binary_features <- c("cta_flag", "is_weekend", "audio_present", "is_trending",
                     "audio_is_original", "uses_named_audio",
                     "flesch_available", "follower_available", "face_flag",
                     "ever_top")

# Factor features (never standardise)
factor_features <- c("lang", "media_type", "topic_cluster", "account_type")

# Cyclic features (keep raw for GAM bs='cc')
cyclic_features <- c("local_hour", "weekday")

# Identifiers (never standardise)
id_features <- c("post_id", "platform", "permalink", "author_hash")

# Outcomes and reference columns (never standardise)
outcome_features <- c("ever_top", "best_rank", "n_captures",
                      "velocity_24h", "velocity_72h", "posted_at_utc")

# Counter columns (reference, never standardise)
counter_features <- grep("^(likes|comments|shares|views)_t", names(df), value = TRUE)

# Everything NOT in the above sets gets standardised (if numeric)
no_standardise <- c(binary_features, factor_features, cyclic_features,
                    id_features, outcome_features, counter_features)

# Helper: z-standardise and return lookup
z_standardise <- function(df_plat, platform_name) {
  # Identify numeric columns to standardise
  all_cols <- names(df_plat)
  numeric_cols <- all_cols[sapply(df_plat, is.numeric)]
  to_std <- setdiff(numeric_cols, no_standardise)

  cat(sprintf("    Standardising %d features for %s\n", length(to_std), platform_name))

  lookup <- tibble(platform = character(), feature = character(),
                   mean = double(), sd = double())

  for (col in to_std) {
    col_mean <- mean(df_plat[[col]], na.rm = TRUE)
    col_sd   <- sd(df_plat[[col]], na.rm = TRUE)

    if (is.na(col_sd) || col_sd == 0) {
      cat(sprintf("    WARNING: %s has SD = 0 (or all NA). Skipping standardisation.\n", col))
      lookup <- bind_rows(lookup, tibble(
        platform = platform_name, feature = col, mean = col_mean, sd = NA_real_
      ))
      next
    }

    df_plat[[col]] <- (df_plat[[col]] - col_mean) / col_sd

    lookup <- bind_rows(lookup, tibble(
      platform = platform_name, feature = col, mean = col_mean, sd = col_sd
    ))
  }

  list(data = df_plat, lookup = lookup)
}

# Apply per platform
result_tt <- z_standardise(df_tt, "tiktok")
result_ig <- z_standardise(df_ig, "instagram")
result_li <- z_standardise(df_li, "linkedin")

df_tt <- result_tt$data
df_ig <- result_ig$data
df_li <- result_li$data

z_lookup <- bind_rows(result_tt$lookup, result_ig$lookup, result_li$lookup)
cat(sprintf("  z_score_lookup: %d entries\n", nrow(z_lookup)))


# =============================================================================
# 8. Verify and write
# =============================================================================
cat("\n== 8. Verification and output ==\n")

# -- Verify dimensions --
cat("\n  Final dimensions:\n")
cat(sprintf("    df_tt: %d rows x %d cols\n", nrow(df_tt), ncol(df_tt)))
cat(sprintf("    df_ig: %d rows x %d cols\n", nrow(df_ig), ncol(df_ig)))
cat(sprintf("    df_li: %d rows x %d cols\n", nrow(df_li), ncol(df_li)))

# -- Verify ever_top balance --
cat("\n  ever_top balance:\n")
for (nm in c("df_tt", "df_ig", "df_li")) {
  d <- get(nm)
  n_top <- sum(d$ever_top == 1, na.rm = TRUE)
  pct <- round(n_top / nrow(d) * 100, 1)
  cat(sprintf("    %s: %d / %d top (%.1f%%)\n", nm, n_top, nrow(d), pct))
}

# -- Verify z-scores (mean ~0, SD ~1 for standardised features) --
cat("\n  Z-score spot checks (first 5 standardised features per platform):\n")
for (nm in c("df_tt", "df_ig", "df_li")) {
  d <- get(nm)
  plat_name <- unique(d$platform)[1]
  std_feats <- z_lookup |> filter(platform == plat_name, !is.na(sd)) |> head(5)

  cat(sprintf("    %s:\n", nm))
  for (i in seq_len(nrow(std_feats))) {
    feat <- std_feats$feature[i]
    if (feat %in% names(d)) {
      m <- round(mean(d[[feat]], na.rm = TRUE), 4)
      s <- round(sd(d[[feat]], na.rm = TRUE), 4)
      cat(sprintf("      %s: mean = %.4f, sd = %.4f\n", feat, m, s))
    }
  }
}

# -- Verify local_hour/weekday NOT standardised --
cat("\n  Cyclic features (should be raw, not standardised):\n")
if ("local_hour" %in% names(df_tt)) {
  cat(sprintf("    df_tt local_hour range: [%.0f, %.0f]\n",
      min(df_tt$local_hour, na.rm = TRUE), max(df_tt$local_hour, na.rm = TRUE)))
}
if ("weekday" %in% names(df_tt)) {
  cat(sprintf("    df_tt weekday range: [%.0f, %.0f]\n",
      min(df_tt$weekday, na.rm = TRUE), max(df_tt$weekday, na.rm = TRUE)))
}

# -- Check for unexpected NAs in predictors --
cat("\n  Remaining NAs in predictor features:\n")
for (nm in c("df_tt", "df_ig", "df_li")) {
  d <- get(nm)
  # Exclude counters, outcomes, identifiers from NA check
  pred_cols <- setdiff(names(d), c(id_features, outcome_features, counter_features, "posted_at_utc"))
  na_counts <- colSums(is.na(d[, pred_cols, drop = FALSE]))
  has_na <- na_counts[na_counts > 0]
  if (length(has_na) > 0) {
    cat(sprintf("    %s: %s\n", nm, paste(names(has_na), has_na, sep = "=", collapse = ", ")))
  } else {
    cat(sprintf("    %s: none\n", nm))
  }
}

# -- Write outputs --
cat("\n  Writing parquet files:\n")

write_parquet(df_tt, file.path(DATA_OUT, "df_tt.parquet"))
cat(sprintf("    df_tt.parquet: %.1f MB\n", file.size(file.path(DATA_OUT, "df_tt.parquet")) / 1e6))

write_parquet(df_ig, file.path(DATA_OUT, "df_ig.parquet"))
cat(sprintf("    df_ig.parquet: %.1f MB\n", file.size(file.path(DATA_OUT, "df_ig.parquet")) / 1e6))

write_parquet(df_li, file.path(DATA_OUT, "df_li.parquet"))
cat(sprintf("    df_li.parquet: %.1f MB\n", file.size(file.path(DATA_OUT, "df_li.parquet")) / 1e6))

write_csv(z_lookup, file.path(OUT_DIR, "z_score_lookup.csv"))
cat(sprintf("    z_score_lookup.csv: %d entries\n", nrow(z_lookup)))

# Update feature decisions
write_csv(decisions, file.path(OUT_DIR, "feature_decisions.csv"))
cat(sprintf("    feature_decisions.csv: %d total rows\n", nrow(decisions)))

# -- Print column lists for reference --
cat("\n  === Column lists ===\n")
cat(sprintf("  df_tt columns (%d):\n    %s\n", ncol(df_tt),
    paste(names(df_tt), collapse = ", ")))
cat(sprintf("\n  df_ig columns (%d):\n    %s\n", ncol(df_ig),
    paste(names(df_ig), collapse = ", ")))
cat(sprintf("\n  df_li columns (%d):\n    %s\n", ncol(df_li),
    paste(names(df_li), collapse = ", ")))


cat("\n== Step 3c complete ==\n")
