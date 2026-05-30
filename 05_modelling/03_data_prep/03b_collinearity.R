# =============================================================================
# 03b_collinearity.R
#
# Master's thesis: Learning the Levers — Which Creator-Controllable
# Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
# from a Swiss German-Language Perspective
# Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026
#
# Purpose:
#   Screens numeric features for high pairwise correlation (|r| > 0.80) and
#   selects the more interpretable feature to keep in each collinear pair.
#   Diagnostic only; the actual drops happen in 03c_transformations.R.
#
# Pipeline position:
#   Step 3b of the modelling pipeline. Depends on 03a_missingness.R; feeds
#   03c_transformations.R via collinear_drops.csv.
#
# Inputs:
#   05_modelling/01_build_analytical_table/data/df_post.parquet  Post-level analytical table
#   05_modelling/03_data_prep/output/feature_decisions.csv       Decision log from 03a
#
# Outputs:
#   05_modelling/03_data_prep/output/collinear_pairs.csv         All pairs with |r| > 0.80
#   05_modelling/03_data_prep/output/collinear_drops.csv         Actionable drop list
#   05_modelling/03_data_prep/output/corr_heatmap_*.png          Per-platform correlation heatmaps
#   05_modelling/03_data_prep/output/feature_decisions.csv       Updated decision log
#
# Usage:
#   Rscript 05_modelling/03_data_prep/03b_collinearity.R
# =============================================================================

rm(list = ls())

# -- Source shared config ------------------------------------------------------
source(file.path("config", "packages.R"))

STEP_DIR <- file.path(BASE_DIR, "03_data_prep")
OUT_DIR  <- file.path(STEP_DIR, "output")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)


# =============================================================================
# 1. Load data and existing decision log
# =============================================================================
cat("== 1. Loading df_post.parquet and feature_decisions.csv ==\n")

df <- read_parquet(file.path(DATA_DIR, "df_post.parquet"))
cat(sprintf("  Loaded: %d rows x %d columns\n", nrow(df), ncol(df)))

# Verify 03a outputs exist
stopifnot("flesch_available missing (run 03a first)" = "flesch_available" %in% names(df))
stopifnot("follower_available missing (run 03a first)" = "follower_available" %in% names(df))
stopifnot("log_post_age missing (run 03a first)" = "log_post_age" %in% names(df))

decisions <- read_csv(file.path(OUT_DIR, "feature_decisions.csv"),
                      show_col_types = FALSE)
cat(sprintf("  Loaded %d existing decisions from 03a.\n", nrow(decisions)))


# =============================================================================
# 2. Define numeric predictor set
# =============================================================================
cat("\n== 2. Defining numeric predictors for correlation check ==\n")

# Continuous features that will enter the GAM as smooth terms or linear terms.
# Exclude: identifiers, outcomes, velocities, counters, PCs (sensitivity only),
#          binary indicators, and factors.
numeric_predictors <- c(
  # Text (continuous)
  "caption_len", "word_count", "hashtag_count", "emoji_count",
  # Style (continuous)
  "sentence_count", "avg_sentence_len", "exclamation_density", "question_density",
  "ellipsis_count", "caps_ratio", "caps_word_count", "line_break_count",
  "url_count", "mention_count", "punct_diversity", "flesch_reading_ease",
  # Visual (continuous)
  "brightness", "contrast", "colourfulness", "face_count", "ocr_text_len",
  # Temporal (continuous -- local_hour/weekday are integer but treated as continuous for correlation)
  "local_hour", "weekday", "post_age_hours",
  # Metadata (continuous)
  "follower_count_final"
)

# Verify all exist
missing_cols <- setdiff(numeric_predictors, names(df))
if (length(missing_cols) > 0) {
  cat(sprintf("  WARNING: Missing columns: %s\n", paste(missing_cols, collapse = ", ")))
}

cat(sprintf("  Numeric predictors for correlation: %d features\n", length(numeric_predictors)))
cat(sprintf("  Features: %s\n", paste(numeric_predictors, collapse = ", ")))


# =============================================================================
# 3. Per-platform correlation matrices
# =============================================================================
cat("\n== 3. Computing per-platform Pearson correlations ==\n")

THRESHOLD <- 0.80
platforms <- c("tiktok", "instagram", "linkedin")
all_pairs <- tibble()

for (plat in platforms) {
  cat(sprintf("\n  --- %s ---\n", plat))

  df_plat <- df |> filter(platform == plat)

  # Select features available for this platform (drop all-NA columns)
  avail_feats <- numeric_predictors[
    sapply(numeric_predictors, function(f) sum(!is.na(df_plat[[f]])) > 10)
  ]
  cat(sprintf("  Available features: %d\n", length(avail_feats)))

  cor_mat <- cor(df_plat[, avail_feats], use = "pairwise.complete.obs")

  # Extract pairs above threshold
  pairs <- which(abs(cor_mat) > THRESHOLD & upper.tri(cor_mat), arr.ind = TRUE)

  if (nrow(pairs) > 0) {
    pair_df <- tibble(
      platform = plat,
      feature_1 = avail_feats[pairs[, 1]],
      feature_2 = avail_feats[pairs[, 2]],
      correlation = map2_dbl(pairs[, 1], pairs[, 2], ~ cor_mat[.x, .y])
    ) |>
      arrange(desc(abs(correlation)))

    cat(sprintf("  Pairs with |r| > %.2f:\n", THRESHOLD))
    print(as.data.frame(pair_df), row.names = FALSE)
    all_pairs <- bind_rows(all_pairs, pair_df)
  } else {
    cat(sprintf("  No pairs with |r| > %.2f\n", THRESHOLD))
  }
}


# =============================================================================
# 4. Save correlation heatmaps
# =============================================================================
cat("\n== 4. Saving correlation heatmaps ==\n")

for (plat in platforms) {
  df_plat <- df |> filter(platform == plat)

  avail_feats <- numeric_predictors[
    sapply(numeric_predictors, function(f) sum(!is.na(df_plat[[f]])) > 10)
  ]

  cor_mat <- cor(df_plat[, avail_feats], use = "pairwise.complete.obs")

  out_file <- file.path(OUT_DIR, sprintf("corr_heatmap_%s.png", plat))
  png(out_file, width = 1200, height = 1000, res = 120)
  corrplot(cor_mat,
           method = "color",
           type = "upper",
           tl.cex = 0.7,
           tl.col = "black",
           addCoef.col = "black",
           number.cex = 0.5,
           title = sprintf("Feature Correlations: %s", plat),
           mar = c(0, 0, 2, 0))
  dev.off()
  cat(sprintf("  Saved: %s\n", out_file))
}


# =============================================================================
# 5. Document drop decisions
# =============================================================================
cat("\n== 5. Collinearity drop decisions ==\n")

# Pre-defined decisions based on modelling plan
expected_drops <- tribble(
  ~drop,            ~keep,                ~rationale,
  "caption_len",    "word_count",         "word_count is more interpretable for content creators (actionable unit).",
  "sentence_count", "word_count",         "word_count subsumes sentence_count; keeping the more general feature.",
  "face_count",     "face_flag",          "face_flag (binary) is simpler and avoids count-to-binary redundancy."
)

# Verify these pairs actually appear above the threshold
cat("  Verifying expected collinear pairs against observed data:\n")
for (i in seq_len(nrow(expected_drops))) {
  feat_a <- expected_drops$drop[i]
  feat_b <- expected_drops$keep[i]

  # Check if this pair was found above threshold on any platform
  found <- all_pairs |>
    filter((feature_1 == feat_a & feature_2 == feat_b) |
           (feature_1 == feat_b & feature_2 == feat_a))

  if (nrow(found) > 0) {
    cat(sprintf("  OK: %s <-> %s (|r| = %s on %s)\n",
        feat_a, feat_b,
        paste(round(found$correlation, 3), collapse = ", "),
        paste(found$platform, collapse = ", ")))
  } else {
    cat(sprintf("  NOTE: %s <-> %s NOT above threshold (%.2f). Checking actual r:\n",
        feat_a, feat_b, THRESHOLD))
    for (plat in platforms) {
      df_plat <- df |> filter(platform == plat)
      if (all(c(feat_a, feat_b) %in% names(df_plat))) {
        r <- cor(df_plat[[feat_a]], df_plat[[feat_b]], use = "pairwise.complete.obs")
        cat(sprintf("    %s: r = %.3f\n", plat, r))
      }
    }
  }
}

# Check for unexpected pairs (above threshold but not in expected list)
known_pairs <- bind_rows(
  expected_drops |> select(feature_1 = drop, feature_2 = keep),
  expected_drops |> select(feature_1 = keep, feature_2 = drop)
)

unexpected <- all_pairs |>
  anti_join(known_pairs, by = c("feature_1", "feature_2"))

if (nrow(unexpected) > 0) {
  cat("\n  WARNING: Unexpected collinear pairs found:\n")
  print(as.data.frame(unexpected), row.names = FALSE)
  cat("  Review these and decide whether to add drop rules.\n")
} else {
  cat("\n  No unexpected collinear pairs. Expected pairs confirmed.\n")
}


# =============================================================================
# 6. Write outputs
# =============================================================================
cat("\n== 6. Writing outputs ==\n")

# Write all pairs (diagnostic)
write_csv(all_pairs, file.path(OUT_DIR, "collinear_pairs.csv"))
cat(sprintf("  collinear_pairs.csv: %d pairs written\n", nrow(all_pairs)))

# Write actionable drop list (consumed by 03c)
write_csv(expected_drops, file.path(OUT_DIR, "collinear_drops.csv"))
cat(sprintf("  collinear_drops.csv: %d drop decisions written\n", nrow(expected_drops)))

# Append to feature decisions
for (i in seq_len(nrow(expected_drops))) {
  decisions <- bind_rows(decisions, tibble(
    feature   = expected_drops$drop[i],
    platform  = "all",
    action    = sprintf("drop (collinear with %s)", expected_drops$keep[i]),
    rationale = expected_drops$rationale[i]
  ))
}

write_csv(decisions, file.path(OUT_DIR, "feature_decisions.csv"))
cat(sprintf("  feature_decisions.csv: %d total rows\n", nrow(decisions)))

cat("\n  NOTE: This script did NOT modify df_post.parquet.\n")
cat("  Collinear features will be dropped in 03c_transformations.R.\n")


cat("\n== Step 3b complete ==\n")
