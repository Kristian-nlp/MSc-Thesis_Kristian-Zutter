# =============================================================================
# 09_strategy_matrix.R
#
# Master's thesis: Learning the Levers — Which Creator-Controllable
# Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
# from a Swiss German-Language Perspective
# Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026
#
# Purpose:
#   Translates statistical findings from all GAM models into an actionable
#   strategy matrix for Swiss German-language marketing teams (RQ3). For
#   each significant feature, determines (a) whether it is
#   creator-controllable, (b) the optimal range from the partial-effect
#   curve, and (c) whether the effect generalises across platforms.
#
# Pipeline position:
#   Step 9 of the modelling pipeline. Depends on Steps 1–8; feeds the
#   strategy heatmap and optimal-range panel in Chapter 5.
#
# Inputs:
#   05_modelling/03_data_prep/data/df_tt.parquet                    TikTok analytical frame
#   05_modelling/03_data_prep/data/df_ig.parquet                    Instagram analytical frame
#   05_modelling/03_data_prep/data/df_li.parquet                    LinkedIn analytical frame
#   05_modelling/05_gam/models/m_*.rds                              All fitted GAMs
#   05_modelling/03_data_prep/output/z_score_lookup.csv             Z-score parameters
#
# Outputs:
#   05_modelling/09_strategy/output/strategy_matrix.csv             Long format
#   05_modelling/09_strategy/output/strategy_matrix_wide_*.csv      Wide per-outcome summary
#   05_modelling/09_strategy/output/strategy_heatmap.png            Thesis figure
#   05_modelling/09_strategy/output/optimal_range_panel.png         Thesis figure
#
# Usage:
#   Rscript 05_modelling/09_strategy/09_strategy_matrix.R
# =============================================================================

rm(list = ls())

# -- Source shared config ------------------------------------------------------
source(file.path("config", "packages.R"))
library(gratia)
library(patchwork)

STEP_DIR  <- file.path(BASE_DIR, "09_strategy")
OUT_DIR   <- file.path(STEP_DIR, "output")
PLOT_DIR  <- OUT_DIR
GAM_DIR   <- file.path(BASE_DIR, "05_gam", "models")
PREP_DIR  <- file.path(BASE_DIR, "03_data_prep", "data")
ZSCORE_FILE <- file.path(BASE_DIR, "03_data_prep", "output", "z_score_lookup.csv")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)


# =============================================================================
# 0. CONTROLLABILITY LOOKUP
# =============================================================================

# Static classification: is this feature controllable by a content creator?
CONTROLLABLE <- c(
  # Per-post controllable (yes)
  word_count          = "yes",
  hashtag_count       = "yes",
  emoji_count         = "yes",
  caps_ratio          = "yes",
  avg_sentence_len    = "yes",
  question_density    = "yes",
  exclamation_density = "yes",
  ellipsis_count      = "yes",
  caps_word_count     = "yes",
  line_break_count    = "yes",
  url_count           = "yes",
  mention_count       = "yes",
  punct_diversity     = "yes",
  flesch_reading_ease = "yes",
  media_type          = "yes",
  face_flag           = "yes",
  brightness          = "yes",
  contrast            = "yes",
  colourfulness       = "yes",
  log_ocr_text_len    = "yes",
  audio_is_original   = "yes",
  uses_named_audio    = "yes",
  cta_flag            = "yes",
  # Timing (posting schedule)
  weekday             = "timing",
  local_hour          = "timing",
  is_weekend          = "timing",
  # Indirect
  topic_cluster       = "indirect",
  is_trending         = "indirect",
  # Not controllable per post
  log_post_age        = "no",
  log_follower        = "no",
  account_type        = "no",
  follower_available  = "no",
  flesch_available    = "no",
  lang                = "no"
)

# Outcome-specific caveats for instrumentation-dominated models
MODEL_CAVEATS <- c(
  ig_inclusion    = "59.4% deviance driven by instrumentation controls (follower_available, media_type=reel)",
  li_velocity_24h = "Historically dominated by log_post_age; revisit under combined-engagement velocity (Decision 38)"
  # ig_velocity_24h removed: Decision 37 (2026-04-17) + Amendment. Instagram
  # velocity is not modelled (T24/T72 field-extraction failure on likes AND
  # comments). No caveat row is needed since no Instagram velocity row will
  # appear in strategy_matrix.csv.
)

# Pretty labels for natural-unit output
FEATURE_LABELS <- c(
  word_count          = "Word count",
  hashtag_count       = "Hashtag count",
  emoji_count         = "Emoji count",
  caps_ratio          = "Capitalisation ratio",
  avg_sentence_len    = "Avg sentence length (words)",
  question_density    = "Question mark density",
  exclamation_density = "Exclamation density",
  ellipsis_count      = "Ellipsis count",
  caps_word_count     = "Capitalised word count",
  line_break_count    = "Line breaks",
  url_count           = "URL count",
  mention_count       = "Mention count",
  punct_diversity     = "Punctuation diversity",
  flesch_reading_ease = "Flesch reading ease",
  media_type          = "Media type",
  face_flag           = "Face present",
  brightness          = "Image brightness",
  contrast            = "Image contrast",
  colourfulness       = "Image colourfulness",
  log_ocr_text_len    = "Overlay text length (log)",
  audio_is_original   = "Original audio",
  uses_named_audio    = "Named audio track",
  cta_flag            = "Call-to-action",
  weekday             = "Day of week",
  local_hour          = "Hour of day",
  is_weekend          = "Weekend",
  is_trending         = "Trending audio",
  topic_cluster       = "Topic cluster",
  log_post_age        = "Post age (log hours)",
  log_follower        = "Follower count (log)",
  account_type        = "Account type",
  follower_available  = "Follower data available",
  flesch_available    = "Readability available",
  lang                = "Language"
)


# =============================================================================
# 0b. HELPER: Extract GAM term info (from 07a)
# =============================================================================

extract_gam_terms <- function(gam_model) {
  terms_out <- tibble(feature = character(), p_value = numeric(),
                      feature_type = character(), direction = character(),
                      edf = numeric())

  # --- Smooth terms ---
  sm <- summary(gam_model)$s.table
  if (!is.null(sm) && nrow(sm) > 0) {
    for (i in seq_len(nrow(sm))) {
      term <- rownames(sm)[i]
      feat <- gsub("^s\\(([^,)]+).*\\)$", "\\1", term)
      edf_val <- sm[i, "edf"]
      dir <- if (edf_val < 1.5) "approx. linear" else "non-linear"
      terms_out <- bind_rows(terms_out, tibble(
        feature = feat, p_value = sm[i, "p-value"],
        feature_type = "smooth", direction = dir, edf = edf_val
      ))
    }
  }

  # --- Parametric terms ---
  pt <- summary(gam_model)$p.table
  p_col <- if ("Pr(>|z|)" %in% colnames(pt)) "Pr(>|z|)" else "Pr(>|t|)"
  if (!is.null(pt) && nrow(pt) > 0) {
    for (i in seq_len(nrow(pt))) {
      term <- rownames(pt)[i]
      if (term == "(Intercept)") next
      feat <- term
      for (fvar in c("lang", "media_type", "topic_cluster", "account_type")) {
        if (startsWith(term, fvar)) { feat <- fvar; break }
      }
      est <- pt[i, "Estimate"]
      dir <- if (is.nan(est)) "aliased" else if (est > 0) "positive" else "negative"
      terms_out <- bind_rows(terms_out, tibble(
        feature = feat, p_value = pt[i, p_col],
        feature_type = "parametric", direction = dir, edf = NA_real_
      ))
    }
  }

  # Collapse factor levels: keep min p-value, first type/direction, mean edf
  terms_out <- terms_out |>
    filter(!is.nan(p_value)) |>
    group_by(feature) |>
    summarise(
      p_value      = min(p_value, na.rm = TRUE),
      feature_type = first(feature_type),
      direction    = first(direction),
      edf          = if (all(is.na(edf))) NA_real_ else mean(edf, na.rm = TRUE),
      .groups      = "drop"
    ) |>
    mutate(significant = p_value < 0.05)

  terms_out
}


# =============================================================================
# 0c. HELPER: Median profile and magnitude (from 07a)
# =============================================================================

make_median_profile <- function(df, model) {
  pred_names <- attr(model$terms, "term.labels")
  sm_names   <- names(model$var.summary)
  all_vars   <- unique(c(pred_names, sm_names))
  profile    <- list()
  for (v in all_vars) {
    if (!(v %in% names(df))) next
    if (is.factor(df[[v]])) {
      profile[[v]] <- names(sort(table(df[[v]]), decreasing = TRUE))[1]
      profile[[v]] <- factor(profile[[v]], levels = levels(df[[v]]))
    } else if (is.logical(df[[v]]) || all(df[[v]] %in% c(0, 1), na.rm = TRUE)) {
      profile[[v]] <- as.numeric(names(sort(table(df[[v]]), decreasing = TRUE))[1])
    } else {
      profile[[v]] <- median(df[[v]], na.rm = TRUE)
    }
  }
  as_tibble(profile)
}

compute_magnitude <- function(feature, model, df, profile) {
  if (!(feature %in% names(df))) return(NA_real_)
  col <- df[[feature]]

  if (is.factor(col)) {
    pt <- summary(model)$p.table
    level_rows <- grep(paste0("^", feature), rownames(pt))
    if (length(level_rows) == 0) return(NA_real_)
    level_freq <- table(col)
    kept_coefs <- c(0)
    for (idx in level_rows) {
      est <- pt[idx, "Estimate"]
      if (is.nan(est)) next
      lvl_name <- sub(paste0("^", feature), "", rownames(pt)[idx])
      if (lvl_name %in% names(level_freq) && level_freq[[lvl_name]] >= 30) {
        kept_coefs <- c(kept_coefs, est)
      }
    }
    if (length(kept_coefs) < 2) {
      all_coefs <- pt[level_rows, "Estimate"]
      all_coefs <- c(0, all_coefs[!is.nan(all_coefs)])
      return(max(all_coefs) - min(all_coefs))
    }
    return(max(kept_coefs) - min(kept_coefs))
  }

  if (is.logical(col) || all(col %in% c(0, 1), na.rm = TRUE)) {
    pt <- summary(model)$p.table
    if (feature %in% rownames(pt)) {
      est <- pt[feature, "Estimate"]
      if (is.nan(est)) return(NA_real_)
      return(est)
    }
    newdata_0 <- profile; newdata_0[[feature]] <- 0
    newdata_1 <- profile; newdata_1[[feature]] <- 1
    pred_0 <- tryCatch(predict(model, newdata = newdata_0, type = "link"),
                       error = function(e) NA_real_)
    pred_1 <- tryCatch(predict(model, newdata = newdata_1, type = "link"),
                       error = function(e) NA_real_)
    if (is.na(pred_0) || is.na(pred_1)) return(NA_real_)
    return(pred_1 - pred_0)
  }

  # Continuous: predict at Q25 and Q75
  q25 <- quantile(col, 0.25, na.rm = TRUE)
  q75 <- quantile(col, 0.75, na.rm = TRUE)
  if (q25 == q75) return(0)
  newdata_lo <- profile; newdata_lo[[feature]] <- q25
  newdata_hi <- profile; newdata_hi[[feature]] <- q75
  pred_lo <- tryCatch(predict(model, newdata = newdata_lo, type = "link"),
                      error = function(e) NA_real_)
  pred_hi <- tryCatch(predict(model, newdata = newdata_hi, type = "link"),
                      error = function(e) NA_real_)
  if (is.na(pred_lo) || is.na(pred_hi)) return(NA_real_)
  pred_hi - pred_lo
}


# =============================================================================
# 0d. HELPER: Optimal range extraction (novel)
# =============================================================================

# Back-transform z-standardised value to natural units
z_to_natural <- function(z_val, feature, platform, z_lookup) {
  # Cyclic features (weekday, local_hour) are NOT z-standardised -- return raw
  if (feature %in% CYCLIC_FEATURES) return(z_val)

  row <- z_lookup |> filter(.data$feature == !!feature, .data$platform == !!platform)
  if (nrow(row) == 0) return(z_val)  # fallback: return z if no lookup
  nat <- z_val * row$sd + row$mean
  # For log-transformed features, exponentiate (log(x+1) -> x)
  if (feature %in% LOG_FEATURES) {
    nat <- exp(nat) - 1
    nat <- max(0, nat)  # floor at 0 (can't have negative counts/followers)
  }
  nat
}

# Day-of-week labels for cyclic weekday feature (ISO: 0=Mon, 6=Sun)
WEEKDAY_LABELS <- c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

# Features that are cyclic (not z-standardised, use raw values)
CYCLIC_FEATURES <- c("weekday", "local_hour")

# Features that were log-transformed before z-standardisation
LOG_FEATURES <- c("log_follower", "log_ocr_text_len", "log_post_age")

# Format a natural-unit value to sensible precision
fmt_natural <- function(x, feature) {
  if (is.na(x) || is.infinite(x)) return("NA")
  if (feature == "weekday") {
    idx <- max(0, min(6, round(x)))
    return(WEEKDAY_LABELS[idx + 1])
  }
  if (feature == "local_hour") return(sprintf("%02d:00", round(x) %% 24))
  if (abs(x) >= 1000) return(format(round(x), big.mark = ","))
  if (abs(x) >= 100)  return(sprintf("%.0f", x))
  if (abs(x) >= 10)   return(sprintf("%.0f", x))
  if (abs(x) >= 1)    return(sprintf("%.1f", x))
  if (abs(x) >= 0.01) return(sprintf("%.2f", x))
  if (x == 0) return("0")
  sprintf("%.3f", x)
}

# Extract optimal range for a smooth term
extract_optimal_smooth <- function(model, feature, platform, good_direction,
                                   z_lookup) {
  # Find the smooth label in the model
  sm_names <- rownames(summary(model)$s.table)
  label <- sm_names[grep(paste0("\\b", feature, "\\b"), sm_names)]
  if (length(label) == 0) return(list(range_text = "n/a", z_lo = NA, z_hi = NA))
  label <- label[1]

  se <- tryCatch(
    smooth_estimates(model, select = label, n = 200),
    error = function(e) NULL
  )
  if (is.null(se) || nrow(se) == 0) {
    return(list(range_text = "Extraction failed", z_lo = NA, z_hi = NA))
  }

  # Get the feature column (the varying one)
  x_col <- feature
  if (!(x_col %in% names(se))) {
    # gratia may use a different column name
    data_cols <- setdiff(names(se), c(".smooth", ".type", ".by", ".estimate",
                                      ".se", ".lower_ci", ".upper_ci"))
    if (length(data_cols) == 1) x_col <- data_cols[1]
    else return(list(range_text = "Column not found", z_lo = NA, z_hi = NA))
  }

  x_vals <- se[[x_col]]
  est    <- se$.estimate
  se_val <- se$.se
  ci_lo  <- est - 1.96 * se_val
  ci_hi  <- est + 1.96 * se_val

  # Determine "good" region based on outcome semantics
  if (good_direction == "positive") {
    # Inclusion/velocity: positive effect is good
    sig_good <- ci_lo > 0
  } else {
    # Rank: negative effect is good (lower rank = better)
    sig_good <- ci_hi < 0
  }

  if (any(sig_good)) {
    # Use the range where effect is significantly in the good direction
    good_x <- x_vals[sig_good]
    z_lo <- min(good_x)
    z_hi <- max(good_x)
  } else {
    # Fallback: peak region within 80% of best effect
    if (good_direction == "positive") {
      peak_val <- max(est)
    } else {
      peak_val <- min(est)
    }
    threshold <- peak_val * 0.8
    if (good_direction == "positive") {
      in_peak <- est >= threshold
    } else {
      in_peak <- est <= threshold
    }
    if (!any(in_peak)) in_peak <- est == peak_val
    peak_x <- x_vals[in_peak]
    z_lo <- min(peak_x)
    z_hi <- max(peak_x)
  }

  # Back-transform to natural units
  nat_lo <- z_to_natural(z_lo, feature, platform, z_lookup)
  nat_hi <- z_to_natural(z_hi, feature, platform, z_lookup)

  # Ensure lo <= hi
  if (nat_lo > nat_hi) { tmp <- nat_lo; nat_lo <- nat_hi; nat_hi <- tmp }

  # Build text
  feat_label <- ifelse(feature %in% names(FEATURE_LABELS),
                       FEATURE_LABELS[feature], feature)
  range_text <- paste0(fmt_natural(nat_lo, feature), " - ",
                       fmt_natural(nat_hi, feature))

  # Flag if fallback was used
  if (!any(sig_good)) {
    range_text <- paste0(range_text, " (peak region, CI includes zero)")
  }

  list(range_text = range_text, z_lo = z_lo, z_hi = z_hi,
       nat_lo = nat_lo, nat_hi = nat_hi)
}

# Extract optimal for binary/factor parametric terms
extract_optimal_parametric <- function(model, feature, df, good_direction) {
  col <- df[[feature]]

  if (is.factor(col)) {
    pt <- summary(model)$p.table
    level_rows <- grep(paste0("^", feature), rownames(pt))
    if (length(level_rows) == 0) return("n/a")
    level_freq <- table(col)
    ref_level <- levels(col)[1]
    best_level <- ref_level
    best_effect <- 0  # reference level
    for (idx in level_rows) {
      est <- pt[idx, "Estimate"]
      if (is.nan(est)) next
      lvl_name <- sub(paste0("^", feature), "", rownames(pt)[idx])
      if (!(lvl_name %in% names(level_freq)) || level_freq[[lvl_name]] < 30) next
      if (good_direction == "positive" && est > best_effect) {
        best_effect <- est; best_level <- lvl_name
      } else if (good_direction == "negative" && est < best_effect) {
        best_effect <- est; best_level <- lvl_name
      }
    }
    return(best_level)
  }

  # Binary
  pt <- summary(model)$p.table
  p_col <- if ("Pr(>|z|)" %in% colnames(pt)) "Pr(>|z|)" else "Pr(>|t|)"
  if (feature %in% rownames(pt)) {
    est <- pt[feature, "Estimate"]
    if (is.nan(est)) return("n/a")
    # Positive coef means level=1 is associated with higher response
    if (good_direction == "positive") {
      return(if (est > 0) "Yes (set to 1)" else "No (set to 0)")
    } else {
      return(if (est < 0) "Yes (set to 1)" else "No (set to 0)")
    }
  }
  "n/a"
}


# =============================================================================
# 0e. HELPER: Recommendation text generator
# =============================================================================

OUTCOME_LABELS <- c(
  inclusion = "top-20 visibility",
  rank      = "rank placement (lower = better)",
  velocity  = "engagement growth"
)

make_recommendation <- function(feature, controllable, direction, optimal_range,
                                outcome, platform, magnitude, effect_label, caveat) {
  outcome_label <- OUTCOME_LABELS[outcome]
  feat_label <- ifelse(feature %in% names(FEATURE_LABELS),
                       FEATURE_LABELS[feature], feature)

  if (controllable == "no") {
    return(paste0(feat_label, ": significant but not creator-controllable. Track as KPI."))
  }

  # For factors/binary, the optimal_range IS the best level, so always say "higher/better"
  # For smooths with non-linear direction, the optimal range already captures the good region
  is_factor_or_binary <- !(direction %in% c("positive", "negative",
                                             "non-linear", "approx. linear"))
  # Default: use direction to determine good_word
  if (outcome == "rank") {
    good_word <- "better"
    if (direction == "positive") good_word <- "worse"
  } else {
    good_word <- "higher"
    if (direction == "negative") good_word <- "lower"
  }
  # For non-linear smooths and factor/binary terms, the optimal range itself
  # points to the best region/level, so always use the positive framing
  if (direction %in% c("non-linear", "approx. linear") || is_factor_or_binary) {
    good_word <- ifelse(outcome == "rank", "better", "higher")
  }

  # Effect size label for the recommendation text
  eff_text <- ""
  if (!is.na(effect_label) && nchar(effect_label) > 0) {
    eff_text <- paste0(" (", effect_label, ")")
  }

  # Build recommendation
  rec <- ""
  if (controllable == "yes") {
    if (optimal_range != "n/a" && optimal_range != "Extraction failed") {
      rec <- paste0("Aim for ", optimal_range,
                    " -- associated with ", good_word, " ", outcome_label,
                    eff_text, ".")
    } else {
      rec <- paste0(feat_label, " is significant for ", outcome_label,
                    " (", direction, " association", eff_text, ").")
    }
  } else if (controllable == "timing") {
    rec <- paste0("Post timing: ", optimal_range,
                  " -- associated with ", good_word, " ", outcome_label,
                  eff_text, ".")
  } else if (controllable == "indirect") {
    rec <- paste0(feat_label, " (indirectly controllable): ", optimal_range,
                  " -- associated with ", good_word, " ", outcome_label,
                  eff_text, ".")
  }

  # Append caveat if present
  if (!is.na(caveat) && nchar(caveat) > 0) {
    rec <- paste0(rec, " [Caveat: ", caveat, "]")
  }

  rec
}


# =============================================================================
# 1. LOAD ALL 10 MODELS + DATA
# =============================================================================

cat("== 1. Loading all GAM models and platform data ==\n")

# Model registry
model_registry <- tribble(
  ~model_id,          ~platform,    ~outcome,    ~good_direction, ~rds_file,
  "tt_inclusion",     "tiktok",     "inclusion", "positive",      "m_tt_inclusion.rds",
  "ig_inclusion",     "instagram",  "inclusion", "positive",      "m_ig_inclusion.rds",
  "li_inclusion",     "linkedin",   "inclusion", "positive",      "m_li_inclusion.rds",
  "tt_rank",          "tiktok",     "rank",      "negative",      "m_tt_rank.rds",
  "ig_rank",          "instagram",  "rank",      "negative",      "m_ig_rank.rds",
  "li_rank",          "linkedin",   "rank",      "negative",      "m_li_rank.rds",
  "tt_velocity_24h",  "tiktok",     "velocity",  "positive",      "m_tt_velocity_24h.rds",
  "tt_velocity_72h",  "tiktok",     "velocity",  "positive",      "m_tt_velocity_72h.rds",
  # ig_velocity_24h removed (Decision 37 + Amendment, 2026-04-17).
  "li_velocity_24h",  "linkedin",   "velocity",  "positive",      "m_li_velocity_24h.rds"
)

# Load models into named list
models <- list()
for (i in seq_len(nrow(model_registry))) {
  mid <- model_registry$model_id[i]
  rds <- file.path(GAM_DIR, model_registry$rds_file[i])
  if (!file.exists(rds)) {
    cat(sprintf("  WARNING: %s not found, skipping\n", basename(rds)))
    next
  }
  models[[mid]] <- readRDS(rds)
  cat(sprintf("  Loaded: %-20s (deviance %.1f%%)\n",
              mid, summary(models[[mid]])$dev.expl * 100))
}

# Load platform data
df_tt <- read_parquet(file.path(PREP_DIR, "df_tt.parquet"))
df_ig <- read_parquet(file.path(PREP_DIR, "df_ig.parquet"))
df_li <- read_parquet(file.path(PREP_DIR, "df_li.parquet"))

# Ensure factors
factor_vars <- c("lang", "media_type", "topic_cluster", "account_type")
for (v in factor_vars) {
  if (v %in% names(df_tt)) df_tt[[v]] <- as.factor(df_tt[[v]])
  if (v %in% names(df_ig)) df_ig[[v]] <- as.factor(df_ig[[v]])
  if (v %in% names(df_li)) df_li[[v]] <- as.factor(df_li[[v]])
}

# Prepare rank subsets (ever_top == 1 only, matching 05b)
df_tt_rank <- df_tt |> filter(ever_top == 1)
df_ig_rank <- df_ig |> filter(ever_top == 1)
df_li_rank <- df_li |> filter(ever_top == 1)

# Prepare velocity subsets + gamma shift (matching 05c)
shift_for_gamma <- function(x) {
  min_val <- min(x, na.rm = TRUE)
  if (min_val <= 0) x + abs(min_val) + 0.001 else x
}

df_tt_v24 <- df_tt |> filter(!is.na(velocity_24h))
df_tt_v24$velocity_24h_g <- shift_for_gamma(df_tt_v24$velocity_24h)

df_tt_v72 <- df_tt |> filter(!is.na(velocity_72h))
df_tt_v72$velocity_72h_g <- shift_for_gamma(df_tt_v72$velocity_72h)

# Instagram velocity data-frame omitted: Decision 37 + Amendment.

df_li_v24 <- df_li |> filter(!is.na(velocity_24h))
df_li_v24$velocity_24h_g <- shift_for_gamma(df_li_v24$velocity_24h)

# Map each model_id to its correct data frame
data_map <- list(
  tt_inclusion    = df_tt,
  ig_inclusion    = df_ig,
  li_inclusion    = df_li,
  tt_rank         = df_tt_rank,
  ig_rank         = df_ig_rank,
  li_rank         = df_li_rank,
  tt_velocity_24h = df_tt_v24,
  tt_velocity_72h = df_tt_v72,
  # ig_velocity_24h entry removed (Decision 37 + Amendment).
  li_velocity_24h = df_li_v24
)

# Load z-score lookup for natural-unit conversion
z_lookup <- read.csv(ZSCORE_FILE, stringsAsFactors = FALSE)
cat(sprintf("  z-score lookup: %d entries\n", nrow(z_lookup)))

cat(sprintf("  Models loaded: %d / %d\n", length(models), nrow(model_registry)))


# =============================================================================
# 2. EXTRACT SIGNIFICANCE FROM ALL MODELS
# =============================================================================

cat("\n== 2. Extracting GAM term significance from all 10 models ==\n")

terms_all <- tibble()
for (i in seq_len(nrow(model_registry))) {
  mid  <- model_registry$model_id[i]
  plat <- model_registry$platform[i]
  outc <- model_registry$outcome[i]
  if (!(mid %in% names(models))) next

  terms_i <- extract_gam_terms(models[[mid]]) |>
    mutate(model_id = mid, platform = plat, outcome = outc)
  terms_all <- bind_rows(terms_all, terms_i)
}

cat(sprintf("  Total terms extracted: %d (across %d models)\n",
            nrow(terms_all), n_distinct(terms_all$model_id)))
cat(sprintf("  Significant terms: %d\n", sum(terms_all$significant)))

# Summary per outcome
for (outc in c("inclusion", "rank", "velocity")) {
  sub <- terms_all |> filter(outcome == outc, significant)
  cat(sprintf("    %s: %d significant terms across %d models\n",
              outc, nrow(sub), n_distinct(sub$model_id)))
}


# =============================================================================
# 3. COMPUTE IQR MAGNITUDE
# =============================================================================

cat("\n== 3. Computing IQR magnitude for all significant terms ==\n")

# Build median profiles per model
profiles <- list()
for (mid in names(models)) {
  profiles[[mid]] <- make_median_profile(data_map[[mid]], models[[mid]])
}

# Compute magnitude for every term (not just significant, for completeness)
magnitudes <- tibble()
for (i in seq_len(nrow(terms_all))) {
  mid  <- terms_all$model_id[i]
  feat <- terms_all$feature[i]
  if (!(mid %in% names(models))) next

  mag <- compute_magnitude(feat, models[[mid]], data_map[[mid]], profiles[[mid]])
  magnitudes <- bind_rows(magnitudes, tibble(
    model_id = mid, feature = feat, magnitude = mag
  ))
}

terms_all <- terms_all |>
  left_join(magnitudes, by = c("model_id", "feature"))

cat(sprintf("  Magnitudes computed: %d / %d terms\n",
            sum(!is.na(terms_all$magnitude)), nrow(terms_all)))

# --- Convert magnitude to interpretable effect sizes per outcome family ---
# Baseline probabilities per platform (for inclusion log-odds -> pp conversion)
baseline_probs <- c(
  tiktok    = mean(df_tt$ever_top),
  instagram = mean(df_ig$ever_top),
  linkedin  = mean(df_li$ever_top)
)
cat(sprintf("  Baseline ever_top rates: TT=%.3f, IG=%.3f, LI=%.3f\n",
            baseline_probs["tiktok"], baseline_probs["instagram"], baseline_probs["linkedin"]))

terms_all <- terms_all |>
  mutate(
    baseline_prob = unname(baseline_probs[platform]),
    magnitude     = as.numeric(magnitude),
    effect_size = case_when(
      is.na(magnitude) ~ NA_real_,
      # Inclusion (binomial logit): log-odds -> pp change at platform baseline
      outcome == "inclusion" ~
        plogis(qlogis(baseline_prob) + magnitude) - baseline_prob,
      # Rank (Gaussian identity): magnitude IS rank-position change already
      outcome == "rank" ~ magnitude,
      # Velocity (Gamma log): magnitude -> proportional change
      outcome == "velocity" ~ exp(magnitude) - 1
    ),
    effect_label = case_when(
      is.na(effect_size) ~ NA_character_,
      outcome == "inclusion" ~ sprintf("%+.1f pp", effect_size * 100),
      outcome == "rank"      ~ sprintf("%+.1f positions", effect_size),
      outcome == "velocity"  ~ sprintf("%+.0f%%", effect_size * 100)
    )
  )

cat(sprintf("  Effect sizes computed: %d / %d terms\n",
            sum(!is.na(terms_all$effect_size)), nrow(terms_all)))


# =============================================================================
# 4. EXTRACT OPTIMAL RANGES
# =============================================================================

cat("\n== 4. Extracting optimal ranges from partial-effect curves ==\n")

terms_all$optimal_range <- NA_character_

for (i in seq_len(nrow(terms_all))) {
  if (!terms_all$significant[i]) {
    terms_all$optimal_range[i] <- "n.s."
    next
  }

  mid  <- terms_all$model_id[i]
  feat <- terms_all$feature[i]
  plat <- terms_all$platform[i]
  ftype <- terms_all$feature_type[i]
  good_dir <- model_registry$good_direction[model_registry$model_id == mid]

  if (!(mid %in% names(models))) {
    terms_all$optimal_range[i] <- "Model not loaded"
    next
  }

  if (ftype == "smooth") {
    result <- extract_optimal_smooth(models[[mid]], feat, plat, good_dir, z_lookup)
    terms_all$optimal_range[i] <- result$range_text
  } else {
    # Parametric (binary or factor)
    result <- extract_optimal_parametric(models[[mid]], feat, data_map[[mid]], good_dir)
    terms_all$optimal_range[i] <- result
  }
}

n_sig <- sum(terms_all$significant)
n_range <- sum(terms_all$significant & terms_all$optimal_range != "n.s." &
                 terms_all$optimal_range != "n/a")
cat(sprintf("  Optimal ranges extracted: %d / %d significant terms\n", n_range, n_sig))


# =============================================================================
# 5. CROSS-PLATFORM GENERALISATION
# =============================================================================

cat("\n== 5. Checking cross-platform generalisation ==\n")

# Count how many platforms a feature is significant on, per outcome type
gen_check <- terms_all |>
  filter(significant) |>
  group_by(feature, outcome) |>
  summarise(
    n_sig_platforms = n_distinct(platform),
    platforms       = paste(sort(unique(platform)), collapse = ", "),
    directions      = paste(unique(direction), collapse = " / "),
    direction_agree = length(unique(direction)) == 1,
    .groups = "drop"
  ) |>
  mutate(generalises = n_sig_platforms >= 2)

terms_all <- terms_all |>
  left_join(
    gen_check |> select(feature, outcome, generalises, n_sig_platforms),
    by = c("feature", "outcome")
  ) |>
  mutate(
    generalises     = replace_na(generalises, FALSE),
    n_sig_platforms = replace_na(n_sig_platforms, 0L)
  )

n_gen <- gen_check |> filter(generalises) |> nrow()
cat(sprintf("  Feature-outcome combinations generalising (2+ platforms): %d\n", n_gen))
if (n_gen > 0) {
  gen_rows <- gen_check |> filter(generalises)
  for (j in seq_len(nrow(gen_rows))) {
    cat(sprintf("    %s (%s): %s [%s]\n",
                gen_rows$feature[j], gen_rows$outcome[j],
                gen_rows$platforms[j],
                ifelse(gen_rows$direction_agree[j], "AGREE", "DIVERGE")))
  }
} else {
  cat("    None -- platform-specific effects dominate (key RQ2 finding).\n")
}


# =============================================================================
# 6. GENERATE RECOMMENDATIONS
# =============================================================================

cat("\n== 6. Generating plain-language recommendations ==\n")

terms_all$controllable <- CONTROLLABLE[terms_all$feature]
terms_all$controllable[is.na(terms_all$controllable)] <- "unknown"

# Add model-level caveats
terms_all$caveat <- MODEL_CAVEATS[terms_all$model_id]
terms_all$caveat[is.na(terms_all$caveat)] <- ""

terms_all$recommendation <- NA_character_
for (i in seq_len(nrow(terms_all))) {
  if (!terms_all$significant[i]) {
    terms_all$recommendation[i] <- "Not significant."
    next
  }
  terms_all$recommendation[i] <- make_recommendation(
    feature       = terms_all$feature[i],
    controllable  = terms_all$controllable[i],
    direction     = terms_all$direction[i],
    optimal_range = terms_all$optimal_range[i],
    outcome       = terms_all$outcome[i],
    platform      = terms_all$platform[i],
    magnitude     = terms_all$magnitude[i],
    effect_label  = terms_all$effect_label[i],
    caveat        = terms_all$caveat[i]
  )
}

n_actionable <- sum(terms_all$significant &
                      terms_all$controllable %in% c("yes", "timing", "indirect"))
cat(sprintf("  Actionable recommendations (controllable + significant): %d\n",
            n_actionable))


# =============================================================================
# 7. BUILD AND SAVE CSVs
# =============================================================================

cat("\n== 7. Building strategy matrix CSVs ==\n")

# p-value stars
terms_all <- terms_all |>
  mutate(
    p_stars = case_when(
      p_value < 0.001 ~ "***",
      p_value < 0.01  ~ "**",
      p_value < 0.05  ~ "*",
      p_value < 0.10  ~ ".",
      TRUE            ~ ""
    )
  )

# --- Long format (all 261 terms with effect sizes) ---
strategy_long <- terms_all |>
  select(feature, controllable, platform, outcome, model_id,
         significant, direction, magnitude, effect_size, effect_label,
         edf, p_value, p_stars,
         optimal_range, recommendation, generalises, n_sig_platforms, caveat) |>
  mutate(magnitude = round(magnitude, 3),
         effect_size = round(effect_size, 4)) |>
  arrange(controllable, feature, outcome, platform)

write.csv(strategy_long, file.path(OUT_DIR, "strategy_matrix.csv"),
          row.names = FALSE)
cat(sprintf("  Saved: strategy_matrix.csv (%d rows, %d features)\n",
            nrow(strategy_long), n_distinct(strategy_long$feature)))

# --- Wide format (one per outcome, ALL features for three-tier view) ---
for (outc in c("inclusion", "rank", "velocity")) {
  wide <- terms_all |>
    filter(outcome == outc) |>
    # Velocity: drop T72 (TikTok-only secondary) to avoid duplicate pivots
    filter(!grepl("72h", model_id)) |>
    mutate(
      cell = case_when(
        !significant ~ "n.s.",
        TRUE ~ paste0(
          ifelse(controllable %in% c("yes", "timing"), optimal_range, direction),
          " (", p_stars, ") [", effect_label, "]"
        )
      )
    ) |>
    select(feature, controllable, platform, cell) |>
    pivot_wider(names_from = platform, values_from = cell, values_fill = "---") |>
    arrange(controllable, feature)

  fname <- paste0("strategy_matrix_wide_", outc, ".csv")
  write.csv(wide, file.path(OUT_DIR, fname), row.names = FALSE)
  cat(sprintf("  Saved: %s (%d features)\n", fname, nrow(wide)))
}


# =============================================================================
# 8. THESIS FIGURES
# =============================================================================

cat("\n== 8. Generating thesis figures ==\n")

# --- Figure 1: Strategy heatmap (significance x platform x outcome) ---

# Prepare heatmap data: only controllable features (yes/timing/indirect)
heat_data <- terms_all |>
  filter(controllable %in% c("yes", "timing", "indirect")) |>
  mutate(
    # Collapse velocity variants for display
    outcome_display = case_when(
      outcome == "inclusion" ~ "Inclusion",
      outcome == "rank"      ~ "Rank",
      outcome == "velocity"  ~ "Velocity"
    ),
    platform_display = PLATFORM_LABELS[platform],
    facet_label = paste(platform_display, outcome_display, sep = "\n"),
    # Encode significance + direction
    fill_val = case_when(
      !significant                                          ~  0,   # not significant
      direction %in% c("positive", "approx. linear") &
        outcome != "rank"                                   ~  1,   # good for inclusion/velocity
      direction %in% c("negative", "approx. linear") &
        outcome == "rank"                                   ~  1,   # good for rank
      direction %in% c("positive") & outcome == "rank"      ~ -1,   # bad for rank
      direction %in% c("negative") & outcome != "rank"      ~ -1,   # bad for inclusion/velocity
      direction == "non-linear"                             ~  0.5, # non-linear (direction ambiguous)
      TRUE                                                  ~  0
    ),
    feature_label = ifelse(feature %in% names(FEATURE_LABELS),
                           FEATURE_LABELS[feature], feature)
  )

# For deduplication: keep one row per feature-platform-outcome (best model)
heat_dedup <- heat_data |>
  group_by(feature, feature_label, controllable, platform, platform_display,
           outcome_display) |>
  summarise(
    fill_val    = first(fill_val[order(p_value)]),
    significant = any(significant),
    p_stars     = first(p_stars[order(p_value)]),
    .groups     = "drop"
  )

# Order features by controllability group, then alphabetically
feat_order <- heat_dedup |>
  distinct(feature, feature_label, controllable) |>
  mutate(ctrl_rank = match(controllable, c("yes", "timing", "indirect"))) |>
  arrange(ctrl_rank, feature_label) |>
  pull(feature_label)

heat_dedup$feature_label <- factor(heat_dedup$feature_label, levels = rev(feat_order))

# Create column ordering
col_order <- c("TikTok\nInclusion", "TikTok\nRank", "TikTok\nVelocity",
               "Instagram\nInclusion", "Instagram\nRank", "Instagram\nVelocity",
               "LinkedIn\nInclusion", "LinkedIn\nRank", "LinkedIn\nVelocity")
heat_dedup$facet_label <- paste(heat_dedup$platform_display,
                                heat_dedup$outcome_display, sep = "\n")
heat_dedup$facet_label <- factor(heat_dedup$facet_label, levels = col_order)

p_heat <- ggplot(heat_dedup, aes(x = facet_label, y = feature_label,
                                  fill = fill_val)) +
  geom_tile(colour = "white", linewidth = 0.5) +
  geom_text(aes(label = ifelse(significant, p_stars, "")),
            size = 3, colour = "white", fontface = "bold") +
  scale_fill_gradient2(
    low = "#D32F2F", mid = "grey90", high = "#1565C0",
    midpoint = 0, limits = c(-1, 1),
    name = "Direction",
    labels = c("Unfavourable", "", "Neutral / n.s.", "", "Favourable"),
    breaks = c(-1, -0.5, 0, 0.5, 1)
  ) +
  labs(x = NULL, y = NULL,
       title = "Strategy Matrix: Feature Significance Across Platforms and Outcomes",
       subtitle = "Stars indicate significance level. Blue = favourable, red = unfavourable, grey = not significant.") +
  theme_thesis() +
  theme(
    axis.text.x = element_text(size = 8, lineheight = 1.1),
    axis.text.y = element_text(size = 8),
    legend.position = "bottom",
    legend.key.width = unit(2, "cm"),
    panel.grid = element_blank()
  )

save_plot(p_heat, "strategy_heatmap.png",
          width = 10, height = max(6, length(feat_order) * 0.35 + 2))
cat("  Saved: strategy_heatmap.png\n")


# --- Figure 2: Optimal range panel (partial-effect curves for key features) ---

# Select significant controllable smooth features for the panel
panel_features <- terms_all |>
  filter(significant,
         controllable %in% c("yes", "timing"),
         feature_type == "smooth",
         outcome == "inclusion") |>  # focus on inclusion for clearest story
  distinct(feature, platform, model_id) |>
  arrange(feature, platform)

if (nrow(panel_features) > 0) {
  cat(sprintf("  Generating optimal range plots for %d feature-platform combos\n",
              nrow(panel_features)))

  plot_list <- list()
  for (j in seq_len(nrow(panel_features))) {
    feat <- panel_features$feature[j]
    plat <- panel_features$platform[j]
    mid  <- panel_features$model_id[j]
    model <- models[[mid]]

    # Find smooth label
    sm_names <- rownames(summary(model)$s.table)
    label <- sm_names[grep(paste0("\\b", feat, "\\b"), sm_names)]
    if (length(label) == 0) next
    label <- label[1]

    se <- tryCatch(smooth_estimates(model, select = label, n = 200),
                   error = function(e) NULL)
    if (is.null(se)) next

    # Get the x column -- gratia uses the feature name directly
    x_col <- feat
    data_cols <- setdiff(names(se), c(".smooth", ".type", ".by", ".estimate",
                                       ".se", ".lower_ci", ".upper_ci"))
    if (!(x_col %in% names(se))) {
      # Try matching without exact name (gratia may rename)
      match_cols <- data_cols[grepl(feat, data_cols, fixed = TRUE)]
      if (length(match_cols) >= 1) x_col <- match_cols[1]
      else if (length(data_cols) == 1) x_col <- data_cols[1]
      else next
    }

    # Back-transform x to natural units
    if (feat %in% CYCLIC_FEATURES) {
      se$x_nat <- se[[x_col]]  # already natural (0-6 weekday, 0-23 hour)
    } else {
      z_row <- z_lookup |> filter(feature == feat, platform == plat)
      if (nrow(z_row) > 0) {
        se$x_nat <- se[[x_col]] * z_row$sd + z_row$mean
        if (feat %in% LOG_FEATURES) {
          se$x_nat <- exp(se$x_nat) - 1
          se$x_nat <- pmax(0, se$x_nat)  # floor at 0
        }
      } else {
        se$x_nat <- se[[x_col]]
      }
    }

    # Remove rows with invalid x_nat (Inf, NaN)
    se <- se |> filter(is.finite(x_nat))

    # Get optimal range for shading
    good_dir <- model_registry$good_direction[model_registry$model_id == mid]
    opt <- extract_optimal_smooth(model, feat, plat, good_dir, z_lookup)

    feat_label <- ifelse(feat %in% names(FEATURE_LABELS),
                         FEATURE_LABELS[feat], feat)
    plat_col <- PLATFORM_COLOURS[plat]

    # Ensure CI columns exist (gratia versions vary)
    if (!(".se" %in% names(se)) && ".lower_ci" %in% names(se)) {
      se$.se <- (se$.upper_ci - se$.lower_ci) / (2 * 1.96)
    }
    if (!(".se" %in% names(se))) {
      se$.se <- 0  # no CI available
    }
    se$ci_lo <- se$.estimate - 1.96 * se$.se
    se$ci_hi <- se$.estimate + 1.96 * se$.se

    if (nrow(se) == 0) next

    p_i <- ggplot(se, aes(x = x_nat, y = .estimate)) +
      geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50") +
      geom_ribbon(aes(ymin = ci_lo, ymax = ci_hi),
                  alpha = 0.15, fill = plat_col) +
      geom_line(colour = plat_col, linewidth = 0.8) +
      labs(x = feat_label,
           y = "Partial effect (link scale)",
           title = paste0(PLATFORM_LABELS[plat], ": ", feat_label)) +
      theme_thesis() +
      theme(plot.title = element_text(size = 9, face = "bold"))

    # Add optimal range shading if available
    if (!is.na(opt$nat_lo) && !is.na(opt$nat_hi)) {
      p_i <- p_i +
        annotate("rect", xmin = opt$nat_lo, xmax = opt$nat_hi,
                 ymin = -Inf, ymax = Inf, alpha = 0.1, fill = "#4CAF50")
    }

    plot_list[[paste0(feat, "_", plat)]] <- p_i
  }

  if (length(plot_list) > 0) {
    n_plots <- length(plot_list)
    ncols <- min(3, n_plots)
    nrows <- ceiling(n_plots / ncols)

    p_panel <- wrap_plots(plot_list, ncol = ncols) +
      plot_annotation(
        title = "Optimal Ranges for Significant Creator-Controllable Features",
        subtitle = "Green bands highlight the favourable range. Dashed line = zero effect.",
        theme = theme_thesis()
      )

    save_plot(p_panel, "optimal_range_panel.png",
              width = min(14, ncols * 4.5), height = nrows * 3.5)
    cat("  Saved: optimal_range_panel.png\n")
  }
} else {
  cat("  No significant controllable smooth features for inclusion -- skipping panel.\n")
}


# =============================================================================
# 9. CONSOLE SUMMARY
# =============================================================================

cat("\n\n================================================================\n")
cat("== STRATEGY MATRIX SUMMARY (Step 9) ==\n")
cat("================================================================\n\n")

# Headline RQ3 stats
cat("--- RQ3: Actionable Strategy Findings ---\n\n")

sig_terms <- terms_all |> filter(significant)
sig_ctrl  <- sig_terms |> filter(controllable %in% c("yes", "timing", "indirect"))

cat(sprintf("Total significant terms across all 10 models: %d\n", nrow(sig_terms)))
cat(sprintf("  Creator-controllable (yes/timing/indirect): %d\n", nrow(sig_ctrl)))
cat(sprintf("  Not controllable (no): %d\n",
            nrow(sig_terms |> filter(controllable == "no"))))

cat("\n--- Significant controllable features by platform and outcome ---\n")
sig_summary <- sig_ctrl |>
  group_by(platform, outcome) |>
  summarise(
    n     = n(),
    feats = paste(unique(feature), collapse = ", "),
    .groups = "drop"
  ) |>
  arrange(platform, outcome)

for (k in seq_len(nrow(sig_summary))) {
  cat(sprintf("  %-10s %-10s (%d): %s\n",
              PLATFORM_LABELS[sig_summary$platform[k]],
              sig_summary$outcome[k],
              sig_summary$n[k],
              sig_summary$feats[k]))
}

cat("\n--- Cross-platform generalisation ---\n")
if (n_gen > 0) {
  gen_rows <- gen_check |> filter(generalises)
  for (j in seq_len(nrow(gen_rows))) {
    ctrl <- CONTROLLABLE[gen_rows$feature[j]]
    cat(sprintf("  %s (%s): %s -- %s [controllable: %s]\n",
                gen_rows$feature[j], gen_rows$outcome[j],
                gen_rows$platforms[j],
                ifelse(gen_rows$direction_agree[j], "AGREE", "DIVERGE"),
                ifelse(is.na(ctrl), "unknown", ctrl)))
  }
} else {
  cat("  NO creator-controllable feature generalises across platforms.\n")
  cat("  Each platform responds to different content signals.\n")
  cat("  This is the central RQ2/RQ3 finding.\n")
}

cat("\n--- Top actionable recommendations (inclusion models) ---\n")
top_recs <- terms_all |>
  filter(significant, outcome == "inclusion",
         controllable %in% c("yes", "timing")) |>
  arrange(p_value) |>
  head(10)

for (k in seq_len(nrow(top_recs))) {
  cat(sprintf("  %d. [%s] %s\n     %s\n",
              k,
              PLATFORM_LABELS[top_recs$platform[k]],
              top_recs$feature[k],
              top_recs$recommendation[k]))
}

cat("\n== Step 9 complete ==\n")
cat(sprintf("Output: %s/strategy_matrix.csv (%d rows)\n", OUT_DIR, nrow(strategy_long)))
cat(sprintf("Output: %s/strategy_matrix_wide_{inclusion,rank,velocity}.csv\n", OUT_DIR))
cat(sprintf("Output: %s/strategy_heatmap.png\n", OUT_DIR))
cat(sprintf("Output: %s/optimal_range_panel.png\n", OUT_DIR))
