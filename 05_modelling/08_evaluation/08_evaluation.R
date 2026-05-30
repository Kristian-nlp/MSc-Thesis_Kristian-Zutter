# =============================================================================
# 08_evaluation.R
#
# Master's thesis: Learning the Levers — Which Creator-Controllable
# Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
# from a Swiss German-Language Perspective
# Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026
#
# Purpose:
#   Evaluates all GAM models and the three ranger cross-checks via 5-fold
#   cross-validation (primary), a temporal-split sensitivity check on
#   TikTok and Instagram, and ranger out-of-bag metrics. Produces the
#   consolidated metrics file, ROC curves, and the model comparison table.
#
# Pipeline position:
#   Step 8 of the modelling pipeline. Depends on Steps 5–7; feeds the
#   model-performance tables and ROC figure in Step 10.
#
# Inputs:
#   05_modelling/03_data_prep/data/df_tt.parquet                   TikTok analytical frame
#   05_modelling/03_data_prep/data/df_ig.parquet                   Instagram analytical frame
#   05_modelling/03_data_prep/data/df_li.parquet                   LinkedIn analytical frame
#   05_modelling/05_gam/models/m_*.rds                             All fitted GAMs
#   05_modelling/06_ranger/models/rf_*_impurity.rds                Ranger forests
#   05_modelling/06_ranger/output/oob_summary.csv                  Ranger OOB summary
#
# Outputs:
#   05_modelling/08_evaluation/output/model_metrics.csv            Long-format metrics
#   05_modelling/08_evaluation/output/model_comparison_table.csv   Wide comparison table
#   05_modelling/08_evaluation/output/roc_curves.png               ROC figure
#   05_modelling/08_evaluation/output/calibration_curves.png       Calibration figure
#
# Usage:
#   Rscript 05_modelling/08_evaluation/08_evaluation.R
# =============================================================================

rm(list = ls())

# -- Source shared config ------------------------------------------------------
source(file.path("config", "packages.R"))
library(patchwork)

STEP_DIR   <- file.path(BASE_DIR, "08_evaluation")
OUT_DIR    <- file.path(STEP_DIR, "output")
PLOT_DIR   <- OUT_DIR   # save_plot() looks for this variable
MODEL_DIR  <- file.path(BASE_DIR, "05_gam", "models")
RANGER_DIR <- file.path(BASE_DIR, "06_ranger")
PREP_DIR   <- file.path(BASE_DIR, "03_data_prep", "data")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

cat("================================================================\n")
cat("== Step 8: Model Evaluation (5-fold CV + temporal split)      ==\n")
cat("================================================================\n")
cat(sprintf("  Date: %s\n", Sys.Date()))
t0_script <- Sys.time()


# =============================================================================
# 1. HELPER FUNCTIONS
# =============================================================================

# -- 1a. Gamma shift (identical to 05c_gam_velocity.R) -----------------------
shift_for_gamma <- function(x) {
  min_val <- min(x, na.rm = TRUE)
  if (min_val <= 0) {
    shift <- abs(min_val) + 0.001
    return(list(shifted = x + shift, shift = shift))
  } else {
    return(list(shifted = x, shift = 0))
  }
}

# -- 1b. Binary classification metrics ---------------------------------------
compute_metrics_binary <- function(truth, prob) {
  # truth: numeric 0/1, prob: predicted probability of class 1
  df <- tibble(
    truth    = factor(truth, levels = c("0", "1")),
    estimate = prob
  )
  auc_roc <- tryCatch(
    yardstick::roc_auc(df, truth, estimate, event_level = "second")$.estimate,
    error = function(e) NA_real_
  )
  pr_auc <- tryCatch(
    yardstick::pr_auc(df, truth, estimate, event_level = "second")$.estimate,
    error = function(e) NA_real_
  )
  brier <- mean((prob - truth)^2, na.rm = TRUE)

  tibble(
    metric = c("AUC-ROC", "PR-AUC", "Brier"),
    value  = c(auc_roc, pr_auc, brier)
  )
}

# -- 1c. Rank metrics (with bootstrap CI for Spearman rho) ------------------
compute_metrics_rank <- function(truth, pred) {
  if (length(truth) < 3 || length(pred) < 3) {
    return(tibble(metric = c("Spearman rho", "rho_ci_lo", "rho_ci_hi",
                              "rho_p", "MAE", "RMSE"),
                  value  = rep(NA_real_, 6)))
  }
  # Point estimate + p-value from cor.test
  ct <- tryCatch(
    cor.test(truth, pred, method = "spearman", exact = FALSE),
    error = function(e) NULL
  )
  spearman <- if (!is.null(ct)) unname(ct$estimate) else NA_real_
  rho_p    <- if (!is.null(ct)) ct$p.value else NA_real_

  # Bootstrap 95% CI (1000 resamples)
  set.seed(42)
  boot_rhos <- replicate(1000, {
    idx <- sample(length(truth), replace = TRUE)
    tryCatch(cor(truth[idx], pred[idx], method = "spearman", use = "complete.obs"),
             error = function(e) NA_real_)
  })
  ci <- quantile(boot_rhos, c(0.025, 0.975), na.rm = TRUE)

  mae  <- mean(abs(truth - pred), na.rm = TRUE)
  rmse <- sqrt(mean((truth - pred)^2, na.rm = TRUE))

  tibble(
    metric = c("Spearman rho", "rho_ci_lo", "rho_ci_hi", "rho_p", "MAE", "RMSE"),
    value  = c(spearman, ci[1], ci[2], rho_p, mae, rmse)
  )
}

# -- 1d. Velocity metrics ----------------------------------------------------
compute_metrics_velocity <- function(truth, pred) {
  if (length(truth) < 3 || length(pred) < 3) {
    return(tibble(metric = c("RMSE", "MAE", "R-squared"),
                  value  = c(NA_real_, NA_real_, NA_real_)))
  }
  rmse <- sqrt(mean((truth - pred)^2, na.rm = TRUE))
  mae  <- mean(abs(truth - pred), na.rm = TRUE)
  ss_res <- sum((truth - pred)^2, na.rm = TRUE)
  ss_tot <- sum((truth - mean(truth, na.rm = TRUE))^2, na.rm = TRUE)
  r_sq <- ifelse(ss_tot > 0, 1 - ss_res / ss_tot, NA_real_)

  tibble(
    metric = c("RMSE", "MAE", "R-squared"),
    value  = c(rmse, mae, r_sq)
  )
}

# -- 1e. Stratified fold indices (for binary outcomes) ------------------------
create_stratified_folds <- function(y, k = 5, seed = 42) {
  set.seed(seed)
  idx_0 <- which(y == 0)
  idx_1 <- which(y == 1)
  folds_0 <- split(sample(idx_0), rep(1:k, length.out = length(idx_0)))
  folds_1 <- split(sample(idx_1), rep(1:k, length.out = length(idx_1)))
  # Each fold's TEST indices
  lapply(1:k, function(i) sort(c(folds_0[[i]], folds_1[[i]])))
}

# -- 1f. Simple random fold indices ------------------------------------------
create_folds <- function(n, k = 5, seed = 42) {
  set.seed(seed)
  split(sample(1:n), rep(1:k, length.out = n))
}

# -- 1g. Adjust formula for CV fold: downgrade smooths with too few unique ----
# mgcv::gam() requires k <= n_unique per smooth term. In CV folds (80% of
# data), features with few unique values (e.g., z-standardized line_break_count
# with 4-5 unique values) may drop below the original k. Solution: use the
# fitted model's smooth specs to detect problematic terms and downgrade
# them from s(x, k=N) to x (linear parametric). This is statistically
# appropriate — with fewer unique values than k, a smooth is overparameterised.
adjust_formula_for_fold <- function(model, data) {
  fml <- formula(model)
  fml_str <- paste(deparse(fml, width.cutoff = 500), collapse = " ")

  for (si in model$smooth) {
    varname <- si$term[1]       # variable name from smooth spec
    k_val   <- si$bs.dim        # k value used in fitting

    if (!varname %in% names(data)) next
    n_uniq <- length(unique(data[[varname]][!is.na(data[[varname]])]))

    if (n_uniq < k_val) {
      # Replace s(varname, ...) with just varname (linear parametric)
      pattern <- sprintf("s\\(%s[^)]*\\)", varname)
      if (grepl(pattern, fml_str)) {
        fml_str <- sub(pattern, varname, fml_str)
        cat(sprintf("    k-fix: s(%s, k=%d) -> %s [linear] (n_unique=%d)\n",
                    varname, k_val, varname, n_uniq))
      }
    }
  }
  as.formula(fml_str)
}

# -- 1h. Safe GAM refit with error handling -----------------------------------
safe_gam_cv <- function(formula, data, family, method = "REML",
                        knots = NULL, fold_label = "") {
  tryCatch(
    gam(formula = formula, data = data, family = family,
        method = method, knots = knots),
    error = function(e) {
      cat(sprintf("    WARNING: GAM refit failed [%s]: %s\n",
                  fold_label, conditionMessage(e)))
      NULL
    },
    warning = function(w) {
      cat(sprintf("    NOTE: GAM warning [%s]: %s\n",
                  fold_label, conditionMessage(w)))
      # Return the model despite warnings (step failures etc.)
      suppressWarnings(
        gam(formula = formula, data = data, family = family,
            method = method, knots = knots)
      )
    }
  )
}


# =============================================================================
# 2. LOAD DATA & ENSURE FACTORS
# =============================================================================
cat("\n== 1. Loading platform data frames ==\n")

df_tt <- read_parquet(file.path(PREP_DIR, "df_tt.parquet"))
df_ig <- read_parquet(file.path(PREP_DIR, "df_ig.parquet"))
df_li <- read_parquet(file.path(PREP_DIR, "df_li.parquet"))

cat(sprintf("  df_tt: %d rows x %d cols\n", nrow(df_tt), ncol(df_tt)))
cat(sprintf("  df_ig: %d rows x %d cols\n", nrow(df_ig), ncol(df_ig)))
cat(sprintf("  df_li: %d rows x %d cols\n", nrow(df_li), ncol(df_li)))

# Ensure factors (same pattern as 05a)
factor_vars <- c("lang", "media_type", "topic_cluster", "account_type")
for (v in factor_vars) {
  if (v %in% names(df_tt)) df_tt[[v]] <- as.factor(df_tt[[v]])
  if (v %in% names(df_ig)) df_ig[[v]] <- as.factor(df_ig[[v]])
  if (v %in% names(df_li)) df_li[[v]] <- as.factor(df_li[[v]])
}

data_list <- list(tt = df_tt, ig = df_ig, li = df_li)


# =============================================================================
# 3. MODEL REGISTRY
# =============================================================================
cat("\n== 2. Building model registry ==\n")

# Knot definitions (same as 05a/05b/05c)
knots_tt_li <- list(local_hour = c(0, 24), weekday = c(0, 7))

registry <- tibble::tribble(
  ~model_id,          ~platform, ~platform_label, ~outcome,        ~outcome_var,    ~model_file,              ~metric_type, ~subset_expr,               ~needs_gamma_shift, ~response_col,    ~knots,      ~temporal_eligible,
  "tt_inclusion",     "tt",      "TikTok",        "inclusion",     "ever_top",      "m_tt_inclusion.rds",     "binary",     NA_character_,              FALSE,              "ever_top",       list(knots_tt_li), TRUE,
  "ig_inclusion",     "ig",      "Instagram",     "inclusion",     "ever_top",      "m_ig_inclusion.rds",     "binary",     NA_character_,              FALSE,              "ever_top",       list(NULL),        TRUE,
  "li_inclusion",     "li",      "LinkedIn",      "inclusion",     "ever_top",      "m_li_inclusion.rds",     "binary",     NA_character_,              FALSE,              "ever_top",       list(knots_tt_li), FALSE,
  "tt_rank",          "tt",      "TikTok",        "rank",          "best_rank",     "m_tt_rank.rds",          "rank",       "ever_top == 1",            FALSE,              "best_rank",      list(knots_tt_li), TRUE,
  "ig_rank",          "ig",      "Instagram",     "rank",          "best_rank",     "m_ig_rank.rds",          "rank",       "ever_top == 1",            FALSE,              "best_rank",      list(NULL),        TRUE,
  "li_rank",          "li",      "LinkedIn",      "rank",          "best_rank",     "m_li_rank.rds",          "rank",       "ever_top == 1",            FALSE,              "best_rank",      list(knots_tt_li), FALSE,
  "tt_velocity_24h",  "tt",      "TikTok",        "velocity_24h",  "velocity_24h",  "m_tt_velocity_24h.rds",  "velocity",   "!is.na(velocity_24h)",     TRUE,               "velocity_24h_g", list(knots_tt_li), TRUE,
  "tt_velocity_72h",  "tt",      "TikTok",        "velocity_72h",  "velocity_72h",  "m_tt_velocity_72h.rds",  "velocity",   "!is.na(velocity_72h)",     TRUE,               "velocity_72h_g", list(knots_tt_li), TRUE,
  # ig_velocity_24h removed: Decision 37 (2026-04-17) + Amendment. The
  # Instagram T24/T72 permalink revisit scraper writes the wrong field into
  # counters.likes and counters.comments; no Instagram velocity model is
  # fitted (see 05c_gam_velocity.R) and no CV is run for it here.
  "li_velocity_24h",  "li",      "LinkedIn",      "velocity_24h",  "velocity_24h",  "m_li_velocity_24h.rds",  "velocity",   "!is.na(velocity_24h)",     TRUE,               "velocity_24h_g", list(knots_tt_li), FALSE
)

cat(sprintf("  Registry: %d models\n", nrow(registry)))
cat(sprintf("  Temporal eligible: %d models\n", sum(registry$temporal_eligible)))


# =============================================================================
# 4. FIVE-FOLD CROSS-VALIDATION ENGINE
# =============================================================================
cat("\n\n================================================================\n")
cat("== 3. Five-fold cross-validation ==\n")
cat("================================================================\n")

K <- 5
all_cv_metrics   <- list()
all_cv_preds     <- list()   # for ROC curves (inclusion models only)

for (i in seq_len(nrow(registry))) {
  reg <- registry[i, ]
  cat(sprintf("\n  [%d/%d] CV for %s (%s, %s) ...\n",
              i, nrow(registry), reg$model_id, reg$platform_label, reg$outcome))
  t0_model <- Sys.time()

  # -- Load fitted model to extract formula and family -----------------------
  model <- readRDS(file.path(MODEL_DIR, reg$model_file))
  fml   <- formula(model)
  fam   <- model$family

  # -- Get platform data and apply subset ------------------------------------
  df <- data_list[[reg$platform]]
  if (!is.na(reg$subset_expr)) {
    df <- df %>% filter(eval(parse(text = reg$subset_expr)))
  }
  n_total <- nrow(df)
  cat(sprintf("    N = %d\n", n_total))

  # -- Extract knots ---------------------------------------------------------
  knots <- reg$knots[[1]]

  # -- Create fold indices ---------------------------------------------------
  if (reg$metric_type == "binary") {
    fold_test_idx <- create_stratified_folds(df[[reg$outcome_var]], k = K)
  } else {
    fold_test_idx <- create_folds(n_total, k = K)
  }

  # -- CV loop ---------------------------------------------------------------
  fold_metrics <- list()
  fold_preds   <- list()

  for (f in 1:K) {
    test_idx  <- fold_test_idx[[f]]
    train_idx <- setdiff(1:n_total, test_idx)
    df_train  <- df[train_idx, ]
    df_test   <- df[test_idx, ]

    # -- Gamma shift for velocity (computed on training fold) ----------------
    gamma_shift <- 0
    if (reg$needs_gamma_shift) {
      resp_col <- reg$outcome_var   # e.g., "velocity_24h"
      g_col    <- reg$response_col  # e.g., "velocity_24h_g"
      gs <- shift_for_gamma(df_train[[resp_col]])
      gamma_shift <- gs$shift
      df_train[[g_col]] <- gs$shifted
      # Apply SAME shift to test fold, clip to ensure positivity
      df_test[[g_col]]  <- pmax(df_test[[resp_col]] + gamma_shift, 0.0001)
    }

    # -- Align factor levels (test = train levels) --------------------------
    for (v in factor_vars) {
      if (v %in% names(df_train) && is.factor(df_train[[v]])) {
        levels(df_test[[v]]) <- levels(df_train[[v]])
      }
    }

    # -- Adjust formula: downgrade smooths with too few unique values --------
    fml_fold <- adjust_formula_for_fold(model, df_train)

    # -- Refit GAM on training fold -----------------------------------------
    fold_label <- sprintf("%s fold %d", reg$model_id, f)
    m_fold <- safe_gam_cv(fml_fold, df_train, fam, method = "REML",
                          knots = knots, fold_label = fold_label)

    if (is.null(m_fold)) {
      fold_metrics[[f]] <- tibble(metric = character(), value = numeric())
      next
    }

    # -- Predict on test fold -----------------------------------------------
    preds <- tryCatch(
      predict(m_fold, newdata = df_test, type = "response"),
      error = function(e) {
        cat(sprintf("    WARNING: predict failed [%s]: %s\n",
                    fold_label, conditionMessage(e)))
        rep(NA_real_, nrow(df_test))
      }
    )

    # Remove NA predictions (from unseen factor levels)
    valid <- !is.na(preds)
    if (sum(!valid) > 0) {
      cat(sprintf("    NOTE: %d/%d NA predictions in %s (unseen factor levels)\n",
                  sum(!valid), length(preds), fold_label))
    }

    # Skip fold if too few valid predictions
    if (sum(valid) < 10) {
      cat(sprintf("    SKIP fold %d: only %d valid predictions\n", f, sum(valid)))
      fold_metrics[[f]] <- tibble(metric = character(), value = numeric())
      next
    }

    # -- Compute metrics based on outcome type ------------------------------
    if (reg$metric_type == "binary") {
      truth_vec <- df_test[[reg$outcome_var]][valid]
      pred_vec  <- preds[valid]
      fold_metrics[[f]] <- compute_metrics_binary(truth_vec, pred_vec)

      # Store predictions for ROC curve
      fold_preds[[f]] <- tibble(
        truth    = truth_vec,
        estimate = pred_vec,
        fold     = f
      )

    } else if (reg$metric_type == "rank") {
      truth_vec <- df_test[[reg$outcome_var]][valid]
      pred_vec  <- preds[valid]
      fold_metrics[[f]] <- compute_metrics_rank(truth_vec, pred_vec)

    } else if (reg$metric_type == "velocity") {
      # Back-transform: predictions are on shifted scale
      truth_vec <- df_test[[reg$outcome_var]][valid]
      pred_vec  <- preds[valid] - gamma_shift
      fold_metrics[[f]] <- compute_metrics_velocity(truth_vec, pred_vec)
    }
  }

  # -- Average metrics across folds ------------------------------------------
  valid_folds <- bind_rows(fold_metrics) %>%
    filter(!is.na(value))

  if (nrow(valid_folds) == 0) {
    cat(sprintf("    FAILED: All folds failed for %s\n", reg$model_id))
    all_cv_metrics[[i]] <- tibble(
      model_id = reg$model_id, platform = reg$platform_label,
      outcome = reg$outcome, model_type = "GAM", cv_type = "5fold_cv",
      metric = NA_character_, value = NA_real_, sd = NA_real_,
      n_train = NA_real_, n_test = NA_real_
    )
  } else {
    n_valid <- length(fold_metrics[sapply(fold_metrics, nrow) > 0])
    # NB: sd must be computed BEFORE value — dplyr summarise() evaluates
    # columns left-to-right, so later columns see earlier overwrites.
    cv_summary <- valid_folds %>%
      group_by(metric) %>%
      summarise(sd    = sd(value, na.rm = TRUE),
                value = mean(value, na.rm = TRUE),
                .groups = "drop") %>%
      mutate(
        model_id  = reg$model_id,
        platform  = reg$platform_label,
        outcome   = reg$outcome,
        model_type = "GAM",
        cv_type   = "5fold_cv",
        n_train   = round(n_total * (K - 1) / K),
        n_test    = round(n_total / K)
      )
    all_cv_metrics[[i]] <- cv_summary

    # Print summary
    for (r in seq_len(nrow(cv_summary))) {
      cat(sprintf("    %s: %.4f (sd=%.4f)\n",
                  cv_summary$metric[r], cv_summary$value[r], cv_summary$sd[r]))
    }
    cat(sprintf("    Valid folds: %d/%d\n", n_valid, K))
  }

  # Store predictions for inclusion models (ROC curves)
  if (reg$metric_type == "binary" && length(fold_preds) > 0) {
    all_cv_preds[[reg$model_id]] <- bind_rows(fold_preds) %>%
      mutate(model_id = reg$model_id, platform = reg$platform_label)
  }

  t1_model <- Sys.time()
  cat(sprintf("    Time: %.1f min\n",
              as.numeric(t1_model - t0_model, units = "mins")))
}

cv_metrics <- bind_rows(all_cv_metrics)
cat(sprintf("\n  CV complete: %d metric rows\n", nrow(cv_metrics)))


# =============================================================================
# 5. TEMPORAL SPLIT (TikTok + Instagram only)
# =============================================================================
cat("\n\n================================================================\n")
cat("== 4. Temporal split sensitivity check ==\n")
cat("================================================================\n")
cat("  (TikTok + Instagram only; LinkedIn excluded — 4.5-day window)\n")

all_temporal_metrics <- list()
temporal_idx <- 0

for (i in seq_len(nrow(registry))) {
  reg <- registry[i, ]
  if (!reg$temporal_eligible) next

  temporal_idx <- temporal_idx + 1
  cat(sprintf("\n  [%d] Temporal split for %s ...\n",
              temporal_idx, reg$model_id))

  # -- Load model to extract formula and family ------------------------------
  model <- readRDS(file.path(MODEL_DIR, reg$model_file))
  fml   <- formula(model)
  fam   <- model$family
  knots <- reg$knots[[1]]

  # -- Get platform data and apply subset ------------------------------------
  df <- data_list[[reg$platform]]

  # Parse posted_at_utc to POSIXct
  df$posted_at_posix <- as.POSIXct(df$posted_at_utc, format = "%Y-%m-%dT%H:%M:%S",
                                    tz = "UTC")

  # Exclude rows with NA timestamps
  n_before <- nrow(df)
  df_temp <- df %>% filter(!is.na(posted_at_posix))
  n_after <- nrow(df_temp)
  n_dropped <- n_before - n_after
  if (n_dropped > 0) {
    cat(sprintf("    Dropped %d rows with NA posted_at_utc (%d -> %d)\n",
                n_dropped, n_before, n_after))
  }

  # Apply model subset (rank or velocity)
  if (!is.na(reg$subset_expr)) {
    df_temp <- df_temp %>% filter(eval(parse(text = reg$subset_expr)))
  }

  n_total <- nrow(df_temp)
  if (n_total < 50) {
    cat(sprintf("    SKIP: Only %d observations after filtering\n", n_total))
    next
  }

  # -- Sort by timestamp and split at 75% -----------------------------------
  df_temp <- df_temp %>% arrange(posted_at_posix)
  cutoff <- floor(n_total * 0.75)
  df_train <- df_temp[1:cutoff, ]
  df_test  <- df_temp[(cutoff + 1):n_total, ]
  cat(sprintf("    Train: %d, Test: %d\n", nrow(df_train), nrow(df_test)))

  # -- Gamma shift for velocity (on training data) --------------------------
  gamma_shift <- 0
  if (reg$needs_gamma_shift) {
    resp_col <- reg$outcome_var
    g_col    <- reg$response_col
    gs <- shift_for_gamma(df_train[[resp_col]])
    gamma_shift <- gs$shift
    df_train[[g_col]] <- gs$shifted
    df_test[[g_col]]  <- pmax(df_test[[resp_col]] + gamma_shift, 0.0001)
  }

  # -- Align factor levels ---------------------------------------------------
  for (v in factor_vars) {
    if (v %in% names(df_train) && is.factor(df_train[[v]])) {
      levels(df_test[[v]]) <- levels(df_train[[v]])
    }
  }

  # -- Adjust formula for training period --------------------------------------
  fml_temp <- adjust_formula_for_fold(model, df_train)

  # -- Refit on training period ----------------------------------------------
  fold_label <- sprintf("%s temporal", reg$model_id)
  m_temp <- safe_gam_cv(fml_temp, df_train, fam, method = "REML",
                        knots = knots, fold_label = fold_label)

  if (is.null(m_temp)) {
    cat(sprintf("    FAILED: Temporal refit failed for %s\n", reg$model_id))
    next
  }

  # -- Predict on test period ------------------------------------------------
  preds <- tryCatch(
    predict(m_temp, newdata = df_test, type = "response"),
    error = function(e) {
      cat(sprintf("    WARNING: predict failed [%s]: %s\n",
                  fold_label, conditionMessage(e)))
      rep(NA_real_, nrow(df_test))
    }
  )
  valid <- !is.na(preds)

  # -- Compute metrics -------------------------------------------------------
  if (reg$metric_type == "binary") {
    truth_vec <- df_test[[reg$outcome_var]][valid]
    pred_vec  <- preds[valid]
    metrics_temp <- compute_metrics_binary(truth_vec, pred_vec)
  } else if (reg$metric_type == "rank") {
    truth_vec <- df_test[[reg$outcome_var]][valid]
    pred_vec  <- preds[valid]
    metrics_temp <- compute_metrics_rank(truth_vec, pred_vec)
  } else if (reg$metric_type == "velocity") {
    truth_vec <- df_test[[reg$outcome_var]][valid]
    pred_vec  <- preds[valid] - gamma_shift
    metrics_temp <- compute_metrics_velocity(truth_vec, pred_vec)
  }

  all_temporal_metrics[[temporal_idx]] <- metrics_temp %>%
    mutate(
      model_id   = reg$model_id,
      platform   = reg$platform_label,
      outcome    = reg$outcome,
      model_type = "GAM",
      cv_type    = "temporal_split",
      sd         = NA_real_,
      n_train    = nrow(df_train),
      n_test     = nrow(df_test)
    )

  for (r in seq_len(nrow(metrics_temp))) {
    cat(sprintf("    %s: %.4f\n", metrics_temp$metric[r], metrics_temp$value[r]))
  }
}

temporal_metrics <- bind_rows(all_temporal_metrics)
cat(sprintf("\n  Temporal split complete: %d metric rows\n", nrow(temporal_metrics)))


# =============================================================================
# 6. RANGER OOB METRICS
# =============================================================================
cat("\n\n================================================================\n")
cat("== 5. Ranger OOB metrics ==\n")
cat("================================================================\n")

oob_summary <- read_csv(file.path(RANGER_DIR, "output", "oob_summary.csv"),
                        show_col_types = FALSE)
cat("  Loaded oob_summary.csv from Step 6\n")

ranger_metrics_list <- list()

# For each platform, load the ranger model and compute additional OOB metrics
for (p in c("tt", "ig", "li")) {
  plat_label <- c(tt = "TikTok", ig = "Instagram", li = "LinkedIn")[[p]]
  rf_file <- file.path(RANGER_DIR, "models", sprintf("rf_%s_impurity.rds", p))

  if (!file.exists(rf_file)) {
    cat(sprintf("  WARNING: %s not found, skipping\n", basename(rf_file)))
    next
  }

  cat(sprintf("  Loading %s ...\n", basename(rf_file)))
  rf <- readRDS(rf_file)

  # OOB predictions (probability of class "1")
  oob_probs <- rf$predictions[, "1"]

  # Get truth vector from complete cases matching ranger's training data
  df_rf <- data_list[[p]]
  feats <- rf$forest$independent.variable.names
  cc_idx <- complete.cases(df_rf[, feats])
  truth_vec <- df_rf$ever_top[cc_idx]

  stopifnot(length(truth_vec) == length(oob_probs))

  oob_metrics <- compute_metrics_binary(truth_vec, oob_probs)

  # Add OOB AUC from the pre-computed summary (more reliable)
  oob_row <- oob_summary %>% filter(platform == plat_label)
  if (nrow(oob_row) > 0) {
    oob_metrics$value[oob_metrics$metric == "AUC-ROC"] <- oob_row$oob_auc
  }

  ranger_metrics_list[[p]] <- oob_metrics %>%
    mutate(
      model_id   = sprintf("rf_%s", p),
      platform   = plat_label,
      outcome    = "inclusion",
      model_type = "ranger",
      cv_type    = "oob",
      sd         = NA_real_,
      n_train    = oob_row$n_fitted,
      n_test     = oob_row$n_fitted  # OOB uses all data
    )

  for (r in seq_len(nrow(oob_metrics))) {
    cat(sprintf("    %s %s: %.4f\n", plat_label,
                oob_metrics$metric[r], oob_metrics$value[r]))
  }

  rm(rf)  # free memory
}

ranger_metrics <- bind_rows(ranger_metrics_list)
cat(sprintf("\n  Ranger metrics complete: %d metric rows\n", nrow(ranger_metrics)))


# =============================================================================
# 7. CONSOLIDATE AND WRITE model_metrics.csv
# =============================================================================
cat("\n\n================================================================\n")
cat("== 6. Consolidating all metrics ==\n")
cat("================================================================\n")

# Standardise column order
cols_out <- c("model_id", "platform", "outcome", "model_type", "cv_type",
              "metric", "value", "sd", "n_train", "n_test")

model_metrics <- bind_rows(
  cv_metrics       %>% select(any_of(cols_out)),
  temporal_metrics %>% select(any_of(cols_out)),
  ranger_metrics   %>% select(any_of(cols_out))
)

write_csv(model_metrics, file.path(OUT_DIR, "model_metrics.csv"))
cat(sprintf("  Written: model_metrics.csv (%d rows)\n", nrow(model_metrics)))

# Summary counts
cat("\n  Rows by cv_type:\n")
model_metrics %>% count(cv_type) %>%
  mutate(line = sprintf("    %s: %d", cv_type, n)) %>%
  pull(line) %>% cat(sep = "\n")
cat("\n  Rows by model_type:\n")
model_metrics %>% count(model_type) %>%
  mutate(line = sprintf("    %s: %d", model_type, n)) %>%
  pull(line) %>% cat(sep = "\n")


# =============================================================================
# 8. ROC CURVES FIGURE
# =============================================================================
cat("\n\n================================================================\n")
cat("== 7. ROC curves (inclusion models) ==\n")
cat("================================================================\n")

# -- GAM out-of-fold ROC curves ----------------------------------------------
gam_roc_data <- list()
for (mid in names(all_cv_preds)) {
  pred_df <- all_cv_preds[[mid]]
  roc_pts <- yardstick::roc_curve(
    tibble(truth = factor(pred_df$truth, levels = c("0", "1")),
           estimate = pred_df$estimate),
    truth, estimate, event_level = "second"
  )
  gam_roc_data[[mid]] <- roc_pts %>%
    mutate(model_id = mid, platform = pred_df$platform[1],
           model_type = "GAM")
}
gam_roc <- bind_rows(gam_roc_data)

# -- Ranger OOB ROC curves ---------------------------------------------------
ranger_roc_data <- list()
for (p in c("tt", "ig", "li")) {
  plat_label <- c(tt = "TikTok", ig = "Instagram", li = "LinkedIn")[[p]]
  rf_file <- file.path(RANGER_DIR, "models", sprintf("rf_%s_impurity.rds", p))
  if (!file.exists(rf_file)) next

  rf <- readRDS(rf_file)
  oob_probs <- rf$predictions[, "1"]

  df_rf <- data_list[[p]]
  feats <- rf$forest$independent.variable.names
  cc_idx <- complete.cases(df_rf[, feats])
  truth_vec <- df_rf$ever_top[cc_idx]

  roc_pts <- yardstick::roc_curve(
    tibble(truth = factor(truth_vec, levels = c("0", "1")),
           estimate = oob_probs),
    truth, estimate, event_level = "second"
  )
  ranger_roc_data[[p]] <- roc_pts %>%
    mutate(model_id = sprintf("rf_%s", p), platform = plat_label,
           model_type = "ranger")

  rm(rf)
}
ranger_roc <- bind_rows(ranger_roc_data)

# -- Combine and plot --------------------------------------------------------
all_roc <- bind_rows(gam_roc, ranger_roc) %>%
  mutate(
    platform   = factor(platform, levels = c("TikTok", "Instagram", "LinkedIn")),
    model_type = factor(model_type, levels = c("GAM", "ranger"))
  )

# Get AUC values for legend labels
auc_labels <- model_metrics %>%
  filter(metric == "AUC-ROC",
         cv_type %in% c("5fold_cv", "oob"),
         outcome == "inclusion") %>%
  mutate(label = sprintf("%s %s (AUC = %.3f)", platform, model_type, value))

# Build legend mapping
all_roc <- all_roc %>%
  left_join(
    auc_labels %>% select(model_id, label),
    by = "model_id"
  )

p_roc <- ggplot(all_roc, aes(x = 1 - specificity, y = sensitivity,
                              colour = platform, linetype = model_type)) +
  geom_path(linewidth = 0.8) +
  geom_abline(slope = 1, intercept = 0, linetype = "dotted", colour = "grey50") +
  scale_colour_manual(
    values = c("TikTok" = "#00f2ea", "Instagram" = "#E1306C", "LinkedIn" = "#0A66C2"),
    name   = "Platform"
  ) +
  scale_linetype_manual(
    values = c("GAM" = "solid", "ranger" = "dashed"),
    name   = "Model"
  ) +
  labs(x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)",
       title = "ROC Curves: Top-20 Inclusion Models") +
  coord_equal() +
  theme_thesis()

save_plot(p_roc, "roc_curves.png", width = 6.5, height = 5.5)
cat("  Saved: roc_curves.png\n")

# Also add AUC annotations to console
cat("\n  Inclusion AUC summary:\n")
auc_labels %>%
  arrange(platform, model_type) %>%
  mutate(line = sprintf("    %s", label)) %>%
  pull(line) %>% cat(sep = "\n")
cat("\n")


# =============================================================================
# 7b. CALIBRATION PLOTS (inclusion models)
# =============================================================================
cat("\n== 7b. Calibration curves (inclusion models) ==\n")

# Pool all GAM out-of-fold predictions into decile bins
cal_data <- tibble()
for (mid in names(all_cv_preds)) {
  preds_df <- all_cv_preds[[mid]]
  if (is.null(preds_df) || nrow(preds_df) == 0) next

  binned <- preds_df |>
    mutate(
      truth_num = as.numeric(as.character(truth)),
      bin = ntile(estimate, 10)
    ) |>
    group_by(bin) |>
    summarise(
      mean_pred = mean(estimate, na.rm = TRUE),
      mean_obs  = mean(truth_num, na.rm = TRUE),
      n         = n(),
      .groups   = "drop"
    ) |>
    mutate(model_id = mid, platform = preds_df$platform[1])

  cal_data <- bind_rows(cal_data, binned)
}

if (nrow(cal_data) > 0) {
  cal_data <- cal_data |>
    mutate(platform = factor(platform, levels = c("TikTok", "Instagram", "LinkedIn")))

  p_cal <- ggplot(cal_data, aes(x = mean_pred, y = mean_obs)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey50") +
    geom_point(aes(colour = platform, size = n), alpha = 0.8) +
    geom_line(aes(colour = platform), linewidth = 0.5) +
    scale_colour_manual(
      values = c("TikTok" = "#00f2ea", "Instagram" = "#E1306C", "LinkedIn" = "#0A66C2")
    ) +
    scale_size_continuous(range = c(2, 5), guide = "none") +
    facet_wrap(~ platform, scales = "free") +
    labs(
      x = "Mean predicted probability (decile bin)",
      y = "Observed proportion (ever_top = 1)",
      title = "Calibration curves — GAM inclusion models (5-fold CV)"
    ) +
    theme_thesis() +
    theme(legend.position = "none")

  save_plot(p_cal, "calibration_curves.png", width = 8, height = 3.5)
  cat("  Saved: calibration_curves.png\n")
} else {
  cat("  WARNING: No inclusion predictions available for calibration plots\n")
}


# =============================================================================
# 9. MODEL COMPARISON SUMMARY TABLE
# =============================================================================
cat("\n\n================================================================\n")
cat("== 8. Model comparison summary table ==\n")
cat("================================================================\n")

# Load deviance explained from fitted models
dev_expl_list <- list()
for (i in seq_len(nrow(registry))) {
  reg <- registry[i, ]
  model <- readRDS(file.path(MODEL_DIR, reg$model_file))
  dev_expl_list[[reg$model_id]] <- tibble(
    model_id = reg$model_id,
    dev_expl = round(summary(model)$dev.expl * 100, 1),
    aic      = round(AIC(model), 1)
  )
  rm(model)
}
dev_expl <- bind_rows(dev_expl_list)

# Pivot CV metrics to wide (one row per model)
primary_metrics <- c("AUC-ROC", "Spearman rho", "RMSE")

cv_wide <- cv_metrics %>%
  filter(metric %in% primary_metrics) %>%
  select(model_id, metric, value, sd) %>%
  pivot_wider(names_from = metric, values_from = c(value, sd),
              names_glue = "cv_{metric}_{.value}")

# Temporal split primary metrics
temporal_wide <- temporal_metrics %>%
  filter(metric %in% primary_metrics) %>%
  select(model_id, metric, value) %>%
  pivot_wider(names_from = metric, values_from = value,
              names_glue = "temp_{metric}")

# Ranger OOB AUC
ranger_wide <- ranger_metrics %>%
  filter(metric == "AUC-ROC") %>%
  transmute(platform,
            ranger_oob_auc = value)

# Assemble
comparison <- registry %>%
  select(model_id, platform = platform_label, outcome) %>%
  left_join(dev_expl, by = "model_id") %>%
  left_join(cv_wide, by = "model_id") %>%
  left_join(temporal_wide, by = "model_id") %>%
  left_join(ranger_wide, by = "platform")

# Clean column names for readability
comparison <- comparison %>%
  rename_with(~ gsub("value_|_value", "", .x)) %>%
  rename_with(~ gsub("`", "", .x))

write_csv(comparison, file.path(OUT_DIR, "model_comparison_table.csv"))
cat(sprintf("  Written: model_comparison_table.csv (%d rows)\n", nrow(comparison)))

# Print to console
cat("\n")
print(comparison %>% select(model_id, platform, outcome, dev_expl, everything()),
      n = Inf, width = 120)


# =============================================================================
# 10. Decision log
# =============================================================================
cat("\n\n================================================================\n")
cat("== 9. DECISION LOG ==\n")
cat("================================================================\n")

cat("\n--- 5-fold CV Results ---\n")
cv_metrics %>%
  filter(!is.na(value)) %>%
  arrange(platform, outcome, metric) %>%
  mutate(line = sprintf("  %s | %-14s | %-12s | %.4f (sd=%.4f)",
                        platform, outcome, metric, value,
                        ifelse(is.na(sd), 0, sd))) %>%
  pull(line) %>% cat(sep = "\n")

cat("\n\n--- Temporal Split Results ---\n")
if (nrow(temporal_metrics) > 0) {
  temporal_metrics %>%
    filter(!is.na(value)) %>%
    arrange(platform, outcome, metric) %>%
    mutate(line = sprintf("  %s | %-14s | %-12s | %.4f (train=%d, test=%d)",
                          platform, outcome, metric, value, n_train, n_test)) %>%
    pull(line) %>% cat(sep = "\n")
} else {
  cat("  No temporal split results available.\n")
}

cat("\n\n--- Ranger OOB Results ---\n")
ranger_metrics %>%
  filter(!is.na(value)) %>%
  arrange(platform, metric) %>%
  mutate(line = sprintf("  %s | %-8s | %.4f", platform, metric, value)) %>%
  pull(line) %>% cat(sep = "\n")

# -- CV vs Temporal divergence check -----------------------------------------
cat("\n\n--- CV vs Temporal Split Divergence (AUC-ROC) ---\n")
cv_auc <- cv_metrics %>%
  filter(metric == "AUC-ROC", !is.na(value)) %>%
  select(model_id, cv_auc = value)
temp_auc <- temporal_metrics %>%
  filter(metric == "AUC-ROC", !is.na(value)) %>%
  select(model_id, temp_auc = value)

divergence <- inner_join(cv_auc, temp_auc, by = "model_id") %>%
  mutate(diff = abs(cv_auc - temp_auc),
         flag = ifelse(diff > 0.05, "*** DIVERGENT ***", "OK"))

if (nrow(divergence) > 0) {
  divergence %>%
    mutate(line = sprintf("  %s: CV=%.3f, Temporal=%.3f, diff=%.3f %s",
                          model_id, cv_auc, temp_auc, diff, flag)) %>%
    pull(line) %>% cat(sep = "\n")
} else {
  cat("  No comparable AUC-ROC pairs available.\n")
}

cat("\n\n================================================================\n")
cat("== Step 8 complete ==\n")
cat("================================================================\n")
cat(sprintf("  Output directory: %s\n", OUT_DIR))
cat("  Files written:\n")
cat("    model_metrics.csv\n")
cat("    roc_curves.png\n")
cat("    model_comparison_table.csv\n")
cat(sprintf("  Total time: %.1f min\n",
            as.numeric(Sys.time() - t0_script, units = "mins")))
