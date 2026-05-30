# =============================================================================
# 05b_gam_rank.R
#
# Master's thesis: Learning the Levers — Which Creator-Controllable
# Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
# from a Swiss German-Language Perspective
# Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026
#
# Purpose:
#   Fits Gaussian GAMs predicting best_rank (1–20) among ever_top = 1 posts
#   only. Same feature sets as the inclusion models. LinkedIn is marginal
#   (N = 421); if convergence is unstable the report falls back to
#   descriptive statistics.
#
# Pipeline position:
#   Step 5b of the modelling pipeline. Depends on Steps 1–4; cross-checks
#   against 05a. Feeds 08_evaluation.R, 09_strategy_matrix.R, and the
#   thesis tables in Step 10.
#
# Inputs:
#   05_modelling/03_data_prep/data/df_tt.parquet           TikTok analytical frame
#   05_modelling/03_data_prep/data/df_ig.parquet           Instagram analytical frame
#   05_modelling/03_data_prep/data/df_li.parquet           LinkedIn analytical frame
#
# Outputs:
#   05_modelling/05_gam/models/m_{tt,ig,li}_rank.rds       Fitted Gaussian GAMs
#   05_modelling/05_gam/output/summary_*_rank.txt          Model summaries
#   05_modelling/05_gam/output/diag_*_rank.png             gam.check 2x2 diagnostics
#   05_modelling/05_gam/output/concurvity_*_rank.csv       Pairwise concurvity matrices
#   05_modelling/05_gam/output/kcheck_*_rank.txt           k-index tables
#   05_modelling/05_gam/output/smooth_*_rank.png           Partial-effect panels
#   05_modelling/05_gam/output/comparison_rank.csv         Cross-platform fit summary
#
# Usage:
#   Rscript 05_modelling/05_gam/05b_gam_rank.R
# =============================================================================

rm(list = ls())

# -- Source shared config ------------------------------------------------------
source(file.path("config", "packages.R"))
library(gratia)

STEP_DIR  <- file.path(BASE_DIR, "05_gam")
MODEL_DIR <- file.path(STEP_DIR, "models")
OUT_DIR   <- file.path(STEP_DIR, "output")
dir.create(MODEL_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(OUT_DIR,   recursive = TRUE, showWarnings = FALSE)


# =============================================================================
# 0. HELPER FUNCTIONS
# =============================================================================

run_diagnostics <- function(model, platform, outcome, df) {
  tag <- paste0(platform, "_", outcome)
  cat(sprintf("\n  --- %s: Diagnostics ---\n", toupper(platform)))

  n_used   <- nobs(model)
  n_data   <- nrow(df)
  dev_expl <- round(summary(model)$dev.expl * 100, 1)
  r_sq     <- round(summary(model)$r.sq, 3)
  aic_val  <- round(AIC(model), 1)

  cat(sprintf("  N fitted: %d / %d (%.1f%% used)\n", n_used, n_data,
              n_used / n_data * 100))
  cat(sprintf("  Deviance explained: %.1f%%\n", dev_expl))
  cat(sprintf("  Adjusted R-squared: %.3f\n", r_sq))
  cat(sprintf("  AIC: %.1f\n", aic_val))

  # -- Summary to text --------------------------------------------------------
  sum_file <- file.path(OUT_DIR, paste0("summary_", tag, ".txt"))
  writeLines(capture.output(summary(model)), sum_file)
  cat(sprintf("  Summary saved: %s\n", basename(sum_file)))

  # -- Significant smooth terms -----------------------------------------------
  sm <- summary(model)$s.table
  if (!is.null(sm) && nrow(sm) > 0) {
    sig_sm <- sm[sm[, "p-value"] < 0.05, , drop = FALSE]
    cat(sprintf("\n  Significant smooth terms (%d / %d, p < 0.05):\n",
                nrow(sig_sm), nrow(sm)))
    if (nrow(sig_sm) > 0) {
      for (i in seq_len(nrow(sig_sm))) {
        cat(sprintf("    %-30s  edf=%5.2f  F=%7.2f  p=%.2e\n",
                    rownames(sig_sm)[i], sig_sm[i, "edf"],
                    sig_sm[i, "F"], sig_sm[i, "p-value"]))
      }
    }
    nonsig <- sm[sm[, "p-value"] >= 0.05, , drop = FALSE]
    if (nrow(nonsig) > 0) {
      cat(sprintf("\n  Non-significant smooths (%d):\n", nrow(nonsig)))
      for (i in seq_len(nrow(nonsig))) {
        cat(sprintf("    %-30s  edf=%5.2f  p=%.3f\n",
                    rownames(nonsig)[i], nonsig[i, "edf"],
                    nonsig[i, "p-value"]))
      }
    }
  }

  # -- Significant parametric terms -------------------------------------------
  pt <- summary(model)$p.table
  if (!is.null(pt) && nrow(pt) > 0) {
    sig_pt <- pt[pt[, "Pr(>|t|)"] < 0.05, , drop = FALSE]
    cat(sprintf("\n  Significant parametric terms (%d / %d):\n",
                nrow(sig_pt), nrow(pt)))
    if (nrow(sig_pt) > 0) {
      for (i in seq_len(nrow(sig_pt))) {
        cat(sprintf("    %-35s  est=%7.3f  t=%6.2f  p=%.2e\n",
                    rownames(sig_pt)[i], sig_pt[i, "Estimate"],
                    sig_pt[i, "t value"], sig_pt[i, "Pr(>|t|)"]))
      }
    }
  }

  # -- k-check ----------------------------------------------------------------
  cat("\n  --- k-check ---\n")
  kc <- k.check(model)
  kc_file <- file.path(OUT_DIR, paste0("kcheck_", tag, ".txt"))
  writeLines(capture.output(print(kc)), kc_file)

  if (!is.null(kc)) {
    kc_df <- as.data.frame(kc)
    if ("k-index" %in% names(kc_df)) {
      low_k <- kc_df[!is.na(kc_df[["k-index"]]) & kc_df[["k-index"]] < 1, ,
                      drop = FALSE]
      if (nrow(low_k) > 0) {
        cat("  *** WARNING: k-index < 1 for:\n")
        for (i in seq_len(nrow(low_k))) {
          cat(sprintf("    %-30s  k-index=%.3f\n",
                      rownames(low_k)[i], low_k[i, "k-index"]))
        }
      } else {
        cat("  All k-index values >= 1. No basis dimension issues.\n")
      }
    }
  }
  cat(sprintf("  k-check saved: %s\n", basename(kc_file)))

  # -- Concurvity --------------------------------------------------------------
  cat("\n  --- Concurvity (pairwise) ---\n")
  cc <- concurvity(model, full = FALSE)
  cc_est <- as.data.frame(cc$estimate)
  cc_file <- file.path(OUT_DIR, paste0("concurvity_", tag, ".csv"))
  write.csv(cc_est, cc_file)

  cc_mat <- as.matrix(cc_est)
  diag(cc_mat) <- 0
  max_cc <- max(cc_mat, na.rm = TRUE)
  if (max_cc > 0.8) {
    idx <- which(cc_mat == max_cc, arr.ind = TRUE)[1, ]
    cat(sprintf("  *** WARNING: Max concurvity = %.3f between %s and %s\n",
                max_cc, rownames(cc_mat)[idx[1]], colnames(cc_mat)[idx[2]]))
  } else {
    cat(sprintf("  Max pairwise concurvity: %.3f (below 0.8 threshold)\n",
                max_cc))
  }
  cat(sprintf("  Concurvity saved: %s\n", basename(cc_file)))

  # -- gam.check diagnostic plots ----------------------------------------------
  diag_file <- file.path(OUT_DIR, paste0("diag_", tag, ".png"))
  png(diag_file, width = 10, height = 8, units = "in", res = 200)
  par(mfrow = c(2, 2))
  gam.check(model, rep = 200)
  title(main = paste0(toupper(platform), " - ", outcome, " diagnostics"),
        outer = TRUE, line = -1)
  dev.off()
  cat(sprintf("  Diagnostics saved: %s\n", basename(diag_file)))

  # -- Partial effect plots (gratia) -------------------------------------------
  smooth_file <- file.path(OUT_DIR, paste0("smooth_", tag, ".png"))
  n_smooths <- length(smooths(model))
  plot_height <- max(8, ceiling(n_smooths / 4) * 3)

  p <- draw(model, residuals = FALSE, rug = TRUE) &
    theme_minimal(base_size = 9)
  ggsave(smooth_file, p, width = 14, height = plot_height, dpi = 200,
         limitsize = FALSE)
  cat(sprintf("  Partial effects saved: %s (%d smooths)\n",
              basename(smooth_file), n_smooths))

  # -- Save model --------------------------------------------------------------
  rds_file <- file.path(MODEL_DIR, paste0("m_", tag, ".rds"))
  saveRDS(model, rds_file)
  cat(sprintf("  Model saved: %s (%.1f MB)\n", basename(rds_file),
              file.size(rds_file) / 1e6))

  cat(sprintf("\n  === %s %s complete ===\n", toupper(platform), outcome))

  invisible(list(
    dev_expl = dev_expl,
    r_sq     = r_sq,
    aic      = aic_val,
    n_used   = n_used
  ))
}


# =============================================================================
# 1. LOAD DATA — subset to ever_top == 1
# =============================================================================
cat("== 0. Loading platform data frames (top posts only) ==\n")

PREP_DIR <- file.path(BASE_DIR, "03_data_prep", "data")
df_tt_all <- read_parquet(file.path(PREP_DIR, "df_tt.parquet"))
df_ig_all <- read_parquet(file.path(PREP_DIR, "df_ig.parquet"))
df_li_all <- read_parquet(file.path(PREP_DIR, "df_li.parquet"))

# Subset to top posts only (ever_top == 1)
df_tt <- df_tt_all |> filter(ever_top == 1)
df_ig <- df_ig_all |> filter(ever_top == 1)
df_li <- df_li_all |> filter(ever_top == 1)

rm(df_tt_all, df_ig_all, df_li_all)

cat(sprintf("  df_tt (top): %d rows (expected ~3,677)\n", nrow(df_tt)))
cat(sprintf("  df_ig (top): %d rows (expected ~2,249)\n", nrow(df_ig)))
cat(sprintf("  df_li (top): %d rows (expected ~421)\n", nrow(df_li)))

# Verify best_rank distribution
cat("\n  best_rank summary:\n")
for (nm in c("df_tt", "df_ig", "df_li")) {
  d <- get(nm)
  cat(sprintf("    %s: mean=%.1f  median=%.0f  min=%d  max=%d  NAs=%d\n",
              nm, mean(d$best_rank, na.rm = TRUE),
              median(d$best_rank, na.rm = TRUE),
              min(d$best_rank, na.rm = TRUE),
              max(d$best_rank, na.rm = TRUE),
              sum(is.na(d$best_rank))))
}


# =============================================================================
# 2. ENSURE FACTORS
# =============================================================================
cat("\n== 1. Preparing factor variables ==\n")

factor_vars <- c("lang", "media_type", "topic_cluster", "account_type")
for (v in factor_vars) {
  if (v %in% names(df_tt)) df_tt[[v]] <- as.factor(df_tt[[v]])
  if (v %in% names(df_ig)) df_ig[[v]] <- as.factor(df_ig[[v]])
  if (v %in% names(df_li)) df_li[[v]] <- as.factor(df_li[[v]])
}

# Check for sparse factor levels in the top subset
for (nm in c("df_tt", "df_ig", "df_li")) {
  d <- get(nm)
  for (v in factor_vars) {
    if (v %in% names(d)) {
      tbl <- table(d[[v]])
      sparse <- tbl[tbl < 5]
      if (length(sparse) > 0) {
        cat(sprintf("  WARNING: %s$%s has sparse levels (<5 obs): %s\n",
                    nm, v, paste(names(sparse), sparse, sep = "=",
                                 collapse = ", ")))
      }
    }
  }
}


# =============================================================================
# 3. TIKTOK RANK GAM
# =============================================================================
cat("\n\n================================================================\n")
cat("== 2. TIKTOK — best_rank GAM (Gaussian, identity) ==\n")
cat("================================================================\n")

t0 <- Sys.time()
knots_tt <- list(local_hour = c(0, 24), weekday = c(0, 7))

m_tt_rank <- gam(
  best_rank ~
    # --- continuous smooths (17; k adjusted for subset unique counts) ---
    s(word_count, k = 8) + s(hashtag_count, k = 8) +
    s(emoji_count, k = 8) + s(avg_sentence_len, k = 8) +
    s(exclamation_density, k = 8) + s(question_density, k = 8) +
    s(ellipsis_count, k = 5) +             # 5 unique in top subset
    s(caps_ratio, k = 8) +
    s(caps_word_count, k = 8) +
    s(line_break_count, k = 5) +           # 5 unique in top subset
    # url_count: only 3 unique -> parametric below
    s(mention_count, k = 8) +
    s(punct_diversity, k = 8) + s(flesch_reading_ease, k = 8) +
    s(brightness, k = 8) + s(contrast, k = 8) +
    s(colourfulness, k = 8) + s(log_post_age, k = 8) +
    s(log_follower, k = 8) + s(log_ocr_text_len, k = 8) +
    # --- cyclic smooths (2) ---
    s(local_hour, bs = "cc", k = 12) +
    s(weekday, bs = "cc", k = 7) +
    # --- binary + low-unique parametric (9) ---
    url_count +                            # 3 unique -> linear parametric
    cta_flag + is_weekend + is_trending + audio_is_original +
    uses_named_audio + flesch_available + follower_available + face_flag +
    # --- factor parametric (3; media_type excluded) ---
    lang + topic_cluster + account_type,
  data    = df_tt,
  family  = gaussian(),
  method  = "REML",
  knots   = knots_tt
)

t1 <- Sys.time()
cat(sprintf("  Fitting time: %.1f seconds\n", as.numeric(t1 - t0, units = "secs")))

diag_tt <- run_diagnostics(m_tt_rank, "tt", "rank", df_tt)


# =============================================================================
# 4. INSTAGRAM RANK GAM
# =============================================================================
cat("\n\n================================================================\n")
cat("== 3. INSTAGRAM — best_rank GAM (Gaussian, identity) ==\n")
cat("================================================================\n")

t0 <- Sys.time()

m_ig_rank <- gam(
  best_rank ~
    # --- continuous smooths (17; k adjusted for subset unique counts) ---
    s(word_count, k = 8) + s(hashtag_count, k = 8) +
    s(emoji_count, k = 8) + s(avg_sentence_len, k = 8) +
    s(exclamation_density, k = 8) + s(question_density, k = 8) +
    s(ellipsis_count, k = 5) +             # 5 unique in top subset
    s(caps_ratio, k = 8) +
    s(caps_word_count, k = 8) +
    s(line_break_count, k = 8) +
    # url_count: 4 unique in top subset -> parametric below
    s(mention_count, k = 8) +
    s(punct_diversity, k = 8) + s(flesch_reading_ease, k = 8) +
    s(brightness, k = 8) + s(contrast, k = 8) +
    s(colourfulness, k = 8) +
    s(log_follower, k = 8) + s(log_ocr_text_len, k = 8) +
    # --- binary + low-unique parametric (5) ---
    url_count +                            # 4 unique in top subset -> parametric
    cta_flag + flesch_available + follower_available + face_flag +
    # --- factor parametric (4) ---
    lang + media_type + topic_cluster + account_type,
  data    = df_ig,
  family  = gaussian(),
  method  = "REML"
)

t1 <- Sys.time()
cat(sprintf("  Fitting time: %.1f seconds\n", as.numeric(t1 - t0, units = "secs")))

diag_ig <- run_diagnostics(m_ig_rank, "ig", "rank", df_ig)


# =============================================================================
# 5. LINKEDIN RANK GAM
# =============================================================================
cat("\n\n================================================================\n")
cat("== 4. LINKEDIN — best_rank GAM (Gaussian, identity) ==\n")
cat("================================================================\n")
cat("  Note: N=421 is marginal. Using k=5 for all smooths.\n")
cat("  Note: If convergence fails, reduce to top 8-10 features.\n")

t0 <- Sys.time()
knots_li <- list(local_hour = c(0, 24), weekday = c(0, 7))

m_li_rank <- gam(
  best_rank ~
    # --- continuous smooths (15; mention_count -> parametric, 3 unique) ---
    s(word_count, k = 5) + s(hashtag_count, k = 5) +
    s(emoji_count, k = 5) + s(avg_sentence_len, k = 5) +
    s(exclamation_density, k = 5) + s(question_density, k = 5) +
    s(ellipsis_count, k = 5) + s(caps_ratio, k = 5) +
    s(caps_word_count, k = 5) + s(line_break_count, k = 5) +
    # url_count: 4 unique in top subset -> parametric below
    # mention_count: only 3 unique -> parametric below
    s(punct_diversity, k = 5) + s(flesch_reading_ease, k = 5) +
    s(log_post_age, k = 5) + s(log_follower, k = 5) +
    # --- cyclic smooths (2) ---
    s(local_hour, bs = "cc", k = 8) +
    s(weekday, bs = "cc", k = 5) +
    # --- binary + low-unique parametric (6) ---
    url_count +                            # 4 unique in top subset -> parametric
    mention_count +                        # 3 unique -> linear parametric
    cta_flag + is_weekend + flesch_available + follower_available +
    # --- factor parametric (4) ---
    lang + media_type + topic_cluster + account_type,
  data    = df_li,
  family  = gaussian(),
  method  = "REML",
  knots   = knots_li
)

t1 <- Sys.time()
cat(sprintf("  Fitting time: %.1f seconds\n", as.numeric(t1 - t0, units = "secs")))

diag_li <- run_diagnostics(m_li_rank, "li", "rank", df_li)


# =============================================================================
# 6. CROSS-PLATFORM SUMMARY
# =============================================================================
cat("\n\n================================================================\n")
cat("== 5. CROSS-PLATFORM SUMMARY ==\n")
cat("================================================================\n")

summary_df <- data.frame(
  platform     = c("TikTok", "Instagram", "LinkedIn"),
  n_fitted     = c(diag_tt$n_used, diag_ig$n_used, diag_li$n_used),
  dev_expl_pct = c(diag_tt$dev_expl, diag_ig$dev_expl, diag_li$dev_expl),
  adj_r_sq     = c(diag_tt$r_sq, diag_ig$r_sq, diag_li$r_sq),
  aic          = c(diag_tt$aic, diag_ig$aic, diag_li$aic)
)
print(summary_df, row.names = FALSE)

write.csv(summary_df,
          file.path(OUT_DIR, "comparison_rank.csv"),
          row.names = FALSE)

cat("\n== Step 5b complete ==\n")


# =============================================================================
# 7. Decision log
# =============================================================================
cat("\n\n--- Step 5b: GAM Rank Models (05b_gam_rank.R) -----\n")
cat(sprintf("Date: %s\n", Sys.Date()))
cat("Outcome: best_rank (1-20, Gaussian GAM, ever_top=1 subset only)\n\n")

cat("RANK GAM RESULTS\n")
for (i in 1:3) {
  cat(sprintf("  %s: N=%d, deviance explained=%.1f%%, adj.R2=%.3f, AIC=%.1f\n",
              summary_df$platform[i], summary_df$n_fitted[i],
              summary_df$dev_expl_pct[i], summary_df$adj_r_sq[i],
              summary_df$aic[i]))
}
cat("\n[Decision 13: LinkedIn rank viability — converged? Stable smooths?]\n")
cat("[If LinkedIn rank GAM failed: report descriptively (median rank by quartile)]\n")
