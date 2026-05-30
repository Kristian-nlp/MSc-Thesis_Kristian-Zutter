# =============================================================================
# 05a_gam_inclusion.R
#
# Master's thesis: Learning the Levers — Which Creator-Controllable
# Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
# from a Swiss German-Language Perspective
# Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026
#
# Purpose:
#   Fits binomial GAMs predicting ever_top (binary: top-20 vs baseline) for
#   each platform. Produces model summaries, gam.check diagnostics,
#   concurvity checks, k-index checks, and gratia partial-effect plots.
#
# Pipeline position:
#   Step 5a of the modelling pipeline. Depends on Steps 1–4; feeds 05b,
#   05c, 05d, 06, 07a, 07b, 08, 09, and the thesis tables in Step 10.
#
# Inputs:
#   05_modelling/03_data_prep/data/df_tt.parquet           TikTok analytical frame
#   05_modelling/03_data_prep/data/df_ig.parquet           Instagram analytical frame
#   05_modelling/03_data_prep/data/df_li.parquet           LinkedIn analytical frame
#
# Outputs:
#   05_modelling/05_gam/models/m_{tt,ig,li}_inclusion.rds  Fitted binomial GAMs
#   05_modelling/05_gam/output/summary_*_inclusion.txt     Model summaries
#   05_modelling/05_gam/output/diag_*_inclusion.png        gam.check 2x2 diagnostics
#   05_modelling/05_gam/output/concurvity_*_inclusion.csv  Pairwise concurvity matrices
#   05_modelling/05_gam/output/kcheck_*_inclusion.txt      k-index tables
#   05_modelling/05_gam/output/smooth_*_inclusion.png      Partial-effect panels
#   05_modelling/05_gam/output/comparison_inclusion.csv    Cross-platform fit summary
#
# Usage:
#   Rscript 05_modelling/05_gam/05a_gam_inclusion.R
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
  # Compact label for filenames
  tag <- paste0(platform, "_", outcome)

  cat(sprintf("\n  --- %s: Diagnostics ---\n", toupper(platform)))

  # -- 0. Basic model info ---------------------------------------------------
  n_used <- nobs(model)
  n_data <- nrow(df)
  dev_expl <- round(summary(model)$dev.expl * 100, 1)
  aic_val  <- round(AIC(model), 1)

  cat(sprintf("  N fitted: %d / %d (%.1f%% used)\n", n_used, n_data,
              n_used / n_data * 100))
  cat(sprintf("  Deviance explained: %.1f%%\n", dev_expl))
  cat(sprintf("  AIC: %.1f\n", aic_val))

  # -- 1. Summary to text file -----------------------------------------------
  sum_file <- file.path(OUT_DIR, paste0("summary_", tag, ".txt"))
  writeLines(capture.output(summary(model)), sum_file)
  cat(sprintf("  Summary saved: %s\n", basename(sum_file)))

  # -- 2. Significant smooth terms -------------------------------------------
  sm <- summary(model)$s.table
  # Binomial uses "Chi.sq", Gaussian/Gamma use "F"
  stat_col <- if ("Chi.sq" %in% colnames(sm)) "Chi.sq" else "F"

  if (!is.null(sm) && nrow(sm) > 0) {
    sig_sm <- sm[sm[, "p-value"] < 0.05, , drop = FALSE]
    cat(sprintf("\n  Significant smooth terms (%d / %d, p < 0.05):\n",
                nrow(sig_sm), nrow(sm)))
    if (nrow(sig_sm) > 0) {
      for (i in seq_len(nrow(sig_sm))) {
        cat(sprintf("    %-30s  edf=%5.2f  %s=%7.2f  p=%.2e\n",
                    rownames(sig_sm)[i],
                    sig_sm[i, "edf"],
                    stat_col,
                    sig_sm[i, stat_col],
                    sig_sm[i, "p-value"]))
      }
    }
    # Non-significant smooths
    nonsig <- sm[sm[, "p-value"] >= 0.05, , drop = FALSE]
    if (nrow(nonsig) > 0) {
      cat(sprintf("\n  Non-significant smooths (%d):\n", nrow(nonsig)))
      for (i in seq_len(nrow(nonsig))) {
        cat(sprintf("    %-30s  edf=%5.2f  p=%.3f\n",
                    rownames(nonsig)[i],
                    nonsig[i, "edf"],
                    nonsig[i, "p-value"]))
      }
    }
  }

  # -- 3. Significant parametric terms ---------------------------------------
  pt <- summary(model)$p.table
  # Binomial uses "z value"/"Pr(>|z|)", Gaussian/Gamma use "t value"/"Pr(>|t|)"
  p_col <- if ("Pr(>|z|)" %in% colnames(pt)) "Pr(>|z|)" else "Pr(>|t|)"
  t_col <- if ("z value"  %in% colnames(pt)) "z value"  else "t value"
  t_lab <- if ("z value"  %in% colnames(pt)) "z"        else "t"

  if (!is.null(pt) && nrow(pt) > 0) {
    sig_pt <- pt[pt[, p_col] < 0.05, , drop = FALSE]
    cat(sprintf("\n  Significant parametric terms (%d / %d):\n",
                nrow(sig_pt), nrow(pt)))
    if (nrow(sig_pt) > 0) {
      for (i in seq_len(nrow(sig_pt))) {
        cat(sprintf("    %-35s  est=%7.3f  %s=%6.2f  p=%.2e\n",
                    rownames(sig_pt)[i],
                    sig_pt[i, "Estimate"],
                    t_lab,
                    sig_pt[i, t_col],
                    sig_pt[i, p_col]))
      }
    }
  }

  # -- 4. k-check -------------------------------------------------------------
  cat("\n  --- k-check ---\n")
  kc <- k.check(model)
  kc_file <- file.path(OUT_DIR, paste0("kcheck_", tag, ".txt"))
  writeLines(capture.output(print(kc)), kc_file)

  # Flag any k-index < 1
  if (!is.null(kc)) {
    kc_df <- as.data.frame(kc)
    if ("k-index" %in% names(kc_df)) {
      low_k <- kc_df[!is.na(kc_df[["k-index"]]) & kc_df[["k-index"]] < 1, ,
                      drop = FALSE]
      if (nrow(low_k) > 0) {
        cat("  *** WARNING: k-index < 1 for:\n")
        for (i in seq_len(nrow(low_k))) {
          cat(sprintf("    %-30s  k-index=%.3f  (consider increasing k)\n",
                      rownames(low_k)[i], low_k[i, "k-index"]))
        }
      } else {
        cat("  All k-index values >= 1. No basis dimension issues.\n")
      }
    }
  }
  cat(sprintf("  k-check saved: %s\n", basename(kc_file)))

  # -- 5. Concurvity ----------------------------------------------------------
  cat("\n  --- Concurvity (pairwise) ---\n")
  cc <- concurvity(model, full = FALSE)
  # cc is a list of matrices; extract the "estimate" component
  cc_est <- as.data.frame(cc$estimate)
  cc_file <- file.path(OUT_DIR, paste0("concurvity_", tag, ".csv"))
  write.csv(cc_est, cc_file)

  # Find worst pairwise concurvity (off-diagonal)
  cc_mat <- as.matrix(cc_est)
  diag(cc_mat) <- 0
  max_cc <- max(cc_mat, na.rm = TRUE)
  if (max_cc > 0.8) {
    # Find the pair
    idx <- which(cc_mat == max_cc, arr.ind = TRUE)[1, ]
    cat(sprintf("  *** WARNING: Max concurvity = %.3f between %s and %s\n",
                max_cc, rownames(cc_mat)[idx[1]], colnames(cc_mat)[idx[2]]))
  } else {
    cat(sprintf("  Max pairwise concurvity: %.3f (below 0.8 threshold)\n",
                max_cc))
  }
  cat(sprintf("  Concurvity saved: %s\n", basename(cc_file)))

  # -- 6. gam.check diagnostic plots -----------------------------------------
  diag_file <- file.path(OUT_DIR, paste0("diag_", tag, ".png"))
  png(diag_file, width = 10, height = 8, units = "in", res = 200)
  par(mfrow = c(2, 2))
  gam.check(model, rep = 200)
  title(main = paste0(toupper(platform), " — ", outcome, " diagnostics"),
        outer = TRUE, line = -1)
  dev.off()
  cat(sprintf("  Diagnostics saved: %s\n", basename(diag_file)))

  # -- 7. Partial effect plots (gratia) --------------------------------------
  smooth_file <- file.path(OUT_DIR, paste0("smooth_", tag, ".png"))
  n_smooths <- length(smooths(model))
  plot_height <- max(8, ceiling(n_smooths / 4) * 3)

  p <- draw(model, residuals = FALSE, rug = TRUE) &
    theme_minimal(base_size = 9)
  ggsave(smooth_file, p, width = 14, height = plot_height, dpi = 200,
         limitsize = FALSE)
  cat(sprintf("  Partial effects saved: %s (%d smooths, %.0f x %.0f in)\n",
              basename(smooth_file), n_smooths, 14, plot_height))

  # -- 8. Save model -----------------------------------------------------------
  rds_file <- file.path(MODEL_DIR, paste0("m_", tag, ".rds"))
  saveRDS(model, rds_file)
  cat(sprintf("  Model saved: %s (%.1f MB)\n", basename(rds_file),
              file.size(rds_file) / 1e6))

  cat(sprintf("\n  === %s %s complete ===\n", toupper(platform), outcome))

  invisible(list(
    dev_expl = dev_expl,
    aic      = aic_val,
    n_used   = n_used
  ))
}


# =============================================================================
# 1. LOAD DATA
# =============================================================================
cat("== 0. Loading platform data frames ==\n")

PREP_DIR <- file.path(BASE_DIR, "03_data_prep", "data")
df_tt <- read_parquet(file.path(PREP_DIR, "df_tt.parquet"))
df_ig <- read_parquet(file.path(PREP_DIR, "df_ig.parquet"))
df_li <- read_parquet(file.path(PREP_DIR, "df_li.parquet"))

cat(sprintf("  df_tt: %d rows x %d cols\n", nrow(df_tt), ncol(df_tt)))
cat(sprintf("  df_ig: %d rows x %d cols\n", nrow(df_ig), ncol(df_ig)))
cat(sprintf("  df_li: %d rows x %d cols\n", nrow(df_li), ncol(df_li)))

# Verify ever_top balance
cat("\n  ever_top balance:\n")
for (nm in c("df_tt", "df_ig", "df_li")) {
  d <- get(nm)
  n_top <- sum(d$ever_top == 1, na.rm = TRUE)
  pct   <- round(n_top / nrow(d) * 100, 1)
  cat(sprintf("    %s: %d / %d top (%.1f%%)\n", nm, n_top, nrow(d), pct))
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

# Verify media_type for TikTok (expected: constant "video" -> excluded)
cat(sprintf("  TikTok media_type levels: %s\n",
            paste(levels(df_tt$media_type), collapse = ", ")))
cat(sprintf("  Instagram media_type levels: %s\n",
            paste(levels(df_ig$media_type), collapse = ", ")))
cat(sprintf("  LinkedIn media_type levels: %s\n",
            paste(levels(df_li$media_type), collapse = ", ")))


# =============================================================================
# 3. TIKTOK INCLUSION GAM
# =============================================================================
cat("\n\n================================================================\n")
cat("== 2. TIKTOK — ever_top inclusion GAM (binomial, logit) ==\n")
cat("================================================================\n")

t0 <- Sys.time()

# Cyclic knots: local_hour wraps 0-24, weekday wraps 0-7
knots_tt <- list(local_hour = c(0, 24), weekday = c(0, 7))

m_tt_incl <- gam(
  ever_top ~
    # --- continuous smooths (17; k adjusted for low-unique features) ---
    s(word_count, k = 10) + s(hashtag_count, k = 10) +
    s(emoji_count, k = 10) + s(avg_sentence_len, k = 10) +
    s(exclamation_density, k = 10) + s(question_density, k = 10) +
    s(ellipsis_count, k = 7) +             # 7 unique values
    s(caps_ratio, k = 10) +
    s(caps_word_count, k = 10) +
    s(line_break_count, k = 5) +           # 5 unique values
    # url_count: only 3 unique -> parametric below
    s(mention_count, k = 10) +
    s(punct_diversity, k = 10) + s(flesch_reading_ease, k = 10) +
    s(brightness, k = 10) + s(contrast, k = 10) +
    s(colourfulness, k = 10) + s(log_post_age, k = 10) +
    s(log_follower, k = 10) + s(log_ocr_text_len, k = 10) +
    # --- cyclic smooths (2) ---
    s(local_hour, bs = "cc", k = 12) +
    s(weekday, bs = "cc", k = 7) +
    # --- binary + low-unique parametric (9) ---
    url_count +                            # 3 unique -> linear parametric
    cta_flag + is_weekend + is_trending + audio_is_original +
    uses_named_audio + flesch_available + follower_available + face_flag +
    # --- factor parametric (3; media_type excluded: constant "video") ---
    lang + topic_cluster + account_type,
  data    = df_tt,
  family  = binomial(link = "logit"),
  method  = "REML",
  knots   = knots_tt
)

t1 <- Sys.time()
cat(sprintf("  Fitting time: %.1f seconds\n", as.numeric(t1 - t0, units = "secs")))

diag_tt <- run_diagnostics(m_tt_incl, "tt", "inclusion", df_tt)


# =============================================================================
# 4. INSTAGRAM INCLUSION GAM
# =============================================================================
cat("\n\n================================================================\n")
cat("== 3. INSTAGRAM — ever_top inclusion GAM (binomial, logit) ==\n")
cat("================================================================\n")
cat("  Note: No temporal features (dropped due to collider bias).\n")
cat("  Note: No audio_is_original, uses_named_audio (TikTok-only).\n")
cat("  Note: No is_trending (constant 0, NZV-dropped).\n")

t0 <- Sys.time()

m_ig_incl <- gam(
  ever_top ~
    # --- continuous smooths (17; k adjusted for low-unique features) ---
    s(word_count, k = 10) + s(hashtag_count, k = 10) +
    s(emoji_count, k = 10) + s(avg_sentence_len, k = 10) +
    s(exclamation_density, k = 10) + s(question_density, k = 10) +
    s(ellipsis_count, k = 8) +             # 8 unique values
    s(caps_ratio, k = 10) +
    s(caps_word_count, k = 10) +
    s(line_break_count, k = 10) +
    s(url_count, k = 5) +                  # 5 unique values
    s(mention_count, k = 10) +
    s(punct_diversity, k = 10) + s(flesch_reading_ease, k = 10) +
    s(brightness, k = 10) + s(contrast, k = 10) +
    s(colourfulness, k = 10) +
    s(log_follower, k = 10) + s(log_ocr_text_len, k = 10) +
    # --- binary parametric (4) ---
    cta_flag + flesch_available + follower_available + face_flag +
    # --- factor parametric (4) ---
    lang + media_type + topic_cluster + account_type,
  data    = df_ig,
  family  = binomial(link = "logit"),
  method  = "REML"
)

t1 <- Sys.time()
cat(sprintf("  Fitting time: %.1f seconds\n", as.numeric(t1 - t0, units = "secs")))

diag_ig <- run_diagnostics(m_ig_incl, "ig", "inclusion", df_ig)


# =============================================================================
# 5. LINKEDIN INCLUSION GAM
# =============================================================================
cat("\n\n================================================================\n")
cat("== 4. LINKEDIN — ever_top inclusion GAM (binomial, logit) ==\n")
cat("================================================================\n")
cat("  Note: No visual features (100% null, design constraint).\n")
cat("  Note: No audio_is_original, uses_named_audio (TikTok-only).\n")
cat("  Note: No is_trending (constant 0, NZV-dropped).\n")
cat("  Note: Reduced k=5 for smooths (N=1,170).\n")

t0 <- Sys.time()

knots_li <- list(local_hour = c(0, 24), weekday = c(0, 7))

m_li_incl <- gam(
  ever_top ~
    # --- continuous smooths (15; mention_count -> parametric, 3 unique) ---
    s(word_count, k = 5) + s(hashtag_count, k = 5) +
    s(emoji_count, k = 5) + s(avg_sentence_len, k = 5) +
    s(exclamation_density, k = 5) + s(question_density, k = 5) +
    s(ellipsis_count, k = 5) + s(caps_ratio, k = 5) +
    s(caps_word_count, k = 5) + s(line_break_count, k = 5) +
    s(url_count, k = 5) +
    # mention_count: only 3 unique -> parametric below
    s(punct_diversity, k = 5) + s(flesch_reading_ease, k = 5) +
    s(log_post_age, k = 5) + s(log_follower, k = 5) +
    # --- cyclic smooths (2) ---
    s(local_hour, bs = "cc", k = 8) +
    s(weekday, bs = "cc", k = 5) +
    # --- binary + low-unique parametric (5) ---
    mention_count +                        # 3 unique -> linear parametric
    cta_flag + is_weekend + flesch_available + follower_available +
    # --- factor parametric (4) ---
    lang + media_type + topic_cluster + account_type,
  data    = df_li,
  family  = binomial(link = "logit"),
  method  = "REML",
  knots   = knots_li
)

t1 <- Sys.time()
cat(sprintf("  Fitting time: %.1f seconds\n", as.numeric(t1 - t0, units = "secs")))

diag_li <- run_diagnostics(m_li_incl, "li", "inclusion", df_li)


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
  aic          = c(diag_tt$aic, diag_ig$aic, diag_li$aic)
)
print(summary_df, row.names = FALSE)

# Save summary comparison
write.csv(summary_df,
          file.path(OUT_DIR, "comparison_inclusion.csv"),
          row.names = FALSE)

cat("\n== Step 5a complete ==\n")


# =============================================================================
# 7. Decision log
# =============================================================================
cat("\n\n--- Step 5a: GAM Inclusion Models (05a_gam_inclusion.R) -----\n")
cat(sprintf("Date: %s\n", Sys.Date()))
cat(sprintf("Output: 05_gam/models/m_{tt,ig,li}_inclusion.rds\n"))
cat(sprintf("        05_gam/output/summary_*_inclusion.txt\n"))
cat(sprintf("        05_gam/output/smooth_*_inclusion.png\n"))
cat(sprintf("        05_gam/output/diag_*_inclusion.png\n\n"))

cat("INCLUSION GAM RESULTS (ever_top ~ features, binomial logit, REML)\n")
for (i in 1:3) {
  cat(sprintf("  %s: N=%d, deviance explained=%.1f%%, AIC=%.1f\n",
              summary_df$platform[i], summary_df$n_fitted[i],
              summary_df$dev_expl_pct[i], summary_df$aic[i]))
}
cat("\n[Decision 12: Model iteration — document any k increases or term drops]\n")
