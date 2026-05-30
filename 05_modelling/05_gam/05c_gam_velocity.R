# =============================================================================
# 05c_gam_velocity.R
#
# Master's thesis: Learning the Levers — Which Creator-Controllable
# Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
# from a Swiss German-Language Perspective
# Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026
#
# Purpose:
#   Fits Gamma GAMs (log link) predicting engagement velocity at T24 (and
#   T72 for TikTok). Velocity is defined as the log-growth in combined
#   engagement (likes + comments + shares) between T0 and the later time
#   point, per Decision 38. Instagram velocity is not modelled (Decision 37
#   + Amendment).
#
# Pipeline position:
#   Step 5c of the modelling pipeline. Depends on Steps 1–4; cross-checks
#   against 05a. Feeds 08_evaluation.R, 09_strategy_matrix.R, and the
#   thesis tables in Step 10.
#
# Inputs:
#   05_modelling/03_data_prep/data/df_tt.parquet                 TikTok analytical frame
#   05_modelling/03_data_prep/data/df_ig.parquet                 Instagram analytical frame
#   05_modelling/03_data_prep/data/df_li.parquet                 LinkedIn analytical frame
#
# Outputs:
#   05_modelling/05_gam/models/m_tt_velocity_24h.rds             TikTok T24 Gamma GAM
#   05_modelling/05_gam/models/m_tt_velocity_72h.rds             TikTok T72 Gamma GAM
#   05_modelling/05_gam/models/m_li_velocity_24h.rds             LinkedIn T24 Gamma GAM
#   05_modelling/05_gam/output/summary_{tt,li}_velocity_*.txt    Model summaries
#   05_modelling/05_gam/output/diag_{tt,li}_velocity_*.png       gam.check diagnostics
#   05_modelling/05_gam/output/concurvity_{tt,li}_velocity_*.csv Pairwise concurvity
#   05_modelling/05_gam/output/kcheck_{tt,li}_velocity_*.txt     k-index tables
#   05_modelling/05_gam/output/smooth_{tt,li}_velocity_*.png     Partial-effect panels
#   05_modelling/05_gam/output/comparison_velocity.csv           Cross-model fit summary
#
# Usage:
#   Rscript 05_modelling/05_gam/05c_gam_velocity.R
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

  # Summary to text
  sum_file <- file.path(OUT_DIR, paste0("summary_", tag, ".txt"))
  writeLines(capture.output(summary(model)), sum_file)

  # Significant smooth terms
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

  # Significant parametric terms
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

  # k-check
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
        cat("  All k-index values >= 1.\n")
      }
    }
  }

  # Concurvity
  cat("\n  --- Concurvity ---\n")
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
    cat(sprintf("  Max pairwise concurvity: %.3f (OK)\n", max_cc))
  }

  # gam.check diagnostic plots
  diag_file <- file.path(OUT_DIR, paste0("diag_", tag, ".png"))
  png(diag_file, width = 10, height = 8, units = "in", res = 200)
  par(mfrow = c(2, 2))
  gam.check(model, rep = 200)
  title(main = paste0(toupper(platform), " - ", outcome),
        outer = TRUE, line = -1)
  dev.off()

  # Partial effect plots (gratia)
  smooth_file <- file.path(OUT_DIR, paste0("smooth_", tag, ".png"))
  n_smooths <- length(smooths(model))
  plot_height <- max(6, ceiling(n_smooths / 4) * 3)
  p <- draw(model, residuals = FALSE, rug = TRUE) &
    theme_minimal(base_size = 9)
  ggsave(smooth_file, p, width = 14, height = plot_height, dpi = 200,
         limitsize = FALSE)

  # Save model
  rds_file <- file.path(MODEL_DIR, paste0("m_", tag, ".rds"))
  saveRDS(model, rds_file)
  cat(sprintf("  Model saved: %s (%.1f MB)\n", basename(rds_file),
              file.size(rds_file) / 1e6))

  cat(sprintf("  === %s %s complete ===\n", toupper(platform), outcome))

  invisible(list(dev_expl = dev_expl, r_sq = r_sq, aic = aic_val,
                 n_used = n_used))
}


# =============================================================================
# 1. LOAD DATA
# =============================================================================
cat("== 0. Loading platform data frames ==\n")

PREP_DIR <- file.path(BASE_DIR, "03_data_prep", "data")
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


# =============================================================================
# 2. PREPARE VELOCITY SUBSETS
# =============================================================================
cat("\n== 1. Preparing velocity subsets ==\n")

# --- TikTok T24 ---
df_tt_v24 <- df_tt |> filter(!is.na(velocity_24h))
cat(sprintf("  TikTok  T24: %d / %d rows (%.1f%%)\n",
            nrow(df_tt_v24), nrow(df_tt),
            nrow(df_tt_v24) / nrow(df_tt) * 100))

# --- TikTok T72 ---
df_tt_v72 <- df_tt |> filter(!is.na(velocity_72h))
cat(sprintf("  TikTok  T72: %d / %d rows (%.1f%%)\n",
            nrow(df_tt_v72), nrow(df_tt),
            nrow(df_tt_v72) / nrow(df_tt) * 100))

# --- Instagram T24: INTENTIONALLY SKIPPED ---
# Per Decision 37 (2026-04-17) + Amendment: the Instagram permalink revisit
# scraper writes the wrong field into counters.likes and counters.comments
# at T24/T72. No Instagram velocity model is fitted. The df_ig parquet is
# still loaded above for cross-reference in case downstream scripts need
# it, but no subset or model is produced here.

# --- LinkedIn T24 ---
df_li_v24 <- df_li |> filter(!is.na(velocity_24h))
cat(sprintf("  LinkedIn  T24: %d / %d rows (%.1f%%)\n",
            nrow(df_li_v24), nrow(df_li),
            nrow(df_li_v24) / nrow(df_li) * 100))

# --- Velocity distribution summaries ---
cat("\n  Velocity distributions (combined engagement, Decision 38):\n")
for (nm in list(
  list(d = df_tt_v24, label = "TT v24", col = "velocity_24h"),
  list(d = df_tt_v72, label = "TT v72", col = "velocity_72h"),
  list(d = df_li_v24, label = "LI v24", col = "velocity_24h")
)) {
  v <- nm$d[[nm$col]]
  cat(sprintf("    %s: mean=%.4f, median=%.4f, sd=%.4f, min=%.4f, max=%.4f\n",
              nm$label, mean(v), median(v), sd(v), min(v), max(v)))
  cat(sprintf("           negative=%d (%.1f%%), zero=%d (%.1f%%)\n",
              sum(v < 0), sum(v < 0) / length(v) * 100,
              sum(v == 0), sum(v == 0) / length(v) * 100))
}


# =============================================================================
# 3. SHIFT VELOCITY FOR GAMMA FAMILY
# =============================================================================
# Gamma requires strictly positive response. Shift all values so min > 0.
# Strategy: add |min| + small epsilon so the minimum becomes epsilon.
# This preserves the relative ordering and shape.

cat("\n== 2. Shifting velocity for Gamma family ==\n")

shift_for_gamma <- function(x, label) {
  min_val <- min(x)
  if (min_val <= 0) {
    shift <- abs(min_val) + 0.001
    cat(sprintf("  %s: min=%.4f, shifting by +%.4f\n", label, min_val, shift))
    return(x + shift)
  } else {
    cat(sprintf("  %s: already positive (min=%.4f), no shift needed\n",
                label, min_val))
    return(x)
  }
}

df_tt_v24$velocity_24h_g <- shift_for_gamma(df_tt_v24$velocity_24h, "TT v24")
df_tt_v72$velocity_72h_g <- shift_for_gamma(df_tt_v72$velocity_72h, "TT v72")
df_li_v24$velocity_24h_g <- shift_for_gamma(df_li_v24$velocity_24h, "LI v24")
# Instagram velocity shift intentionally omitted per Decision 37.


# =============================================================================
# 4. TIKTOK T24 VELOCITY GAM — full feature set
# =============================================================================
cat("\n\n================================================================\n")
cat("== 3. TIKTOK T24 — velocity Gamma GAM (log link) ==\n")
cat("================================================================\n")

t0 <- Sys.time()
knots_tt <- list(local_hour = c(0, 24), weekday = c(0, 7))

m_tt_v24 <- gam(
  velocity_24h_g ~
    s(word_count, k = 10) + s(hashtag_count, k = 10) +
    s(emoji_count, k = 10) + s(avg_sentence_len, k = 10) +
    s(exclamation_density, k = 10) + s(question_density, k = 10) +
    s(ellipsis_count, k = 7) + s(caps_ratio, k = 10) +
    s(caps_word_count, k = 10) + s(line_break_count, k = 5) +
    s(mention_count, k = 10) +
    s(punct_diversity, k = 10) + s(flesch_reading_ease, k = 10) +
    s(brightness, k = 10) + s(contrast, k = 10) +
    s(colourfulness, k = 10) + s(log_post_age, k = 10) +
    s(log_follower, k = 10) + s(log_ocr_text_len, k = 10) +
    s(local_hour, bs = "cc", k = 12) +
    s(weekday, bs = "cc", k = 7) +
    url_count +
    cta_flag + is_weekend + is_trending + audio_is_original +
    uses_named_audio + flesch_available + follower_available + face_flag +
    lang + topic_cluster + account_type,
  data    = df_tt_v24,
  family  = Gamma(link = "log"),
  method  = "REML",
  knots   = knots_tt
)

t1 <- Sys.time()
cat(sprintf("  Fitting time: %.1f seconds\n", as.numeric(t1 - t0, units = "secs")))
diag_tt_v24 <- run_diagnostics(m_tt_v24, "tt", "velocity_24h", df_tt_v24)


# =============================================================================
# 5. TIKTOK T72 VELOCITY GAM — full feature set
# =============================================================================
cat("\n\n================================================================\n")
cat("== 4. TIKTOK T72 — velocity Gamma GAM (log link) ==\n")
cat("================================================================\n")

t0 <- Sys.time()

m_tt_v72 <- gam(
  velocity_72h_g ~
    s(word_count, k = 10) + s(hashtag_count, k = 10) +
    s(emoji_count, k = 10) + s(avg_sentence_len, k = 10) +
    s(exclamation_density, k = 10) + s(question_density, k = 10) +
    s(ellipsis_count, k = 6) + s(caps_ratio, k = 10) +  # 6 unique in T72 subset
    s(caps_word_count, k = 10) + s(line_break_count, k = 5) +
    s(mention_count, k = 10) +
    s(punct_diversity, k = 10) + s(flesch_reading_ease, k = 10) +
    s(brightness, k = 10) + s(contrast, k = 10) +
    s(colourfulness, k = 10) + s(log_post_age, k = 10) +
    s(log_follower, k = 10) + s(log_ocr_text_len, k = 10) +
    s(local_hour, bs = "cc", k = 12) +
    s(weekday, bs = "cc", k = 7) +
    url_count +
    cta_flag + is_weekend + is_trending + audio_is_original +
    uses_named_audio + flesch_available + follower_available + face_flag +
    lang + topic_cluster + account_type,
  data    = df_tt_v72,
  family  = Gamma(link = "log"),
  method  = "REML",
  knots   = knots_tt
)

t1 <- Sys.time()
cat(sprintf("  Fitting time: %.1f seconds\n", as.numeric(t1 - t0, units = "secs")))
diag_tt_v72 <- run_diagnostics(m_tt_v72, "tt", "velocity_72h", df_tt_v72)


# =============================================================================
# 6. INSTAGRAM T24 VELOCITY GAM — SKIPPED (Decision 37, 2026-04-17)
# =============================================================================
cat("\n\n================================================================\n")
cat("== 5. INSTAGRAM T24 — SKIPPED (Decision 37 + Amendment) ==\n")
cat("================================================================\n")
cat("  Instagram velocity is NOT modelled. The Instagram permalink revisit\n")
cat("  scraper extracts the wrong field into counters.likes and\n")
cat("  counters.comments at T24/T72 (mean ratio t24/t0 = 0.15 and 0.27\n")
cat("  respectively, versus healthy TikTok 1.03 and LinkedIn 2.56/2.01).\n")
cat("  Full diagnostic: 10_thesis_tables/output/appendix_ig_velocity_diagnostic.csv\n")
cat("  Rationale: see the thesis methodology decision log Decision 37 + Amendment.\n\n")


# =============================================================================
# 7. LINKEDIN T24 VELOCITY GAM — MINIMAL feature set
# =============================================================================
cat("\n\n================================================================\n")
cat("== 6. LINKEDIN T24 — velocity Gamma GAM (MINIMAL, log link) ==\n")
cat("================================================================\n")
cat("  *** CAUTION: N~507, marginal sample. Heavy caveats. ***\n\n")

cat("  Using minimal feature set (~8 smooths + factors) for N=507.\n")

t0 <- Sys.time()
knots_li <- list(local_hour = c(0, 24), weekday = c(0, 7))

m_li_v24 <- gam(
  velocity_24h_g ~
    # --- minimal continuous smooths (6) ---
    s(word_count, k = 5) + s(hashtag_count, k = 5) +
    s(emoji_count, k = 5) + s(flesch_reading_ease, k = 5) +
    s(log_follower, k = 5) + s(log_post_age, k = 5) +
    # --- cyclic (2) ---
    s(local_hour, bs = "cc", k = 6) +
    s(weekday, bs = "cc", k = 5) +
    # --- binary parametric (2) ---
    cta_flag + is_weekend +
    # --- factor parametric (2) ---
    media_type + account_type,
  data    = df_li_v24,
  family  = Gamma(link = "log"),
  method  = "REML",
  knots   = knots_li
)

t1 <- Sys.time()
cat(sprintf("  Fitting time: %.1f seconds\n", as.numeric(t1 - t0, units = "secs")))
diag_li_v24 <- run_diagnostics(m_li_v24, "li", "velocity_24h", df_li_v24)


# =============================================================================
# 8. CROSS-PLATFORM SUMMARY
# =============================================================================
cat("\n\n================================================================\n")
cat("== 7. CROSS-PLATFORM VELOCITY SUMMARY ==\n")
cat("================================================================\n")

summary_df <- data.frame(
  model        = c("TikTok T24", "TikTok T72", "LinkedIn T24"),
  n_fitted     = c(diag_tt_v24$n_used, diag_tt_v72$n_used, diag_li_v24$n_used),
  dev_expl_pct = c(diag_tt_v24$dev_expl, diag_tt_v72$dev_expl, diag_li_v24$dev_expl),
  adj_r_sq     = c(diag_tt_v24$r_sq, diag_tt_v72$r_sq, diag_li_v24$r_sq),
  aic          = c(diag_tt_v24$aic, diag_tt_v72$aic, diag_li_v24$aic)
)
print(summary_df, row.names = FALSE)

write.csv(summary_df,
          file.path(OUT_DIR, "comparison_velocity.csv"),
          row.names = FALSE)

cat("\n== Step 5c complete ==\n")


# =============================================================================
# 9. Decision log
# =============================================================================
cat("\n\n--- Step 5c: GAM Velocity Models (05c_gam_velocity.R) -----\n")
cat(sprintf("Date: %s\n", Sys.Date()))
cat("Outcome: velocity (Gamma GAM, log link)\n\n")

cat("VELOCITY GAM RESULTS (combined engagement, Decision 38)\n")
for (i in 1:nrow(summary_df)) {
  cat(sprintf("  %s: N=%d, deviance explained=%.1f%%, adj.R2=%.3f, AIC=%.1f\n",
              summary_df$model[i], summary_df$n_fitted[i],
              summary_df$dev_expl_pct[i], summary_df$adj_r_sq[i],
              summary_df$aic[i]))
}
cat("\nNOTES:\n")
cat("  - Instagram velocity NOT fitted (Decision 37 + Amendment, 2026-04-17)\n")
cat("  - Velocity is combined engagement: log(L+C+S+1)_tX - log(L+C+S+1)_t0\n")
cat("  - LinkedIn T24: marginal N (507), results exploratory only\n")
cat("  - Velocity shifted by |min|+0.001 for Gamma positivity requirement\n")
cat("\n[See Decision 14 (velocity viability) and Decision 38 (combined definition)]\n")
