# =============================================================================
# 06_ranger_crosscheck.R
#
# Master's thesis: Learning the Levers — Which Creator-Controllable
# Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
# from a Swiss German-Language Perspective
# Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026
#
# Purpose:
#   Fits random forests (ranger) on ever_top for each platform as a
#   cross-check against the GAM inclusion models. Compares variable
#   importance rankings between ranger and GAM significance to identify
#   robust features and candidate interaction effects.
#
# Pipeline position:
#   Step 6 of the modelling pipeline. Depends on 05a_gam_inclusion.R;
#   feeds 08_evaluation.R and the thesis tables in Step 10.
#
# Inputs:
#   05_modelling/03_data_prep/data/df_tt.parquet                   TikTok analytical frame
#   05_modelling/03_data_prep/data/df_ig.parquet                   Instagram analytical frame
#   05_modelling/03_data_prep/data/df_li.parquet                   LinkedIn analytical frame
#   05_modelling/05_gam/models/m_{tt,ig,li}_inclusion.rds          Inclusion GAMs (for comparison)
#
# Outputs:
#   05_modelling/06_ranger/models/rf_{tt,ig,li}_impurity.rds       Fitted ranger models
#   05_modelling/06_ranger/output/vi_{tt,ig,li}_impurity.csv       Impurity importance
#   05_modelling/06_ranger/output/vi_{tt,ig,li}_permutation.csv    Permutation importance
#   05_modelling/06_ranger/output/vi_comparison_all.csv            Combined VI vs GAM
#   05_modelling/06_ranger/output/vi_comparison.png                Thesis figure
#   05_modelling/06_ranger/output/oob_summary.csv                  OOB error and AUC
#
# Usage:
#   Rscript 05_modelling/06_ranger/06_ranger_crosscheck.R
# =============================================================================

rm(list = ls())

# -- Source shared config ------------------------------------------------------
source(file.path("config", "packages.R"))
library(patchwork)

STEP_DIR  <- file.path(BASE_DIR, "06_ranger")
OUT_DIR   <- file.path(STEP_DIR, "output")
MODEL_DIR <- file.path(STEP_DIR, "models")
GAM_DIR   <- file.path(BASE_DIR, "05_gam", "models")
dir.create(OUT_DIR,   recursive = TRUE, showWarnings = FALSE)
dir.create(MODEL_DIR, recursive = TRUE, showWarnings = FALSE)


# =============================================================================
# 0. HELPER: Extract GAM significant features
# =============================================================================

extract_gam_sig <- function(gam_model) {
  # Returns a data frame of all features and their GAM significance
  sig_features <- tibble(feature = character(), gam_p = numeric(),
                         gam_type = character(), gam_direction = character())

  # Smooth terms
  sm <- summary(gam_model)$s.table
  if (!is.null(sm) && nrow(sm) > 0) {
    for (i in seq_len(nrow(sm))) {
      # Extract feature name from s(feature_name) or s(feature_name, bs=...)
      term <- rownames(sm)[i]
      feat <- gsub("^s\\(([^,)]+).*\\)$", "\\1", term)
      edf  <- sm[i, "edf"]
      direction <- if (edf < 1.5) "approx. linear" else "non-linear"
      sig_features <- bind_rows(sig_features, tibble(
        feature = feat, gam_p = sm[i, "p-value"],
        gam_type = "smooth", gam_direction = direction
      ))
    }
  }

  # Parametric terms
  pt <- summary(gam_model)$p.table
  p_col <- if ("Pr(>|z|)" %in% colnames(pt)) "Pr(>|z|)" else "Pr(>|t|)"
  if (!is.null(pt) && nrow(pt) > 0) {
    for (i in seq_len(nrow(pt))) {
      term <- rownames(pt)[i]
      if (term == "(Intercept)") next
      # Map factor levels back to base feature name
      feat <- term
      for (fvar in c("lang", "media_type", "topic_cluster", "account_type")) {
        if (startsWith(term, fvar)) { feat <- fvar; break }
      }
      est <- pt[i, "Estimate"]
      direction <- if (is.nan(est)) "aliased" else if (est > 0) "positive" else "negative"
      sig_features <- bind_rows(sig_features, tibble(
        feature = feat, gam_p = pt[i, p_col],
        gam_type = "parametric", gam_direction = direction
      ))
    }
  }

  # For factors with multiple levels, keep the most significant p-value
  sig_features <- sig_features |>
    filter(!is.nan(gam_p)) |>
    group_by(feature) |>
    summarise(
      gam_p         = min(gam_p, na.rm = TRUE),
      gam_type      = first(gam_type),
      gam_direction = first(gam_direction),
      .groups = "drop"
    ) |>
    mutate(gam_significant = gam_p < 0.05)

  sig_features
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


# =============================================================================
# 3. BUILD FEATURE MATRICES
# =============================================================================
cat("\n== 2. Building feature matrices (same predictors as GAM inclusion) ==\n")

# Whitelist approach: explicitly list predictors per platform (matches GAM formulas)
# Using whitelist avoids accidentally including non-predictor columns with NAs

FEATURES_TT <- c(
  # continuous smooths (19)
  "word_count", "hashtag_count", "emoji_count", "avg_sentence_len",
  "exclamation_density", "question_density", "ellipsis_count",
  "caps_ratio", "caps_word_count", "line_break_count", "url_count",
  "mention_count", "punct_diversity", "flesch_reading_ease",
  "brightness", "contrast", "colourfulness",
  "log_post_age", "log_follower", "log_ocr_text_len",
  # cyclic (2)
  "local_hour", "weekday",
  # binary (9)
  "cta_flag", "is_weekend", "is_trending", "audio_is_original",
  "uses_named_audio", "flesch_available", "follower_available", "face_flag",
  # factors (3; media_type excluded: constant "video")
  "lang", "topic_cluster", "account_type"
)

FEATURES_IG <- c(
  # continuous smooths (19; no temporal, no log_post_age)
  "word_count", "hashtag_count", "emoji_count", "avg_sentence_len",
  "exclamation_density", "question_density", "ellipsis_count",
  "caps_ratio", "caps_word_count", "line_break_count", "url_count",
  "mention_count", "punct_diversity", "flesch_reading_ease",
  "brightness", "contrast", "colourfulness",
  "log_follower", "log_ocr_text_len",
  # binary (4; no temporal/audio TikTok-only/is_trending)
  "cta_flag", "flesch_available", "follower_available", "face_flag",
  # factors (4)
  "lang", "media_type", "topic_cluster", "account_type"
)

FEATURES_LI <- c(
  # continuous smooths (15; no visual, no log_ocr_text_len)
  "word_count", "hashtag_count", "emoji_count", "avg_sentence_len",
  "exclamation_density", "question_density", "ellipsis_count",
  "caps_ratio", "caps_word_count", "line_break_count", "url_count",
  "punct_diversity", "flesch_reading_ease",
  "log_post_age", "log_follower",
  # cyclic (2)
  "local_hour", "weekday",
  # binary (5; mention_count is parametric in GAM but ranger handles it)
  "mention_count", "cta_flag", "is_weekend", "flesch_available", "follower_available",
  # factors (4)
  "lang", "media_type", "topic_cluster", "account_type"
)

build_rf_data <- function(df, features) {
  # Select only the outcome + whitelisted predictors
  cols <- intersect(c("ever_top", features), names(df))
  out <- df[, cols]
  out$ever_top <- as.factor(out$ever_top)

  missing <- setdiff(features, names(df))
  if (length(missing) > 0) {
    cat(sprintf("  WARNING: features not in data: %s\n",
                paste(missing, collapse = ", ")))
  }
  cat(sprintf("  %d rows x %d predictors\n", nrow(out), ncol(out) - 1))
  out
}

cat("  TikTok:    ")
rf_tt <- build_rf_data(df_tt, FEATURES_TT)
cat("  Instagram: ")
rf_ig <- build_rf_data(df_ig, FEATURES_IG)
cat("  LinkedIn:  ")
rf_li <- build_rf_data(df_li, FEATURES_LI)


# =============================================================================
# 4. FIT RANGER MODELS
# =============================================================================
cat("\n== 3. Fitting ranger models ==\n")

fit_ranger <- function(df, platform_name, imp_type, seed = 42) {
  cat(sprintf("\n  --- %s (%s importance) ---\n",
              toupper(platform_name), imp_type))
  t0 <- Sys.time()

  # Drop rows with NA in predictors (match GAM's na.action)
  n_before <- nrow(df)
  df <- na.omit(df)
  n_after  <- nrow(df)
  if (n_before != n_after) {
    cat(sprintf("  Dropped %d rows with NAs (%d -> %d)\n",
                n_before - n_after, n_before, n_after))
  }

  m <- ranger(
    ever_top ~ .,
    data                       = df,
    importance                 = imp_type,
    num.trees                  = 2000,
    seed                       = seed,
    probability                = TRUE,
    respect.unordered.factors  = "order"
  )

  t1 <- Sys.time()
  cat(sprintf("  Fitting time: %.1f seconds\n",
              as.numeric(t1 - t0, units = "secs")))
  cat(sprintf("  OOB prediction error: %.4f\n", m$prediction.error))
  cat(sprintf("  Num trees: %d | Mtry: %d\n", m$num.trees, m$mtry))

  # OOB AUC (probability predictions)
  oob_preds <- m$predictions[, "1"]  # probability of class "1" (top)
  oob_truth <- df$ever_top
  auc_df <- tibble(truth = oob_truth, estimate = oob_preds)
  auc_val <- roc_auc(auc_df, truth, estimate, event_level = "second")$.estimate
  cat(sprintf("  OOB AUC: %.4f\n", auc_val))

  list(model = m, auc = auc_val, n_used = n_after)
}

# -- Impurity importance (primary) --
cat("\n  ===== IMPURITY IMPORTANCE =====\n")
res_tt_imp <- fit_ranger(rf_tt, "tiktok",    "impurity")
res_ig_imp <- fit_ranger(rf_ig, "instagram",  "impurity")
res_li_imp <- fit_ranger(rf_li, "linkedin",   "impurity")

# -- Permutation importance (robustness check) --
cat("\n  ===== PERMUTATION IMPORTANCE =====\n")
res_tt_perm <- fit_ranger(rf_tt, "tiktok",    "permutation")
res_ig_perm <- fit_ranger(rf_ig, "instagram",  "permutation")
res_li_perm <- fit_ranger(rf_li, "linkedin",   "permutation")

# Save models
saveRDS(res_tt_imp$model, file.path(MODEL_DIR, "rf_tt_impurity.rds"))
saveRDS(res_ig_imp$model, file.path(MODEL_DIR, "rf_ig_impurity.rds"))
saveRDS(res_li_imp$model, file.path(MODEL_DIR, "rf_li_impurity.rds"))
cat("\n  Models saved to 06_ranger/models/\n")


# =============================================================================
# 5. EXTRACT VARIABLE IMPORTANCE
# =============================================================================
cat("\n== 4. Extracting variable importance ==\n")

extract_vi <- function(model, platform_name, imp_type) {
  vi <- importance(model)
  vi_df <- tibble(
    feature    = names(vi),
    importance = unname(vi)
  ) |>
    arrange(desc(importance)) |>
    mutate(rank = row_number())

  # Save
  fname <- sprintf("vi_%s_%s.csv",
                    c(tiktok="tt", instagram="ig", linkedin="li")[[platform_name]],
                    imp_type)
  write.csv(vi_df, file.path(OUT_DIR, fname), row.names = FALSE)
  cat(sprintf("  %s %s: saved %s (%d features)\n",
              platform_name, imp_type, fname, nrow(vi_df)))

  # Print top 15
  cat(sprintf("  Top 15 (%s %s):\n", platform_name, imp_type))
  top15 <- head(vi_df, 15)
  for (i in seq_len(nrow(top15))) {
    cat(sprintf("    %2d. %-25s  %.4f\n",
                top15$rank[i], top15$feature[i], top15$importance[i]))
  }
  vi_df
}

vi_tt_imp  <- extract_vi(res_tt_imp$model,  "tiktok",    "impurity")
vi_ig_imp  <- extract_vi(res_ig_imp$model,  "instagram", "impurity")
vi_li_imp  <- extract_vi(res_li_imp$model,  "linkedin",  "impurity")
vi_tt_perm <- extract_vi(res_tt_perm$model, "tiktok",    "permutation")
vi_ig_perm <- extract_vi(res_ig_perm$model, "instagram", "permutation")
vi_li_perm <- extract_vi(res_li_perm$model, "linkedin",  "permutation")


# =============================================================================
# 6. LOAD GAM MODELS AND BUILD COMPARISON TABLE
# =============================================================================
cat("\n== 5. Building ranger vs GAM comparison table ==\n")

# Load GAM inclusion models
m_tt_gam <- readRDS(file.path(GAM_DIR, "m_tt_inclusion.rds"))
m_ig_gam <- readRDS(file.path(GAM_DIR, "m_ig_inclusion.rds"))
m_li_gam <- readRDS(file.path(GAM_DIR, "m_li_inclusion.rds"))

gam_sig_tt <- extract_gam_sig(m_tt_gam)
gam_sig_ig <- extract_gam_sig(m_ig_gam)
gam_sig_li <- extract_gam_sig(m_li_gam)

build_comparison <- function(vi_imp, vi_perm, gam_sig, platform_name) {
  # Merge impurity rank
  comp <- vi_imp |>
    select(feature, imp_importance = importance, ranger_impurity_rank = rank)

  # Merge permutation rank
  comp <- comp |>
    left_join(
      vi_perm |> select(feature, perm_importance = importance,
                         ranger_perm_rank = rank),
      by = "feature"
    )

  # Merge GAM significance
  comp <- comp |>
    left_join(gam_sig, by = "feature") |>
    mutate(
      gam_significant = replace_na(gam_significant, FALSE),
      gam_p           = replace_na(gam_p, 1),
      gam_type        = replace_na(gam_type, "not_in_gam"),
      gam_direction   = replace_na(gam_direction, "none"),
      ranger_top15    = ranger_impurity_rank <= 15,
      agreement       = case_when(
        ranger_top15 & gam_significant  ~ "both",
        ranger_top15 & !gam_significant ~ "ranger_only",
        !ranger_top15 & gam_significant ~ "gam_only",
        TRUE                            ~ "neither"
      ),
      platform = platform_name
    )

  cat(sprintf("\n  %s agreement summary:\n", platform_name))
  cat(sprintf("    both:        %d\n", sum(comp$agreement == "both")))
  cat(sprintf("    ranger_only: %d\n", sum(comp$agreement == "ranger_only")))
  cat(sprintf("    gam_only:    %d\n", sum(comp$agreement == "gam_only")))
  cat(sprintf("    neither:     %d\n", sum(comp$agreement == "neither")))

  comp
}

comp_tt <- build_comparison(vi_tt_imp, vi_tt_perm, gam_sig_tt, "tiktok")
comp_ig <- build_comparison(vi_ig_imp, vi_ig_perm, gam_sig_ig, "instagram")
comp_li <- build_comparison(vi_li_imp, vi_li_perm, gam_sig_li, "linkedin")

# Combined table
comp_all <- bind_rows(comp_tt, comp_ig, comp_li)
write.csv(comp_all, file.path(OUT_DIR, "vi_comparison_all.csv"),
          row.names = FALSE)
cat(sprintf("\n  Combined comparison saved: vi_comparison_all.csv (%d rows)\n",
            nrow(comp_all)))


# =============================================================================
# 7. OOB SUMMARY TABLE
# =============================================================================
cat("\n== 6. OOB performance summary ==\n")

oob_summary <- tibble(
  platform = c("TikTok", "Instagram", "LinkedIn"),
  n_fitted = c(res_tt_imp$n_used, res_ig_imp$n_used, res_li_imp$n_used),
  oob_error = c(res_tt_imp$model$prediction.error,
                res_ig_imp$model$prediction.error,
                res_li_imp$model$prediction.error),
  oob_auc   = c(res_tt_imp$auc, res_ig_imp$auc, res_li_imp$auc)
)
print(oob_summary)
write.csv(oob_summary, file.path(OUT_DIR, "oob_summary.csv"),
          row.names = FALSE)


# =============================================================================
# 8. VISUALIZATION: vi_comparison.png (thesis figure)
# =============================================================================
cat("\n== 7. Creating vi_comparison.png ==\n")

PLOT_DIR <- OUT_DIR  # save_plot() uses this

make_vi_panel <- function(comp_df, platform_name, platform_colour) {
  # Top 15 by impurity importance + any GAM-only features
  top15 <- comp_df |> filter(ranger_impurity_rank <= 15)
  gam_only <- comp_df |> filter(agreement == "gam_only")

  plot_data <- bind_rows(top15, gam_only) |>
    distinct(feature, .keep_all = TRUE) |>
    mutate(
      fill_colour = case_when(
        agreement == "both"        ~ platform_colour,
        agreement == "ranger_only" ~ "#CCCCCC",
        agreement == "gam_only"    ~ "white",
        TRUE                       ~ "#CCCCCC"
      ),
      border_colour = case_when(
        agreement == "gam_only" ~ platform_colour,
        TRUE                    ~ "grey30"
      ),
      # Clean feature names for display
      feature_label = feature |>
        str_replace_all("_", " ") |>
        str_to_title() |>
        str_replace("Log ", "log ") |>
        str_replace("Ocr", "OCR") |>
        str_replace("Cta", "CTA"),
      # Use impurity importance for bar length; GAM-only gets 0
      bar_value = if_else(is.na(imp_importance), 0, imp_importance)
    ) |>
    arrange(desc(bar_value)) |>
    mutate(feature_label = fct_inorder(feature_label))

  p <- ggplot(plot_data, aes(x = bar_value,
                              y = fct_rev(feature_label))) +
    geom_col(aes(fill = fill_colour, colour = border_colour),
             linewidth = 0.4, width = 0.7) +
    scale_fill_identity() +
    scale_colour_identity() +
    # Mark GAM-only features with a triangle
    geom_point(
      data = plot_data |> filter(agreement == "gam_only"),
      aes(x = max(plot_data$bar_value) * 0.02, y = fct_rev(feature_label)),
      shape = 17, size = 2.5, colour = platform_colour
    ) +
    labs(
      title = PLATFORM_LABELS[[platform_name]],
      x = "Impurity importance",
      y = NULL
    ) +
    theme_thesis(base_size = 9) +
    theme(plot.title = element_text(colour = platform_colour, face = "bold"))

  p
}

p_tt <- make_vi_panel(comp_tt, "tiktok",    PLATFORM_COLOURS_LC[["tiktok"]])
p_ig <- make_vi_panel(comp_ig, "instagram", PLATFORM_COLOURS_LC[["instagram"]])
p_li <- make_vi_panel(comp_li, "linkedin",  PLATFORM_COLOURS_LC[["linkedin"]])

p_combined <- p_tt / p_ig / p_li +
  plot_annotation(
    title    = "Ranger Variable Importance vs GAM Significance",
    subtitle = "Solid colour = both agree | Grey = ranger only | Triangle = GAM only",
    theme = theme_thesis(base_size = 10)
  )

save_plot(p_combined, "vi_comparison.png", width = 6.5, height = 12)


# =============================================================================
# 9. DECISION: INTERACTION TERMS
# =============================================================================
cat("\n\n================================================================\n")
cat("== 8. DECISION: Interaction terms ==\n")
cat("================================================================\n")
cat("Criteria: features ranked top-5 by ranger impurity but NOT significant\n")
cat("(p >= 0.05) in the GAM. These may indicate interaction-driven effects.\n\n")

assess_interactions <- function(comp_df, platform_name) {
  candidates <- comp_df |>
    filter(ranger_impurity_rank <= 5, !gam_significant) |>
    arrange(ranger_impurity_rank)

  cat(sprintf("  %s: %d interaction candidates\n",
              toupper(platform_name), nrow(candidates)))
  if (nrow(candidates) > 0) {
    for (i in seq_len(nrow(candidates))) {
      cat(sprintf("    Rank %d: %-25s (ranger imp=%.4f, GAM p=%.3f, perm rank=%d)\n",
                  candidates$ranger_impurity_rank[i],
                  candidates$feature[i],
                  candidates$imp_importance[i],
                  candidates$gam_p[i],
                  candidates$ranger_perm_rank[i]))
    }
  } else {
    cat("    No candidates (all top-5 ranger features are GAM-significant,\n")
    cat("    or no divergence detected).\n")
  }
  candidates
}

cand_tt <- assess_interactions(comp_tt, "tiktok")
cand_ig <- assess_interactions(comp_ig, "instagram")
cand_li <- assess_interactions(comp_li, "linkedin")

cat("\n  INTERACTION DECISION:\n")
cat("  Review the candidates above. Per the modelling plan, limit to 1-2\n")
cat("  tensor product smooths per platform. Only proceed if:\n")
cat("    (a) The feature is top-5 in ranger AND confirmed by permutation.\n")
cat("    (b) There is a plausible substantive reason for an interaction.\n")
cat("  Do NOT go on a fishing expedition.\n")


# =============================================================================
# 10. CROSS-PLATFORM SUMMARY
# =============================================================================
cat("\n\n================================================================\n")
cat("== 9. CROSS-PLATFORM SUMMARY ==\n")
cat("================================================================\n")

for (plat in c("tiktok", "instagram", "linkedin")) {
  comp <- comp_all |> filter(platform == plat)
  cat(sprintf("\n  %s:\n", PLATFORM_LABELS[[plat]]))
  cat(sprintf("    Features in both:     %s\n",
              paste(comp$feature[comp$agreement == "both"], collapse = ", ")))
  cat(sprintf("    Ranger-only (top 15): %s\n",
              paste(comp$feature[comp$agreement == "ranger_only"], collapse = ", ")))
  cat(sprintf("    GAM-only:             %s\n",
              paste(comp$feature[comp$agreement == "gam_only"], collapse = ", ")))
}


# =============================================================================
# 11. Decision log
# =============================================================================
cat("\n\n--- Step 6: Ranger Cross-Check (06_ranger_crosscheck.R) ------\n")
cat(sprintf("Date: %s\n", Sys.Date()))
cat("Output: 06_ranger/output/vi_*_impurity.csv, vi_*_permutation.csv\n")
cat("        06_ranger/output/vi_comparison_all.csv\n")
cat("        06_ranger/output/vi_comparison.png (thesis figure)\n")
cat("        06_ranger/output/oob_summary.csv\n\n")

cat("RANGER OOB PERFORMANCE (ever_top, probability forest, 2000 trees)\n")
for (i in 1:3) {
  cat(sprintf("  %s: N=%d, OOB error=%.4f, OOB AUC=%.4f\n",
              oob_summary$platform[i], oob_summary$n_fitted[i],
              oob_summary$oob_error[i], oob_summary$oob_auc[i]))
}

cat("\nTOP-15 VARIABLE IMPORTANCE (impurity) PER PLATFORM\n")
for (plat in c("tiktok", "instagram", "linkedin")) {
  comp <- comp_all |> filter(platform == plat, ranger_impurity_rank <= 15)
  cat(sprintf("\n  %s:\n", PLATFORM_LABELS[[plat]]))
  for (i in seq_len(nrow(comp))) {
    marker <- if (comp$gam_significant[i]) "*" else " "
    cat(sprintf("    %2d. %s %-25s  imp=%.4f  (perm rank=%d, GAM p=%.3f)\n",
                comp$ranger_impurity_rank[i], marker, comp$feature[i],
                comp$imp_importance[i], comp$ranger_perm_rank[i],
                comp$gam_p[i]))
  }
}

cat("\n  * = also significant in GAM inclusion model (p < 0.05)\n")

cat("\nAGREEMENT SUMMARY\n")
for (plat in c("tiktok", "instagram", "linkedin")) {
  comp <- comp_all |> filter(platform == plat)
  cat(sprintf("  %s: both=%d, ranger_only=%d, gam_only=%d, neither=%d\n",
              PLATFORM_LABELS[[plat]],
              sum(comp$agreement == "both"),
              sum(comp$agreement == "ranger_only"),
              sum(comp$agreement == "gam_only"),
              sum(comp$agreement == "neither")))
}

cat("\nDecision 20: Interaction terms — see the thesis methodology decision log\n")

cat("\n== Step 6 complete ==\n")
