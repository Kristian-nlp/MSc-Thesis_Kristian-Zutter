# =============================================================================
# 07a_comparison_table.R
#
# Master's thesis: Learning the Levers — Which Creator-Controllable
# Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
# from a Swiss German-Language Perspective
# Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026
#
# Purpose:
#   Cross-platform comparison table for RQ2. For every feature available on
#   at least two platforms, extracts from the inclusion GAMs: significance,
#   direction, approximate magnitude (change in log-odds across the
#   feature's IQR), and edf. Produces a features-by-platforms matrix and
#   the effect-comparison heatmap.
#
# Pipeline position:
#   Step 7a of the modelling pipeline. Depends on Steps 1–6; feeds the
#   cross-platform thesis tables in Step 10.
#
# Inputs:
#   05_modelling/03_data_prep/data/df_tt.parquet                   TikTok analytical frame
#   05_modelling/03_data_prep/data/df_ig.parquet                   Instagram analytical frame
#   05_modelling/03_data_prep/data/df_li.parquet                   LinkedIn analytical frame
#   05_modelling/05_gam/models/m_{tt,ig,li}_inclusion.rds          Inclusion GAMs
#
# Outputs:
#   05_modelling/07_cross_platform/output/effect_comparison_table.csv     Long format
#   05_modelling/07_cross_platform/output/effect_comparison_wide.csv      Wide summary
#   05_modelling/07_cross_platform/output/effect_comparison_heatmap.png   Thesis figure
#
# Usage:
#   Rscript 05_modelling/07_cross_platform/07a_comparison_table.R
# =============================================================================

rm(list = ls())

# -- Source shared config ------------------------------------------------------
source(file.path("config", "packages.R"))

STEP_DIR  <- file.path(BASE_DIR, "07_cross_platform")
OUT_DIR   <- file.path(STEP_DIR, "output")
PLOT_DIR  <- OUT_DIR
GAM_DIR   <- file.path(BASE_DIR, "05_gam", "models")
PREP_DIR  <- file.path(BASE_DIR, "03_data_prep", "data")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)


# =============================================================================
# 0. HELPER: Extract GAM term info (reused from 06_ranger_crosscheck.R)
# =============================================================================

extract_gam_terms <- function(gam_model) {
  # Returns a tibble with: feature, p_value, feature_type, direction, edf
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
# 1. LOAD MODELS AND DATA
# =============================================================================

cat("== 1. Loading GAM inclusion models and platform data ==\n")

m_tt <- readRDS(file.path(GAM_DIR, "m_tt_inclusion.rds"))
m_ig <- readRDS(file.path(GAM_DIR, "m_ig_inclusion.rds"))
m_li <- readRDS(file.path(GAM_DIR, "m_li_inclusion.rds"))

df_tt <- read_parquet(file.path(PREP_DIR, "df_tt.parquet"))
df_ig <- read_parquet(file.path(PREP_DIR, "df_ig.parquet"))
df_li <- read_parquet(file.path(PREP_DIR, "df_li.parquet"))

# Ensure factors match model expectations
factor_vars <- c("lang", "media_type", "topic_cluster", "account_type")
for (v in factor_vars) {
  if (v %in% names(df_tt)) df_tt[[v]] <- as.factor(df_tt[[v]])
  if (v %in% names(df_ig)) df_ig[[v]] <- as.factor(df_ig[[v]])
  if (v %in% names(df_li)) df_li[[v]] <- as.factor(df_li[[v]])
}

cat(sprintf("  TikTok:    %d obs, model deviance %.1f%%\n",
            nrow(df_tt), summary(m_tt)$dev.expl * 100))
cat(sprintf("  Instagram: %d obs, model deviance %.1f%%\n",
            nrow(df_ig), summary(m_ig)$dev.expl * 100))
cat(sprintf("  LinkedIn:  %d obs, model deviance %.1f%%\n",
            nrow(df_li), summary(m_li)$dev.expl * 100))


# =============================================================================
# 2. EXTRACT GAM TERM INFO PER PLATFORM
# =============================================================================

cat("\n== 2. Extracting GAM term significance ==\n")

terms_tt <- extract_gam_terms(m_tt) |> mutate(platform = "tiktok")
terms_ig <- extract_gam_terms(m_ig) |> mutate(platform = "instagram")
terms_li <- extract_gam_terms(m_li) |> mutate(platform = "linkedin")

terms_all <- bind_rows(terms_tt, terms_ig, terms_li)

# Identify features present on 2+ platforms
feature_counts <- terms_all |>
  distinct(feature, platform) |>
  count(feature, name = "n_platforms") |>
  filter(n_platforms >= 2)

cat(sprintf("  Features on 2+ platforms: %d\n", nrow(feature_counts)))
cat(sprintf("  Features on all 3 platforms: %d\n",
            sum(feature_counts$n_platforms == 3)))


# =============================================================================
# 3. COMPUTE IQR MAGNITUDE (change in log-odds from Q25 to Q75)
# =============================================================================

cat("\n== 3. Computing IQR magnitude ==\n")

# Helper: create a "median profile" data frame for prediction
make_median_profile <- function(df, model) {
  # Get all predictor names from model
  pred_names <- attr(model$terms, "term.labels")
  # Also get smooth term variable names
  sm_names <- names(model$var.summary)

  all_vars <- unique(c(pred_names, sm_names))
  profile <- list()
  for (v in all_vars) {
    if (!(v %in% names(df))) next
    if (is.factor(df[[v]])) {
      # Mode: most frequent level
      profile[[v]] <- names(sort(table(df[[v]]), decreasing = TRUE))[1]
      profile[[v]] <- factor(profile[[v]], levels = levels(df[[v]]))
    } else if (is.logical(df[[v]]) || all(df[[v]] %in% c(0, 1), na.rm = TRUE)) {
      # Binary: mode
      profile[[v]] <- as.numeric(names(sort(table(df[[v]]), decreasing = TRUE))[1])
    } else {
      # Continuous: median
      profile[[v]] <- median(df[[v]], na.rm = TRUE)
    }
  }
  as_tibble(profile)
}

# Helper: compute magnitude for one feature on one platform
compute_magnitude <- function(feature, model, df, profile) {
  if (!(feature %in% names(df))) return(NA_real_)

  col <- df[[feature]]

  if (is.factor(col)) {
    # Factor: extract coefficient range from parametric table, filtering out
    # sparse levels (< 30 obs) whose coefficients are unreliable. Without this,
    # LinkedIn factor magnitudes are extreme (e.g., topic_cluster 178 log-odds)
    # due to convergence issues and levels with 2-3 observations.
    pt <- summary(model)$p.table
    level_rows <- grep(paste0("^", feature), rownames(pt))
    if (length(level_rows) == 0) return(NA_real_)

    # Get level frequencies to filter sparse levels
    level_freq <- table(col)
    kept_coefs <- c(0)  # reference level (coef = 0)
    for (idx in level_rows) {
      est <- pt[idx, "Estimate"]
      if (is.nan(est)) next
      # Extract level name: remove feature prefix from row name
      lvl_name <- sub(paste0("^", feature), "", rownames(pt)[idx])
      # Keep only if level has >= 30 observations
      if (lvl_name %in% names(level_freq) && level_freq[[lvl_name]] >= 30) {
        kept_coefs <- c(kept_coefs, est)
      }
    }
    if (length(kept_coefs) < 2) {
      # Fallback: use all non-aliased coefficients if no level passes filter
      all_coefs <- pt[level_rows, "Estimate"]
      all_coefs <- c(0, all_coefs[!is.nan(all_coefs)])
      return(max(all_coefs) - min(all_coefs))
    }
    return(max(kept_coefs) - min(kept_coefs))
  }

  if (is.logical(col) || all(col %in% c(0, 1), na.rm = TRUE)) {
    # Binary: extract coefficient directly from parametric table
    pt <- summary(model)$p.table
    if (feature %in% rownames(pt)) {
      est <- pt[feature, "Estimate"]
      if (is.nan(est)) return(NA_real_)
      return(est)
    }
    # Fallback to predict if feature name doesn't match exactly
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

# Build median profiles
profile_tt <- make_median_profile(df_tt, m_tt)
profile_ig <- make_median_profile(df_ig, m_ig)
profile_li <- make_median_profile(df_li, m_li)

# Compute magnitude for all shared features on all platforms
models   <- list(tiktok = m_tt, instagram = m_ig, linkedin = m_li)
dfs      <- list(tiktok = df_tt, instagram = df_ig, linkedin = df_li)
profiles <- list(tiktok = profile_tt, instagram = profile_ig, linkedin = profile_li)

magnitudes <- tibble(feature = character(), platform = character(),
                     magnitude_logodds = numeric())

for (feat in feature_counts$feature) {
  for (plat in c("tiktok", "instagram", "linkedin")) {
    # Only compute if feature is in this platform's model
    if (feat %in% (terms_all |> filter(platform == plat) |> pull(feature))) {
      mag <- compute_magnitude(feat, models[[plat]], dfs[[plat]], profiles[[plat]])
      magnitudes <- bind_rows(magnitudes, tibble(
        feature = feat, platform = plat, magnitude_logodds = mag
      ))
    }
  }
}


# =============================================================================
# 4. BUILD LONG-FORMAT COMPARISON TABLE
# =============================================================================

cat("\n== 4. Building comparison table ==\n")

# Join term info with magnitudes
comparison_long <- terms_all |>
  inner_join(feature_counts |> select(feature), by = "feature") |>
  left_join(magnitudes, by = c("feature", "platform")) |>
  mutate(
    p_stars = case_when(
      p_value < 0.001 ~ "***",
      p_value < 0.01  ~ "**",
      p_value < 0.05  ~ "*",
      p_value < 0.10  ~ ".",
      TRUE            ~ ""
    ),
    magnitude_logodds = round(magnitude_logodds, 3)
  ) |>
  arrange(feature, platform)

# Save long format
write.csv(comparison_long, file.path(OUT_DIR, "effect_comparison_table.csv"),
          row.names = FALSE)
cat(sprintf("  Saved: effect_comparison_table.csv (%d rows, %d features)\n",
            nrow(comparison_long), n_distinct(comparison_long$feature)))


# =============================================================================
# 5. BUILD WIDE-FORMAT SUMMARY
# =============================================================================

cat("\n== 5. Building wide-format summary ==\n")

# Create a compact summary string per feature-platform cell
comparison_wide <- comparison_long |>
  mutate(
    cell = paste0(
      ifelse(significant, direction, "n.s."),
      " (", sprintf("%.3f", p_value), p_stars, ")",
      ifelse(!is.na(magnitude_logodds),
             paste0(" [", sprintf("%+.2f", magnitude_logodds), "]"), "")
    )
  ) |>
  select(feature, platform, cell) |>
  pivot_wider(names_from = platform, values_from = cell, values_fill = "---")

write.csv(comparison_wide, file.path(OUT_DIR, "effect_comparison_wide.csv"),
          row.names = FALSE)
cat(sprintf("  Saved: effect_comparison_wide.csv (%d features)\n",
            nrow(comparison_wide)))


# =============================================================================
# 6. Console summary
# =============================================================================

cat("\n\n== CROSS-PLATFORM COMPARISON SUMMARY ==\n")
cat("=========================================\n\n")

# Shared features with significance on 2+ platforms
sig_on_2plus <- comparison_long |>
  filter(significant) |>
  count(feature, name = "n_sig_platforms") |>
  filter(n_sig_platforms >= 2)

cat(sprintf("Features significant on 2+ platforms: %d\n", nrow(sig_on_2plus)))
if (nrow(sig_on_2plus) > 0) {
  for (i in seq_len(nrow(sig_on_2plus))) {
    feat <- sig_on_2plus$feature[i]
    cat(sprintf("  %s (%d platforms)\n", feat, sig_on_2plus$n_sig_platforms[i]))
    # Show details
    details <- comparison_long |> filter(feature == feat, significant)
    for (j in seq_len(nrow(details))) {
      cat(sprintf("    %s: %s, p=%.2e, mag=%+.3f log-odds\n",
                  PLATFORM_LABELS[details$platform[j]],
                  details$direction[j],
                  details$p_value[j],
                  details$magnitude_logodds[j]))
    }
  }
}

# Features significant on exactly 1 platform
sig_on_1 <- comparison_long |>
  filter(significant) |>
  count(feature, name = "n_sig_platforms") |>
  filter(n_sig_platforms == 1)

cat(sprintf("\nFeatures significant on exactly 1 platform: %d\n", nrow(sig_on_1)))
if (nrow(sig_on_1) > 0) {
  for (i in seq_len(nrow(sig_on_1))) {
    feat <- sig_on_1$feature[i]
    detail <- comparison_long |> filter(feature == feat, significant)
    cat(sprintf("  %s: %s only (%s, p=%.2e)\n",
                feat,
                PLATFORM_LABELS[detail$platform],
                detail$direction,
                detail$p_value))
  }
}

# Direction agreement check
cat("\n--- Direction agreement (features significant on 2+ platforms) ---\n")
if (nrow(sig_on_2plus) > 0) {
  for (i in seq_len(nrow(sig_on_2plus))) {
    feat <- sig_on_2plus$feature[i]
    dirs <- comparison_long |> filter(feature == feat, significant) |> pull(direction)
    agree <- length(unique(dirs)) == 1
    cat(sprintf("  %s: %s  %s\n",
                feat,
                paste(dirs, collapse = " / "),
                ifelse(agree, "(AGREE)", "*** DIVERGE ***")))
  }
}

# Magnitude ranking (top 10 by absolute magnitude across all platforms)
cat("\n--- Top 10 features by absolute IQR magnitude ---\n")
top_mag <- comparison_long |>
  filter(!is.na(magnitude_logodds)) |>
  mutate(abs_mag = abs(magnitude_logodds)) |>
  arrange(desc(abs_mag)) |>
  head(10)
for (i in seq_len(nrow(top_mag))) {
  cat(sprintf("  %2d. %-25s  %s  mag=%+.3f  p=%.2e %s\n",
              i,
              top_mag$feature[i],
              PLATFORM_LABELS[top_mag$platform[i]],
              top_mag$magnitude_logodds[i],
              top_mag$p_value[i],
              top_mag$p_stars[i]))
}

# =============================================================================
# EFFECT COMPARISON HEATMAP (Thesis Figure 18)
# =============================================================================
cat("\n== Generating effect comparison heatmap ==\n")

# Exclude pure controls for a cleaner content-focused figure
control_features <- c("account_type", "follower_available", "flesch_available")

heat_data <- comparison_long |>
  filter(!feature %in% control_features) |>
  mutate(
    platform_label = factor(PLATFORM_LABELS[platform],
                            levels = c("TikTok", "Instagram", "LinkedIn")),
    # Encode: significant + direction -> fill value
    fill_val = case_when(
      !significant                                     ~  0,    # n.s.
      direction %in% c("positive", "approx. linear")   ~  magnitude_logodds,
      direction == "negative"                          ~  magnitude_logodds,
      direction == "non-linear"                        ~  magnitude_logodds,
      TRUE                                             ~  0
    ),
    sig_label = ifelse(significant, p_stars, ""),
    # Clean feature names for display
    feature_label = gsub("_", " ", feature) |> tools::toTitleCase()
  )

# Order features: significant on more platforms first, then alphabetical
feat_order <- heat_data |>
  group_by(feature_label) |>
  summarise(n_sig = sum(significant), max_mag = max(abs(fill_val), na.rm = TRUE),
            .groups = "drop") |>
  arrange(desc(n_sig), desc(max_mag)) |>
  pull(feature_label)

heat_data$feature_label <- factor(heat_data$feature_label, levels = rev(feat_order))

p_heat <- ggplot(heat_data, aes(x = platform_label, y = feature_label)) +
  geom_tile(aes(fill = fill_val), colour = "white", linewidth = 0.3) +
  geom_text(aes(label = sig_label), size = 3, vjust = 0.5) +
  scale_fill_gradient2(
    low = "#B2182B", mid = "grey95", high = "#2166AC",
    midpoint = 0, name = "IQR magnitude\n(log-odds)",
    limits = c(-1, 1),
    oob = scales::squish
  ) +
  labs(
    title = "Cross-platform effect comparison — inclusion models",
    x = NULL, y = NULL
  ) +
  theme_thesis() +
  theme(
    axis.text.x = element_text(face = "bold"),
    axis.text.y = element_text(size = 8),
    panel.grid = element_blank()
  )

save_plot(p_heat, "effect_comparison_heatmap.png", width = 5.5, height = 8)
cat("  Saved: effect_comparison_heatmap.png\n")


cat("\n== Step 7a complete ==\n")
cat("Output: 07_cross_platform/output/effect_comparison_table.csv\n")
cat("Output: 07_cross_platform/output/effect_comparison_wide.csv\n")
cat("Output: 07_cross_platform/output/effect_comparison_heatmap.png\n")
