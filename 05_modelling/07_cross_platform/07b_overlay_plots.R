# =============================================================================
# 07b_overlay_plots.R
#
# Master's thesis: Learning the Levers — Which Creator-Controllable
# Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
# from a Swiss German-Language Perspective
# Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026
#
# Purpose:
#   Overlays partial-effect curves from the inclusion GAMs across platforms
#   on shared axes. Visual centrepiece of the cross-platform analysis (RQ2):
#   shows where platforms agree (curves overlap) and where they diverge
#   (curves separate or invert). Uses gratia::smooth_estimates() to extract
#   smooth predictions with platform-coloured lines and 95% ribbons.
#
# Pipeline position:
#   Step 7b of the modelling pipeline. Depends on Steps 1–6; feeds the
#   cross-platform thesis figures in Step 10.
#
# Inputs:
#   05_modelling/03_data_prep/data/df_tt.parquet                   TikTok analytical frame
#   05_modelling/03_data_prep/data/df_ig.parquet                   Instagram analytical frame
#   05_modelling/03_data_prep/data/df_li.parquet                   LinkedIn analytical frame
#   05_modelling/05_gam/models/m_{tt,ig,li}_inclusion.rds          Inclusion GAMs
#
# Outputs:
#   05_modelling/07_cross_platform/output/overlay_*.png            Individual overlays
#   05_modelling/07_cross_platform/output/overlay_panel.png        Combined thesis figure
#
# Usage:
#   Rscript 05_modelling/07_cross_platform/07b_overlay_plots.R
# =============================================================================

rm(list = ls())

# -- Source shared config ------------------------------------------------------
source(file.path("config", "packages.R"))
library(gratia)
library(patchwork)

STEP_DIR  <- file.path(BASE_DIR, "07_cross_platform")
OUT_DIR   <- file.path(STEP_DIR, "output")
PLOT_DIR  <- OUT_DIR
GAM_DIR   <- file.path(BASE_DIR, "05_gam", "models")
PREP_DIR  <- file.path(BASE_DIR, "03_data_prep", "data")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)


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

# Ensure factors
factor_vars <- c("lang", "media_type", "topic_cluster", "account_type")
for (v in factor_vars) {
  if (v %in% names(df_tt)) df_tt[[v]] <- as.factor(df_tt[[v]])
  if (v %in% names(df_ig)) df_ig[[v]] <- as.factor(df_ig[[v]])
  if (v %in% names(df_li)) df_li[[v]] <- as.factor(df_li[[v]])
}


# =============================================================================
# 2. IDENTIFY SHARED SMOOTH FEATURES
# =============================================================================

cat("\n== 2. Identifying shared smooth features ==\n")

# Extract smooth term names from each model
get_smooth_vars <- function(model) {
  sm <- summary(model)$s.table
  if (is.null(sm) || nrow(sm) == 0) return(character())
  sapply(rownames(sm), function(term) gsub("^s\\(([^,)]+).*\\)$", "\\1", term),
         USE.NAMES = FALSE)
}

smooths_tt <- get_smooth_vars(m_tt)
smooths_ig <- get_smooth_vars(m_ig)
smooths_li <- get_smooth_vars(m_li)

cat(sprintf("  TikTok smooths:    %d terms\n", length(smooths_tt)))
cat(sprintf("  Instagram smooths: %d terms\n", length(smooths_ig)))
cat(sprintf("  LinkedIn smooths:  %d terms\n", length(smooths_li)))

# Find features that appear as smooths on 2+ platforms
all_smooth_features <- tibble(
  feature  = c(smooths_tt, smooths_ig, smooths_li),
  platform = c(rep("tiktok", length(smooths_tt)),
               rep("instagram", length(smooths_ig)),
               rep("linkedin", length(smooths_li)))
)

shared_smooths <- all_smooth_features |>
  distinct(feature, platform) |>
  count(feature, name = "n_platforms") |>
  filter(n_platforms >= 2) |>
  arrange(desc(n_platforms), feature)

cat(sprintf("  Shared smooth features (2+ platforms): %d\n", nrow(shared_smooths)))
for (i in seq_len(nrow(shared_smooths))) {
  feat <- shared_smooths$feature[i]
  plats <- all_smooth_features |>
    filter(feature == feat) |>
    pull(platform) |>
    unique()
  cat(sprintf("    %s (%d): %s\n", feat, shared_smooths$n_platforms[i],
              paste(plats, collapse = ", ")))
}


# =============================================================================
# 3. SELECT TOP FEATURES FOR OVERLAY PLOTS
# =============================================================================

cat("\n== 3. Selecting features for overlay plots ==\n")

# Prioritise: features significant (p < 0.05) on at least 1 platform,
# available as a smooth on 2+ platforms.
# Then fill to 8 with the most interesting non-significant shared smooths.

get_smooth_pvalue <- function(model, feature) {
  sm <- summary(model)$s.table
  if (is.null(sm)) return(NA_real_)
  term_row <- grep(paste0("^s\\(", feature, "[,)]"), rownames(sm))
  if (length(term_row) == 0) return(NA_real_)
  sm[term_row[1], "p-value"]
}

# Build significance table for shared smooth features
sig_table <- shared_smooths |>
  rowwise() |>
  mutate(
    p_tt = get_smooth_pvalue(m_tt, feature),
    p_ig = get_smooth_pvalue(m_ig, feature),
    p_li = get_smooth_pvalue(m_li, feature),
    min_p = min(c(p_tt, p_ig, p_li), na.rm = TRUE),
    n_sig = sum(c(p_tt < 0.05, p_ig < 0.05, p_li < 0.05), na.rm = TRUE)
  ) |>
  ungroup() |>
  arrange(desc(n_sig), min_p)

cat("  Significance across platforms:\n")
for (i in seq_len(nrow(sig_table))) {
  cat(sprintf("    %-25s  TT=%.3f  IG=%.3f  LI=%.3f  (sig on %d)\n",
              sig_table$feature[i],
              sig_table$p_tt[i],
              sig_table$p_ig[i],
              sig_table$p_li[i],
              sig_table$n_sig[i]))
}

# Select top 8 features
overlay_features <- sig_table |>
  head(8) |>
  pull(feature)

cat(sprintf("\n  Selected %d features for overlay plots:\n", length(overlay_features)))
cat(paste0("    ", overlay_features, collapse = "\n"), "\n")


# =============================================================================
# 4. EXTRACT SMOOTH ESTIMATES WITH gratia
# =============================================================================

cat("\n== 4. Extracting smooth estimates ==\n")

# Helper: extract smooth estimates for one feature from one model
extract_smooth <- function(model, feature, platform_name, n_points = 200) {
  # Check if this smooth exists in the model
  sm_names <- get_smooth_vars(model)
  if (!(feature %in% sm_names)) return(NULL)

  # Get the smooth term label as it appears in the model
  sm_table <- summary(model)$s.table
  term_idx <- grep(paste0("^s\\(", feature, "[,)]"), rownames(sm_table))
  if (length(term_idx) == 0) return(NULL)
  smooth_label <- rownames(sm_table)[term_idx[1]]

  # Extract smooth estimates (select= replaces deprecated smooth= in gratia >=0.8.9.9)
  se <- tryCatch(
    smooth_estimates(model, select = smooth_label, n = n_points),
    error = function(e) {
      cat(sprintf("  Warning: smooth_estimates failed for %s on %s: %s\n",
                  feature, platform_name, e$message))
      NULL
    }
  )
  if (is.null(se)) return(NULL)

  # Rename the feature column and standardise output
  se_out <- se |>
    mutate(
      platform = platform_name,
      feature_name = feature,
      x = .data[[feature]],
      y = .estimate,
      se = .se,
      ci_lo = .estimate - 1.96 * .se,
      ci_hi = .estimate + 1.96 * .se
    ) |>
    select(platform, feature_name, x, y, se, ci_lo, ci_hi)

  se_out
}

# Extract for all selected features across all platforms
all_estimates <- list()

for (feat in overlay_features) {
  cat(sprintf("  Processing: %s\n", feat))

  est_tt <- extract_smooth(m_tt, feat, "TikTok")
  est_ig <- extract_smooth(m_ig, feat, "Instagram")
  est_li <- extract_smooth(m_li, feat, "LinkedIn")

  combined <- bind_rows(est_tt, est_ig, est_li)
  if (nrow(combined) > 0) {
    all_estimates[[feat]] <- combined
  }
}

cat(sprintf("  Extracted smooths for %d features\n", length(all_estimates)))


# =============================================================================
# 5. GENERATE INDIVIDUAL OVERLAY PLOTS
# =============================================================================

cat("\n== 5. Generating overlay plots ==\n")

# Pretty feature labels for axis titles
feature_labels <- c(
  word_count = "Word Count (z)",
  hashtag_count = "Hashtag Count (z)",
  emoji_count = "Emoji Count (z)",
  avg_sentence_len = "Avg. Sentence Length (z)",
  exclamation_density = "Exclamation Density (z)",
  question_density = "Question Density (z)",
  ellipsis_count = "Ellipsis Count (z)",
  caps_ratio = "Caps Ratio (z)",
  caps_word_count = "Caps Word Count (z)",
  line_break_count = "Line Break Count (z)",
  url_count = "URL Count (z)",
  mention_count = "Mention Count (z)",
  punct_diversity = "Punctuation Diversity (z)",
  flesch_reading_ease = "Flesch Reading Ease (z)",
  brightness = "Brightness (z)",
  contrast = "Contrast (z)",
  colourfulness = "Colourfulness (z)",
  log_post_age = "Log Post Age (z)",
  log_follower = "Log Follower Count (z)",
  log_ocr_text_len = "Log OCR Text Length (z)",
  local_hour = "Local Hour",
  weekday = "Weekday"
)

# Helper: create a single overlay plot
make_overlay_plot <- function(est_df, feat_name) {
  x_label <- ifelse(feat_name %in% names(feature_labels),
                    feature_labels[[feat_name]], feat_name)

  # Determine which platforms are present
  plats_present <- unique(est_df$platform)
  colours_use <- PLATFORM_COLOURS[plats_present]

  # Compute y-axis limits from point estimates + the tightest platform's CIs.
  # LinkedIn's SEs can be ±100,000 due to convergence issues. For plots with
  # only TikTok + LinkedIn, even the median SE is enormous. Fix: use the
  # *minimum* per-platform median SE so the well-estimated platform (TikTok)
  # drives the scale, not the unstable one (LinkedIn).
  y_range <- range(est_df$y, na.rm = TRUE)
  min_platform_se <- est_df |>
    group_by(platform) |>
    summarise(med_se = median(se, na.rm = TRUE), .groups = "drop") |>
    pull(med_se) |>
    min()
  y_buffer <- max(abs(diff(y_range)) * 0.3, min_platform_se * 1.96 * 1.5, 0.1)
  y_lims <- c(y_range[1] - y_buffer, y_range[2] + y_buffer)

  p <- ggplot(est_df, aes(x = x, y = y, colour = platform, fill = platform)) +
    geom_ribbon(aes(ymin = ci_lo, ymax = ci_hi), alpha = 0.15, colour = NA) +
    geom_line(linewidth = 0.8) +
    geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50",
               linewidth = 0.3) +
    scale_colour_manual(values = colours_use, name = "Platform") +
    scale_fill_manual(values = colours_use, name = "Platform") +
    coord_cartesian(ylim = y_lims) +
    labs(
      x = x_label,
      y = "Partial effect (log-odds)",
      title = feat_name
    ) +
    theme(legend.position = "bottom")

  p
}

# Generate and save individual plots
plot_list <- list()

for (feat in names(all_estimates)) {
  est_df <- all_estimates[[feat]]
  p <- make_overlay_plot(est_df, feat)
  plot_list[[feat]] <- p

  # Save individual
  save_plot(p, paste0("overlay_", feat, ".png"), width = 6.5, height = 4.5)
}

cat(sprintf("  Saved %d individual overlay plots\n", length(plot_list)))


# =============================================================================
# 6. COMBINED PANEL FIGURE (thesis centrepiece)
# =============================================================================

cat("\n== 6. Building combined panel figure ==\n")

if (length(plot_list) > 0) {
  # Remove individual legends, add shared legend at bottom
  plots_no_legend <- lapply(plot_list, function(p) p + theme(legend.position = "none"))

  # Determine grid layout
  n_plots <- length(plots_no_legend)
  n_cols <- 2
  n_rows <- ceiling(n_plots / n_cols)

  # Add shared legend via patchwork guide collection
  # Keep one plot with legend, then use patchwork to collect guides
  p_final <- wrap_plots(
    c(list(plot_list[[1]]),                             # first plot keeps legend
      lapply(plot_list[-1], function(p)                 # rest: no legend
        p + theme(legend.position = "none"))),
    ncol = n_cols
  ) +
    plot_layout(guides = "collect") +
    plot_annotation(
      title = "Cross-Platform Partial Effects (Inclusion GAM)",
      subtitle = "Shared smooth terms with 95% confidence ribbons",
      theme = theme(
        plot.title = element_text(size = 13, face = "bold", hjust = 0),
        plot.subtitle = element_text(size = 11, hjust = 0),
        legend.position = "bottom"
      )
    )

  plot_height <- max(8, n_rows * 4)
  save_plot(p_final, "overlay_panel.png", width = 12, height = plot_height)
  cat(sprintf("  Saved: overlay_panel.png (%d x %d layout)\n", n_rows, n_cols))
}


# =============================================================================
# 7. CONSOLE SUMMARY
# =============================================================================

cat("\n\n== OVERLAY PLOT SUMMARY ==\n")
cat("==========================\n\n")

cat(sprintf("Features plotted: %d\n", length(all_estimates)))
models_lookup <- list(tiktok = m_tt, instagram = m_ig, linkedin = m_li)
for (feat in names(all_estimates)) {
  est_df <- all_estimates[[feat]]
  plats <- unique(est_df$platform)

  # Get p-values for annotation
  p_vals <- c()
  for (pl in plats) {
    pname <- tolower(pl)
    pv <- get_smooth_pvalue(models_lookup[[pname]], feat)
    p_vals <- c(p_vals, sprintf("%s p=%.3f%s",
                                pl, pv,
                                ifelse(!is.na(pv) && pv < 0.05, "*", "")))
  }
  cat(sprintf("  %s (%s): %s\n", feat,
              paste(plats, collapse = "/"),
              paste(p_vals, collapse = ", ")))
}

cat("\n== Step 7b complete ==\n")
cat("Output: 07_cross_platform/output/overlay_*.png\n")
cat("Output: 07_cross_platform/output/overlay_panel.png\n")
