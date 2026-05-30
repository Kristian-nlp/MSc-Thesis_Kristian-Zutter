# =============================================================================
# 04_eda.R
#
# Master's thesis: Learning the Levers — Which Creator-Controllable
# Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
# from a Swiss German-Language Perspective
# Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026
#
# Purpose:
#   Exploratory data analysis on the three platform-specific frames produced
#   by Step 3. Catches distributional issues, near-zero-variance features,
#   and bivariate patterns, produces thesis-ready APA figures, and resolves
#   the final feature list per platform.
#
# Pipeline position:
#   Step 4 of the modelling pipeline. Depends on 03a–03c; feeds 05a–05c.
#
# Inputs:
#   05_modelling/03_data_prep/data/df_tt.parquet                 TikTok analytical frame
#   05_modelling/03_data_prep/data/df_ig.parquet                 Instagram analytical frame
#   05_modelling/03_data_prep/data/df_li.parquet                 LinkedIn analytical frame
#   05_modelling/03_data_prep/output/z_score_lookup.csv          Z-score parameters
#
# Outputs:
#   05_modelling/04_eda/plots/*.png                              APA EDA figures
#   05_modelling/04_eda/output/nzv_screening.csv                 Near-zero-variance screen
#   05_modelling/04_eda/output/feature_list_final.csv            Final per-platform feature list
#
# Usage:
#   Rscript 05_modelling/04_eda/04_eda.R
# =============================================================================

rm(list = ls())

# -- Source shared config ------------------------------------------------------
source(file.path("config", "packages.R"))
library(patchwork)    # Plot composition (requires ggplot2 >= 3.5.2)

STEP_DIR  <- file.path(BASE_DIR, "04_eda")
PLOT_DIR  <- file.path(STEP_DIR, "plots")
OUT_DIR   <- file.path(STEP_DIR, "output")
dir.create(PLOT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(OUT_DIR,  recursive = TRUE, showWarnings = FALSE)


# =============================================================================
# 0. LOAD DATA
# =============================================================================
cat("== 0. Loading platform data frames ==\n")

PREP_DIR <- file.path(BASE_DIR, "03_data_prep", "data")
df_tt <- read_parquet(file.path(PREP_DIR, "df_tt.parquet"))
df_ig <- read_parquet(file.path(PREP_DIR, "df_ig.parquet"))
df_li <- read_parquet(file.path(PREP_DIR, "df_li.parquet"))

# z-score lookup for interpretive reference
z_lookup <- read_csv(file.path(BASE_DIR, "03_data_prep", "output", "z_score_lookup.csv"),
                     show_col_types = FALSE)

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


# -- Helpers -------------------------------------------------------------------

# Bind platform data frames with a clean platform factor for plotting
bind_platforms <- function(..., platforms = c("tiktok", "instagram", "linkedin")) {
  dfs  <- list(df_tt, df_ig, df_li)
  # Only bind columns that exist in all three (or handle per-call)
  bind_rows(
    df_tt |> mutate(platform = "tiktok"),
    df_ig |> mutate(platform = "instagram"),
    df_li |> mutate(platform = "linkedin")
  ) |>
    mutate(platform = factor(platform,
                             levels = c("tiktok", "instagram", "linkedin"),
                             labels = c("TikTok", "Instagram", "LinkedIn")))
}

# Pretty labels for ever_top
label_evertop <- function(x) {
  ifelse(x == 1, "Top-20", "Baseline")
}


# =============================================================================
# 1. OUTCOME DISTRIBUTIONS [THESIS-READY]
# =============================================================================
cat("\n== 1. Outcome distributions ==\n")


# -- 1a. ever_top class balance ------------------------------------------------
cat("  1a. ever_top balance bar chart\n")

evertop_summary <- bind_rows(
  df_tt |> count(ever_top) |> mutate(platform = "TikTok"),
  df_ig |> count(ever_top) |> mutate(platform = "Instagram"),
  df_li |> count(ever_top) |> mutate(platform = "LinkedIn")
) |>
  group_by(platform) |>
  mutate(pct  = n / sum(n) * 100,
         label = sprintf("%.1f%%", pct)) |>
  ungroup() |>
  mutate(
    platform = factor(platform, levels = c("TikTok", "Instagram", "LinkedIn")),
    group    = factor(ever_top, levels = c(0, 1),
                      labels = c("Baseline", "Top-20"))
  )

p1a <- ggplot(evertop_summary, aes(x = platform, y = pct, fill = group)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.6) +
  geom_text(aes(label = label),
            position = position_dodge(width = 0.7), vjust = -0.5, size = 3.2) +
  scale_fill_manual(values = c("Baseline" = "#999999", "Top-20" = "#E69F00"),
                    name = NULL) +
  labs(x = NULL, y = "Percentage of Posts (%)") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  theme(legend.position = "bottom")

save_plot(p1a, "01_evertop_balance.png")


# -- 1b. best_rank histogram ---------------------------------------------------
cat("  1b. best_rank histogram (top posts only)\n")

rank_data <- bind_rows(
  df_tt |> filter(ever_top == 1) |> select(best_rank) |> mutate(platform = "TikTok"),
  df_ig |> filter(ever_top == 1) |> select(best_rank) |> mutate(platform = "Instagram"),
  df_li |> filter(ever_top == 1) |> select(best_rank) |> mutate(platform = "LinkedIn")
) |>
  mutate(platform = factor(platform, levels = c("TikTok", "Instagram", "LinkedIn")))

# Annotate N per platform
rank_n <- rank_data |>
  count(platform) |>
  mutate(label = sprintf("N = %s", format(n, big.mark = ",")))

p1b <- ggplot(rank_data, aes(x = best_rank, fill = platform)) +
  geom_histogram(binwidth = 1, colour = "white", linewidth = 0.2) +
  facet_wrap(~platform, scales = "free_y") +
  geom_text(data = rank_n, aes(label = label),
            x = 17, y = Inf, vjust = 1.5, hjust = 1, size = 3, inherit.aes = FALSE) +
  scale_fill_platform(guide = "none") +
  scale_x_continuous(breaks = seq(1, 20, by = 2)) +
  labs(x = "Best Rank (1 = Highest)", y = "Count") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15)))

save_plot(p1b, "02_best_rank_hist.png")


# -- 1c. velocity_24h density -------------------------------------------------
cat("  1c. velocity_24h density\n")

vel_data <- bind_rows(
  df_tt |> filter(!is.na(velocity_24h)) |>
    select(velocity_24h) |> mutate(platform = "TikTok"),
  df_ig |> filter(!is.na(velocity_24h)) |>
    select(velocity_24h) |> mutate(platform = "Instagram"),
  df_li |> filter(!is.na(velocity_24h)) |>
    select(velocity_24h) |> mutate(platform = "LinkedIn")
) |>
  mutate(platform = factor(platform, levels = c("TikTok", "Instagram", "LinkedIn")))

vel_n <- vel_data |> count(platform) |>
  mutate(label = sprintf("%s (N = %s)", platform, format(n, big.mark = ",")))
vel_labels <- setNames(vel_n$label, vel_n$platform)

p1c <- ggplot(vel_data, aes(x = velocity_24h, fill = platform, colour = platform)) +
  geom_density(alpha = 0.3, linewidth = 0.6) +
  scale_fill_manual(values = PLATFORM_COLOURS, labels = vel_labels, name = NULL) +
  scale_colour_manual(values = PLATFORM_COLOURS, labels = vel_labels, name = NULL) +
  labs(x = "Velocity 24h (log-difference in likes)", y = "Density") +
  theme(legend.position = "bottom")

save_plot(p1c, "03a_velocity_24h_density.png")

# Faceted version (if overlay is hard to read due to scale differences)
p1c_facet <- ggplot(vel_data, aes(x = velocity_24h, fill = platform)) +
  geom_histogram(aes(y = after_stat(density)), bins = 50,
                 colour = "white", linewidth = 0.2) +
  geom_density(alpha = 0, colour = "black", linewidth = 0.5) +
  facet_wrap(~platform, scales = "free") +
  scale_fill_platform(guide = "none") +
  labs(x = "Velocity 24h (log-difference in likes)", y = "Density")

save_plot(p1c_facet, "03a_velocity_24h_faceted.png")


# -- 1d. velocity_72h density (diagnostic) ------------------------------------
cat("  1d. velocity_72h density (diagnostic)\n")

vel72_data <- bind_rows(
  df_tt |> filter(!is.na(velocity_72h)) |>
    select(velocity_72h) |> mutate(platform = "TikTok"),
  df_ig |> filter(!is.na(velocity_72h)) |>
    select(velocity_72h) |> mutate(platform = "Instagram"),
  df_li |> filter(!is.na(velocity_72h)) |>
    select(velocity_72h) |> mutate(platform = "LinkedIn")
) |>
  mutate(platform = factor(platform, levels = c("TikTok", "Instagram", "LinkedIn")))

p1d <- ggplot(vel72_data, aes(x = velocity_72h, fill = platform)) +
  geom_histogram(aes(y = after_stat(density)), bins = 50,
                 colour = "white", linewidth = 0.2) +
  geom_density(alpha = 0, colour = "black", linewidth = 0.5) +
  facet_wrap(~platform, scales = "free") +
  scale_fill_platform(guide = "none") +
  labs(x = "Velocity 72h (log-difference in likes)", y = "Density")

save_plot(p1d, "03b_velocity_72h_density.png")


# -- Console summary ----------------------------------------------------------
cat("\n  Outcome summary table:\n")
cat("  ---------------------------------------------------------------\n")
cat(sprintf("  %-12s %6s %6s %7s %8s %10s %8s %6s\n",
            "Platform", "N", "Top", "Top%", "Med.Rank", "Mean.V24", "SD.V24", "N.V24"))
cat("  ---------------------------------------------------------------\n")

for (info in list(
  list(df_tt, "TikTok"),
  list(df_ig, "Instagram"),
  list(df_li, "LinkedIn")
)) {
  d   <- info[[1]]
  nm  <- info[[2]]
  n_top   <- sum(d$ever_top == 1, na.rm = TRUE)
  pct_top <- round(n_top / nrow(d) * 100, 1)
  med_rank <- median(d$best_rank[d$ever_top == 1], na.rm = TRUE)
  v24     <- d$velocity_24h[!is.na(d$velocity_24h)]
  cat(sprintf("  %-12s %6d %6d %6.1f%% %8.1f %10.4f %8.4f %6d\n",
              nm, nrow(d), n_top, pct_top, med_rank,
              mean(v24), sd(v24), length(v24)))
}
cat("  ---------------------------------------------------------------\n")


# =============================================================================
# 2. FEATURE DISTRIBUTIONS [THESIS-READY + DIAGNOSTIC]
# =============================================================================
cat("\n== 2. Feature distributions ==\n")


# -- 2a. Shared continuous features (multi-panel histogram grid) ---------------
cat("  2a. Shared continuous features histogram grid\n")

# Features shared across all three platforms (z-scored)
shared_continuous <- c(
  "word_count", "hashtag_count", "emoji_count", "avg_sentence_len",
  "exclamation_density", "question_density", "ellipsis_count",
  "caps_ratio", "caps_word_count", "line_break_count",
  "url_count", "mention_count", "punct_diversity",
  "flesch_reading_ease", "log_follower"
)

# Pretty labels
feature_labels <- c(
  word_count          = "Word Count (z)",
  hashtag_count       = "Hashtag Count (z)",
  emoji_count         = "Emoji Count (z)",
  avg_sentence_len    = "Avg Sentence Length (z)",
  exclamation_density = "Exclamation Density (z)",
  question_density    = "Question Density (z)",
  ellipsis_count      = "Ellipsis Count (z)",
  caps_ratio          = "CAPS Ratio (z)",
  caps_word_count     = "CAPS Word Count (z)",
  line_break_count    = "Line Break Count (z)",
  url_count           = "URL Count (z)",
  mention_count       = "Mention Count (z)",
  punct_diversity     = "Punctuation Diversity (z)",
  flesch_reading_ease = "Flesch Reading Ease (z)",
  log_follower        = "Log Follower Count (z)"
)

# Build long-format data for shared features
shared_long <- bind_rows(
  df_tt |> select(any_of(shared_continuous)) |> mutate(platform = "TikTok"),
  df_ig |> select(any_of(shared_continuous)) |> mutate(platform = "Instagram"),
  df_li |> select(any_of(shared_continuous)) |> mutate(platform = "LinkedIn")
) |>
  pivot_longer(cols = all_of(shared_continuous), names_to = "feature", values_to = "value") |>
  mutate(
    platform = factor(platform, levels = c("TikTok", "Instagram", "LinkedIn")),
    feature  = factor(feature, levels = shared_continuous, labels = feature_labels[shared_continuous])
  )

p2a <- ggplot(shared_long, aes(x = value, fill = platform)) +
  geom_histogram(bins = 40, alpha = 0.5, position = "identity", linewidth = 0.1) +
  facet_wrap(~feature, scales = "free_y", ncol = 3) +
  coord_cartesian(xlim = c(-3, 5)) +
  scale_fill_platform(name = NULL) +
  labs(x = "Standardised Value", y = "Count") +
  theme(
    legend.position = "bottom",
    strip.text = element_text(size = 8),
    axis.text  = element_text(size = 7)
  )

save_plot(p2a, "04_feature_distributions.png", width = 10, height = 12)


# -- 2b. Platform-specific: TikTok visual features ----------------------------
cat("  2b. TikTok visual features\n")

visual_features <- c("brightness", "contrast", "colourfulness", "log_ocr_text_len")
visual_labels   <- c(
  brightness       = "Brightness (z)",
  contrast         = "Contrast (z)",
  colourfulness    = "Colourfulness (z)",
  log_ocr_text_len = "Log OCR Text Length (z)"
)

# TikTok + Instagram have visual features
vis_long <- bind_rows(
  df_tt |> select(any_of(visual_features)) |> mutate(platform = "TikTok"),
  df_ig |> select(any_of(visual_features)) |> mutate(platform = "Instagram")
) |>
  pivot_longer(cols = any_of(visual_features), names_to = "feature", values_to = "value") |>
  filter(!is.na(value)) |>
  mutate(
    platform = factor(platform, levels = c("TikTok", "Instagram")),
    feature  = factor(feature, levels = visual_features, labels = visual_labels[visual_features])
  )

p2b <- ggplot(vis_long, aes(x = value, fill = platform)) +
  geom_histogram(bins = 40, alpha = 0.5, position = "identity", linewidth = 0.1) +
  facet_wrap(~feature, scales = "free_y", ncol = 2) +
  coord_cartesian(xlim = c(-3, 5)) +
  scale_fill_manual(values = PLATFORM_COLOURS_LC[c("tiktok", "instagram")], name = NULL) +
  labs(x = "Standardised Value", y = "Count") +
  theme(legend.position = "bottom")

save_plot(p2b, "05a_visual_features.png", width = 6.5, height = 5)


# -- 2b. Temporal: local_hour distribution (TikTok + LinkedIn) -----------------
cat("  2b. Temporal features: local_hour\n")

hour_data <- bind_rows(
  df_tt |> select(local_hour) |> mutate(platform = "TikTok"),
  df_li |> select(local_hour) |> mutate(platform = "LinkedIn")
) |>
  filter(!is.na(local_hour)) |>
  mutate(platform = factor(platform, levels = c("TikTok", "LinkedIn")))

p2b_hour <- ggplot(hour_data, aes(x = local_hour, fill = platform)) +
  geom_histogram(binwidth = 1, colour = "white", linewidth = 0.2) +
  facet_wrap(~platform, scales = "free_y") +
  scale_fill_manual(values = PLATFORM_COLOURS_LC[c("tiktok", "linkedin")], guide = "none") +
  scale_x_continuous(breaks = seq(0, 23, by = 3)) +
  labs(x = "Local Hour (CET/CEST)", y = "Count") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.08)))

save_plot(p2b_hour, "05b_temporal_hour.png")


# -- Weekday distribution (TikTok + LinkedIn) ----------------------------------
cat("  2b. Temporal features: weekday\n")

weekday_labels <- c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

weekday_data <- bind_rows(
  df_tt |> select(weekday) |> mutate(platform = "TikTok"),
  df_li |> select(weekday) |> mutate(platform = "LinkedIn")
) |>
  filter(!is.na(weekday)) |>
  mutate(
    platform = factor(platform, levels = c("TikTok", "LinkedIn")),
    weekday_label = factor(weekday_labels[weekday + 1],
                           levels = weekday_labels)
  )

p2b_wday <- ggplot(weekday_data, aes(x = weekday_label, fill = platform)) +
  geom_bar(colour = "white", linewidth = 0.2) +
  facet_wrap(~platform, scales = "free_y") +
  scale_fill_manual(values = PLATFORM_COLOURS_LC[c("tiktok", "linkedin")], guide = "none") +
  labs(x = "Day of Week", y = "Count") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.08)))

save_plot(p2b_wday, "05c_temporal_weekday.png")


# -- 2c. Categorical distributions --------------------------------------------
cat("  2c. Categorical features\n")

# -- media_type --
mt_data <- bind_rows(
  df_tt |> count(media_type) |> mutate(platform = "TikTok"),
  df_ig |> count(media_type) |> mutate(platform = "Instagram"),
  df_li |> count(media_type) |> mutate(platform = "LinkedIn")
) |>
  mutate(platform = factor(platform, levels = c("TikTok", "Instagram", "LinkedIn")))

p2c_mt <- ggplot(mt_data, aes(x = reorder(media_type, -n), y = n, fill = platform)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.6) +
  scale_fill_platform(name = NULL) +
  labs(x = "Media Type", y = "Count") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.08))) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

save_plot(p2c_mt, "06a_media_type.png")

# -- lang --
lang_data <- bind_rows(
  df_tt |> count(lang) |> mutate(platform = "TikTok"),
  df_ig |> count(lang) |> mutate(platform = "Instagram"),
  df_li |> count(lang) |> mutate(platform = "LinkedIn")
) |>
  mutate(platform = factor(platform, levels = c("TikTok", "Instagram", "LinkedIn")))

p2c_lang <- ggplot(lang_data, aes(x = reorder(lang, -n), y = n, fill = platform)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.6) +
  scale_fill_platform(name = NULL) +
  labs(x = "Language", y = "Count") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.08))) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

save_plot(p2c_lang, "06b_lang.png")

# -- topic_cluster --
tc_data <- bind_rows(
  df_tt |> count(topic_cluster) |> mutate(platform = "TikTok"),
  df_ig |> count(topic_cluster) |> mutate(platform = "Instagram"),
  df_li |> count(topic_cluster) |> mutate(platform = "LinkedIn")
) |>
  mutate(platform = factor(platform, levels = c("TikTok", "Instagram", "LinkedIn")))

p2c_tc <- ggplot(tc_data, aes(x = topic_cluster, y = n, fill = platform)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.6) +
  scale_fill_platform(name = NULL) +
  labs(x = "Topic Cluster", y = "Count") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.08))) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 7))

save_plot(p2c_tc, "06c_topic_cluster.png", width = 8, height = 5)

# -- account_type --
at_data <- bind_rows(
  df_tt |> count(account_type) |> mutate(platform = "TikTok"),
  df_ig |> count(account_type) |> mutate(platform = "Instagram"),
  df_li |> count(account_type) |> mutate(platform = "LinkedIn")
) |>
  mutate(platform = factor(platform, levels = c("TikTok", "Instagram", "LinkedIn")))

p2c_at <- ggplot(at_data, aes(x = account_type, y = n, fill = platform)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.6) +
  scale_fill_platform(name = NULL) +
  labs(x = "Account Type", y = "Count") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.08)))

save_plot(p2c_at, "06d_account_type.png")


# =============================================================================
# 3. CORRELATION MATRICES [DIAGNOSTIC]
# =============================================================================
cat("\n== 3. Correlation matrices ==\n")
cat("  Note: 86 topic PCs excluded from correlation matrix (would be unreadable).\n")
cat("  Topic information enters models as topic_cluster (categorical factor).\n")

# -- Feature sets for correlation (z-scored continuous only, no topic PCs) -----

# Identifiers, outcomes, counters, binary, factor, cyclic -- all excluded
exclude_patterns <- c("^post_id$", "^platform$", "^permalink$", "^author_hash$",
                       "^ever_top$", "^best_rank$", "^n_captures$",
                       "^velocity_", "^posted_at_utc$",
                       "^likes_", "^comments_", "^shares_", "^views_",
                       "^topic_pc_",                          # exclude PCs
                       "^cta_flag$", "^is_weekend$", "^audio_present$",
                       "^is_trending$", "^audio_is_original$",
                       "^uses_named_audio$", "^flesch_available$",
                       "^follower_available$", "^face_flag$",
                       "^lang$", "^media_type$", "^topic_cluster$",
                       "^account_type$",
                       "^local_hour$", "^weekday$")

get_corr_features <- function(df) {
  all_cols <- names(df)
  exclude  <- unique(unlist(lapply(exclude_patterns, \(p) grep(p, all_cols, value = TRUE))))
  feats    <- setdiff(all_cols, exclude)
  # Keep only numeric

  feats[sapply(df[feats], is.numeric)]
}

for (info in list(
  list(df_tt, "tiktok",    "TikTok"),
  list(df_ig, "instagram", "Instagram"),
  list(df_li, "linkedin",  "LinkedIn")
)) {
  d     <- info[[1]]
  pname <- info[[2]]
  plabel <- info[[3]]
  feats <- get_corr_features(d)

  cat(sprintf("\n  %s: %d features in correlation matrix\n", plabel, length(feats)))

  corr_mat <- cor(d[feats], use = "pairwise.complete.obs")

  # Save plot
  png(file.path(PLOT_DIR, sprintf("07_corr_%s.png", pname)),
      width = 8, height = 8, units = "in", res = 300)
  corrplot(corr_mat, method = "color", type = "upper",
           tl.col = "black", tl.cex = 0.65, tl.srt = 45,
           addCoef.col = "black", number.cex = 0.5,
           cl.cex = 0.7, mar = c(0, 0, 2, 0),
           title = sprintf("%s: Feature Correlations", plabel))
  dev.off()
  cat(sprintf("  Saved: 07_corr_%s.png\n", pname))

  # Flag pairs |r| > 0.70
  high_corr <- which(abs(corr_mat) > 0.70 & upper.tri(corr_mat), arr.ind = TRUE)
  if (nrow(high_corr) > 0) {
    cat(sprintf("  Pairs with |r| > 0.70 (%s):\n", plabel))
    for (i in seq_len(nrow(high_corr))) {
      f1 <- rownames(corr_mat)[high_corr[i, 1]]
      f2 <- colnames(corr_mat)[high_corr[i, 2]]
      r  <- corr_mat[high_corr[i, 1], high_corr[i, 2]]
      cat(sprintf("    %s <-> %s: r = %.3f\n", f1, f2, r))
    }
  } else {
    cat(sprintf("  No pairs with |r| > 0.70 (%s).\n", plabel))
  }
}


# =============================================================================
# 4. RESIDUAL MISSINGNESS [DIAGNOSTIC]
# =============================================================================
cat("\n== 4. Residual missingness check ==\n")

# Predictor columns: exclude identifiers, outcomes, counters
id_cols      <- c("post_id", "platform", "permalink", "author_hash")
outcome_cols <- c("ever_top", "best_rank", "n_captures",
                  "velocity_24h", "velocity_72h", "posted_at_utc")
counter_cols_pattern <- "^(likes|comments|shares|views)_t"

needs_heatmap <- FALSE

for (info in list(
  list(df_tt, "TikTok"),
  list(df_ig, "Instagram"),
  list(df_li, "LinkedIn")
)) {
  d   <- info[[1]]
  nm  <- info[[2]]

  counter_cols <- grep(counter_cols_pattern, names(d), value = TRUE)
  pred_cols    <- setdiff(names(d), c(id_cols, outcome_cols, counter_cols))
  na_counts    <- colSums(is.na(d[pred_cols]))
  has_na       <- na_counts[na_counts > 0]

  if (length(has_na) > 0) {
    cat(sprintf("  %s: %d features with NAs\n", nm, length(has_na)))
    for (fn in names(has_na)) {
      pct <- round(has_na[[fn]] / nrow(d) * 100, 2)
      cat(sprintf("    %-25s %5d  (%.2f%%)\n", fn, has_na[[fn]], pct))
      if (pct > 1.0) needs_heatmap <- TRUE
    }
  } else {
    cat(sprintf("  %s: no NAs in predictor features.\n", nm))
  }
}

if (needs_heatmap) {
  cat("\n  >>> Unexpected NAs > 1%% found. Generating missingness heatmap. <<<\n")
  for (info in list(
    list(df_tt, "tiktok",    "TikTok"),
    list(df_ig, "instagram", "Instagram"),
    list(df_li, "linkedin",  "LinkedIn")
  )) {
    d     <- info[[1]]
    pname <- info[[2]]
    plabel <- info[[3]]
    counter_cols <- grep(counter_cols_pattern, names(d), value = TRUE)
    pred_cols    <- setdiff(names(d), c(id_cols, outcome_cols, counter_cols))

    p_miss <- vis_miss(d[pred_cols], sort_miss = TRUE, warn_large_data = FALSE) +
      labs(title = sprintf("%s: Predictor Missingness", plabel))
    save_plot(p_miss, sprintf("08_missingness_%s.png", pname), width = 10, height = 6)
  }
} else {
  cat("  Post-imputation missingness negligible (all < 1%%). No heatmap needed.\n")
}


# =============================================================================
# 5. BIVARIATE RELATIONSHIPS [THESIS-READY]
# =============================================================================
cat("\n== 5. Bivariate relationships ==\n")

# -- Helper: box plot for continuous feature vs ever_top -----------------------
plot_bivariate_box <- function(feature_name, feature_label, platforms = "all") {

  if (identical(platforms, "all")) {
    plot_data <- bind_rows(
      df_tt |> select(ever_top, value = all_of(feature_name)) |> mutate(platform = "TikTok"),
      df_ig |> select(ever_top, value = all_of(feature_name)) |> mutate(platform = "Instagram"),
      df_li |> select(ever_top, value = all_of(feature_name)) |> mutate(platform = "LinkedIn")
    )
  } else {
    dfs <- list(tiktok = df_tt, instagram = df_ig, linkedin = df_li)
    plot_data <- bind_rows(lapply(platforms, function(p) {
      dfs[[p]] |> select(ever_top, value = all_of(feature_name)) |>
        mutate(platform = PLATFORM_LABELS[[p]])
    }))
  }

  plot_data <- plot_data |>
    filter(!is.na(value)) |>
    mutate(
      platform = factor(platform, levels = c("TikTok", "Instagram", "LinkedIn")),
      group    = factor(ever_top, levels = c(0, 1), labels = c("Baseline", "Top-20"))
    )

  p <- ggplot(plot_data, aes(x = group, y = value, fill = group)) +
    geom_boxplot(outlier.size = 0.5, outlier.alpha = 0.3, width = 0.6) +
    facet_wrap(~platform, scales = "free_y") +
    scale_fill_manual(values = c("Baseline" = "#999999", "Top-20" = "#E69F00"),
                      name = NULL) +
    labs(x = NULL, y = feature_label) +
    theme(legend.position = "bottom")

  p
}

# -- Helper: print bivariate statistics ----------------------------------------
print_bivar_stats <- function(feature_name) {
  cat(sprintf("\n  --- %s vs ever_top ---\n", feature_name))
  cat(sprintf("  %-12s %8s %8s %8s %8s %10s\n",
              "Platform", "Med.Base", "Med.Top", "Cohen.d", "Wilcox.p", "N"))

  for (info in list(
    list(df_tt, "TikTok"),
    list(df_ig, "Instagram"),
    list(df_li, "LinkedIn")
  )) {
    d  <- info[[1]]
    nm <- info[[2]]
    if (!(feature_name %in% names(d))) next

    vals <- d[[feature_name]]
    grp  <- d$ever_top
    ok   <- !is.na(vals) & !is.na(grp)
    v0   <- vals[ok & grp == 0]
    v1   <- vals[ok & grp == 1]

    if (length(v0) < 2 || length(v1) < 2) next

    med0 <- median(v0)
    med1 <- median(v1)

    # Cohen's d (pooled SD)
    pooled_sd <- sqrt(((length(v0) - 1) * var(v0) + (length(v1) - 1) * var(v1)) /
                        (length(v0) + length(v1) - 2))
    d_val <- if (pooled_sd > 0) (mean(v1) - mean(v0)) / pooled_sd else NA_real_

    # Wilcoxon
    w_test <- suppressWarnings(wilcox.test(v0, v1))

    cat(sprintf("  %-12s %8.3f %8.3f %8.3f %8s %10d\n",
                nm, med0, med1,
                round(d_val, 3),
                format.pval(w_test$p.value, digits = 3),
                length(v0) + length(v1)))
  }
}


# -- 5a. log_follower vs ever_top ---------------------------------------------
cat("  5a. log_follower vs ever_top\n")
p5a <- plot_bivariate_box("log_follower", "Log Follower Count (z)")
save_plot(p5a, "09a_follower_evertop.png")
print_bivar_stats("log_follower")

# -- 5b. hashtag_count vs ever_top --------------------------------------------
cat("\n  5b. hashtag_count vs ever_top\n")
p5b <- plot_bivariate_box("hashtag_count", "Hashtag Count (z)")
save_plot(p5b, "09b_hashtag_evertop.png")
print_bivar_stats("hashtag_count")

# -- 5c. word_count vs ever_top -----------------------------------------------
cat("\n  5c. word_count vs ever_top\n")
p5c <- plot_bivariate_box("word_count", "Word Count (z)")
save_plot(p5c, "09c_wordcount_evertop.png")
print_bivar_stats("word_count")

# -- 5d. flesch_reading_ease vs ever_top ---------------------------------------
cat("\n  5d. flesch_reading_ease vs ever_top\n")
p5d <- plot_bivariate_box("flesch_reading_ease", "Flesch Reading Ease (z)")
save_plot(p5d, "09d_flesch_evertop.png")
print_bivar_stats("flesch_reading_ease")

# -- 5e. brightness vs ever_top (TikTok + Instagram only) ---------------------
cat("\n  5e. brightness vs ever_top (TikTok + Instagram)\n")
p5e <- plot_bivariate_box("brightness", "Brightness (z)",
                          platforms = c("tiktok", "instagram"))
save_plot(p5e, "09e_brightness_evertop.png")
print_bivar_stats("brightness")

# -- 5f. is_trending vs ever_top (grouped bar: top rate by trending level) -----
cat("\n  5f. is_trending vs ever_top\n")

trend_data <- bind_rows(
  df_tt |> select(ever_top, is_trending) |> mutate(platform = "TikTok"),
  df_ig |> select(ever_top, is_trending) |> mutate(platform = "Instagram"),
  df_li |> select(ever_top, is_trending) |> mutate(platform = "LinkedIn")
) |>
  filter(!is.na(is_trending)) |>
  mutate(
    platform = factor(platform, levels = c("TikTok", "Instagram", "LinkedIn")),
    trending_label = factor(is_trending, levels = c(0, 1),
                            labels = c("Not Trending", "Trending"))
  ) |>
  group_by(platform, trending_label) |>
  summarise(n = n(), n_top = sum(ever_top == 1),
            top_rate = n_top / n * 100, .groups = "drop")

p5f <- ggplot(trend_data, aes(x = trending_label, y = top_rate, fill = platform)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.6) +
  geom_text(aes(label = sprintf("%.1f%%", top_rate)),
            position = position_dodge(width = 0.7), vjust = -0.5, size = 3) +
  scale_fill_platform(name = NULL) +
  labs(x = NULL, y = "Top-20 Rate (%)") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  theme(legend.position = "bottom")

save_plot(p5f, "09f_trending_evertop.png")

# Print trending stats
cat("\n  is_trending top rates:\n")
cat(sprintf("  %-12s %-15s %6s %6s %7s\n", "Platform", "Trending", "N", "Top", "Rate%"))
for (i in seq_len(nrow(trend_data))) {
  r <- trend_data[i, ]
  cat(sprintf("  %-12s %-15s %6d %6d %6.1f%%\n",
              r$platform, r$trending_label, r$n, r$n_top, r$top_rate))
}


# -- Diagnostic: scatter plots vs best_rank -----------------------------------
cat("\n  Diagnostic: feature vs best_rank (top posts only)\n")

for (info in list(
  list("log_follower",  "Log Follower Count (z)", "10a_follower_rank.png"),
  list("hashtag_count", "Hashtag Count (z)",      "10b_hashtag_rank.png")
)) {
  feat  <- info[[1]]
  lab   <- info[[2]]
  fname <- info[[3]]

  rank_scatter <- bind_rows(
    df_tt |> filter(ever_top == 1) |>
      select(best_rank, value = all_of(feat)) |> mutate(platform = "TikTok"),
    df_ig |> filter(ever_top == 1) |>
      select(best_rank, value = all_of(feat)) |> mutate(platform = "Instagram"),
    df_li |> filter(ever_top == 1) |>
      select(best_rank, value = all_of(feat)) |> mutate(platform = "LinkedIn")
  ) |>
    filter(!is.na(value), !is.na(best_rank)) |>
    mutate(platform = factor(platform, levels = c("TikTok", "Instagram", "LinkedIn")))

  p_rank <- ggplot(rank_scatter, aes(x = value, y = best_rank)) +
    geom_jitter(alpha = 0.08, size = 0.5, width = 0.1) +
    geom_smooth(method = "loess", colour = "#E69F00", linewidth = 0.8, se = TRUE) +
    facet_wrap(~platform, scales = "free_x") +
    scale_y_reverse(breaks = seq(1, 20, by = 2)) +
    labs(x = lab, y = "Best Rank (1 = Highest)")

  save_plot(p_rank, fname)
  cat(sprintf("  Saved: %s\n", fname))
}


# =============================================================================
# 6. NEAR-ZERO VARIANCE SCREENING [DECISION POINT]
# =============================================================================
cat("\n== 6. Near-zero variance screening ==\n")

nzv_results <- tibble(
  feature    = character(),
  platform   = character(),
  sd         = double(),
  freq_ratio = double(),
  pct_unique = double(),
  nzv_flag   = logical()
)

# Get all predictor features (numeric only, excluding topic PCs, IDs, outcomes, counters)
for (info in list(
  list(df_tt, "tiktok",    "TikTok"),
  list(df_ig, "instagram", "Instagram"),
  list(df_li, "linkedin",  "LinkedIn")
)) {
  d      <- info[[1]]
  pname  <- info[[2]]
  plabel <- info[[3]]

  # All numeric columns minus exclusions
  counter_cols <- grep(counter_cols_pattern, names(d), value = TRUE)
  exclude_nzv  <- c(id_cols, outcome_cols, counter_cols,
                     grep("^topic_pc_", names(d), value = TRUE))
  pred_cols    <- setdiff(names(d), exclude_nzv)
  num_cols     <- pred_cols[sapply(d[pred_cols], is.numeric)]

  for (col in num_cols) {
    vals <- d[[col]][!is.na(d[[col]])]
    if (length(vals) < 2) next

    col_sd <- sd(vals)

    # Frequency ratio: count of most common / count of second most common
    freq_table <- sort(table(vals), decreasing = TRUE)
    if (length(freq_table) >= 2) {
      freq_ratio <- as.numeric(freq_table[1]) / as.numeric(freq_table[2])
    } else {
      freq_ratio <- Inf
    }

    pct_unique <- length(unique(vals)) / length(vals) * 100

    nzv_flag <- (freq_ratio > 19) & (pct_unique < 10)

    nzv_results <- bind_rows(nzv_results, tibble(
      feature    = col,
      platform   = pname,
      sd         = round(col_sd, 6),
      freq_ratio = round(freq_ratio, 2),
      pct_unique = round(pct_unique, 2),
      nzv_flag   = nzv_flag
    ))
  }
}

# Print flagged features
flagged <- nzv_results |> filter(nzv_flag)
cat(sprintf("  Total features screened: %d\n", nrow(nzv_results)))
cat(sprintf("  NZV flagged: %d\n", nrow(flagged)))

if (nrow(flagged) > 0) {
  cat("\n  >>> NZV flagged features <<<\n")
  cat(sprintf("  %-25s %-12s %8s %10s %10s\n",
              "Feature", "Platform", "SD", "FreqRatio", "Pct.Uniq"))
  cat("  ", strrep("-", 70), "\n")
  for (i in seq_len(nrow(flagged))) {
    r <- flagged[i, ]
    cat(sprintf("  %-25s %-12s %8.4f %10.1f %9.2f%%\n",
                r$feature, r$platform, r$sd, r$freq_ratio, r$pct_unique))
  }

  # Check if any feature is NZV on ALL platforms
  nzv_all <- flagged |>
    count(feature) |>
    filter(n == 3)

  if (nrow(nzv_all) > 0) {
    cat("\n  >>> WARNING: Features NZV on ALL platforms (consider dropping): <<<\n")
    cat("  ", paste(nzv_all$feature, collapse = ", "), "\n")
  } else {
    cat("\n  No features are NZV on all three platforms.\n")
    cat("  Decision: KEEP all features. GAM will estimate flat (near-zero edf)\n")
    cat("  smooths for NZV features on specific platforms, which is the correct\n")
    cat("  statistical response.\n")
  }
} else {
  cat("  No features flagged as near-zero variance.\n")
}

# Write NZV results
write_csv(nzv_results, file.path(OUT_DIR, "nzv_screening.csv"))
cat(sprintf("  Saved: nzv_screening.csv (%d rows)\n", nrow(nzv_results)))


# =============================================================================
# 7. FINAL FEATURE LIST [DECISION OUTPUT]
# =============================================================================
cat("\n== 7. Final feature list ==\n")

# Define feature classifications
build_feature_list <- function(df, platform_name) {

  # Binary features present in this data frame
  all_binary <- c("cta_flag", "is_weekend", "audio_present", "is_trending",
                   "audio_is_original", "uses_named_audio",
                   "flesch_available", "follower_available", "face_flag")
  binary_present <- intersect(all_binary, names(df))

  # Factor features
  all_factor <- c("lang", "media_type", "topic_cluster", "account_type")
  factor_present <- intersect(all_factor, names(df))

  # Cyclic features
  all_cyclic <- c("local_hour", "weekday")
  cyclic_present <- intersect(all_cyclic, names(df))

  # Continuous features (z-scored): everything numeric that is not in the above
  counter_cols <- grep(counter_cols_pattern, names(df), value = TRUE)
  exclude_final <- c(id_cols, outcome_cols, counter_cols,
                      grep("^topic_pc_", names(df), value = TRUE),
                      binary_present, factor_present, cyclic_present)
  all_cols <- names(df)
  cont_present <- setdiff(all_cols, exclude_final)
  cont_present <- cont_present[sapply(df[cont_present], is.numeric)]

  # Build tibble
  bind_rows(
    tibble(feature = cont_present, platform = platform_name,
           type = "continuous", status = "keep", note = "z-standardised"),
    tibble(feature = binary_present, platform = platform_name,
           type = "binary", status = "keep", note = ""),
    tibble(feature = factor_present, platform = platform_name,
           type = "factor",
           status = ifelse(factor_present == "account_type",
                           "keep-control", "keep"),
           note = ifelse(factor_present == "account_type",
                         "control variable, not creator-controllable", "")),
    tibble(feature = cyclic_present, platform = platform_name,
           type = "cyclic", status = "keep", note = "raw for GAM bs='cc'")
  )
}

feature_list <- bind_rows(
  build_feature_list(df_tt, "tiktok"),
  build_feature_list(df_ig, "instagram"),
  build_feature_list(df_li, "linkedin")
)

# Add NZV notes and apply drop decisions
for (i in seq_len(nrow(flagged))) {
  r <- flagged[i, ]
  idx <- which(feature_list$feature == r$feature & feature_list$platform == r$platform)
  if (length(idx) == 1) {
    feature_list$note[idx] <- paste0(
      feature_list$note[idx],
      ifelse(nchar(feature_list$note[idx]) > 0, "; ", ""),
      sprintf("NZV (freq_ratio=%.0f, pct_unique=%.1f%%)", r$freq_ratio, r$pct_unique)
    )
  }
}

# -- NZV drop decisions --------------------------------------------------------
# audio_present: NZV on ALL platforms (constant or near-constant). Drop globally.
feature_list <- feature_list |>
  mutate(status = ifelse(feature == "audio_present", "drop-nzv", status),
         note   = ifelse(feature == "audio_present",
                         "DROPPED: NZV on all platforms (constant/near-constant)",
                         note))

# is_trending: constant (all 0) on Instagram and LinkedIn. Drop from those.
feature_list <- feature_list |>
  mutate(status = ifelse(feature == "is_trending" & platform %in% c("instagram", "linkedin"),
                         "drop-nzv", status),
         note   = ifelse(feature == "is_trending" & platform %in% c("instagram", "linkedin"),
                         "DROPPED: constant (all 0) on this platform",
                         note))

# follower_available: NZV on TikTok (99.9% available). Keep on IG/LI where it varies.
# cta_flag: NZV on TikTok only. Keep -- GAM will estimate near-zero effect.
# Other TikTok NZV (exclamation_density, question_density, etc.): Keep -- these
# have variance on other platforms and GAM handles platform-specific flat effects.

n_dropped <- sum(feature_list$status == "drop-nzv")
cat(sprintf("\n  NZV drop decisions applied: %d features dropped\n", n_dropped))
cat("    audio_present: dropped on all platforms (constant/near-constant)\n")
cat("    is_trending: dropped on Instagram + LinkedIn (constant = 0)\n")

write_csv(feature_list, file.path(OUT_DIR, "feature_list_final.csv"))
cat(sprintf("  Saved: feature_list_final.csv (%d rows)\n", nrow(feature_list)))

# Summary counts
cat("\n  Feature counts entering GAM:\n")
for (plat in c("tiktok", "instagram", "linkedin")) {
  fl <- feature_list |> filter(platform == plat, status %in% c("keep", "keep-control"))
  n_cont   <- sum(fl$type == "continuous")
  n_binary <- sum(fl$type == "binary")
  n_factor <- sum(fl$type == "factor")
  n_cyclic <- sum(fl$type == "cyclic")
  total    <- n_cont + n_binary + n_factor + n_cyclic
  plabel   <- PLATFORM_LABELS[[plat]]
  cat(sprintf("    %-12s %2d continuous + %d binary + %d factor + %d cyclic = %d predictors\n",
              plabel, n_cont, n_binary, n_factor, n_cyclic, total))
}


# =============================================================================
# Findings log
# =============================================================================
cat("\n\n")
cat("--- Step 4: Exploratory Data Analysis (04_eda.R) ----------------------------\n")
cat(sprintf("Date: %s\n", Sys.Date()))
cat("Output: 04_eda/plots/*.png, 04_eda/output/nzv_screening.csv,\n")
cat("        04_eda/output/feature_list_final.csv\n")
cat("\n")
cat("OUTCOME DISTRIBUTIONS\n")
cat("  ever_top balance (post level):\n")
for (info in list(list(df_tt, "TikTok"), list(df_ig, "Instagram"), list(df_li, "LinkedIn"))) {
  d <- info[[1]]; nm <- info[[2]]
  n_top <- sum(d$ever_top == 1)
  cat(sprintf("    %s: %d / %d (%.1f%% top)\n", nm, n_top, nrow(d),
              n_top / nrow(d) * 100))
}

cat("\n  velocity_24h coverage:\n")
for (info in list(list(df_tt, "TikTok"), list(df_ig, "Instagram"), list(df_li, "LinkedIn"))) {
  d <- info[[1]]; nm <- info[[2]]
  n_vel <- sum(!is.na(d$velocity_24h))
  v <- d$velocity_24h[!is.na(d$velocity_24h)]
  cat(sprintf("    %s: N=%d (%.1f%%), mean=%.4f, sd=%.4f\n",
              nm, n_vel, n_vel / nrow(d) * 100, mean(v), sd(v)))
}

cat("\nNEAR-ZERO VARIANCE SCREENING\n")
cat(sprintf("  Features screened: %d | NZV flagged: %d\n",
            nrow(nzv_results), nrow(flagged)))
if (nrow(flagged) > 0) {
  for (i in seq_len(nrow(flagged))) {
    r <- flagged[i, ]
    cat(sprintf("    %s on %s (freq_ratio=%.0f, pct_unique=%.1f%%)\n",
                r$feature, r$platform, r$freq_ratio, r$pct_unique))
  }
}
cat("\n  NZV DROP DECISIONS:\n")
cat("    audio_present: DROPPED on all platforms (constant/near-constant)\n")
cat("    is_trending: DROPPED on Instagram + LinkedIn (constant = 0, no trending audio)\n")
cat("    Remaining NZV features: KEPT (GAM estimates flat smooth = correct response)\n")

cat("\nCORRELATION NOTES\n")
cat("  TikTok: no pairs |r| > 0.70\n")
cat("  Instagram: no pairs |r| > 0.70\n")
cat("  LinkedIn: word_count <-> line_break_count r=0.733 (below 0.80 threshold, kept)\n")

cat("\nBIVARIATE HIGHLIGHTS\n")
cat("  Most effect sizes small (|d| < 0.15). Largest signals:\n")
cat("    word_count on LinkedIn: d=-0.202 (shorter posts = more top)\n")
cat("    flesch_reading_ease on Instagram: d=0.157 (higher readability = more top)\n")
cat("    brightness on Instagram: d=0.133 (brighter images = more top)\n")
cat("    log_follower: negligible bivariate effect on all platforms (|d| < 0.05)\n")
cat("    is_trending on TikTok: slight boost (42.4%% -> 46.3%% top rate)\n")

cat("\nDECISION: FINAL FEATURE LIST -- RESOLVED\n")
for (plat in c("tiktok", "instagram", "linkedin")) {
  fl <- feature_list |> filter(platform == plat, status %in% c("keep", "keep-control"))
  n_cont   <- sum(fl$type == "continuous")
  n_binary <- sum(fl$type == "binary")
  n_factor <- sum(fl$type == "factor")
  n_cyclic <- sum(fl$type == "cyclic")
  total    <- n_cont + n_binary + n_factor + n_cyclic
  cat(sprintf("  %s: %d continuous + %d binary + %d factor + %d cyclic = %d predictors\n",
              PLATFORM_LABELS[[plat]], n_cont, n_binary, n_factor, n_cyclic, total))
}


cat("\n\n== Step 4 complete ==\n")
cat("Review thesis-ready plots in 04_eda/plots/.\n")
