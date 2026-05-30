# =============================================================================
# fig_05_cross_platform_velocity.R
#
# Master's thesis: Learning the Levers — Which Creator-Controllable
# Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
# from a Swiss German-Language Perspective
# Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026
#
# Purpose:
#   Produces Chapter 4 Figure 5: two-panel cross-platform comparison of
#   velocity effects on the TikTok and LinkedIn 24-hour engagement-velocity
#   GAMs. Instagram is excluded (no IG velocity model; Decision 37). Panel A
#   overlays s(log_post_age) smooths for TT vs LI on a raw-hours x-axis and
#   a velocity-multiplier (exp(partial)) y-axis. Panel B is an is_weekend
#   coefficient forest plot showing percent change in 24-hour engagement
#   velocity (exp(beta) - 1) with 95% confidence intervals.
#
# Pipeline position:
#   Thesis figure builder; reads the fitted TT and LI velocity GAMs and the
#   shared plotting theme. Cross-checks is_weekend against the results
#   register if present.
#
# Inputs:
#   05_modelling/config/theme_thesis.R                                    theme_thesis() and PLATFORM_COLOURS
#   05_modelling/05_gam/models/m_tt_velocity_24h.rds                       TikTok 24-hour velocity GAM
#   05_modelling/05_gam/models/m_li_velocity_24h.rds                       LinkedIn 24-hour velocity GAM
#   05_modelling/01_build_analytical_table/data/df_post.parquet           per-platform log_post_age standardisation
#   05_modelling/11_data_results/_ground_truth/results_register.csv        is_weekend cross-check (optional)
#
# Outputs:
#   05_modelling/10_figures/output/figure_r5_cross-comparison.png  Chapter 4 Figure 5
#
# Usage:
#   Rscript 05_modelling/10_figures/R/fig_05_cross_platform_velocity.R
# =============================================================================

suppressPackageStartupMessages({
  library(mgcv)
  library(arrow)
  library(ggplot2)
  library(patchwork)
  library(dplyr)
  library(gratia)
  library(readr)
})

# -- Paths --------------------------------------------------------------------
REPO     <- here::here()
BASE_DIR <- file.path(REPO, "05_modelling")
THEME    <- file.path(BASE_DIR, "config", "theme_thesis.R")
OUT_DIR  <- file.path(BASE_DIR, "10_figures", "output")
OUT_PNG  <- file.path(OUT_DIR, "figure_r5_cross-comparison.png")
OUT_PDF  <- file.path(OUT_DIR, "fig_05_cross_platform_velocity.pdf")
REGISTER <- file.path(REPO, "05_modelling", "11_data_results", "_ground_truth", "results_register.csv")

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# -- Theme + platform palette (read from shared config) ----------------------
stopifnot("theme_thesis.R missing" = file.exists(THEME))
source(THEME)            # defines theme_thesis(), PLATFORM_COLOURS
TT_COL <- unname(PLATFORM_COLOURS[["TikTok"]])
LI_COL <- unname(PLATFORM_COLOURS[["LinkedIn"]])
theme_set(theme_thesis(base_size = 9))

# -- Discover model paths -----------------------------------------------------
find_model <- function(name) {
  candidates <- c(
    file.path(BASE_DIR, "05_gam", "models",      paste0(name, ".rds")),
    file.path(BASE_DIR, "05_gam", "sensitivity", paste0(name, ".rds"))
  )
  hit <- candidates[file.exists(candidates)]
  if (length(hit) == 0) stop("Model RDS not found: ", name, call. = FALSE)
  hit[1]
}
M_TT_PATH <- find_model("m_tt_velocity_24h")
M_LI_PATH <- find_model("m_li_velocity_24h")

m_tt <- readRDS(M_TT_PATH)
m_li <- readRDS(M_LI_PATH)
stopifnot(family(m_tt)$family == "Gamma", family(m_tt)$link == "log",
          family(m_li)$family == "Gamma", family(m_li)$link == "log")

# -- Recover log_post_age z-standardisation per platform ---------------------
# log_post_age = z-score of log1p(post_age_hours), computed per-platform
# (same recipe as figure_tt_velocity_partial_effects.R lines 77-93).
DF_POST <- file.path(BASE_DIR, "01_build_analytical_table", "data",
                     "df_post.parquet")
stopifnot("df_post.parquet missing" = file.exists(DF_POST))
df_post <- arrow::read_parquet(DF_POST)
get_pa_params <- function(plat) {
  s <- df_post |> dplyr::filter(platform == plat,
                                !is.na(post_age_hours))
  lpa <- log1p(s$post_age_hours)
  list(mu = mean(lpa, na.rm = TRUE),
       sd = stats::sd(lpa, na.rm = TRUE),
       n  = nrow(s))
}
pp_tt <- get_pa_params("tiktok")
pp_li <- get_pa_params("linkedin")
bt_hours <- function(z, params) expm1(z * params$sd + params$mu)

cat("\n========================================================================\n")
cat("== fig_05_cross_platform_velocity.R\n")
cat("========================================================================\n")

# =============================================================================
# Sanity check 1 -- platform colours and source of palette
# =============================================================================
cat("\n[Sanity 1] Platform colours read from:\n")
cat(sprintf("  source: %s\n", THEME))
cat(sprintf("  TikTok    = %s\n", TT_COL))
cat(sprintf("  LinkedIn  = %s\n", LI_COL))

cat("\n[Models]\n")
cat(sprintf("  TT velocity_t24: %s  (N=%d)\n", M_TT_PATH, nrow(m_tt$model)))
cat(sprintf("  LI velocity_t24: %s  (N=%d)\n", M_LI_PATH, nrow(m_li$model)))

cat("\n[log_post_age z-standardisation parameters (recovered from df_post)]\n")
cat(sprintf("  TikTok    mu=%.4f  sd=%.4f  N=%d  (log1p(hours))\n",
            pp_tt$mu, pp_tt$sd, pp_tt$n))
cat(sprintf("  LinkedIn  mu=%.4f  sd=%.4f  N=%d  (log1p(hours))\n",
            pp_li$mu, pp_li$sd, pp_li$n))

# =============================================================================
# Panel A:  s(log_post_age) smooths overlaid, raw-hours x-axis, multiplier y
# =============================================================================
build_pa_curve <- function(model, params, label) {
  z_in <- model$model$log_post_age
  qs   <- as.numeric(quantile(z_in, c(0.01, 0.99), na.rm = TRUE))  # P1-P99, per V5 figure audit Issue H option 2
  est  <- gratia::smooth_estimates(model, select = "s(log_post_age)",
                                   n = 400) |>
    dplyr::filter(log_post_age >= qs[1], log_post_age <= qs[2])
  est$mult  <- exp(est$.estimate)
  est$lo    <- exp(est$.estimate - 1.96 * est$.se)
  est$hi    <- exp(est$.estimate + 1.96 * est$.se)
  est$hours <- bt_hours(est$log_post_age, params)
  est$platform <- label
  est
}
sm_tt <- build_pa_curve(m_tt, pp_tt, "TikTok")
sm_li <- build_pa_curve(m_li, pp_li, "LinkedIn")
sm_both <- bind_rows(sm_tt, sm_li) |>
  mutate(platform = factor(platform, levels = c("TikTok", "LinkedIn")))

# Multiplier at the median log_post_age of each platform's training frame.
# Look the smooth value up from the dense smooth_estimates curve we already
# built rather than re-evaluating gratia::smooth_estimates with a custom data
# frame (which would require synthesising every other predictor).
multiplier_at_median <- function(curve_df, model, params, label) {
  z_med <- median(model$model$log_post_age, na.rm = TRUE)
  i     <- which.min(abs(curve_df$log_post_age - z_med))
  list(label = label, z_med = z_med,
       hours_med = bt_hours(z_med, params),
       mult = curve_df$mult[i],
       z_used = curve_df$log_post_age[i])
}
m_tt_med <- multiplier_at_median(sm_tt, m_tt, pp_tt, "TikTok")
m_li_med <- multiplier_at_median(sm_li, m_li, pp_li, "LinkedIn")

cat("\n[Sanity 2] Panel A: predicted velocity multiplier at median log_post_age\n")
cat("  (gratia centers smooths on the empirical distribution, not the\n")
cat("   median; on a right-skewed predictor like log_post_age the median is\n")
cat("   below the mean, so the multiplier at the median can deviate from 1.\n")
cat("   The flag below is informational, not a defect.)\n")
for (e in list(m_tt_med, m_li_med)) {
  flag <- if (abs(e$mult - 1) > 0.05) "   <-- |mult - 1| > 0.05 (informational)" else ""
  cat(sprintf("  %-9s  z_med=%+.3f  raw=%.1fh  multiplier=%.4f%s\n",
              e$label, e$z_med, e$hours_med, e$mult, flag))
}

# Additionally report the multiplier at z = 0 (empirical mean), which IS
# expected to sit near 1.0 by the centering constraint.
mult_at_zero <- function(curve_df, label) {
  i <- which.min(abs(curve_df$log_post_age - 0))
  if (length(i) == 0 || abs(curve_df$log_post_age[i]) > 0.5) {
    return(sprintf("  %-9s  z=0 outside plotted [P5, P95] window\n", label))
  }
  sprintf("  %-9s  z=0    raw=%.1fh  multiplier=%.4f\n",
          label, bt_hours(curve_df$log_post_age[i],
                          if (label == "TikTok") pp_tt else pp_li),
          curve_df$mult[i])
}
cat("\n[Sanity 2b] Multiplier at z = 0 (empirical mean of log_post_age):\n")
cat(mult_at_zero(sm_tt, "TikTok"))
cat(mult_at_zero(sm_li, "LinkedIn"))

# X-axis breaks at interpretable raw hours, restricted to the union of both
# platforms' plotted ranges.
raw_range <- range(sm_both$hours, na.rm = TRUE)
cat(sprintf("\n[Panel A x-range] %.2f to %.0f hours (union of TT/LI [P1, P99])\n",
            raw_range[1], raw_range[2]))
candidate_breaks <- c(1, 6, 24, 24*3, 24*7, 24*14)
break_labels     <- c("1h", "6h", "24h", "3d", "7d", "14d")
keep <- candidate_breaks >= raw_range[1] & candidate_breaks <= raw_range[2]
hr_breaks <- candidate_breaks[keep]
hr_labels <- break_labels[keep]

p_A <- ggplot(sm_both, aes(x = hours, y = mult,
                           colour = platform, fill = platform)) +
  geom_hline(yintercept = 1, linetype = "dashed",
             colour = "grey60", linewidth = 0.4) +
  geom_ribbon(aes(ymin = lo, ymax = hi),
              alpha = 0.20, colour = NA) +
  geom_line(linewidth = 0.8) +
  scale_x_continuous(trans = "log10",
                     breaks = hr_breaks, labels = hr_labels,
                     expand = c(0.01, 0.01)) +
  scale_y_continuous(trans = "log10") +
  scale_colour_manual(values = c(TikTok = TT_COL, LinkedIn = LI_COL),
                      name = NULL, drop = FALSE) +
  scale_fill_manual  (values = c(TikTok = TT_COL, LinkedIn = LI_COL),
                      name = NULL, drop = FALSE) +
  labs(x = "Post age at capture (hours, log scale)",
       y = "Velocity (multiple of mean)")

# =============================================================================
# Panel B:  is_weekend forest plot
# =============================================================================
extract_is_weekend <- function(model, label) {
  pt <- summary(model)$p.table
  stopifnot("is_weekend not found in p.table" = "is_weekend" %in% rownames(pt))
  b  <- unname(pt["is_weekend", "Estimate"])
  se <- unname(pt["is_weekend", "Std. Error"])
  pv <- unname(pt["is_weekend", ncol(pt)])
  data.frame(
    platform = label,
    beta     = b,
    se       = se,
    pct      = (exp(b)             - 1) * 100,
    lo       = (exp(b - 1.96 * se) - 1) * 100,
    hi       = (exp(b + 1.96 * se) - 1) * 100,
    p_value  = pv
  )
}
isw <- bind_rows(
  extract_is_weekend(m_tt, "TikTok"),
  extract_is_weekend(m_li, "LinkedIn")
) |>
  mutate(platform = factor(platform, levels = c("LinkedIn", "TikTok")))

# APA p-value formatting: drop leading zero, three decimals, "p < .001" floor
fmt_p <- function(p) {
  if (is.na(p))     return("")
  if (p < .001)     return("p < .001")
  s <- sprintf("p = %.3f", p)
  sub("p = 0", "p = ", s, fixed = TRUE)
}
isw$p_label <- vapply(isw$p_value, fmt_p, character(1))

cat("\n[Sanity 3] Panel B: is_weekend percent change with 95% CI\n")
for (i in seq_len(nrow(isw))) {
  r <- isw[i, ]
  cat(sprintf("  %-9s  %+5.1f%% [%+5.1f, %+5.1f]   %s   beta=%+.4f, SE=%.4f\n",
              as.character(r$platform), r$pct, r$lo, r$hi,
              r$p_label, r$beta, r$se))
}

# Cross-check vs results_register.csv if present
if (file.exists(REGISTER)) {
  reg <- read_csv(REGISTER, show_col_types = FALSE)
  cmp <- reg |>
    dplyr::filter(feature == "is_weekend",
                  outcome == "velocity_t24",
                  platform %in% c("tt", "li")) |>
    dplyr::select(platform, value, p_value, sig_marker)
  cat("\n[Sanity 3b] Register cross-check (05_modelling/11_data_results/_ground_truth/results_register.csv):\n")
  print(cmp)
} else {
  cat("\n[Sanity 3b] results_register.csv not found at: ", REGISTER, "\n", sep = "")
}

# Build forest plot. Inline p-value labels removed per V5 figure audit (§3 #9);
# significance now lives in the chapter Note only.

p_B <- ggplot(isw, aes(y = platform, x = pct, colour = platform)) +
  geom_vline(xintercept = 0, linetype = "dashed",
             colour = "grey60", linewidth = 0.4) +
  geom_pointrange(aes(xmin = lo, xmax = hi),
                  size = 0.45, linewidth = 0.8) +
  scale_colour_manual(values = c(TikTok = TT_COL, LinkedIn = LI_COL),
                      name = NULL, drop = FALSE, guide = "none") +
  scale_x_continuous(labels = function(x) sprintf("%+d%%", as.integer(x)),
                     expand = expansion(mult = c(0.05, 0.05))) +
  labs(x = "Weekend vs weekday (% change in 24h velocity)",
       y = NULL)

# =============================================================================
# Combine and save
# =============================================================================
fig <- (p_A | p_B) +
  plot_annotation(tag_levels = "A") +
  plot_layout(widths = c(1, 1), guides = "collect") &
  theme(legend.position = "bottom",
        plot.tag        = element_text(face = "bold", size = 11))

# 16 cm wide x 7 cm tall -> inches
W_IN <- 16 / 2.54
H_IN <- 7  / 2.54

ggsave(OUT_PNG, fig, width = W_IN, height = H_IN,
       dpi = 300, bg = "white", device = "png")
# Default pdf() device avoids the X11/cairo dependency on this system.
# ggsave(OUT_PDF, fig, width = W_IN, height = H_IN,
#        device = "pdf", bg = "white")  # PDF disabled

cat(sprintf("\nWrote: %s  (%.1f x %.1f in)\n", OUT_PNG, W_IN, H_IN))
cat(sprintf("Wrote: %s  (%.1f x %.1f in)\n", OUT_PDF, W_IN, H_IN))
cat("Done.\n")

invisible(list(panel_A = p_A, panel_B = p_B, fig = fig))
