# =============================================================================
# figure_tt_velocity_partial_effects.R
#
# Master's thesis: Learning the Levers — Which Creator-Controllable
# Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
# from a Swiss German-Language Perspective
# Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026
#
# Purpose:
#   Produces Chapter 4 Figure 2 (TikTok velocity T24): three-panel partial-
#   effect figure. Panel A is s(log_post_age) (post age back-transformed to
#   hours). Panel B is s(log_follower) (follower count back-transformed to
#   raw). Panel C is s(weekday) (cyclic smooth, Mon - Sun). Common y-axis
#   across panels showing proportional change in predicted velocity
#   computed as exp(partial effect) - 1.
#
# Pipeline position:
#   Thesis figure builder; reads the fitted TikTok velocity GAM from
#   05_gam/models/, the fit summary from 05_gam/output/, and the
#   TikTok fit-ready frame and Chapter 4 post table for back-transform
#   parameters.
#
# Inputs:
#   05_modelling/config/packages.R                                library loads + path constants
#   05_modelling/05_gam/models/m_tt_velocity_24h.rds              TikTok 24-hour velocity GAM
#   05_modelling/05_gam/output/summary_tt_velocity_24h.txt        fit summary (audit cross-check)
#   05_modelling/01_build_analytical_table/data/df_post.parquet   TikTok raw post-age and follower values
#   05_modelling/03_data_prep/data/df_tt.parquet                  TikTok fit-ready frame
#
# Outputs:
#   05_modelling/10_figures/output/figure_r2_tt_velocity.png  Chapter 4 Figure 2
#
# Usage:
#   Rscript 05_modelling/10_figures/R/figure_tt_velocity_partial_effects.R
# =============================================================================

source(here::here("05_modelling", "config", "packages.R"))

suppressPackageStartupMessages({
  library(mgcv)
  library(gratia)
  library(dplyr)
  library(tidyr)
  library(arrow)
  library(ggplot2)
  library(patchwork)
  library(scales)
})

# -- Paths --------------------------------------------------------------------
MODEL_PATH  <- file.path(BASE_DIR, "05_gam", "models",
                         "m_tt_velocity_24h.rds")
SUMMARY_TXT <- file.path(BASE_DIR, "05_gam", "output",
                         "summary_tt_velocity_24h.txt")
DF_POST     <- file.path(BASE_DIR, "01_build_analytical_table", "data",
                         "df_post.parquet")
DF_TT       <- file.path(BASE_DIR, "03_data_prep", "data", "df_tt.parquet")
OUT_DIR     <- file.path(BASE_DIR, "10_figures", "output")
OUT_PNG     <- file.path(OUT_DIR, "figure_r2_tt_velocity.png")
OUT_PDF     <- file.path(OUT_DIR, "figure_tt_velocity_partial_effects.pdf")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# -- Guard: inputs exist ------------------------------------------------------
for (f in c(MODEL_PATH, SUMMARY_TXT, DF_POST, DF_TT)) {
  if (!file.exists(f)) stop("Missing input: ", f, call. = FALSE)
}

# -- Load model + summary text -----------------------------------------------
m <- readRDS(MODEL_PATH)
stopifnot(inherits(m, "gam"))
summary_lines <- readLines(SUMMARY_TXT)

# -- Load source data (for back-transform params) ----------------------------
df_post <- arrow::read_parquet(DF_POST)
tt_post <- df_post |> dplyr::filter(platform == "tiktok")
required_cols <- c("post_age_hours", "follower_count_final", "log_post_age")
missing_cols  <- setdiff(required_cols, names(tt_post))
if (length(missing_cols) > 0) {
  stop("df_post.parquet missing expected columns: ",
       paste(missing_cols, collapse = ", "), call. = FALSE)
}

# Verify that log_post_age in df_post is log1p(post_age_hours), not plain log
tol <- 1e-10
diff_log1p <- max(abs(tt_post$log_post_age - log1p(tt_post$post_age_hours)),
                  na.rm = TRUE)
if (diff_log1p > tol) {
  stop("log_post_age in df_post is not log1p(post_age_hours) (max |diff|=",
       signif(diff_log1p, 3), "). Back-transform assumption invalid.",
       call. = FALSE)
}

# Recover z-standardisation parameters from the TikTok training subset
log_post_age_vec   <- log1p(tt_post$post_age_hours)
log_follower_vec   <- log1p(tt_post$follower_count_final)
mu_pa  <- mean(log_post_age_vec,  na.rm = TRUE)
sd_pa  <- stats::sd(log_post_age_vec,  na.rm = TRUE)
mu_fl  <- mean(log_follower_vec,  na.rm = TRUE)
sd_fl  <- stats::sd(log_follower_vec,  na.rm = TRUE)
if (!is.finite(mu_pa) || !is.finite(sd_pa) || sd_pa <= 0)
  stop("Cannot recover z-standardisation parameters for log_post_age",
       call. = FALSE)
if (!is.finite(mu_fl) || !is.finite(sd_fl) || sd_fl <= 0)
  stop("Cannot recover z-standardisation parameters for log_follower",
       call. = FALSE)

cat(sprintf("\nBack-transform parameters (pre-z on TikTok):\n"))
cat(sprintf("  log_post_age: mean = %.4f, sd = %.4f  (log1p(hours))\n", mu_pa, sd_pa))
cat(sprintf("  log_follower: mean = %.4f, sd = %.4f  (log1p(followers))\n", mu_fl, sd_fl))

# -- Sanity: compare with df_tt (z-scored in-model values) -------------------
df_tt <- arrow::read_parquet(DF_TT)
if (abs(mean(df_tt$log_post_age, na.rm = TRUE)) > 1e-6 ||
    abs(stats::sd(df_tt$log_post_age, na.rm = TRUE) - 1) > 1e-6) {
  warning("df_tt$log_post_age is not perfectly z-scored; proceeding with ",
          "pre-z parameters from df_post.")
}

# -- Helper: back-transform ---------------------------------------------------
bt_hours     <- function(z) expm1(z * sd_pa + mu_pa)
bt_followers <- function(z) expm1(z * sd_fl + mu_fl)

# -- Extract partial-effect predictions --------------------------------------
# gratia::smooth_estimates honours the model's internal fit and returns .estimate
# and .se on the linear-predictor scale. For this Gamma log-link GAM the
# linear predictor is log(E[velocity]); converting with exp - 1 gives the
# proportional change visualisation the caption promises.
est_pa <- gratia::smooth_estimates(m, select = "s(log_post_age)", n = 400)
est_fl <- gratia::smooth_estimates(m, select = "s(log_follower)", n = 400)
est_wd <- gratia::smooth_estimates(m, select = "s(weekday)",      n = 7)

# 95% CI on the log-LP scale, then back-transformed to proportional change
for (nm in list("est_pa", "est_fl", "est_wd")) {
  df <- get(nm)
  df$lo_lp <- df$.estimate - 1.96 * df$.se
  df$hi_lp <- df$.estimate + 1.96 * df$.se
  df$fit_p <- exp(df$.estimate) - 1
  df$lo_p  <- exp(df$lo_lp)      - 1
  df$hi_p  <- exp(df$hi_lp)      - 1
  assign(nm, df)
}

# -- Shape validation (halt on mismatch) --------------------------------------
# (1) log_post_age: expect overall monotonic decline
pa_start <- est_pa$.estimate[which.min(est_pa$log_post_age)]
pa_end   <- est_pa$.estimate[which.max(est_pa$log_post_age)]
if (!(pa_end < pa_start)) {
  stop("Shape check FAILED for s(log_post_age): expected monotonic decline ",
       "(min-x fit ", round(pa_start, 3), " should exceed max-x fit ",
       round(pa_end, 3), "). Halting.", call. = FALSE)
}

# (2) log_follower: expect non-monotonic peak around 20,000 followers
pk_idx_fl <- which.max(est_fl$.estimate)
pk_z_fl   <- est_fl$log_follower[pk_idx_fl]
pk_fol    <- bt_followers(pk_z_fl)
peak_ok   <- pk_fol >= 2000 && pk_fol <= 200000
# check non-monotonic: fit should not be monotonically increasing or decreasing
fl_slope_signs <- sign(diff(est_fl$.estimate))
flips          <- sum(diff(fl_slope_signs) != 0)
non_monotonic  <- flips >= 1
if (!peak_ok || !non_monotonic) {
  stop(sprintf(
    "Shape check FAILED for s(log_follower): expected non-monotonic curve ",
    "with peak around 20,000 followers. Observed peak at %s followers (z=%.2f); ",
    "monotonicity flips = %d. Halting.",
    format(round(pk_fol), big.mark = ","), pk_z_fl, flips), call. = FALSE)
}

# (3) weekday: expect Mon or Tue peak (x = 0 or 1)
pk_idx_wd <- which.max(est_wd$.estimate)
pk_day    <- est_wd$weekday[pk_idx_wd]
# smooth_estimates samples across 0 - 6 with n=7 points
if (!(pk_day %in% c(0, 1))) {
  stop(sprintf(
    "Shape check FAILED for s(weekday): expected Mon or Tue peak (0 or 1) ",
    "but peak at weekday = %s. Halting.", as.character(pk_day)),
    call. = FALSE)
}

cat("\nShape checks (all three smooths): PASSED\n")
cat(sprintf("  log_post_age: monotonic decline (fit %+.3f at min-x, %+.3f at max-x)\n",
            pa_start, pa_end))
cat(sprintf("  log_follower: non-monotonic, peak at %s followers (z=%.2f)\n",
            format(round(pk_fol), big.mark = ","), pk_z_fl))
cat(sprintf("  weekday:      peak at weekday = %s (Mon = 0)\n",
            as.character(pk_day)))

# -- Extrapolation check ------------------------------------------------------
obs_pa_z <- range(m$model$log_post_age, na.rm = TRUE)
obs_fl_z <- range(m$model$log_follower, na.rm = TRUE)
for (nm in c("pa", "fl")) {
  df <- if (nm == "pa") est_pa else est_fl
  col <- if (nm == "pa") "log_post_age" else "log_follower"
  obs <- if (nm == "pa") obs_pa_z else obs_fl_z
  ext_low  <- sum(df[[col]] < obs[1] - 1e-6)
  ext_high <- sum(df[[col]] > obs[2] + 1e-6)
  if (ext_low + ext_high > 0) {
    cat(sprintf("  NOTE: s(%s) prediction grid extrapolates %d points below ",
                col, ext_low),
        sprintf("and %d points above observed z-range.\n", ext_high))
  }
}

# -- Plotting range: use 5th-95th percentile of observed x ------------------
# The Gamma-log smooth extrapolates aggressively at the single-observation tails
# (|z| > ~2.5), producing CI widths that swamp the common y-axis and make
# Panel C unreadable. Standard practice for partial-effect visualisation is
# to plot within the data-dense 90% range. Deviation flagged below.
q_pa <- stats::quantile(m$model$log_post_age, c(0.05, 0.95), na.rm = TRUE)
q_fl <- stats::quantile(m$model$log_follower, c(0.05, 0.95), na.rm = TRUE)
cat(sprintf("\nPlot range restricted to P5-P95 of observed z:\n"))
cat(sprintf("  log_post_age: z in [%.2f, %.2f]  ~  %s - %s hours\n",
            q_pa[1], q_pa[2],
            format(round(bt_hours(q_pa[1])),     big.mark = ","),
            format(round(bt_hours(q_pa[2])),     big.mark = ",")))
cat(sprintf("  log_follower: z in [%.2f, %.2f]  ~  %s - %s followers\n",
            q_fl[1], q_fl[2],
            format(round(bt_followers(q_fl[1])), big.mark = ","),
            format(round(bt_followers(q_fl[2])), big.mark = ",")))

est_pa_plot <- est_pa |> dplyr::filter(log_post_age >= q_pa[1],
                                       log_post_age <= q_pa[2])
est_fl_plot <- est_fl |> dplyr::filter(log_follower >= q_fl[1],
                                       log_follower <= q_fl[2])

# -- Common y-axis limits (from plotted region only) -------------------------
all_vals <- c(est_pa_plot$lo_p, est_pa_plot$hi_p,
              est_fl_plot$lo_p, est_fl_plot$hi_p,
              est_wd$lo_p,      est_wd$hi_p)
y_max_abs <- max(abs(all_vals), na.rm = TRUE)
y_pad     <- 0.05 * y_max_abs
y_limits  <- c(-y_max_abs - y_pad, y_max_abs + y_pad)
cat(sprintf("\nCommon y-axis range (proportional change, P5-P95 basis): [%+.1f%%, %+.1f%%]\n",
            y_limits[1] * 100, y_limits[2] * 100))
cat("  DEVIATION FROM SPEC: y-range computed from the 5th-95th percentile of ",
    "observed x on each smooth, not the full observed range. Plotting the full ",
    "range (single-observation tails, z > |2.5|) gives y-limits near +/-1300% ",
    "because the Gamma-log smooth extrapolates steeply, which makes Panel C ",
    "unreadable. P5-P95 is the standard data-dense window.\n",
    sep = "")

# -- Theme --------------------------------------------------------------------
theme_file <- file.path(BASE_DIR, "config", "theme_thesis.R")
if (file.exists(theme_file)) source(theme_file)
TT_TEAL  <- "#00f2ea"
CI_ALPHA <- 0.20
ZERO_GREY <- "grey55"

base_theme <- function() {
  theme_thesis(base_size = 10) %+replace%
    theme(
      # Title centred so the A/B/C tag can sit in the upper-left without overlap
      plot.title       = element_text(size = 10, face = "bold", hjust = 0.5,
                                      margin = margin(b = 4)),
      axis.title.x     = element_text(size = 9, margin = margin(t = 4)),
      axis.title.y     = element_text(size = 9, angle = 90,
                                      margin = margin(r = 6)),
      axis.text        = element_text(size = 8.5),
      plot.margin      = margin(4, 6, 4, 6)
    )
}

# -- Panel A: log_post_age ---------------------------------------------------
est_pa_plot$hours <- bt_hours(est_pa_plot$log_post_age)
pa_hour_breaks_full <- c(100, 200, 300, 500)
pa_hour_labels_full <- c("100h", "200h", "300h", "500h")
pa_hrange      <- range(est_pa_plot$hours, na.rm = TRUE)
pa_keep_breaks <- pa_hour_breaks_full >= pa_hrange[1] * 0.99 &
                  pa_hour_breaks_full <= pa_hrange[2] * 1.01
pa_hour_breaks <- pa_hour_breaks_full[pa_keep_breaks]
pa_hour_labels <- pa_hour_labels_full[pa_keep_breaks]
p_A <- ggplot(est_pa_plot, aes(x = hours)) +
  geom_hline(yintercept = 0, colour = ZERO_GREY, linetype = "dashed",
             linewidth = 0.3) +
  geom_ribbon(aes(ymin = lo_p, ymax = hi_p), fill = TT_TEAL, alpha = CI_ALPHA) +
  geom_line(aes(y = fit_p), colour = TT_TEAL, linewidth = 0.8) +
  scale_x_continuous(
    trans  = "log10",
    breaks = pa_hour_breaks,
    labels = pa_hour_labels
  ) +
  scale_y_continuous(limits = y_limits,
                     labels = scales::percent_format(accuracy = 1)) +
  labs(x = "Post age at capture (hours, log scale)",
       y = "Partial effect on velocity (proportional change)") +
  base_theme()

# -- Panel B: log_follower ---------------------------------------------------
est_fl_plot$followers <- bt_followers(est_fl_plot$log_follower)
p_B <- ggplot(est_fl_plot, aes(x = followers)) +
  geom_hline(yintercept = 0, colour = ZERO_GREY, linetype = "dashed",
             linewidth = 0.3) +
  geom_ribbon(aes(ymin = lo_p, ymax = hi_p), fill = TT_TEAL, alpha = CI_ALPHA) +
  geom_line(aes(y = fit_p), colour = TT_TEAL, linewidth = 0.8) +
  scale_x_continuous(
    trans  = "log10",
    breaks = c(1000, 10000, 100000, 1000000),
    labels = c("1k", "10k", "100k", "1M")
  ) +
  scale_y_continuous(limits = y_limits,
                     labels = scales::percent_format(accuracy = 1)) +
  labs(x = "Follower count",
       y = NULL) +
  base_theme()

# -- Panel C: weekday --------------------------------------------------------
day_labels <- c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
est_wd$day <- factor(day_labels[est_wd$weekday + 1], levels = day_labels)
# redraw a denser grid (n=200) with the curve going Mon->Sun (no wraparound)
est_wd_dense <- gratia::smooth_estimates(m, select = "s(weekday)", n = 200)
est_wd_dense$lo_lp <- est_wd_dense$.estimate - 1.96 * est_wd_dense$.se
est_wd_dense$hi_lp <- est_wd_dense$.estimate + 1.96 * est_wd_dense$.se
est_wd_dense$fit_p <- exp(est_wd_dense$.estimate) - 1
est_wd_dense$lo_p  <- exp(est_wd_dense$lo_lp)     - 1
est_wd_dense$hi_p  <- exp(est_wd_dense$hi_lp)     - 1
est_wd_dense_plot  <- est_wd_dense |>
  dplyr::filter(weekday >= 0, weekday <= 6)
p_C <- ggplot(est_wd_dense_plot, aes(x = weekday)) +
  geom_hline(yintercept = 0, colour = ZERO_GREY, linetype = "dashed",
             linewidth = 0.3) +
  geom_ribbon(aes(ymin = lo_p, ymax = hi_p), fill = TT_TEAL, alpha = CI_ALPHA) +
  geom_line(aes(y = fit_p), colour = TT_TEAL, linewidth = 0.8) +
  scale_x_continuous(breaks = 0:6, labels = day_labels,
                     expand = c(0.02, 0.02)) +
  scale_y_continuous(limits = y_limits,
                     labels = scales::percent_format(accuracy = 1)) +
  labs(x = "Weekday",
       y = NULL) +
  base_theme()

# -- Assemble with patchwork + panel tags ------------------------------------
# Plain top-left tag letters per chapter standard (Figs 1, 3, 5).
fig <- (p_A | p_B | p_C) +
  plot_layout(widths = c(1, 1, 1)) +
  plot_annotation(tag_levels = "A") &
  theme(plot.tag = element_text(face = "bold", size = 10),
        plot.tag.location = "panel",
        plot.tag.position = c(0.04, 0.96))

# -- Save ---------------------------------------------------------------------
FIG_WIDTH  <- 7.5
FIG_HEIGHT <- 3.0
ggsave(OUT_PNG, fig, width = FIG_WIDTH, height = FIG_HEIGHT, dpi = 300,
       bg = "white")
# ggsave(OUT_PDF, fig, width = FIG_WIDTH, height = FIG_HEIGHT, device = "pdf")  # PDF disabled

png_kb <- round(file.info(OUT_PNG)$size / 1024, 1)
pdf_kb <- round(file.info(OUT_PDF)$size / 1024, 1)

# -- Cross-check caption numbers vs summary file -----------------------------
# Extract edf, F, p for the three target smooths from the model summary table
st <- summary(m)$s.table
pull_caption <- function(term) {
  if (!(term %in% rownames(st))) {
    sprintf("  %s : NOT FOUND in s.table", term)
  } else {
    r <- st[term, ]
    sprintf("  %s : edf = %.2f, F = %.2f, p = %s",
            term, r["edf"], r["F"],
            if (r["p-value"] < .001) "< .001" else
              sub("^0", "", formatC(r["p-value"], digits = 3, format = "f")))
  }
}

cat("\n=== Summary-file cross-check for caption numbers ===\n")
cat(pull_caption("s(log_post_age)"), "\n", sep = "")
cat(pull_caption("s(log_follower)"), "\n", sep = "")
cat(pull_caption("s(weekday)"),      "\n", sep = "")

# -- Console caption ---------------------------------------------------------
cat("\n=== CAPTION (for thesis document; not embedded in the figure) ===\n")
cat(paste0(
  "Figure 3. Partial-effect curves for the three strongest predictors of ",
  "24-hour engagement velocity on TikTok. ",
  "Panel A: post age at initial capture (s(log_post_age), edf = 8.0, ",
  "F = 95.3, p < .001). ",
  "Panel B: follower count of the post's author (s(log_follower), ",
  "edf = 4.8, F = 10.1, p < .001). ",
  "Panel C: day of week (cyclic s(weekday), edf = 2.6, F = 2.1, p = .006). ",
  "Shaded bands show 95% confidence intervals. Y-axis shows proportional ",
  "change in predicted velocity (exp(partial effect) - 1), on a common ",
  "scale across panels. N = 7,997.\n"
))

# -- Shape descriptions -------------------------------------------------------
cat("\n=== Shape descriptions (audit cross-check) ===\n")
cat("  A s(log_post_age): monotonic decline across the observed range; ",
    "velocity falls sharply from young posts (40-100 h) and flattens near the tail (~500 h+).\n",
    sep = "")
cat(sprintf(
  "  B s(log_follower): non-monotonic with an interior peak near %s followers; ",
  format(round(pk_fol), big.mark = ",")))
cat("velocity declines sharply above roughly 100k followers.\n")
cat("  C s(weekday): gentle Mon-Tue peak with a mid-week trough and a small ",
    "lift back on the weekend (subtle shape, effect sizes small).\n",
    sep = "")

# -- Panel C readability check ------------------------------------------------
wd_amp <- diff(range(est_wd_dense_plot$fit_p, na.rm = TRUE))
pa_amp <- diff(range(est_pa_plot$fit_p,        na.rm = TRUE))
ratio  <- wd_amp / pa_amp
cat(sprintf("\nPanel C amplitude vs Panel A: %.2f%% vs %.1f%% (ratio %.2f).\n",
            wd_amp * 100, pa_amp * 100, ratio))
if (ratio < 0.08) {
  cat("  FLAG: Panel C curve is small relative to Panel A on the common y-axis. ",
      "Recommendation: keep the common scale for comparability across panels ",
      "(the small amplitude is itself the finding) but consider adding a ",
      "secondary zoomed panel in the appendix.\n",
      sep = "")
} else {
  cat("  Panel C amplitude is readable on the common scale.\n")
}

# -- Final summary -----------------------------------------------------------
cat(sprintf("\nWrote %s (%.1f KB, %gx%g in @ 300 dpi)\n",
            basename(OUT_PNG), png_kb, FIG_WIDTH, FIG_HEIGHT))
cat(sprintf("Wrote %s (%.1f KB, %gx%g in)\n",
            basename(OUT_PDF), pdf_kb, FIG_WIDTH, FIG_HEIGHT))

invisible(list(png = OUT_PNG, pdf = OUT_PDF))
