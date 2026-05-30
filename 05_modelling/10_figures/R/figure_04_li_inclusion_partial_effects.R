# =============================================================================
# figure_04_li_inclusion_partial_effects.R
#
# Master's thesis: Learning the Levers — Which Creator-Controllable
# Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
# from a Swiss German-Language Perspective
# Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026
#
# Purpose:
#   Produces Chapter 4 Figure 4 (LinkedIn inclusion GAM): two-panel
#   partial-effect plot of the two significant predictors of top-20
#   inclusion. Panel A is s(log_post_age) (continuous smooth, x back-
#   transformed to hours). Panel B is a media_type coefficient (forest)
#   plot for non-reference levels relative to the article reference.
#   Both panels are on the log-odds scale; see Table 6 for the
#   percentage-point translation.
#
# Pipeline position:
#   Thesis figure builder; reads the fitted LinkedIn inclusion GAM from
#   05_gam/models/ and the LinkedIn fit-ready frame and Chapter 4 post
#   table for back-transform parameters.
#
# Inputs:
#   05_modelling/config/packages.R                                library loads + path constants
#   05_modelling/config/theme_thesis.R                            theme_thesis() and PLATFORM_COLOURS
#   05_modelling/05_gam/models/m_li_inclusion.rds                 LinkedIn inclusion GAM
#   05_modelling/01_build_analytical_table/data/df_post.parquet   LinkedIn raw post-age values for back-transform
#   05_modelling/03_data_prep/data/df_li.parquet                  LinkedIn fit-ready frame
#
# Outputs:
#   05_modelling/10_figures/output/figure_r4_li_inclusion_partial_effects.png  Chapter 4 Figure 4
#
# Usage:
#   Rscript 05_modelling/10_figures/R/figure_04_li_inclusion_partial_effects.R
# =============================================================================

source(here::here("05_modelling", "config", "packages.R"))

suppressPackageStartupMessages({
  library(mgcv)
  library(gratia)
  library(dplyr)
  library(arrow)
  library(ggplot2)
  library(patchwork)
  library(scales)
})

# -- Paths --------------------------------------------------------------------
MODEL_PATH <- file.path(BASE_DIR, "05_gam", "models", "m_li_inclusion.rds")
DF_POST    <- file.path(BASE_DIR, "01_build_analytical_table", "data",
                        "df_post.parquet")
DF_LI      <- file.path(BASE_DIR, "03_data_prep", "data", "df_li.parquet")
OUT_DIR    <- file.path(BASE_DIR, "10_figures", "output")
OUT_PNG    <- file.path(OUT_DIR, "figure_r4_li_inclusion_partial_effects.png")
OUT_PDF    <- file.path(OUT_DIR, "figure_04_li_inclusion_partial_effects.pdf")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

for (f in c(MODEL_PATH, DF_POST, DF_LI)) {
  if (!file.exists(f)) stop("Missing input: ", f, call. = FALSE)
}

# -- Theme --------------------------------------------------------------------
theme_file <- file.path(BASE_DIR, "config", "theme_thesis.R")
if (file.exists(theme_file)) source(theme_file)
LI_BLUE   <- unname(PLATFORM_COLOURS[["LinkedIn"]])  # #0A66C2
CI_ALPHA  <- 0.20
ZERO_GREY <- "grey55"

base_theme <- function() {
  theme_thesis(base_size = 10) %+replace%
    theme(
      plot.title   = element_text(size = 10, face = "bold", hjust = 0,
                                  margin = margin(b = 4)),
      axis.title.x = element_text(size = 9, margin = margin(t = 4)),
      axis.title.y = element_text(size = 9, angle = 90,
                                  margin = margin(r = 6)),
      axis.text    = element_text(size = 8.5, colour = "black"),
      plot.margin  = margin(4, 8, 4, 6)
    )
}

# =============================================================================
# Load model and verify caption numbers
# =============================================================================
m <- readRDS(MODEL_PATH)
stopifnot(inherits(m, "gam"))

st <- summary(m)$s.table
pt <- summary(m)$p.table

cat("\n== Verification against thesis caption ==\n")
if (!"s(log_post_age)" %in% rownames(st)) {
  stop("s(log_post_age) not in model smooth table", call. = FALSE)
}
pa_row <- st["s(log_post_age)", ]
cat(sprintf("  s(log_post_age): edf = %.2f, Chi.sq = %.2f, p = %s\n",
            pa_row["edf"], pa_row["Chi.sq"],
            if (pa_row["p-value"] < .001) "< .001"
            else sub("^0", "", formatC(pa_row["p-value"], digits = 3, format = "f"))))

# Sanity-check the v8 caption numbers (edf ~= 3.05, Chi.sq ~= 26.34, p < .001)
if (abs(pa_row["edf"] - 3.05) > 0.02)
  cat("  NOTE: edf differs from caption's 3.05 by > 0.02; caption needs update.\n")
if (abs(pa_row["Chi.sq"] - 26.34) > 0.5)
  cat("  NOTE: Chi.sq differs from caption's 26.34 by > 0.5; caption needs update.\n")
if (pa_row["p-value"] >= .001)
  cat("  NOTE: s(log_post_age) no longer p < .001; caption needs update.\n")

# -- Load data for back-transform and row-count verification -----------------
df_post <- arrow::read_parquet(DF_POST)
li_post <- df_post |> dplyr::filter(platform == "linkedin")
n_li_post <- nrow(li_post)
mean_top  <- mean(li_post$ever_top, na.rm = TRUE)
cat(sprintf("  df_post (linkedin subset): N = %d, mean(ever_top) = %.3f\n",
            n_li_post, mean_top))
if (n_li_post != 1170)
  cat(sprintf("  NOTE: N = %d (expected 1,170); caption needs update.\n",
              n_li_post))
if (abs(mean_top - 0.360) > 0.005)
  cat(sprintf("  NOTE: mean(ever_top) = %.3f (expected ~ 0.360).\n", mean_top))

df_li <- arrow::read_parquet(DF_LI)
n_li_model <- nrow(df_li)
cat(sprintf("  df_li (fit-ready): N = %d\n", n_li_model))

# -- Verify required columns --------------------------------------------------
req <- c("post_age_hours", "log_post_age", "media_type", "ever_top")
missing_cols <- setdiff(req, names(li_post))
if (length(missing_cols))
  stop("df_post (linkedin) missing: ",
       paste(missing_cols, collapse = ", "), call. = FALSE)

# =============================================================================
# PANEL A: s(log_post_age) partial-effect curve
# =============================================================================
# df_post holds log_post_age = log1p(post_age_hours) in raw units (not z).
# df_li (fitted on) has log_post_age z-standardised within platform.
# Recover the z-transform (mu, sd) from the LinkedIn pre-z vector so the
# z-axis of the smooth can be back-transformed to raw hours.

tol <- 1e-8
diff_log1p <- max(abs(li_post$log_post_age - log1p(li_post$post_age_hours)),
                  na.rm = TRUE)
if (diff_log1p > tol)
  stop("df_post linkedin log_post_age is not log1p(post_age_hours); ",
       "back-transform assumption invalid.", call. = FALSE)

log_pa_li <- log1p(li_post$post_age_hours)
mu_pa <- mean(log_pa_li, na.rm = TRUE)
sd_pa <- stats::sd(log_pa_li, na.rm = TRUE)
if (!is.finite(mu_pa) || !is.finite(sd_pa) || sd_pa <= 0)
  stop("Cannot recover z parameters for log_post_age.", call. = FALSE)

cat(sprintf("\n== Back-transform parameters (LinkedIn pre-z) ==\n"))
cat(sprintf("  log_post_age: mean = %.4f, sd = %.4f  (log1p(hours))\n",
            mu_pa, sd_pa))

# Confirm the in-model z-column is roughly standardised
if (abs(mean(df_li$log_post_age, na.rm = TRUE)) > 1e-4 ||
    abs(stats::sd(df_li$log_post_age, na.rm = TRUE) - 1) > 1e-4) {
  cat("  NOTE: df_li$log_post_age is not perfectly z-scored; using df_post params.\n")
}

bt_hours <- function(z) expm1(z * sd_pa + mu_pa)

# Extract partial-effect values and CIs via predict(type="terms").
# gratia::smooth_estimates() reports a constant SE ~ 81,000 for this smooth,
# a known pathology when the full-model VCov includes near-singular entries
# from the step-failure / negative Hessian warning. predict(m, type="terms",
# se.fit=TRUE) uses the standard mgcv machinery and returns the correct SE
# band (around 0.1-0.2 on the log-odds scale).

q_pa <- stats::quantile(m$model$log_post_age, c(0.05, 0.95), na.rm = TRUE)
cat(sprintf("\n== Panel A plot range (P5-P95 of observed z) ==\n"))
cat(sprintf("  z in [%.2f, %.2f] ~ %s - %s hours\n",
            q_pa[1], q_pa[2],
            format(round(bt_hours(q_pa[1])), big.mark = ","),
            format(round(bt_hours(q_pa[2])), big.mark = ",")))

# Build newdata: log_post_age sweeps the plot range; every other predictor
# fixed at a valid reference (numerics at the z-scaled mean = 0; factors at
# their first level). type="terms" isolates the s(log_post_age) partial, so
# the choice of constants for the other terms does not affect the returned
# fit/SE for this smooth.
model_frame_vars <- setdiff(names(m$model), "ever_top")
newdata_pa <- data.frame(log_post_age = seq(q_pa[1], q_pa[2],
                                            length.out = 400))
for (v in setdiff(model_frame_vars, "log_post_age")) {
  col <- m$model[[v]]
  if (is.factor(col)) {
    newdata_pa[[v]] <- factor(levels(col)[1], levels = levels(col))
  } else if (is.logical(col)) {
    newdata_pa[[v]] <- FALSE
  } else {
    newdata_pa[[v]] <- 0
  }
}

pred_pa <- predict(m, newdata = newdata_pa, type = "terms",
                   terms = "s(log_post_age)", se.fit = TRUE)
fit_pa <- as.numeric(pred_pa$fit[, "s(log_post_age)"])
se_pa  <- as.numeric(pred_pa$se.fit[, "s(log_post_age)"])
cat(sprintf("  predict() fit range: [%+.3f, %+.3f]; SE range: [%.3f, %.3f]\n",
            min(fit_pa), max(fit_pa), min(se_pa), max(se_pa)))

est_pa_plot <- data.frame(
  log_post_age = newdata_pa$log_post_age,
  .estimate    = fit_pa,
  .se          = se_pa,
  lower        = fit_pa - 1.96 * se_pa,
  upper        = fit_pa + 1.96 * se_pa
) |>
  mutate(hours = bt_hours(log_post_age))

# Reader-friendly hour breaks on a log10 axis
hour_breaks <- c(1, 6, 24, 72, 168, 336)
hour_labels <- c("1h", "6h", "24h", "3d", "7d", "14d")
hrange <- range(est_pa_plot$hours, na.rm = TRUE)
keep_br <- hour_breaks >= hrange[1] & hour_breaks <= hrange[2]
if (any(!keep_br))
  cat(sprintf("  NOTE: dropping hour break(s) outside plotted range: %s\n",
              paste(hour_labels[!keep_br], collapse = ", ")))
hour_breaks_kept <- hour_breaks[keep_br]
hour_labels_kept <- hour_labels[keep_br]

p_A <- ggplot(est_pa_plot, aes(x = hours)) +
  geom_hline(yintercept = 0, colour = ZERO_GREY, linetype = "dashed",
             linewidth = 0.3) +
  geom_ribbon(aes(ymin = lower, ymax = upper),
              fill = LI_BLUE, alpha = CI_ALPHA) +
  geom_line(aes(y = .estimate), colour = LI_BLUE, linewidth = 0.8) +
  scale_x_continuous(trans = "log10",
                     breaks = hour_breaks_kept,
                     labels = hour_labels_kept,
                     expand = c(0.01, 0.01)) +
  labs(x = "Post age at capture (hours, log scale)",
       y = "Partial effect (log-odds)") +
  base_theme()

# =============================================================================
# PANEL B: media_type coefficient (forest) plot
# =============================================================================
# Extract non-reference media_type rows from the parametric table and build
# a 95% CI as estimate +/- 1.96 * SE on the log-odds scale.

mt_rows <- rownames(pt)[grepl("^media_type", rownames(pt))]
if (length(mt_rows) == 0)
  stop("No media_type rows in p.table; model specification changed.",
       call. = FALSE)

mt_df <- data.frame(
  term     = mt_rows,
  estimate = pt[mt_rows, "Estimate"],
  se       = pt[mt_rows, "Std. Error"],
  p        = pt[mt_rows, "Pr(>|z|)"],
  row.names = NULL
) |>
  mutate(
    level = sub("^media_type", "", term),
    lower = estimate - 1.96 * se,
    upper = estimate + 1.96 * se
  ) |>
  arrange(desc(estimate))

cat("\n== Panel B: media_type coefficients ==\n")
print(mt_df |> mutate(across(c(estimate, se, lower, upper),
                             ~ sprintf("%+.3f", .))))

expected_levels <- c("carousel", "document", "image", "text", "video")
missing_lv <- setdiff(expected_levels, mt_df$level)
if (length(missing_lv))
  cat(sprintf("  NOTE: non-reference media_type levels missing: %s\n",
              paste(missing_lv, collapse = ", ")))
extra_lv <- setdiff(mt_df$level, expected_levels)
if (length(extra_lv))
  cat(sprintf("  NOTE: unexpected media_type levels present: %s\n",
              paste(extra_lv, collapse = ", ")))

mt_df$level <- factor(mt_df$level, levels = rev(mt_df$level))  # most positive on top

p_B <- ggplot(mt_df, aes(x = estimate, y = level)) +
  geom_vline(xintercept = 0, colour = ZERO_GREY, linetype = "dashed",
             linewidth = 0.3) +
  geom_errorbar(aes(xmin = lower, xmax = upper),
                orientation = "y",
                colour = LI_BLUE, width = 0.15, linewidth = 0.6) +
  geom_point(colour = LI_BLUE, size = 2.2) +
  labs(x = "Partial effect (log-odds)",
       y = NULL) +
  base_theme() +
  theme(panel.grid.major.y = element_line(colour = "grey92",
                                          linewidth = 0.3))

# =============================================================================
# Combine and save
# =============================================================================
fig <- (p_A | p_B) +
  plot_layout(widths = c(1, 1)) +
  plot_annotation(tag_levels = "A") &
  theme(plot.tag = element_text(face = "bold", size = 10))

FIG_WIDTH  <- 7
FIG_HEIGHT <- 3.5
ggsave(OUT_PNG, fig, width = FIG_WIDTH, height = FIG_HEIGHT, dpi = 300,
       bg = "white")
# ggsave(OUT_PDF, fig, width = FIG_WIDTH, height = FIG_HEIGHT, device = "pdf")  # PDF disabled

png_kb <- round(file.info(OUT_PNG)$size / 1024, 1)
pdf_kb <- round(file.info(OUT_PDF)$size / 1024, 1)
cat(sprintf("\nWrote %s (%.1f KB, %gx%g in @ 300 dpi)\n",
            basename(OUT_PNG), png_kb, FIG_WIDTH, FIG_HEIGHT))
cat(sprintf("Wrote %s (%.1f KB, %gx%g in)\n",
            basename(OUT_PDF), pdf_kb, FIG_WIDTH, FIG_HEIGHT))

# =============================================================================
# Console caption (APA 7, for the thesis document)
# =============================================================================
cat("\n=== CAPTION (for the thesis document; not embedded in the figure) ===\n")
cat(paste0(
  "Figure 4. LinkedIn: Partial effects of the two significant predictors of ",
  "top-20 inclusion. ",
  "Panel A shows the partial-effect curve for post age at initial capture ",
  "(s(log_post_age), edf = ", sprintf("%.2f", pa_row["edf"]),
  ", Chi-squared = ", sprintf("%.2f", pa_row["Chi.sq"]),
  ", p ", if (pa_row["p-value"] < .001) "< .001"
         else paste0("= ", sub("^0", "", formatC(pa_row["p-value"],
                                                 digits = 3, format = "f"))),
  "). Panel B shows the coefficient plot for media format, with each level's ",
  "effect estimated relative to the article reference category. The y-axis of ",
  "Panel A and the x-axis of Panel B show the partial effect on the log-odds ",
  "of top-20 inclusion; see Table 6 for effect magnitudes expressed as ",
  "percentage-point changes at the LinkedIn baseline rate of 36.0%. Shaded ",
  "band (Panel A) and horizontal bars (Panel B) show 95% confidence intervals. ",
  "Panel A is plotted across the 5th to 95th percentile of observed ",
  "log_post_age. N = ", format(n_li_post, big.mark = ","), ".\n"
))

invisible(list(png = OUT_PNG, pdf = OUT_PDF))
