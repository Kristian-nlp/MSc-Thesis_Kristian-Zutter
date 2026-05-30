# =============================================================================
# figure_r3_ig_inclusion_partials_joined.R
#
# Master's thesis: Learning the Levers — Which Creator-Controllable
# Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
# from a Swiss German-Language Perspective
# Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026
#
# Purpose:
#   Produces Chapter 4 Figure 3 (Instagram inclusion GAM): two-panel
#   partial-effect plot. Panel A is s(flesch_reading_ease) (linear);
#   Panel B is s(word_count) (non-linear). Built on the joined-only
#   sensitivity refit (m_ig_inclusion_joined.rds, N = 4,928), which the
#   thesis methodology names as the authoritative basis for Instagram
#   inclusion content claims because the full model is instrumentation-
#   affected by DOM-only capture asymmetry in the Explore grid.
#
# Pipeline position:
#   Thesis figure builder; reads the IG joined-only sensitivity refit
#   from 05_gam/sensitivity/ and the shared plotting theme.
#
# Inputs:
#   05_modelling/config/theme_thesis.R                                  theme_thesis() and PLATFORM_COLOURS
#   05_modelling/05_gam/sensitivity/m_ig_inclusion_joined.rds           Instagram joined-only inclusion GAM
#
# Outputs:
#   05_modelling/10_figures/output/figure_r3_ig_inclusion_partials_joined.png  Chapter 4 Figure 3
#
# Usage:
#   Rscript 05_modelling/10_figures/R/figure_r3_ig_inclusion_partials_joined.R
# =============================================================================

suppressMessages({
  library(mgcv)
  library(gratia)
  library(ggplot2)
  library(patchwork)
  library(dplyr)
})

# -- Paths --------------------------------------------------------------------
REPO  <- here::here()
BASE  <- file.path(REPO, "05_modelling", "10_figures")
OUT   <- file.path(BASE, "output")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

THEME <- file.path(REPO, "05_modelling", "config", "theme_thesis.R")
MODEL <- file.path(REPO, "05_modelling", "05_gam", "sensitivity",
                   "m_ig_inclusion_joined.rds")

# -- Theme --------------------------------------------------------------------
if (file.exists(THEME)) {
  source(THEME)                        # defines theme_thesis() + PLATFORM_COLOURS
  IG_PINK <- unname(PLATFORM_COLOURS[["Instagram"]])
  theme_set(theme_thesis(base_size = 11))
} else {
  IG_PINK <- "#E1306C"
  theme_set(theme_minimal(base_size = 11) +
              theme(panel.grid.minor = element_blank(),
                    panel.grid.major = element_line(colour = "grey92", linewidth = 0.3),
                    panel.background = element_blank()))
}

# -- Load model ---------------------------------------------------------------
m <- readRDS(MODEL)

cat(sprintf("Source: %s\n", basename(MODEL)))
cat(sprintf("Joined-only N = %d\n", length(m$y)))

# -- Verify required smooths are present --------------------------------------
st <- summary(m)$s.table
cat("\nsummary(model)$s.table:\n")
print(round(st, 5))

need <- c("s(flesch_reading_ease)", "s(word_count)")
missing <- setdiff(need, rownames(st))
if (length(missing) > 0) {
  stop(sprintf("Required smooth(s) missing from model: %s",
               paste(missing, collapse = ", ")))
}
for (nm in need) {
  pv <- st[nm, "p-value"]
  if (pv >= 0.05) {
    cat(sprintf("NOTE: %s has p = %.4f (not < .05); plotting anyway.\n",
                nm, pv))
  } else {
    cat(sprintf("OK: %s  p = %.4f\n", nm, pv))
  }
}

# =============================================================================
# Extract partial-effect curves
# -----------------------------------------------------------------------------
# Design note. gratia::smooth_estimates() returns a constant astronomical SE
# on the IG joined-only refit because the VCov has near-singular entries
# from three separated cells (media_type = reel, topic_cluster = 6, lang =
# fr -- see the thesis methodology decision log). The same
# pathology is documented for the LinkedIn inclusion figure (lines 151-156
# of figure_04_li_inclusion_partial_effects.R).
#
# Workaround: predict(m, type = "terms", se.fit = TRUE) uses the standard
# mgcv machinery and isolates the smooth's partial fit + SE without
# tripping on the singular off-diagonal blocks.
#
# Plot range: q05-q95 of each predictor on the joined-only fitting frame
# (m$model). Trims long-tail outliers and matches the convention used by
# the LinkedIn partial-effect figure.
# =============================================================================

q_fr <- stats::quantile(m$model$flesch_reading_ease,
                        c(0.05, 0.95), na.rm = TRUE)
q_wc <- stats::quantile(m$model$word_count,
                        c(0.05, 0.95), na.rm = TRUE)

# Build a single newdata template: every predictor at a valid neutral value.
# Numerics at the z-scaled mean (0); factors at their first level; logicals
# at FALSE. terms = "s(...)" then isolates the smooth's partial fit and SE,
# so the choice of fixed values for the other predictors does not affect
# the curve being plotted.
model_frame_vars <- setdiff(names(m$model), "ever_top")

build_newdata <- function(focal_var, focal_grid) {
  nd <- data.frame(.placeholder = focal_grid)
  nd[[focal_var]] <- focal_grid
  nd$.placeholder <- NULL
  for (v in setdiff(model_frame_vars, focal_var)) {
    col <- m$model[[v]]
    if (is.factor(col)) {
      nd[[v]] <- factor(levels(col)[1], levels = levels(col))
    } else if (is.logical(col)) {
      nd[[v]] <- FALSE
    } else {
      nd[[v]] <- 0
    }
  }
  nd
}

# Panel A: s(flesch_reading_ease) -------------------------------------------
nd_fr   <- build_newdata("flesch_reading_ease",
                         seq(q_fr[1], q_fr[2], length.out = 400))
pred_fr <- predict(m, newdata = nd_fr, type = "terms",
                   terms = "s(flesch_reading_ease)", se.fit = TRUE)
sm_fr <- data.frame(
  flesch_reading_ease = nd_fr$flesch_reading_ease,
  .estimate = as.numeric(pred_fr$fit[, "s(flesch_reading_ease)"]),
  .se       = as.numeric(pred_fr$se.fit[, "s(flesch_reading_ease)"])
) |>
  mutate(lower = .estimate - 1.96 * .se,
         upper = .estimate + 1.96 * .se)

# Panel B: s(word_count) -----------------------------------------------------
nd_wc   <- build_newdata("word_count",
                         seq(q_wc[1], q_wc[2], length.out = 400))
pred_wc <- predict(m, newdata = nd_wc, type = "terms",
                   terms = "s(word_count)", se.fit = TRUE)
sm_wc <- data.frame(
  word_count = nd_wc$word_count,
  .estimate = as.numeric(pred_wc$fit[, "s(word_count)"]),
  .se       = as.numeric(pred_wc$se.fit[, "s(word_count)"])
) |>
  mutate(lower = .estimate - 1.96 * .se,
         upper = .estimate + 1.96 * .se)

cat(sprintf("\nPanel A fit range: [%+.3f, %+.3f]; SE range: [%.3f, %.3f]\n",
            min(sm_fr$.estimate), max(sm_fr$.estimate),
            min(sm_fr$.se),       max(sm_fr$.se)))
cat(sprintf("Panel B fit range: [%+.3f, %+.3f]; SE range: [%.3f, %.3f]\n",
            min(sm_wc$.estimate), max(sm_wc$.estimate),
            min(sm_wc$.se),       max(sm_wc$.se)))

# =============================================================================
# Back-transform axes from z-units to raw units
# -----------------------------------------------------------------------------
# Both features enter the Instagram model as within-platform z-standardised
# values. Pre-z mean / sd on the Instagram training subset (from
# 05_modelling/03_data_prep/output/z_score_lookup.csv):
#   flesch_reading_ease  mean = 47.73786   sd = 34.00537
#   word_count           mean = 46.47550   sd = 57.96952
# Breaks are chosen in raw units (meaningful to a reader) and converted
# to z-units for placement; labels show the raw-unit values.
# =============================================================================

FR_MEAN <- 47.73786
FR_SD   <- 34.00537
fr_refs    <- c(0, 25, 50, 75, 100)
fr_z_breaks <- (fr_refs - FR_MEAN) / FR_SD
fr_labels   <- as.character(fr_refs)

xrange_fr <- range(sm_fr$flesch_reading_ease)
keep_fr   <- fr_z_breaks >= xrange_fr[1] & fr_z_breaks <= xrange_fr[2]
if (any(!keep_fr)) {
  cat(sprintf("NOTE: dropping %d Flesch break(s) outside observed x-range: %s\n",
              sum(!keep_fr), paste(fr_labels[!keep_fr], collapse = ", ")))
}
fr_z_breaks_kept <- fr_z_breaks[keep_fr]
fr_labels_kept   <- fr_labels[keep_fr]

WC_MEAN <- 46.47550
WC_SD   <- 57.96952
wc_refs    <- c(0, 50, 100, 200, 400)
wc_z_breaks <- (wc_refs - WC_MEAN) / WC_SD
wc_labels   <- c("0", "50", "100", "200", "400")

xrange_wc <- range(sm_wc$word_count)
keep_wc   <- wc_z_breaks >= xrange_wc[1] & wc_z_breaks <= xrange_wc[2]
if (any(!keep_wc)) {
  cat(sprintf("NOTE: dropping %d word-count break(s) outside observed x-range: %s\n",
              sum(!keep_wc), paste(wc_labels[!keep_wc], collapse = ", ")))
}
wc_z_breaks_kept <- wc_z_breaks[keep_wc]
wc_labels_kept   <- wc_labels[keep_wc]

# =============================================================================
# Panel A: s(flesch_reading_ease) linear smooth
# =============================================================================
p_A <- ggplot(sm_fr, aes(x = flesch_reading_ease)) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey60",
             linewidth = 0.4) +
  geom_ribbon(aes(ymin = lower, ymax = upper),
              fill = IG_PINK, alpha = 0.2) +
  geom_line(aes(y = .estimate), colour = IG_PINK, linewidth = 0.8) +
  scale_x_continuous(breaks = fr_z_breaks_kept, labels = fr_labels_kept,
                     expand = c(0.01, 0.01)) +
  scale_y_continuous(labels = scales::label_number(accuracy = 0.1)) +
  labs(x = "Flesch reading ease", y = "Partial effect (log-odds)")

# =============================================================================
# Panel B: s(word_count) non-linear smooth
# =============================================================================
p_B <- ggplot(sm_wc, aes(x = word_count)) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey60",
             linewidth = 0.4) +
  geom_ribbon(aes(ymin = lower, ymax = upper),
              fill = IG_PINK, alpha = 0.2) +
  geom_line(aes(y = .estimate), colour = IG_PINK, linewidth = 0.8) +
  scale_x_continuous(breaks = wc_z_breaks_kept, labels = wc_labels_kept,
                     expand = c(0.01, 0.01)) +
  scale_y_continuous(labels = scales::label_number(accuracy = 0.1)) +
  labs(x = "Caption word count", y = "Partial effect (log-odds)")

# =============================================================================
# Combine and save
# =============================================================================
fig <- (p_A | p_B) + plot_annotation(tag_levels = "A")

out_png <- file.path(OUT, "figure_r3_ig_inclusion_partials_joined.png")
ggsave(out_png, fig, width = 6, height = 2.8, dpi = 300, bg = "white")

sz_kb <- round(file.info(out_png)$size / 1024, 1)
cat(sprintf("\nSaved: %s (%.1f KB)\n", out_png, sz_kb))

# =============================================================================
# Caption-ready diagnostics block (joined-only refit)
# =============================================================================
N_joined <- length(m$y)
fr_row   <- summary(m)$s.table["s(flesch_reading_ease)", , drop = FALSE]
wc_row   <- summary(m)$s.table["s(word_count)",          , drop = FALSE]

fmt_p <- function(p) {
  if (p < .001) "< .001" else sprintf("= %.3f", p)
}

cat("\n--- Figure 3 caption diagnostics (joined-only refit) ---\n")
cat(sprintf("- N = %d (joined-only refit; m_ig_inclusion_joined.rds)\n", N_joined))
cat(sprintf("- Panel A  s(flesch_reading_ease):  edf = %.2f, chi-sq = %.2f, p %s\n",
            fr_row[1, "edf"], fr_row[1, "Chi.sq"], fmt_p(fr_row[1, "p-value"])))
cat(sprintf("- Panel B  s(word_count):           edf = %.2f, chi-sq = %.2f, p %s\n",
            wc_row[1, "edf"], wc_row[1, "Chi.sq"], fmt_p(wc_row[1, "p-value"])))
cat(sprintf("- Panel A x-axis range (z-units, 5th-95th pct): [%.3f, %.3f]\n",
            q_fr[1], q_fr[2]))
cat(sprintf("- Panel B x-axis range (z-units, 5th-95th pct): [%.3f, %.3f]\n",
            q_wc[1], q_wc[2]))
