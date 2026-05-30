# =============================================================================
# fig_02_tt_inclusion.R
#
# Master's thesis: Learning the Levers — Which Creator-Controllable
# Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
# from a Swiss German-Language Perspective
# Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026
#
# Purpose:
#   Produces Chapter 4 Figure 1 (TikTok inclusion GAM): two-panel partial-
#   effect plot. Panel A is s(weekday) (cyclic); Panel B is
#   s(log_ocr_text_len) (linear). Both panels are on the log-odds scale.
#
# Pipeline position:
#   Thesis figure builder; reads the fitted TikTok inclusion GAM from
#   05_gam/models/ and the shared plotting theme from config/.
#
# Inputs:
#   05_modelling/config/theme_thesis.R               theme_thesis() and PLATFORM_COLOURS
#   05_modelling/05_gam/models/m_tt_inclusion.rds    TikTok inclusion GAM
#
# Outputs:
#   05_modelling/10_figures/output/figure_r1_tt_inclusion.png  Chapter 4 Figure 1
#
# Usage:
#   Rscript 05_modelling/10_figures/R/fig_02_tt_inclusion.R
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
MODEL <- file.path(REPO, "05_modelling", "05_gam", "models", "m_tt_inclusion.rds")

# -- Theme --------------------------------------------------------------------
if (file.exists(THEME)) {
  source(THEME)                        # defines theme_thesis() + PLATFORM_COLOURS
  TT_TEAL <- unname(PLATFORM_COLOURS[["TikTok"]])
  theme_set(theme_thesis(base_size = 11))
} else {
  TT_TEAL <- "#00f2ea"
  theme_set(theme_minimal(base_size = 11) +
              theme(panel.grid.minor = element_blank(),
                    panel.grid.major = element_line(colour = "grey92", linewidth = 0.3),
                    panel.background = element_blank()))
}

# -- Load model ---------------------------------------------------------------
m <- readRDS(MODEL)

# -- Verify required smooths are present and significant ----------------------
st <- summary(m)$s.table
cat("summary(model)$s.table:\n")
print(round(st, 5))

need <- c("s(weekday)", "s(log_ocr_text_len)")
missing <- setdiff(need, rownames(st))
if (length(missing) > 0) {
  stop(sprintf("Required smooth(s) missing from model: %s",
               paste(missing, collapse = ", ")))
}
for (nm in need) {
  pv <- st[nm, "p-value"]
  if (pv >= 0.05) {
    cat(sprintf("NOTE: %s has p = %.4f (not < .05); proceeding anyway.\n",
                nm, pv))
  } else {
    cat(sprintf("OK: %s  p = %.4f\n", nm, pv))
  }
}

# =============================================================================
# Extract partial-effect curves
# -----------------------------------------------------------------------------
# Design note. The brief specified gratia::draw(). Using smooth_estimates()
# instead so the CI ribbon alpha, zero line, and axis labels can be set
# exactly as specified; draw() produces its own composed ggplot that is
# awkward to post-hoc restyle. smooth_estimates() is the same mechanism
# draw() uses internally for the partial-effect curve data.
# =============================================================================

sm_wk <- smooth_estimates(m, select = "s(weekday)",          n = 200) |>
  mutate(lower = .estimate - 1.96 * .se,
         upper = .estimate + 1.96 * .se)

sm_oc <- smooth_estimates(m, select = "s(log_ocr_text_len)", n = 400) |>
  mutate(lower = .estimate - 1.96 * .se,
         upper = .estimate + 1.96 * .se)

# =============================================================================
# Panel A: s(weekday) cyclic smooth
# =============================================================================
day_labels <- c("Mon","Tue","Wed","Thu","Fri","Sat","Sun")

p_A <- ggplot(sm_wk, aes(x = weekday)) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey60",
             linewidth = 0.4) +
  geom_ribbon(aes(ymin = lower, ymax = upper),
              fill = TT_TEAL, alpha = 0.2) +
  geom_line(aes(y = .estimate), colour = TT_TEAL, linewidth = 0.8) +
  scale_x_continuous(breaks = 0:6, labels = day_labels, expand = c(0.01, 0.01)) +
  labs(x = "Weekday", y = "Partial effect (log-odds)")

# =============================================================================
# Panel B: s(log_ocr_text_len) linear smooth
# -----------------------------------------------------------------------------
# The feature enters the model as z-standardised log1p(ocr_text_len).
# Pre-z mean / sd on the TikTok training subset:
#   mean(log1p(ocr_text_len)) = 4.901885
#   sd(log1p(ocr_text_len))   = 1.116966
# To label the x-axis in raw character counts (0, 10, 100, 1k, 10k) we back-
# transform the char-count values into z-units and place breaks there.
# =============================================================================
LOG_OCR_MEAN <- 4.901885
LOG_OCR_SD   <- 1.116966
char_refs    <- c(0, 10, 100, 1000, 10000)
z_breaks     <- (log1p(char_refs) - LOG_OCR_MEAN) / LOG_OCR_SD
char_labels  <- c("0", "10", "100", "1k", "10k")

# Keep breaks inside the plotted range (10000 back-transforms to z = 3.86,
# which is just past the right edge of the observed data, z_max = 3.72).
xrange <- range(sm_oc$log_ocr_text_len)
keep   <- z_breaks >= xrange[1] & z_breaks <= xrange[2]
if (any(!keep)) {
  cat(sprintf("NOTE: dropping %d break(s) outside observed x-range: %s\n",
              sum(!keep), paste(char_labels[!keep], collapse = ", ")))
}
z_breaks_kept    <- z_breaks[keep]
char_labels_kept <- char_labels[keep]

p_B <- ggplot(sm_oc, aes(x = log_ocr_text_len)) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey60",
             linewidth = 0.4) +
  geom_ribbon(aes(ymin = lower, ymax = upper),
              fill = TT_TEAL, alpha = 0.2) +
  geom_line(aes(y = .estimate), colour = TT_TEAL, linewidth = 0.8) +
  scale_x_continuous(breaks = z_breaks_kept, labels = char_labels_kept,
                     expand = c(0.01, 0.01)) +
  labs(x = "OCR text length (characters, log scale)", y = "Partial effect (log-odds)")

# =============================================================================
# Combine and save
# =============================================================================
fig <- (p_A | p_B) + plot_annotation(tag_levels = "A")

out_path <- file.path(OUT, "figure_r1_tt_inclusion.png")
ggsave(out_path, fig,
       width = 6, height = 2.8, dpi = 300, bg = "white")

sz_kb <- round(file.info(out_path)$size / 1024, 1)
cat(sprintf("Saved: %s (%.1f KB)\n", out_path, sz_kb))
