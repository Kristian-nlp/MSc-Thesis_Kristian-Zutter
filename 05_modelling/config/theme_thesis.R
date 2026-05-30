# =============================================================================
# theme_thesis.R
#
# Master's thesis: Learning the Levers — Which Creator-Controllable
# Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
# from a Swiss German-Language Perspective
# Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026
#
# Purpose:
#   APA 7th edition ggplot2 theme, platform colour scales, and save_plot()
#   helper. Sourced automatically by packages.R; all plots in the pipeline
#   inherit theme_thesis() via theme_set().
#
# Pipeline position:
#   Sourced by every script in the modelling pipeline (steps 00–11).
#
# Usage:
#   source("config/theme_thesis.R")   # normally loaded via packages.R
# =============================================================================


# =============================================================================
# 1. THEME
# =============================================================================

theme_thesis <- function(base_size = 11) {
  theme_minimal(base_size = base_size, base_family = "") %+replace%
    theme(
      # -- Text --
      plot.title         = element_text(size = base_size + 1, face = "bold",
                                        hjust = 0, margin = margin(b = 8)),
      plot.subtitle      = element_text(size = base_size, hjust = 0,
                                        margin = margin(b = 8)),
      axis.title         = element_text(size = base_size),
      axis.text          = element_text(size = base_size - 1, colour = "black"),
      strip.text         = element_text(size = base_size, face = "bold",
                                        margin = margin(b = 4, t = 4)),
      legend.title       = element_text(size = base_size, face = "bold"),
      legend.text        = element_text(size = base_size - 1),

      # -- Panel --
      panel.background   = element_rect(fill = "white", colour = NA),
      plot.background    = element_rect(fill = "white", colour = NA),
      panel.grid.major   = element_blank(),
      panel.grid.minor   = element_blank(),
      panel.border       = element_blank(),

      # -- Axes --
      axis.line          = element_line(colour = "black", linewidth = 0.4),
      axis.ticks         = element_line(colour = "black", linewidth = 0.3),
      axis.ticks.length  = unit(3, "pt"),

      # -- Legend --
      legend.position    = "bottom",
      legend.key         = element_rect(fill = "white", colour = NA),
      legend.background  = element_rect(fill = "white", colour = NA),

      # -- Margins --
      plot.margin        = margin(t = 5, r = 10, b = 5, l = 5, unit = "pt"),

      # -- Strip (facet labels) --
      strip.background   = element_rect(fill = "grey95", colour = NA)
    )
}


# =============================================================================
# 2. PLATFORM COLOUR PALETTE
# =============================================================================

PLATFORM_COLOURS <- c(
  TikTok    = "#00f2ea",
  Instagram = "#E1306C",
  LinkedIn  = "#0A66C2"
)

# Lowercase-keyed lookup for label retrieval (e.g., PLATFORM_LABELS[["tiktok"]] = "TikTok")
PLATFORM_LABELS <- c(tiktok = "TikTok", instagram = "Instagram", linkedin = "LinkedIn")

# Lowercase-keyed alias for subsetting (e.g., PLATFORM_COLOURS_LC[c("tiktok", "instagram")])
PLATFORM_COLOURS_LC <- setNames(unname(PLATFORM_COLOURS), c("tiktok", "instagram", "linkedin"))

scale_colour_platform <- function(...) {
  scale_colour_manual(values = PLATFORM_COLOURS, ...)
}

scale_fill_platform <- function(...) {
  scale_fill_manual(values = PLATFORM_COLOURS, ...)
}


# =============================================================================
# 3. EVER-TOP PALETTE
# =============================================================================

EVERTOP_COLOURS <- c(
  `0` = "#999999",
  `1` = "#E69F00"
)

EVERTOP_LABELS <- c(
  `0` = "Baseline",
  `1` = "Top-20"
)

scale_colour_evertop <- function(...) {
  scale_colour_manual(values = EVERTOP_COLOURS, labels = EVERTOP_LABELS, ...)
}

scale_fill_evertop <- function(...) {
  scale_fill_manual(values = EVERTOP_COLOURS, labels = EVERTOP_LABELS, ...)
}


# =============================================================================
# 4. save_plot() HELPER
# =============================================================================

#' Save a ggplot to PNG with APA defaults.
#'
#' @param p       A ggplot object.
#' @param filename  Basename only (e.g. "01_evertop_balance.png").
#' @param width   Width in inches (default: 6.5, APA single-column max).
#' @param height  Height in inches (default: 4.5).
#' @param dpi     Resolution (default: 300).
save_plot <- function(p, filename, width = 6.5, height = 4.5, dpi = 300) {
  # Use PLOT_DIR from parent environment if available
  out_dir <- if (exists("PLOT_DIR", envir = parent.frame())) {
    get("PLOT_DIR", envir = parent.frame())
  } else {
    file.path(BASE_DIR, "04_eda", "plots")
  }
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  fpath <- file.path(out_dir, filename)
  ggsave(fpath, plot = p, width = width, height = height, dpi = dpi,
         bg = "white", device = "png")
  cat(sprintf("  Saved: %s (%.1f x %.1f in, %d DPI)\n", filename, width, height, dpi))
}


# =============================================================================
# 5. SET GLOBAL THEME
# =============================================================================

theme_set(theme_thesis())

cat("theme_thesis.R loaded | APA theme set | Platform + ever-top palettes defined\n")
