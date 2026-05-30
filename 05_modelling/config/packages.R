# =============================================================================
# packages.R
#
# Master's thesis: Learning the Levers — Which Creator-Controllable
# Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
# from a Swiss German-Language Perspective
# Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026
#
# Purpose:
#   Shared library loading and path definitions sourced by every script in
#   the modelling pipeline. Defines DB_PATH, BASE_DIR, and DATA_DIR.
#
# Pipeline position:
#   Sourced by every script in the modelling pipeline (steps 00–11).
#
# Usage:
#   source("config/packages.R")
# =============================================================================

# -- Libraries ----------------------------------------------------------------
library(DBI)
library(RSQLite)
library(tidyverse)
library(arrow)        # Parquet I/O
# -- Packages needed later (require ggplot2 >=3.5.2; loaded in later steps) ----
# library(gratia)       # GAM visualisation   (Step 5)
# library(patchwork)    # Plot composition    (Step 4+)
# library(kableExtra)   # Nice tables         (Step 4+)

# -- Packages safe to load now -------------------------------------------------
library(mgcv)         # GAMs
library(ranger)       # Random forests
library(yardstick)    # Model metrics
library(corrplot)     # Correlation matrices
library(naniar)       # Missingness diagnostics
library(openxlsx)     # Excel output
library(cluster)      # Silhouette analysis (Step 2)

# -- Paths --------------------------------------------------------------------
DB_PATH  <- here::here("04_database", "scraper.db")
BASE_DIR <- here::here("05_modelling")
DATA_DIR <- file.path(BASE_DIR, "01_build_analytical_table", "data")

# Ensure output directories exist
dir.create(DATA_DIR, recursive = TRUE, showWarnings = FALSE)

# -- Theme (loaded separately) ------------------------------------------------
theme_file <- file.path(BASE_DIR, "config", "theme_thesis.R")
if (file.exists(theme_file)) source(theme_file)

cat("packages.R loaded | DB:", DB_PATH, "\n")
