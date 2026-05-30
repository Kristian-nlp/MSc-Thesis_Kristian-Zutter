# =============================================================================
# build_results_register.R
#
# Master's thesis: Learning the Levers — Which Creator-Controllable
# Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
# from a Swiss German-Language Perspective
# Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026
#
# Purpose:
#   Builds the ground-truth Results-chapter findings register. One row per
#   (feature x platform x outcome) finding the chapter reports. Applies F1
#   (per-level factor pp at platform baseline) and F2 (per-platform vs
#   cross-platform reporting) conventions, plus IQR-based delta conversion
#   for smooth terms. IG inclusion content claims use the joined-only
#   refit; the full IG inclusion model is reported only for fit metrics.
#
# Pipeline position:
#   Step 11 of the modelling pipeline. Depends on Steps 5–10 (all fitted
#   models and the per-level factor tables); feeds the chapter audit and
#   any downstream prose checks.
#
# Inputs:
#   05_modelling/05_gam/models/m_*.rds                                            All fitted GAMs
#   05_modelling/05_gam/sensitivity/m_ig_inclusion_joined.rds                     IG joined-only refit
#   05_modelling/10_thesis_tables/output/factor_pp_per_level.csv                  Per-level factor pp
#   05_modelling/10_thesis_tables/output/factor_omnibus_tests.csv                 Factor omnibus tests
#
# Outputs:
#   05_modelling/11_data_results/_ground_truth/results_register.csv               Findings register
#
# Usage:
#   Rscript 05_modelling/11_data_results/_ground_truth/build_results_register.R
# =============================================================================

source(here::here("05_modelling", "config", "packages.R"))

suppressPackageStartupMessages({
  library(dplyr)
  library(tibble)
  library(arrow)
  library(mgcv)
  library(readr)
})

REPO_ROOT <- here::here()

PREP_DIR  <- file.path(BASE_DIR, "03_data_prep", "data")
MODEL_DIR <- file.path(BASE_DIR, "05_gam", "models")
SENS_DIR  <- file.path(BASE_DIR, "05_gam", "sensitivity")
TABLES_DIR <- file.path(BASE_DIR, "10_thesis_tables", "output")
OUT_DIR   <- file.path(BASE_DIR, "11_data_results", "_ground_truth")
OUT_CSV   <- file.path(OUT_DIR, "results_register.csv")

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

BASELINES <- c(tt = 0.430, ig = 0.325, li = 0.360)

# Cyclic smooths in any model: shape-only output
CYCLIC_SMOOTHS <- c("local_hour", "weekday")

# Reference categories for multi-level factors per platform/model_source.
# Reads off levels(model$model[[var]])[1].
REF_CATS <- list(
  tt = list(media_type = NA_character_,        # constant on TT
            topic_cluster = "1",
            lang = NA_character_,                # determined at runtime
            account_type = NA_character_),       # determined at runtime
  ig = list(media_type = "carousel",
            topic_cluster = "1",
            lang = NA_character_,
            account_type = NA_character_),
  li = list(media_type = "article",
            topic_cluster = NA_character_,
            lang = NA_character_,
            account_type = NA_character_)
)

# -- Load all models ----------------------------------------------------------

cat("\n========================================================================\n")
cat("== build_results_register.R\n")
cat("========================================================================\n")
cat(sprintf("Timestamp: %s\n\n", Sys.time()))

cat("-- Loading models --\n")
models <- list(
  tt_inclusion       = readRDS(file.path(MODEL_DIR, "m_tt_inclusion.rds")),
  tt_rank            = readRDS(file.path(MODEL_DIR, "m_tt_rank.rds")),
  tt_velocity_t24    = readRDS(file.path(MODEL_DIR, "m_tt_velocity_24h.rds")),
  tt_velocity_t72    = readRDS(file.path(MODEL_DIR, "m_tt_velocity_72h.rds")),
  ig_inclusion_full  = readRDS(file.path(MODEL_DIR, "m_ig_inclusion.rds")),
  ig_inclusion       = readRDS(file.path(SENS_DIR,  "m_ig_inclusion_joined.rds")),
  ig_rank            = readRDS(file.path(MODEL_DIR, "m_ig_rank.rds")),
  li_inclusion       = readRDS(file.path(MODEL_DIR, "m_li_inclusion.rds")),
  li_rank            = readRDS(file.path(MODEL_DIR, "m_li_rank.rds")),
  li_velocity_t24    = readRDS(file.path(MODEL_DIR, "m_li_velocity_24h.rds"))
)

# Per-anchor metadata: one entry per chapter-anchor model.
# IG inclusion uses joined-only refit (per IG-specific stance).
# ig_inclusion_full is held as a separate auxiliary entry for footnote use.
ANCHORS <- tibble::tribble(
  ~anchor,           ~platform, ~outcome,        ~model_key,           ~model_file,                                                ~source_label,
  "tt_inclusion",    "tt",      "inclusion",     "tt_inclusion",       "05_modelling/05_gam/models/m_tt_inclusion.rds",            "summary(m_tt_inclusion)",
  "tt_rank",         "tt",      "rank",          "tt_rank",            "05_modelling/05_gam/models/m_tt_rank.rds",                 "summary(m_tt_rank)",
  "tt_velocity_t24", "tt",      "velocity_t24",  "tt_velocity_t24",    "05_modelling/05_gam/models/m_tt_velocity_24h.rds",         "summary(m_tt_velocity_24h)",
  "tt_velocity_t72", "tt",      "velocity_t72",  "tt_velocity_t72",    "05_modelling/05_gam/models/m_tt_velocity_72h.rds",         "summary(m_tt_velocity_72h)",
  "ig_inclusion",    "ig",      "inclusion",     "ig_inclusion",       "05_modelling/05_gam/sensitivity/m_ig_inclusion_joined.rds","summary(m_ig_inclusion_joined)",
  "ig_rank",         "ig",      "rank",          "ig_rank",            "05_modelling/05_gam/models/m_ig_rank.rds",                 "summary(m_ig_rank)",
  "li_inclusion",    "li",      "inclusion",     "li_inclusion",       "05_modelling/05_gam/models/m_li_inclusion.rds",            "summary(m_li_inclusion)",
  "li_rank",         "li",      "rank",          "li_rank",            "05_modelling/05_gam/models/m_li_rank.rds",                 "summary(m_li_rank)",
  "li_velocity_t24", "li",      "velocity_t24",  "li_velocity_t24",    "05_modelling/05_gam/models/m_li_velocity_24h.rds",         "summary(m_li_velocity_24h)"
)

cat(sprintf("  Loaded %d anchor models.\n", nrow(ANCHORS)))


# -- Load decision-level CSV inputs ------------------------------------------

cat("\n-- Loading factor_pp_per_level.csv and factor_omnibus_tests.csv --\n")
# pp_at_baseline must stay character: source CSV stores formatted "+/-X.X"
# strings; readr would auto-coerce them to numeric and strip the sign.
pp_per_level <- read_csv(file.path(TABLES_DIR, "factor_pp_per_level.csv"),
                         show_col_types = FALSE,
                         col_types = cols(pp_at_baseline = col_character()))
omnibus     <- read_csv(file.path(TABLES_DIR, "factor_omnibus_tests.csv"),
                        show_col_types = FALSE)
cat(sprintf("  factor_pp_per_level.csv: %d rows\n", nrow(pp_per_level)))
cat(sprintf("  factor_omnibus_tests.csv: %d rows\n", nrow(omnibus)))


# -- Helpers ------------------------------------------------------------------

sig_stars <- function(p) {
  if (is.na(p) || is.nan(p)) return("n.s.")
  if (p < 0.001) return("***")
  if (p < 0.01)  return("**")
  if (p < 0.05)  return("*")
  "n.s."
}

is_binomial <- function(model) family(model)$family == "binomial"
is_gamma    <- function(model) family(model)$family == "Gamma"
is_gaussian <- function(model) family(model)$family == "gaussian"

# Format numeric value to a stable string. Returns "" when NA.
fmt_val <- function(x, digits = 2) {
  if (is.na(x)) return(NA_character_)
  formatC(x, digits = digits, format = "f")
}

# Build the IQR-based partial-effect difference for a smooth term.
# Returns a list with link-scale delta (delta_link), the q25 and q75 values
# of the predictor used, and a non-monotonicity flag based on whether the
# partial effect is monotonic over the IQR window.
smooth_iqr_delta <- function(model, feature) {
  df <- model$model
  if (!feature %in% names(df)) return(NULL)
  x <- df[[feature]]
  if (!is.numeric(x)) return(NULL)
  qs <- quantile(x, probs = c(0.25, 0.50, 0.75), na.rm = TRUE,
                 names = FALSE)
  if (any(is.na(qs))) return(NULL)
  # If the predictor is essentially constant over the IQR (q25==q75) the IQR-
  # based delta is not defined. Returning a flag tells the caller to fall
  # through to a shape-only description.
  if (qs[1] == qs[3]) return(list(degenerate_iqr = TRUE,
                                  q25 = qs[1], q75 = qs[3]))

  # Build a newdata that holds all other variables at modal/median values
  newdf <- df[1, , drop = FALSE]
  for (nm in names(newdf)) {
    v <- df[[nm]]
    if (is.factor(v)) {
      newdf[[nm]] <- factor(levels(v)[1], levels = levels(v))
    } else if (is.logical(v)) {
      newdf[[nm]] <- FALSE
    } else if (is.numeric(v)) {
      newdf[[nm]] <- median(v, na.rm = TRUE)
    } else {
      newdf[[nm]] <- v[1]
    }
  }
  qs5 <- quantile(x, probs = c(0.10, 0.25, 0.50, 0.75, 0.90), na.rm = TRUE,
                  names = FALSE)
  newdf5 <- newdf[rep(1, 5), , drop = FALSE]
  newdf5[[feature]] <- qs5
  smooth_term <- paste0("s(", feature, ")")

  # type="terms" with a specific term: partial effect at link scale
  pe <- tryCatch(
    predict(model, newdata = newdf5, type = "terms", terms = smooth_term),
    error = function(e) NULL
  )
  if (is.null(pe)) return(NULL)
  partial <- as.numeric(pe[, smooth_term])
  if (any(!is.finite(partial))) return(NULL)

  delta_link <- partial[4] - partial[2]                  # q75 - q25
  range_link <- max(partial) - min(partial)              # over q10..q90
  # Heuristic: non-monotonic if partial reverses direction in q25..q75 window
  d1 <- partial[3] - partial[2]   # q50 - q25
  d2 <- partial[4] - partial[3]   # q75 - q50
  non_monotonic <- !is.na(d1) && !is.na(d2) && (sign(d1) * sign(d2) < 0)
  list(delta_link = delta_link,
       range_link = range_link,
       q25 = qs[1], q50 = qs[2], q75 = qs[3],
       non_monotonic = non_monotonic,
       partial_q10 = partial[1], partial_q25 = partial[2],
       partial_q50 = partial[3], partial_q75 = partial[4],
       partial_q90 = partial[5])
}

# Convert a log-scale delta (or beta) to the platform/outcome reporting scale.
# Returns a tibble of (value_unit, value_numeric, value_string).
convert_value <- function(delta_link, outcome, platform) {
  if (is.na(delta_link)) {
    return(tibble(value_unit = NA_character_,
                  value_numeric = NA_real_,
                  value_string = NA_character_))
  }
  if (outcome == "inclusion") {
    B <- BASELINES[[platform]]
    pp <- plogis(qlogis(B) + delta_link) - B
    return(tibble(value_unit = "pp",
                  value_numeric = pp * 100,
                  value_string = sprintf("%+.1f", pp * 100)))
  } else if (outcome == "rank") {
    return(tibble(value_unit = "positions",
                  value_numeric = delta_link,
                  value_string = sprintf("%+.2f", delta_link)))
  } else if (grepl("^velocity", outcome)) {
    pct <- (exp(delta_link) - 1) * 100
    return(tibble(value_unit = "percent",
                  value_numeric = pct,
                  value_string = sprintf("%+.1f", pct)))
  } else {
    return(tibble(value_unit = NA_character_,
                  value_numeric = NA_real_,
                  value_string = NA_character_))
  }
}

# Compute qualitative shape description for a cyclic or shape-only smooth.
shape_describe <- function(model, feature) {
  df <- model$model
  if (!feature %in% names(df)) return(NA_character_)
  x <- df[[feature]]
  rng <- range(x, na.rm = TRUE)
  newdf <- df[1, , drop = FALSE]
  for (nm in names(newdf)) {
    v <- df[[nm]]
    if (is.factor(v))      newdf[[nm]] <- factor(levels(v)[1], levels = levels(v))
    else if (is.logical(v)) newdf[[nm]] <- FALSE
    else if (is.numeric(v)) newdf[[nm]] <- median(v, na.rm = TRUE)
    else                    newdf[[nm]] <- v[1]
  }
  grid_x <- seq(rng[1], rng[2], length.out = 25)
  newdf25 <- newdf[rep(1, 25), , drop = FALSE]
  newdf25[[feature]] <- grid_x
  smooth_term <- paste0("s(", feature, ")")
  pe <- tryCatch(
    predict(model, newdata = newdf25, type = "terms", terms = smooth_term),
    error = function(e) NULL
  )
  if (is.null(pe)) return(NA_character_)
  p <- as.numeric(pe[, smooth_term])
  i_max <- which.max(p)
  i_min <- which.min(p)
  rng_link <- max(p) - min(p)
  if (feature == "weekday") {
    # weekday is encoded 0..6 (Python convention: Mon=0, Sun=6) in
    # df_*.parquet across all platforms, verified against posted_at_utc
    # ISO weekday. The figure assets fig_02_tt_inclusion.R (line 97) and
    # figure_tt_velocity_partial_effects.R (line 289) use day_labels[wd+1].
    # The earlier `max(1, min(7, round(wd)))` formula collapsed wd=0 and
    # wd=1 both to "Mon" and shifted everything else by one day; fixed.
    weekdays_ <- c("Mon","Tue","Wed","Thu","Fri","Sat","Sun")
    idx_pk <- max(0, min(6, round(grid_x[i_max]))) + 1
    idx_tr <- max(0, min(6, round(grid_x[i_min]))) + 1
    pk <- weekdays_[idx_pk]
    tr <- weekdays_[idx_tr]
    return(sprintf("cyclic; peak ~%s / trough ~%s; link-scale range %.2f",
                   pk, tr, rng_link))
  }
  if (feature == "local_hour") {
    return(sprintf("cyclic; peak ~%dh / trough ~%dh; link-scale range %.2f",
                   round(grid_x[i_max]) %% 24, round(grid_x[i_min]) %% 24,
                   rng_link))
  }
  sprintf("non-monotonic; peak at %s=%.2f, trough at %s=%.2f; link-scale range %.2f",
          feature, grid_x[i_max], feature, grid_x[i_min], rng_link)
}


# -- Identify all features per model -----------------------------------------

# Returns a tibble with one row per feature in the formula RHS, classifying it
# as "smooth", "smooth_cyclic", "binary", "linear", or "factor".
classify_features <- function(model) {
  fml <- formula(model)
  rhs <- attr(terms(fml), "term.labels")
  smooths <- if (length(model$smooth) > 0) {
    sapply(model$smooth, function(s) {
      list(term = s$term[1], bs = class(s)[1])
    }, simplify = FALSE)
  } else list()
  smooth_terms <- vapply(smooths, function(s) s$term, character(1))
  smooth_bs    <- vapply(smooths, function(s) s$bs,    character(1))
  smooth_is_cc <- grepl("cyclic", smooth_bs, ignore.case = TRUE) |
                  grepl("cc",      smooth_bs)

  # Factors
  factor_terms <- character()
  data <- model$model
  for (term in rhs) {
    bare_var <- gsub("^s\\(|,.*$|\\)$", "", term)
    if (bare_var %in% smooth_terms) next
    if (term %in% names(data) && is.factor(data[[term]])) {
      factor_terms <- c(factor_terms, term)
    }
  }

  # Binary parametric (logical, or numeric 0/1, or character with 2 unique
  # values) and linear continuous parametric
  binary_terms <- character()
  linear_terms <- character()
  for (term in rhs) {
    if (term %in% smooth_terms) next
    bare_var <- gsub("^s\\(|,.*$|\\)$", "", term)
    if (bare_var %in% smooth_terms) next
    if (!(term %in% names(data))) next
    if (term %in% factor_terms) next
    v <- data[[term]]
    if (is.logical(v)) {
      binary_terms <- c(binary_terms, term)
    } else if (is.numeric(v)) {
      uv <- unique(v[!is.na(v)])
      if (length(uv) <= 2 && all(uv %in% c(0, 1))) {
        binary_terms <- c(binary_terms, term)
      } else {
        linear_terms <- c(linear_terms, term)
      }
    } else if (is.character(v)) {
      uv <- unique(v[!is.na(v)])
      if (length(uv) == 2) {
        binary_terms <- c(binary_terms, term)
      } else {
        linear_terms <- c(linear_terms, term)
      }
    }
  }

  bind_rows(
    tibble(feature = smooth_terms,
           kind = ifelse(smooth_is_cc, "smooth_cyclic", "smooth")),
    tibble(feature = binary_terms, kind = "binary"),
    tibble(feature = linear_terms, kind = "linear"),
    tibble(feature = factor_terms, kind = "factor")
  )
}


# -- Extraction helpers per kind ---------------------------------------------

extract_smooth_row <- function(anchor_row, model, feature, cyclic = FALSE) {
  s <- summary(model)
  st <- s$s.table
  smooth_term <- paste0("s(", feature, ")")
  st_row <- if (smooth_term %in% rownames(st)) st[smooth_term, ] else NULL
  if (is.null(st_row)) {
    # smooth not on default symbol (cyclic might have bs annotation)
    candidates <- grep(paste0("^s\\(", feature, "(?:,|\\))"),
                       rownames(st), value = TRUE)
    if (length(candidates) > 0) st_row <- st[candidates[1], ]
  }
  if (is.null(st_row)) {
    return(tibble(
      feature = feature, platform = anchor_row$platform,
      outcome = anchor_row$outcome,
      value_unit = NA_character_, value = NA_character_,
      p_value = NA_real_, sig_marker = "n.s.",
      n_obs = nrow(model$model), model_file = anchor_row$model_file,
      source_row = sprintf("%s$s.table", anchor_row$source_label),
      reference_category = NA_character_,
      notes = "smooth term not found in summary"))
  }
  edf <- unname(st_row["edf"])
  stat <- if ("Chi.sq" %in% colnames(st)) unname(st_row["Chi.sq"]) else unname(st_row["F"])
  stat_label <- if ("Chi.sq" %in% colnames(st)) "chi-sq" else "F"
  p <- unname(st_row["p-value"])

  if (cyclic) {
    desc <- shape_describe(model, feature)
    return(tibble(
      feature = feature, platform = anchor_row$platform,
      outcome = anchor_row$outcome,
      value_unit = "shape-only",
      value = desc,
      p_value = p, sig_marker = sig_stars(p),
      n_obs = nrow(model$model), model_file = anchor_row$model_file,
      source_row = sprintf("%s$s.table[%s]", anchor_row$source_label, smooth_term),
      reference_category = NA_character_,
      notes = sprintf("cyclic smooth (bs=cc); edf=%.2f; %s=%.2f",
                      edf, stat_label, stat)))
  }

  iqr <- smooth_iqr_delta(model, feature)
  if (is.null(iqr)) {
    return(tibble(
      feature = feature, platform = anchor_row$platform,
      outcome = anchor_row$outcome,
      value_unit = NA_character_, value = NA_character_,
      p_value = p, sig_marker = sig_stars(p),
      n_obs = nrow(model$model), model_file = anchor_row$model_file,
      source_row = sprintf("%s$s.table[%s]", anchor_row$source_label, smooth_term),
      reference_category = NA_character_,
      notes = sprintf("smooth IQR delta unavailable; edf=%.2f; %s=%.2f",
                      edf, stat_label, stat)))
  }
  # Predictor effectively constant over IQR (e.g. LI velocity log_follower):
  # report shape-only with a qualitative descriptor.
  if (isTRUE(iqr$degenerate_iqr)) {
    desc <- shape_describe(model, feature)
    return(tibble(
      feature = feature, platform = anchor_row$platform,
      outcome = anchor_row$outcome,
      value_unit = "shape-only",
      value = desc,
      p_value = p, sig_marker = sig_stars(p),
      n_obs = nrow(model$model), model_file = anchor_row$model_file,
      source_row = sprintf("%s$s.table[%s] (q25==q75 in fitting frame; IQR delta degenerate)",
                           anchor_row$source_label, smooth_term),
      reference_category = NA_character_,
      notes = sprintf("predictor essentially constant over IQR (q25=q75=%.3g); shape evaluated over q10..q90 instead; edf=%.2f; %s=%.2f",
                      iqr$q25, edf, stat_label, stat)))
  }

  # Decide whether the smooth is shape-only based on cyclic-only rule (above) or
  # heavily non-monotonic. We keep IQR value but flag.
  conv <- convert_value(iqr$delta_link, anchor_row$outcome, anchor_row$platform)
  shape_note <- if (iqr$non_monotonic) {
    sprintf("non-monotonic in IQR (q25=%.3g, q50=%.3g, q75=%.3g of %s; link-range over q10..q90 = %.2f)",
            iqr$q25, iqr$q50, iqr$q75, feature, iqr$range_link)
  } else {
    sprintf("monotonic over IQR (q25=%.3g, q75=%.3g of %s)",
            iqr$q25, iqr$q75, feature)
  }
  notes_str <- sprintf("smooth IQR (q75-q25); edf=%.2f; %s=%.2f; %s",
                       edf, stat_label, stat, shape_note)

  tibble(
    feature = feature, platform = anchor_row$platform,
    outcome = anchor_row$outcome,
    value_unit = conv$value_unit,
    value = conv$value_string,
    p_value = p, sig_marker = sig_stars(p),
    n_obs = nrow(model$model), model_file = anchor_row$model_file,
    source_row = sprintf("%s$s.table[%s] + IQR partial(q25,q75)",
                         anchor_row$source_label, smooth_term),
    reference_category = NA_character_,
    notes = notes_str)
}

extract_parametric_row <- function(anchor_row, model, feature, kind) {
  s <- summary(model)
  pt <- s$p.table
  if (!(feature %in% rownames(pt))) {
    return(tibble(
      feature = feature, platform = anchor_row$platform,
      outcome = anchor_row$outcome,
      value_unit = NA_character_, value = NA_character_,
      p_value = NA_real_, sig_marker = "n.s.",
      n_obs = nrow(model$model), model_file = anchor_row$model_file,
      source_row = sprintf("%s$p.table", anchor_row$source_label),
      reference_category = NA_character_,
      notes = sprintf("%s parametric term not found", kind)))
  }
  row <- pt[feature, ]
  beta <- unname(row["Estimate"])
  se   <- unname(row["Std. Error"])
  test_stat <- unname(row[3])
  p    <- unname(row[4])
  test_label <- if (is_binomial(model)) "z" else "t"

  # Aliased coefficient (e.g. uses_named_audio in TT inclusion; audio_is_original
  # in TT rank): Estimate=0, SE=0
  if (!is.na(beta) && !is.na(se) && beta == 0 && se == 0) {
    return(tibble(
      feature = feature, platform = anchor_row$platform,
      outcome = anchor_row$outcome,
      value_unit = NA_character_, value = NA_character_,
      p_value = NA_real_, sig_marker = "n.s.",
      n_obs = nrow(model$model), model_file = anchor_row$model_file,
      source_row = sprintf("%s$p.table[%s]", anchor_row$source_label, feature),
      reference_category = NA_character_,
      notes = sprintf("%s parametric: aliased (coef=0, SE=0); not estimable",
                      kind)))
  }
  # Separation diagnostic: |estimate| > 100 with SE > 1e3 means full separation
  is_sep <- !is.na(beta) && !is.na(se) && abs(beta) > 100 && se > 1e3
  if (is_sep) {
    return(tibble(
      feature = feature, platform = anchor_row$platform,
      outcome = anchor_row$outcome,
      value_unit = NA_character_, value = NA_character_,
      p_value = p, sig_marker = "n.s.",
      n_obs = nrow(model$model), model_file = anchor_row$model_file,
      source_row = sprintf("%s$p.table[%s]", anchor_row$source_label, feature),
      reference_category = NA_character_,
      notes = sprintf("complete separation (beta=%.2f, SE=%.2g); not estimable",
                      beta, se)))
  }
  conv <- convert_value(beta, anchor_row$outcome, anchor_row$platform)
  notes_str <- sprintf("%s; beta=%.4f, SE=%.4f, %s=%.2f",
                       kind, beta, se, test_label, test_stat)
  tibble(
    feature = feature, platform = anchor_row$platform,
    outcome = anchor_row$outcome,
    value_unit = conv$value_unit,
    value = conv$value_string,
    p_value = p, sig_marker = sig_stars(p),
    n_obs = nrow(model$model), model_file = anchor_row$model_file,
    source_row = sprintf("%s$p.table[%s]", anchor_row$source_label, feature),
    reference_category = NA_character_,
    notes = notes_str)
}

# Extract one row for the omnibus test of a multi-level factor on the model.
# Uses pre-computed factor_omnibus_tests.csv when available; else falls back to
# pTerms.table or anova.gam(reduced) on the fly. For cross-platform Table 7
# topic_cluster this is the F2 cell value.
extract_factor_omnibus_row <- function(anchor_row, model, feature) {
  s <- summary(model)
  pt_tab <- s$pTerms.table
  ref_cat <- if (feature %in% names(REF_CATS[[anchor_row$platform]])) {
    REF_CATS[[anchor_row$platform]][[feature]]
  } else NA_character_
  # If the registered REF_CATS entry is NA, read levels(model$model[[feature]])[1]
  if (is.na(ref_cat) && feature %in% names(model$model)) {
    lv <- levels(model$model[[feature]])
    if (length(lv) > 0) ref_cat <- lv[1]
  }

  # Map anchor name -> column value used in factor_omnibus_tests.csv. The CSV
  # uses ig_inclusion_joined (joined-only refit, per IG-specific stance) and
  # tt_velocity_24h (mgcv velocity_24h_g model name); other anchors match.
  csv_model_id <- switch(
    anchor_row$anchor,
    ig_inclusion    = "ig_inclusion_joined",
    tt_velocity_t24 = "tt_velocity_24h",
    tt_velocity_t72 = "tt_velocity_72h",
    li_velocity_t24 = "li_velocity_24h",
    anchor_row$anchor
  )
  csv_row <- omnibus[omnibus$model == csv_model_id & omnibus$feature == feature, ]
  if (nrow(csv_row) == 1) {
    stat_value <- csv_row$statistic_value
    df_val <- csv_row$df
    p <- csv_row$p_value
    note <- csv_row$note
    if (!is.na(p)) {
      stat_lbl <- if (csv_row$statistic_type == "Chisq") "chi-sq" else "F"
      val_str <- if (is.na(stat_value)) NA_character_ else
                 sprintf("%s(%.2f) = %.2f", stat_lbl, df_val, stat_value)
      note_str <- sprintf("omnibus from factor_omnibus_tests.csv (anova.gam two-model deviance test)%s",
                          ifelse(is.na(note) || note == "",
                                 "", paste0("; ", note)))
      return(tibble(
        feature = feature, platform = anchor_row$platform,
        outcome = anchor_row$outcome,
        value_unit = stat_lbl,
        value = val_str,
        p_value = p, sig_marker = sig_stars(p),
        n_obs = csv_row$n_obs, model_file = anchor_row$model_file,
        source_row = sprintf("factor_omnibus_tests.csv [model=%s, feature=%s]",
                             csv_model_id, feature),
        reference_category = ref_cat,
        notes = note_str))
    } else {
      # CSV row exists but is constant (TT media_type)
      return(tibble(
        feature = feature, platform = anchor_row$platform,
        outcome = anchor_row$outcome,
        value_unit = "n/a",
        value = "n/a",
        p_value = NA_real_, sig_marker = "n.s.",
        n_obs = csv_row$n_obs, model_file = anchor_row$model_file,
        source_row = sprintf("factor_omnibus_tests.csv [model=%s, feature=%s]",
                             csv_model_id, feature),
        reference_category = ref_cat,
        notes = sprintf("structurally constant on TikTok; test not defined (%s)",
                        note)))
    }
  }

  # Fallback: pTerms.table
  if (!is.null(pt_tab) && feature %in% rownames(pt_tab)) {
    row <- pt_tab[feature, ]
    df_val <- unname(row["df"])
    p <- unname(row["p-value"])
    stat_col <- setdiff(colnames(pt_tab), c("df", "p-value"))[1]
    stat_value <- unname(row[stat_col])
    stat_lbl <- if (stat_col == "Chi.sq") "chi-sq" else "F"
    val_str <- sprintf("%s(%d) = %.2f", stat_lbl, as.integer(df_val), stat_value)
    return(tibble(
      feature = feature, platform = anchor_row$platform,
      outcome = anchor_row$outcome,
      value_unit = stat_lbl,
      value = val_str,
      p_value = p, sig_marker = sig_stars(p),
      n_obs = nrow(model$model), model_file = anchor_row$model_file,
      source_row = sprintf("%s$pTerms.table[%s]",
                           anchor_row$source_label, feature),
      reference_category = ref_cat,
      notes = sprintf("omnibus Wald via pTerms.table (parametric factor)")))
  }

  tibble(
    feature = feature, platform = anchor_row$platform,
    outcome = anchor_row$outcome,
    value_unit = NA_character_, value = NA_character_,
    p_value = NA_real_, sig_marker = "n.s.",
    n_obs = nrow(model$model), model_file = anchor_row$model_file,
    source_row = sprintf("%s$pTerms.table", anchor_row$source_label),
    reference_category = ref_cat,
    notes = "factor omnibus not found")
}

# Per-level rows for media_type and topic_cluster (Convention F1).
# Strategy:
#   - For inclusion: prefer factor_pp_per_level.csv when available; otherwise
#     compute pp = plogis(qlogis(B) + beta) - B from p.table.
#   - For rank: positions = beta directly from p.table (Gaussian identity).
#   - For velocity: percent = (exp(beta)-1)*100 from p.table.
extract_factor_level_rows <- function(anchor_row, model, feature) {
  s <- summary(model)
  pt <- s$p.table
  data <- model$model
  if (!feature %in% names(data)) return(tibble())
  lv <- levels(data[[feature]])
  if (length(lv) < 2) return(tibble())
  ref <- lv[1]
  results <- list()
  csv_platform <- if (anchor_row$platform == "tt") "tiktok"
                  else if (anchor_row$platform == "ig") "instagram"
                  else "linkedin"
  csv_model_source <- if (anchor_row$anchor == "ig_inclusion") "joined_only" else "full"

  for (level in lv[-1]) {
    feat_label <- sprintf("%s:%s", feature, level)
    rn <- paste0(feature, level)

    # Cell counts in this fitting frame to detect separation
    outcome_col <- data[[1]]
    in_lvl <- as.character(data[[feature]]) == level
    if (anchor_row$outcome == "inclusion") {
      n0 <- sum(in_lvl & outcome_col == 0, na.rm = TRUE)
      n1 <- sum(in_lvl & outcome_col == 1, na.rm = TRUE)
      sep_check <- !is.na(n0) && !is.na(n1) && (n0 == 0 || n1 == 0)
    } else {
      sep_check <- FALSE
    }

    if (!(rn %in% rownames(pt))) {
      results[[length(results)+1]] <- tibble(
        feature = feat_label, platform = anchor_row$platform,
        outcome = anchor_row$outcome,
        value_unit = NA_character_, value = NA_character_,
        p_value = NA_real_, sig_marker = "n.s.",
        n_obs = nrow(data), model_file = anchor_row$model_file,
        source_row = sprintf("%s$p.table[%s]", anchor_row$source_label, rn),
        reference_category = ref,
        notes = sprintf("level absent (droplevels removed it from refit)"))
      next
    }
    row <- pt[rn, ]
    beta <- unname(row["Estimate"])
    se <- unname(row["Std. Error"])
    test_stat <- unname(row[3])
    p <- unname(row[4])

    # IG joined-only inclusion: prefer factor_pp_per_level.csv values for
    # full match (audit trail). For rank/velocity, fall back to direct
    # extraction since the CSV does not cover those.
    csv_row <- pp_per_level[
      pp_per_level$platform == csv_platform &
      pp_per_level$model_source == csv_model_source &
      pp_per_level$feature == feature &
      as.character(pp_per_level$level) == as.character(level),
    ]
    use_csv <- nrow(csv_row) == 1 && anchor_row$outcome == "inclusion"

    cell_sep    <- isTRUE(sep_check)
    coef_sep    <- !is.na(beta) && !is.na(se) && abs(beta) > 100 && se > 1e3
    is_sep      <- cell_sep || coef_sep
    if (is_sep) {
      sep_kind <- if (cell_sep) "cell"
                  else "coupled/rank-deficient"
      cell_str <- if (anchor_row$outcome == "inclusion")
                    sprintf("n_top=%d, n_baseline=%d at level=%s",
                            n1, n0, level)
                  else sprintf("level=%s", level)
      sensitivity_note <- if (anchor_row$anchor == "ig_inclusion")
                            "; full-model beta not robust to source"
                          else ""
      results[[length(results)+1]] <- tibble(
        feature = feat_label, platform = anchor_row$platform,
        outcome = anchor_row$outcome,
        value_unit = "pp", value = NA_character_,
        p_value = NA_real_, sig_marker = "n.s.",
        n_obs = nrow(data), model_file = anchor_row$model_file,
        source_row = sprintf("%s$p.table[%s] + cell-count check",
                             anchor_row$source_label, rn),
        reference_category = ref,
        notes = sprintf("%s separation in fitting frame (%s); per-level not estimable%s",
                        sep_kind, cell_str, sensitivity_note))
      next
    }

    if (use_csv) {
      # pp_at_baseline column is read as character (col_types = col_character()),
      # so the existing "+/-X.X" formatting is preserved. Defensive re-format
      # in case the upstream CSV format changes to numeric.
      raw_pp <- csv_row$pp_at_baseline
      pp_str <- if (is.character(raw_pp) && grepl("^[+-]", raw_pp)) raw_pp
                else if (!is.na(suppressWarnings(as.numeric(raw_pp))))
                  sprintf("%+.1f", as.numeric(raw_pp))
                else NA_character_
      results[[length(results)+1]] <- tibble(
        feature = feat_label, platform = anchor_row$platform,
        outcome = anchor_row$outcome,
        value_unit = "pp",
        value = pp_str,
        p_value = csv_row$p_value, sig_marker = sig_stars(csv_row$p_value),
        n_obs = nrow(data), model_file = anchor_row$model_file,
        source_row = sprintf("factor_pp_per_level.csv [platform=%s, model_source=%s, feature=%s, level=%s]",
                             csv_platform, csv_model_source, feature, level),
        reference_category = ref,
        notes = sprintf("F1 per-level pp at platform baseline %.3f; beta=%.4f, SE=%.4f",
                        BASELINES[[anchor_row$platform]], beta, se))
      next
    }

    conv <- convert_value(beta, anchor_row$outcome, anchor_row$platform)
    note_main <- sprintf(
      "F1 per-level at platform baseline (%.3f); beta=%.4f, SE=%.4f, %s=%.2f",
      BASELINES[[anchor_row$platform]], beta, se,
      ifelse(is_binomial(model), "z", "t"), test_stat)
    results[[length(results)+1]] <- tibble(
      feature = feat_label, platform = anchor_row$platform,
      outcome = anchor_row$outcome,
      value_unit = conv$value_unit,
      value = conv$value_string,
      p_value = p, sig_marker = sig_stars(p),
      n_obs = nrow(data), model_file = anchor_row$model_file,
      source_row = sprintf("%s$p.table[%s]", anchor_row$source_label, rn),
      reference_category = ref,
      notes = note_main)
  }
  bind_rows(results)
}


# -- Build all rows ----------------------------------------------------------

cat("\n-- Building rows --\n")
all_rows <- list()
for (i in seq_len(nrow(ANCHORS))) {
  anchor_row <- ANCHORS[i, ]
  m <- models[[anchor_row$model_key]]
  cat(sprintf("  anchor=%s (N=%d, family=%s)\n",
              anchor_row$anchor, nrow(m$model), family(m)$family))
  feats <- classify_features(m)
  for (j in seq_len(nrow(feats))) {
    fr <- feats[j, ]
    if (fr$kind == "smooth") {
      r <- extract_smooth_row(anchor_row, m, fr$feature, cyclic = FALSE)
    } else if (fr$kind == "smooth_cyclic") {
      r <- extract_smooth_row(anchor_row, m, fr$feature, cyclic = TRUE)
    } else if (fr$kind == "binary") {
      r <- extract_parametric_row(anchor_row, m, fr$feature, "binary")
    } else if (fr$kind == "linear") {
      r <- extract_parametric_row(anchor_row, m, fr$feature, "linear")
    } else if (fr$kind == "factor") {
      r <- extract_factor_omnibus_row(anchor_row, m, fr$feature)
      # Add per-level rows for media_type and topic_cluster only
      if (fr$feature %in% c("media_type", "topic_cluster")) {
        r2 <- extract_factor_level_rows(anchor_row, m, fr$feature)
        r <- bind_rows(r, r2)
      }
    } else {
      next
    }
    all_rows[[length(all_rows)+1]] <- r
  }
}

register <- bind_rows(all_rows)


# -- Add structural rows for missing-by-design cells -------------------------

# TT media_type does not appear in any TT model formula (constant). Add a
# structural row per outcome so the cross-platform Table 7 cells (and any
# prose) have a register entry to point at.
tt_outcomes <- c("inclusion", "rank", "velocity_t24", "velocity_t72")
tt_extras <- bind_rows(lapply(tt_outcomes, function(o) {
  anchor_row <- ANCHORS[ANCHORS$anchor == paste0("tt_", o), ][1,]
  tibble(
    feature = "media_type", platform = "tt", outcome = o,
    value_unit = "n/a", value = "n/a",
    p_value = NA_real_, sig_marker = "n.s.",
    n_obs = nrow(models[[anchor_row$model_key]]$model),
    model_file = anchor_row$model_file,
    source_row = "structural (TT all video)",
    reference_category = NA_character_,
    notes = "structurally constant on TikTok; test not defined")
}))
register <- bind_rows(register, tt_extras)

# IG velocity is withdrawn (Decision 37). Add explicit n/a rows for every
# feature that exists in the IG analytical table but is not in any IG model
# formula for velocity. This lets the cross-platform Table 7 register a clear
# "withdrawn" status. Using a single feature='__platform_outcome__' marker is
# noisy; instead, add ONE row per (feature x ig x velocity_t24 / velocity_t72)
# only for the chapter-cited features (none, since the chapter does not cite
# IG velocity at all). Convention: include a single sentinel row per outcome.
ig_velocity_extras <- bind_rows(
  tibble(feature = "__withdrawn__", platform = "ig", outcome = "velocity_t24",
         value_unit = "n/a", value = "n/a",
         p_value = NA_real_, sig_marker = "n.s.",
         n_obs = NA_integer_, model_file = NA_character_,
         source_row = "thesis methodology decision log: Decision 37",
         reference_category = NA_character_,
         notes = "Instagram velocity withdrawn from RQ1 reporting (Decision 37); no IG velocity model anchored in the chapter"),
  tibble(feature = "__withdrawn__", platform = "ig", outcome = "velocity_t72",
         value_unit = "n/a", value = "n/a",
         p_value = NA_real_, sig_marker = "n.s.",
         n_obs = NA_integer_, model_file = NA_character_,
         source_row = "thesis methodology decision log: Decision 37",
         reference_category = NA_character_,
         notes = "Instagram velocity withdrawn from RQ1 reporting (Decision 37); no IG velocity model anchored in the chapter")
)
register <- bind_rows(register, ig_velocity_extras)


# -- Final ordering & write ---------------------------------------------------

platform_levels <- c("tt", "ig", "li")
outcome_levels  <- c("inclusion", "rank", "velocity_t24", "velocity_t72")
register <- register |>
  mutate(
    platform = factor(platform, levels = platform_levels),
    outcome  = factor(outcome,  levels = outcome_levels)
  ) |>
  arrange(platform, outcome, feature) |>
  mutate(platform = as.character(platform), outcome = as.character(outcome))

cat(sprintf("\nWriting %d rows to %s\n", nrow(register), OUT_CSV))
write.csv(register, OUT_CSV, row.names = FALSE, na = "")

cat("\n-- Summary --\n")
print(register |> count(platform, outcome), n = Inf)
cat("\nFeature kind breakdown:\n")
register |>
  mutate(
    is_factor_level = grepl(":", feature),
    is_omnibus = !is_factor_level & feature %in% c("media_type","topic_cluster","lang","account_type"),
    is_smooth_shape = value_unit == "shape-only",
    is_separation_unidentifiable = is.na(value) & grepl("separation|UNIDENTIFIABLE", notes, ignore.case = TRUE),
    is_constant = value_unit == "n/a"
  ) |>
  summarise(
    factor_levels = sum(is_factor_level, na.rm = TRUE),
    omnibus_factor = sum(is_omnibus, na.rm = TRUE),
    shape_only = sum(is_smooth_shape, na.rm = TRUE),
    separation = sum(is_separation_unidentifiable, na.rm = TRUE),
    constant = sum(is_constant, na.rm = TRUE),
    total = n()
  ) |>
  print()

cat("\nDone.\n")
invisible(register)
