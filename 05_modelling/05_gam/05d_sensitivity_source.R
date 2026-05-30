# =============================================================================
# 05d_sensitivity_source.R
#
# Master's thesis: Learning the Levers — Which Creator-Controllable
# Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
# from a Swiss German-Language Perspective
# Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026
#
# Purpose:
#   Sensitivity analyses for the inclusion GAMs. Part A refits inclusion
#   models on the joined-only capture subset to test whether feature
#   significance is driven by scraper instrumentation. Part B fits a
#   reduced-feature LinkedIn inclusion GAM to check whether the step
#   failure in the full model disappears with fewer features.
#
# Pipeline position:
#   Step 5d of the modelling pipeline. Depends on 05a_gam_inclusion.R;
#   consumed by Chapter 4 robustness checks and the appendix tables.
#
# Inputs:
#   04_database/scraper.db                                          SQLite raw database (source field)
#   05_modelling/03_data_prep/data/df_tt.parquet                    TikTok analytical frame
#   05_modelling/03_data_prep/data/df_ig.parquet                    Instagram analytical frame
#   05_modelling/03_data_prep/data/df_li.parquet                    LinkedIn analytical frame
#   05_modelling/05_gam/models/m_{tt,ig,li}_inclusion.rds           Full inclusion GAMs
#
# Outputs:
#   05_modelling/05_gam/output/sensitivity_source_breakdown.csv     Source x ever_top counts
#   05_modelling/05_gam/output/sensitivity_source_comparison.csv    Full vs joined comparison
#   05_modelling/05_gam/output/sensitivity_li_reduced.csv           LinkedIn reduced-model comparison
#
# Usage:
#   Rscript 05_modelling/05_gam/05d_sensitivity_source.R
# =============================================================================

setwd(here::here())
source("05_modelling/config/packages.R")
library(gratia)
library(patchwork)

PREP_DIR  <- file.path(BASE_DIR, "03_data_prep", "data")
MODEL_DIR <- file.path(BASE_DIR, "05_gam", "models")
OUT_DIR   <- file.path(BASE_DIR, "05_gam", "output")

cat("\n================================================================\n")
cat("== 05d: Sensitivity Analysis ==\n")
cat("================================================================\n")
cat(sprintf("  Timestamp: %s\n\n", Sys.time()))


# =============================================================================
# PART A: Source Sensitivity Check
# =============================================================================

cat("== PART A: Source sensitivity check ==\n")

# -- A1. Query database for capture source per post ----------------------------
cat("\n-- A1. Querying scraper.db for capture source --\n")

con <- dbConnect(SQLite(), DB_PATH)

source_per_post <- dbGetQuery(con, "
  SELECT p.post_id, p.platform,
         c.source,
         COUNT(*) AS n_captures
  FROM posts p
  JOIN captures c ON c.post_id = p.post_id
  GROUP BY p.post_id, p.platform, c.source
") |> as_tibble()

dbDisconnect(con)
cat(sprintf("  Source-level rows: %d\n", nrow(source_per_post)))

# Determine dominant source per post (most frequent capture method)
dominant_source <- source_per_post |>
  group_by(post_id, platform) |>
  slice_max(n_captures, n = 1, with_ties = FALSE) |>
  ungroup() |>
  select(post_id, source_dominant = source)

cat(sprintf("  Posts with source assigned: %d\n", nrow(dominant_source)))


# -- A2. Load platform data and join source ------------------------------------
cat("\n-- A2. Loading platform data and joining source --\n")

df_tt <- read_parquet(file.path(PREP_DIR, "df_tt.parquet")) |>
  left_join(dominant_source, by = "post_id")
df_ig <- read_parquet(file.path(PREP_DIR, "df_ig.parquet")) |>
  left_join(dominant_source, by = "post_id")
df_li <- read_parquet(file.path(PREP_DIR, "df_li.parquet")) |>
  left_join(dominant_source, by = "post_id")

# Ensure factors
factor_vars <- c("lang", "media_type", "topic_cluster", "account_type")
for (v in factor_vars) {
  if (v %in% names(df_tt)) df_tt[[v]] <- as.factor(df_tt[[v]])
  if (v %in% names(df_ig)) df_ig[[v]] <- as.factor(df_ig[[v]])
  if (v %in% names(df_li)) df_li[[v]] <- as.factor(df_li[[v]])
}


# -- A3. Source breakdown (descriptive) ----------------------------------------
cat("\n-- A3. Source breakdown --\n")

source_breakdown <- bind_rows(
  df_tt |> count(platform, source_dominant, ever_top, name = "n"),
  df_ig |> count(platform, source_dominant, ever_top, name = "n"),
  df_li |> count(platform, source_dominant, ever_top, name = "n")
)

write.csv(source_breakdown, file.path(OUT_DIR, "sensitivity_source_breakdown.csv"),
          row.names = FALSE)

# Print summary
for (plat in c("tiktok", "instagram", "linkedin")) {
  sub <- source_breakdown |> filter(platform == plat)
  total <- sum(sub$n)
  cat(sprintf("\n  %s (N=%d):\n", plat, total))
  for (src in unique(sub$source_dominant)) {
    src_n <- sum(sub$n[sub$source_dominant == src])
    src_top <- sum(sub$n[sub$source_dominant == src & sub$ever_top == 1])
    cat(sprintf("    %-8s  N=%5d (%4.1f%%)  top-rate=%.1f%%\n",
                src, src_n, src_n/total*100, src_top/src_n*100))
  }
}


# -- A4. Filter to joined-only and refit inclusion GAMs ------------------------
cat("\n\n-- A4. Refitting inclusion GAMs on joined-only subset --\n")

df_tt_j <- df_tt |> filter(source_dominant == "joined")
df_ig_j <- df_ig |> filter(source_dominant == "joined")
df_li_j <- df_li |> filter(source_dominant == "joined")

cat(sprintf("  Joined-only N: TT=%d (%.0f%%), IG=%d (%.0f%%), LI=%d (%.0f%%)\n",
            nrow(df_tt_j), nrow(df_tt_j)/nrow(df_tt)*100,
            nrow(df_ig_j), nrow(df_ig_j)/nrow(df_ig)*100,
            nrow(df_li_j), nrow(df_li_j)/nrow(df_li)*100))

# Load full models to extract formulas
m_tt_full <- readRDS(file.path(MODEL_DIR, "m_tt_inclusion.rds"))
m_ig_full <- readRDS(file.path(MODEL_DIR, "m_ig_inclusion.rds"))
m_li_full <- readRDS(file.path(MODEL_DIR, "m_li_inclusion.rds"))

# Helper: adjust formula if joined-only subset has fewer unique values
adjust_formula_for_data <- function(model, data) {
  fml <- formula(model)
  fml_str <- paste(deparse(fml, width.cutoff = 500), collapse = " ")

  for (si in model$smooth) {
    varname <- si$term[1]
    k_val   <- si$bs.dim

    if (!varname %in% names(data)) next
    n_uniq <- length(unique(data[[varname]][!is.na(data[[varname]])]))

    if (n_uniq < k_val) {
      pattern <- sprintf("s\\(%s[^)]*\\)", varname)
      if (grepl(pattern, fml_str)) {
        if (n_uniq <= 3) {
          fml_str <- sub(pattern, varname, fml_str)
          cat(sprintf("    k-fix: s(%s, k=%d) -> %s [linear] (n_unique=%d)\n",
                      varname, k_val, varname, n_uniq))
        } else {
          new_k <- n_uniq
          new_term <- sub(sprintf("k\\s*=\\s*%d", k_val),
                          sprintf("k = %d", new_k),
                          regmatches(fml_str, regexpr(pattern, fml_str)))
          fml_str <- sub(pattern, new_term, fml_str)
          cat(sprintf("    k-fix: s(%s, k=%d) -> s(%s, k=%d) (n_unique=%d)\n",
                      varname, k_val, varname, new_k, n_uniq))
        }
      }
    }
  }
  as.formula(fml_str)
}

# Helper: extract significant terms from a GAM
extract_sig_terms <- function(model, label) {
  sm <- summary(model)$s.table
  pt <- summary(model)$p.table
  p_col <- if ("Pr(>|z|)" %in% colnames(pt)) "Pr(>|z|)" else "Pr(>|t|)"

  terms <- tibble(feature = character(), p_value = numeric(),
                  significant = logical(), source = character())

  # Smooth terms
  if (!is.null(sm) && nrow(sm) > 0) {
    for (i in seq_len(nrow(sm))) {
      feat <- gsub("^s\\(([^,)]+).*\\)$", "\\1", rownames(sm)[i])
      terms <- bind_rows(terms, tibble(
        feature = feat, p_value = sm[i, "p-value"],
        significant = sm[i, "p-value"] < 0.05, source = label
      ))
    }
  }

  # Parametric terms (aggregate factor levels to feature level)
  if (!is.null(pt) && nrow(pt) > 0) {
    for (i in seq_len(nrow(pt))) {
      term <- rownames(pt)[i]
      if (term == "(Intercept)") next
      if (is.nan(pt[i, "Estimate"])) next  # aliased

      # Strip factor level suffixes
      feat <- term
      for (fv in c("lang", "media_type", "topic_cluster", "account_type")) {
        if (startsWith(term, fv)) { feat <- fv; break }
      }

      terms <- bind_rows(terms, tibble(
        feature = feat, p_value = pt[i, p_col],
        significant = pt[i, p_col] < 0.05, source = label
      ))
    }
  }

  # For factors with multiple levels, keep minimum p-value
  terms |>
    group_by(feature, source) |>
    summarise(p_value = min(p_value, na.rm = TRUE),
              significant = any(significant),
              .groups = "drop")
}

# Knot definitions
knots_tt_li <- list(local_hour = c(0, 24), weekday = c(0, 7))

# Refit models
platforms <- list(
  list(name = "TikTok",    code = "tt", data_full = df_tt, data_joined = df_tt_j,
       model_full = m_tt_full, knots = knots_tt_li),
  list(name = "Instagram", code = "ig", data_full = df_ig, data_joined = df_ig_j,
       model_full = m_ig_full, knots = NULL),
  list(name = "LinkedIn",  code = "li", data_full = df_li, data_joined = df_li_j,
       model_full = m_li_full, knots = knots_tt_li)
)

comparison_rows <- list()

for (p in platforms) {
  n_joined <- nrow(p$data_joined)
  cat(sprintf("\n  Refitting %s inclusion on joined-only (N=%d) ...\n",
              p$name, n_joined))

  # Skip platforms with no joined posts (e.g. LinkedIn = all api)
  if (n_joined < 50) {
    cat(sprintf("    SKIP: Only %d joined posts (platform uses '%s' capture).\n",
                n_joined,
                paste(unique(p$data_full$source_dominant), collapse = "/")))
    cat("    This is itself a finding: source sensitivity is not testable here.\n")
    next
  }

  # Check class balance in joined subset
  top_rate_j <- mean(p$data_joined$ever_top)
  top_rate_f <- mean(p$data_full$ever_top)
  cat(sprintf("    ever_top rate: joined=%.1f%% (full=%.1f%%)\n",
              top_rate_j * 100, top_rate_f * 100))
  if (top_rate_j < 0.03 || top_rate_j > 0.97) {
    cat("    WARNING: Extreme class imbalance in joined subset. Results may be unstable.\n")
  }

  # Adjust formula for joined subset
  fml_j <- adjust_formula_for_data(p$model_full, p$data_joined)

  # Align factor levels: drop levels absent in joined subset
  df_j <- p$data_joined
  for (v in factor_vars) {
    if (v %in% names(df_j) && is.factor(df_j[[v]])) {
      df_j[[v]] <- droplevels(df_j[[v]])
    }
  }

  # Refit (suppress warnings, catch errors — step failures are expected)
  m_joined <- tryCatch(
    withCallingHandlers(
      gam(formula = fml_j, data = df_j,
          family = binomial(link = "logit"), method = "REML",
          knots = p$knots),
      warning = function(w) {
        cat(sprintf("    WARNING: %s\n", conditionMessage(w)))
        invokeRestart("muffleWarning")
      }
    ),
    error = function(e) {
      cat(sprintf("    ERROR: Refit failed: %s\n", conditionMessage(e)))
      NULL
    }
  )

  if (is.null(m_joined)) next

  cat(sprintf("    Deviance explained: %.1f%% (full: %.1f%%)\n",
              summary(m_joined)$dev.expl * 100,
              summary(p$model_full)$dev.expl * 100))

  # Extract terms from both models
  terms_full   <- extract_sig_terms(p$model_full, "full")
  terms_joined <- extract_sig_terms(m_joined, "joined")

  # Compare
  comp <- terms_full |>
    rename(p_full = p_value, sig_full = significant) |>
    select(-source) |>
    full_join(
      terms_joined |>
        rename(p_joined = p_value, sig_joined = significant) |>
        select(-source),
      by = "feature"
    ) |>
    mutate(
      platform  = p$name,
      changed   = case_when(
        is.na(sig_full) | is.na(sig_joined) ~ "missing_in_one",
        sig_full & !sig_joined               ~ "lost",
        !sig_full & sig_joined               ~ "gained",
        sig_full & sig_joined                ~ "stable_sig",
        !sig_full & !sig_joined              ~ "stable_ns",
        TRUE                                 ~ "other"
      ),
      n_full    = nrow(p$data_full),
      n_joined  = nrow(p$data_joined)
    )

  comparison_rows <- c(comparison_rows, list(comp))

  # Print summary
  n_stable  <- sum(comp$changed %in% c("stable_sig", "stable_ns"), na.rm = TRUE)
  n_lost    <- sum(comp$changed == "lost", na.rm = TRUE)
  n_gained  <- sum(comp$changed == "gained", na.rm = TRUE)
  cat(sprintf("    Stable: %d  |  Lost significance: %d  |  Gained: %d\n",
              n_stable, n_lost, n_gained))
  if (n_lost > 0) {
    lost <- comp |> filter(changed == "lost")
    cat("    Lost features:\n")
    for (j in seq_len(nrow(lost))) {
      cat(sprintf("      - %s (full p=%.4f -> joined p=%.4f)\n",
                  lost$feature[j], lost$p_full[j], lost$p_joined[j]))
    }
  }
}

# Save comparison
source_comparison <- bind_rows(comparison_rows) |>
  arrange(platform, changed, feature)
write.csv(source_comparison, file.path(OUT_DIR, "sensitivity_source_comparison.csv"),
          row.names = FALSE)
cat(sprintf("\n  Saved: sensitivity_source_comparison.csv (%d rows)\n",
            nrow(source_comparison)))


# =============================================================================
# PART B: LinkedIn Reduced Inclusion Model
# =============================================================================

cat("\n\n================================================================\n")
cat("== PART B: LinkedIn reduced inclusion model ==\n")
cat("================================================================\n")

# The full LinkedIn inclusion GAM (N=1,170, 17 smooths at k=5) produced a
# step failure (negative Hessian eigenvalue). Test whether a reduced model
# with 8 features (selected by ranger permutation importance) converges cleanly.

# Reduced feature set (ranger permutation top + GAM significant + control):
#   log_post_age, media_type, word_count, line_break_count,
#   punct_diversity, log_follower, hashtag_count, account_type

cat("\n-- B1. Fitting reduced LinkedIn inclusion model --\n")

# Ensure factors in df_li
df_li_r <- df_li   # use full LinkedIn data (not joined-only)
for (v in factor_vars) {
  if (v %in% names(df_li_r) && !is.factor(df_li_r[[v]])) {
    df_li_r[[v]] <- as.factor(df_li_r[[v]])
  }
}

m_li_reduced <- tryCatch(
  gam(
    ever_top ~
      s(log_post_age, k = 5) +
      s(word_count, k = 5) +
      s(line_break_count, k = 5) +
      s(punct_diversity, k = 5) +
      s(log_follower, k = 5) +
      s(hashtag_count, k = 5) +
      s(local_hour, bs = "cc", k = 8) +
      s(weekday, bs = "cc", k = 5) +
      media_type + account_type,
    data = df_li_r,
    family = binomial(link = "logit"),
    method = "REML",
    knots = knots_tt_li
  ),
  warning = function(w) {
    cat(sprintf("  WARNING during fit: %s\n", conditionMessage(w)))
    suppressWarnings(
      gam(
        ever_top ~
          s(log_post_age, k = 5) +
          s(word_count, k = 5) +
          s(line_break_count, k = 5) +
          s(punct_diversity, k = 5) +
          s(log_follower, k = 5) +
          s(hashtag_count, k = 5) +
          s(local_hour, bs = "cc", k = 8) +
          s(weekday, bs = "cc", k = 5) +
          media_type + account_type,
        data = df_li_r,
        family = binomial(link = "logit"),
        method = "REML",
        knots = knots_tt_li
      )
    )
  }
)

cat(sprintf("  Deviance explained: %.1f%% (full model: %.1f%%)\n",
            summary(m_li_reduced)$dev.expl * 100,
            summary(m_li_full)$dev.expl * 100))
cat(sprintf("  AIC: %.1f (full: %.1f)\n",
            AIC(m_li_reduced), AIC(m_li_full)))

# Check convergence
conv_ok <- m_li_reduced$converged
cat(sprintf("  Converged: %s\n", ifelse(conv_ok, "YES (clean)", "NO")))

# Run diagnostics
cat("\n  k.check results:\n")
kc <- k.check(m_li_reduced)
for (j in seq_len(nrow(kc))) {
  flag <- ifelse(kc[j, "p-value"] < 0.05, " <--", "")
  cat(sprintf("    %-20s  k-index=%.3f  p=%.4f%s\n",
              rownames(kc)[j], kc[j, "k-index"], kc[j, "p-value"], flag))
}

# Compare significant terms
terms_full_li <- extract_sig_terms(m_li_full, "full")
terms_reduced <- extract_sig_terms(m_li_reduced, "reduced")

comp_li <- terms_full_li |>
  filter(feature %in% c("log_post_age", "word_count", "line_break_count",
                         "punct_diversity", "log_follower", "hashtag_count",
                         "local_hour", "weekday", "media_type", "account_type")) |>
  rename(p_full = p_value, sig_full = significant) |>
  select(-source) |>
  full_join(
    terms_reduced |>
      rename(p_reduced = p_value, sig_reduced = significant) |>
      select(-source),
    by = "feature"
  ) |>
  mutate(
    changed = case_when(
      sig_full & !sig_reduced  ~ "lost",
      !sig_full & sig_reduced  ~ "gained",
      sig_full & sig_reduced   ~ "stable_sig",
      !sig_full & !sig_reduced ~ "stable_ns",
      TRUE                     ~ "other"
    )
  )

write.csv(comp_li, file.path(OUT_DIR, "sensitivity_li_reduced.csv"),
          row.names = FALSE)

cat("\n  Feature comparison (full vs reduced):\n")
for (j in seq_len(nrow(comp_li))) {
  cat(sprintf("    %-20s  full: p=%.4f %s  |  reduced: p=%.4f %s  [%s]\n",
              comp_li$feature[j],
              comp_li$p_full[j],
              ifelse(comp_li$sig_full[j], "*", " "),
              comp_li$p_reduced[j],
              ifelse(comp_li$sig_reduced[j], "*", " "),
              comp_li$changed[j]))
}

cat(sprintf("\n  Saved: sensitivity_li_reduced.csv (%d features)\n",
            nrow(comp_li)))


# =============================================================================
# SUMMARY
# =============================================================================
cat("\n\n================================================================\n")
cat("== Summary ==\n")
cat("================================================================\n")
cat(sprintf("  Source comparison: %d feature-platform rows\n",
            nrow(source_comparison)))
cat(sprintf("  LinkedIn reduced: %d features compared, converged=%s\n",
            nrow(comp_li), ifelse(conv_ok, "YES", "NO")))

t_end <- Sys.time()
cat(sprintf("\n  Completed at: %s\n", t_end))
