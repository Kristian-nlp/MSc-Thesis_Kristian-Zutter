# =============================================================================
# 00_data_recon.R
#
# Master's thesis: Learning the Levers — Which Creator-Controllable
# Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
# from a Swiss German-Language Perspective
# Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026
#
# Purpose:
#   Fully dynamic data reconnaissance from scraper.db. Auto-discovers every
#   table and every column (nothing hardcoded) and writes the inventory and
#   cross-table diagnostics to an Excel workbook.
#
# Pipeline position:
#   Step 0 of the modelling pipeline. Standalone exploratory script; no
#   downstream dependencies.
#
# Inputs:
#   04_database/scraper.db                              SQLite raw database (read-only)
#
# Outputs:
#   05_modelling/00_recon/00_data_recon.xlsx            Multi-sheet inventory + diagnostics
#
# Usage:
#   Rscript 05_modelling/00_recon/00_data_recon.R
# =============================================================================

# -- Clean slate --------------------------------------------------------------
rm(list = ls())

# -- Packages -----------------------------------------------------------------
pkgs <- c("DBI", "RSQLite", "tidyverse", "openxlsx")
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
}

library(DBI)
library(RSQLite)
library(tidyverse)
library(openxlsx)

# -- Config -------------------------------------------------------------------
# >>> ADJUST THESE TWO PATHS <<<
db_path  <- "scraper.db"
out_path <- "00_data_recon.xlsx"

# Create output directory if needed
dir.create(dirname(out_path), showWarnings = FALSE, recursive = TRUE)

# -- Connect ------------------------------------------------------------------
con <- dbConnect(SQLite(), db_path)

# -- Fresh workbook (safe to re-run) ------------------------------------------
wb <- createWorkbook()

# -- Helper function ----------------------------------------------------------
sheet_counter <- 0

add_sheet <- function(sheet_name, df, caption = NULL) {
  sheet_counter <<- sheet_counter + 1
  # Prefix with number to guarantee uniqueness and ordering
  safe_name <- substr(paste0(sheet_counter, "_", sheet_name), 1, 31)

  addWorksheet(wb, safe_name)
  start_row <- 1
  if (!is.null(caption)) {
    writeData(wb, safe_name, data.frame(x = caption), startRow = 1, colNames = FALSE)
    addStyle(wb, safe_name, style = createStyle(textDecoration = "bold", fontSize = 12),
             rows = 1, cols = 1)
    start_row <- 3
  }
  writeDataTable(wb, safe_name, df, startRow = start_row, tableStyle = "TableStyleLight9")
  setColWidths(wb, safe_name, cols = seq_len(ncol(df)), widths = "auto")

  cat(sprintf("\n-- [%s] --\n", safe_name))
  print(as.data.frame(df), right = FALSE)
  cat("\n")
}

# =============================================================================
# PART 1: SCHEMA DISCOVERY
# =============================================================================

cat("\n##########################################################\n")
cat("# PART 1: SCHEMA DISCOVERY\n")
cat("##########################################################\n")

tables <- dbListTables(con)

# 1. Row counts
row_counts <- map_dfr(tables, function(tbl) {
  n <- dbGetQuery(con, paste0("SELECT COUNT(*) AS n FROM [", tbl, "]"))$n
  tibble(table_name = tbl, n_rows = n)
})
add_sheet("Row_Counts", row_counts, "Row counts per table")

# 2. Full column inventory
schema_all <- map_dfr(tables, function(tbl) {
  info <- dbGetQuery(con, sprintf("PRAGMA table_info([%s])", tbl))
  info |> mutate(
    table_name = tbl,
    is_pk      = if_else(pk == 1, "YES", ""),
    not_null   = if_else(notnull == 1, "YES", "")
  ) |> select(table_name, column_name = name, type, is_pk, not_null)
})
add_sheet("All_Columns", schema_all, "Complete column inventory (all tables)")

# =============================================================================
# PART 2: COLUMN-LEVEL PROFILING
# For EVERY table and EVERY column: nulls, distinct values,
# min/mean/max (numeric) or top values (text/other).
# =============================================================================

cat("\n##########################################################\n")
cat("# PART 2: COLUMN-LEVEL PROFILING\n")
cat("##########################################################\n")

profile_rows <- list()

for (tbl in tables) {
  cols   <- dbGetQuery(con, sprintf("PRAGMA table_info([%s])", tbl))
  n_rows <- dbGetQuery(con, sprintf("SELECT COUNT(*) AS n FROM [%s]", tbl))$n

  if (n_rows == 0) {
    cat(sprintf("\n  %s -- empty table, skipping\n", tbl))
    next
  }

  cat(sprintf("\n  Profiling %s (%d rows, %d cols)...\n", tbl, n_rows, nrow(cols)))

  for (i in seq_len(nrow(cols))) {
    col_name <- cols$name[i]
    col_type <- tolower(cols$type[i])

    # Basic stats: total, null, distinct
    basic <- tryCatch(
      dbGetQuery(con, sprintf("
        SELECT COUNT(*)              AS n_total,
               SUM(CASE WHEN [%s] IS NULL THEN 1 ELSE 0 END) AS n_null,
               COUNT(DISTINCT [%s])  AS n_distinct
        FROM [%s]
      ", col_name, col_name, tbl)),
      error = function(e) tibble(n_total = n_rows, n_null = NA_integer_, n_distinct = NA_integer_)
    )

    null_pct <- if (!is.na(basic$n_null)) round(basic$n_null / basic$n_total * 100, 1) else NA_real_

    is_numeric <- grepl("int|real|float|numeric|double", col_type)
    is_blob    <- grepl("blob", col_type)

    val_min    <- NA_character_
    val_mean   <- NA_character_
    val_max    <- NA_character_
    top_values <- NA_character_

    if (is_blob) {
      non_null <- basic$n_total - basic$n_null
      top_values <- sprintf("%d non-null BLOBs", non_null)

    } else if (is_numeric) {
      rs <- tryCatch(
        dbGetQuery(con, sprintf("
          SELECT MIN([%s]) AS vmin, ROUND(AVG([%s]), 4) AS vmean, MAX([%s]) AS vmax
          FROM [%s] WHERE [%s] IS NOT NULL
        ", col_name, col_name, col_name, tbl, col_name)),
        error = function(e) NULL
      )
      if (!is.null(rs) && nrow(rs) > 0 && !is.na(rs$vmin)) {
        val_min  <- as.character(rs$vmin)
        val_mean <- as.character(rs$vmean)
        val_max  <- as.character(rs$vmax)
      }

    } else {
      tv <- tryCatch(
        dbGetQuery(con, sprintf("
          SELECT CAST([%s] AS TEXT) AS val, COUNT(*) AS n
          FROM [%s] WHERE [%s] IS NOT NULL
          GROUP BY [%s] ORDER BY n DESC LIMIT 8
        ", col_name, tbl, col_name, col_name)),
        error = function(e) data.frame()
      )
      if (nrow(tv) > 0) {
        tv$val <- str_trunc(as.character(tv$val), 60)
        top_values <- paste(sprintf("%s (%d)", tv$val, tv$n), collapse = " | ")
      }
    }

    profile_rows[[length(profile_rows) + 1]] <- tibble(
      table_name  = tbl,
      column_name = col_name,
      column_type = cols$type[i],
      n_total     = basic$n_total,
      n_null      = basic$n_null,
      null_pct    = null_pct,
      n_distinct  = basic$n_distinct,
      val_min     = val_min,
      val_mean    = val_mean,
      val_max     = val_max,
      top_values  = top_values
    )
  }
}

profile_df <- bind_rows(profile_rows)
add_sheet("Column_Profiles", profile_df, "Every column in every table -- auto-discovered")

# =============================================================================
# PART 3: CROSS-TABLE MODELLING DIAGNOSTICS
# =============================================================================

cat("\n##########################################################\n")
cat("# PART 3: CROSS-TABLE DIAGNOSTICS\n")
cat("##########################################################\n")

# -- 3A. Platform x is_top balance --------------------------------------------
tryCatch({
  q <- dbGetQuery(con, "
    SELECT p.platform, c.is_top, COUNT(*) AS n,
           ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY p.platform), 1) AS pct
    FROM captures c JOIN posts p ON c.post_id = p.post_id
    GROUP BY p.platform, c.is_top ORDER BY p.platform, c.is_top
  ")
  add_sheet("Is_Top_Balance", q, "Platform x is_top balance (captures level)")
}, error = function(e) cat("  3A skipped:", e$message, "\n"))

# -- 3B. Collection window per platform x account_type -----------------------
tryCatch({
  q <- dbGetQuery(con, "
    SELECT platform, account_type, COUNT(*) AS n_snapshots,
           MIN(captured_at_utc) AS first_snapshot,
           MAX(captured_at_utc) AS last_snapshot,
           ROUND(julianday(MAX(captured_at_utc)) - julianday(MIN(captured_at_utc)), 1) AS span_days
    FROM snapshots GROUP BY platform, account_type ORDER BY platform, account_type
  ")
  add_sheet("Collection_Window", q, "Collection window per platform x account type")
}, error = function(e) cat("  3B skipped:", e$message, "\n"))

# -- 3C. Account type split at captures level ---------------------------------
tryCatch({
  q <- dbGetQuery(con, "
    SELECT p.platform, s.account_type, COUNT(*) AS n_captures,
           SUM(c.is_top) AS n_top, COUNT(*) - SUM(c.is_top) AS n_baseline
    FROM captures c JOIN posts p ON c.post_id = p.post_id
    JOIN snapshots s ON c.snapshot_id = s.snapshot_id
    GROUP BY p.platform, s.account_type ORDER BY p.platform, s.account_type
  ")
  add_sheet("Account_Type", q, "Account type split at captures level")
}, error = function(e) cat("  3C skipped:", e$message, "\n"))

# -- 3D. Unique authors vs posts ----------------------------------------------
tryCatch({
  q <- dbGetQuery(con, "
    SELECT platform,
           COUNT(*) AS n_posts,
           COUNT(DISTINCT author_hash) AS n_unique_authors,
           ROUND(COUNT(*) * 1.0 / COUNT(DISTINCT author_hash), 2) AS posts_per_author
    FROM posts GROUP BY platform
  ")
  add_sheet("Authors", q, "Unique authors vs total posts")
}, error = function(e) cat("  3D skipped:", e$message, "\n"))

# -- 3E. Rank distribution (top posts only) -----------------------------------
tryCatch({
  q <- dbGetQuery(con, "
    SELECT p.platform,
           MIN(c.rank_observed) AS rank_min, MAX(c.rank_observed) AS rank_max,
           ROUND(AVG(c.rank_observed), 1) AS rank_mean,
           COUNT(DISTINCT c.rank_observed) AS n_distinct_ranks,
           COUNT(*) AS n_top_captures
    FROM captures c JOIN posts p ON c.post_id = p.post_id
    WHERE c.is_top = 1 GROUP BY p.platform
  ")
  add_sheet("Rank_Distribution", q, "Rank distribution among Top posts")
}, error = function(e) cat("  3E skipped:", e$message, "\n"))

# -- 3F. Revisit coverage (T0 / T24 / T72) -----------------------------------
tryCatch({
  q <- dbGetQuery(con, "
    WITH pr AS (
      SELECT p.platform, p.post_id,
             MAX(CASE WHEN co.revisit_type = 't0'  THEN 1 ELSE 0 END) AS has_t0,
             MAX(CASE WHEN co.revisit_type = 't24' THEN 1 ELSE 0 END) AS has_t24,
             MAX(CASE WHEN co.revisit_type = 't72' THEN 1 ELSE 0 END) AS has_t72
      FROM posts p LEFT JOIN counters co ON p.post_id = co.post_id
      GROUP BY p.platform, p.post_id
    )
    SELECT platform, COUNT(*) AS n_posts,
           SUM(has_t0) AS with_t0, SUM(has_t24) AS with_t24, SUM(has_t72) AS with_t72,
           ROUND(SUM(has_t0)  * 100.0 / COUNT(*), 1) AS pct_t0,
           ROUND(SUM(has_t24) * 100.0 / COUNT(*), 1) AS pct_t24,
           ROUND(SUM(has_t72) * 100.0 / COUNT(*), 1) AS pct_t72
    FROM pr GROUP BY platform
  ")
  add_sheet("Revisit_Coverage", q, "T0 / T24 / T72 revisit coverage per platform")
}, error = function(e) cat("  3F skipped:", e$message, "\n"))

# -- 3G. Counter stats (all engagement metrics) -------------------------------
tryCatch({
  q <- dbGetQuery(con, "
    SELECT p.platform, co.revisit_type, COUNT(*) AS n,
           ROUND(AVG(co.likes), 1)    AS avg_likes,
           ROUND(AVG(co.comments), 1) AS avg_comments,
           ROUND(AVG(co.shares), 1)   AS avg_shares,
           ROUND(AVG(co.views), 1)    AS avg_views,
           MAX(co.likes) AS max_likes, MAX(co.comments) AS max_comments,
           MAX(co.shares) AS max_shares, MAX(co.views) AS max_views,
           SUM(CASE WHEN co.likes IS NULL THEN 1 ELSE 0 END)    AS likes_null,
           SUM(CASE WHEN co.comments IS NULL THEN 1 ELSE 0 END) AS comments_null,
           SUM(CASE WHEN co.shares IS NULL THEN 1 ELSE 0 END)   AS shares_null,
           SUM(CASE WHEN co.views IS NULL THEN 1 ELSE 0 END)    AS views_null
    FROM counters co JOIN posts p ON co.post_id = p.post_id
    GROUP BY p.platform, co.revisit_type ORDER BY p.platform, co.revisit_type
  ")
  add_sheet("Counter_Stats", q, "Engagement counter statistics")
}, error = function(e) cat("  3G skipped:", e$message, "\n"))

# -- 3H. Feature table join coverage ------------------------------------------
tryCatch({
  feature_tables <- tables[grepl("^features_", tables)]
  if (length(feature_tables) > 0) {
    q <- map_dfr(feature_tables, function(ft) {
      dbGetQuery(con, sprintf("
        SELECT '%s' AS feature_table, p.platform,
               COUNT(DISTINCT p.post_id) AS total_posts,
               COUNT(DISTINCT f.post_id) AS with_features,
               ROUND(COUNT(DISTINCT f.post_id) * 100.0 / COUNT(DISTINCT p.post_id), 1) AS pct_covered
        FROM posts p LEFT JOIN [%s] f ON p.post_id = f.post_id
        GROUP BY p.platform
      ", ft, ft))
    })
    add_sheet("Feature_Coverage", q, "Feature table join coverage")
  }
}, error = function(e) cat("  3H skipped:", e$message, "\n"))

# -- 3I. Scrape log summary ---------------------------------------------------
tryCatch({
  if ("scrape_log" %in% tables) {
    q <- dbGetQuery(con, "
      SELECT platform, account_type, status, COUNT(*) AS n_runs,
             ROUND(AVG(posts_captured), 1) AS avg_posts,
             SUM(posts_captured) AS total_captured
      FROM scrape_log
      GROUP BY platform, account_type, status ORDER BY platform, account_type, status
    ")
    add_sheet("Scrape_Log", q, "Scrape log summary")
  }
}, error = function(e) cat("  3I skipped:", e$message, "\n"))

# -- 3J. Scrape errors --------------------------------------------------------
tryCatch({
  if ("scrape_log" %in% tables) {
    q <- dbGetQuery(con, "
      SELECT platform, account_type, error_message, COUNT(*) AS n
      FROM scrape_log WHERE status != 'success' AND error_message IS NOT NULL
      GROUP BY platform, account_type, error_message ORDER BY n DESC LIMIT 20
    ")
    if (nrow(q) > 0) add_sheet("Scrape_Errors", q, "Scrape errors (top 20)")
  }
}, error = function(e) cat("  3J skipped:", e$message, "\n"))

# -- 3K. Revisit log (if exists) ----------------------------------------------
tryCatch({
  if ("revisit_log" %in% tables) {
    cols_rl <- dbGetQuery(con, "PRAGMA table_info(revisit_log)")
    cat(sprintf("\n  revisit_log has %d columns: %s\n", nrow(cols_rl), paste(cols_rl$name, collapse = ", ")))

    q <- dbGetQuery(con, "SELECT * FROM revisit_log LIMIT 20")
    add_sheet("Revisit_Log_Sample", q, "revisit_log -- first 20 rows (sample)")

    # Summary by platform if platform column exists
    if ("platform" %in% cols_rl$name) {
      q2 <- dbGetQuery(con, "
        SELECT platform, COUNT(*) AS n FROM revisit_log GROUP BY platform
      ")
      add_sheet("Revisit_Log_Summary", q2, "revisit_log summary by platform")
    }
  }
}, error = function(e) cat("  3K skipped:", e$message, "\n"))

# =============================================================================
# PART 4: PER-PLATFORM NULL RATES FOR ALL FEATURE COLUMNS
# One row per platform x column -- easy to filter in Excel.
# =============================================================================

cat("\n##########################################################\n")
cat("# PART 4: PER-PLATFORM FEATURE NULL RATES\n")
cat("##########################################################\n")

tryCatch({
  feature_tables <- tables[grepl("^features_", tables)]
  platform_null_rows <- list()

  for (ft in feature_tables) {
    ft_cols <- dbGetQuery(con, sprintf("PRAGMA table_info([%s])", ft))
    # Skip post_id and processed_at -- not features
    feat_cols <- ft_cols$name[!ft_cols$name %in% c("post_id", "processed_at")]

    for (fc in feat_cols) {
      q <- tryCatch(
        dbGetQuery(con, sprintf("
          SELECT p.platform,
                 COUNT(*) AS n_posts,
                 SUM(CASE WHEN f.[%s] IS NULL THEN 1 ELSE 0 END) AS n_null,
                 ROUND(SUM(CASE WHEN f.[%s] IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS null_pct
          FROM posts p
          LEFT JOIN [%s] f ON p.post_id = f.post_id
          GROUP BY p.platform
        ", fc, fc, ft)),
        error = function(e) NULL
      )

      if (!is.null(q)) {
        q$feature_table <- ft
        q$column_name   <- fc
        platform_null_rows[[length(platform_null_rows) + 1]] <- q
      }
    }
  }

  if (length(platform_null_rows) > 0) {
    platform_nulls <- bind_rows(platform_null_rows) |>
      select(feature_table, column_name, platform, n_posts, n_null, null_pct) |>
      arrange(feature_table, column_name, platform)
    add_sheet("Platform_Null_Rates", platform_nulls,
              "NULL rates per platform for every feature column")
  }
}, error = function(e) cat("  Part 4 error:", e$message, "\n"))

# =============================================================================
# SAVE
# =============================================================================

saveWorkbook(wb, out_path, overwrite = TRUE)
dbDisconnect(con)

cat(sprintf("\n\nExcel saved to: %s\n", normalizePath(out_path, mustWork = FALSE)))
cat("\n##########################################################\n")
cat("# == Step 0 complete ==\n")
cat("##########################################################\n")
