# =============================================================================
# 03d_patch_audio.R
#
# Master's thesis: Learning the Levers — Which Creator-Controllable
# Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
# from a Swiss German-Language Perspective
# Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026
#
# Purpose:
#   Hotfix that expands the uses_named_audio pattern to cover missed
#   "original sound" translations (Italian, Dutch, Ukrainian, Chinese,
#   Romanian, Danish). Re-derives uses_named_audio and audio_is_original
#   from the database and patches df_tt.parquet in place (~222 TikTok posts
#   reclassified). Safe to run at any later step.
#
# Pipeline position:
#   Step 3d of the modelling pipeline. Depends on 03c_transformations.R
#   (needs df_tt.parquet); feeds Step 4 onwards.
#
# Inputs:
#   04_database/scraper.db                                       SQLite raw database (audio_name lookup)
#   05_modelling/03_data_prep/data/df_tt.parquet                 TikTok analytical frame
#
# Outputs:
#   05_modelling/03_data_prep/data/df_tt.parquet                 Updated in place
#
# Usage:
#   Rscript 05_modelling/03_data_prep/03d_patch_audio.R
# =============================================================================

rm(list = ls())
source(file.path("config", "packages.R"))

STEP_DIR <- file.path(BASE_DIR, "03_data_prep")
TT_PATH  <- file.path(STEP_DIR, "data", "df_tt.parquet")

# -- 1. Load current TikTok data -----------------------------------------------
cat("== 1. Loading df_tt.parquet ==\n")
df_tt <- read_parquet(TT_PATH)
cat(sprintf("  Loaded: %d rows x %d columns\n", nrow(df_tt), ncol(df_tt)))

old_named   <- sum(df_tt$uses_named_audio == 1, na.rm = TRUE)
old_original <- sum(df_tt$uses_named_audio == 0, na.rm = TRUE)
cat(sprintf("  BEFORE: uses_named_audio=1 (named): %d | =0 (original): %d | NA: %d\n",
    old_named, old_original, sum(is.na(df_tt$uses_named_audio))))

# -- 2. Pull audio_name from DB ------------------------------------------------
cat("\n== 2. Pulling audio_name from database ==\n")
con <- dbConnect(SQLite(), DB_PATH)

audio_lookup <- dbGetQuery(con, "
  SELECT fa.post_id, fa.audio_name
  FROM features_audio fa
  JOIN posts p ON fa.post_id = p.post_id
  WHERE p.platform = 'tiktok'
")
dbDisconnect(con)
cat(sprintf("  Retrieved audio_name for %d TikTok posts\n", nrow(audio_lookup)))

# -- 3. Expanded pattern (original list + missed translations) ------------------
original_sound_pattern <- paste(
  # Original 12 patterns from 01_build_post_table.R
  "original sound", "Originalton", "Original Sound",
  "originalljud", "son original", "sonido original",
  "som original", "suara asli", "orijinal ses",
  "оригинальный звук", "الصوت الأصلي", "πρωτότυπος ήχος",
  # --- ADDED: missed translations (2026-04-05 patch) ---
  "audio originale",          # Italian

  "suono originale",          # Italian
  "origineel geluid",         # Dutch
  "оригінальний звук",        # Ukrainian
  "оригінальний аудіозапис",  # Ukrainian (alt.)
  "原創音樂",                  # Chinese Traditional
  "原聲",                      # Chinese Traditional (short form)
  "sunet original",           # Romanian
  "original lyd",             # Danish / Norwegian
  sep = "|"
)

# -- 4. Re-derive ---------------------------------------------------------------
cat("\n== 3. Re-deriving uses_named_audio ==\n")

audio_lookup <- audio_lookup |>
  mutate(
    uses_named_audio_new = case_when(
      is.na(audio_name) ~ NA_integer_,
      grepl(original_sound_pattern, audio_name, ignore.case = TRUE) ~ 0L,
      TRUE ~ 1L
    ),
    audio_is_original_new = case_when(
      is.na(audio_name) ~ NA_integer_,
      grepl(original_sound_pattern, audio_name, ignore.case = TRUE) ~ 1L,
      TRUE ~ 0L
    )
  )

new_named    <- sum(audio_lookup$uses_named_audio_new == 1, na.rm = TRUE)
new_original <- sum(audio_lookup$uses_named_audio_new == 0, na.rm = TRUE)
n_fixed      <- old_named - new_named

cat(sprintf("  AFTER:  uses_named_audio=1 (named): %d | =0 (original): %d | NA: %d\n",
    new_named, new_original, sum(is.na(audio_lookup$uses_named_audio_new))))
cat(sprintf("  Posts reclassified (named -> original): %d\n", n_fixed))

# -- 5. Patch df_tt -------------------------------------------------------------
cat("\n== 4. Patching df_tt ==\n")

df_tt <- df_tt |>
  select(-uses_named_audio, -audio_is_original) |>
  left_join(
    audio_lookup |> select(post_id, uses_named_audio = uses_named_audio_new,
                           audio_is_original = audio_is_original_new),
    by = "post_id"
  )

# Verify
stopifnot("Row count changed" = nrow(df_tt) == 8551)
cat(sprintf("  Verified: %d rows (unchanged)\n", nrow(df_tt)))

# -- 6. Write back ---------------------------------------------------------------
cat("\n== 5. Writing patched df_tt.parquet ==\n")
write_parquet(df_tt, TT_PATH)
cat(sprintf("  Written to: %s (%.1f MB)\n", TT_PATH, file.size(TT_PATH) / 1e6))

cat(sprintf("\n== Step 3d complete: %d posts reclassified ==\n", n_fixed))
