# =============================================================================
# 02a_pca_embeddings.R
#
# Master's thesis: Learning the Levers — Which Creator-Controllable
# Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
# from a Swiss German-Language Perspective
# Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026
#
# Purpose:
#   Runs PCA on the 384 topic embedding dimensions to reduce dimensionality.
#   Targets ~80% cumulative variance explained and replaces the raw topic
#   dimension columns with principal-component scores.
#
# Pipeline position:
#   Step 2a of the modelling pipeline. Depends on 01_build_post_table.R;
#   feeds 02b_topic_clustering.R.
#
# Inputs:
#   05_modelling/01_build_analytical_table/data/df_post.parquet  Post-level analytical table
#
# Outputs:
#   05_modelling/02_embedding_reduction/output/pca_object.rds    Fitted prcomp object
#   05_modelling/02_embedding_reduction/output/pca_scree.png     Scree plot (first 50 PCs)
#   05_modelling/02_embedding_reduction/output/pca_cumvar.png    Cumulative variance plot
#   05_modelling/01_build_analytical_table/data/df_post.parquet  Updated in place with topic_pc_*
#
# Usage:
#   Rscript 05_modelling/02_embedding_reduction/02a_pca_embeddings.R
# =============================================================================

rm(list = ls())

# -- Source shared config ------------------------------------------------------
source(file.path("config", "packages.R"))

STEP_DIR  <- file.path(BASE_DIR, "02_embedding_reduction")
OUT_DIR   <- file.path(STEP_DIR, "output")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)


# =============================================================================
# 1. Load df_post.parquet and extract embeddings
# =============================================================================
cat("== 1. Loading df_post.parquet ==\n")

df_post <- read_parquet(file.path(DATA_DIR, "df_post.parquet"))
cat(sprintf("  Loaded: %d rows x %d columns\n", nrow(df_post), ncol(df_post)))

# Extract embedding columns
emb_cols <- grep("^topic_dim_", names(df_post), value = TRUE)
cat(sprintf("  Embedding columns found: %d\n", length(emb_cols)))
stopifnot(length(emb_cols) == 384)

emb_matrix <- as.matrix(df_post[, emb_cols])


# =============================================================================
# 2. Handle missing embeddings
# =============================================================================
cat("\n== 2. Checking for missing embeddings ==\n")

has_embedding <- complete.cases(emb_matrix)
n_missing <- sum(!has_embedding)
cat(sprintf("  Valid embeddings:  %d\n", sum(has_embedding)))
cat(sprintf("  Missing (NA/zero): %d (%.1f%%)\n",
    n_missing, n_missing / nrow(emb_matrix) * 100))

if (n_missing > 0) {
  cat("  Platform breakdown of missing embeddings:\n")
  print(table(df_post$platform[!has_embedding]))
}

# PCA on complete cases only
emb_complete <- emb_matrix[has_embedding, ]
cat(sprintf("  PCA input: %d rows x %d columns\n",
    nrow(emb_complete), ncol(emb_complete)))


# =============================================================================
# 3. Run PCA
# =============================================================================
cat("\n== 3. Running PCA (center=TRUE, scale.=TRUE) ==\n")
cat("  This may take 30-60 seconds...\n")

pca_result <- prcomp(emb_complete, center = TRUE, scale. = TRUE)

cat("  PCA complete.\n")


# =============================================================================
# 4. Variance explained diagnostics
# =============================================================================
cat("\n== 4. Variance explained ==\n")

var_explained <- pca_result$sdev^2 / sum(pca_result$sdev^2)
cum_var       <- cumsum(var_explained)

# Print thresholds
thresholds <- c(0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
cat("\n  Cumulative variance thresholds:\n")
for (thr in thresholds) {
  n_pcs <- which(cum_var >= thr)[1]
  cat(sprintf("    %.0f%% variance: %d PCs (cumvar = %.3f)\n",
      thr * 100, n_pcs, cum_var[n_pcs]))
}

# Print first 50 PCs individually
cat("\n  First 50 PCs (individual + cumulative variance):\n")
pc_table <- tibble(
  PC            = 1:50,
  var_pct       = round(var_explained[1:50] * 100, 2),
  cumvar_pct    = round(cum_var[1:50] * 100, 2)
)
print(as.data.frame(pc_table), row.names = FALSE)


# =============================================================================
# 5. Scree plot
# =============================================================================
cat("\n== 5. Saving scree plot ==\n")

plot_data <- tibble(PC = 1:50, variance = var_explained[1:50] * 100)

p_scree <- ggplot(plot_data, aes(x = PC, y = variance)) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 1.5) +
  labs(
    title = "PCA Scree Plot (Topic Embeddings)",
    x = "Principal Component",
    y = "Variance Explained (%)"
  ) +
  scale_x_continuous(breaks = seq(0, 50, 5)) +
  theme_minimal(base_size = 11)

ggsave(file.path(OUT_DIR, "pca_scree.png"), p_scree,
       width = 8, height = 5, dpi = 150)
cat(sprintf("  Saved: %s\n", file.path(OUT_DIR, "pca_scree.png")))


# =============================================================================
# 6. Cumulative variance plot
# =============================================================================
cat("\n== 6. Saving cumulative variance plot ==\n")

plot_data_cum <- tibble(PC = 1:100, cumvar = cum_var[1:100] * 100)

p_cumvar <- ggplot(plot_data_cum, aes(x = PC, y = cumvar)) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 1, alpha = 0.5) +
  geom_hline(yintercept = 80, linetype = "dashed", colour = "red", linewidth = 0.4) +
  annotate("text", x = 80, y = 82, label = "80% threshold",
           colour = "red", size = 3.5, hjust = 0) +
  labs(
    title = "Cumulative Variance Explained (Topic Embeddings)",
    x = "Number of Principal Components",
    y = "Cumulative Variance (%)"
  ) +
  scale_x_continuous(breaks = seq(0, 100, 10)) +
  scale_y_continuous(breaks = seq(0, 100, 10)) +
  theme_minimal(base_size = 11)

ggsave(file.path(OUT_DIR, "pca_cumvar.png"), p_cumvar,
       width = 8, height = 5, dpi = 150)
cat(sprintf("  Saved: %s\n", file.path(OUT_DIR, "pca_cumvar.png")))


# =============================================================================
# 7. Save PCA object
# =============================================================================
cat("\n== 7. Saving PCA object ==\n")

saveRDS(pca_result, file.path(OUT_DIR, "pca_object.rds"))
cat(sprintf("  Saved: %s\n", file.path(OUT_DIR, "pca_object.rds")))


# =============================================================================
# 8. Write PC scores to df_post.parquet
# =============================================================================
cat("\n== 8. Writing PC scores to df_post.parquet ==\n")

# ------------------------------------------------------------------
# DECISION: Number of PCs
# Review the scree plot and cumulative variance output above.
# Set N_PCS to the chosen number (target: 80% variance explained).
# ------------------------------------------------------------------
# >>> After reviewing output, update this value if needed <<<
n_at_80 <- which(cum_var >= 0.80)[1]
N_PCS <- n_at_80
cat(sprintf("  Using %d PCs (%.1f%% cumulative variance)\n",
    N_PCS, cum_var[N_PCS] * 100))

# Extract scores for complete cases
scores_complete <- pca_result$x[, 1:N_PCS]
colnames(scores_complete) <- sprintf("topic_pc_%02d", 1:N_PCS)

# Build full-size score matrix (NAs for posts without embeddings)
scores_full <- matrix(NA_real_, nrow = nrow(df_post), ncol = N_PCS)
colnames(scores_full) <- sprintf("topic_pc_%02d", 1:N_PCS)
scores_full[has_embedding, ] <- scores_complete

# Drop old topic_dim columns, add PC columns
df_post <- df_post |>
  select(-all_of(emb_cols)) |>
  bind_cols(as_tibble(scores_full))

cat(sprintf("  Updated df_post: %d rows x %d columns\n",
    nrow(df_post), ncol(df_post)))
cat(sprintf("  Embedding columns replaced: 384 dims -> %d PCs\n", N_PCS))

# Write
out_path <- file.path(DATA_DIR, "df_post.parquet")
write_parquet(df_post, out_path)
cat(sprintf("  Written to: %s\n", out_path))
cat(sprintf("  File size: %.1f MB\n", file.size(out_path) / 1e6))

# Verify
cat("\n  Verification:\n")
cat(sprintf("    PC columns: %s ... %s\n",
    sprintf("topic_pc_%02d", 1), sprintf("topic_pc_%02d", N_PCS)))
cat(sprintf("    NAs in topic_pc_01: %d (should be %d)\n",
    sum(is.na(df_post$topic_pc_01)), n_missing))


cat("\n== Step 2a complete ==\n")
cat("Next step: 02b_topic_clustering.R\n")
