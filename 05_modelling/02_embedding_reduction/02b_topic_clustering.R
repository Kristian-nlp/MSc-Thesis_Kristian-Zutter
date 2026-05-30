# =============================================================================
# 02b_topic_clustering.R
#
# Master's thesis: Learning the Levers — Which Creator-Controllable
# Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
# from a Swiss German-Language Perspective
# Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026
#
# Purpose:
#   Clusters PCA-reduced topic embeddings via k-means. Produces silhouette
#   analysis for k selection, sample captions per cluster for manual
#   labelling, and a UMAP projection of the chosen k = 10 solution.
#
# Pipeline position:
#   Step 2b of the modelling pipeline. Depends on 02a_pca_embeddings.R;
#   feeds 03a_missingness.R.
#
# Inputs:
#   05_modelling/01_build_analytical_table/data/df_post.parquet      Post-level analytical table
#   05_modelling/02_embedding_reduction/output/pca_object.rds        Fitted PCA object
#   04_database/scraper.db                                           SQLite raw database (captions)
#
# Outputs:
#   05_modelling/02_embedding_reduction/output/silhouette_plot.png   Silhouette by k
#   05_modelling/02_embedding_reduction/output/elbow_plot.png        Within-cluster SS by k
#   05_modelling/02_embedding_reduction/output/cluster_captions_k*.txt Captions per cluster
#   05_modelling/02_embedding_reduction/output/umap_topic_clusters.png UMAP projection (k = 10)
#   05_modelling/01_build_analytical_table/data/df_post.parquet      Updated in place with topic_cluster
#
# Usage:
#   Rscript 05_modelling/02_embedding_reduction/02b_topic_clustering.R
# =============================================================================

rm(list = ls())

# -- Source shared config ------------------------------------------------------
source(file.path("config", "packages.R"))

STEP_DIR <- file.path(BASE_DIR, "02_embedding_reduction")
OUT_DIR  <- file.path(STEP_DIR, "output")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)


# =============================================================================
# 1. Load data and PCA object
# =============================================================================
cat("== 1. Loading df_post.parquet and PCA object ==\n")

df_post <- read_parquet(file.path(DATA_DIR, "df_post.parquet"))
cat(sprintf("  Loaded: %d rows x %d columns\n", nrow(df_post), ncol(df_post)))

pca_result <- readRDS(file.path(OUT_DIR, "pca_object.rds"))
cat("  PCA object loaded.\n")

# ------------------------------------------------------------------
# DECISION: Number of PCs for clustering
# The 80% variance threshold required 86 PCs -- impractical for k-means
# (curse of dimensionality degrades Euclidean distances). The scree elbow
# at ~15 PCs captures the dominant semantic structure (35.8% cumvar).
# This is typical for multilingual sentence embeddings.
# ------------------------------------------------------------------
N_PCS_CLUSTER <- 15

pc_cols_all <- grep("^topic_pc_", names(df_post), value = TRUE)
pc_cols     <- sprintf("topic_pc_%02d", 1:N_PCS_CLUSTER)
cat(sprintf("  Total PC columns in parquet: %d\n", length(pc_cols_all)))
cat(sprintf("  Using first %d PCs for clustering (%.1f%% cumvar)\n",
    N_PCS_CLUSTER, 35.8))

# Work with complete cases only (same rows as PCA input)
pc_matrix    <- as.matrix(df_post[, pc_cols])
has_pcs      <- complete.cases(pc_matrix)
pc_complete  <- pc_matrix[has_pcs, ]
cat(sprintf("  Complete cases for clustering: %d / %d\n",
    sum(has_pcs), nrow(df_post)))


# =============================================================================
# 2. K-means for candidate k values
# =============================================================================
cat("\n== 2. Running k-means for candidate k values ==\n")

K_CANDIDATES <- c(5, 8, 10, 12, 15)
set.seed(42)

km_results <- list()
for (k in K_CANDIDATES) {
  cat(sprintf("  k = %d ... ", k))
  km <- kmeans(pc_complete, centers = k, nstart = 25, iter.max = 100)
  km_results[[as.character(k)]] <- km
  cat(sprintf("converged (tot.withinss = %.0f, iter = %d)\n",
      km$tot.withinss, km$iter))
}


# =============================================================================
# 3. Silhouette analysis
# =============================================================================
cat("\n== 3. Computing silhouette scores ==\n")

# Distance matrix on PC scores (subsample if >10k for speed)
MAX_SIL_N <- 5000
if (nrow(pc_complete) > MAX_SIL_N) {
  cat(sprintf("  Subsampling %d / %d rows for silhouette (speed)\n",
      MAX_SIL_N, nrow(pc_complete)))
  set.seed(42)
  sil_idx <- sample(nrow(pc_complete), MAX_SIL_N)
} else {
  sil_idx <- seq_len(nrow(pc_complete))
}

sil_dist <- dist(pc_complete[sil_idx, ])

sil_scores <- tibble(k = integer(), avg_silhouette = double())

for (k in K_CANDIDATES) {
  km <- km_results[[as.character(k)]]
  sil <- silhouette(km$cluster[sil_idx], sil_dist)
  avg_sil <- mean(sil[, "sil_width"])
  sil_scores <- bind_rows(sil_scores, tibble(k = k, avg_silhouette = avg_sil))
  cat(sprintf("  k = %2d | avg silhouette = %.4f\n", k, avg_sil))
}

cat("\n  Silhouette summary:\n")
print(as.data.frame(sil_scores))


# =============================================================================
# 4. Silhouette plot
# =============================================================================
cat("\n== 4. Saving silhouette plot ==\n")

p_sil <- ggplot(sil_scores, aes(x = k, y = avg_silhouette)) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 2.5) +
  geom_text(aes(label = sprintf("%.3f", avg_silhouette)),
            vjust = -1, size = 3) +
  labs(
    title = "Average Silhouette Score by Number of Clusters",
    x = "k (number of clusters)",
    y = "Average Silhouette Width"
  ) +
  scale_x_continuous(breaks = K_CANDIDATES) +
  theme_minimal(base_size = 11)

ggsave(file.path(OUT_DIR, "silhouette_plot.png"), p_sil,
       width = 7, height = 5, dpi = 150)
cat(sprintf("  Saved: %s\n", file.path(OUT_DIR, "silhouette_plot.png")))


# =============================================================================
# 5. Elbow plot (within-cluster sum of squares)
# =============================================================================
cat("\n== 5. Saving elbow plot ==\n")

wss_data <- tibble(
  k   = K_CANDIDATES,
  wss = map_dbl(K_CANDIDATES, ~ km_results[[as.character(.x)]]$tot.withinss)
)

p_elbow <- ggplot(wss_data, aes(x = k, y = wss)) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 2.5) +
  labs(
    title = "Elbow Plot (Within-Cluster Sum of Squares)",
    x = "k (number of clusters)",
    y = "Total Within-Cluster SS"
  ) +
  scale_x_continuous(breaks = K_CANDIDATES) +
  theme_minimal(base_size = 11)

ggsave(file.path(OUT_DIR, "elbow_plot.png"), p_elbow,
       width = 7, height = 5, dpi = 150)
cat(sprintf("  Saved: %s\n", file.path(OUT_DIR, "elbow_plot.png")))


# =============================================================================
# 6. Sample captions nearest each centroid
# =============================================================================
cat("\n== 6. Extracting sample captions per cluster ==\n")

# We need raw captions -- read from DB
con <- dbConnect(SQLite(), DB_PATH)
captions <- dbGetQuery(con, "
  SELECT post_id, caption_raw, platform
  FROM posts
  WHERE caption_raw IS NOT NULL
")
dbDisconnect(con)

# Map post_ids for complete cases
post_ids_complete <- df_post$post_id[has_pcs]

for (k in K_CANDIDATES) {
  km <- km_results[[as.character(k)]]
  out_file <- file.path(OUT_DIR, sprintf("cluster_captions_k%02d.txt", k))

  sink(out_file)
  cat(sprintf("=== K-MEANS CLUSTERING: k = %d ===\n", k))
  cat(sprintf("Cluster sizes: %s\n\n",
      paste(sprintf("%d: %d", 1:k, km$size), collapse = " | ")))

  for (cl in 1:k) {
    cat(sprintf("--- Cluster %d (N = %d) ---\n", cl, km$size[cl]))

    # Indices of posts in this cluster
    cl_idx <- which(km$cluster == cl)

    # Distance from each post to its centroid
    centroid <- km$centers[cl, , drop = FALSE]
    dists <- apply(pc_complete[cl_idx, , drop = FALSE], 1, function(row) {
      sqrt(sum((row - centroid)^2))
    })

    # 10 nearest to centroid
    nearest <- cl_idx[order(dists)[1:min(10, length(dists))]]
    nearest_ids <- post_ids_complete[nearest]

    # Look up captions
    cap_df <- captions |> filter(post_id %in% nearest_ids)

    for (i in seq_len(nrow(cap_df))) {
      # Truncate long captions for readability
      cap_text <- substr(cap_df$caption_raw[i], 1, 200)
      cap_text <- gsub("\n", " ", cap_text)
      cat(sprintf("  [%s] %s\n", cap_df$platform[i], cap_text))
    }
    cat("\n")
  }
  sink()

  cat(sprintf("  Saved: %s\n", out_file))
}


# =============================================================================
# 7. Cluster size distribution per platform
# =============================================================================
cat("\n== 7. Cluster size distribution by platform ==\n")

for (k in K_CANDIDATES) {
  km <- km_results[[as.character(k)]]
  cat(sprintf("\n  k = %d:\n", k))

  cluster_platform <- tibble(
    cluster  = km$cluster,
    platform = df_post$platform[has_pcs]
  )

  tbl <- cluster_platform |>
    count(cluster, platform) |>
    pivot_wider(names_from = platform, values_from = n, values_fill = 0) |>
    arrange(cluster)

  print(as.data.frame(tbl))
}


# =============================================================================
# 8. DECISION: Choose k and assign topic_cluster
# =============================================================================
cat("\n== 8. DECISION: Choose k ==\n")
cat("  Review the silhouette plot, elbow plot, and cluster_captions_k*.txt files.\n")
cat("  Then set CHOSEN_K below and re-run from this section.\n\n")

# ------------------------------------------------------------------
# >>> DECISION: Set CHOSEN_K after reviewing output <<<
# ------------------------------------------------------------------
CHOSEN_K <- 10   # <- Set this after review (e.g. CHOSEN_K <- 10)

if (!is.null(CHOSEN_K)) {
  cat(sprintf("  Chosen k = %d\n", CHOSEN_K))

  km_final <- km_results[[as.character(CHOSEN_K)]]

  # Assign cluster labels to all posts (NA for posts without embeddings)
  cluster_full <- rep(NA_integer_, nrow(df_post))
  cluster_full[has_pcs] <- km_final$cluster

  df_post$topic_cluster <- factor(cluster_full)

  cat(sprintf("  topic_cluster assigned: %d non-NA values\n",
      sum(!is.na(df_post$topic_cluster))))

  # Distribution
  cat("\n  topic_cluster distribution:\n")
  print(table(df_post$topic_cluster, df_post$platform,
              dnn = c("cluster", "platform")))

  # Write updated parquet
  out_path <- file.path(DATA_DIR, "df_post.parquet")
  write_parquet(df_post, out_path)
  cat(sprintf("\n  Written to: %s\n", out_path))
  cat(sprintf("  File size: %.1f MB\n", file.size(out_path) / 1e6))
  cat(sprintf("  Columns: %d (new: topic_cluster)\n", ncol(df_post)))

} else {
  cat("  CHOSEN_K is NULL. Set it after reviewing diagnostics and re-run.\n")
  cat("  No changes written to df_post.parquet.\n")
}


# =============================================================================
# 5. UMAP VISUALISATION (Thesis Figure 14)
# =============================================================================
cat("\n== 5. UMAP 2D projection of topic clusters ==\n")

if (!requireNamespace("uwot", quietly = TRUE)) {
  install.packages("uwot")
}
library(uwot)

# Use the same 15 PCs as clustering
pc_cols <- paste0("topic_pc_", sprintf("%02d", 1:N_PCS_CLUSTER))
pc_mat  <- as.matrix(df_post[has_pcs, pc_cols])

set.seed(42)
umap_coords <- umap(pc_mat, n_neighbors = 30, min_dist = 0.3, n_components = 2,
                     metric = "euclidean", n_threads = 1)

# Build plot data
umap_df <- tibble(
  UMAP1         = umap_coords[, 1],
  UMAP2         = umap_coords[, 2],
  topic_cluster = factor(km_results[[as.character(CHOSEN_K)]]$cluster),
  platform      = df_post$platform[has_pcs]
)

# Cluster labels (from the manual labelling in the findings log)
CLUSTER_LABELS <- c(
  "1"  = "Business & Professional",
  "2"  = "Architecture & Design",
  "3"  = "Hashtag-Only / Minimal",
  "4"  = "Interactive & Products",
  "5"  = "Sports & Action",
  "6"  = "Audio Metadata (No Caption)",
  "7"  = "Casual Commentary",
  "8"  = "Travel & Nature",
  "9"  = "DACH & European",
  "10" = "Ultra-Short & Emoji"
)

umap_df$cluster_label <- CLUSTER_LABELS[as.character(umap_df$topic_cluster)]

# Compute cluster centroids for labels
centroids <- umap_df |>
  group_by(topic_cluster, cluster_label) |>
  summarise(x = median(UMAP1), y = median(UMAP2), .groups = "drop")

p_umap <- ggplot(umap_df, aes(x = UMAP1, y = UMAP2, colour = topic_cluster)) +
  geom_point(alpha = 0.15, size = 0.4, stroke = 0) +
  geom_label(data = centroids, aes(x = x, y = y, label = cluster_label),
             size = 2.5, fontface = "bold", alpha = 0.85,
             label.padding = unit(0.15, "lines"), show.legend = FALSE) +
  scale_colour_brewer(palette = "Set3", name = "Cluster") +
  labs(title = "UMAP projection of topic embeddings (k = 10 clusters)",
       x = "UMAP 1", y = "UMAP 2") +
  theme_thesis() +
  theme(legend.position = "none")

PLOT_DIR <- OUT_DIR
save_plot(p_umap, "umap_topic_clusters.png", width = 7, height = 6)
cat("  Saved: umap_topic_clusters.png\n")


cat("\n== Step 2b complete ==\n")
