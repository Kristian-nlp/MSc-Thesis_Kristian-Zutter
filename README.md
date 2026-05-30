# Learning the Levers

**Which Creator-Controllable Features Predict Post Visibility on TikTok, Instagram, and LinkedIn from a Swiss German-Language Perspective**

Replication repository for the Master's thesis by **Kristian Zutter**, HSLU MSc Applied Information and Data Science, Spring Semester 2026.

This repository contains the full code base behind the thesis: the Python scraping and feature-extraction pipeline and the R modelling pipeline (GAMs and random forests), through to the figures cited in the thesis. The large binary artefacts (the raw database, the derived datasets, and the fitted models) are **not** stored in the repository; they are shared separately (see [Data availability](#data-availability)) because they are too large to version-control.

---

## What is in this repository

| Stage | Folder | Purpose |
|-------|--------|---------|
| 1. Configuration | `01_config/` | Platform URLs, logging, alerting, fallback handlers, cookie structure |
| 2. Scraping | `02_scraper/` | Hourly cron-driven collection from TikTok, Instagram, LinkedIn discovery surfaces (Python + Selenium) |
| 3. Feature extraction | `03_features/` | Text, style, visual, audio, temporal, and topic-embedding feature extractors (Python) |
| 4. Database | `04_database/` | SQLite schema, ETL helpers, and the parquet exporter (the `scraper.db` file itself is shared via the cloud bundle) |
| 5. Modelling | `05_modelling/` | The full R modelling pipeline (`00_recon` through `11_data_results`) and the figure scripts (`10_figures/R/`) |

### What is committed vs. shared separately

- **Committed to this repository:** all code, configuration, the five Chapter 4 figures (`05_modelling/10_figures/output/`), and the canonical results register (`05_modelling/11_data_results/_ground_truth/results_register.csv`).
- **Shared via the cloud bundle (not committed):** the raw database, the derived `.parquet` datasets, and the fitted `.rds` models. See [Data availability](#data-availability).
- **Not included:** intermediate diagnostic outputs (EDA plots, GAM diagnostics, variable-importance tables, etc.) are regenerable by running the pipeline and are omitted to keep the repository lean.

---

## Data availability

The raw database and all derived/fitted binaries are available as a separate download:

> **Cloud bundle:** <https://hsluzern-my.sharepoint.com/:f:/g/personal/kristian_zutter_stud_hslu_ch/IgD6ySg2XQM-S5EFakwgznG-ASxeoycE_zgMxElv5xQOb64?e=lgfAH0>

The bundle mirrors this repository's folder structure. To use it, download it and copy its contents into the root of your clone so the files land at the paths below.

| Artefact | Target path in the repo |
|----------|--------------------------|
| Raw SQLite database (77 MB) | `04_database/scraper.db` |
| Post-level analytical table | `05_modelling/01_build_analytical_table/data/df_post.parquet` |
| Per-platform analytical tables | `05_modelling/03_data_prep/data/df_{tt,ig,li}.parquet` |
| Topic embeddings (PCA object) | `05_modelling/02_embedding_reduction/output/pca_object.rds` |
| Fitted GAM models (9) | `05_modelling/05_gam/models/m_*.rds` |
| Instagram inclusion sensitivity refit | `05_modelling/05_gam/sensitivity/m_ig_inclusion_joined.rds` |
| Ranger cross-check models (3) | `05_modelling/06_ranger/models/rf_*.rds` |
| Out-of-fold CV predictions | `05_modelling/08_evaluation/output/oof_*.rds` |

With only `scraper.db` in place you can regenerate everything else by running the pipeline (see [Reproduce from scratch](#reproduce-from-raw-data)). With the models in place you can rebuild the figures and the results register directly, without re-running the heavy steps.

### Dataset summary

| Metric | Value |
|--------|-------|
| Collection window | 11 Mar 2026 - 24 Mar 2026 |
| Unique posts | 16,638 |
| TikTok / Instagram / LinkedIn posts | 8,551 / 6,917 / 1,170 |
| Sock-puppet accounts | 6 (one fresh + one light-seeded per platform) |

Schema definitions live in `04_database/schema.sql`. The database is opened **read-only** by every R script.

---

## Quick start

### Prerequisites

- **Python 3.10+** with a CPU-only PyTorch install (see `requirements.txt` header) — for the scraper and feature extractors.
- **R 4.3+** with packages: `DBI`, `RSQLite`, `tidyverse`, `arrow`, `mgcv`, `ranger`, `yardstick`, `corrplot`, `naniar`, `openxlsx`, `cluster`, `here`, `gratia`, `patchwork`, `readr`, `scales`.
- **SQLite 3** (for inspecting the database).

```sh
# Python deps
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# R deps (install once)
Rscript -e 'install.packages(c("DBI","RSQLite","tidyverse","arrow","mgcv","ranger","yardstick","corrplot","naniar","openxlsx","cluster","here","gratia","patchwork","readr","scales"))'
```

All R scripts use `here::here()` to resolve paths from the repository root (anchored by the `.here` file), so they can be run from anywhere inside the repo.

---

## Reproduce the figures

The five Chapter 4 figures are committed under `05_modelling/10_figures/output/`. To regenerate them you need the fitted GAM models from the [cloud bundle](#data-availability) in place, then run any figure script from the repo root:

```sh
Rscript 05_modelling/10_figures/R/fig_02_tt_inclusion.R
Rscript 05_modelling/10_figures/R/figure_tt_velocity_partial_effects.R
Rscript 05_modelling/10_figures/R/figure_r3_ig_inclusion_partials_joined.R
Rscript 05_modelling/10_figures/R/figure_04_li_inclusion_partial_effects.R
Rscript 05_modelling/10_figures/R/fig_05_cross_platform_velocity.R
```

| Script | Output | Thesis figure |
|--------|--------|---------------|
| `fig_02_tt_inclusion.R` | `figure_r1_tt_inclusion.png` | TikTok: smooth predictors of top-20 inclusion |
| `figure_tt_velocity_partial_effects.R` | `figure_r2_tt_velocity.png` | TikTok: 24-hour velocity partial effects |
| `figure_r3_ig_inclusion_partials_joined.R` | `figure_r3_ig_inclusion_partials_joined.png` | Instagram: caption features on inclusion (joined-only refit) |
| `figure_04_li_inclusion_partial_effects.R` | `figure_r4_li_inclusion_partial_effects.png` | LinkedIn: post age and media format on inclusion |
| `fig_05_cross_platform_velocity.R` | `figure_r5_cross-comparison.png` | Cross-platform velocity comparison (TikTok + LinkedIn) |

The Chapter 3 "Audit pipeline overview" and the Chapter 5 "posting strategy" figures are hand-designed schematics produced outside the repository and are not regenerable from the code.

### Thesis tables

The thesis tables are typeset in the thesis document itself. Every numerical claim in those tables is traceable to the **results register** (`05_modelling/11_data_results/_ground_truth/results_register.csv`), which maps each (feature, platform, outcome) triple to the model it came from, the reported value, the p-value, and the significance marker. See `results_register_README.md` for the full schema.

---

## Reproduce from raw data

To regenerate everything from scratch, place `scraper.db` from the [cloud bundle](#data-availability) at `04_database/scraper.db`, then run the pipeline in order from the repo root:

```sh
# 1. Build the analytical post table from scraper.db
Rscript 05_modelling/01_build_analytical_table/01_build_post_table.R

# 2. Topic embeddings + clustering
Rscript 05_modelling/02_embedding_reduction/02a_pca_embeddings.R
Rscript 05_modelling/02_embedding_reduction/02b_topic_clustering.R

# 3. Data prep (missingness, collinearity, transforms, audio patch)
Rscript 05_modelling/03_data_prep/03a_missingness.R
Rscript 05_modelling/03_data_prep/03b_collinearity.R
Rscript 05_modelling/03_data_prep/03c_transformations.R
Rscript 05_modelling/03_data_prep/03d_patch_audio.R

# 4. Exploratory analysis (optional)
Rscript 05_modelling/04_eda/04_eda.R

# 5. Fit GAMs (inclusion, rank, velocity, sensitivity)
Rscript 05_modelling/05_gam/05a_gam_inclusion.R
Rscript 05_modelling/05_gam/05b_gam_rank.R
Rscript 05_modelling/05_gam/05c_gam_velocity.R
Rscript 05_modelling/05_gam/05d_sensitivity_source.R

# 6. Ranger cross-check
Rscript 05_modelling/06_ranger/06_ranger_crosscheck.R

# 7. Cross-platform comparison
Rscript 05_modelling/07_cross_platform/07a_comparison_table.R
Rscript 05_modelling/07_cross_platform/07b_overlay_plots.R

# 8. Evaluation (CV + temporal split)
Rscript 05_modelling/08_evaluation/08_evaluation.R

# 9. Strategy matrix (RQ3 effect labels)
Rscript 05_modelling/09_strategy/09_strategy_matrix.R

# 10. Canonical results register
Rscript 05_modelling/11_data_results/_ground_truth/build_results_register.R

# 11. Figures (05_modelling/10_figures/R/)
```

Running the pipeline recreates the intermediate outputs (parquets, models, diagnostic plots, intermediate CSVs) in their respective `data/` and `output/` folders.

To re-collect data with the scrapers, you additionally need:

- Authenticated session cookies for the six sock-puppet accounts (see `01_config/cookies/README.md` for the required structure — cookies are not bundled for security).
- A Chromium browser and `undetected-chromedriver`.
- A residential proxy (the thesis used Evomi; configured via `01_config/settings.py`).

---

## Not included in this repository

This is a clean replication snapshot. The following are deliberately absent:

- **Auth cookies** (`01_config/cookies/`) — real session tokens are not bundled; only a README documenting the expected structure.
- **Raw database, derived datasets, and fitted models** — shared via the [cloud bundle](#data-availability) rather than committed.
- **Intermediate diagnostic outputs** (EDA plots, GAM diagnostics, variable-importance and comparison CSVs) — regenerable by running the pipeline.
- **Internal planning notes, drafts, and the test suite** — not required to reproduce the thesis results.
- **Hard-coded user paths** — all path resolution is repo-relative via `here::here()` and the `.here` anchor file.
