# Results-chapter findings register (`results_register.csv`)

**Built:** 2026-05-02
**Repo HEAD at build time:** `d2ba8d41fca1156ead3431c04733af00431704d6`
**Modelling pipeline last touched at:** `d5d887bfa70e900b198acb0742eb84f1458683f7` ("redefine velocity as combined engagement for TT and LI (Path B, Decision 38)", 2026-04-17)
**Build script:** `05_modelling/11_data_results/_ground_truth/build_results_register.R`

This file is the source of truth for the prose-vs-tables alignment pass on
`Results_V2.docx`. It contains one row per (feature x platform x outcome)
finding the Results chapter can cite, with full provenance back to a CSV row
or model summary line.

This is read-only output for the alignment pass. The chapter draft, the audit
file (the thesis methodology decision log), the audit blocks, and the rewrite plan
are not modified by this build.

---

## Conventions in force

The register encodes the conventions documented in
the thesis methodology decision log and refined by audit Blocks
01-05. The decision-level documents below are authoritative; the summaries
here are pointers, not redefinitions.

### F1 - per-level factor inclusion pp at platform baseline

For multi-level factor variables (`media_type`, `topic_cluster`), per-level
pp values are computed as

    pp_level = plogis(qlogis(baseline) + beta_level) - baseline

where `baseline` is the platform's `mean(ever_top)`:

| Platform  | baseline |
|-----------|---------:|
| TikTok    |    0.430 |
| Instagram |    0.325 |
| LinkedIn  |    0.360 |

Reference categories named in the table footnotes:

| Platform  | media_type | topic_cluster |
|-----------|------------|---------------|
| TikTok    | n/a (constant: all video) | 1 (Business and professional) |
| Instagram | carousel  | 1 (Business and professional) |
| LinkedIn  | article   | 1 (Business and professional) |

The register uses `factor_pp_per_level.csv` as the authoritative input for the
seven cells the audit explicitly locked (Instagram joined-only media_type
[reel/image/video] and topic_cluster [6/8]; LinkedIn full-model media_type
[video/image]). Other per-level rows on `media_type` (e.g. LinkedIn carousel /
document / text) and on `topic_cluster` are computed on the fly using the
same F1 formula from `summary(model)$p.table`. The audit doc covered only
significant levels in the body tables; the register includes every level for
cross-platform Table 7 traceability.

### F2 - per-platform vs cross-platform reporting

Per-platform tables (4, 5, 6) report one row per significant level. The
cross-platform Table 7 cells use:

- `media_type` row: the strongest significant per-level pp value
  (`factor_pp_per_level.csv`).
- `topic_cluster` row: the omnibus chi-squared (binomial) or F (Gamma /
  Gaussian) statistic from `factor_omnibus_tests.csv` (Block 05; computed via
  `anova.gam` two-model deviance test, not pTerms Wald).

The register exposes both the omnibus rows (one per multi-level factor per
anchor model) and the per-level rows (one per non-reference level on
`media_type` and `topic_cluster`), so the alignment pass has direct access to
either reading.

### IG-specific stance for Instagram inclusion

For Instagram inclusion, the joined-only refit
(`05_modelling/05_gam/sensitivity/m_ig_inclusion_joined.rds`) is the
authoritative source for content-effect claims, because the full model is
documented as instrumentation-affected (Decision 31, audit Block 04). The
full model `m_ig_inclusion.rds` is anchored only for the Section 4.5 fit
metrics (deviance, AUC), which are not part of this register.

The register's `ig_inclusion` rows therefore come from
`m_ig_inclusion_joined.rds`. The chapter draft currently cites several IG
inclusion pp values (e.g. `face_flag` -3.8 pp, `flesch_reading_ease` +4.8 pp)
that were computed from the full model via `strategy_matrix.csv`. These differ
from the joined-only register values (`face_flag` -10.6 pp,
`flesch_reading_ease` +7.4 pp). This is a known divergence the alignment pass
must reconcile; it is not a register defect.

### TikTok structural constraints

`media_type` is constant on TikTok (all 8,551 captured posts are
`media_type == "video"`). It is not in any TT model formula. The register
contains four structural sentinel rows (one per TT outcome) with
`value = "n/a"` and a footnote that the test is not defined.

### Instagram velocity (withdrawn)

Instagram velocity was withdrawn from RQ1 reporting under Decision 37
(the thesis methodology decision log). No IG velocity model
is anchored in the chapter. The register contains two sentinel rows (one per
velocity outcome) recording the withdrawal so the cross-platform Table 7
cells have an explicit source citation.

---

## Anchor models

Nine GAMs are anchored in the chapter. The register has one outcome family
per anchor:

| Anchor          | Platform | Outcome       | Model file                                                  | N     | Family       |
|-----------------|----------|---------------|-------------------------------------------------------------|-------|--------------|
| tt_inclusion    | tt       | inclusion     | `05_modelling/05_gam/models/m_tt_inclusion.rds`             | 8,546 | binomial/logit |
| tt_rank         | tt       | rank          | `05_modelling/05_gam/models/m_tt_rank.rds`                  | 3,672 | gaussian/identity |
| tt_velocity_t24 | tt       | velocity_t24  | `05_modelling/05_gam/models/m_tt_velocity_24h.rds`          | 7,997 | Gamma/log    |
| tt_velocity_t72 | tt       | velocity_t72  | `05_modelling/05_gam/models/m_tt_velocity_72h.rds`          | 7,545 | Gamma/log    |
| ig_inclusion    | ig       | inclusion     | `05_modelling/05_gam/sensitivity/m_ig_inclusion_joined.rds` | 4,928 | binomial/logit |
| ig_rank         | ig       | rank          | `05_modelling/05_gam/models/m_ig_rank.rds`                  | 2,249 | gaussian/identity |
| li_inclusion    | li       | inclusion     | `05_modelling/05_gam/models/m_li_inclusion.rds`             | 1,170 | binomial/logit |
| li_rank         | li       | rank          | `05_modelling/05_gam/models/m_li_rank.rds`                  |   421 | gaussian/identity |
| li_velocity_t24 | li       | velocity_t24  | `05_modelling/05_gam/models/m_li_velocity_24h.rds`          |   507 | Gamma/log    |

The full IG inclusion model `m_ig_inclusion.rds` is loaded at build time but
not anchored as a per-feature row source; its only role is as a footnote
reference for chapter Section 4.5.

---

## Source files

### Decision-level (define which convention applies)

The conventions encoded in this register are documented in the thesis
methodology decision log. Five formal decision blocks define:

- Separation handling in the Instagram joined-only refit (at
  `media_type == reel`, `topic_cluster == 6`, `langfr`, and the coupled
  `(Intercept) / follower_available` pair). The register's separation
  rows cite this block.
- Disambiguation of the three anchor conventions in the codebase
  (strategy_matrix swing / F1 baseline / AME). The Convention F1 baseline
  (+16.2 pp video; -12.6 pp image for LinkedIn) is the chapter convention
  for `media_type`. The register honours this.
- The worst-to-best swing convention used by `strategy_matrix.csv`, which
  is NOT the right source for body tables. The register has zero rows
  sourced from `strategy_matrix.csv`.
- The full sig_BOTH / sig_FULL_only / sig_JOINED_only / sig_NEITHER
  inventory for IG inclusion. Used to set the significance markers and
  notes on every IG inclusion row.
- Pre-computed omnibus chi-sq and F values for the cross-platform Table
  cells.

Decisions 37 (Instagram velocity withdrawal) and 38 (combined velocity
refit for TT and LI) also apply.

### Data-level (provide the values)

Priority order, as specified by the spec:

1. `05_modelling/10_thesis_tables/output/factor_pp_per_level.csv` (mtime
   2026-04-30) - F1 per-level inclusion pp for IG joined-only and LI full.
2. `05_modelling/10_thesis_tables/output/factor_omnibus_tests.csv` (mtime
   2026-05-01) - Block 05 omnibus chi-sq / F for Table 7.
3. `05_modelling/05_gam/models/m_*.rds` plus
   `05_modelling/05_gam/sensitivity/m_ig_inclusion_joined.rds` (mtime
   2026-04-30) - smooth statistics (edf, F or chi-sq, p) and parametric
   coefficients (estimate, SE, z or t, p) for binaries, smooths, and any
   value not covered by the two CSVs.
4. `05_modelling/09_strategy/output/strategy_matrix.csv` is **NOT** used.
   Zero register rows source from it; verified post-build.

### Build inputs (script reads only these)

- 9 GAM RDS files listed in the anchor table
- 1 sensitivity GAM RDS (joined-only IG inclusion)
- The two CSVs above
- Reference-category levels from `levels(model$model[[factor_var]])`

The build script is hermetic: it does not query `scraper.db` or refit any
model. The joined-only refit was already cached in `sensitivity/` by the
upstream `factor_pp_per_level.R` run.

---

## Output schema

`results_register.csv` has 12 columns and 352 rows.

| Column               | Type    | Notes                                                                                                         |
|----------------------|---------|---------------------------------------------------------------------------------------------------------------|
| `feature`            | string  | Feature name as it appears in the model formula. For multi-level factors, per-level rows use `feature:level`. |
| `platform`           | string  | One of `tt`, `ig`, `li`.                                                                                      |
| `outcome`            | string  | One of `inclusion`, `rank`, `velocity_t24`, `velocity_t72`.                                                   |
| `value_unit`         | string  | One of `pp`, `positions`, `percent`, `chi-sq`, `F`, `shape-only`, `n/a`. Empty when value is unidentifiable.  |
| `value`              | string  | Reporting-scale value. Numeric values are formatted with explicit signs (`+/-X.X`).                           |
| `p_value`            | numeric | Raw p-value from the relevant test. Empty when not estimable (separation, withdrawn).                         |
| `sig_marker`         | string  | One of `***` (p<.001), `**` (p<.01), `*` (p<.05), `n.s.` (p>=.05 OR not estimable).                           |
| `n_obs`              | integer | Fitting-frame N for the anchor model.                                                                         |
| `model_file`         | string  | Repo-relative path to the RDS file. Empty for sentinel rows.                                                  |
| `source_row`         | string  | Provenance hint: which CSV row or summary call the value came from.                                           |
| `reference_category` | string  | For multi-level factor levels and omnibus rows: the named reference category. Empty for non-factor rows.      |
| `notes`              | string  | Free-form human-readable annotations. Includes IQR shape diagnostics, separation explanations, formula info. |

### Row counts

| Platform  | inclusion | rank | velocity_t24 | velocity_t72 | total |
|-----------|----------:|-----:|-------------:|-------------:|------:|
| tt        |        43 |   43 |           43 |           43 |   172 |
| ig        |        40 |   40 |       1\*    |        1\*   |    82 |
| li        |        41 |   40 |           17 |              |    98 |
| total     |       124 |  123 |           61 |           44 |   352 |

\* IG velocity is withdrawn; the single row per outcome is a sentinel.

### Feature-kind breakdown

| Kind                                                       |   n |
|-------------------------------------------------------------|----:|
| Single-feature row (smooth, binary, linear)                | 220 |
| Multi-level factor omnibus (`media_type`, `topic_cluster`, |  34 |
|   `lang`, `account_type`)                                  |     |
| Multi-level factor level (`media_type:level`,              |  96 |
|   `topic_cluster:level`)                                   |     |
| Withdrawn sentinel (`__withdrawn__`)                       |   2 |

---

## Conversion conventions per outcome

For each model the register uses the following conversion from the link-scale
delta or beta to the reporting-scale value.

### Inclusion (binomial / logit)

`value_unit = "pp"`, value is signed pp at the platform baseline.

- Binary parametric: `pp = plogis(qlogis(B) + beta) - B`, then `* 100`.
- Multi-level factor (per-level): same formula with the level coefficient.
- Smooth (non-cyclic): `pp = plogis(qlogis(B) + delta_link) - B` with
  `delta_link = partial(q75) - partial(q25)` of the predictor in the fitting
  frame.

### Rank (Gaussian / identity, lower = better)

`value_unit = "positions"`, value is the position shift directly (no link
inversion).

- Binary parametric: `value = beta`.
- Multi-level factor (per-level): `value = beta_level`.
- Smooth (non-cyclic): `value = partial(q75) - partial(q25)`.

### Velocity (Gamma / log)

`value_unit = "percent"`, value is signed percentage change.

- Binary parametric: `value = (exp(beta) - 1) * 100`.
- Multi-level factor (per-level): `value = (exp(beta_level) - 1) * 100`.
- Smooth (non-cyclic): `value = (exp(delta_link) - 1) * 100` with the same
  IQR-based `delta_link`.

This matches the spec instruction:

> Smooths reported as proportional change at IQR (TT log_post_age velocity,
> etc.): compute `exp(partial_effect_at_q75) / exp(partial_effect_at_q25) - 1`
> and report value as a percentage.

The two formulations are algebraically identical:
`exp(p75)/exp(p25) - 1 == exp(p75 - p25) - 1 == exp(delta_link) - 1`.

### Cyclic and degenerate-IQR smooths

Cyclic smooths (`bs="cc"`: `local_hour`, `weekday`) and any smooth where the
predictor's q25 equals q75 in the fitting frame (e.g. zero-inflated style
features like `caps_word_count`, `ellipsis_count`, `mention_count`,
`question_density`, `exclamation_density`, `url_count`, plus `log_follower`
on LI velocity_t24 and IG rank) get `value_unit = "shape-only"`. The `value`
column carries a short qualitative description. The `notes` column records
the link-scale range over q10..q90 and the location of the peak / trough,
matching the spec's guidance:

> Non-monotonic smooths ... value_unit = "shape-only", value = a short
> qualitative description. p_value and sig_marker still populated from the
> smooth table.

The shape descriptions are produced by `shape_describe()` in the build
script: it predicts the smooth's partial effect on a 25-point grid spanning
the predictor's full range and reports the peak / trough.

### Multi-level factor omnibus rows

`value_unit = "chi-sq"` for binomial models and `"F"` for Gaussian / Gamma
models. The `value` column carries `"chi-sq(df) = stat"` or `"F(df) = stat"`
formatted to two decimal places.

For the four cells that the audit Block 05 pre-computed (IG joined-only
inclusion media_type and topic_cluster, LI inclusion media_type and
topic_cluster, TT velocity_t24 topic_cluster), the register reads
`factor_omnibus_tests.csv`. For all other multi-level factor omnibus rows,
it falls back to `summary(model)$pTerms.table` (Wald, parametric factor df).
The two test families differ - `anova.gam` two-model deviance test versus
pTerms Wald - and the register's `notes` column flags which one was used so
the alignment pass can choose appropriately.

---

## Edge cases handled

- **Separation in IG joined-only refit (Block 01)**: `media_type:reel`
  (n_top=57, n_baseline=0) and `topic_cluster:6` (n_top=0, n_baseline=47) are
  full-separation cells. `follower_available` is coupled with the intercept
  under the same numerical singularity. All three rows have empty `value`,
  empty `p_value`, `sig_marker = "n.s."`, and a notes column that explains
  the unidentifiable status and references Section 4.3.2 / the separation
  doc.
- **Separation in LI inclusion**: `topic_cluster:6` has only 1 obs total
  (full-separation cell); `topic_cluster:unknown` is a coupled rank-
  deficiency case (n_top=3, n_baseline=6, but the SE inflates to 1e+07,
  diagnostic of paired separation). Both flagged via the same mechanism.
- **TT media_type structurally constant**: 4 sentinel rows with
  `value = "n/a"`.
- **IG velocity withdrawn**: 2 sentinel rows referencing Decision 37.
- **Aliased coefficients** (Estimate=0, SE=0): handled in the parametric
  extractor (`uses_named_audio` in TT inclusion; `audio_is_original` in TT
  rank) - `value` is empty and `notes` flags the aliasing.
- **Heavy zero-inflation smooths**: predictors with q25 == q75 (e.g.
  `caps_word_count`, `ellipsis_count`, `mention_count`, `question_density`,
  `exclamation_density`, `url_count` on most platforms; `log_follower` on LI
  velocity_t24 and IG rank) report `shape-only` with the q10-q90 shape
  description rather than a degenerate IQR delta.
- **Non-monotonic smooths inside the IQR**: still report a numeric IQR
  delta, with a `notes` flag stating the smooth is non-monotonic over
  q25..q75 and giving the link-scale range over q10..q90 for context. The
  alignment pass can choose to override these to shape-only if the chapter's
  prose framing benefits from a qualitative description.

---

## Cross-platform Table 7 cell map (quick reference)

The cells the chapter cites for cross-platform Table 7 resolve to these
register rows:

| Outcome  | Feature       | TT cell                         | IG cell                                         | LI cell                          |
|----------|---------------|---------------------------------|-------------------------------------------------|----------------------------------|
| inclusion | media_type   | `media_type` (n/a, structural)  | `media_type` (chi-sq(10.71)=245.60, ***)       | `media_type:video` (+16.2, *)    |
| inclusion | topic_cluster | n/a (no TT inclusion in scope) | `topic_cluster` (chi-sq(13.88)=25.93, *)        | `topic_cluster` (chi-sq(8.03)=6.75, n.s.) |
| velocity_t24 | media_type | `media_type` (n/a, structural) | n/a (IG velocity withdrawn)                    | `media_type` (F(5)=0.18, n.s.)   |
| velocity_t24 | topic_cluster | `topic_cluster` (F(5.78)=6.24, ***) | n/a (IG velocity withdrawn)              | n/a (no topic_cluster in LI velocity_t24) |
| rank     | media_type    | `media_type` (n/a, structural)  | `media_type` (F(3)=16.14, ***)                  | `media_type` (F(5)=4.98, ***)    |

For per-platform tables (4, 5, 6) the chapter cites `feature:level` rows
directly; the register has them all.

---

## Manual decisions taken at build time

1. **IQR formula chosen**: q75 - q25 partial-effect difference, evaluated
   on a 5-point grid (q10, q25, q50, q75, q90) at all-other-covariates-
   median (or first-level for factors / FALSE for binaries) profile. This
   matches the strategy_matrix script's `compute_magnitude` continuous
   branch (lines 261-275 of `09_strategy_matrix.R`), but applied per-
   anchor-model rather than via the strategy_matrix output.

2. **Profile for the partial-effect prediction**: median for numerics,
   first level (alphabetical) for factors, FALSE for binaries. This is the
   simplest defensible choice; AME-style averaging over the empirical
   covariate distribution would shrink magnitudes by 1-2 pp on LinkedIn
   (per Block 02) but the chapter's continuous-feature pp values use the
   single-point baseline anchor (Decision 30), so the IQR partial here is
   computed at the corresponding single-point profile. Documented in
   Block 02's "Why AME differs from Convention F1" section.

3. **Per-level rows for `media_type` and `topic_cluster` only** (not for
   `lang` or `account_type`). `lang` and `account_type` are controls, not
   creator-controllable content features, and the chapter narrative treats
   them via the omnibus only. The register reflects this.

4. **`__withdrawn__` sentinel rows for IG velocity** rather than expanding
   one row per IG-feature x outcome combination. Decision 37 withdraws IG
   velocity entirely; there is no anchor model to extract from. A single
   sentinel per outcome is the minimum needed to give Table 7 cells a
   register entry to point at.

5. **TT media_type structural rows added explicitly** (4 rows, one per
   outcome). The omnibus extractor would otherwise miss these cells because
   `media_type` is not in any TT model formula.

6. **IG full model `m_ig_inclusion.rds` is loaded but not anchored.** It
   would only be needed for Section 4.5 fit metrics, which are not part of
   this register. The register's IG inclusion rows come exclusively from
   the joined-only refit `m_ig_inclusion_joined.rds`. Where this differs
   from the chapter draft (e.g. `face_flag` -10.6 pp from joined vs -3.8
   pp from full model in the current draft), the register reflects the
   spec, not the chapter; the alignment pass resolves the divergence.

---

## Build reproducibility

To reproduce the register from a clean checkout:

```bash
# From repo root
Rscript 05_modelling/11_data_results/_ground_truth/build_results_register.R
```

The script reads the GAM RDS files and the two CSV inputs, and writes
`05_modelling/11_data_results/_ground_truth/results_register.csv`. It does not modify any other file.
The build is deterministic: rerunning produces a byte-identical CSV (modulo
the `Sys.time()` log line, which is not part of the output).

If `m_ig_inclusion_joined.rds` is missing (e.g. on a fresh checkout), the
upstream script `05_modelling/10_thesis_tables/R/factor_pp_per_level.R` must
run first to produce the cached refit; that script in turn requires
`scraper.db` to derive the joined-only subset.

---

## Out of scope for this register

- Section 4.5 model performance (deviance, AIC, AUC, Spearman rho with CI).
  These come from `08_evaluation/output/model_metrics.csv` and are not
  per-feature findings.
- Strategy heatmap and RQ3 outputs, which use the swing-at-baseline
  convention from `strategy_matrix.csv` (not the chapter body).
- Tables 1 (platform overview), 2 (topic clusters), 3 (legend), 9 (model
  fit). These are descriptive / structural tables, not findings tables.
- Smooth-shape figures (figure_*.pdf / .png in
  `10_thesis_tables/output/figures`). The register provides the underlying
  numeric anchors and shape descriptors, but figure plotting is a separate
  pipeline.
