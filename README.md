# Prescriptive Care Lab

This is an independent, reproducible research prototype. It goes from **prediction** to a **decision** in three steps:

1. Predict next-year hospital admission from one year of Medicare-style claims.
2. Group members into phenotypes.
3. Choose whom a capacity-limited care-management program should contact, by budgeted integer optimization under **explicitly assumed** intervention effects.

The choices are compared with random, highest-cost and highest-risk targeting. The project also runs sensitivity and subgroup checks and sizes a randomized pilot that would measure the real effect.

> **Synthetic data, assumed effects.**
> - All data are CMS **DE-SynPUF synthetic** Medicare claims. CMS warns that inferences about Medicare beneficiaries from these files "will be misleading and often incorrect". No clinical or epidemiological conclusion is valid from these results. Their value is the pipeline and the methods.
> - The intervention's relative risk reduction (10–30%) is an **assumption** throughout, not an estimate.
> - CPU only; no PHI, GPU, paid API, cloud or LLM calls. Not a clinical tool.

## Question

Among fee-for-service beneficiaries, three questions:

- How well do 2008 claims rank and calibrate risk of **any inpatient admission in 2009**? The secondary outcome is top-decile 2009 cost.
- Which phenotypes emerge?
- With capacity to contact 2%, 5% or 10% of members, which selection maximizes avoided admissions under the stated assumptions, and how much does the answer depend on them?

The full design is in [`docs/protocol.md`](docs/protocol.md). It was committed (6d7fcc2) before any model was fit. Deviations are listed at its end.

## Data

| Item | Value |
|---|---|
| Source | CMS 2008–2010 DE-SynPUF, **DE1.0 Sample 1**: 2008 and 2009 Beneficiary Summary; 2008–2010 Inpatient, Outpatient and Carrier (1A + 1B) claims |
| Terms | Free public-use download from cms.gov, no data use agreement, synthetic, no PHI. Terms text, URLs, retrieval dates and SHA-256 are in [`data/manifest.json`](data/manifest.json). Raw files are gitignored. |
| Cohort | 116,352 in the 2008 file → alive at end of 2008 and present in 2009: 114,538 → 12 months Part A + B in 2008: 95,605 → 0 HMO months: **72,071** |
| Split (SHA-256 of ID, beneficiary-level) | train 43,488 · validation 14,295 · **test 14,288 (2,696 admitted, 18.9%)** |
| Features (54, 2008 only) | demographics; 11 CMS chronic-condition flags; 2008 payments; claim-derived utilization (inpatient stays/days, outpatient, carrier, ED visits); 19 ICD-9 chapter flags; distinct 3-digit diagnoses. Race is audit-only. |

**Timing rule.** Feature claims must start **and** end in 2008. The 2009 beneficiary file is read only for outcomes. Tests check both rules.

## Methods

- **Predictive.**
  - Models: L2 logistic regression and LightGBM, fixed hyperparameters, early stopping on validation. Preprocessing is fit on train only.
  - Targeting score: LightGBM, chosen by validation AUROC (0.728 vs 0.726).
  - Test metrics: AUROC, AUPRC, Brier, calibration intercept/slope and reliability curves, with 1,000 beneficiary-bootstrap CIs and paired differences.
  - Interpretation: SHAP on 5,000 validation rows.
- **Phenotyping.**
  - k-means for k = 3–8 on scaled chronic-condition and utilization profiles, train only.
  - Selection rule: highest silhouette among k with mean stability ARI ≥ 0.75 (5 seeds × 80% subsamples). Gaussian-mixture BIC was a cross-check.
- **Prescriptive.**
  - Problem: maximize Σ xᵢ·pᵢ·rᵢ subject to Σ xᵢ·wᵢ ≤ B. Solved as an ILP with PuLP/CBC; the LP bound is reported.
  - Budgets: a head-count budget and an assumed care-manager-hours budget (4–8 h by phenotype risk rank).
  - Effect scenarios: uniform 10/20/30%; "sickest least helped" (30%→10% by phenotype rank); "sickest most helped" (10%→30%).
  - Evaluation: **realized** avoided admissions = Σ over contacted members of *observed* 2009 admission × assumed rᵢ. The model is never scored against its own predictions.
  - Uncertainty: 300 bootstrap resamples with re-selection in each.
  - Subgroups: age group, sex, race.
- **Freeze / one-shot test.**
  - `dev` (train + validation) → `dryrun` (full evaluation rehearsed on validation) → `freeze` (hashes committed at e427563) → `test`.
  - `test` writes `results/test_results.json` in exclusive-create mode.

## Results (held-out test, n = 14,288)

**Prediction**

| Outcome | Model | AUROC [95% CI] | AUPRC [95% CI] | Brier | Calibration slope [95% CI] |
|---|---|---|---|---|---|
| Any admission 2009 (18.9%) | Logistic | 0.743 [0.734, 0.752] | 0.378 [0.361, 0.398] | 0.136 | 1.04 [0.98, 1.10] |
| | **LightGBM** (targeting) | 0.741 [0.732, 0.751] | 0.376 [0.359, 0.395] | 0.136 | 1.11 [1.05, 1.17] |
| Top-decile cost 2009 (10.3%) | Logistic | 0.800 [0.789, 0.810] | 0.360 [0.336, 0.386] | 0.079 | 1.06 [1.00, 1.12] |
| | LightGBM | 0.799 [0.788, 0.809] | 0.373 [0.349, 0.397] | 0.078 | 1.06 [1.00, 1.12] |

- **Admission.** LightGBM − logistic ΔAUROC = −0.002 [−0.004, 0.001]: no gain from boosting on this synthetic data.
- **Calibration.** For the admission models, calibration intercepts are about 0.04 (CI −0.00 to 0.08). A slope above 1 means LightGBM's admission predictions are slightly too compressed.
- **SHAP.** The model relies mostly on diagnostic breadth (distinct 3-digit codes), 2008 Medicare payments and visit counts. This is association, not cause.

**Phenotypes (k = 8; silhouette 0.245, mean stability ARI 0.85).** Test admission rates by phenotype:

| Phenotype | Test admission rate |
|---|---|
| No recorded 2008 care (21%; likely a synthesis artefact) | 1.0% |
| Light use | 13.0% |
| Cardiometabolic | 24.2% |
| Cardiometabolic + arthritis | 27.5% |
| Cancer | 27.1% |
| ESRD | 33.3% |
| Recent hospital stays, complex | 35.8% |
| Stroke/TIA + stays | 36.0% |

The Gaussian-mixture BIC cross-check falls steadily through k = 8 with no elbow (`results/dev_results.json`), so it does not help choose k. Silhouette is only 0.22–0.25 for every k, and several clusters are defined by a single condition flag. These are descriptive groupings, not discovered subtypes.

**Targeting.** "Captured" = contacted members actually admitted in 2009; it is assumption-free. "Avoided" uses the assumed uniform 20% effect. Head-count budget:

| Capacity | Random: captured / avoided | Highest 2008 cost | Highest risk (= ILP when effect uniform) | Risk − cost, captured [95% CI] |
|---|---|---|---|---|
| 2% (286) | 53.9 / 10.8 | 107 / 21.4 | **152 / 30.4** [27.9, 33.7] | 45 [25, 66] |
| 5% (714) | 135.1 / 27.0 | 277 / 55.4 | **345 / 69.0** [64.6, 73.8] | 68 [39, 98] |
| 10% (1,429) | 269.1 / 53.8 | 525 / 105.0 | **627 / 125.4** [118.0, 132.8] | 102 [63, 138] |

**Optimized − highest risk** (avoided admissions, 95% CI) **depends only on the assumptions:**

| Budget | Effect assumption | 2% | 5% | 10% |
|---|---|---|---|---|
| Head-count | Uniform 10/20/30% | 0 (identical by construction) | 0 | 0 |
| Head-count | Sickest least helped | +5.9 [1.7, 9.6] | +10.8 [5.5, 16.3] | +17.9 [11.1, 25.6] |
| Head-count | Sickest most helped | +1.1 [−1.8, 3.6] | +2.7 [−1.9, 8.4] | +16.4 [10.0, 22.6] |
| Hours | Uniform 20% | +2.8 [−0.6, 5.7] | +1.8 [−2.6, 6.4] | 0.0 [−5.8, 5.5] |
| Hours | Sickest least helped | +11.5 [7.2, 15.1] | +22.5 [16.1, 29.4] | +37.0 [28.6, 44.9] |

- **ILP quality.** Every ILP solution is within 0.003% of its LP upper bound. Its planned value exceeds greedy by at most 0.04 admissions.
- **Subgroups (5% capacity).** The capture rate is 12.8% overall but 6.6% for ages 65–74, 18.3% for under-65s and 22.0% for 85+. All three are flagged by the prespecified >5 pp rule. Mean predicted risk for Black members is 18.2% against 20.7% observed.
- **Pilot.** For a top-5% list (observed control admission rate 48.3%), an individually randomized pilot needs **412 per arm** to detect a 20% RRR. It needs 180 per arm for 30% and 1,669 for 10% (α 0.05, power 0.8). Cluster designs need 1.5–3.5× more.
- **Sensitivity.** Excluding the 3 test members with no 2009 Part A gives AUROC 0.741 and the same selected set.

All numbers: [`results/tables.md`](results/tables.md), generated from `results/test_results.json` and `results/dev_results.json`. Plain-language brief: [`reports/report.md`](reports/report.md). Static page: [`reports/summary.html`](reports/summary.html).

## Reproduce

```bash
uv sync
uv run python -m src.data            # download (hash-checked), cohort, 2008 features, 2009 labels, split (~1 min)
uv run pytest -q                     # split integrity, no future leakage, train-only preprocessing, optimizer capacity
uv run python -m src.run dev         # train/validation only (~3.5 min)
uv run python -m src.run dryrun      # full evaluation rehearsed on validation
uv run python -m src.run freeze      # requires committed inputs; writes results/freeze.json
uv run python -m src.run test        # one shot (~12 min); refuses if anything changed since freeze
uv run python scripts/make_figures.py
```

Peak RSS: `dev` 1.4 GB and `test` 0.5 GB (`results/*.json` → `compute`). Carrier CSVs are about 1.24 GB each uncompressed (`results/data_checks.json`) and are streamed in 200k-row chunks.

## Limitations

- **Synthetic data.** DE-SynPUF weakens or breaks real multivariate structure. For example, HMO enrollment makes no difference to observed claims (16.3% vs 16.5% admission rates, `results/data_checks.json`). About 21% of full-year fee-for-service members have essentially no claims. AUROC values, phenotypes and subgroup gaps say nothing about real Medicare.
- **Assumed, not estimated, effects.** Claims data contain no intervention assignment. Any "optimized beats risk ranking" gain is produced by the assumed heterogeneity in effect or cost. Only the "captured admissions" comparison is empirical.
- **Deaths not modeled.** Deaths in 2009 (1.5%) are a competing event. A member who died without an admission counts as not admitted.
- **No external or temporal validation.** One sample and one year.
- **Weak cluster structure.** See the phenotype results above.
- **Uninterpretable logistic coefficients.** They are collinear across payment and inpatient fields.

## Repository map

- `src/data.py`: provenance, cohort, features, outcomes, split.
- `src/features.py`: train-only preprocessing.
- `src/models.py`: models, metrics, bootstrap.
- `src/cluster.py`: phenotyping and k selection.
- `src/prescriptive.py`: ILP, greedy, policies, pilot arithmetic.
- `src/run.py`: dev / dryrun / freeze / test phases.
- `models/`: fitted artefacts, hashed at freeze.
- `results/`: all numbers.
- `reports/`: brief, figures, static HTML.
- `docs/protocol.md`: the prespecified design.



## License

Code: MIT ([`LICENSE`](LICENSE)). Data attribution: [`THIRD_PARTY.md`](THIRD_PARTY.md). Implementation was AI-assisted.
