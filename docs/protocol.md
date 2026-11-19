# Prespecified protocol (v1.0)

Machine-readable twin: [`../protocol.json`](../protocol.json). This file, `protocol.json`, the data build (`src/data.py`) and `data/manifest.json` were committed **before any model was fit, any cluster was formed, and before the test split was used for anything except counting its rows and events**. Later changes are listed under "Deviations" with reason and timing.

> **Synthetic-data warning.** All data are CMS DE-SynPUF *synthetic* Medicare claims. CMS states that "analyses using the DE-SynPUF to draw inferences about Medicare beneficiaries, providers, or the Medicare program will be misleading and often incorrect" (DE-SynPUF User Manual, section 6). Nothing here is a clinical or epidemiological finding. The project demonstrates a pipeline and methods only.

## 1. Questions

1. **Predictive.** From one year of claims (2008), how well can we rank and calibrate the risk of any inpatient admission in the next year (2009)? Secondary: risk of being in the top decile of 2009 Medicare payments.
2. **Phenotyping.** Do 2008 chronic-condition and utilization profiles fall into a small number of stable, describable groups, and how do next-year admission rates differ between them?
3. **Prescriptive.** With limited care-management capacity, whom should we contact to maximize admissions avoided, under explicitly *assumed* intervention effects? How does that choice compare with random, highest-cost and highest-risk targeting, and how sensitive is it to the assumptions?
4. **Validation design.** What pilot would test the assumed effect for real?

## 2. Data

- **Source.** CMS 2008–2010 Data Entrepreneurs' Synthetic Public Use File (DE-SynPUF), DE1.0 **Sample 1 only**: 2008 and 2009 Beneficiary Summary Files, 2008–2010 Inpatient, Outpatient and Carrier (1A + 1B) claims. URLs, retrieval dates, SHA-256 and the CMS terms text are in `data/manifest.json`. Raw files are gitignored. Prescription Drug Events and the 2010 file are not used.
- **Terms.** Free public-use download, no data use agreement, synthetic, no PHI.
- **Known synthetic artefacts** that matter for this design:
  - HMO months make no difference to observed claims. Inpatient-admission rates in 2009 are 16.3% with any HMO month and 16.5% with none (exploration, before this protocol; recomputed after the test by `scripts/data_checks.py` into `results/data_checks.json`). In real data HMO members have no fee-for-service claims. We still apply the real-world fee-for-service rule so the code would transfer to CMS Limited Data Sets.
  - Claim-line payments do not match beneficiary-file totals exactly (carrier-claims payments average higher than `MEDREIMB_CAR`).
  - The synthesis weakens multivariate relationships. **We expect modest discrimination and will not tune to chase it.**

### Cohort (counts from `data/manifest.json`)

| Step | N |
|---|---|
| 2008 Beneficiary Summary File, Sample 1 | 116,352 |
| Alive at end of 2008 | 114,538 |
| Also present in the 2009 Beneficiary Summary File | 114,538 |
| 12 months Part A and 12 months Part B in 2008 | 95,605 |
| 0 HMO months in 2008 (fee-for-service baseline) | **72,071** |

We do **not** condition on 2009 enrollment, because a deployed model could not know it. Only 18 cohort members have 0 Part A months in 2009. A sensitivity analysis excludes them. 1,117 members (1.5%) die in 2009. Death is a competing event that is **not modeled**. A member who dies without admission counts as "no admission".

## 3. Timing and leakage rules

- **Features use 2008 only:**
  - the 2008 Beneficiary Summary File;
  - claims whose from-date **and** thru-date are both in 2008.
- Stays spanning 2007/2008 or 2008/2009 are therefore not claim features. Their 2008 payments are still inside the 2008 beneficiary-file totals.
- **The 2009 Beneficiary Summary File is never read for features.** Its chronic-condition flags and payment totals are 2009 quantities. It is used only for the cost outcome, 2009 deaths and 2009 Part A months.
- **Primary outcome `y_ip`.** At least one inpatient claim whose admission date (`CLM_ADMSN_DT`, else `CLM_FROM_DT`) falls in 2009. Rows are deduplicated on `CLM_ID`.
- **Secondary outcome `high_cost`.** 2009 `MEDREIMB_IP + MEDREIMB_OP + MEDREIMB_CAR` ≥ the 90th percentile of that sum in the **training split**.
- **Excluded from predictors (`FORBIDDEN_PREDICTORS` in `src/data.py`, enforced by a test):**
  - every outcome-year field;
  - ID, state and county;
  - **race**. Race is used only for the subgroup audit; we choose not to build it into a targeting score. Sex and age are predictors.

### Features (54)

- **Demographics:** age (at 2008-12-31), female, ESRD.
- **Chronic conditions:** 11 CMS chronic-condition flags, plus their count.
- **2008 beneficiary-file payments:** Medicare, beneficiary and primary-payer amounts for inpatient, outpatient and carrier, plus the Medicare total.
- **Claim-derived 2008 utilization:**
  - inpatient claims, days and paid;
  - outpatient claims and paid;
  - carrier claims and paid;
  - ED visits (claims carrying HCPCS 99281–99285);
  - any inpatient stay.
- **Coarse diagnoses:** 19 ICD-9-CM chapter flags (17 numeric chapters, plus V and E codes) from any 2008 claim, and the number of distinct 3-digit codes.

## 4. Split

- **Rule.** Beneficiary-level and deterministic: `u = SHA256(DESYNPUF_ID | "prescriptive-care-lab-v1")` mapped to [0, 1). Train if u < 0.6, validation if u < 0.8, otherwise test. The rule uses the ID only. `data/membership.json` is committed, and a test recomputes it.
- **Result.** Train 43,488 (7,927 admissions), validation 14,295 (2,657), test 14,288 (2,696).
- **Use of each part:**
  - **Train:** every fit, including scaling, clustering and the high-cost threshold.
  - **Validation:** LightGBM early stopping, model selection for targeting, and SHAP.
  - **Test:** used once, after `freeze`, by `run test`, which writes `results/test_results.json` in exclusive-create mode.

## 5. Predictive models (fixed hyperparameters, no search)

- **Preprocessing** is fit on train only:
  - `log1p` on count and payment features (list in `protocol.json`);
  - then standardization for logistic regression.
  - LightGBM uses the same log-transformed matrix, unscaled.
- **Models:**
  - **Logistic regression:** L2, C = 1.0.
  - **LightGBM:**
    - learning rate 0.03, 31 leaves, min_child_samples 100;
    - feature and bagging fraction 0.8, λ₂ = 1;
    - up to 3,000 trees, with early stopping (100 rounds) on **validation** log-loss.
  - Both are fit for each outcome.
- **No recalibration.** Calibration is reported as-is. This matters for the prescriptive step (section 7.5).
- **Targeting score.** The model with the higher **validation** AUROC for `y_ip` becomes the risk score used for targeting. It is fixed before test.
- **Metrics on test:**
  - AUROC and AUPRC (average precision);
  - Brier score;
  - calibration intercept and slope (logistic recalibration of logit(p));
  - a 10-bin quantile reliability curve.
- **Uncertainty:**
  - 1,000 beneficiary-bootstrap resamples (seed 2026), percentile 95% CIs;
  - paired difference LightGBM − logistic, on the same resamples.
- **Interpretation.** SHAP TreeExplainer on 5,000 validation rows: mean |SHAP| ranking. Interpretation only, not causal.

## 6. Phenotyping (train only)

- **Features:**
  - the 11 chronic-condition flags and ESRD;
  - log1p of inpatient claims and days, outpatient and carrier claims, ED visits, distinct 3-digit diagnoses, and 2008 Medicare total.
- **Scaling.** Standardized with train statistics.
- **What is left out:**
  - **Age and sex are excluded.** A phenotype should be a clinical and utilization pattern, not a demographic one.
  - Outcomes are never used.
- **Candidates and selection:**
  - **Candidates:** k-means for k = 3…8 (n_init 10). The range is set by actionability: at least 3 groups so programs can differ, at most 8 so leaders can read them.
  - **Silhouette:** on a fixed 10,000-row train subsample.
  - **Stability:** 5 seeds, each fit on an 80% train subsample. Score the adjusted Rand index (ARI) of each fit's labels on full train versus the full-train reference fit.
  - **Cross-check:** Gaussian-mixture BIC (diagonal covariance).
  - **Selection rule:** among k with mean stability ARI ≥ 0.75, pick the highest silhouette. Ties go to the smaller k. If no k qualifies, pick the most stable k.
- **Outputs:**
  - Validation and test members are assigned to the nearest train centroid.
  - Phenotypes are described in plain language from their **train** profiles.
  - Admission rates are reported by phenotype for train and test.
- **Phenotype risk rank** = the train 2009 admission rate of each phenotype. It orders phenotypes for the heterogeneous-effect and hours scenarios (section 7). It is fixed in the dev phase.

## 7. Prescriptive allocation (evaluated on test)

### 7.1 Decision problem

- **Set-up:**
  - For each test member *i*: planned risk p_i (the frozen targeting model), an **assumed** relative risk reduction r_i, and a contact cost w_i.
  - x_i ∈ {0, 1} marks who is contacted.
- **Optimization:** maximize Σ x_i · p_i · r_i (planned avoided admissions), subject to Σ x_i · w_i ≤ B.
- **Solver:** an ILP in PuLP/CBC.
- **Equivalence with ranking:**
  - With uniform cost (w_i = 1) the ILP is exactly "sort by p_i·r_i and take the top B". A test checks the ILP against greedy.
  - With uniform r it is **identical to highest-risk targeting by construction**. Any difference between "optimized" and "highest risk" therefore comes only from the assumed heterogeneity in r_i or w_i, and will be described that way.

### 7.2 Assumptions (all labeled as assumptions, not estimates)

- **Capacity:** 2%, 5% and 10% of test members (B = 286, 714, 1,429 contacts).
- **Effect scenarios** for r:
  - U10, U20, U30: uniform 10%, 20%, 30%.
  - H_dec ("the sickest are least impactable"): 30% in the lowest-risk phenotype, falling linearly with phenotype risk rank to 10% in the highest-risk phenotype.
  - H_inc: the reverse, 10% → 30%.
- **Cost scenarios:**
  - `uniform`: one contact = 1 unit, so the budget is a head-count.
  - `hours`: assumed care-manager hours of 4 h in the lowest-risk phenotype, rising linearly to 8 h in the highest. Budget = 6 h × head-count capacity. This is a true 0/1 knapsack, solved by ILP and compared with the greedy ratio rule.

### 7.3 Policies (same budget)

1. **Random:** 1,000 draws; report the mean and the 2.5–97.5 percentiles.
2. **Highest 2008 cost:** `bene08_medreimb_total`.
3. **Highest predicted risk:** p_i.
4. **Optimized:** the ILP above.
5. **Oracle:** an upper bound that knows the realized 2009 outcome. Context only, not achievable.

Under the hours budget, rank-based policies take members in rank order and skip anyone who no longer fits.

### 7.4 Evaluation

- **Realized avoided admissions** = Σ over contacted members of y_i · r_i, where y_i is the **observed** 2009 outcome. We do not score with p_i, because that would let the model grade its own predictions.
- **Admissions captured** = Σ y_i over contacted members. This is the empirical, assumption-free part.
- **Planned avoided** = Σ p_i r_i, reported alongside so the gap between planned and realized (calibration) is visible.
- **Uncertainty:** 300 bootstrap resamples of the test set, with policies **re-selected in each resample**.
  - Uniform cost: the greedy sort, which is provably equal to the ILP.
  - Hours: the greedy ratio rule. The ILP–greedy gap is reported on the point estimate.
  - Paired differences: optimized − highest risk, optimized − highest cost, highest risk − highest cost, highest risk − random (expected).
- **Decision rule:** a policy is reported as "better" than another only if the paired 95% CI of the difference lies entirely above 0. Even then the claim is conditional on the assumed effects.
- **Sensitivity:**
  - all 5 effect × 2 cost × 3 capacity combinations;
  - repeat the U20 / uniform / 5% setting excluding the 18 members with 0 Part A months in 2009.

### 7.5 Why calibration matters here

- With uniform r, only the ranking matters.
- With heterogeneous r, the optimizer trades p_i against r_i across phenotypes. Miscalibration by phenotype therefore changes who is selected. We report calibration by phenotype on test.

### 7.6 Subgroup and fairness checks (test, descriptive)

- **Subgroups:** age group (<65, 65–74, 75–84, 85+), sex, and race (white, black, hispanic, other; SynPUF codes).
- **Risk model:** AUROC and calibration-in-the-large (mean p − observed rate) per subgroup.
- **Targeting policy (5%, U20, uniform cost):**
  - selection rate per group;
  - observed admission rate per group;
  - capture rate per group (share of the group's admissions that fall among contacted members);
  - bootstrap CIs.
- **Flag.** A group is flagged if its capture rate differs from the overall rate by > 5 percentage points with a CI that excludes the overall rate. Synthetic subgroup structure is not evidence about real disparities.

## 8. Pilot design (validation of the assumed effect)

- **Main design:** an individually randomized pilot among members the model would target.
  - **Inputs:** baseline risk = the observed 2009 admission rate among the top-5% targeted test members; RRR ∈ {10, 20, 30}%; two-sided α = 0.05; power 0.8.
  - **Output:** the two-proportion sample size per arm.
- **Alternative:** a stepped-wedge or cluster design by practice, with ICC ∈ {0.01, 0.05} and 50 targeted members per cluster. We report the design effect.
- **Primary pilot endpoint:** 12-month any-admission. Death is reported as a competing outcome.
- These numbers are planning arithmetic, not results.

## 9. Freeze and one-shot test

- `python -m src.run dev`:
  - fits on train;
  - early stopping, model choice and SHAP on validation;
  - k selection and the phenotype rank on train;
  - writes `results/dev_results.json` and the model artefacts.
- `python -m src.run freeze`:
  - refuses if the frozen files are uncommitted;
  - records the commit and SHA-256 of the code, protocol, lockfile, manifest, membership and model artefacts in `results/freeze.json`.
- `python -m src.run test`:
  - refuses unless every hash matches the committed `freeze.json`;
  - writes `results/test_results.json` in exclusive-create mode, so it runs once.

## 10. Compute

CPU only, no GPU, no paid API, no cloud, no LLM calls. Carrier CSVs (about 1.2 GB each) are streamed in 200k-row chunks. Peak RSS is recorded in the results.

## Deviations

All of the following were made before any model output was seen, unless stated otherwise.

1. **Negative payments clipped before `log1p`.**
   - 42 cohort members have negative 2008 payment amounts. These are claim adjustments: 8 in `bene08_medreimb_ip`/`ip_paid`, 34 in `bene08_medreimb_op`/`op_paid`, and 2 in the total.
   - The first `dev` call stopped on a guard that rejected negative values. The transform now clips at 0 before `log1p`. No model had been fit.
2. **Outcome code refactor.** The 2009 admission count was moved into the function `admissions_in_year` so it can be unit-tested. Rebuilt derived files are byte-identical (same SHA-256).
3. **ILP solver settings.** CBC runs with a relative MIP gap of 1e-5 and a 60 s time limit. Without the gap setting, a small synthetic test knapsack ran for more than 100 s. The fractional-knapsack LP bound is reported next to every ILP solution, so the remaining optimality gap is visible.
4. **LightGBM reproducibility settings.** `deterministic=True`, `force_row_wise=True` and 4 threads are computational settings, not hyperparameters.
5. **Validation rehearsal.** `dev` runs the prescriptive pipeline once on the validation split (20 bootstrap resamples) to catch bugs before the one-shot test. It is labeled a rehearsal, not a result.
   - The first rehearsal used the ILP. Its hours-budget solves often ran to the time limit, so the rehearsal took about 15 minutes.
   - The final `dev` run uses greedy in the rehearsal. The ILP path is covered by tests and runs in `test`.
   - That first run also failed to write JSON because of one NaN SHAP correlation (an unused feature). NaN is now written as null.
6. **Full validation dry run.** After `dev` and before `freeze`:
   - The test-phase evaluation was refactored into `evaluate_part`. A `dryrun` phase runs it on the **validation** split: 50 metric and 20 policy resamples, greedy instead of ILP. It writes `results/val_dryrun.json`.
   - Purpose: exercise every code path, including figures and tables, before the one-shot test.
   - `dev` logic, models and phenotypes are unchanged.
   - Every result now also records the optimized policy's relative gap to the LP bound.
7. **Phenotype names.** Plain-language phenotype names in `scripts/make_figures.py` were written from the train profiles before the test run.

---

# Addendum A (v1.1): program-effect estimation and demand forecasting

This addendum was written and committed **after** the one-shot test (9c15e92) and **before** any code for these two modules existed or was run. It adds two analyses requested for a second application (forecasting; causal/program-effectiveness analysis). It changes nothing in sections 1–10, and their results stand as frozen.

Before writing it, one descriptive look was taken at the monthly inpatient-admission counts for the whole Sample 1 population. No model was fit. That look showed:
- **January–February 2008 ramp-up:** 1,533 and 1,805 admissions, against roughly 2,300–2,600 from March 2008 on.
- **2010 collapse:** from 1,786 in January 2010 to 257 in December 2010. CMS documents lower 2010 claim counts "due to attrition from death, and some effects of disclosure treatment".
- **Steady decline through 2009**, consistent with a closed 2008 cohort that ages and dies with no new entrants.

These observations shaped section B.1 and are reported as findings, not hidden.

## A. Semi-synthetic program-effect (causal) estimation

**Purpose.** Show, against a *known* simulated truth, that a naive enrolled-vs-not comparison is confounded, and test whether propensity-score methods recover the effect. **The program and its effect are simulated.** Nothing here estimates a real care-management effect.

### A.1 Population and covariates

- **Population:** the 14,288 test-split members.
  - Their predictive evaluation is already complete and frozen, so re-using their 2009 outcomes here cannot affect it.
- **Covariates X:** the 54 frozen predictors, transformed with the frozen train-fit preprocessor (log1p, then standardization).
- **Untreated potential outcome:** Y0 = each member's **observed** 2009 any-admission.

### A.2 Data-generating process (simulated)

- **Risk index:** s = standardized logit of the frozen LightGBM 2009-admission risk. It is a non-linear function of X.
- **Enrollment:** T ~ Bernoulli(sigmoid(α + γ·s)).
  - α is solved numerically per scenario so that mean enrollment probability = 20%.
  - γ is the confounding strength: higher-risk members enroll more.
- **Effect:**
  - D ~ Bernoulli(RRR), drawn independently;
  - Y1 = Y0·(1 − D), so enrollment removes each would-be admission with probability RRR;
  - observed Y = T·Y1 + (1 − T)·Y0.
- **Estimand:** the sample average treatment effect on the risk-difference scale, ATE = mean(Y1 − Y0). It is computed exactly from the simulated potential outcomes in each replicate.
- **Grid:**
  - γ ∈ {0 (randomized), 0.5, 1, 2} × RRR ∈ {0, 0.15, 0.30}, giving 12 scenarios.
  - Plus an **unmeasured-confounding** stress test at γ = 1, RRR ∈ {0, 0.15}. Here enrollment uses s' = (s + u)/√2, where u = standardized(Y0 + N(0, 1)) is a hidden severity signal. u is correlated with the outcome and is **not** in X.

### A.3 Estimators (adjustment set = X only)

1. **Naive:** mean(Y | T=1) − mean(Y | T=0).
2. **Outcome regression (g-computation):** L2 logistic regression (C = 1) of Y on X, fit separately in each arm, then averaged over all members.
3. **IPW (Hájek, normalized weights):** propensity from L2 logistic regression (C = 1) on X, clipped to [0.01, 0.99].
4. **AIPW (doubly robust):** the propensity model and arm-specific outcome models above.

There is no cross-fitting: the models are low-dimensional (54 covariates, about 14k rows). This choice is stated as a limitation.

### A.4 Diagnostics and uncertainty

- **Balance:** absolute standardized mean differences (SMD) for each covariate, before and after IPW weighting.
  - Reported: the maximum |SMD| and the count of covariates with |SMD| > 0.1.
  - Also reported: propensity range and the Kish effective sample size of each arm.
- **Bootstrap CIs:** replicate 1 of each scenario gets 200 member-bootstrap resamples. All models are refit in each resample, with percentile 95% CIs, for all four estimators.
- **Repeated simulation:** 100 replicates per scenario, re-drawing T and D. Reported per estimator:
  - mean bias against the true ATE, and RMSE;
  - for IPW and AIPW, coverage of influence-function Wald 95% CIs. These ignore propensity-estimation error, which is usually conservative for IPW.
- **Prespecified reading:**
  - An estimator "recovers" the effect in a scenario if |mean bias| < 0.005 (half a percentage point) **and**, for IPW and AIPW, coverage is ≥ 0.90.
  - The unmeasured-confounding scenarios are expected to fail. Their purpose is to show that the methods cannot fix a confounder that was not measured.

## B. Monthly admission forecasting

### B.1 Series and window

- **Series:** monthly count of distinct inpatient claims (deduplicated on `CLM_ID`) by admission month, for **all** Sample 1 beneficiaries. The secondary series is monthly inpatient payments (`CLM_PMT_AMT`).
- **Modeling window: March 2008 – December 2009 (22 months).**
  - **January–February 2008 is excluded.** The ramp-up is left truncation at the start of the data window.
  - **2010 is excluded.** It shows the CMS-documented attrition/disclosure decline.
  - Both exclusions are shown in a figure and reported.
- **Consequence:** with 22 usable months there is less than two full seasonal cycles. **SARIMA and seasonal ETS are not identifiable and are not fit.** That is a finding about the data, not a modeling choice.

### B.2 Methods

1. **Naive:** last value.
2. **Seasonal naive:** same month one year earlier.
3. **Drift:** random walk with drift estimated on the training window.
4. **Damped additive-trend ETS** (statsmodels `ETSModel`, error="add", trend="add", damped_trend=True).

### B.3 Rolling-origin backtest

- **Origins:** first training window March 2008 – February 2009 (12 months), expanding by one month per origin. Forecasts at horizons h = 1 and h = 3, with targets up to December 2009. This gives 10 one-step and 8 three-step forecasts.
- **Accuracy metrics:** MAE, MAPE, and MASE (scaled by the in-sample one-step naive MAE of the first training window).
- **Prediction intervals:** 80% and 95%.
  - Naive, seasonal naive and drift use Gaussian intervals. The SD comes from the training-window one-step residuals of that method, scaled by √h.
  - Seasonal naive has no training residuals until month 13, so its residual SD is borrowed from naive.
  - ETS uses statsmodels' simulation-based intervals (1,000 paths, seed 2026).
- **Coverage** is the share of backtest targets inside the interval. With only 8–10 targets per horizon, coverage is reported together with its exact binomial 95% CI. **No method is declared calibrated or better on this evidence alone.**

## Addendum deviations

- **`statsmodels` dependency added after the test.** The frozen state is recoverable at commit e427563. The one-shot test cannot be re-run in any case, because `results/test_results.json` exists.
