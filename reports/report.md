# Whom should care managers call first?

*A short brief for clinical and operations leaders. A one-page visual version is in [`summary.html`](summary.html).*

> **Read this first.**
> - **The data are synthetic.** Everything here was built on CMS's *synthetic* Medicare claims (DE-SynPUF). CMS warns that conclusions about real Medicare patients drawn from these files "will be misleading and often incorrect".
> - **The benefit of a care-management call is assumed.** We assumed it lowers a patient's chance of admission by 10–30%. We did not measure it.
> - **What this shows.** It demonstrates *how* the decision would be made and checked. It does not tell you what would happen in your population.

## The decision

- We have enough care managers to contact 2%, 5% or 10% of members this year.
- We want to choose the members in a way that prevents the most hospital admissions next year.

## What we did, in plain terms

1. **Predicted next-year admission risk.** We used one year of claims history: diagnoses, chronic conditions, visits, hospital stays and costs.
   - We tested the prediction on 14,288 members the model had never seen. Of them, 2,696 (18.9%) were admitted the following year.
2. **Grouped members into 8 recognizable profiles.** Examples are "cardiometabolic multimorbidity", "kidney failure / dialysis" and "recent hospital stays, complex".
3. **Chose whom to contact under a fixed budget.** We compared four ways of choosing:
   - at random;
   - the most expensive members last year (a common rule of thumb);
   - the highest predicted risk;
   - an optimizer that also weighs how much each profile is *assumed* to benefit and how many care-manager hours each profile needs.

## What we found (test members only)

**1. Choosing by predicted risk reaches many more of next year's admissions than choosing by last year's cost.** This finding needs no assumption about how well care management works.

| If we can contact… | Random choice: members later admitted | Highest last-year cost | Highest predicted risk |
|---|---|---|---|
| 2% (286 people) | 54 | 107 | **152** (53% of those contacted) |
| 5% (714 people) | 135 | 277 | **345** (48%) |
| 10% (1,429 people) | 269 | 525 | **627** (44%) |

- At 5% capacity, risk-based targeting reaches **68 more** members who go on to be admitted than cost-based targeting. The 95% interval is 39 to 98.
- Last year's cost is a weaker signal than a model that also uses diagnoses and visit patterns.

**2. How many admissions that prevents depends entirely on how well the program works.**

- **If calls cut each person's admission risk by 20%:** contacting the highest-risk 5% would prevent about **69 admissions** (95% interval 65–74). The alternatives prevent fewer:
  - highest-cost targeting: about 55;
  - random choice: about 27.
- **If the effect is 10% or 30%:** every figure scales down or up in proportion. The ranking of the approaches stays the same.

**3. The "smart optimizer" only beats simple risk ranking if the program works better for some kinds of patients than others.**

- **If every member benefits equally,** the optimizer picks exactly the same people as "highest risk first". There is nothing extra to gain.
- **If the sickest benefit least** (for example, people on dialysis or with repeated recent stays), the optimizer shifts calls toward moderately ill members:
  - with a head-count budget, it prevents about **11 more admissions at 5% capacity** (95% interval 5–16);
  - if sicker members also need more care-manager hours, about **22 more** (16–29).
- **If the sickest benefit most,** the optimizer's gain is small and uncertain at 2–5% capacity. At 10% it is about 16 more.
- In the "sickest benefit least" case, cost-based targeting even does slightly *worse* than random once care-manager hours are counted.

**In short:** the "right" list depends on something we don't yet know, namely who actually benefits. That is the reason to run a pilot before scaling.

**4. Profiles.** Next-year admission rates in test members ranged from **1%** to **36%**:

| Profile | Share of members | Admission rate |
|---|---|---|
| No recorded care in the prior year | about 21% | 1% (a likely artefact of the synthetic data) |
| Light use, few conditions | | 13% |
| Cardiometabolic | | 24–28% |
| Cancer | | 27% |
| Kidney failure / dialysis | | 33% |
| Recent hospital stays, complex | | 36% |
| Stroke / TIA with hospital stays | | 36% |

The profiles are useful for describing members and planning programs. They are not sharp, distinct disease types.

**5. Fairness check.**
- At 5% capacity, the risk-based list reaches a smaller share of admissions among **65–74-year-olds**: 6.6%, versus 12.8% overall. It reaches more among **under-65s** (18%) and **over-85s** (22%).
- Predicted risk was slightly **below** the observed rate for Black members: 18.2% predicted versus 20.7% observed.
- Race was **not** used to make predictions. It was used only for this check.
- In a real deployment these gaps would need review before go-live. In synthetic data they are not evidence about real disparities.

**6. Are the risk percentages trustworthy?** On average, yes:
- in the decile predicted at 31% on average, 33% were admitted;
- in the top decile, predicted 42% and observed 44%;
- the model's extremes are slightly too cautious (calibration slope 1.11).

This matters because the optimizer compares risk *levels* across profiles, not just the ranking.

## What it would take to know the real effect

- **Pilot design.** Randomize members whom the model would select. Half receive care management and half receive usual care. Compare admissions over 12 months.
- **Size.** Using the observed admission rate among the top-5% list (48%), each arm needs about:
  - **180** members to detect a 30% reduction;
  - **412** to detect 20%;
  - **1,669** to detect 10%;
  - at the usual 5% significance and 80% power.
- **If randomizing by clinic instead of by member** (cluster or stepped-wedge rollout): the numbers grow by roughly 1.5–3.5×, depending on how similar patients within a clinic are.
- **To learn whether the effect differs by profile:** randomize within each profile. That needs a larger second phase.
- **Afterwards:** the measured effect replaces the assumption, and the call list is recomputed.

## Limits

- **Synthetic data.** Real Medicare data would give different numbers.
- **Assumed benefit.** It was not measured.
- **Deaths not modeled.** 1.5% of members died during the outcome year, and these deaths are not modeled.
- **Single dataset, no outside validation.** It used one year of history and one sample of the synthetic files.
- **Research only.** This is research software. It is not a clinical tool, and no patient was contacted.

*All figures come from `results/test_results.json` and `results/dev_results.json`. Full tables with confidence intervals are in `results/tables.md`.*
