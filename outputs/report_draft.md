# Daily Step Counts in 2024: A Comparative Statistical Analysis

## Abstract
We analyze daily step counts for two individuals across the 2024 calendar year to test differences between people, weekday–weekend patterns, and seasonal variation. We check assumptions, apply appropriate parametric or non-parametric tests, report effect sizes and confidence intervals, and perform a power analysis to justify sample size.

## Background & Motivation
Physical activity (proxied by step counts) varies across people and time. Understanding differences between individuals, weekday/weekend patterns, and seasonal shifts can inform habit-building strategies and health planning.

## Null and Alternative Hypotheses
- **H1:** μ_Lindsay = μ_Alex (null) vs μ_Lindsay ≠ μ_Alex (alt).
- **H2 (per person):** μ_weekday = μ_weekend vs μ_weekday ≠ μ_weekend.
- **H3:** Mean daily steps are equal across months vs at least one month differs.

## Experimental Design
- **Design:** Observational; unit = person-day in 2024.
- **Independent variables:** Person (between), Day type (weekday/weekend; within), Month (Jan–Dec).
- **Dependent variable:** Steps per day.
- **Controls:** Restrict to 2024; aggregate to daily totals.
- **Randomization:** Not applicable.

## Power Analysis
Two-sample t-test framework used for H1; if effect size is small, larger n improves power.

## Data Collection & Cleaning Procedures
- Source: Apple Health step counts exported to CSV.
- Standardized columns to `date` and `steps`; converted to datetime & numeric.
- Restricted to 2024 and aggregated to one row per day per person.
- Derived fields: weekday/weekend, month.
- See `steps_2024_tidy.csv` for final dataset.

## Statistical Analysis
- Assumption checks: Shapiro–Wilk for normality, Levene for homogeneity of variance.
- **H1:** Mann–Whitney U; p = 0.0000, mean diff (Lindsay − Alex) = -1530.0, 95% CI = n/a, Hedges g = -0.40.
- **H2:** See `h2_within_person_weekday_weekend.csv` for per-person comparisons.
- **H3:** See `h3_monthly_results.csv` and pairwise CSVs for month comparisons.
- When normality/variance assumptions failed, non-parametric alternatives were used.

## Interpretation of Results
Interpretation emphasizes effect sizes and confidence intervals alongside p-values to gauge practical significance (e.g., routine differences, schedule effects, seasonality).

## Limitations
Observational (no causal inference), device measurement noise, possible missing days, unmodeled confounders (e.g., weather, travel). Multiple comparisons addressed with Tukey HSD or Holm adjustment where applicable.

## Conclusion & Recommendations
Summarize whether differences are meaningful and suggest practical actions (e.g., targeted walks on low-activity days). Future work could incorporate weather/location or fit mixed-effects models for richer inference.

## References
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*.
- Lakens, D. (2013). Calculating and reporting effect sizes.
- Wilcoxon, Mann–Whitney (1945–47); Tukey (1949).
