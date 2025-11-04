# Daily Step Count Analysis (2024)

**Authors:** Lindsay Gross & Alex Oh  
**Course:** AIPI 510 — Statistical Analysis Project  

---

## Overview
This project compares **daily step counts** for Lindsay and Alex using Apple Health data from **2024**.  
We analyzed differences in activity **between people**, **across weekdays and weekends**, and **by month** using reproducible Python scripts and nonparametric statistical tests.

pip install -r requirements.txt
```

## Data Sources

- `alex_steps_per_day.csv` - Alex's daily step counts
- `lindsay_steps_per_day.csv` - Lindsay's daily step counts
- **Analysis Period**: 2024-2025 (655 days for Alex, 670 days for Lindsay)

## Statistical Analysis Process (`finaltests.ipynb`)

### 1. Data Preprocessing

**Rolling Average Methodology**:
- **Window Size**: 3-day rolling averages to smooth daily variations
- **Bootstrap Sampling**: 500 randomly selected 3-day blocks per individual
- **Purpose**: Reduce noise and capture underlying activity patterns

```python
# Key parameters
ROLLING_WINDOW = 3  # 3 day blocks
SAMPLE_SIZE = 500   # Bootstrap sample size
RANDOM_STATE = 42   # For reproducibility
```

### 2. Normality Assessment

**Shapiro-Wilk Test Results**:
- **Alex**: W = 0.9421, p = 8.88e-24
- **Lindsay**: W = 0.9128, p = 1.61e-28
- **Conclusion**: Both distributions are significantly non-normal (p < 0.05)

**Implication**: Non-parametric tests are required for valid statistical inference.

### 3. Distribution Comparison

**Descriptive Statistics**:
- **Alex Mean**: 7,545.7 steps/day
- **Lindsay Mean**: 7,115.9 steps/day
- **Difference**: 429.8 steps/day (Alex higher)

**Effect Size**:
- **Cohen's d**: 0.105 (small effect size)
- **Interpretation**: Practically small but statistically detectable difference

### 4. Statistical Tests Performed

#### Kolmogorov-Smirnov Two-Sample Test
- **Purpose**: Compare entire distribution shapes
- **Result**: D = 0.157, p = 1.78e-16
- **Conclusion**: Highly significant difference in distributions

#### Bootstrap Confidence Intervals for KS Statistic
- **Method**: 1,000 bootstrap resamples
- **95% CI**: [0.1326, 0.1873]
- **Original KS**: 0.1567 ✓ (falls within CI)
- **Validation**: Bootstrap confirms the reliability of the KS test result

#### Mann-Whitney U Test
- **Purpose**: Non-parametric test for median differences
- **Result**: U = 1,233,835, p = 4.47e-06
- **Conclusion**: Significant difference in central tendencies

#### Wilcoxon Signed-Rank Test Analysis
- **Method**: Tested multiple hypothesized medians (6,400-7,800 steps)
- **Purpose**: Identify confidence intervals for true medians
- **Visualization**: P-value curves showing median estimation ranges

### 5. Key Visualizations

1. **Distribution Comparison Histograms**: Overlaid histograms showing step count distributions
2. **Q-Q Plots**: Demonstrate non-normal distributions for both individuals
3. **Empirical Cumulative Distribution Functions (ECDF)**: Visualize the maximum difference (D = 0.157) between distributions
4. **Wilcoxon P-value Curves**: Show confidence regions for median estimates

## Key Findings

### Statistical Significance
- **Highly significant differences** detected across multiple tests (all p < 0.001)
- **Robust results** confirmed through bootstrap validation
- **Non-parametric approach** appropriate due to non-normal distributions

### Practical Significance
- **Small effect size** (Cohen's d = 0.105) suggests limited practical importance
- **Mean difference** of ~430 steps/day is detectable but modest
- **Individual variation** is substantial within each person's data

### Methodological Strengths
- **Rolling averages** reduce daily noise while preserving patterns
- **Bootstrap confidence intervals** provide robust uncertainty quantification
- **Multiple statistical tests** ensure comprehensive analysis
- **Reproducible methodology** with fixed random seeds

## Technical Implementation

**Key Libraries**:
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computations
- `scipy.stats` - Statistical tests (Shapiro-Wilk, KS, Mann-Whitney, Wilcoxon)
- `matplotlib` - Data visualization
- `statsmodels` - Q-Q plots and advanced statistical modeling

**Reproducibility**:
- Fixed random state (RANDOM_STATE = 42)
- Documented parameters and methodology
- Clear data filtering criteria (2024-2025 period)

## Files Structure

- `finaltests.ipynb` - **Main analysis notebook** (this README focuses on this file)
- `assumptions.ipynb` - Preliminary assumption testing
- `stats.ipynb` - Basic statistical exploration
- `step_analysis_2024.py` - Python script version of analysis
- `outputs/` - Generated plots and results

## Conclusion

The analysis demonstrates **statistically significant but practically small differences** in daily step patterns between Alex and Lindsay. The robust statistical methodology, including bootstrap validation and multiple non-parametric tests, provides high confidence in the results while acknowledging the limited practical significance of the observed differences.


README built by Claude Sonnet 4 via the Windsurf Application.
---

## Methods
- **Data:** Apple Health exports filtered to 2024  
- **Variables:** Person, Day Type (Weekday/Weekend), Month  
- **Tests:** Mann–Whitney U (independent) & Wilcoxon signed-rank (paired)  
- **Effect Sizes:** Cliff’s δ and rank-biserial r  
- **Libraries:** `pandas`, `scipy`, `matplotlib`

---

## Results

| Comparison | Median 1 | Median 2 | p-value | Effect Size | Interpretation |
|:------------|:----------|:----------|:---------|:--------------|:----------------|
| **Lindsay vs Alex** | 6,391 | 7,615 | 2.4 × 10⁻⁸ | δ = –0.24 | Alex walked significantly more |
| **Lindsay Weekday–Weekend** | 6,845 | 4,985 | 0.13 | δ = +0.10 | Not significant |
| **Alex Weekday–Weekend** | 7,552 | 7,788 | 0.46 | δ = –0.05 | Not significant |

**Monthly trends:** Natural variation, no consistent seasonal pattern.

---

## Visuals
-  `box_lindsay_vs_alex.png` — Between-person step distributions  
-  `box_by_month.png` — Monthly step trends  

---

## Interpretation
Alex consistently walked more than Lindsay, with a small-to-moderate effect size.  
Neither participant showed significant weekday-weekend or monthly differences, suggesting **stable daily routines**.

---

## Limitations
Observational data only; device differences and contextual factors (weather, travel, etc.) were not controlled.

---

## Future Work
Add contextual data (e.g., weather or events) and explore long-term or clustered activity patterns.

---

**References:**  
Schäfer & Schwarz (2019). *The Meaningfulness of Effect Sizes in Psychological Research.* *Frontiers in Psychology*, 10. [https://doi.org/10.3389/fpsyg.2019.00813](https://doi.org/10.3389/fpsyg.2019.00813)
