# Daily Step Count Analysis (2024)

**Authors:** Lindsay Gross & Alex Oh  
**Course:** AIPI 510 — Statistical Analysis Project  

---

## Overview
This project compares **daily step counts** for Lindsay and Alex using Apple Health data from **2024**.  
We analyzed differences in activity **between people**, **across weekdays and weekends**, and **by month** using reproducible Python scripts and nonparametric statistical tests.

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
