# Daily Step Count Analysis (2024)

**Author:** Lindsay Gross and Alex Oh  
**Course:** AIPI 510 — Statistical Analysis Project  

---

## Overview
This project analyzes **daily step counts** for Lindsay and Alex using Apple Health data from **2024**.  
The goal was to test whether there were meaningful differences in daily activity patterns across people, weekdays vs. weekends, and months.

The analysis covers the full pipeline from **data cleaning and hypothesis testing** to **visualization and interpretation**, using Python and reproducible scripts.

---

## Research Question
> Do Lindsay and Alex differ in their average daily step counts, and do activity patterns vary across weekdays, weekends, or months?

---

##  Hypotheses
- **H1:** Lindsay and Alex’s mean daily steps are significantly different.  
- **H2:** Each person’s weekday and weekend step counts differ.  
- **H3:** Mean daily steps differ across months.

---

##  Experimental Design
- **Type:** Observational study  
- **Data source:** Apple Health step count exports  
- **Time frame:** January–December 2024  
- **Variables:**  
  - Independent: Person, Day Type (Weekday/Weekend), Month  
  - Dependent: Daily step count  
- **Controls:** Filtered to 2024 only, aggregated to one row per day  
- **Randomization:** Not applicable (observational dataset)

---

##  Data Collection & Cleaning
1. Exported Apple Health data to `export.xml`  
2. Converted to daily CSVs using `apple_steps.py`  
3. Filtered to include only **2024** using `filter_to_2024.py`  
4. Produced cleaned datasets:
   - `lindsay_steps_per_day_2024.csv`
   - `alex_steps_per_day_2024.csv`
5. Removed incomplete or invalid rows and standardized column names

---

## Statistical Analysis
Performed using `analysis.py`.

- **Descriptive Stats:** Mean, median, standard deviation, min, max  
- **Inferential Tests:**  
  - Welch’s t-test (Lindsay vs. Alex)  
  - Welch’s t-tests for weekday vs. weekend per person  
- **Power Analysis:** Estimated required sample size (α=0.05, power=0.80)

**Results Summary:**
| Test | Comparison | Mean₁ | Mean₂ | p-value | Significant? |
|------|-------------|-------|-------|----------|---------------|
| H1 | Lindsay vs Alex | 64
