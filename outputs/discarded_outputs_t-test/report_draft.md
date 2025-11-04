
### Summary of Results

**H1: Lindsay vs Alex (between-person difference)**

* Mean daily steps:

  * Lindsay = 6,457
  * Alex = 7,987
* Welch t = –5.41, p < 0.001 → **statistically significant difference**
* Interpretation: Alex walked significantly more on average than Lindsay.
* Effect size ≈ 0.40 (Cohen’s d ≈ medium).
* **Power analysis:** With d = 0.40, about 98 days per person would be needed for 80 % power—your data (357–366 days) easily meets that.

---

**H2: Weekday vs Weekend (within-person patterns)**

* **Lindsay:** Weekdays = 6,657 steps, Weekends = 5,953 steps (p = 0.12)
  → Not statistically significant but trend = more weekday steps.
* **Alex:** Weekdays = 7,893 steps, Weekends = 8,221 steps (p = 0.44)
  → No significant difference.

---

**Assumption checks (normality + variance):**

* Shapiro p values ≈ 1e-10 → non-normal distributions (but Welch t-test is robust).
* Levene p = 0.0006 → variances differ → Welch test was the right choice.

---

### Visuals

* **Lindsay vs Alex:** Boxplot clearly shows Alex’s median and upper range higher.
* **Monthly patterns:** Steps fluctuate across months (lowest in mid-year, peaks late year).

---

### Inference for Report

* **Null (H1):** μₗ = μₐ → Reject (p < 0.001).
* **H2:** Fail to reject (p > 0.05).
* **Conclusion:** Alex walks more overall; no consistent weekday/weekend or monthly pattern that’s statistically strong.