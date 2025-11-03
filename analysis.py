
#!/usr/bin/env python3
"""
Simple 2024 step-count analysis (Lindsay vs Alex)

What it does:
- Loads two CSVs: lindsay_steps_per_day_2024.csv, alex_steps_per_day_2024.csv
- Filters to 2024 (safeguard), computes basic descriptives
- H1: Lindsay vs Alex (Welch t-test)  + quick assumption checks
- H2: Weekday vs Weekend (per person; Welch t-test)
- Power analysis (two-sample, using observed effect size)
- Saves two plots to outputs/: boxplot by person, boxplot by month
- Prints all key stats to the console

Run:
    python3 analysis.py
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestIndPower
import matplotlib
matplotlib.use("Agg")  # headless (saves images only)
import matplotlib.pyplot as plt

# ---- Settings (hard-coded for simplicity) ----
LINDSAY_CSV = "lindsay_steps_per_day_2024.csv"
ALEX_CSV    = "alex_steps_per_day_2024.csv"
OUTDIR      = "outputs"

def load_steps(csv_path: str) -> pd.DataFrame:
    """Load a CSV with columns date, steps; filter to 2024; return DataFrame."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    # Normalize headers a bit, then ensure required columns
    df.columns = df.columns.str.strip().str.lower().str.replace("\ufeff", "", regex=False)
    if "date" not in df.columns or "steps" not in df.columns:
        raise ValueError(f"{csv_path} must have columns: date, steps (found: {list(df.columns)})")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["steps"] = pd.to_numeric(df["steps"], errors="coerce")
    df = df.dropna(subset=["date", "steps"])
    df = df[df["date"].dt.year == 2024].copy()  # safeguard
    return df[["date", "steps"]]

def describe(name: str, s: pd.Series):
    print(f"\n{name} — Descriptives (daily steps, 2024)")
    print(f"  days:   {s.shape[0]}")
    print(f"  mean:   {s.mean():.0f}")
    print(f"  median: {s.median():.0f}")
    print(f"  std:    {s.std():.0f}")
    print(f"  min:    {s.min():.0f}")
    print(f"  max:    {s.max():.0f}")

def welch_t(x, y, label=""):
    res = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
    print(f"\n{label} — Welch t-test")
    print(f"  mean(x)={np.nanmean(x):.0f}, mean(y)={np.nanmean(y):.0f}")
    print(f"  t={res.statistic:.3f}, p={res.pvalue:.4f}")
    return res

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # Load data
    l = load_steps(LINDSAY_CSV)
    a = load_steps(ALEX_CSV)

    # Basic descriptives
    describe("Lindsay", l["steps"])
    describe("Alex",    a["steps"])

    # ---- H1: Lindsay vs Alex (Welch t-test) ----
    # Assumption checks (lightweight)
    try:
        p_shap_l = stats.shapiro(l["steps"]).pvalue if len(l) <= 5000 else np.nan
        p_shap_a = stats.shapiro(a["steps"]).pvalue if len(a) <= 5000 else np.nan
    except Exception:
        p_shap_l = p_shap_a = np.nan  # if Shapiro fails on large n, that's fine
    p_levene = stats.levene(l["steps"], a["steps"], center="median").pvalue

    print("\nAssumption checks (H1)")
    print(f"  Shapiro p (Lindsay): {p_shap_l if not np.isnan(p_shap_l) else 'n/a'}")
    print(f"  Shapiro p (Alex):    {p_shap_a if not np.isnan(p_shap_a) else 'n/a'}")
    print(f"  Levene p:            {p_levene:.4f}")

    res_h1 = welch_t(l["steps"], a["steps"], label="H1: Lindsay vs Alex")

    # ---- H2: Weekday vs Weekend (per person; Welch t-tests) ----
    l["weekday"] = l["date"].dt.dayofweek  # Mon=0 ... Sun=6
    a["weekday"] = a["date"].dt.dayofweek
    l_wd, l_we = l[l["weekday"] < 5]["steps"], l[l["weekday"] >= 5]["steps"]
    a_wd, a_we = a[a["weekday"] < 5]["steps"], a[a["weekday"] >= 5]["steps"]

    welch_t(l_wd, l_we, label="H2: Lindsay Weekday vs Weekend")
    welch_t(a_wd, a_we, label="H2: Alex Weekday vs Weekend")

    # ---- Power analysis (two-sample; based on observed effect size) ----
    # Cohen's d using pooled SD
    pooled_sd = np.sqrt((l["steps"].std(ddof=1)**2 + a["steps"].std(ddof=1)**2) / 2.0)
    d = abs(l["steps"].mean() - a["steps"].mean()) / pooled_sd if pooled_sd > 0 else 0.0
    analysis = TTestIndPower()
    try:
        n_needed = analysis.solve_power(effect_size=d if d > 0 else 0.5, power=0.80, alpha=0.05)
    except Exception:
        n_needed = np.nan
    print("\nPower analysis (two-sample t-test, α=0.05, power=0.80)")
    print(f"  Observed effect size (d): {d:.2f}")
    print(f"  Approx. required n per group: {int(round(n_needed)) if np.isfinite(n_needed) else 'n/a'}")

    # ---- Simple plots ----
    # Boxplot by person
    plt.figure()
    plt.boxplot([l["steps"], a["steps"]], labels=["Lindsay", "Alex"])
    plt.ylabel("Daily steps")
    plt.title("2024 Daily Steps — Lindsay vs Alex")
    plt.savefig(os.path.join(OUTDIR, "box_lindsay_vs_alex.png"), bbox_inches="tight")
    plt.close()

    # Boxplot by month (overall, both people)
    la = pd.concat([l.assign(person="Lindsay"), a.assign(person="Alex")], ignore_index=True)
    la["month"] = la["date"].dt.month
    month_order = sorted(la["month"].unique())
    data_for_box = [la[la["month"] == m]["steps"].values for m in month_order]
    plt.figure()
    plt.boxplot(data_for_box, labels=[str(m) for m in month_order])
    plt.xlabel("Month (2024)")
    plt.ylabel("Daily steps")
    plt.title("Steps by Month (2024, combined)")
    plt.savefig(os.path.join(OUTDIR, "box_by_month.png"), bbox_inches="tight")
    plt.close()

    # Optional: small descriptives table to file
    desc = pd.DataFrame({
        "person": ["Lindsay", "Alex"],
        "days":   [l.shape[0], a.shape[0]],
        "mean":   [l["steps"].mean(), a["steps"].mean()],
        "median": [l["steps"].median(), a["steps"].median()],
        "std":    [l["steps"].std(), a["steps"].std()],
        "min":    [l["steps"].min(), a["steps"].min()],
        "max":    [l["steps"].max(), a["steps"].max()],
    })
    desc.to_csv(os.path.join(OUTDIR, "descriptives.csv"), index=False)

    print(f"\nSaved figures to '{OUTDIR}/':")
    print("  - box_lindsay_vs_alex.png")
    print("  - box_by_month.png")
    print(f"Saved table: {OUTDIR}/descriptives.csv")
    print("\nDone.")

if __name__ == "__main__":
    main()
