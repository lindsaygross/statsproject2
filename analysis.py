
"""
Simple 2024 step-count analysis (Lindsay vs Alex)
- Uses nonparametric tests due to non-normality.
- Mann–Whitney U for independent samples (daily data).
- Wilcoxon signed-rank on week-level paired means (weekday vs weekend).
- Reports nonparametric effect sizes: Cliff's delta (independent), rank-biserial (Wilcoxon).
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LINDSAY_CSV = "lindsay_steps_per_day_2024.csv"
ALEX_CSV    = "alex_steps_per_day_2024.csv"
OUTDIR      = "outputs"

USE_WILCOXON_WEEK_PAIRS = False  

def load_steps(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip().str.lower().str.replace("\ufeff", "", regex=False)
    if "date" not in df.columns or "steps" not in df.columns:
        raise ValueError(f"{csv_path} must have columns: date, steps (found: {list(df.columns)})")
    df["date"]  = pd.to_datetime(df["date"], errors="coerce") # Used chatgpt5 at 5:30pm on 11/1/2025 to help with date parsing and to add error handling
    df["steps"] = pd.to_numeric(df["steps"], errors="coerce") # Used chatgpt5 at 5:30pm on 11/1/2025 to help with date parsing and to add error handling
    df = df.dropna(subset=["date", "steps"])
    df = df[df["date"].dt.year == 2024].copy()
    return df[["date", "steps"]]

def describe(name: str, s: pd.Series):
    print(f"\n{name} — Descriptives (daily steps, 2024)")
    print(f"  days:   {s.shape[0]}")
    print(f"  mean:   {s.mean():.0f}")
    print(f"  median: {s.median():.0f}")
    print(f"  std:    {s.std():.0f}")
    print(f"  min:    {s.min():.0f}")
    print(f"  max:    {s.max():.0f}")

# Nonparametric tests + effect sizes
def cliff_delta(x, y):
    # Used chatgpt at 6:30pm on 11/1/2025 to help write this function for cliff's delta/understand the formula
    """Cliff's delta: P(X>Y) - P(X<Y)."""
    x = np.asarray(x); y = np.asarray(y)
    n_x, n_y = len(x), len(y)
    count = 0
    for xi in x:
        # Used chatgpt at 7:00pm on 11//2025 to help write this loop
        count += np.sum(xi > y) - np.sum(xi < y)
    return count / (n_x * n_y)

def rank_biserial_from_wilcoxon(T, n):
    # Rank-biserial correlation for Wilcoxon signed-rank:
    # Used chatgpt at 7:10pm on 11/1/2025 to help with this formula
    return (2*T)/(n*(n+1)) - 1

def mann_whitney_test(x, y, label=""):
    # Used chatgpt at 6:50pm on 11/1/2025 to help write this function for mann-whitney
    res = stats.mannwhitneyu(x, y, alternative="two-sided")
    d = cliff_delta(x, y)
    print(f"\n{label} — Mann–Whitney U")
    print(f"  median(x)={np.nanmedian(x):.0f}, median(y)={np.nanmedian(y):.0f}")
    print(f"  U={res.statistic:.0f}, p={res.pvalue:.4g}")
    print(f"  Cliff's delta={d:.3f}") # small: 0.11, medium 0.28, and large effect ≈ 0.43
    return res

def wilcoxon_week_pairs(df, label=""):
    # Used chatgpt at 7:20pm on 11/1/2025 to help write this function for wilcoxon week pairs
    """
    Build paired week-level weekday/weekend means, then Wilcoxon signed-rank.
    """
    tmp = df.copy()
    tmp["dow"] = tmp["date"].dt.dayofweek
    tmp["year_week"] = tmp["date"].dt.strftime("%G-%V")   
    weekday_means = (tmp[tmp["dow"] < 5]
                     .groupby("year_week")["steps"].mean()
                     .rename("weekday_mean"))
    weekend_means = (tmp[tmp["dow"] >= 5]
                     .groupby("year_week")["steps"].mean()
                     .rename("weekend_mean"))
    pairs = pd.concat([weekday_means, weekend_means], axis=1).dropna()
    if pairs.empty or len(pairs) < 3:
        return None

    x = pairs["weekday_mean"].values
    y = pairs["weekend_mean"].values
    res = stats.wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
    n = len(pairs)
    r_rb = rank_biserial_from_wilcoxon(res.statistic, n)
    print(f"\n{label} — Wilcoxon signed-rank (weekly paired means)")
    print(f"  weeks paired: {n}")
    print(f"  median(weekday_means)={np.median(x):.0f}, median(weekend_means)={np.median(y):.0f}")
    print(f"  W={res.statistic:.0f}, p={res.pvalue:.4g}")
    print(f"  Rank-biserial r ≈ {r_rb:.3f}  (~0.1 small, ~0.3 medium, ~0.5 large)")
    return res

def normality_and_variance_checks(l_steps, a_steps):
    try:
        p_shap_l = stats.shapiro(l_steps).pvalue if len(l_steps) <= 5000 else np.nan # Used chatgpt at 8:00pm on 11/1/2025 to help add shapiro test with exception handling
        p_shap_a = stats.shapiro(a_steps).pvalue if len(a_steps) <= 5000 else np.nan
    except Exception:
        p_shap_l = p_shap_a = np.nan
    p_levene = stats.levene(l_steps, a_steps, center="median").pvalue
    print("\nAssumption checks (reporting only)")
    print(f"Shapiro p (Lindsay): {p_shap_l if not np.isnan(p_shap_l) else 'n/a'}")
    print(f"Shapiro p (Alex):    {p_shap_a if not np.isnan(p_shap_a) else 'n/a'}")
    print(f"Levene p:            {p_levene:.4f}")

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # Load
    l = load_steps(LINDSAY_CSV)
    a = load_steps(ALEX_CSV)

    # Descriptives
    describe("Lindsay", l["steps"])
    describe("Alex",    a["steps"])

    # Normality/variance
    normality_and_variance_checks(l["steps"], a["steps"])

    # H1: Lindsay vs Alex (independent) — Mann–Whitney U
    mann_whitney_test(l["steps"], a["steps"], label="H1: Lindsay vs Alex")

    # Weekday vs Weekend per person
    l["dow"] = l["date"].dt.dayofweek
    a["dow"] = a["date"].dt.dayofweek
    # Used chatgpt at 8:00pm on 11/1/2025 to edit this section for weekday vs weekend to make more readable/efficient
    l_wd, l_we = l[l["dow"] < 5]["steps"].values, l[l["dow"] >= 5]["steps"].values
    a_wd, a_we = a[a["dow"] < 5]["steps"].values, a[a["dow"] >= 5]["steps"].values

    if USE_WILCOXON_WEEK_PAIRS:
        # Pair by week, then Wilcoxon
        wilcoxon_week_pairs(l, label="H2: Lindsay Weekday vs Weekend")
        wilcoxon_week_pairs(a, label="H2: Alex Weekday vs Weekend")
    else:
        # Treat daily obs as independent, use Mann–Whitney
        mann_whitney_test(l_wd, l_we, label="H2: Lindsay Weekday vs Weekend")
        mann_whitney_test(a_wd, a_we, label="H2: Alex Weekday vs Weekend")

    # PLOTS 
    plt.figure()
    plt.boxplot([l["steps"], a["steps"]], labels=["Lindsay", "Alex"])
    plt.ylabel("Daily steps")
    plt.title("2024 Daily Steps — Lindsay vs Alex")
    plt.savefig(os.path.join(OUTDIR, "box_lindsay_vs_alex.png"), bbox_inches="tight")
    plt.close()

    # Boxplot by month
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

    # Descriptives table 
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
