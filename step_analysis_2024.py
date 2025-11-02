

#!/usr/bin/env python3
"""
Step Count Statistical Analysis (2024, Lindsay vs Alex)

- Validates & loads two CSVs with columns: date, steps
- Forces 2024-only data (safeguard)
- Adds derived fields (weekday, weekend, month)
- Descriptives + hypothesis tests:
  H1: Between persons (t-test or Mann–Whitney, with effect sizes + CIs)
  H2: Weekday vs weekend within each person
  H3: Monthly differences (ANOVA+Tukey or Kruskal+Holm-adjusted pairwise)
- Power analysis (two-sample t-test, effect size from H1)
- Plots (time series, histograms, monthly boxplot)
- Draft Markdown report with all sections required in the assignment

Usage:
  python3 step_analysis_2024.py --lindsay lindsay_steps_per_day_2024.csv --alex alex_steps_per_day_2024.csv --outdir outputs
"""

import argparse
import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless savefig; prevents GUI hangs
import matplotlib.pyplot as plt
from scipy import stats

# Optional; we fall back gracefully if missing
HAVE_SM = True
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.weightstats import DescrStatsW, CompareMeans
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.stats.power import TTestIndPower
except Exception:
    HAVE_SM = False

# -------------------- helpers --------------------

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def load_steps_csv(path: str, person_label: str) -> pd.DataFrame:
    # Read and normalize headers (handle BOMs, spaces, case)
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)  # strip BOM if present
        .str.strip()
        .str.lower()
    )

    # Map common variants to required names
    colmap = {}

    # date-like
    for c in df.columns:
        if c in ("date", "day", "date_time"):
            colmap[c] = "date"
            break

    # steps-like
    for c in df.columns:
        if c in ("steps", "step", "stepcount", "step_count", "value", "count"):
            colmap[c] = "steps"
            break

    if "date" not in colmap.values() or "steps" not in colmap.values():
        raise ValueError(
            f"{path} must contain columns like 'date' and 'steps'. "
            f"Found columns: {list(df.columns)}"
        )

    df = df.rename(columns={k: v for k, v in colmap.items()})

    # Parse types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["steps"] = pd.to_numeric(df["steps"], errors="coerce")
    df = df.dropna(subset=["date", "steps"])

    # Keep only 2024
    df = df[df["date"].dt.year == 2024].copy()

    # Make date a real column (avoid FutureWarning), then aggregate
    df["date"] = df["date"].dt.date
    df = df.groupby("date", as_index=False)["steps"].sum()

    # Features
    df["person"] = person_label
    dt = pd.to_datetime(df["date"])
    df["weekday"] = dt.dt.day_name()
    df["is_weekend"] = df["weekday"].isin(["Saturday", "Sunday"])
    df["month"] = dt.dt.month
    df["month_name"] = dt.dt.month_name()
    return df



def cohen_d_from_groups(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    sx = np.nanvar(x, ddof=1)
    sy = np.nanvar(y, ddof=1)
    sp = ((nx-1)*sx + (ny-1)*sy) / (nx+ny-2)
    if sp <= 0:
        return np.nan
    return (np.nanmean(x) - np.nanmean(y)) / np.sqrt(sp)

def hedges_g(d, nx, ny):
    df = nx + ny - 2
    if df <= 0 or not np.isfinite(d):
        return np.nan
    J = 1 - (3 / (4*df - 1))
    return J * d

def write_markdown(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

# -------------------- main analysis --------------------

def run(args):
    print("[1/10] Starting analysis…")
    print(f"  Lindsay CSV: {args.lindsay}")
    print(f"  Alex CSV:    {args.alex}")
    print(f"  Outdir:      {args.outdir}")

    if not os.path.exists(args.lindsay):
        raise FileNotFoundError(f"Lindsay file not found: {args.lindsay}")
    if not os.path.exists(args.alex):
        raise FileNotFoundError(f"Alex file not found: {args.alex}")

    outdir = ensure_outdir(args.outdir)

    lindsay = load_steps_csv(args.lindsay, "Lindsay")
    alex    = load_steps_csv(args.alex,    "Alex")
    df = pd.concat([lindsay, alex], ignore_index=True).sort_values(["person","date"])
    tidy_path = os.path.join(outdir, "steps_2024_tidy.csv")
    df.to_csv(tidy_path, index=False)
    
    outdir = ensure_outdir(args.outdir)
    print("[2/10] Created output dir:", outdir)

    # ---------- Descriptives ----------
    desc = df.groupby("person").agg(
        days=("date","nunique"),
        mean_steps=("steps","mean"),
        median_steps=("steps","median"),
        std_steps=("steps","std"),
        min_steps=("steps","min"),
        max_steps=("steps","max"),
    ).reset_index()
    desc_path = os.path.join(outdir, "descriptives_by_person.csv")
    desc.to_csv(desc_path, index=False)
    
    outdir = ensure_outdir(args.outdir)
    print("[3/10] Created output dir:", outdir)

    # ---------- H1: Between-person ----------
    persons = ["Lindsay","Alex"]
    x = df.loc[df["person"]=="Lindsay","steps"].dropna().values
    y = df.loc[df["person"]=="Alex","steps"].dropna().values

    sw1 = stats.shapiro(x) if len(x) <= 5000 else (np.nan, np.nan)
    sw2 = stats.shapiro(y) if len(y) <= 5000 else (np.nan, np.nan)
    lev = stats.levene(x, y, center="median")

    normalish = (isinstance(sw1, tuple) and (np.isnan(sw1[1]) or sw1[1] > 0.05)) and \
                (isinstance(sw2, tuple) and (np.isnan(sw2[1]) or sw2[1] > 0.05))
    equal_var = lev.pvalue > 0.05

    if normalish:
        t_res = stats.ttest_ind(x, y, equal_var=equal_var)
        test_used = f"Two-sample t-test ({'pooled' if equal_var else 'Welch'})"
    else:
        t_res = stats.mannwhitneyu(x, y, alternative="two-sided")
        test_used = "Mann–Whitney U"

    d = cohen_d_from_groups(x, y)
    g = hedges_g(d, len(x), len(y))

    if HAVE_SM and normalish:
        cm = CompareMeans(DescrStatsW(x), DescrStatsW(y))
        ci_low, ci_high = cm.tconfint_diff(usevar='pooled' if equal_var else 'unequal')
    else:
        ci_low, ci_high = (np.nan, np.nan)

    h1_df = pd.DataFrame([{
        "test": test_used,
        "n_Lindsay": len(x), "n_Alex": len(y),
        "mean_Lindsay": float(np.nanmean(x)), "mean_Alex": float(np.nanmean(y)),
        "median_Lindsay": float(np.nanmedian(x)), "median_Alex": float(np.nanmedian(y)),
        "p_value": float(t_res.pvalue),
        "cohen_d": float(d) if np.isfinite(d) else np.nan,
        "hedges_g": float(g) if np.isfinite(g) else np.nan,
        "mean_diff_Lindsay_minus_Alex": float(np.nanmean(x) - np.nanmean(y)),
        "ci95_low": float(ci_low) if np.isfinite(ci_low) else np.nan,
        "ci95_high": float(ci_high) if np.isfinite(ci_high) else np.nan,
        "shapiro_p_Lindsay": float(sw1[1]) if isinstance(sw1, tuple) else np.nan,
        "shapiro_p_Alex": float(sw2[1]) if isinstance(sw2, tuple) else np.nan,
        "levene_p": float(lev.pvalue)
    }])
    h1_path = os.path.join(outdir, "h1_between_person_results.csv")
    h1_df.to_csv(h1_path, index=False)
    
    outdir = ensure_outdir(args.outdir)
    print("[4/10] Created output dir:", outdir)

    # ---------- H2: Weekday vs Weekend within-person ----------
    h2_records = []
    for p in persons:
        a = df[(df["person"]==p) & (~df["is_weekend"])]["steps"].dropna().values
        b = df[(df["person"]==p) & (df["is_weekend"])]["steps"].dropna().values
        if len(a) >= 2 and len(b) >= 2:
            sw_a = stats.shapiro(a) if len(a) <= 5000 else (np.nan, np.nan)
            sw_b = stats.shapiro(b) if len(b) <= 5000 else (np.nan, np.nan)
            lev2 = stats.levene(a, b, center='median')
            normalish2 = (isinstance(sw_a, tuple) and (np.isnan(sw_a[1]) or sw_a[1] > 0.05)) and \
                         (isinstance(sw_b, tuple) and (np.isnan(sw_b[1]) or sw_b[1] > 0.05))
            equal_var2 = lev2.pvalue > 0.05
            if normalish2:
                res = stats.ttest_ind(a, b, equal_var=equal_var2)
                used = f"Two-sample t-test ({'pooled' if equal_var2 else 'Welch'})"
            else:
                res = stats.mannwhitneyu(a, b, alternative="two-sided")
                used = "Mann–Whitney U"
            d2 = cohen_d_from_groups(a, b)
            g2 = hedges_g(d2, len(a), len(b))
            if HAVE_SM and normalish2:
                cm2 = CompareMeans(DescrStatsW(a), DescrStatsW(b))
                ci2_low, ci2_high = cm2.tconfint_diff(usevar='pooled' if equal_var2 else 'unequal')
            else:
                ci2_low, ci2_high = (np.nan, np.nan)
            h2_records.append({
                "person": p,
                "test": used,
                "n_weekday": len(a), "n_weekend": len(b),
                "mean_weekday": float(np.nanmean(a)), "mean_weekend": float(np.nanmean(b)),
                "median_weekday": float(np.nanmedian(a)), "median_weekend": float(np.nanmedian(b)),
                "p_value": float(res.pvalue),
                "cohen_d": float(d2) if np.isfinite(d2) else np.nan,
                "hedges_g": float(g2) if np.isfinite(g2) else np.nan,
                "mean_diff_weekday_minus_weekend": float(np.nanmean(a)-np.nanmean(b)),
                "ci95_low": float(ci2_low) if np.isfinite(ci2_low) else np.nan,
                "ci95_high": float(ci2_high) if np.isfinite(ci2_high) else np.nan,
                "shapiro_p_weekday": float(sw_a[1]) if isinstance(sw_a, tuple) else np.nan,
                "shapiro_p_weekend": float(sw_b[1]) if isinstance(sw_b, tuple) else np.nan,
                "levene_p": float(lev2.pvalue)
            })
    h2_df = pd.DataFrame(h2_records)
    h2_path = os.path.join(outdir, "h2_within_person_weekday_weekend.csv")
    h2_df.to_csv(h2_path, index=False)
    
    outdir = ensure_outdir(args.outdir)
    print("[5/10] Created output dir:", outdir)

    # ---------- H3: Monthly differences ----------
    results, pairwise_written = [], []
    scopes = [("Overall", df)] + [(p, df[df["person"]==p]) for p in persons]
    for scope_name, data in scopes:
        groups = [g["steps"].dropna().values for _, g in data.groupby("month")]
        
        
        outdir = ensure_outdir(args.outdir)
        print("[6/10] Created output dir:", outdir)
        
        # normality per group (Shapiro; skip if tiny/huge)
        ps = []
        for arr in groups:
            if 3 <= len(arr) <= 5000:
                ps.append(stats.shapiro(arr)[1])
            else:
                ps.append(np.nan)
        normalish3 = all(np.isnan(p) or p > 0.05 for p in ps)
        lev3 = stats.levene(*[a for a in groups if len(a)>=2], center='median') if len(groups) >= 2 else None
        equal_var3 = (lev3.pvalue > 0.05) if lev3 is not None else True

        tukey_df = pd.DataFrame()
        if normalish3 and equal_var3 and len(groups) >= 2:
            if HAVE_SM:
                model = smf.ols("steps ~ C(month)", data=data).fit()
                anova_tbl = sm.stats.anova_lm(model, typ=2)
                pval = float(anova_tbl["PR(>F)"].iloc[0])
                test_name = "One-way ANOVA"
                tk = pairwise_tukeyhsd(endog=data["steps"], groups=data["month"], alpha=0.05)
                tukey_df = pd.DataFrame(tk._results_table.data[1:], columns=tk._results_table.data[0])
            else:
                _, pval = stats.f_oneway(*groups)
                test_name = "One-way ANOVA (scipy)"
        else:
            if len(groups) >= 2:
                _, pval = stats.kruskal(*groups)
                test_name = "Kruskal–Wallis"
            else:
                pval = np.nan
                test_name = "Kruskal–Wallis"
            # pairwise via Mann–Whitney with Holm adjustment
            months = sorted(data["month"].unique().tolist())
            pairs = []
            for i in range(len(months)):
                for j in range(i+1, len(months)):
                    a = data[data["month"]==months[i]]["steps"].dropna().values
                    b = data[data["month"]==months[j]]["steps"].dropna().values
                    if len(a) >= 2 and len(b) >= 2:
                        u = stats.mannwhitneyu(a, b, alternative="two-sided")
                        pairs.append((months[i], months[j], u.pvalue))
            if pairs:
                ranked = sorted(pairs, key=lambda t: t[2])
                m = len(ranked)
                adj = []
                for k, (m1, m2, pv) in enumerate(ranked, start=1):
                    adj_p = min((m - k + 1) * pv, 1.0)  # Holm step-down
                    adj.append((m1, m2, pv, adj_p))
                tukey_df = pd.DataFrame(adj, columns=["group1","group2","raw_p","p_adj"])

        results.append({"scope": scope_name, "test": test_name, "p_value": pval})
        if not tukey_df.empty:
            path_pw = os.path.join(outdir, f"h3_month_pairwise_{scope_name.replace(' ','_')}.csv")
            tukey_df.to_csv(path_pw, index=False)
            pairwise_written.append(path_pw)

    h3_table = pd.DataFrame(results)
    h3_path = os.path.join(outdir, "h3_monthly_results.csv")
    h3_table.to_csv(h3_path, index=False)

    # ---------- Power analysis ----------
    power_plot_path = None
    if HAVE_SM and len(x) >= 5 and len(y) >= 5:
        d_eff = abs(cohen_d_from_groups(x, y))
        analysis = TTestIndPower()
        # Required n per group for 80% power (if effect size too tiny, fallback to 0.5)
        try:
            required_n = analysis.solve_power(effect_size=d_eff if np.isfinite(d_eff) and d_eff>0 else 0.5,
                                              power=0.8, alpha=0.05, ratio=len(y)/max(len(x),1))
        except Exception:
            required_n = np.nan
        ns = np.arange(10, min(366, max(len(x), len(y)) + 50), 5)
        powers = analysis.power(effect_size=d_eff if np.isfinite(d_eff) and d_eff>0 else 0.5,
                                nobs1=ns, alpha=0.05, ratio=len(y)/max(len(x),1))
        plt.figure()
        plt.plot(ns, powers)
        plt.axhline(0.8)
        plt.xlabel("Sample size (group 1)")
        plt.ylabel("Power")
        plt.title("Power Curve (Two-sample t-test)")
        power_plot_path = os.path.join(outdir, "power_curve.png")
        plt.savefig(power_plot_path, bbox_inches="tight")
        plt.close()
        pd.DataFrame({"n_group1": ns, "power": powers}).to_csv(os.path.join(outdir,"power_table.csv"), index=False)

    # ---------- Plots ----------
    for p in persons:
        sub = df[df["person"]==p].sort_values("date")
        plt.figure()
        plt.plot(pd.to_datetime(sub["date"]), sub["steps"])
        plt.xlabel("Date"); plt.ylabel("Steps"); plt.title(f"Daily Steps in 2024 — {p}")
        plt.savefig(os.path.join(outdir, f"ts_daily_{p}.png"), bbox_inches="tight"); plt.close()

        plt.figure()
        plt.hist(sub["steps"].dropna().values, bins=30)
        plt.xlabel("Steps"); plt.ylabel("Count of Days"); plt.title(f"Distribution of Daily Steps — {p}")
        plt.savefig(os.path.join(outdir, f"hist_{p}.png"), bbox_inches="tight"); plt.close()

    plt.figure()
    month_order = sorted(df["month"].unique().tolist())
    data_for_box = [df[df["month"]==m]["steps"].dropna().values for m in month_order]
    plt.boxplot(data_for_box, labels=[datetime(2024, m, 1).strftime("%b") for m in month_order])
    plt.xlabel("Month"); plt.ylabel("Steps"); plt.title("Steps by Month (Overall)")
    plt.savefig(os.path.join(outdir, "box_by_month_overall.png"), bbox_inches="tight"); plt.close()

    # ---------- Draft report ----------
    h1 = pd.read_csv(h1_path).iloc[0].to_dict()
    effect_str = f"Hedges g = {h1['hedges_g']:.2f}" if np.isfinite(h1['hedges_g']) else "Hedges g not computed"
    ci_str = f"[{h1['ci95_low']:.1f}, {h1['ci95_high']:.1f}]" if np.isfinite(h1['ci95_low']) and np.isfinite(h1['ci95_high']) else "n/a"

    power_snippet = ""
    if power_plot_path:
        d_eff = abs(cohen_d_from_groups(x, y))
        power_snippet = f"Estimated effect size (Cohen's d) ≈ {d_eff:.2f}. See power_curve.png and power_table.csv."

    title = "Daily Step Counts in 2024: A Comparative Statistical Analysis"
    abstract = (
        "We analyze daily step counts for two individuals across the 2024 calendar year to test "
        "differences between people, weekday–weekend patterns, and seasonal variation. We check assumptions, "
        "apply appropriate parametric or non-parametric tests, report effect sizes and confidence intervals, "
        "and perform a power analysis to justify sample size."
    )

    report_md = f"""# {title}

## Abstract
{abstract}

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
{power_snippet or "Two-sample t-test framework used for H1; if effect size is small, larger n improves power."}

## Data Collection & Cleaning Procedures
- Source: Apple Health step counts exported to CSV.
- Standardized columns to `date` and `steps`; converted to datetime & numeric.
- Restricted to 2024 and aggregated to one row per day per person.
- Derived fields: weekday/weekend, month.
- See `steps_2024_tidy.csv` for final dataset.

## Statistical Analysis
- Assumption checks: Shapiro–Wilk for normality, Levene for homogeneity of variance.
- **H1:** {h1['test']}; p = {h1['p_value']:.4f}, mean diff (Lindsay − Alex) = {h1['mean_diff_Lindsay_minus_Alex']:.1f}, 95% CI = {ci_str}, {effect_str}.
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
"""
    write_markdown(os.path.join(outdir, "report_draft.md"), report_md)

    print("\nDone. Key artifacts written to:", outdir)

# -------------------- CLI --------------------

if __name__ == "__main__":
    try:
        ap = argparse.ArgumentParser(description="2024 step-count analysis (Lindsay vs Alex)")
        ap.add_argument("--lindsay", required=True, help="CSV with columns date,steps for Lindsay")
        ap.add_argument("--alex", required=True, help="CSV with columns date,steps for Alex")
        ap.add_argument("--outdir", default="outputs", help="Output directory (default: outputs)")
        run(ap.parse_args())
    except Exception as e:
        import traceback
        print("\n ERROR:", e)
        traceback.print_exc()
        raise

