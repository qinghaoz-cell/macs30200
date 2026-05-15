import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"
FIG = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def clean_data() -> pd.DataFrame:
    df = pd.read_csv(DATA / "salaries.csv")
    df = df[df["employment_type"] == "FT"].copy()
    df = df.dropna(
        subset=[
            "salary_in_usd",
            "remote_ratio",
            "experience_level",
            "company_size",
            "company_location",
            "work_year",
            "job_title",
        ]
    )
    remote_map = {0: "On-site", 50: "Hybrid", 100: "Remote"}
    exp_map = {"EN": "Entry", "MI": "Mid", "SE": "Senior", "EX": "Executive"}
    df["remote_category"] = df["remote_ratio"].map(remote_map)
    df["experience_label"] = df["experience_level"].map(exp_map)
    df = df[df["remote_category"].notna()].copy()
    df["remote_category"] = pd.Categorical(
        df["remote_category"],
        categories=["On-site", "Hybrid", "Remote"],
        ordered=True,
    )
    df["log_salary"] = np.log(df["salary_in_usd"])
    return df


def mean_ci(values: pd.Series) -> tuple[float, float]:
    values = values.dropna()
    mean = values.mean()
    se = values.std(ddof=1) / math.sqrt(len(values))
    return mean, 1.96 * se


def plot_box_with_means(df: pd.DataFrame) -> None:
    order = ["On-site", "Hybrid", "Remote"]
    data = [df.loc[df["remote_category"] == cat, "salary_in_usd"] for cat in order]
    means = [d.mean() for d in data]
    counts = [len(d) for d in data]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bp = ax.boxplot(data, tick_labels=order, showfliers=False, patch_artist=True)
    colors = ["#6d8fb3", "#c8a45d", "#8a9f73"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    ax.scatter(range(1, len(order) + 1), means, marker="D", s=48, color="#8b1e3f", label="Mean", zorder=4)
    for i, (series, mean, n) in enumerate(zip(data, means, counts), start=1):
        q1, q3 = np.percentile(series.dropna(), [25, 75])
        label_y = q3 - 0.12 * (q3 - q1)
        ax.text(
            i,
            label_y,
            f"Mean ${mean:,.0f}\nN={n:,}",
            ha="center",
            va="top",
            fontsize=7,
            linespacing=0.95,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 1.2},
            zorder=5,
        )

    ax.set_title("Salary Distribution by Work Category")
    ax.set_xlabel("Work Category")
    ax.set_ylabel("Salary in USD")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "salary_distribution_with_means.png", dpi=300)
    plt.close(fig)


def exact_matched_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df["remote_ratio"].isin([0, 100])].copy()
    base["treat_remote"] = (base["remote_ratio"] == 100).astype(int)

    rows = []
    for label, keys in [
        ("Experience + location + company size", ["experience_level", "company_location", "company_size"]),
        (
            "Experience + location + company size + work year",
            ["experience_level", "company_location", "company_size", "work_year"],
        ),
        (
            "Experience + location + company size + job title",
            ["experience_level", "company_location", "company_size", "job_title"],
        ),
    ]:
        counts = base.groupby(keys)["treat_remote"].agg(["sum", "count"]).reset_index()
        counts["onsite_count"] = counts["count"] - counts["sum"]
        valid = counts[(counts["sum"] > 0) & (counts["onsite_count"] > 0)][keys]
        matched = base.merge(valid, on=keys, how="inner")
        for treat, name in [(0, "On-site"), (1, "Remote")]:
            sub = matched[matched["treat_remote"] == treat]["salary_in_usd"]
            mean, ci = mean_ci(sub)
            rows.append(
                {
                    "matching_level": label,
                    "work_category": name,
                    "count": len(sub),
                    "mean_salary": mean,
                    "ci95": ci,
                    "median_salary": sub.median(),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "matched_salary_comparisons.csv", index=False)
    return result


def plot_matched_comparison(match_df: pd.DataFrame) -> None:
    focus = match_df[match_df["matching_level"] == "Experience + location + company size + job title"].copy()
    order = ["On-site", "Remote"]
    focus["work_category"] = pd.Categorical(focus["work_category"], categories=order, ordered=True)
    focus = focus.sort_values("work_category")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(focus["work_category"], focus["mean_salary"], yerr=focus["ci95"], capsize=6, color=["#6d8fb3", "#8a9f73"])
    for i, row in enumerate(focus.itertuples(), start=0):
        ax.text(
            i,
            row.mean_salary - 13500,
            f"${row.mean_salary:,.0f}\nN={row.count:,}",
            ha="center",
            va="top",
            fontsize=10,
            color="white",
            fontweight="bold",
        )
    ax.set_title("Matched Remote vs. On-site Salary", pad=12)
    ax.set_xlabel("Work Category")
    ax.set_ylabel("Average salary in USD")
    ax.set_ylim(0, max(focus["mean_salary"] + focus["ci95"]) * 1.18)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "matched_remote_onsite_salary_clean.png", dpi=300)
    plt.close(fig)


def ols_log_salary(df: pd.DataFrame) -> pd.DataFrame:
    model_df = df.copy()
    x = pd.get_dummies(
        model_df[["remote_category", "experience_level", "company_size", "work_year", "company_location"]],
        columns=["remote_category", "experience_level", "company_size", "work_year", "company_location"],
        drop_first=True,
        dtype=float,
    )
    x.insert(0, "Intercept", 1.0)
    y = model_df["log_salary"].to_numpy(dtype=float)
    X = x.to_numpy(dtype=float)

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    sigma2 = (resid @ resid) / (n - k)
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * xtx_inv))

    out = pd.DataFrame({"term": x.columns, "coef": beta, "se": se})
    out["ci_low"] = out["coef"] - 1.96 * out["se"]
    out["ci_high"] = out["coef"] + 1.96 * out["se"]
    out["percent_diff"] = (np.exp(out["coef"]) - 1) * 100
    out["percent_low"] = (np.exp(out["ci_low"]) - 1) * 100
    out["percent_high"] = (np.exp(out["ci_high"]) - 1) * 100
    out.to_csv(OUT / "ols_log_salary_coefficients.csv", index=False)
    return out


def plot_remote_coefficients(coefs: pd.DataFrame) -> None:
    keep = coefs[coefs["term"].isin(["remote_category_Hybrid", "remote_category_Remote"])].copy()
    keep["label"] = keep["term"].map(
        {"remote_category_Hybrid": "Hybrid vs on-site", "remote_category_Remote": "Remote vs on-site"}
    )
    keep = keep.set_index("label").loc[["Hybrid vs on-site", "Remote vs on-site"]].reset_index()
    y = np.arange(len(keep))
    xerr = np.vstack([keep["percent_diff"] - keep["percent_low"], keep["percent_high"] - keep["percent_diff"]])

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.errorbar(keep["percent_diff"], y, xerr=xerr, fmt="o", color="#8b1e3f", capsize=5)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y, keep["label"])
    ax.set_xlabel("Estimated salary difference after controls (%)")
    ax.set_title("Adjusted Association Between Work Category and Salary")
    ax.grid(axis="x", alpha=0.25)
    for row in keep.itertuples():
        ax.text(row.percent_diff, row.Index + 0.16, f"{row.percent_diff:.1f}%", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "adjusted_remote_salary_coefficients.png", dpi=300)
    plt.close(fig)


def write_latex_summary(match_df: pd.DataFrame, coefs: pd.DataFrame) -> None:
    focus = match_df[match_df["matching_level"] == "Experience + location + company size + job title"].copy()
    remote = coefs[coefs["term"] == "remote_category_Remote"].iloc[0]
    hybrid = coefs[coefs["term"] == "remote_category_Hybrid"].iloc[0]
    onsite = focus[focus["work_category"] == "On-site"].iloc[0]
    rem = focus[focus["work_category"] == "Remote"].iloc[0]
    text = f"""% Auto-generated by scripts/extended_analysis.py
\\begin{{table}}[H]
\\centering
\\caption{{Additional adjusted and matched comparisons}}
\\begin{{tabular}}{{llr}}
\\toprule
Comparison & Estimate & Interpretation \\\\
\\midrule
OLS, remote vs. on-site & {remote.percent_diff:.1f}\\% & Controls: experience, company size, work year, company location \\\\
OLS, hybrid vs. on-site & {hybrid.percent_diff:.1f}\\% & Hybrid estimate is less stable because N is small \\\\
Matched on experience, location, size, and job title & Remote mean: \\${rem.mean_salary:,.0f} & N={int(rem['count']):,} \\\\
Matched on experience, location, size, and job title & On-site mean: \\${onsite.mean_salary:,.0f} & N={int(onsite['count']):,} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    (OUT / "additional_analysis_table.tex").write_text(text)


def main() -> None:
    df = clean_data()
    plot_box_with_means(df)
    match_df = exact_matched_comparisons(df)
    plot_matched_comparison(match_df)
    coefs = ols_log_salary(df)
    plot_remote_coefficients(coefs)
    write_latex_summary(match_df, coefs)
    print("Cleaned rows:", len(df))
    print(match_df)
    print(coefs[coefs["term"].isin(["remote_category_Hybrid", "remote_category_Remote"])][
        ["term", "coef", "se", "percent_diff", "percent_low", "percent_high"]
    ])


if __name__ == "__main__":
    main()
