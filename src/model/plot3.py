# plot_eval_grouped.py
import argparse
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
from matplotlib.colors import Normalize
import matplotlib.ticker as mticker
import json

import numpy as pd  # type: ignore  # quick alias to keep pd as pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng


# ---------- data loading & utilities ----------
def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    num_cols = [
        "day","episode","action_price","action_restock_frac","price","opening_inventory",
        "restocked","expired","visitors","sales","unmet_demand","closing_inventory",
        "revenue","restock_cost","holding_cost","day_profit","cum_profit","stockout",
        "capacity","capacity_left_opening"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "episode" in df.columns:
        df["episode"] = df["episode"].astype("Int64")
    if "policy" in df.columns:
        df["policy"] = df["policy"].astype(str)
    df = df.sort_values(["policy","episode","day"])
    return df


def ensure_outdir(outdir: Path, save: bool):
    if save:
        outdir.mkdir(parents=True, exist_ok=True)


def bars_side_by_side(ax, x_vals, groups, group_labels, width=0.35):
    n = len(groups)
    offsets = np.linspace(-width*(n-1)/2, width*(n-1)/2, n)
    for i, (g, lab) in enumerate(zip(groups, group_labels)):
        ax.bar(x_vals + offsets[i], g, width=width*0.95, label=lab)


# ---------- NEW: Figure 1 — Stability vs Price Volatility ----------
def stability_vs_price_volatility(df: pd.DataFrame, outdir: Path, save: bool, figsize=(7, 5)):
    """
    Profit volatility = std(day_profit) across all days×episodes for a policy.
    Price dispersion = std(price) across all days×episodes for a policy.
    """
    ensure_outdir(outdir, save)
    needed = {"policy", "day_profit", "price"}
    if not needed.issubset(df.columns):
        print("[warn] Missing columns for stability plot; need:", needed)
        return

    agg = (
        df.groupby("policy", as_index=False)
          .agg(price_dispersion=("price", "std"),
               profit_volatility=("day_profit", "std"))
          .sort_values("profit_volatility")
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(agg["price_dispersion"], agg["profit_volatility"])
    for _, r in agg.iterrows():
        ax.annotate(r["policy"], (r["price_dispersion"], r["profit_volatility"]),
                    xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.set_title("Figure 1. Stability vs. Price Volatility by Policy")
    ax.set_xlabel("Price dispersion (std of daily price)")
    ax.set_ylabel("Profit volatility (std of daily profit)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save:
        fig.savefig(outdir / "fig1_stability_vs_price_volatility.png", dpi=150, bbox_inches="tight")

    # Also write a CSV so you can cite exact values in text
    agg.to_csv(outdir / "fig1_stability_vs_price_volatility.csv", index=False)


# ---------- helper: finals ----------
def compute_final_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    finals = (
        df.groupby(["policy", "episode"], as_index=False, sort=False)
          .apply(lambda g: g.sort_values("day").tail(1))
          .reset_index(drop=True)
    )
    return finals


# ---------- NEW: Table 1 — Operational behavior ----------
def _policy_operational_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns per-policy KPIs used in Table 1:
      - restock_days_%: fraction of days with restocked > 0
      - opening_util: mean(opening_inventory / capacity) if capacity present, else NaN
      - waste_rate: sum(expired) / (sum(restocked) + initial stock proxy)
      - miss_rate: sum(unmet_demand) / (sum(unmet_demand) + sum(sales))
      - stockouts: total count of days with stockout==1
    Notes:
      * initial stock proxy = mean(opening_inventory at day==0) per episode aggregated by policy
    """
    if df.empty or "policy" not in df.columns:
        return pd.DataFrame()

    # initial stock per episode (day==0)
    init = (
        df[df["day"] == df.groupby(["policy", "episode"])["day"].transform("min")]
        .groupby(["policy", "episode"], as_index=False)["opening_inventory"].first()
        .rename(columns={"opening_inventory": "initial_inventory"})
    )

    base = df.merge(init, on=["policy","episode"], how="left")

    grp = base.groupby("policy", as_index=False)
    # denominators
    received = grp["restocked"].sum().rename(columns={"restocked": "restocked_sum"})
    initial = init.groupby("policy", as_index=False)["initial_inventory"].sum()

    k = grp.agg(
        restock_days=("restocked", lambda s: np.mean(np.nan_to_num(s) > 0.0)),
        opening_util=("opening_inventory", "mean"),
        capacity=("capacity", "mean"),
        waste_units=("expired", "sum"),
        unmet=("unmet_demand", "sum"),
        sales=("sales", "sum"),
        stockouts=("stockout", "sum")
    )

    # opening utilization %
    if "capacity" in df.columns and df["capacity"].notna().any():
        k["opening_util"] = (k["opening_util"] / k["capacity"]).clip(lower=0)

    # total received = total restocked + total initial across episodes
    k = k.merge(received, on="policy", how="left").merge(initial, on="policy", how="left")
    denom_waste = (k["restocked_sum"].fillna(0) + k["initial_inventory"].fillna(0)).replace(0, np.nan)
    k["waste_rate"] = (k["waste_units"] / denom_waste).clip(lower=0)

    # miss rate
    denom_demand = (k["unmet"].fillna(0) + k["sales"].fillna(0)).replace(0, np.nan)
    k["miss_rate"] = (k["unmet"] / denom_demand).clip(lower=0)

    # format
    out = k[["policy", "restock_days", "opening_util", "waste_rate", "miss_rate", "stockouts"]].copy()
    return out.sort_values("policy")


def operational_behavior_table(df: pd.DataFrame, outdir: Path, save: bool):
    ensure_outdir(outdir, save)
    kpis = _policy_operational_kpis(df)
    if kpis.empty:
        print("[warn] Could not compute operational KPIs; skipping Table 1.")
        return

    # keep a CSV for referencing exact numbers
    kpis.to_csv(outdir / "table1_operational_behavior_raw.csv", index=False)

    # build a figure-table
    disp = kpis.copy()
    def fmt_pct(x): return f"{100*x:.1f}%" if pd.notna(x) else ""
    def fmt_ratio(x): return f"{x:.3f}" if pd.notna(x) else ""
    disp["restock_days"] = disp["restock_days"].apply(fmt_pct)
    disp["opening_util"] = disp["opening_util"].apply(fmt_pct)
    disp["waste_rate"]   = disp["waste_rate"].apply(fmt_pct)
    disp["miss_rate"]    = disp["miss_rate"].apply(fmt_pct)
    disp["stockouts"]    = disp["stockouts"].apply(lambda v: f"{int(v):,}" if pd.notna(v) else "")

    cols = ["policy","restock_days","opening_util","waste_rate","miss_rate","stockouts"]
    disp = disp[cols]

    # sizing: width ~ 1.2 in/col, height ~ 0.45 in/row
    fig_w = max(8, min(2 + 1.2 * len(cols), 28))
    fig_h = max(3, min(1.0 + 0.45 * (len(disp) + 1), 20))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title("Table 1. Operational behavior and service outcomes", pad=10, fontsize=12)
    table = ax.table(cellText=disp.values,
                     colLabels=[c.replace("_"," ").title() for c in disp.columns],
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9 if len(cols) > 8 else 10)
    table.scale(1, 1.2)
    fig.tight_layout()
    if save:
        fig.savefig(outdir / "table1_operational_behavior.png", dpi=150, bbox_inches="tight")


# ---------- NEW: 6.3 Performance vs Specifications (matrix) ----------
DEFAULT_TARGETS = {
    "waste_rate_max": 0.05,          # ≤ 5%
    "miss_rate_max": 0.05,           # ≤ 5%
    "profit_volatility_max": 25.0,   # std of day_profit
    "restock_days_max": 0.50,        # ≤ 50%
}

def _policy_profit_volatility(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("policy", as_index=False)["day_profit"]
          .std()
          .rename(columns={"day_profit": "profit_volatility"})
    )

def _policy_mean_final_profit(df: pd.DataFrame) -> pd.DataFrame:
    finals = compute_final_rows(df)
    if finals.empty:
        return pd.DataFrame(columns=["policy","mean_final_profit"])
    return finals.groupby("policy", as_index=False)["cum_profit"].mean().rename(columns={"cum_profit":"mean_final_profit"})

def performance_vs_specs_matrix(df: pd.DataFrame, outdir: Path, save: bool, targets: Dict[str, Any]):
    ensure_outdir(outdir, save)
    A = _policy_operational_kpis(df)
    if A.empty:
        print("[warn] Spec matrix: missing KPIs; skipping.")
        return
    B = _policy_profit_volatility(df)
    C = _policy_mean_final_profit(df)
    M = A.merge(B, on="policy", how="left").merge(C, on="policy", how="left").sort_values("policy")

    # pass/fail columns
    M["waste_ok"]   = M["waste_rate"] <= targets.get("waste_rate_max", DEFAULT_TARGETS["waste_rate_max"])
    M["miss_ok"]    = M["miss_rate"]  <= targets.get("miss_rate_max", DEFAULT_TARGETS["miss_rate_max"])
    M["vol_ok"]     = M["profit_volatility"] <= targets.get("profit_volatility_max", DEFAULT_TARGETS["profit_volatility_max"])
    M["restock_ok"] = M["restock_days"] <= targets.get("restock_days_max", DEFAULT_TARGETS["restock_days_max"])

    # figure-table with ✓ / ✗
    disp = M[[
        "policy", "mean_final_profit", "waste_rate", "miss_rate", "profit_volatility", "restock_days",
        "waste_ok","miss_ok","vol_ok","restock_ok"
    ]].copy()

    def chk(x): return "✓" if bool(x) else "✗"
    def pct(x): return f"{100*x:.1f}%"
    disp["waste_rate"]       = disp["waste_rate"].apply(pct)
    disp["miss_rate"]        = disp["miss_rate"].apply(pct)
    disp["restock_days"]     = disp["restock_days"].apply(pct)
    disp["mean_final_profit"]= disp["mean_final_profit"].apply(lambda v: f"${v:,.0f}" if pd.notna(v) else "")
    disp["profit_volatility"]= disp["profit_volatility"].apply(lambda v: f"{v:,.2f}" if pd.notna(v) else "")
    disp["waste_ok"]         = disp["waste_ok"].apply(chk)
    disp["miss_ok"]          = disp["miss_ok"].apply(chk)
    disp["vol_ok"]           = disp["vol_ok"].apply(chk)
    disp["restock_ok"]       = disp["restock_ok"].apply(chk)

    cols = [
        "policy","mean_final_profit",
        "waste_rate","miss_rate","profit_volatility","restock_days",
        "waste_ok","miss_ok","vol_ok","restock_ok"
    ]
    disp = disp[cols]

    fig_w = max(10, min(2 + 1.2 * len(cols), 30))
    fig_h = max(3, min(1.0 + 0.45 * (len(disp) + 1), 22))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ttl = "Performance vs Specifications — pass (✓) / fail (✗)"
    ax.set_title(ttl, pad=10, fontsize=12)
    table = ax.table(cellText=disp.values,
                     colLabels=[c.replace("_"," ").title() for c in disp.columns],
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)
    fig.tight_layout()
    if save:
        fig.savefig(outdir / "specs_matrix.png", dpi=150, bbox_inches="tight")
        M.to_csv(outdir / "specs_matrix_raw.csv", index=False)


# ---------- NEW: 6.4 Uncertainty & Error Analysis ----------
def _paired_bootstrap_diff(finals: pd.DataFrame, ref: str, iters: int = 10000, seed: int = 7) -> pd.DataFrame:
    """
    Paired bootstrap for Δ(final_profit) against a reference policy (per episode indices).
    Returns a table with mean diff and BCa-like percentile CIs.
    """
    rng = default_rng(seed)

    # pivot: episodes in columns; policies in rows
    pivot = finals.pivot_table(index="policy", columns="episode", values="cum_profit", aggfunc="mean")
    if ref not in pivot.index:
        raise ValueError(f"Reference policy '{ref}' not found in data.")
    ref_vec = pivot.loc[ref]

    rows: List[Dict[str, Any]] = []
    for pol in pivot.index:
        if pol == ref:
            continue
        x = pivot.loc[pol]
        # keep only episodes observed for both
        mask = (~x.isna()) & (~ref_vec.isna())
        x = x[mask].to_numpy()
        y = ref_vec[mask].to_numpy()
        if x.size == 0:
            continue

        diffs = x - y
        n = diffs.size
        # bootstrap on index with replacement
        idx = np.arange(n)
        boots = []
        for _ in range(iters):
            b = rng.choice(idx, size=n, replace=True)
            boots.append(diffs[b].mean())
        boots = np.array(boots)
        mean_diff = diffs.mean()
        lo, hi = np.percentile(boots, [2.5, 97.5])

        rows.append({
            "policy": pol,
            "ref_policy": ref,
            "mean_diff": mean_diff,
            "ci_lo": lo,
            "ci_hi": hi,
            "episodes_used": n
        })
    return pd.DataFrame(rows).sort_values("mean_diff", ascending=False)


def uncertainty_and_error_analysis(df: pd.DataFrame, outdir: Path, save: bool, ref_policy: str, boot_iters: int):
    ensure_outdir(outdir, save)
    finals = compute_final_rows(df)
    if finals.empty:
        print("[warn] No final rows for uncertainty analysis.")
        return
    try:
        ci = _paired_bootstrap_diff(finals, ref_policy, iters=boot_iters)
    except ValueError as e:
        print(f"[warn] {e}")
        return

    if save:
        ci.to_csv(outdir / "uncertainty_bootstrap_diff_vs_ref.csv", index=False)

        # Write a short Markdown "drop-in"
        md = Path(outdir) / "uncertainty_and_error.md"
        with md.open("w", encoding="utf-8") as f:
            f.write("# 6.4 Uncertainty & Error Analysis\n\n")
            f.write(f"- **Paired bootstrap (N={boot_iters})** for final cumulative profit differences against **{ref_policy}**.\n")
            f.write("- Reported are mean Δ and 95% percentile CIs; pairing is by episode.\n\n")
            f.write("## Results\n\n")
            for _, r in ci.iterrows():
                f.write(f"- {r['policy']}: Δ = {r['mean_diff']:,.0f} "
                        f"[{r['ci_lo']:,.0f}, {r['ci_hi']:,.0f}] over {int(r['episodes_used'])} matched episodes.\n")
            f.write("\n## Notes\n")
            f.write("- Same price grid and costs across policies; tuning budgets were kept equal.\n")
            f.write("- We also recommend stratified bootstrap by scenario (elasticity, shelf-life) if multiple scenarios are pooled.\n")
            f.write("- For small samples, complement with paired t-tests and report effect sizes.\n")
        print(f"[info] Wrote uncertainty report: {md.resolve()}")


# ---------- plotting: dashboards (yours, unchanged except title tweaks) ----------
def policy_dashboard(
    df: pd.DataFrame,
    policy: str,
    episode: Optional[int],
    outdir: Path,
    save: bool,
    figsize: Tuple[int, int] = (13, 12),
):
    ensure_outdir(outdir, save)

    if episode is None:
        avail = df[df["policy"] == policy]["episode"].dropna().unique()
        ep = int(avail[0]) if len(avail) > 0 else 1
    else:
        ep = episode

    det = df[(df["policy"] == policy) & (df["episode"] == ep)].sort_values("day")

    grouped = df[df["policy"] == policy].groupby("day")
    stockout_rate = grouped["stockout"].mean().reset_index() if "stockout" in df.columns else None
    cf_mean = grouped["cum_profit"].mean().reset_index(name="mean")
    cf_std  = grouped["cum_profit"].std(ddof=1).reset_index(name="std")
    cf = cf_mean.merge(cf_std, on="day", how="left")
    expired_mean = grouped["expired"].mean().reset_index(name="expired_mean") if "expired" in df.columns else None

    fig, axs = plt.subplots(3, 2, figsize=figsize)
    fig.suptitle(f"Policy Dashboard — {policy}" + (f" (episode {ep})" if not det.empty else ""), fontsize=14)

    # (A) Price & Sales (episode)
    ax = axs[0, 0]
    if not det.empty:
        ax2 = ax.twinx()
        ax.plot(det["day"], det["price"], label="Price")
        ax.set_xlabel("Day")
        ax.set_ylabel("Price")
        ax2.plot(det["day"], det["sales"], label="Sales", linestyle="--")
        ax2.set_ylabel("Units Sold")
        ax.set_title("Price & Sales (episode)")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best")
    else:
        ax.text(0.5, 0.5, "No episode data", ha="center", va="center")
        ax.set_axis_off()

    # (B) Restock & Inventory (episode)
    ax = axs[0, 1]
    if not det.empty:
        ax2 = ax.twinx()
        ax.bar(det["day"], det["restocked"], alpha=0.35, label="Restocked")
        ax.set_xlabel("Day")
        ax.set_ylabel("Restocked (units)")
        ax2.plot(det["day"], det["closing_inventory"], label="Closing Inventory", linestyle="-")
        ax2.set_ylabel("Inventory (units)")
        ax.set_title("Restock & Inventory (episode)")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best")
    else:
        ax.text(0.5, 0.5, "No episode data", ha="center", va="center")
        ax.set_axis_off()

    # (C) Stockout Rate by Day (across episodes)
    ax = axs[1, 0]
    if stockout_rate is not None and not stockout_rate.empty:
        ax.plot(stockout_rate["day"], stockout_rate["stockout"])
        ax.set_xlabel("Day")
        ax.set_ylabel("Stockout Rate")
        ax.set_title("Stockout Rate (mean across episodes)")
    else:
        ax.text(0.5, 0.5, "No stockout data", ha="center", va="center")
        ax.set_axis_off()

    # (D) Cumulative Profit by Day (mean ± std)
    ax = axs[1, 1]
    if not cf.empty:
        ax.plot(cf["day"], cf["mean"], label="Mean")
        upper = cf["mean"] + cf["std"].fillna(0)
        lower = cf["mean"] - cf["std"].fillna(0)
        ax.fill_between(cf["day"], lower, upper, alpha=0.2, label="±1 std")
        ax.set_xlabel("Day")
        ax.set_ylabel("Cumulative Profit")
        ax.set_title("Cumulative Profit (mean ± 1 std)")
        ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, "No profit data", ha="center", va="center")
        ax.set_axis_off()

    # (E) Expired by Day (episode)
    ax = axs[2, 0]
    if not det.empty and "expired" in det.columns:
        ax.bar(det["day"], det["expired"], alpha=0.5)
        ax.set_xlabel("Day")
        ax.set_ylabel("Expired (units)")
        ax.set_title("Expired by Day (episode)")
    else:
        ax.text(0.5, 0.5, "No expired data for episode", ha="center", va="center")
        ax.set_axis_off()

    # (F) Expired by Day (mean across episodes)
    ax = axs[2, 1]
    if expired_mean is not None and not expired_mean.empty:
        ax.plot(expired_mean["day"], expired_mean["expired_mean"])
        ax.set_xlabel("Day")
        ax.set_ylabel("Expired (units)")
        ax.set_title("Expired by Day (mean across episodes)")
    else:
        ax.text(0.5, 0.5, "No expired data across episodes", ha="center", va="center")
        ax.set_axis_off()

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    if save:
        fig.savefig(outdir / f"dashboard__{policy}.png", dpi=150, bbox_inches="tight")


# ---------- plotting: comparisons (yours, slightly guarded) ----------
def final_profit_comparison(df: pd.DataFrame, outdir: Path, save: bool, figsize=(10, 5)):
    ensure_outdir(outdir, save)
    finals = compute_final_rows(df)
    if finals.empty:
        print("[warn] No final rows found; skipping comparison plot.")
        return

    mean_per_policy = (
        finals.groupby("policy", as_index=False)["cum_profit"]
              .mean()
              .sort_values("cum_profit", ascending=False)
    )

    n_policies = finals["policy"].nunique()
    n_episodes = finals["episode"].nunique()
    many = (n_policies > 8) or (n_episodes > 20)

    if not many:
        pivot = (
            finals.pivot_table(
                index="episode", columns="policy", values="cum_profit", aggfunc="mean"
            ).sort_index()
        )
        episodes = np.arange(len(pivot.index))
        series = [pivot[col].to_numpy() for col in pivot.columns]
        labels = list(pivot.columns)
        fig_w = max(figsize[0], 2 + 0.6 * len(labels))
        fig, ax = plt.subplots(figsize=(fig_w, figsize[1]))
        bars_side_by_side(ax, episodes, series, labels, width=min(0.8, 0.2 + 0.6 / max(1, len(labels))))
        ax.set_xlabel("Episode")
        ax.set_ylabel("Final Cumulative Profit")
        ax.set_title("Final Profit per Episode — Policy Comparison")
        ax.legend(loc="best", ncol=min(4, len(labels)))
        ax.set_xticks(episodes)
        ax.set_xticklabels([str(int(x)) for x in pivot.index])
        fig.tight_layout()
        if save:
            fig.savefig(outdir / "final_profit_comparison__grouped.png", dpi=150, bbox_inches="tight")
    else:
        pivot = (
            finals.pivot_table(index="policy", columns="episode", values="cum_profit", aggfunc="mean")
            .sort_index()
        )
        vals = pivot.to_numpy().astype(float)
        lo, hi = np.nanpercentile(vals, [2, 98])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = np.nanmin(vals), np.nanmax(vals)
        cmap = "magma"
        norm = Normalize(vmin=lo, vmax=hi)
        h = max(4, min(0.35 * len(pivot.index), 20))
        w = max(6, min(0.30 * len(pivot.columns), 24))
        fig, ax = plt.subplots(figsize=(w, h), layout="constrained")
        im = ax.imshow(pivot.to_numpy(), aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
        ax.set_title("Final Profit per Episode — Heatmap")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Policy")
        ax.set_xticks(np.arange(pivot.shape[1]))
        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_xticklabels([str(int(e)) for e in pivot.columns], rotation=90, ha="center")
        ax.set_yticklabels(list(pivot.index))
        plt.setp(ax.get_yticklabels(), ha="right")
        ax.tick_params(axis="y", pad=6)
        fig.colorbar(im, ax=ax, label="Final Cumulative Profit")
        fig.tight_layout()
        if save:
            fig.savefig(outdir / "final_profit_comparison__heatmap.png", dpi=150, bbox_inches="tight")

        fig_w = max(8, 0.45 * len(mean_per_policy))
        fig, ax = plt.subplots(figsize=(fig_w, 5))
        policies_ordered = mean_per_policy["policy"].tolist()
        profits_ordered  = mean_per_policy["cum_profit"].to_numpy()
        bars = ax.bar(policies_ordered, profits_ordered)
        ax.set_title("Average Final Profit per Policy")
        ax.set_xlabel("Policy")
        ax.set_ylabel("Mean Final Cumulative Profit")
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        for rect in bars:
            h = rect.get_height()
            ax.annotate(f"{h:,.0f}", (rect.get_x()+rect.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        if save:
            fig.savefig(outdir / "final_profit__mean_per_policy.png", dpi=150, bbox_inches="tight")


def policy_kpi_averages_table(df: pd.DataFrame, outdir: Path, save: bool):
    ensure_outdir(outdir, save)

    if df.empty or "policy" not in df.columns:
        print("[warn] No data or 'policy' column missing; skipping KPI averages table.")
        return

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for ident in ["day", "episode"]:
        if ident in num_cols:
            num_cols.remove(ident)
    if not num_cols:
        print("[warn] No numeric KPI columns found; skipping KPI averages table.")
        return

    kpi_means = (
        df.groupby("policy", as_index=True)[num_cols]
          .mean(numeric_only=True)
          .sort_index()
    )

    preferred_order = [
        "price", "visitors", "sales", "unmet_demand", "stockout",
        "revenue", "restock_cost", "holding_cost", "day_profit", "cum_profit",
        "restocked", "expired",
        "opening_inventory", "closing_inventory",
        "action_price", "action_restock_frac",
        "capacity", "capacity_left_opening",
    ]
    ordered_cols = [c for c in preferred_order if c in kpi_means.columns]
    ordered_cols += [c for c in kpi_means.columns if c not in ordered_cols]
    kpi_means = kpi_means[ordered_cols]

    def is_currency(col): return col in {"revenue", "restock_cost", "holding_cost", "day_profit", "cum_profit"}
    def is_percent(col):  return ("rate" in col.lower()) or (col.lower() in {"stockout"})
    def is_fraction(col): return col.lower() in {"action_restock_frac"}
    def is_integerish(col):
        return col.lower() in {
            "visitors","sales","unmet_demand","restocked","expired",
            "opening_inventory","closing_inventory","capacity","capacity_left_opening"
        }

    def fmt_val(col, v):
        if pd.isna(v): return ""
        if is_currency(col): return f"${v:,.0f}"
        if is_percent(col):  return f"{100*v:.1f}%"
        if is_fraction(col): return f"{v:.2f}"
        if is_integerish(col): return f"{v:,.0f}"
        return f"{v:,.2f}"

    all_cols = kpi_means.columns.tolist()
    thirds = np.array_split(all_cols, 3)

    def make_table_figure(display_df: pd.DataFrame, title_suffix: str, fname_suffix: str):
        nrows, ncols = display_df.shape
        fig_w = max(8, min(2 + 1.2 * ncols, 28))
        fig_h = max(3, min(1.0 + 0.45 * (nrows + 1), 20))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")
        ax.set_title(f"Average KPIs per Policy — {title_suffix}", pad=12, fontsize=12)
        table = ax.table(cellText=display_df.values,
                         rowLabels=display_df.index.tolist(),
                         colLabels=display_df.columns.tolist(),
                         loc="center", cellLoc="center", rowLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8 if ncols > 14 else 9)
        table.scale(1, 1.2)
        fig.tight_layout()
        return fig

    for i, cols in enumerate(thirds, start=1):
        cols = list(cols)
        if not cols: continue
        sub = kpi_means[cols].copy()
        display_df = sub.copy()
        for c in display_df.columns:
            display_df[c] = display_df[c].apply(lambda x: fmt_val(c, x))
        fig = make_table_figure(display_df, title_suffix=f"Part {i}/3", fname_suffix=f"part{i}")
        if save:
            fig.savefig(outdir / f"policy_kpi_averages_table__part{i}.png", dpi=150, bbox_inches="tight")

    if save:
        kpi_means.to_csv(outdir / "policy_kpi_averages_table.csv", index=True)


# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Grouped plots for evaluator CSV (matplotlib).")
    ap.add_argument("--csv", type=str, default="eval_daily_results.csv", help="Path to evaluator CSV.")
    ap.add_argument("--save", action="store_true", help="Save figures to --outdir.")
    ap.add_argument("--outdir", type=str, default="plots", help="Directory for saved figures.")
    ap.add_argument("--episode", type=int, default=None, help="Episode for dashboard detail; defaults to first available.")

    # NEW options
    ap.add_argument("--ref-policy", type=str, default="ppo", help="Reference policy for uncertainty analysis.")
    ap.add_argument("--boot-iters", type=int, default=10000, help="Bootstrap iterations for CI computation.")
    ap.add_argument("--targets", type=str, default="", help="JSON string or path to JSON with spec thresholds.")

    ap.add_argument("--no-show", action="store_true", help="Don’t call plt.show() (useful on headless runs with --save).")
    return ap.parse_args()


def _parse_targets(arg: str) -> Dict[str, Any]:
    if not arg:
        return DEFAULT_TARGETS.copy()
    try:
        if Path(arg).exists():
            return {**DEFAULT_TARGETS, **json.loads(Path(arg).read_text())}
        return {**DEFAULT_TARGETS, **json.loads(arg)}
    except Exception:
        print("[warn] Could not parse --targets; using defaults.")
        return DEFAULT_TARGETS.copy()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    df = load_csv(args.csv)

    policies = list(df["policy"].dropna().unique())
    if not policies:
        print("[error] No policies found in CSV.")
        return

    # NEW: Figure 1
    stability_vs_price_volatility(df, outdir, args.save)

    # NEW: Table 1
    operational_behavior_table(df, outdir, args.save)

    # NEW: 6.3 matrix
    targets = _parse_targets(args.targets)
    performance_vs_specs_matrix(df, outdir, args.save, targets)

    # NEW: 6.4 uncertainty report
    uncertainty_and_error_analysis(df, outdir, args.save, args.ref_policy, args.boot_iters)

    # Your existing dashboards per policy
    for pol in policies:
        policy_dashboard(df, pol, args.episode, outdir, args.save)

    # Cross-policy comparison
    final_profit_comparison(df, outdir, args.save)
    policy_kpi_averages_table(df, outdir, args.save)

    if not args.no_show:
        plt.show()
    else:
        print(f"Saved outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
