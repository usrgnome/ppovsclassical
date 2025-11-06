# plot_eval_grouped.py
import argparse
from pathlib import Path
from typing import Tuple, Optional
from matplotlib.colors import Normalize, TwoSlopeNorm, BoundaryNorm
import matplotlib.ticker as mticker

import numpy as pd  # type: ignore  # quick alias to keep pd as pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
    df = df.sort_values(["policy","episode","day"])
    return df


def ensure_outdir(outdir: Path, save: bool):
    if save:
        outdir.mkdir(parents=True, exist_ok=True)


def bars_side_by_side(ax, x_vals, groups, group_labels, width=0.35):
    """
    Draw grouped bars for each x across multiple series.
    groups: list of arrays (one per label) same length as x_vals
    """
    n = len(groups)
    offsets = np.linspace(-width*(n-1)/2, width*(n-1)/2, n)
    for i, (g, lab) in enumerate(zip(groups, group_labels)):
        ax.bar(x_vals + offsets[i], g, width=width*0.95, label=lab)


# ---------- plotting: dashboards ----------
def policy_dashboard(
    df: pd.DataFrame,
    policy: str,
    episode: Optional[int],
    outdir: Path,
    save: bool,
    figsize: Tuple[int, int] = (13, 9),
):
    """
    One figure per policy:
      (A) Price & Sales (episode detail; twin axis)
      (B) Restock & Closing Inventory (episode detail; twin axis)
      (C) Stockout Rate by Day (mean across episodes)
      (D) Cumulative Profit by Day (mean ± 1 std across episodes)
    """
    ensure_outdir(outdir, save)

    # episode detail data (falls back to first episode if not provided)
    if episode is None:
        avail = df[df["policy"] == policy]["episode"].dropna().unique()
        ep = int(avail[0]) if len(avail) > 0 else 1
    else:
        ep = episode

    det = df[(df["policy"] == policy) & (df["episode"] == ep)].sort_values("day")
    if det.empty:
        print(f"[warn] No rows for policy='{policy}', episode={ep}; skipping detail panels.")

    # aggregates
    grouped = df[df["policy"] == policy].groupby("day")
    stockout_rate = grouped["stockout"].mean().reset_index() if "stockout" in df.columns else None
    cf_mean = grouped["cum_profit"].mean().reset_index(name="mean")
    cf_std  = grouped["cum_profit"].std(ddof=1).reset_index(name="std")
    cf = cf_mean.merge(cf_std, on="day", how="left")

    # figure layout
    fig, axs = plt.subplots(2, 2, figsize=figsize)
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
        # combined legend
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

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    if save:
        fig.savefig(outdir / f"dashboard__{policy}.png", dpi=150, bbox_inches="tight")


# ---------- plotting: comparisons ----------
def compute_final_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return one row per (policy, episode): the final day of that episode.
    """
    if df.empty:
        return df
    finals = (
        df.groupby(["policy", "episode"], as_index=False, sort=False)
          .apply(lambda g: g.sort_values("day").tail(1))
          .reset_index(drop=True)
    )
    return finals


def final_profit_comparison(df: pd.DataFrame, outdir: Path, save: bool, figsize=(10, 5)):
    """
    Adaptive comparison of final cumulative profit across policies & episodes.
    - If counts are modest -> grouped bars
    - If large -> heatmap (policies × episodes)
    - Always provide an auxiliary 'mean per policy' bar chart for quick read
    """
    ensure_outdir(outdir, save)

    finals = compute_final_rows(df)
    if finals.empty:
        print("[warn] No final rows found; skipping comparison plot.")
        return

    # --- summary: mean final profit per policy (always plotted) ---
    mean_per_policy = (
        finals.groupby("policy", as_index=False)["cum_profit"]
              .mean()
              .sort_values("cum_profit", ascending=False)
    )

    # thresholds for switching layout
    n_policies = finals["policy"].nunique()
    n_episodes = finals["episode"].nunique()
    many = (n_policies > 8) or (n_episodes > 20)

    if not many:
        # ===== grouped bars =====
        # wide: index=episode, columns=policy; allow missing episodes
        pivot = (
            finals.pivot_table(
                index="episode", columns="policy", values="cum_profit", aggfunc="mean"
            )
            .sort_index()
        )

        # uniform x positions so missing episodes don't create large gaps
        episodes = np.arange(len(pivot.index))  # 0..N-1
        series = [pivot[col].to_numpy() for col in pivot.columns]
        labels = list(pivot.columns)

        # figure width grows with #policies to prevent squish
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
        # ===== heatmap =====
        # dense view that fits many policies × episodes
        pivot = (
            finals.pivot_table(
                index="policy", columns="episode", values="cum_profit", aggfunc="mean"
            )
            .sort_index()              # policies alpha
            .reindex(sorted(pivot.columns) if (pivot := finals.pivot_table(
                index="policy", columns="episode", values="cum_profit", aggfunc="mean"
            )) is not None else None, axis=1)
        )
        # dynamic figure size
        h = max(4, min(0.35 * len(pivot.index), 20))
        w = max(6, min(0.30 * len(pivot.columns), 24))

        
        vals = pivot.to_numpy().astype(float)
        # 1) robust min/max to avoid being dominated by outliers
        lo, hi = np.nanpercentile(vals, [2, 98])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = np.nanmin(vals), np.nanmax(vals)

        cmap = "magma"                      # alternatives: "inferno", "turbo", "plasma"
        norm = Normalize(vmin=lo, vmax=hi)
        
        fig, ax = plt.subplots(figsize=(w, h), layout="constrained")
        im = ax.imshow(pivot.to_numpy(), aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
        ax.set_title("Final Profit per Episode — Heatmap")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Policy")
        ax.set_xticks(np.arange(pivot.shape[1]))
        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_xticklabels([str(int(e)) for e in pivot.columns], rotation=90, ha="center")
        ax.set_yticklabels(list(pivot.index))

        # 1) right-align y labels and add some padding so they don't sit on the axis
        plt.setp(ax.get_yticklabels(), ha="right")
        ax.tick_params(axis="y", pad=6)

        # 2) ensure enough left margin; try to size it to the longest label
        try:
            # if supported, use constrained layout (best), otherwise adjust left margin
            # (safe to call even if already using constrained layout)
            fig.set_constrained_layout(True)
        except Exception:
            pass

        # If constrained layout isn't available/effective (older Matplotlib/backends), fall back to manual adjust
        try:
            fig.canvas.draw()  # needed to get accurate text extents
            renderer = fig.canvas.get_renderer()
            maxw_px = max(t.get_window_extent(renderer=renderer).width for t in ax.get_yticklabels())
            fig_w_in = fig.get_size_inches()[0]
            dpi = fig.dpi
            # fraction of figure width needed for labels + small gap
            left_needed = maxw_px / (fig_w_in * dpi) + 0.04
            # clamp to a sane range so colorbar/title don't get crushed
            fig.subplots_adjust(left=min(0.45, max(0.12, left_needed)))
        except Exception:
            # conservative fallback if we couldn't measure
            fig.subplots_adjust(left=0.22)
        
        ticks = np.linspace(lo, hi, num=6)
        cbar = fig.colorbar(im, ax=ax, ticks=ticks)
        cbar.set_label("Final Cumulative Profit")
        if not getattr(fig, "get_constrained_layout", lambda: False)():
            fig.tight_layout()
        if save:
            fig.savefig(outdir / "final_profit_comparison__heatmap.png", dpi=150, bbox_inches="tight")

    # --- auxiliary: mean bar per policy (compact, always useful) ---
        fig_w = max(8, 0.45 * len(mean_per_policy))
        fig, ax = plt.subplots(figsize=(fig_w, 5))

        policies_ordered = mean_per_policy["policy"].tolist()
        profits_ordered  = mean_per_policy["cum_profit"].to_numpy()

        bars = ax.bar(policies_ordered, profits_ordered)

        # axis titles/labels
        ax.set_title("Average Final Profit per Policy")
        ax.set_xlabel("Policy")
        ax.set_ylabel("Mean Final Cumulative Profit")

        # nice y-axis formatting (1,234 style)
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

        # rotate x labels without set_xticklabels warning
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        # annotate each bar with its profit
        def _label_bars(ax, bars, fmt="{:,.0f}", offset=3, fontsize=9):
            for rect in bars:
                height = rect.get_height()
                x = rect.get_x() + rect.get_width() / 2
                y = height
                va = "bottom" if height >= 0 else "top"
                dy = offset if height >= 0 else -offset
                ax.annotate(fmt.format(height),
                            xy=(x, y),
                            xytext=(0, dy),
                            textcoords="offset points",
                            ha="center", va=va,
                            fontsize=fontsize)

        _label_bars(ax, bars)

        fig.tight_layout()
        if save:
            fig.savefig(outdir / "final_profit__mean_per_policy.png", dpi=150, bbox_inches="tight")



# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Grouped plots for evaluator CSV (matplotlib).")
    ap.add_argument("--csv", type=str, default="eval_daily_results.csv", help="Path to evaluator CSV.")
    ap.add_argument("--save", action="store_true", help="Save figures to --outdir.")
    ap.add_argument("--outdir", type=str, default="plots", help="Directory for saved figures.")
    ap.add_argument("--episode", type=int, default=None, help="Episode to use for detail panels; defaults to first available.")
    ap.add_argument("--no-show", action="store_true", help="Don’t call plt.show() (useful on headless runs with --save).")
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    df = load_csv(args.csv)

    policies = list(df["policy"].dropna().unique())
    if not policies:
        print("[error] No policies found in CSV.")
        return

    # Build a dashboard per policy
    for pol in policies:
        policy_dashboard(df, pol, args.episode, outdir, args.save)

    # Cross-policy comparison
    final_profit_comparison(df, outdir, args.save)

    if not args.no_show:
        plt.show()
    else:
        print(f"Saved plots to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
