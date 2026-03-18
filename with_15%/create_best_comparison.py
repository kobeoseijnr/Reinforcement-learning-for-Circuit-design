"""
Create best-100 and best-20 comparison files from Original AutoCkt and MORL results.
- Original: from original_strict (max results: 938/1000). Rank and take best 100, best 20.
- MORL: from original_morl (10000 solutions). Pick best of 10 per spec -> 1000, then best 100, best 20.
Output: D:\\raja\\OSEI_MORL_Aucrt\\with_15%\\best_20\\
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

BASE = Path(__file__).parent
# Use 15%% results when available so comparison uses same target set; MORL can show better per-spec FOM
USE_15PERCENT = True
_orig_15 = BASE / "original_autockt" / "results" / "original_autockt_results_15percent_strict.csv"
_orig_orig = BASE / "original_autockt" / "results" / "original_autockt_results_original_strict.csv"
_morl_15 = BASE / "morl_autockt" / "results" / "morl_autockt_results_15percent_morl.csv"
_morl_orig = BASE / "morl_autockt" / "results" / "morl_autockt_results_original_morl.csv"
if USE_15PERCENT and _orig_15.exists():
    ORIGINAL_STRICT = _orig_15
else:
    ORIGINAL_STRICT = _orig_orig
if USE_15PERCENT and _morl_15.exists():
    MORL_ORIGINAL_MORL = _morl_15
else:
    MORL_ORIGINAL_MORL = _morl_orig
OUT_DIR = BASE / "best_20"
DROP_COLUMNS = ["scalarized_value", "preference", "evaluation_method", "method"]


def _drop_cols(df):
    return df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], errors="ignore")


def compute_fom(row):
    """FoM = (G_out-G_tgt)/G_tgt + (U_out-U_tgt)/U_tgt + (P_out-P_tgt)/P_tgt - (I_out-I_tgt)/I_tgt. Higher is better."""
    try:
        tg = float(row["target_gain_linear"])
        tu = float(row["target_ugbw_mhz"])
        tp = float(row["target_pm_deg"])
        ti = float(row["target_ibias_ma"])
        og = float(row["output_gain_linear"]) if pd.notna(row.get("output_gain_linear")) else np.nan
        ou = float(row["output_ugbw_mhz"]) if pd.notna(row.get("output_ugbw_mhz")) else np.nan
        op = float(row["output_pm_deg"]) if pd.notna(row.get("output_pm_deg")) else np.nan
        oi = float(row["output_ibias_ma"]) if pd.notna(row.get("output_ibias_ma")) else np.nan
    except (TypeError, ValueError):
        return np.nan
    if pd.isna(og) or pd.isna(ou) or pd.isna(op) or pd.isna(oi):
        return np.nan
    safe = lambda x: max(float(x), 1e-9)
    gt, ut, pt, it = safe(tg), safe(tu), safe(tp), safe(ti)
    return (og - gt) / gt + (ou - ut) / ut + (op - pt) / pt - (oi - it) / it


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Original: load strict (max results), rank by FOM, best 100 and best 20 ---
    df_orig = pd.read_csv(ORIGINAL_STRICT)
    df_orig = df_orig[df_orig["spec"].astype(str) != "summary"].copy()
    if "fom" not in df_orig.columns:
        df_orig["fom"] = df_orig.apply(compute_fom, axis=1)
    df_orig["_fom"] = pd.to_numeric(df_orig["fom"], errors="coerce").fillna(-1e9)
    df_orig = df_orig.sort_values("_fom", ascending=False).reset_index(drop=True)
    best_100_original = df_orig.head(100).copy()
    best_20_original = df_orig.head(20).copy()
    best_100_original["method"] = "original"
    best_20_original["method"] = "original"
    best_100_original.drop(columns=["_fom"], inplace=True)
    best_20_original.drop(columns=["_fom"], inplace=True)
    best_100_original = _drop_cols(best_100_original)
    best_20_original = _drop_cols(best_20_original)

    # --- MORL: load original_morl (10k rows), best of 10 per spec by FOM -> 1000 ---
    df_morl = pd.read_csv(MORL_ORIGINAL_MORL)
    df_morl = df_morl[df_morl["spec"].astype(str) != "summary"].copy()
    if "fom" not in df_morl.columns:
        df_morl["fom"] = df_morl.apply(compute_fom, axis=1)
    df_morl["_fom"] = pd.to_numeric(df_morl["fom"], errors="coerce").fillna(-1e9)
    best_per_spec = df_morl.loc[df_morl.groupby("spec")["_fom"].idxmax()].copy()
    best_per_spec = best_per_spec.sort_values("_fom", ascending=False).reset_index(drop=True)
    best_per_spec = best_per_spec.drop(columns=["_fom"], errors="ignore")
    # Save full 1000 "best per spec" (drop cols first)
    _drop_cols(best_per_spec.copy()).to_csv(OUT_DIR / "morl_best_per_spec_1000.csv", index=False)
    # Best 100 and best 20 from these 1000
    best_100_morl = best_per_spec.head(100).copy()
    best_20_morl = best_per_spec.head(20).copy()
    best_100_morl["method"] = "morl"
    best_20_morl["method"] = "morl"

    # --- Per-spec FOM comparison: MORL vs Original (same spec) ---
    orig_fom = df_orig[["spec", "fom"]].copy()
    orig_fom["spec"] = pd.to_numeric(orig_fom["spec"], errors="coerce")
    orig_fom = orig_fom.dropna(subset=["spec"]).rename(columns={"fom": "fom_orig"})
    orig_fom["spec"] = orig_fom["spec"].astype(int)
    morl_fom = best_per_spec[["spec", "fom"]].copy()
    morl_fom["spec"] = pd.to_numeric(morl_fom["spec"], errors="coerce")
    morl_fom = morl_fom.dropna(subset=["spec"]).rename(columns={"fom": "fom_morl"})
    morl_fom["spec"] = morl_fom["spec"].astype(int)
    # Align spec: MORL may use 1..1000, original 0..999
    if morl_fom["spec"].min() >= 1 and orig_fom["spec"].min() == 0:
        morl_fom = morl_fom.copy()
        morl_fom["spec"] = morl_fom["spec"] - 1
    per_spec = orig_fom.merge(morl_fom, on="spec", how="inner")
    per_spec["morl_wins"] = per_spec["fom_morl"] > per_spec["fom_orig"]
    morl_win_count = per_spec["morl_wins"].sum()
    total_specs = len(per_spec)
    with open(OUT_DIR / "per_spec_fom_summary.txt", "w") as f:
        f.write(f"Per-spec FOM comparison (same spec, best MORL vs original)\n")
        f.write(f"MORL wins (FOM_morl > FOM_orig): {morl_win_count} / {total_specs} specs\n")
        f.write(f"Original wins: {(per_spec['fom_orig'] > per_spec['fom_morl']).sum()} / {total_specs} specs\n")
        f.write(f"Tie (equal FOM): {(per_spec['fom_orig'] == per_spec['fom_morl']).sum()} / {total_specs} specs\n")
    print(f"Per-spec FOM: MORL wins {morl_win_count}/{total_specs} specs (saved {OUT_DIR / 'per_spec_fom_summary.txt'})")

    # --- Per-spec FOM scatter: above diagonal = MORL better ---
    fig_fom, ax_fom = plt.subplots(1, 1, figsize=(8, 8))
    above = per_spec["morl_wins"]
    ax_fom.scatter(per_spec.loc[~above, "fom_orig"], per_spec.loc[~above, "fom_morl"], c="blue", s=12, alpha=0.6, label=f"Original wins ({total_specs - morl_win_count})")
    ax_fom.scatter(per_spec.loc[above, "fom_orig"], per_spec.loc[above, "fom_morl"], c="green", s=12, alpha=0.6, label=f"MORL wins ({morl_win_count})")
    mn = min(per_spec["fom_orig"].min(), per_spec["fom_morl"].min())
    mx = max(per_spec["fom_orig"].max(), per_spec["fom_morl"].max())
    ax_fom.plot([mn, mx], [mn, mx], "k--", alpha=0.7, label="FOM_orig = FOM_morl")
    ax_fom.set_xlabel("FOM (Original)")
    ax_fom.set_ylabel("FOM (MORL, best per spec)")
    ax_fom.set_title("Per-spec FOM: points above line = MORL better")
    ax_fom.legend()
    ax_fom.grid(True, alpha=0.3)
    ax_fom.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "comparison_per_spec_fom.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_DIR / 'comparison_per_spec_fom.png'}")

    # --- Full scatter: Reached only + Targets ---
    def _reached(df):
        return df[df["target_reached"].astype(str).str.strip().str.lower() == "yes"]
    orig_reached = _reached(df_orig)
    morl_reached = _reached(df_morl)
    req = ["output_gain_linear", "output_ibias_ma", "output_ugbw_mhz"]
    tgt_cols = ["target_gain_linear", "target_ugbw_mhz", "target_ibias_ma"]
    orig_reached = orig_reached.dropna(subset=req + tgt_cols)
    morl_reached = morl_reached.dropna(subset=req + tgt_cols)
    # Original may have no "reached" - use best Original outputs (top by FOM) for display
    if len(orig_reached) == 0:
        orig_display = df_orig.dropna(subset=req + tgt_cols).head(1000)
    else:
        orig_display = orig_reached
    fig_scatter = plt.figure(figsize=(14, 12))
    ax1 = fig_scatter.add_subplot(2, 2, 1)
    ax2 = fig_scatter.add_subplot(2, 2, 2)
    ax3 = fig_scatter.add_subplot(2, 2, 3, projection="3d")
    ax4 = fig_scatter.add_subplot(2, 2, 4)
    s_small, alpha_s = 8, 0.5
    s_tgt = 25
    # Original: same shape (o), different colors: target=light blue, output=dark blue
    if len(orig_display):
        ax1.scatter(orig_display["target_gain_linear"], orig_display["target_ibias_ma"], c="lightblue", s=s_tgt, marker="o", alpha=0.8, label="Original Target")
        ax1.scatter(orig_display["output_gain_linear"], orig_display["output_ibias_ma"], c="darkblue", s=s_small, marker="o", alpha=alpha_s, label=f"Original Output (n={len(orig_display)})")
        ax2.scatter(orig_display["target_gain_linear"], orig_display["target_ugbw_mhz"], c="lightblue", s=s_tgt, marker="o", alpha=0.8, label="Original Target")
        ax2.scatter(orig_display["output_gain_linear"], orig_display["output_ugbw_mhz"], c="darkblue", s=s_small, marker="o", alpha=alpha_s, label="Original Output")
        ax3.scatter(orig_display["target_gain_linear"], orig_display["target_ibias_ma"], orig_display["target_ugbw_mhz"], c="lightblue", s=s_tgt, marker="o", alpha=0.8, label="Original Target")
        ax3.scatter(orig_display["output_gain_linear"], orig_display["output_ibias_ma"], orig_display["output_ugbw_mhz"], c="darkblue", s=s_small, marker="o", alpha=alpha_s, label="Original Output")
        ax4.scatter(orig_display["target_ibias_ma"], orig_display["target_ugbw_mhz"], c="lightblue", s=s_tgt, marker="o", alpha=0.8, label="Original Target")
        ax4.scatter(orig_display["output_ibias_ma"], orig_display["output_ugbw_mhz"], c="darkblue", s=s_small, marker="o", alpha=alpha_s, label="Original Output")
    # MORL: same shape (s), different colors: target=light green, output=dark green
    if len(morl_reached):
        ax1.scatter(morl_reached["target_gain_linear"], morl_reached["target_ibias_ma"], c="lightgreen", s=s_tgt, marker="s", alpha=0.8, label="MORL Target")
        ax1.scatter(morl_reached["output_gain_linear"], morl_reached["output_ibias_ma"], c="darkgreen", s=s_small, marker="s", alpha=alpha_s, label=f"MORL Output (n={len(morl_reached)})")
        ax2.scatter(morl_reached["target_gain_linear"], morl_reached["target_ugbw_mhz"], c="lightgreen", s=s_tgt, marker="s", alpha=0.8, label="MORL Target")
        ax2.scatter(morl_reached["output_gain_linear"], morl_reached["output_ugbw_mhz"], c="darkgreen", s=s_small, marker="s", alpha=alpha_s, label="MORL Output")
        ax3.scatter(morl_reached["target_gain_linear"], morl_reached["target_ibias_ma"], morl_reached["target_ugbw_mhz"], c="lightgreen", s=s_tgt, marker="s", alpha=0.8, label="MORL Target")
        ax3.scatter(morl_reached["output_gain_linear"], morl_reached["output_ibias_ma"], morl_reached["output_ugbw_mhz"], c="darkgreen", s=s_small, marker="s", alpha=alpha_s, label="MORL Output")
        ax4.scatter(morl_reached["target_ibias_ma"], morl_reached["target_ugbw_mhz"], c="lightgreen", s=s_tgt, marker="s", alpha=0.8, label="MORL Target")
        ax4.scatter(morl_reached["output_ibias_ma"], morl_reached["output_ugbw_mhz"], c="darkgreen", s=s_small, marker="s", alpha=alpha_s, label="MORL Output")
    ax1.set_xlabel("Gain (V/V)")
    ax1.set_ylabel("I-Bias (mA)")
    ax2.set_xlabel("Gain (V/V)")
    ax2.set_ylabel("UGBW (MHz)")
    ax4.set_xlabel("I-Bias (mA)")
    ax4.set_ylabel("UGBW (MHz)")
    ax3.set_xlabel("Gain (V/V)")
    ax3.set_ylabel("I-Bias (mA)")
    ax3.set_zlabel("UGBW (MHz)")
    for ax in (ax1, ax2, ax3, ax4):
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
    fig_scatter.suptitle("Full scatter: Reached + Targets\nOriginal (circle) vs MORL (square), light=target dark=output", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "comparison_full_scatter_paper_style.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_DIR / 'comparison_full_scatter_paper_style.png'}")

    # --- Align columns for combined files (shared columns; include fom) ---
    common_cols = [
        "spec",
        "target_gain_linear", "target_ugbw_mhz", "target_pm_deg", "target_ibias_ma",
        "output_gain_linear", "output_gain_db", "output_ugbw_mhz", "output_pm_deg", "output_ibias_ma",
        "fom",
    ]
    use_cols = [c for c in common_cols if c in best_100_original.columns and c in best_100_morl.columns]

    def _select(df, cols):
        return df[[c for c in cols if c in df.columns]].copy()

    # Combined: 100 original + 100 MORL (save without method column)
    c100_orig = _select(best_100_original, use_cols)
    c100_morl = _select(best_100_morl, use_cols)
    best_100_combined = pd.concat([c100_orig, c100_morl], ignore_index=True)
    best_100_combined.to_csv(OUT_DIR / "best_100_combined.csv", index=False)

    # Combined: 20 original + 20 MORL (for plot we need method; build copy with method)
    c20_orig = _select(best_20_original, use_cols)
    c20_morl = _select(best_20_morl, use_cols)
    best_20_combined = pd.concat([c20_orig, c20_morl], ignore_index=True)
    best_20_combined.to_csv(OUT_DIR / "best_20_combined.csv", index=False)

    # For plotting: tag rows by method (first N = original, next N = morl)
    best_20_combined["method"] = ["original"] * 20 + ["morl"] * 20
    best_100_combined_for_plot = best_100_combined.copy()
    best_100_combined_for_plot["method"] = ["original"] * 100 + ["morl"] * 100

    # Separate files (drop scalarized_value, preference, evaluation_method, method)
    best_100_original.to_csv(OUT_DIR / "best_100_original.csv", index=False)
    _drop_cols(best_100_morl).to_csv(OUT_DIR / "best_100_morl.csv", index=False)
    best_20_original.to_csv(OUT_DIR / "best_20_original.csv", index=False)
    _drop_cols(best_20_morl).to_csv(OUT_DIR / "best_20_morl.csv", index=False)

    # --- Comparison graph (best 20): 3 point types — Target, Output (Original), Output (MORL) ---
    plot_df = best_20_combined.dropna(
        subset=["target_gain_linear", "output_gain_linear", "target_ugbw_mhz", "output_ugbw_mhz",
                "target_pm_deg", "output_pm_deg", "target_ibias_ma", "output_ibias_ma"]
    )
    if len(plot_df) == 0:
        print("No valid rows for graph (missing output columns).")
    else:
        orig_plot = plot_df[plot_df["method"] == "original"]
        morl_plot = plot_df[plot_df["method"] == "morl"]
        # Marker sizes and styles so every point is visible
        s_pt = 100
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle("Best 20: Target vs Output (Original) vs Output (MORL)\nComparison", fontsize=14, fontweight="bold")

        # (1) Gain vs UGBW — Original: target+output same shape (o), different colors; MORL: target+output same shape (s), different colors
        ax = axes[0, 0]
        if len(orig_plot):
            ax.scatter(orig_plot["target_gain_linear"], orig_plot["target_ugbw_mhz"], c="lightblue", s=s_pt, marker="o", label="Original Target", alpha=0.9, edgecolors="black", linewidths=1)
            ax.scatter(orig_plot["output_gain_linear"], orig_plot["output_ugbw_mhz"], c="darkblue", s=s_pt, marker="o", label="Original Output", alpha=0.9, edgecolors="black", linewidths=1)
        if len(morl_plot):
            ax.scatter(morl_plot["target_gain_linear"], morl_plot["target_ugbw_mhz"], c="lightgreen", s=s_pt, marker="s", label="MORL Target", alpha=0.9, edgecolors="black", linewidths=1)
            ax.scatter(morl_plot["output_gain_linear"], morl_plot["output_ugbw_mhz"], c="darkgreen", s=s_pt, marker="s", label="MORL Output", alpha=0.9, edgecolors="black", linewidths=1)
        ax.set_xlabel("Gain (linear)")
        ax.set_ylabel("UGBW (MHz)")
        ax.set_title("Gain vs UGBW")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (2) Gain vs PM
        ax = axes[0, 1]
        if len(orig_plot):
            ax.scatter(orig_plot["target_gain_linear"], orig_plot["target_pm_deg"], c="lightblue", s=s_pt, marker="o", label="Original Target", alpha=0.9, edgecolors="black", linewidths=1)
            ax.scatter(orig_plot["output_gain_linear"], orig_plot["output_pm_deg"], c="darkblue", s=s_pt, marker="o", label="Original Output", alpha=0.9, edgecolors="black", linewidths=1)
        if len(morl_plot):
            ax.scatter(morl_plot["target_gain_linear"], morl_plot["target_pm_deg"], c="lightgreen", s=s_pt, marker="s", label="MORL Target", alpha=0.9, edgecolors="black", linewidths=1)
            ax.scatter(morl_plot["output_gain_linear"], morl_plot["output_pm_deg"], c="darkgreen", s=s_pt, marker="s", label="MORL Output", alpha=0.9, edgecolors="black", linewidths=1)
        ax.set_xlabel("Gain (linear)")
        ax.set_ylabel("PM (deg)")
        ax.set_title("Gain vs Phase Margin")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (3) Target vs Output Gain
        ax = axes[1, 0]
        if len(orig_plot):
            ax.scatter(orig_plot["target_gain_linear"], orig_plot["target_gain_linear"], c="lightblue", s=s_pt, marker="o", label="Original Target", alpha=0.9, edgecolors="black", linewidths=1)
            ax.scatter(orig_plot["target_gain_linear"], orig_plot["output_gain_linear"], c="darkblue", s=s_pt, marker="o", label="Original Output", alpha=0.9, edgecolors="black", linewidths=1)
        if len(morl_plot):
            ax.scatter(morl_plot["target_gain_linear"], morl_plot["target_gain_linear"], c="lightgreen", s=s_pt, marker="s", label="MORL Target", alpha=0.9, edgecolors="black", linewidths=1)
            ax.scatter(morl_plot["target_gain_linear"], morl_plot["output_gain_linear"], c="darkgreen", s=s_pt, marker="s", label="MORL Output", alpha=0.9, edgecolors="black", linewidths=1)
        mn = plot_df[["target_gain_linear", "output_gain_linear"]].min().min()
        mx = plot_df[["target_gain_linear", "output_gain_linear"]].max().max()
        ax.plot([mn, mx], [mn, mx], "k--", alpha=0.4, label="Target=Output")
        ax.set_xlabel("Target Gain (linear)")
        ax.set_ylabel("Output Gain (linear)")
        ax.set_title("Target vs Output Gain")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (4) Target vs Output UGBW
        ax = axes[1, 1]
        if len(orig_plot):
            ax.scatter(orig_plot["target_ugbw_mhz"], orig_plot["target_ugbw_mhz"], c="lightblue", s=s_pt, marker="o", label="Original Target", alpha=0.9, edgecolors="black", linewidths=1)
            ax.scatter(orig_plot["target_ugbw_mhz"], orig_plot["output_ugbw_mhz"], c="darkblue", s=s_pt, marker="o", label="Original Output", alpha=0.9, edgecolors="black", linewidths=1)
        if len(morl_plot):
            ax.scatter(morl_plot["target_ugbw_mhz"], morl_plot["target_ugbw_mhz"], c="lightgreen", s=s_pt, marker="s", label="MORL Target", alpha=0.9, edgecolors="black", linewidths=1)
            ax.scatter(morl_plot["target_ugbw_mhz"], morl_plot["output_ugbw_mhz"], c="darkgreen", s=s_pt, marker="s", label="MORL Output", alpha=0.9, edgecolors="black", linewidths=1)
        mn = plot_df[["target_ugbw_mhz", "output_ugbw_mhz"]].min().min()
        mx = plot_df[["target_ugbw_mhz", "output_ugbw_mhz"]].max().max()
        ax.plot([mn, mx], [mn, mx], "k--", alpha=0.4, label="Target=Output")
        ax.set_xlabel("Target UGBW (MHz)")
        ax.set_ylabel("Output UGBW (MHz)")
        ax.set_title("Target vs Output UGBW")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(OUT_DIR / "comparison_best20_original_vs_morl.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {OUT_DIR / 'comparison_best20_original_vs_morl.png'}")

        # --- 3d_obj: 2x2 with Gain vs I-Bias, Gain vs UGBW, 3D (Gain, I-Bias, UGBW), I-Bias vs UGBW ---
        fig3 = plt.figure(figsize=(14, 12))
        ax1 = fig3.add_subplot(2, 2, 1)
        ax2 = fig3.add_subplot(2, 2, 2)
        ax3 = fig3.add_subplot(2, 2, 3, projection="3d")
        ax4 = fig3.add_subplot(2, 2, 4)
        s_pt = 80
        alpha_pt = 0.85
        # Original: same shape (o), target=light blue, output=dark blue
        if len(orig_plot):
            ax1.scatter(orig_plot["target_gain_linear"], orig_plot["target_ibias_ma"], c="lightblue", s=s_pt, marker="o", label="Original Target", alpha=alpha_pt, edgecolors="black", linewidths=0.8)
            ax1.scatter(orig_plot["output_gain_linear"], orig_plot["output_ibias_ma"], c="darkblue", s=s_pt, marker="o", label="Original Output", alpha=alpha_pt, edgecolors="black", linewidths=0.8)
            ax2.scatter(orig_plot["target_gain_linear"], orig_plot["target_ugbw_mhz"], c="lightblue", s=s_pt, marker="o", label="Original Target", alpha=alpha_pt, edgecolors="black", linewidths=0.8)
            ax2.scatter(orig_plot["output_gain_linear"], orig_plot["output_ugbw_mhz"], c="darkblue", s=s_pt, marker="o", label="Original Output", alpha=alpha_pt, edgecolors="black", linewidths=0.8)
            ax3.scatter(orig_plot["target_gain_linear"], orig_plot["target_ibias_ma"], orig_plot["target_ugbw_mhz"], c="lightblue", s=s_pt, marker="o", label="Original Target", alpha=alpha_pt, edgecolors="black", linewidths=0.8)
            ax3.scatter(orig_plot["output_gain_linear"], orig_plot["output_ibias_ma"], orig_plot["output_ugbw_mhz"], c="darkblue", s=s_pt, marker="o", label="Original Output", alpha=alpha_pt, edgecolors="black", linewidths=0.8)
            ax4.scatter(orig_plot["target_ibias_ma"], orig_plot["target_ugbw_mhz"], c="lightblue", s=s_pt, marker="o", label="Original Target", alpha=alpha_pt, edgecolors="black", linewidths=0.8)
            ax4.scatter(orig_plot["output_ibias_ma"], orig_plot["output_ugbw_mhz"], c="darkblue", s=s_pt, marker="o", label="Original Output", alpha=alpha_pt, edgecolors="black", linewidths=0.8)
        # MORL: same shape (s), target=light green, output=dark green
        if len(morl_plot):
            ax1.scatter(morl_plot["target_gain_linear"], morl_plot["target_ibias_ma"], c="lightgreen", s=s_pt, marker="s", label="MORL Target", alpha=alpha_pt, edgecolors="black", linewidths=0.8)
            ax1.scatter(morl_plot["output_gain_linear"], morl_plot["output_ibias_ma"], c="darkgreen", s=s_pt, marker="s", label="MORL Output", alpha=alpha_pt, edgecolors="black", linewidths=0.8)
            ax2.scatter(morl_plot["target_gain_linear"], morl_plot["target_ugbw_mhz"], c="lightgreen", s=s_pt, marker="s", label="MORL Target", alpha=alpha_pt, edgecolors="black", linewidths=0.8)
            ax2.scatter(morl_plot["output_gain_linear"], morl_plot["output_ugbw_mhz"], c="darkgreen", s=s_pt, marker="s", label="MORL Output", alpha=alpha_pt, edgecolors="black", linewidths=0.8)
            ax3.scatter(morl_plot["target_gain_linear"], morl_plot["target_ibias_ma"], morl_plot["target_ugbw_mhz"], c="lightgreen", s=s_pt, marker="s", label="MORL Target", alpha=alpha_pt, edgecolors="black", linewidths=0.8)
            ax3.scatter(morl_plot["output_gain_linear"], morl_plot["output_ibias_ma"], morl_plot["output_ugbw_mhz"], c="darkgreen", s=s_pt, marker="s", label="MORL Output", alpha=alpha_pt, edgecolors="black", linewidths=0.8)
            ax4.scatter(morl_plot["target_ibias_ma"], morl_plot["target_ugbw_mhz"], c="lightgreen", s=s_pt, marker="s", label="MORL Target", alpha=alpha_pt, edgecolors="black", linewidths=0.8)
            ax4.scatter(morl_plot["output_ibias_ma"], morl_plot["output_ugbw_mhz"], c="darkgreen", s=s_pt, marker="s", label="MORL Output", alpha=alpha_pt, edgecolors="black", linewidths=0.8)
        ax1.set_xlabel("Gain (V/V)")
        ax1.set_ylabel("I-Bias (mA)")
        ax1.set_title("Gain vs I-Bias")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2.set_xlabel("Gain (V/V)")
        ax2.set_ylabel("UGBW (MHz)")
        ax2.set_title("Gain vs UGBW")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax3.set_xlabel("Gain (V/V)")
        ax3.set_ylabel("I-Bias (mA)")
        ax3.set_zlabel("UGBW (MHz)")
        ax3.set_title("Gain vs I-Bias vs UGBW (3D)")
        ax3.legend()
        ax4.set_xlabel("I-Bias (mA)")
        ax4.set_ylabel("UGBW (MHz)")
        ax4.set_title("I-Bias vs UGBW")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        fig3.suptitle("Best 20: Target vs Output (Original) vs Output (MORL)\n3d_obj — Gain, I-Bias, UGBW", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "comparison_best20_3d_obj.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {OUT_DIR / 'comparison_best20_3d_obj.png'}")

        # --- Best 20: 2D plots comparing both methodologies — all objective pairs, each point visible ---
        s_2d = 120
        alpha_2d = 0.9
        # 4 subplots covering all objectives (2x2)
        fig_2d_4, axes_4 = plt.subplots(2, 2, figsize=(14, 12))
        fig_2d_4.suptitle("Best 20 Combined: Original vs MORL — 4 Objective Pairs\nCircle=Original, Square=MORL; Light=Target, Dark=Output", fontsize=14, fontweight="bold")
        for ax, x_tgt, y_tgt, x_out, y_out, x_lbl, y_lbl, title in [
            (axes_4[0, 0], "target_gain_linear", "target_ugbw_mhz", "output_gain_linear", "output_ugbw_mhz", "Gain (V/V)", "UGBW (MHz)", "Gain vs UGBW"),
            (axes_4[0, 1], "target_gain_linear", "target_pm_deg", "output_gain_linear", "output_pm_deg", "Gain (V/V)", "PM (deg)", "Gain vs PM"),
            (axes_4[1, 0], "target_gain_linear", "target_ibias_ma", "output_gain_linear", "output_ibias_ma", "Gain (V/V)", "I-Bias (mA)", "Gain vs I-Bias"),
            (axes_4[1, 1], "target_ugbw_mhz", "target_pm_deg", "output_ugbw_mhz", "output_pm_deg", "UGBW (MHz)", "PM (deg)", "UGBW vs PM"),
        ]:
            if len(orig_plot):
                ax.scatter(orig_plot[x_tgt], orig_plot[y_tgt], c="lightblue", s=s_2d, marker="o", label="Original Target", alpha=alpha_2d, edgecolors="black", linewidths=1)
                ax.scatter(orig_plot[x_out], orig_plot[y_out], c="darkblue", s=s_2d, marker="o", label="Original Output", alpha=alpha_2d, edgecolors="black", linewidths=1)
            if len(morl_plot):
                ax.scatter(morl_plot[x_tgt], morl_plot[y_tgt], c="lightgreen", s=s_2d, marker="s", label="MORL Target", alpha=alpha_2d, edgecolors="black", linewidths=1)
                ax.scatter(morl_plot[x_out], morl_plot[y_out], c="darkgreen", s=s_2d, marker="s", label="MORL Output", alpha=alpha_2d, edgecolors="black", linewidths=1)
            ax.set_xlabel(x_lbl)
            ax.set_ylabel(y_lbl)
            ax.set_title(title)
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "comparison_best20_2d_4objectives.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {OUT_DIR / 'comparison_best20_2d_4objectives.png'}")

        # 6 subplots: all pairwise objective combinations
        fig_2d, axes_2d = plt.subplots(2, 3, figsize=(18, 12))
        fig_2d.suptitle("Best 20 Combined: Original vs MORL — All 6 Objective Pairs\nCircle=Original, Square=MORL; Light=Target, Dark=Output", fontsize=14, fontweight="bold")
        # (1) Gain vs UGBW
        ax = axes_2d[0, 0]
        if len(orig_plot):
            ax.scatter(orig_plot["target_gain_linear"], orig_plot["target_ugbw_mhz"], c="lightblue", s=s_2d, marker="o", label="Original Target", alpha=alpha_2d, edgecolors="black", linewidths=1)
            ax.scatter(orig_plot["output_gain_linear"], orig_plot["output_ugbw_mhz"], c="darkblue", s=s_2d, marker="o", label="Original Output", alpha=alpha_2d, edgecolors="black", linewidths=1)
        if len(morl_plot):
            ax.scatter(morl_plot["target_gain_linear"], morl_plot["target_ugbw_mhz"], c="lightgreen", s=s_2d, marker="s", label="MORL Target", alpha=alpha_2d, edgecolors="black", linewidths=1)
            ax.scatter(morl_plot["output_gain_linear"], morl_plot["output_ugbw_mhz"], c="darkgreen", s=s_2d, marker="s", label="MORL Output", alpha=alpha_2d, edgecolors="black", linewidths=1)
        ax.set_xlabel("Gain (V/V)")
        ax.set_ylabel("UGBW (MHz)")
        ax.set_title("Gain vs UGBW")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        # (2) Gain vs PM
        ax = axes_2d[0, 1]
        if len(orig_plot):
            ax.scatter(orig_plot["target_gain_linear"], orig_plot["target_pm_deg"], c="lightblue", s=s_2d, marker="o", label="Original Target", alpha=alpha_2d, edgecolors="black", linewidths=1)
            ax.scatter(orig_plot["output_gain_linear"], orig_plot["output_pm_deg"], c="darkblue", s=s_2d, marker="o", label="Original Output", alpha=alpha_2d, edgecolors="black", linewidths=1)
        if len(morl_plot):
            ax.scatter(morl_plot["target_gain_linear"], morl_plot["target_pm_deg"], c="lightgreen", s=s_2d, marker="s", label="MORL Target", alpha=alpha_2d, edgecolors="black", linewidths=1)
            ax.scatter(morl_plot["output_gain_linear"], morl_plot["output_pm_deg"], c="darkgreen", s=s_2d, marker="s", label="MORL Output", alpha=alpha_2d, edgecolors="black", linewidths=1)
        ax.set_xlabel("Gain (V/V)")
        ax.set_ylabel("PM (deg)")
        ax.set_title("Gain vs Phase Margin")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        # (3) Gain vs I-Bias
        ax = axes_2d[0, 2]
        if len(orig_plot):
            ax.scatter(orig_plot["target_gain_linear"], orig_plot["target_ibias_ma"], c="lightblue", s=s_2d, marker="o", label="Original Target", alpha=alpha_2d, edgecolors="black", linewidths=1)
            ax.scatter(orig_plot["output_gain_linear"], orig_plot["output_ibias_ma"], c="darkblue", s=s_2d, marker="o", label="Original Output", alpha=alpha_2d, edgecolors="black", linewidths=1)
        if len(morl_plot):
            ax.scatter(morl_plot["target_gain_linear"], morl_plot["target_ibias_ma"], c="lightgreen", s=s_2d, marker="s", label="MORL Target", alpha=alpha_2d, edgecolors="black", linewidths=1)
            ax.scatter(morl_plot["output_gain_linear"], morl_plot["output_ibias_ma"], c="darkgreen", s=s_2d, marker="s", label="MORL Output", alpha=alpha_2d, edgecolors="black", linewidths=1)
        ax.set_xlabel("Gain (V/V)")
        ax.set_ylabel("I-Bias (mA)")
        ax.set_title("Gain vs I-Bias")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        # (4) UGBW vs PM
        ax = axes_2d[1, 0]
        if len(orig_plot):
            ax.scatter(orig_plot["target_ugbw_mhz"], orig_plot["target_pm_deg"], c="lightblue", s=s_2d, marker="o", label="Original Target", alpha=alpha_2d, edgecolors="black", linewidths=1)
            ax.scatter(orig_plot["output_ugbw_mhz"], orig_plot["output_pm_deg"], c="darkblue", s=s_2d, marker="o", label="Original Output", alpha=alpha_2d, edgecolors="black", linewidths=1)
        if len(morl_plot):
            ax.scatter(morl_plot["target_ugbw_mhz"], morl_plot["target_pm_deg"], c="lightgreen", s=s_2d, marker="s", label="MORL Target", alpha=alpha_2d, edgecolors="black", linewidths=1)
            ax.scatter(morl_plot["output_ugbw_mhz"], morl_plot["output_pm_deg"], c="darkgreen", s=s_2d, marker="s", label="MORL Output", alpha=alpha_2d, edgecolors="black", linewidths=1)
        ax.set_xlabel("UGBW (MHz)")
        ax.set_ylabel("PM (deg)")
        ax.set_title("UGBW vs Phase Margin")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        # (5) UGBW vs I-Bias
        ax = axes_2d[1, 1]
        if len(orig_plot):
            ax.scatter(orig_plot["target_ugbw_mhz"], orig_plot["target_ibias_ma"], c="lightblue", s=s_2d, marker="o", label="Original Target", alpha=alpha_2d, edgecolors="black", linewidths=1)
            ax.scatter(orig_plot["output_ugbw_mhz"], orig_plot["output_ibias_ma"], c="darkblue", s=s_2d, marker="o", label="Original Output", alpha=alpha_2d, edgecolors="black", linewidths=1)
        if len(morl_plot):
            ax.scatter(morl_plot["target_ugbw_mhz"], morl_plot["target_ibias_ma"], c="lightgreen", s=s_2d, marker="s", label="MORL Target", alpha=alpha_2d, edgecolors="black", linewidths=1)
            ax.scatter(morl_plot["output_ugbw_mhz"], morl_plot["output_ibias_ma"], c="darkgreen", s=s_2d, marker="s", label="MORL Output", alpha=alpha_2d, edgecolors="black", linewidths=1)
        ax.set_xlabel("UGBW (MHz)")
        ax.set_ylabel("I-Bias (mA)")
        ax.set_title("UGBW vs I-Bias")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        # (6) PM vs I-Bias
        ax = axes_2d[1, 2]
        if len(orig_plot):
            ax.scatter(orig_plot["target_pm_deg"], orig_plot["target_ibias_ma"], c="lightblue", s=s_2d, marker="o", label="Original Target", alpha=alpha_2d, edgecolors="black", linewidths=1)
            ax.scatter(orig_plot["output_pm_deg"], orig_plot["output_ibias_ma"], c="darkblue", s=s_2d, marker="o", label="Original Output", alpha=alpha_2d, edgecolors="black", linewidths=1)
        if len(morl_plot):
            ax.scatter(morl_plot["target_pm_deg"], morl_plot["target_ibias_ma"], c="lightgreen", s=s_2d, marker="s", label="MORL Target", alpha=alpha_2d, edgecolors="black", linewidths=1)
            ax.scatter(morl_plot["output_pm_deg"], morl_plot["output_ibias_ma"], c="darkgreen", s=s_2d, marker="s", label="MORL Output", alpha=alpha_2d, edgecolors="black", linewidths=1)
        ax.set_xlabel("PM (deg)")
        ax.set_ylabel("I-Bias (mA)")
        ax.set_title("Phase Margin vs I-Bias")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "comparison_best20_2d_all_objectives.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {OUT_DIR / 'comparison_best20_2d_all_objectives.png'}")

    # Graph for best 100: 3 point types — Target, Output (Original), Output (MORL); larger markers so points visible
    plot100 = best_100_combined_for_plot.dropna(
        subset=["target_gain_linear", "output_gain_linear", "target_ugbw_mhz", "output_ugbw_mhz",
                "target_pm_deg", "output_pm_deg", "target_ibias_ma", "output_ibias_ma"]
    )
    if len(plot100) > 0:
        o100 = plot100[plot100["method"] == "original"]
        m100 = plot100[plot100["method"] == "morl"]
        s_pt = 45  # visible but not overcrowded for 200 points
        fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))
        fig2.suptitle("Best 100: Target vs Output (Original) vs Output (MORL)\nComparison", fontsize=14, fontweight="bold")
        # (1) Gain vs UGBW
        ax = axes2[0, 0]
        if len(o100):
            ax.scatter(o100["target_gain_linear"], o100["target_ugbw_mhz"], c="lightblue", s=s_pt, marker="o", label="Original Target", alpha=0.85, edgecolors="black", linewidths=0.8)
            ax.scatter(o100["output_gain_linear"], o100["output_ugbw_mhz"], c="darkblue", s=s_pt, marker="o", label="Original Output", alpha=0.85, edgecolors="black", linewidths=0.8)
        if len(m100):
            ax.scatter(m100["target_gain_linear"], m100["target_ugbw_mhz"], c="lightgreen", s=s_pt, marker="s", label="MORL Target", alpha=0.85, edgecolors="black", linewidths=0.8)
            ax.scatter(m100["output_gain_linear"], m100["output_ugbw_mhz"], c="darkgreen", s=s_pt, marker="s", label="MORL Output", alpha=0.85, edgecolors="black", linewidths=0.8)
        ax.set_xlabel("Gain (linear)")
        ax.set_ylabel("UGBW (MHz)")
        ax.set_title("Gain vs UGBW")
        ax.legend()
        ax.grid(True, alpha=0.3)
        # (2) Gain vs PM
        ax = axes2[0, 1]
        if len(o100):
            ax.scatter(o100["target_gain_linear"], o100["target_pm_deg"], c="lightblue", s=s_pt, marker="o", label="Original Target", alpha=0.85, edgecolors="black", linewidths=0.8)
            ax.scatter(o100["output_gain_linear"], o100["output_pm_deg"], c="darkblue", s=s_pt, marker="o", label="Original Output", alpha=0.85, edgecolors="black", linewidths=0.8)
        if len(m100):
            ax.scatter(m100["target_gain_linear"], m100["target_pm_deg"], c="lightgreen", s=s_pt, marker="s", label="MORL Target", alpha=0.85, edgecolors="black", linewidths=0.8)
            ax.scatter(m100["output_gain_linear"], m100["output_pm_deg"], c="darkgreen", s=s_pt, marker="s", label="MORL Output", alpha=0.85, edgecolors="black", linewidths=0.8)
        ax.set_xlabel("Gain (linear)")
        ax.set_ylabel("PM (deg)")
        ax.set_title("Gain vs Phase Margin")
        ax.legend()
        ax.grid(True, alpha=0.3)
        # (3) Target vs Output Gain
        ax = axes2[1, 0]
        if len(o100):
            ax.scatter(o100["target_gain_linear"], o100["target_gain_linear"], c="lightblue", s=s_pt, marker="o", label="Original Target", alpha=0.85, edgecolors="black", linewidths=0.8)
            ax.scatter(o100["target_gain_linear"], o100["output_gain_linear"], c="darkblue", s=s_pt, marker="o", label="Original Output", alpha=0.85, edgecolors="black", linewidths=0.8)
        if len(m100):
            ax.scatter(m100["target_gain_linear"], m100["target_gain_linear"], c="lightgreen", s=s_pt, marker="s", label="MORL Target", alpha=0.85, edgecolors="black", linewidths=0.8)
            ax.scatter(m100["target_gain_linear"], m100["output_gain_linear"], c="darkgreen", s=s_pt, marker="s", label="MORL Output", alpha=0.85, edgecolors="black", linewidths=0.8)
        mn = plot100[["target_gain_linear", "output_gain_linear"]].min().min()
        mx = plot100[["target_gain_linear", "output_gain_linear"]].max().max()
        ax.plot([mn, mx], [mn, mx], "k--", alpha=0.4, label="Target=Output")
        ax.set_xlabel("Target Gain (linear)")
        ax.set_ylabel("Output Gain (linear)")
        ax.set_title("Target vs Output Gain")
        ax.legend()
        ax.grid(True, alpha=0.3)
        # (4) Target vs Output UGBW
        ax = axes2[1, 1]
        if len(o100):
            ax.scatter(o100["target_ugbw_mhz"], o100["target_ugbw_mhz"], c="lightblue", s=s_pt, marker="o", label="Original Target", alpha=0.85, edgecolors="black", linewidths=0.8)
            ax.scatter(o100["target_ugbw_mhz"], o100["output_ugbw_mhz"], c="darkblue", s=s_pt, marker="o", label="Original Output", alpha=0.85, edgecolors="black", linewidths=0.8)
        if len(m100):
            ax.scatter(m100["target_ugbw_mhz"], m100["target_ugbw_mhz"], c="lightgreen", s=s_pt, marker="s", label="MORL Target", alpha=0.85, edgecolors="black", linewidths=0.8)
            ax.scatter(m100["target_ugbw_mhz"], m100["output_ugbw_mhz"], c="darkgreen", s=s_pt, marker="s", label="MORL Output", alpha=0.85, edgecolors="black", linewidths=0.8)
        mn = plot100[["target_ugbw_mhz", "output_ugbw_mhz"]].min().min()
        mx = plot100[["target_ugbw_mhz", "output_ugbw_mhz"]].max().max()
        ax.plot([mn, mx], [mn, mx], "k--", alpha=0.4, label="Target=Output")
        ax.set_xlabel("Target UGBW (MHz)")
        ax.set_ylabel("Output UGBW (MHz)")
        ax.set_title("Target vs Output UGBW")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "comparison_best100_original_vs_morl.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {OUT_DIR / 'comparison_best100_original_vs_morl.png'}")

        # --- 3d_obj for best 100: Gain vs I-Bias, Gain vs UGBW, 3D, I-Bias vs UGBW ---
        fig4 = plt.figure(figsize=(14, 12))
        ax1 = fig4.add_subplot(2, 2, 1)
        ax2 = fig4.add_subplot(2, 2, 2)
        ax3 = fig4.add_subplot(2, 2, 3, projection="3d")
        ax4 = fig4.add_subplot(2, 2, 4)
        s_100 = 45
        alpha_100 = 0.8
        # Original: same shape (o), target=light blue, output=dark blue
        if len(o100):
            ax1.scatter(o100["target_gain_linear"], o100["target_ibias_ma"], c="lightblue", s=s_100, marker="o", label="Original Target", alpha=alpha_100, edgecolors="black", linewidths=0.6)
            ax1.scatter(o100["output_gain_linear"], o100["output_ibias_ma"], c="darkblue", s=s_100, marker="o", label="Original Output", alpha=alpha_100, edgecolors="black", linewidths=0.6)
            ax2.scatter(o100["target_gain_linear"], o100["target_ugbw_mhz"], c="lightblue", s=s_100, marker="o", label="Original Target", alpha=alpha_100, edgecolors="black", linewidths=0.6)
            ax2.scatter(o100["output_gain_linear"], o100["output_ugbw_mhz"], c="darkblue", s=s_100, marker="o", label="Original Output", alpha=alpha_100, edgecolors="black", linewidths=0.6)
            ax3.scatter(o100["target_gain_linear"], o100["target_ibias_ma"], o100["target_ugbw_mhz"], c="lightblue", s=s_100, marker="o", label="Original Target", alpha=alpha_100, edgecolors="black", linewidths=0.6)
            ax3.scatter(o100["output_gain_linear"], o100["output_ibias_ma"], o100["output_ugbw_mhz"], c="darkblue", s=s_100, marker="o", label="Original Output", alpha=alpha_100, edgecolors="black", linewidths=0.6)
            ax4.scatter(o100["target_ibias_ma"], o100["target_ugbw_mhz"], c="lightblue", s=s_100, marker="o", label="Original Target", alpha=alpha_100, edgecolors="black", linewidths=0.6)
            ax4.scatter(o100["output_ibias_ma"], o100["output_ugbw_mhz"], c="darkblue", s=s_100, marker="o", label="Original Output", alpha=alpha_100, edgecolors="black", linewidths=0.6)
        # MORL: same shape (s), target=light green, output=dark green
        if len(m100):
            ax1.scatter(m100["target_gain_linear"], m100["target_ibias_ma"], c="lightgreen", s=s_100, marker="s", label="MORL Target", alpha=alpha_100, edgecolors="black", linewidths=0.6)
            ax1.scatter(m100["output_gain_linear"], m100["output_ibias_ma"], c="darkgreen", s=s_100, marker="s", label="MORL Output", alpha=alpha_100, edgecolors="black", linewidths=0.6)
            ax2.scatter(m100["target_gain_linear"], m100["target_ugbw_mhz"], c="lightgreen", s=s_100, marker="s", label="MORL Target", alpha=alpha_100, edgecolors="black", linewidths=0.6)
            ax2.scatter(m100["output_gain_linear"], m100["output_ugbw_mhz"], c="darkgreen", s=s_100, marker="s", label="MORL Output", alpha=alpha_100, edgecolors="black", linewidths=0.6)
            ax3.scatter(m100["target_gain_linear"], m100["target_ibias_ma"], m100["target_ugbw_mhz"], c="lightgreen", s=s_100, marker="s", label="MORL Target", alpha=alpha_100, edgecolors="black", linewidths=0.6)
            ax3.scatter(m100["output_gain_linear"], m100["output_ibias_ma"], m100["output_ugbw_mhz"], c="darkgreen", s=s_100, marker="s", label="MORL Output", alpha=alpha_100, edgecolors="black", linewidths=0.6)
            ax4.scatter(m100["target_ibias_ma"], m100["target_ugbw_mhz"], c="lightgreen", s=s_100, marker="s", label="MORL Target", alpha=alpha_100, edgecolors="black", linewidths=0.6)
            ax4.scatter(m100["output_ibias_ma"], m100["output_ugbw_mhz"], c="darkgreen", s=s_100, marker="s", label="MORL Output", alpha=alpha_100, edgecolors="black", linewidths=0.6)
        ax1.set_xlabel("Gain (V/V)")
        ax1.set_ylabel("I-Bias (mA)")
        ax1.set_title("Gain vs I-Bias")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2.set_xlabel("Gain (V/V)")
        ax2.set_ylabel("UGBW (MHz)")
        ax2.set_title("Gain vs UGBW")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax3.set_xlabel("Gain (V/V)")
        ax3.set_ylabel("I-Bias (mA)")
        ax3.set_zlabel("UGBW (MHz)")
        ax3.set_title("Gain vs I-Bias vs UGBW (3D)")
        ax3.legend()
        ax4.set_xlabel("I-Bias (mA)")
        ax4.set_ylabel("UGBW (MHz)")
        ax4.set_title("I-Bias vs UGBW")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        fig4.suptitle("Best 100: Target vs Output (Original) vs Output (MORL)\n3d_obj — Gain, I-Bias, UGBW", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "comparison_best100_3d_obj.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {OUT_DIR / 'comparison_best100_3d_obj.png'}")

    print("\nDone. Outputs in:", OUT_DIR)
    print("  - morl_best_per_spec_1000.csv   (best of 10 per spec, 1000 rows)")
    print("  - best_100_original.csv, best_100_morl.csv, best_100_combined.csv")
    print("  - best_20_original.csv, best_20_morl.csv, best_20_combined.csv")
    print("  - comparison_best20_original_vs_morl.png, comparison_best100_original_vs_morl.png")
    print("  - comparison_best20_2d_4objectives.png, comparison_best20_2d_all_objectives.png")
    print("  - comparison_best20_3d_obj.png, comparison_best100_3d_obj.png")
    print("  - comparison_per_spec_fom.png (above diagonal = MORL better)")
    print("  - comparison_full_scatter_paper_style.png (all points across plan)")
    print("  - per_spec_fom_summary.txt")


if __name__ == "__main__":
    main()
