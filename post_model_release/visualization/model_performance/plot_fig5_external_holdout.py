"""Plot frozen Total/Inside-AD/Outside-AD metrics; no model fitting.

Adapted from the original make_figB.py. The legacy pr_auc columns contain AP.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", required=True, type=Path, help="Combined frozen AD metrics CSV")
parser.add_argument("--output-dir", required=True, type=Path, help="Separate output directory")
args = parser.parse_args()
CSV = args.input.expanduser().resolve()
OUT = args.output_dir.expanduser().resolve()
expected_outputs = [f"Fig5_external_holdout.{ext}" for ext in ("pdf", "svg", "png", "tiff")]
expected_outputs += ["external_validation_summary.csv", "plot_manifest.json"]
if any((OUT / name).exists() for name in expected_outputs):
    parser.error("An output file already exists. Choose a new output directory.")

# ---------- settings ----------
FONT, TICK, LETTER = 8, 7, 11
YLAB = 6.3                 # row-label size
VOFF = 0.22                # vertical shift between Total / Inside AD / Outside AD
SHOW_CI = True             # 95% CI bars from the *_ci_low / *_ci_high columns
FIG_SIZE = (7.5, 7.4)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": FONT, "axes.linewidth": 0.6, "pdf.fonttype": 42, "svg.fonttype": "none",
    "xtick.labelsize": TICK, "ytick.labelsize": TICK,
})

df = pd.read_csv(CSV)
required = ["Dataset", "representation", "analysis_group", "model", "budget", "rows", "ad_coverage_percent"]
required += [m + suffix for m in ("mcc", "pr_auc", "roc_auc") for suffix in ("", "_ci_low", "_ci_high")]
missing = sorted(set(required) - set(df.columns))
if missing:
    parser.error(f"Input is missing columns: {missing}")
if df[required].isna().any().any():
    parser.error("Missing values in required plot columns.")
representations = {"2d", "3d", "morgan", "2d_3d", "2d_3d_morgan"}
if set(df.representation) != representations:
    parser.error("Expected exactly the five verified molecular representations.")
if set(df.analysis_group) != {"Total", "Inside_AD", "Outside_AD"}:
    parser.error("Expected Total, Inside_AD and Outside_AD groups.")
if df.Dataset.nunique() != 2 or len(df) != 30:
    parser.error("Expected 30 rows: two assay datasets x five representations x three AD groups.")
if df.duplicated(["Dataset", "representation", "analysis_group"]).any():
    parser.error("Duplicate dataset/representation/AD-group combinations.")
numeric = ["budget", "rows", "ad_coverage_percent"] + required[7:]
if not np.isfinite(df[numeric].to_numpy(dtype=float)).all():
    parser.error("Non-finite numerical values in the figure input.")
for metric in ("mcc", "pr_auc", "roc_auc"):
    if (df[metric + "_ci_low"] > df[metric + "_ci_high"]).any():
        parser.error(f"Reversed confidence interval bounds for {metric}.")
OUT.mkdir(parents=True, exist_ok=True)

# the file has both 'representation' (lowercase) and 'Representation'; use the lowercase one
def rep_label(x):
    s = str(x).lower()
    m, d2, d3 = "morgan" in s, "2d" in s, "3d" in s
    if m and d2 and d3: return "2D + 3D + Morgan"   # three-way check first
    if m and d3:        return "Morgan + 3D"
    if d2 and d3:       return "2D + 3D"
    if m:               return "Morgan FP"
    if d3:              return "3D"
    if d2:              return "2D"
    return str(x)
df["Rep"] = df["representation"].map(rep_label)

def group_label(x):
    s = str(x).lower()
    if "out" in s: return "Outside AD"
    if "in" in s:  return "Inside AD"
    return "Total"
df["Grp"] = df["analysis_group"].map(group_label)

def pretty_algo(a):
    a = str(a).lower().replace("-", "_")
    names = {"svm_rbf": "SVM-RBF", "extra_trees": "Extra Trees", "lightgbm": "LightGBM",
             "xgboost": "XGBoost", "random_forest": "Random Forest",
             "knn": "KNN", "logistic": "Logistic"}
    return names.get(a, a.replace("_", " ").title())

ORDER = ["2D", "3D", "Morgan FP", "2D + 3D", "2D + 3D + Morgan"]
reps = [r for r in ORDER if r in df.Rep.unique()]
dsets = sorted(df.Dataset.unique(), key=lambda s: (0 if "single" in str(s).lower() else 1, str(s)))

dup = df.groupby(["Dataset", "Rep", "Grp"]).size()

METRICS = [("MCC", "mcc", "mcc_ci_low", "mcc_ci_high"),
           ("AP", "pr_auc", "pr_auc_ci_low", "pr_auc_ci_high"),
           ("ROC-AUC", "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high")]
STYLE = {"Total": ("#0072B2", "o", -VOFF),
         "Inside AD": ("#E69F00", "s", 0.0),
         "Outside AD": ("#009E73", "^", VOFF)}

fig, axes = plt.subplots(3, len(dsets), figsize=FIG_SIZE)
axes = np.asarray(axes).reshape(3, len(dsets))

for i, (mname, col, lo, hi) in enumerate(METRICS):
    for j, ds in enumerate(dsets):
        ax = axes[i, j]
        d = df[df.Dataset == ds]
        ylabels = []
        for k, rep in enumerate(reps):
            r = d[d.Rep == rep]
            if r.empty:
                ylabels.append(rep); continue
            tot = r[r.Grp == "Total"]
            r0 = tot.iloc[0] if len(tot) else r.iloc[0]
            ylabels.append(f"{rep}\n({pretty_algo(r0['model'])}, {int(r0['budget'])})")
            ax.axhspan(k - 0.5, k + 0.5, color="0.96" if k % 2 == 0 else "white", zorder=0)
            for g, (c, m, off) in STYLE.items():
                t = r[r.Grp == g]
                if t.empty: continue
                v = t[col].iloc[0]
                if SHOW_CI:
                    ax.hlines(k + off, t[lo].iloc[0], t[hi].iloc[0], color=c, lw=0.9,
                              alpha=0.85, zorder=2)
                ax.scatter(v, k + off, color=c, marker=m, s=20, zorder=3,
                           edgecolor="white", linewidth=0.3)
        ax.set_yticks(range(len(reps)))
        ax.set_yticklabels(ylabels, fontsize=YLAB)
        ax.set_ylim(len(reps) - 0.5, -0.5)
        ax.set_xlabel(mname, fontsize=FONT)
        ax.grid(axis="x", lw=0.3, alpha=0.5)
        ax.tick_params(axis="y", length=0)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        if i == 0: ax.set_title(str(ds), fontsize=FONT + 0.5)
        ax.text(-0.50, 1.04, "ABCDEF"[i * len(dsets) + j], transform=ax.transAxes,
                fontsize=LETTER, fontweight="bold")

h = [Line2D([], [], color="w", marker=m, markerfacecolor=c, ms=6, label=g)
     for g, (c, m, _) in STYLE.items()]
fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, fontsize=7.5,
           bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.96))
for ext in ("pdf", "svg"):
    fig.savefig(f"{OUT}/Fig5_external_holdout.{ext}", bbox_inches="tight")
fig.savefig(f"{OUT}/Fig5_external_holdout.tiff", dpi=600, bbox_inches="tight")
fig.savefig(f"{OUT}/Fig5_external_holdout.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# summary table for the supplement
cols = ["Dataset", "Rep", "model", "budget", "Grp", "rows", "ad_coverage_percent",
        "mcc", "mcc_ci_low", "mcc_ci_high", "pr_auc", "pr_auc_ci_low", "pr_auc_ci_high",
        "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high"]
df[cols].to_csv(f"{OUT}/external_validation_summary.csv", index=False)
manifest = {
    "input": str(CSV),
    "input_sha256": hashlib.sha256(CSV.read_bytes()).hexdigest(),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "rows_plotted": len(df),
    "precision_recall_metric": "Average precision (AP), retained in legacy pr_auc columns",
    "count_unit": "Observations in rows; feature budget labels are requested budgets",
    "confidence_intervals": "Precomputed 95% compound-cluster percentile bootstrap intervals",
    "model_training_performed": False,
    "adaptation": "Original make_figB.py layout; configurable paths, AP label, input checks, PNG and provenance manifest",
    "numpy_version": np.__version__,
    "pandas_version": pd.__version__,
    "matplotlib_version": matplotlib.__version__,
}
(OUT / "plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
(OUT / "caption_Fig5.txt").write_text("Figure 5. Source-held-out external validation of the final models in the single-protein and cell-based datasets. Panels A and B show Matthews correlation coefficient (MCC), C and D show average precision (AP), and E and F show ROC-AUC, respectively. Points indicate performance on the total external cohort, inside the applicability domain (AD), and outside AD. Horizontal bars show precomputed 95% compound-cluster bootstrap confidence intervals. Labels identify the molecular representation, algorithm and requested feature budget. AP was calculated using average_precision_score.\n", encoding="utf-8")
print("Saved to", OUT)
