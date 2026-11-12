import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import argparse, json, hashlib
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output-dir", required=True)
args = parser.parse_args()
CSV = str(Path(args.input).resolve())
OUT = str(Path(args.output_dir).resolve())
if Path(OUT).exists() and any(Path(OUT).iterdir()):
    raise FileExistsError("Output directory must be new or empty")
Path(OUT).mkdir(parents=True, exist_ok=True)

# ---------- settings ----------
FONT, TICK, LETTER = 8, 7, 11
DODGE = 0.10                  # wider shift so error bars don't touch (0.08-0.12)
LOW_ALGOS = []      # ignored when setting y-range; set [] to show everything
FIG_SIZE = (7.5, 7.2)         # taller panels = less crowding
LINE_ALPHA = 0.55             # connecting lines lighter, so error bars stand out

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": FONT, "axes.linewidth": 0.6, "pdf.fonttype": 42, "svg.fonttype": "none",
    "xtick.labelsize": TICK, "ytick.labelsize": TICK,
})

df = pd.read_csv(CSV)
df = df.rename(columns={"Feature_Budget": "Budget", "Mean_Inner_CV_MCC": "MCC",
                        "SD_Inner_CV_MCC": "SD"})
df["Selected"] = df["Selected_Final_Model"].astype(str).str.upper().eq("TRUE")

def rep_label(x):
    s = str(x).lower()
    m, d2, d3 = "morgan" in s, "2d" in s, "3d" in s
    if m and d2 and d3: return "2D + 3D + Morgan"
    if m and d3:        return "Morgan + 3D"
    if d2 and d3:       return "2D + 3D"
    if m:               return "Morgan FP"
    if d3:              return "3D"
    if d2:              return "2D"
    return str(x)
df["Rep"] = df["Representation"].map(rep_label)

ORDER = ["2D", "3D", "Morgan FP", "2D + 3D", "2D + 3D + Morgan"]
reps = [r for r in ORDER if r in df.Rep.unique()]
dsets = sorted(df.Dataset.unique(), key=lambda s: (0 if "single" in str(s).lower() else 1, str(s)))
algos = sorted(df.Algorithm.unique())
budgets = sorted(df.Budget.unique())
xmap = {b: i for i, b in enumerate(budgets)}

df = df.sort_values(["Selected", "MCC"], ascending=[False, False])
best = df.drop_duplicates(["Dataset", "Rep", "Algorithm", "Budget"]).copy()
best["X"] = best.Budget.map(xmap)
sel = df[df.Selected].drop_duplicates(["Dataset", "Rep"]).copy()
sel["X"] = sel.Budget.map(xmap)
print("Rows kept:", len(best), "| selected configs:", len(sel))

colors = {"Extra Trees":"#0072B2", "KNN":"#E69F00", "LightGBM":"#009E73", "Logistic Regression":"#CC79A7", "Random Forest":"#D55E00", "SVM-RBF":"#56B4E9", "XGBoost":"#7F7F7F"}
mk = {"Extra Trees":"o", "KNN":"s", "LightGBM":"^", "Logistic Regression":"D", "Random Forest":"v", "SVM-RBF":"P", "XGBoost":"X"}
if not set(algos).issubset(colors):
    raise ValueError("Unexpected algorithm: set an explicit manuscript style")
offs = dict(zip(algos, (np.arange(len(algos)) - (len(algos) - 1) / 2) * DODGE))

fig, axes = plt.subplots(len(dsets), len(reps), figsize=FIG_SIZE, sharex=True)
axes = np.asarray(axes).reshape(len(dsets), len(reps))

for i, ds in enumerate(dsets):
    d = best[best.Dataset == ds]
    core = d[~d.Algorithm.isin(LOW_ALGOS)]
    lo, hi = (core.MCC - core.SD).min(), (core.MCC + core.SD).max()   # range includes SD
    pad = (hi - lo) * 0.05
    ylim = (lo - pad, hi + pad)
    for j, rep in enumerate(reps):
        ax = axes[i, j]
        s = d[d.Rep == rep]
        for a in algos:
            t = s[s.Algorithm == a].sort_values("X")
            if t.empty: continue
            x = t.X + offs[a]
            ax.plot(x, t.MCC, color=colors[a], lw=0.6, alpha=LINE_ALPHA, zorder=2)
            ax.errorbar(x, t.MCC, yerr=t.SD, fmt=mk[a], color=colors[a], ms=3,
                        markeredgecolor="white", markeredgewidth=0.3,
                        elinewidth=0.7, capsize=1.5, capthick=0.7, zorder=3)
        b = sel[(sel.Dataset == ds) & (sel.Rep == rep)]
        if len(b):
            ax.scatter(b.X.iloc[0] + offs[b.Algorithm.iloc[0]], b.MCC.iloc[0], marker="*",
                       s=85, color="#1b2a49", edgecolor="white", linewidth=0.5, zorder=6)
        ax.set_ylim(*ylim)
        ax.set_xlim(-0.6, len(budgets) - 0.4)
        ax.set_xticks(range(len(budgets)))
        ax.set_xticklabels(budgets, fontsize=6, rotation=90)
        ax.grid(axis="y", lw=0.3, alpha=0.5)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        if i == 0: ax.set_title(rep, fontsize=FONT)
        ax.tick_params(labelleft=True)
        if j == 0: ax.set_ylabel("Mean development-CV MCC", fontsize=FONT)
        if i == len(dsets) - 1: ax.set_xlabel("Feature budget", fontsize=FONT)
    axes[i, 0].text(-0.62, 1.06, "AB"[i], transform=axes[i, 0].transAxes,
                    fontsize=LETTER, fontweight="bold")
    axes[i, -1].text(1.08, 0.5, str(ds), transform=axes[i, -1].transAxes,
                     rotation=270, va="center", fontsize=FONT)
    below = d[(d.MCC + d.SD) < ylim[0]].groupby("Algorithm").MCC.agg(["min", "max"])
    if len(below):
        print(f"\n{ds}: below plotted range ({ylim[0]:.3f}):\n{below.round(3)}")

h = [Line2D([], [], color=colors[a], marker=mk[a], ms=4, lw=0.8, label=a) for a in algos]
h.append(Line2D([], [], color="#1b2a49", marker="*", ms=8, lw=0, label="Selected configuration"))
fig.legend(handles=h, loc="upper center", ncol=len(h), frameon=False, fontsize=7,
           columnspacing=1.0, handletextpad=0.3, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 0.97, 0.94))
for ext in ("pdf", "svg"):
    fig.savefig(f"{OUT}/Fig4_development_cv.{ext}", bbox_inches="tight")
fig.savefig(f"{OUT}/Fig4_development_cv.tiff", dpi=600, bbox_inches="tight")
fig.savefig(f"{OUT}/Fig4_development_cv.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved to", OUT)

(Path(OUT) / "plot_manifest.json").write_text(json.dumps({"input": CSV, "sha256": hashlib.sha256(Path(CSV).read_bytes()).hexdigest(), "model_training_performed": False}, indent=2) + "\n")
