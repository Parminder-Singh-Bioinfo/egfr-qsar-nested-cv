import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "pdf.fonttype": 42,
})

df = pd.read_csv(CSV, encoding="utf-8")
df = df.rename(columns={"Representation": "Feature", "Mean_MCC": "MCC", "SD_MCC": "SD"})

DESIRED = ["2D", "3D", "Morgan", "2D + 3D", "2D + 3D + Morgan"]
feats = [f for f in DESIRED if f in df.Feature.unique()] + \
        [f for f in df.Feature.unique() if f not in DESIRED]
algos = sorted(df.Algorithm.unique())
budgets = sorted(df.Budget.unique())
datasets = list(df.Dataset.unique())

CMAP = "viridis"
vmin, vmax = df.MCC.min(), df.MCC.max()
GAP = 0.6
nb, na = len(budgets), len(algos)

# label styling: change these to taste
GROUP_WEIGHT = "normal"     # feature-type names (2D, 3D, ...)
GROUP_SIZE = 8.5
DATASET_WEIGHT = "normal"   # "Single-protein" etc.
DATASET_SIZE = 8.5

def draw(ax, d, panel_letter=None, title=None):
    centers, x0 = [], 0
    for feat in feats:
        pv = (d[d.Feature == feat].pivot_table(index="Algorithm", columns="Budget", values="MCC")
              .reindex(index=algos, columns=budgets))
        xe = np.arange(nb + 1) + x0
        ye = np.arange(na + 1)
        mesh = ax.pcolormesh(xe, ye, pv.values, cmap=CMAP, vmin=vmin, vmax=vmax,
                             edgecolor="white", linewidth=0.5)
        for i in range(na):
            for j in range(nb):
                v = pv.values[i, j]
                if np.isnan(v):
                    continue
                rgba = plt.get_cmap(CMAP)((v - vmin) / (vmax - vmin))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                ax.text(x0 + j + 0.5, i + 0.5, f"{v:.2f}".lstrip("0"),
                        ha="center", va="center", fontsize=5.2,
                        color="black" if lum > 0.55 else "white")
        centers.append((x0 + nb / 2, feat))
        x0 += nb + GAP
        ax._last_mesh = mesh

    ticks, labels, x0 = [], [], 0
    for _ in feats:
        ticks += [x0 + j + 0.5 for j in range(nb)]
        labels += [str(b) for b in budgets]
        x0 += nb + GAP
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=6, rotation=90)
    ax.set_yticks(np.arange(na) + 0.5)
    ax.set_yticklabels([a.replace("_", " ") for a in algos], fontsize=7)
    ax.set_xlim(-0.1, x0 - GAP + 0.1)
    ax.set_ylim(na, 0)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    for cx, feat in centers:
        ax.text(cx, -0.45, feat, ha="center", va="bottom",
                fontsize=GROUP_SIZE, fontweight=GROUP_WEIGHT)
        ax.plot([cx - nb / 2 + 0.1, cx + nb / 2 - 0.1], [-0.25, -0.25],
                color="0.5", lw=0.6, clip_on=False)
    if title:
        ax.set_ylabel(title, fontsize=DATASET_SIZE, fontweight=DATASET_WEIGHT, labelpad=6)
    if panel_letter:
        ax.text(-0.13, 1.12, panel_letter, transform=ax.transAxes,
                fontsize=11, fontweight="bold")   # panel letters stay bold (journal convention)

# ---------- combined figure ----------
fig, axes = plt.subplots(len(datasets), 1, figsize=(7.1, 2.9 * len(datasets)),
                         constrained_layout=True)
axes = np.atleast_1d(axes)
for k, (ax, ds) in enumerate(zip(axes, datasets)):
    draw(ax, df[df.Dataset == ds], panel_letter="AB"[k] if k < 2 else None, title=ds)
axes[-1].set_xlabel("Feature budget (within each representation)", fontsize=8)
cb = fig.colorbar(axes[0]._last_mesh, ax=axes, orientation="vertical", fraction=0.02, pad=0.01)
cb.set_label("Mean inner-CV MCC")
for ext in ("pdf", "svg"):
    fig.savefig(f"{OUT}/Fig2_inner_cv.{ext}", bbox_inches="tight")
fig.savefig(f"{OUT}/Fig2_inner_cv.tiff", dpi=600, bbox_inches="tight")
fig.savefig(f"{OUT}/Fig2_inner_cv.png", dpi=300, bbox_inches="tight")
plt.close(fig)


(Path(OUT) / "plot_manifest.json").write_text(json.dumps({"input": CSV, "sha256": hashlib.sha256(Path(CSV).read_bytes()).hexdigest(), "model_training_performed": False}, indent=2) + "\n")
