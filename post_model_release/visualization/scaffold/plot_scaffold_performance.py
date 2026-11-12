"""Portable plotting adaptation of make_fig_scaffold_bars_2.py."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---------- paths ----------
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--performance', required=True, type=Path)
parser.add_argument('--ad-performance', required=True, type=Path)
parser.add_argument('--output-dir', required=True, type=Path)
args = parser.parse_args()
PERF = args.performance.expanduser().resolve()
ADP = args.ad_performance.expanduser().resolve()
OUT = args.output_dir.expanduser().resolve()
names = ['FigC_scaffold_bars_heatmap.' + ext for ext in ('pdf','svg','png','tiff')]
names += ['plot_manifest.json', 'caption.txt']
if any((OUT / name).exists() for name in names):
    parser.error('An output file already exists. Choose a new output directory.')

# ---------- style (same as the other figures) ----------
FONT, TICK, LETTER = 8, 7, 11
MIN_N = 20                      # cells with fewer observations are marked with a dagger
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": FONT, "axes.linewidth": 0.6, "pdf.fonttype": 42, "svg.fonttype": "none",
    "xtick.labelsize": TICK, "ytick.labelsize": TICK,
})

REP = {"2d": "2D", "3d": "3D", "morgan": "Morgan FP",
       "2d_3d": "2D + 3D", "2d_3d_morgan": "2D + 3D + Morgan"}
ORDER = list(REP.values())
def short(r): return r.replace(" + ", "\n+ ")

perf = pd.read_csv(PERF); ad = pd.read_csv(ADP)
required = {'Dataset','Representation','Algorithm','Budget','Scaffold_status','Observations','MCC'}
for table, label in ((perf,'performance'),(ad,'AD performance')):
    missing = required - set(table.columns)
    if missing:
        parser.error(f'{label} missing columns: {sorted(missing)}')
    if table[list(required - {'MCC'})].isna().any().any():
        parser.error(f'{label} has missing group labels or counts')
if 'Unique_scaffolds' not in perf or 'AD_status' not in ad:
    parser.error('Unique_scaffolds and AD_status are required in the respective tables')
expected_datasets = {'Single-protein','Cell-based'}
if set(perf.Dataset) != expected_datasets or set(ad.Dataset) != expected_datasets:
    parser.error('Expected Single-protein and Cell-based datasets')
for table in (perf,ad):
    if set(table.Representation) != set(REP):
        parser.error('Expected the five verified molecular representations')
for dataset in sorted(expected_datasets):
    for rep in REP:
        for status in ('Total','Seen','Unseen'):
            group = perf[(perf.Dataset==dataset)&(perf.Representation==rep)&(perf.Scaffold_status==status)]
            if len(group)!=1 or not np.isfinite(group.MCC.iloc[0]):
                parser.error(f'Expected one finite performance row: {dataset}/{rep}/{status}')
        for status in ('Seen','Unseen'):
            for domain in ('Inside','Outside'):
                group=ad[(ad.Dataset==dataset)&(ad.Representation==rep)&(ad.Scaffold_status==status)&(ad.AD_status==domain)]
                if len(group)!=1 or not np.isfinite(group.MCC.iloc[0]):
                    parser.error(f'Expected one finite heatmap row: {dataset}/{rep}/{status}/{domain}')
OUT.mkdir(parents=True, exist_ok=True)
for d in (perf, ad):
    d["Rep"] = d["Representation"].str.lower().map(REP)
    d["Alg"] = d["Algorithm"]
perf = perf[perf.Scaffold_status.isin(["Total", "Seen", "Unseen"])]
ad = ad[ad.Scaffold_status.isin(["Seen", "Unseen"])]
dsets = sorted(perf.Dataset.unique(), key=lambda s: (0 if "single" in s.lower() else 1, s))

C_SEEN, C_UNSEEN = "#0072B2", "#E69F00"
fig = plt.figure(figsize=(7.5, 6.8))
gs = fig.add_gridspec(2, len(dsets), height_ratios=[1, 1.05], hspace=0.60, wspace=0.62,
                      left=0.19, right=0.895, top=0.88, bottom=0.07)
LET_X = 0.065               # panel letters share one x position per column

bars_axes, heat_axes = [], []
# =============== row 1: grouped bars (MCC) ===============
for j, ds in enumerate(dsets):
    ax = fig.add_subplot(gs[0, j])
    p = perf[perf.Dataset == ds]
    w = 0.36
    allv = []
    for k, rep in enumerate(ORDER):
        r = p[p.Rep == rep].set_index("Scaffold_status")
        for off, st, c in ((-w/2, "Seen", C_SEEN), (w/2, "Unseen", C_UNSEEN)):
            v = r.loc[st, "MCC"]; allv.append(v)
            ax.bar(k + off, v, w, color=c, edgecolor="white", linewidth=0.4, zorder=3)
            total = r.loc['Total', 'MCC']
            label_y = max(v, total) + 0.025 if v >= 0 else min(v, total) - 0.025
            ax.text(k + off, label_y, f"{v:.2f}".replace("0.", ".").replace("-.", "\u2212."),
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=5.5, zorder=4)
        ax.hlines(r.loc["Total", "MCC"], k - 0.42, k + 0.42, color="black", lw=1.1, zorder=5)
    ax.axhline(0, color="0.3", lw=0.6, zorder=2)
    ax.set_xticks(range(len(ORDER))); ax.set_xticklabels([short(r) for r in ORDER], fontsize=6.3)
    ax.set_ylabel("External MCC", fontsize=FONT)
    lo, hi = min(allv), max(allv)
    ax.set_ylim(min(lo - 0.10, -0.05), hi + 0.12)
    ax.grid(axis="y", lw=0.3, alpha=0.5, zorder=0)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    s = p[p.Scaffold_status == "Seen"].iloc[0]; u = p[p.Scaffold_status == "Unseen"].iloc[0]
    ax.set_title(f"{ds}\nSeen: n = {int(s.Observations)}, {int(s.Unique_scaffolds)} scaffolds\n"
                 f"Unseen: n = {int(u.Observations)}, {int(u.Unique_scaffolds)} scaffolds",
                 fontsize=7, pad=6)
    bars_axes.append(ax)

# =============== row 2: heatmaps (MCC, scaffold x AD) ===============
cols = [("Seen", "Inside"), ("Seen", "Outside"), ("Unseen", "Inside"), ("Unseen", "Outside")]
cmap = plt.get_cmap("RdBu"); norm = TwoSlopeNorm(vmin=-0.3, vcenter=0, vmax=1.0)
for j, ds in enumerate(dsets):
    ax = fig.add_subplot(gs[1, j])
    a = ad[ad.Dataset == ds]
    M = np.full((len(ORDER), 4), np.nan); N = np.zeros((len(ORDER), 4), int)
    for i, rep in enumerate(ORDER):
        for c, (st, adv) in enumerate(cols):
            t = a[(a.Rep == rep) & (a.Scaffold_status == st) & (a.AD_status == adv)]
            if len(t): M[i, c] = t.MCC.iloc[0]; N[i, c] = int(t.Observations.iloc[0])
    mesh = ax.pcolormesh(np.arange(5), np.arange(len(ORDER) + 1), np.ma.masked_invalid(M),
                         cmap=cmap, norm=norm, edgecolor="white", linewidth=0.8)
    for i in range(len(ORDER)):
        for c in range(4):
            if np.isnan(M[i, c]): continue
            rgba = cmap(norm(M[i, c])); lum = .299*rgba[0] + .587*rgba[1] + .114*rgba[2]
            dag = "\u2020" if N[i, c] < MIN_N else ""
            ax.text(c + .5, i + .5, f"{(0.0 if abs(M[i, c]) < 0.005 else M[i, c]):.2f}{dag}\n(n={N[i, c]})".replace("-", "\u2212"),
                    ha="center", va="center", fontsize=5.6, color="black" if lum > .5 else "white")
    ax.set_xticks([0.5, 1.5, 2.5, 3.5])
    ax.set_xticklabels(["Seen\nInside AD", "Seen\nOutside AD", "Unseen\nInside AD", "Unseen\nOutside AD"], fontsize=6.3)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(ORDER)) + .5)
    ax.set_yticklabels([f"{r}\n({a[a.Rep == r].Alg.iloc[0]}, {int(a[a.Rep == r].Budget.iloc[0])})" for r in ORDER], fontsize=5.8)
    ax.set_ylim(len(ORDER), 0); ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.axvline(2, color="white", lw=3)
    heat_axes.append(ax)
for j in range(len(dsets)):
    xl = heat_axes[j].get_position().x0 - 0.115      # left edge of the panel incl. its labels
    fig.text(xl, bars_axes[j].get_position().y1 + 0.065, "AB"[j], fontsize=LETTER, fontweight="bold")
    fig.text(xl, heat_axes[j].get_position().y1 + 0.045, "CD"[j], fontsize=LETTER, fontweight="bold")
cax = fig.add_axes([0.925, 0.10, 0.012, 0.30])
cb = fig.colorbar(mesh, cax=cax); cb.set_label("External MCC", fontsize=FONT); cb.ax.tick_params(labelsize=TICK)

h = [Patch(color=C_SEEN, label="Seen scaffolds"), Patch(color=C_UNSEEN, label="Unseen scaffolds"),
     Line2D([], [], color="black", lw=1.1, label="All external observations")]
fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.045))
for ext in ("pdf", "svg"): fig.savefig(f"{OUT}/FigC_scaffold_bars_heatmap.{ext}", bbox_inches="tight")
fig.savefig(f"{OUT}/FigC_scaffold_bars_heatmap.png", dpi=200, bbox_inches="tight")
fig.savefig(f"{OUT}/FigC_scaffold_bars_heatmap.tiff", dpi=600, bbox_inches="tight")
print("saved", OUT)
plt.close(fig)
caption = ('Scaffold generalization and applicability-domain-stratified performance. '
    '(A,B) MCC for seen and unseen scaffolds across five final molecular representations; '
    'black markers show MCC for all external observations. '
    '(C,D) MCC by scaffold status and AD membership. n denotes observations; '
    'daggers mark groups with fewer than 20 observations. '
    'The single scaffold-less cell-based observation is excluded from seen/unseen groups '
    'but included in all-external MCC. Algorithm labels show the requested feature budgets.\n')
(OUT/'caption.txt').write_text(caption,encoding='utf-8')
manifest = {'adapted_from':'make_fig_scaffold_bars_2.py',
    'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'inputs':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in (PERF,ADP)],
    'counts':'observations; distinct scaffold counts are reported separately',
    'model_training_performed':False,
    'changes':'Configurable paths, input validation, observation labels, wrapped titles, bar labels clear of total markers, caption and manifest',
    'matplotlib_version':matplotlib.__version__}
(OUT/'plot_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
