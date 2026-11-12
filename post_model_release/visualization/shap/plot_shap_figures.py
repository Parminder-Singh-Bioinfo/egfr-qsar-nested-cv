import os, re, json, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

# ---------------- explicit publication paths ----------------
import argparse
import hashlib
from pathlib import Path
parser = argparse.ArgumentParser(description="Replay the two SHAP figures from frozen tables; no model fitting or SHAP calculation.")
parser.add_argument("--input-root", required=True)
parser.add_argument("--output-dir", required=True)
args = parser.parse_args()
ROOT = str(Path(args.input_root).resolve())
OUT = str(Path(args.output_dir).resolve())
if Path(OUT).exists() and any(Path(OUT).iterdir()):
    raise FileExistsError("Output directory must be new or empty: " + OUT)
Path(OUT).mkdir(parents=True, exist_ok=True)

# per-dataset layout: left/right margins and the gap between the two columns (tuned to label length)
DATASETS = [
    dict(tag="Single_protein_Morgan", title="Single-protein, Extra Trees, Morgan FP",
         folder=f"{ROOT}/01_Single_Protein_Morgan", morgan=True,
         left=0.12, right=0.97, wspace=0.30),
    dict(tag="Cell_based_3D", title="Cell-based, SVM-RBF, 3D descriptors",
         folder=f"{ROOT}/02_Cell_Based_3D", morgan=False,
         left=0.20, right=0.93, wspace=0.50),
]

# (label, file under the dataset folder, colour, marker)
SUBGROUPS = [
    ("All observations", "03_Classification_Errors/TP_vs_FN_feature_comparison.csv", "#222222", "o"),
    ("Inside AD", "04_AD_Analysis/Inside_AD/TP_vs_FN_feature_comparison.csv", "#0072B2", "s"),
    ("Outside AD", "04_AD_Analysis/Outside_AD/TP_vs_FN_feature_comparison.csv", "#E69F00", "s"),
    ("Seen scaffold", "05_Scaffold_Analysis/Scaffold_Seen/TP_vs_FN_feature_comparison.csv", "#009E73", "^"),
    ("Unseen scaffold", "05_Scaffold_Analysis/Scaffold_Unseen/TP_vs_FN_feature_comparison.csv", "#CC79A7", "^"),
]

MIN_GROUP = 5                  # subgroup shown only if TP and FN both have >= 5 observations
TOP_N, TOP_DIFF = 20, 10
FONT, TICK, LETTER = 8, 7, 11
POS, NEG = "#B2182B", "#2166AC"         # red / blue
CMAP = "coolwarm"                       # red = high descriptor value, blue = low
TP_C, FN_C = "#0072B2", "#E69F00"
FAM = {"PaDEL": "#0072B2", "Mordred": "#009E73", "MoRSE": "#CC79A7", "RDF": "#D55E00",
       "GETAWAY": "#E69F00", "AUTOCORR": "#56B4E9", "WHIM": "#F0E442", "Other": "0.6"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": FONT, "axes.linewidth": 0.6, "pdf.fonttype": 42, "svg.fonttype": "none",
    "xtick.labelsize": TICK, "ytick.labelsize": TICK,
})


# ---------------- helpers ----------------
def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def pick(d, must, avoid=()):
    for c in d.columns:
        n = norm(c)
        if all(m in n for m in must) and not any(a in n for a in avoid):
            return c
    return None


def pretty(f, morgan):
    """Morgan: trailing bit number, e.g. rdkit_morgan_r2_2048bit_0959 -> 'Bit 0959'."""
    f = str(f)
    if morgan:
        m = re.search(r"(\d+)$", f)
        return f"Bit {int(m.group(1)):04d}" if m else f
    return (f.replace("rdkit_3d_", "RDKit ").replace("mordred_3d_", "Mordred ")
             .replace("padel_3d_", "PaDEL ").replace("_", " "))


def family(f, morgan):
    if morgan:
        return "Morgan"
    f = str(f)
    if f.startswith("padel"): return "PaDEL"
    if f.startswith("mordred"): return "Mordred"
    if "GETAWAY" in f: return "GETAWAY"
    if "MORSE" in f: return "MoRSE"
    if "RDF" in f: return "RDF"
    if "AUTOCORR" in f: return "AUTOCORR"
    if "WHIM" in f: return "WHIM"
    return "Other"


def is_binary(x):
    u = np.unique(x[~np.isnan(x)])
    return set(u.tolist()) <= {0.0, 1.0}


def get_counts(d):
    """TP and FN counts: this export uses n_first (TP) and n_second (FN)."""
    if {"n_first", "n_second"} <= set(d.columns):
        return int(d["n_first"].iloc[0]), int(d["n_second"].iloc[0])
    for a, b in [("n_tp", "n_fn"), ("ntp", "nfn"), ("tp_n", "fn_n"), ("tp_count", "fn_count")]:
        if a in d.columns and b in d.columns:
            return int(d[a].iloc[0]), int(d[b].iloc[0])
    return None, None


def load_imp(folder):
    d = pd.read_csv(f"{folder}/01_Global_SHAP/shap_molecular_importance.csv")
    f = pick(d, ["feature"]); a = pick(d, ["meanabsshap"]); s = pick(d, ["meanshap"], avoid=["abs"])
    if None in (f, a, s):
        raise KeyError(f"importance file columns: {list(d.columns)}")
    out = pd.DataFrame({"feature": d[f].astype(str), "mean_abs": d[a].astype(float),
                        "mean": d[s].astype(float)})
    return out.sort_values("mean_abs", ascending=False).reset_index(drop=True)


def load_cmp(path):
    """Table indexed by feature (tp, fn, delta = FN - TP) with n_tp and n_fn.
    Returns None if the file is missing, has no TP_vs_FN rows, or either group has < MIN_GROUP."""
    label = os.path.relpath(path, ROOT)
    if not os.path.exists(path):
        print(f"  not generated: {label}")
        return None
    d = pd.read_csv(path)
    if "comparison" in d.columns:
        d = d[d["comparison"].astype(str).str.contains("TP_vs_FN", case=False)]
    if d.empty:
        print(f"  no TP_vs_FN rows: {label}")
        return None
    fcol = pick(d, ["feature"])
    tpc = pick(d, ["tp", "mean", "signed"]) or pick(d, ["tp", "mean", "shap"], avoid=["abs"])
    fnc = pick(d, ["fn", "mean", "signed"]) or pick(d, ["fn", "mean", "shap"], avoid=["abs"])
    if None in (fcol, tpc, fnc):
        raise KeyError(f"{label}: needed feature/TP/FN columns. Columns: {list(d.columns)}")
    n_tp, n_fn = get_counts(d)
    if n_tp is None:
        raise KeyError(f"{label}: no TP/FN count columns. Columns: {list(d.columns)}")
    if n_tp < MIN_GROUP or n_fn < MIN_GROUP:
        print(f"  omitted (TP {n_tp}, FN {n_fn} < {MIN_GROUP}): {label}")
        return None
    out = pd.DataFrame({"tp": d[tpc].astype(float).values, "fn": d[fnc].astype(float).values},
                       index=d[fcol].astype(str).values)
    out = out[~out.index.duplicated()]
    out["delta"] = out.fn - out.tp                   # FN - TP, as labelled in panel D
    return out, n_tp, n_fn


def prepare(ds):
    print(f"\n{ds['title']}")
    subs = {name: load_cmp(os.path.join(ds["folder"], rel)) for name, rel, _, _ in SUBGROUPS}
    if subs["All observations"] is None:
        raise FileNotFoundError(f"Overall TP vs FN file missing or too small in {ds['folder']}")
    feats = subs["All observations"][0].delta.abs().sort_values(ascending=False).index[:TOP_DIFF].tolist()
    shown = [(name, col, mk) for name, _, col, mk in SUBGROUPS if subs[name] is not None]
    return dict(feats=feats, subs=subs, shown=shown)


def check_metadata(ds):
    path = Path(ds["folder"]) / "shap_metadata.json"
    m = json.loads(path.read_text())
    print(ds["tag"], json.dumps(m)[:600])


def save(fig, name):
    for ext in ("pdf", "svg"):
        fig.savefig(f"{OUT}/{name}.{ext}", bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def label_left_edge(fig, ax):
    """Left edge (figure fraction) of the y tick labels: panel letters are aligned to it."""
    r = fig.canvas.get_renderer()
    xs = [t.get_window_extent(r).x0 for t in ax.get_yticklabels() if t.get_text()]
    return fig.transFigure.inverted().transform((min(xs), 0))[0]


# ---------------- one figure per dataset ----------------
def make_figure(ds, prep, xlim):
    folder, morgan = ds["folder"], ds["morgan"]
    feats, subs, shown = prep["feats"], prep["subs"], prep["shown"]

    imp = load_imp(folder).head(TOP_N).copy()
    imp["label"] = imp.feature.map(lambda f: pretty(f, morgan))
    vals = pd.read_csv(f"{folder}/01_Global_SHAP/shap_molecular_values.csv")
    inp = pd.read_csv(f"{folder}/01_Global_SHAP/shap_molecular_inputs.csv")

    fig = plt.figure(figsize=(7.5, 7.4))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1.15], height_ratios=[1, 1],
                  left=ds["left"], right=ds["right"], top=0.93, bottom=0.16,
                  wspace=ds["wspace"], hspace=0.30)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axC, axD = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    # ---- A: global importance (bar length = mean |SHAP|) ----
    y = np.arange(len(imp))[::-1]
    if morgan:
        bar_cols = [POS if v > 0 else NEG for v in imp["mean"]]
    else:
        bar_cols = [FAM[family(f, morgan)] for f in imp.feature]
    axA.barh(y, imp.mean_abs, color=bar_cols, height=0.72)
    axA.set_yticks(y); axA.set_yticklabels(imp.label, fontsize=5.8)
    axA.set_xlabel("Mean |SHAP| (probability units)", fontsize=FONT)
    axA.set_title(f"Global importance (top {TOP_N})", fontsize=FONT)
    axA.grid(axis="x", lw=0.3, alpha=0.5)
    for sp in ("top", "right"): axA.spines[sp].set_visible(False)
    if morgan:
        axA.legend(handles=[Patch(color=POS, label="Mean signed SHAP > 0"),
                            Patch(color=NEG, label="Mean signed SHAP < 0")],
                   title="Colour", title_fontsize=6, fontsize=6, frameon=False, loc="lower right")
    else:
        fams = [f for f in FAM if f in set(family(x, morgan) for x in imp.feature)]
        axA.legend(handles=[Patch(color=FAM[f], label=f) for f in fams],
                   title="Descriptor family", title_fontsize=6, fontsize=6, frameon=False, loc="lower right")

    # ---- B: beeswarm (red = high value / bit present, blue = low / absent) ----
    rng = np.random.default_rng(1)
    n = len(imp)
    for k, f in enumerate(imp.feature):
        if f not in vals.columns or f not in inp.columns:
            continue
        sv = vals[f].to_numpy(float); fv = inp[f].to_numpy(float)
        yy = (n - 1 - k) + rng.uniform(-0.3, 0.3, len(sv))
        if morgan and is_binary(fv):
            axB.scatter(sv, yy, c=np.where(fv > 0.5, POS, NEG), s=5, lw=0, alpha=0.8)
        else:
            lo, hi = np.nanpercentile(fv, [1, 99])
            c = np.clip((fv - lo) / (hi - lo + 1e-12), 0, 1)
            axB.scatter(sv, yy, c=c, cmap=CMAP, vmin=0, vmax=1, s=5, lw=0, alpha=0.8)
    axB.axvline(0, color="0.3", lw=0.6)
    axB.set_yticks(range(n)); axB.set_yticklabels(imp.label[::-1], fontsize=5.8)
    axB.set_ylim(-0.7, n - 0.3)
    axB.set_xlabel("SHAP value (impact on P(active))", fontsize=FONT)
    axB.set_title("SHAP beeswarm", fontsize=FONT)
    axB.grid(axis="x", lw=0.3, alpha=0.5)
    for sp in ("top", "right"): axB.spines[sp].set_visible(False)
    if morgan:
        axB.legend(handles=[Patch(color=POS, label="Bit present"), Patch(color=NEG, label="Bit absent")],
                   fontsize=6, frameon=False, loc="lower right")
    else:
        sm = plt.cm.ScalarMappable(cmap=CMAP, norm=Normalize(0, 1))
        cb = fig.colorbar(sm, ax=axB, fraction=0.05, pad=0.03, aspect=25)
        cb.set_ticks([0.02, 0.98]); cb.set_ticklabels(["Low", "High"])
        cb.ax.tick_params(labelsize=TICK, length=0)
        cb.set_label("Feature value", fontsize=FONT, labelpad=-4)

    # ---- C: actual mean SHAP for TP and FN (all observations) ----
    cmp, n_tp, n_fn = subs["All observations"]
    d = cmp.loc[feats]
    yc = np.arange(len(d))
    for i, (tpv, fnv) in enumerate(zip(d.tp, d.fn)):
        axC.plot([tpv, fnv], [i, i], color="0.7", lw=1, zorder=1)
    axC.scatter(d.tp, yc, color=TP_C, s=22, zorder=3, label=f"TP (n = {n_tp})")
    axC.scatter(d.fn, yc, color=FN_C, marker="s", s=22, zorder=3, label=f"FN (n = {n_fn})")
    axC.axvline(0, color="0.3", lw=0.6)
    axC.set_ylim(len(d) - 0.5, -0.5)
    axC.set_yticks(yc); axC.set_yticklabels([pretty(f, morgan) for f in d.index], fontsize=5.8)
    axC.set_xlabel("Mean SHAP value (TP or FN)", fontsize=FONT)
    axC.set_title("TP vs FN, all observations", fontsize=FONT)
    axC.grid(axis="x", lw=0.3, alpha=0.5)
    for sp in ("top", "right"): axC.spines[sp].set_visible(False)
    # legend placed below the axes so it never covers data points
    axC.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.20),
               ncol=2, handletextpad=0.3, columnspacing=1.2)

    # ---- D: FN - TP shift by subgroup (one lane per subgroup) ----
    k_count = len(shown)
    offs = np.linspace(0.32, -0.32, k_count) if k_count > 1 else [0.0]
    for i, f in enumerate(feats):
        axD.axhline(i, color="0.93", lw=0.6, zorder=0)
        for k, (name, col, mk) in enumerate(shown):
            t = subs[name][0]
            if f in t.index:
                axD.scatter(t.loc[f, "delta"], i + offs[k], s=16, color=col, marker=mk,
                            zorder=3, edgecolor="white", linewidth=0.3)
    axD.axvline(0, color="0.3", lw=0.6)
    axD.set_xlim(*xlim)
    axD.set_ylim(len(feats) - 0.5, -0.5)
    axD.set_yticks(range(len(feats)))
    axD.set_yticklabels([pretty(f, morgan) for f in feats], fontsize=5.8)
    axD.set_xlabel("Mean SHAP, FN − TP", fontsize=FONT)
    axD.set_title("FN − TP shift by subgroup", fontsize=FONT)
    axD.grid(axis="x", lw=0.3, alpha=0.5)
    for sp in ("top", "right"): axD.spines[sp].set_visible(False)
    handles = [Line2D([], [], marker=mk, ls="", color=col, markersize=4,
                      label=f"{name} (TP n = {subs[name][1]}, FN n = {subs[name][2]})")
               for name, col, mk in shown]
    axD.legend(handles=handles, fontsize=5.8, frameon=False, loc="upper center",
               bbox_to_anchor=(0.42, -0.20), ncol=1, handletextpad=0.3)

    # ---- panel letters, aligned to the left edge of each panel's y tick labels ----
    fig.canvas.draw()
    for ax, s in zip((axA, axB, axC, axD), "ABCD"):
        p = ax.get_position()
        fig.text(label_left_edge(fig, ax), p.y1 + 0.010, s, fontsize=LETTER,
                 fontweight="bold", va="bottom", ha="left")
    fig.suptitle(ds["title"], fontsize=FONT + 1, y=0.985)
    save(fig, f"Fig_{ds['tag']}")


def write_caption(ds, prep):
    subs = prep["subs"]
    shown = [f"{name} (TP n = {subs[name][1]}, FN n = {subs[name][2]})" for name, _, _ in prep["shown"]]
    omitted = [name for name, _, _, _ in SUBGROUPS if subs[name] is None]
    bit_note = ("Morgan fingerprints: radius 2, 2048 bits; bit labels are the bit index. "
                if ds["morgan"] else "")
    text = (
        f"{ds['title']}. SHAP values were computed for the external set with the frozen model and are "
        f"in units of predicted probability P(active). {bit_note}"
        f"(A) Top {TOP_N} features by mean |SHAP|. For Morgan bits, bar colour shows the sign of the mean "
        f"signed SHAP; for 3D descriptors, colour shows descriptor family. "
        f"(B) SHAP values for the same features. For Morgan bits, red marks bit present and blue bit absent; "
        f"for 3D descriptors, red marks high and blue low feature values. "
        f"(C) Mean SHAP for true positives (TP) and false negatives (FN), all observations, for the {TOP_DIFF} "
        f"features with the largest absolute FN − TP difference. "
        f"(D) FN − TP difference in mean SHAP for the same features in each subgroup. "
        f"Subgroups shown: {'; '.join(shown)}. "
        f"Comparisons not shown ({', '.join(omitted) if omitted else 'none'}) were not generated or had fewer "
        f"than {MIN_GROUP} TP or FN observations. "
        f"Applicability-domain and scaffold subgroups overlap, and are descriptive comparisons rather than "
        f"independent effects. Small groups should be interpreted as exploratory.\n"
    )
    with open(f"{OUT}/caption_{ds['tag']}.txt", "w") as fh:
        fh.write(text)


def shift_limit(preps):
    vals = []
    for p in preps:
        for name, _, _ in p["shown"]:
            t = p["subs"][name][0]
            vals += [t.loc[f, "delta"] for f in p["feats"] if f in t.index]
    m = max(abs(v) for v in vals) if vals else 0.05
    return (-m * 1.25, m * 1.25)


# ---------------- run ----------------
if __name__ == "__main__":
    for ds in DATASETS:
        check_metadata(ds)
    preps = {ds["tag"]: prepare(ds) for ds in DATASETS}
    XLIM = shift_limit(preps.values())               # same x-range in both panels D
    for ds in DATASETS:
        make_figure(ds, preps[ds["tag"]], XLIM)
        write_caption(ds, preps[ds["tag"]])
    print("\nSaved to", OUT)

    files = sorted(Path(ROOT).rglob("*.csv")) + sorted(Path(ROOT).rglob("shap_metadata.json"))
    manifest = {"inputs": [{"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files],
                "model_training_performed": False, "shap_recalculated": False,
                "subgroup_delta": "FN minus TP", "minimum_observations_per_group": MIN_GROUP,
                "outputs": [p.name for p in sorted(Path(OUT).iterdir())]}
    (Path(OUT) / "plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")