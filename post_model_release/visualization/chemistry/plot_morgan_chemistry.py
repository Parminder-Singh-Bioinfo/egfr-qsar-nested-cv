import os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

# ---------------- explicit release paths ----------------
import argparse, hashlib, json
from pathlib import Path
parser = argparse.ArgumentParser(description="Replay Morgan chemical interpretation composite from saved tables and structure images")
parser.add_argument("--input-root", required=True)
parser.add_argument("--output-dir", required=True)
args = parser.parse_args()
ROOT = str(Path(args.input_root).resolve())
TABLES = f"{ROOT}/tables"
IMG_DIR = f"{ROOT}/figures/individual_bits"
OUT = str(Path(args.output_dir).resolve())
if Path(OUT).exists() and any(Path(OUT).iterdir()):
    raise FileExistsError("Output directory must be new or empty")
Path(OUT).mkdir(parents=True, exist_ok=True)

FONT, LETTER = 8, 11
ROW_H, LABEL_H = 0.74, 0.42          # inches per bit row / per group-label band
FIG_W = 7.5
TP_C, FN_C = "#0072B2", "#E69F00"
ENV_COLS = ["#5B8DB8", "#E69F00", "#009E73"]      # most frequent / 2nd / 3rd (high contrast)
OTHER_C = "0.80"
BAND_C = "0.955"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": FONT, "axes.linewidth": 0.6, "pdf.fonttype": 42, "svg.fonttype": "none",
    "xtick.labelsize": 7, "ytick.labelsize": 7,
})


# ---------------- helpers ----------------
def load_image(path):
    """Read a PNG, put it on white, drop the tiny caption RDKit writes under the molecule, trim the border."""
    im = mpimg.imread(path)
    if im.ndim == 2:
        im = np.stack([im] * 3, axis=-1)
    if im.shape[2] == 4:
        a = im[..., 3:4]
        im = im[..., :3] * a + (1 - a)
    H = im.shape[0]
    rows = (im < 0.96).any(axis=2).any(axis=1)
    runs, start = [], None
    for i, v in enumerate(rows):
        if v and start is None: start = i
        if (not v) and start is not None: runs.append((start, i - 1)); start = None
    if start is not None: runs.append((start, H - 1))
    if len(runs) >= 2:
        (l0, l1), prev_end = runs[-1], runs[-2][1]
        if (l1 - l0) < 0.06 * H and (l0 - prev_end) > 0.01 * H:
            im = im[: prev_end + 1]
    ink = (im < 0.96).any(axis=2)
    if ink.any():
        ys, xs = np.where(ink)
        pad = 6
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, im.shape[0])
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, im.shape[1])
        im = im[y0:y1, x0:x1]
    return im


def save(fig, name):
    for ext in ("pdf", "svg"):
        fig.savefig(f"{OUT}/{name}.{ext}", bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


# ---------------- load tables ----------------
prev = pd.read_csv(f"{TABLES}/tp_fn_bit_prevalence.csv")
envf = pd.read_csv(f"{TABLES}/environment_frequency.csv")
rep = pd.read_csv(f"{TABLES}/representative_environment_images.csv")
div = pd.read_csv(f"{TABLES}/bit_environment_diversity.csv")

pw = prev.pivot(index="bit", columns="group", values="prevalence_percent")
npw = prev.groupby("group")["n"].first()
n_tp, n_fn = int(npw["TP"]), int(npw["FN"])
pw["diff"] = pw["TP"] - pw["FN"]
tp_bits = pw[pw["diff"] > 0].sort_values("diff", ascending=False).index.astype(int).tolist()
fn_bits = pw[pw["diff"] <= 0].sort_values("diff", ascending=True).index.astype(int).tolist()
print(f"TP n = {n_tp}, FN n = {n_fn}")
print("TP-enriched bits:", tp_bits)
print("FN-enriched bits:", fn_bits)

tot = envf.groupby("bit")["environment_occurrences"].sum()
envf["share"] = 100 * envf["environment_occurrences"] / envf["bit"].map(tot)
n_distinct = div.set_index("bit")["distinct_environment_representations"].to_dict()
rep = rep.sort_values(["bit", "environment_rank"])
rep["key"] = "r" + rep["radius"].astype(str) + ":" + rep["fragment_smiles"]
share_lookup = envf.set_index(["bit", "environment_key"])["share"].to_dict()

# row layout: a label band before each group, then its bits
layout = [("label", "Higher prevalence among true positives (TP)")] + [("bit", b) for b in tp_bits]
if fn_bits:
    layout += [("label", "Higher prevalence among false negatives (FN)")] + [("bit", b) for b in fn_bits]
heights = [LABEL_H / ROW_H if t == "label" else 1.0 for t, _ in layout]
total_u = sum(heights)
centers, acc = [], 0.0
for h in heights:
    centers.append(acc + h / 2)
    acc += h
n_bits = len(tp_bits) + len(fn_bits)


# ---------------- figure builder ----------------
def build(n_env, name, width_ratios, title_suffix=""):
    fig_h = ROW_H * total_u + 2.15
    fig = plt.figure(figsize=(FIG_W, fig_h))
    top_f, bot_f = 1 - 0.95 / fig_h, 0.85 / fig_h
    gs = GridSpec(len(layout), 2 + n_env, figure=fig, hspace=0.0, wspace=0.07,
                  height_ratios=heights, width_ratios=width_ratios,
                  left=0.095, right=0.965, top=top_f, bottom=bot_f)

    # ---- A: prevalence of each bit in TP and FN compounds ----
    axA = fig.add_subplot(gs[:, 0])
    axA.patch.set_alpha(0)
    ybit = {b: c for (t, b), c in zip(layout, centers) if t == "bit"}
    for (t, v), c, h in zip(layout, centers, heights):
        if t == "label":
            pass  # full-width group heading is placed in the label band below
        else:
            axA.plot([pw.loc[v, "FN"], pw.loc[v, "TP"]], [c, c], color="0.72", lw=1, zorder=1)
    bl = list(ybit)
    axA.scatter([pw.loc[b, "TP"] for b in bl], [ybit[b] for b in bl], color=TP_C, s=28, zorder=3)
    axA.scatter([pw.loc[b, "FN"] for b in bl], [ybit[b] for b in bl], color=FN_C, marker="s", s=28, zorder=3)
    axA.set_ylim(total_u, 0)
    axA.set_yticks([ybit[b] for b in bl]); axA.set_yticklabels([f"Bit {b:04d}" for b in bl], fontsize=6.8)
    axA.set_xlim(-3, 103)
    axA.set_xlabel("Compounds containing bit (%)", fontsize=FONT)
    axA.grid(axis="x", lw=0.3, alpha=0.5)
    for sp in ("top", "right"): axA.spines[sp].set_visible(False)
    axA.legend(handles=[
        Line2D([], [], marker="o", ls="", color=TP_C, markersize=5, label=f"True positives (n = {n_tp})"),
        Line2D([], [], marker="s", ls="", color=FN_C, markersize=5, label=f"False negatives (n = {n_fn})")],
        frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=1)

    # ---- B: structures, caption line below each ----
    cell_w_in = None
    tile_axes = []
    for (t, v), c, h in zip(layout, centers, heights):
        if t != "bit":
            continue
        ridx = [i for i, (tt, vv) in enumerate(layout) if tt == "bit" and vv == v][0]
        r = rep[rep.bit == v]
        for k in range(n_env):
            ax = fig.add_subplot(gs[ridx, 1 + k])
            ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            row = r[r.environment_rank == k + 1]
            if len(row):
                row = row.iloc[0]
                path = f"{IMG_DIR}/{row.image_file}"
                if os.path.exists(path):
                    img = load_image(path)
                    pos = ax.get_position()
                    cw, ch = pos.width * FIG_W, pos.height * fig_h
                    box_w, box_h = cw * 0.97, ch * 0.80 * 0.97
                    s_ = min(box_w / img.shape[1], box_h / img.shape[0])
                    w_f, h_f = img.shape[1] * s_ / cw, img.shape[0] * s_ / ch
                    x0, y1 = 0.5 - w_f / 2, 0.98
                    ax.imshow(img, extent=(x0, x0 + w_f, y1 - h_f, y1), aspect="auto", interpolation="lanczos")
                else:
                    print("  missing image:", path)
                sh = share_lookup.get((v, row.key))
                stxt = "" if sh is None else ("<1%" if sh < 1 else f"{sh:.0f}%")
                cap = " \u00b7 ".join([x for x in (stxt, f"r{int(row.radius)}", str(row.group)) if x])
                ax.text(0.5, 0.0, cap, transform=ax.transAxes, fontsize=6.4, ha="center", va="bottom", color="0.15")
            if ridx == 1:
                ax.set_title(["Most observations", "2nd", "3rd"][k], fontsize=7, pad=3)
            tile_axes.append(ax)

    # ---- C: composition of each bit (share of environment occurrences) ----
    axC = fig.add_subplot(gs[:, 1 + n_env])
    axC.patch.set_alpha(0)
    for b in bl:
        c = ybit[b]
        g = envf[envf.bit == b].sort_values("share", ascending=False)
        left = 0.0
        for k in range(3):
            if k < len(g):
                w = g.share.iloc[k]
                axC.barh(c, w, left=left, color=ENV_COLS[k], height=0.58, edgecolor="white", linewidth=0.5)
                if w >= 12:
                    value_label = f"{w:.1f}" if 99.5 <= w < 100 else f"{w:.0f}"
                    axC.text(left + w / 2, c, value_label, ha="center", va="center", fontsize=5.8,
                             color="white" if k != 1 else "black")
                left += w
        if left < 99.99:
            axC.barh(c, 100 - left, left=left, color=OTHER_C, height=0.58, edgecolor="white",
                     linewidth=0.5, hatch="////")
        axC.text(103, c, f"{n_distinct.get(b, len(g))}", va="center", ha="left", fontsize=6.4, color="0.2")
    axC.set_xlim(0, 112)
    axC.set_xticks([0, 50, 100])
    axC.set_ylim(total_u, 0)
    axC.set_yticks([])
    axC.set_xlabel("Share of environment\noccurrences (%)", fontsize=FONT - 0.5)
    for sp in ("top", "right", "left"): axC.spines[sp].set_visible(False)
    axC.legend(handles=[Patch(color=ENV_COLS[0], label="Most frequent"), Patch(color=ENV_COLS[1], label="2nd"),
                        Patch(color=ENV_COLS[2], label="3rd"),
                        Patch(facecolor=OTHER_C, edgecolor="white", hatch="////", label="Remaining environments")],
               frameon=False, fontsize=5.9, loc="upper center", bbox_to_anchor=(0.45, -0.105), ncol=2,
               handlelength=1.1, columnspacing=0.8)

    # ---- group bands behind the label rows, spanning the full width ----
    pa = axA.get_position()
    for (t, v), c, h in zip(layout, centers, heights):
        if t == "label":
            y_top = pa.y1 - ((c - h / 2) / total_u) * pa.height
            fig.add_artist(Rectangle((0.0, y_top - h / total_u * pa.height), 1.0, h / total_u * pa.height,
                                     transform=fig.transFigure, facecolor=BAND_C, edgecolor="none", zorder=-5))
            y_mid = y_top - (h / total_u * pa.height) / 2
            fig.text(pa.x0, y_mid, v, ha="left", va="center", fontsize=7,
                     style="italic", color="0.2")

    # ---- letters and headers ----

    pb, pc = tile_axes[0].get_position(), axC.get_position()
    top_y = pa.y1 + 0.030

    # Separate two-line Panel C heading to prevent collision with Panel B.
    # Each panel title is constrained to its own horizontal bounds.
    fig.text(0.012, top_y, "A", fontsize=LETTER, fontweight="bold", va="bottom")
    fig.text(pb.x0 - 0.012, top_y, "B", fontsize=LETTER, fontweight="bold", va="bottom")
    fig.text(pc.x0 - 0.014, top_y, "C", fontsize=LETTER, fontweight="bold", va="bottom")
    fig.text((pa.x0 + pa.x1) / 2, top_y, "Bit prevalence, TP vs FN",
             fontsize=7.3, ha="center", va="bottom")
    fig.text((pb.x0 + tile_axes[n_env - 1].get_position().x1) / 2, top_y + 0.017,
             "Atom environments", fontsize=7.3, ha="center", va="bottom")
    fig.text((pb.x0 + tile_axes[n_env - 1].get_position().x1) / 2, top_y + 0.003,
             "(share of occurrences · radius · compound group)",
             fontsize=6.0, ha="center", va="bottom", color="0.3")
    c_title = fig.text((pc.x0 + pc.x1) / 2 + 0.012, top_y + 0.005,
             "Distribution of\natom-centered environments",
             fontsize=7.0, linespacing=1.05, ha="center", va="bottom")
    fig.text(pc.x1 - 0.002, pa.y1 + 0.003, "n env.", fontsize=6, ha="right", va="bottom", color="0.3")
    fig.suptitle("Single-protein, Extra Trees, Morgan FP (radius 2, 2048 bits)" + title_suffix,
                 fontsize=FONT + 1, y=0.985)
    # Render and check header widths against their assigned panel boundaries.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    title_bb = c_title.get_window_extent(renderer).transformed(fig.transFigure.inverted())
    if title_bb.x0 < pc.x0 or title_bb.x1 > pc.x1:
        # Scale down only when necessary; guarantees the long title fits Panel C.
        available = pc.x1 - pc.x0 - 0.012
        c_title.set_fontsize(c_title.get_fontsize() * available / title_bb.width)
        c_title.set_position(((pc.x0 + pc.x1) / 2, top_y + 0.005))
    h_mm = fig_h / FIG_W * 170
    print(f"{name}: {FIG_W:.1f} x {fig_h:.1f} in; at 170 mm width it is {h_mm:.0f} mm tall"
          + ("  (WARNING: above the usual 225 mm limit; lower ROW_H)" if h_mm > 225 else ""))
    save(fig, name)


build(2, "Fig_Morgan_chemical_interpretation",
      width_ratios=[1.45, 1.8, 1.8, 1.35])

# ---------------- caption text ----------------
cap = f"""Figure legend (draft)
(A) Percentage of external-set compounds containing each Morgan bit, among true positives (TP; actives predicted active, n = {n_tp}) and false negatives (FN; actives predicted inactive, n = {n_fn}). Bits are grouped by whether they are more prevalent among TP or among FN.
(B) Representative atom-centred environments for two environment representations with the largest observation counts for each bit. Orange highlighting marks the atoms of the environment; r0, r1 and r2 give the Morgan radius (0, 1 or 2 bonds from the central atom). Text under each structure gives the environment's share of all occurrences of that bit, its radius, and the group (TP, FN, TN or FP) of the compound drawn. The drawn compound is one example, not a summary of the group.
(C) Composition of each hashed bit. Stacked bars show proportions of environment occurrences (an occurrence is one atom environment in one compound; a compound can contribute several), not proportions of compounds. Colours give the most frequent, second and third most frequent environment; the hatched grey segment is all remaining environments. Numbers inside segments are percentages. 'n env.' is the number of distinct environment representations, which include stereochemical and aromaticity variants of the same fragment.
The Morgan fingerprint used radius 2 and 2048 bits, so different environments can map to the same bit.
"""
with open(f"{OUT}/caption_Morgan_chemical_interpretation.txt", "w") as fh:
    fh.write(cap)
print("Saved to", OUT)


files = sorted(Path(TABLES).glob("*.csv")) + sorted(Path(IMG_DIR).glob("*.png"))
(Path(OUT) / "plot_manifest.json").write_text(json.dumps({"inputs": [{"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files], "model_training_performed": False}, indent=2) + "\n")