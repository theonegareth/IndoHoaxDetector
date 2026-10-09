"""Publication figures for the revised paper (answers Q18).

300 dpi PNG + vector PDF, sized to IEEE column widths (3.5 in single, 7.16 in
double). Palette is the validated categorical default, slots 1-3, which passes
all-pairs CVD and normal-vision separation in light mode; the one contrast WARN
on aqua is relieved by direct labels, which every bar carries.

Re-runnable: figures that depend on the GPU queue simply include whichever
models have finished.
"""
import os, json, glob
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

S = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(S, "figures")
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d6d2"

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": GRID, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.labelcolor": INK,
    "xtick.major.size": 0, "ytick.major.size": 0,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "{}.{}".format(name, ext)))
    plt.close(fig)
    print("  wrote", name)


def bare(ax, xgrid=True):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)


def load(n):
    p = os.path.join(S, n)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


san = load("out_classical_sanitised.json")
dl2 = load("out_deleaked2.json")
lk = load("out_leakage_baseline.json")
cross = load("out_cross_source.json")            # original corpus: labelled as such where used
crs = load("out_cross_source_sanitised.json")     # sanitised corpus: what the paper reports
lenb = load("out_length_baseline.json")


def model_rows():
    """(name, f1, train_seconds) for every model on the sanitised corpus."""
    rows = []
    if san:
        for m, r in san["results"].items():
            rows.append((m, r["test_f1_weighted"], r["total_train_time_s"]))
    ib = load("out_indobert_sanitised_seed42.json")
    if ib:
        rows.append(("IndoBERT (stemmed)", ib["f1_weighted"], ib["training_time_s"]))
    for f in sorted(glob.glob(os.path.join(S, "out_hf_*.json"))):
        r = json.load(open(f, encoding="utf-8"))
        # "IndoBERT" is the headline run, which uses minimal preprocessing (Section IV-B)
        pretty = {"indobert_minimal": "IndoBERT", "indobertweet": "IndoBERTweet",
                  "xlmr": "XLM-R (frozen emb.)", "mbert": "mBERT"}.get(r["tag"])
        if pretty:
            rows.append((pretty, r["f1_weighted"], r["training_time_s"]))
    return rows


# ---- Fig 1: preprocessing pipeline ---------------------------------------
def fig_pipeline():
    steps = ["Raw corpus\n68,139 docs", "Deduplicate\n63,000", "Sanitise\nverdict tags",
             "Clean & stem\nSastrawi", "TF-IDF /\ntokeniser", "Train &\nevaluate"]
    fig, ax = plt.subplots(figsize=(7.16, 1.35))
    ax.set_xlim(0, len(steps)); ax.set_ylim(0, 1); ax.axis("off")
    for i, s in enumerate(steps):
        c = ORANGE if i == 2 else BLUE
        ax.add_patch(FancyBboxPatch((i + 0.06, 0.22), 0.88, 0.56,
                                    boxstyle="round,pad=0.015,rounding_size=0.06",
                                    fc="white", ec=c, lw=1.1))
        ax.text(i + 0.5, 0.5, s, ha="center", va="center", fontsize=7, color=INK, linespacing=1.4)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((i + 0.95, 0.5), (i + 1.04, 0.5),
                                         arrowstyle="-|>", mutation_scale=7, color=INK2, lw=0.8))
    ax.text(2.5, 0.06, "new in this revision", ha="center", fontsize=6.5,
            color=ORANGE, style="italic")
    save(fig, "fig1_preprocessing_pipeline")


# ---- Fig 2: model comparison ---------------------------------------------
def fig_models():
    # Dot plot, not bars: the values sit in a narrow band near 1.0, and a bar
    # chart there would need a truncated baseline, which misencodes magnitude.
    rows = sorted(model_rows(), key=lambda r: r[1])
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(3.5, 0.34 * len(rows) + 0.6))
    yy = np.arange(len(rows))
    vals = [r[1] for r in rows]
    lo = min(vals) - 0.012
    hi = 1.0 + (1.0 - lo) * 0.20
    for i, (name, v, _) in enumerate(rows):
        c = ORANGE if ("BERT" in name or "XLM" in name) else BLUE
        ax.plot([lo, v], [i, i], color=GRID, lw=0.7, zorder=1)
        ax.scatter(v, i, s=42, color=c, zorder=3, edgecolor="white", lw=0.8)
        ax.text(v + (hi - lo) * 0.022, i, "{:.4f}".format(v), va="center",
                ha="left", fontsize=7, color=INK)
    ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(lo, hi); ax.set_xlabel("Weighted F1-score")
    bare(ax)
    save(fig, "fig3_model_comparison")


# ---- Fig 3: accuracy vs training cost ------------------------------------
def fig_efficiency():
    # short names here; the model-comparison figure carries the XLM-R disclosure
    rows = [(n.replace(" (frozen emb.)", ""), f, t) for n, f, t in model_rows()]
    if not rows:
        return
    # merge points that coincide on screen, clustering on visual distance rather
    # than rounded values: SVM and LogReg differ in the 4th decimal but are one dot
    xs = [np.log10(max(t, 1e-9)) for _, _, t in rows]
    ys = [f1 for _, f1, _ in rows]
    xr = max(max(xs) - min(xs), 1e-9)
    yr = max(max(ys) - min(ys), 1e-9)
    pts = []
    for (name, f1, t), x, y in zip(rows, xs, ys):
        for d in pts:
            if abs(x - d["x"]) / xr < 0.035 and abs(y - d["y"]) / yr < 0.035:
                d["names"].append(name)
                break
        else:
            pts.append({"names": [name], "f1": f1, "t": t, "x": x, "y": y})
    pts.sort(key=lambda d: d["t"])

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for d in pts:
        c = ORANGE if any(s in " ".join(d["names"]) for s in ("BERT", "XLM")) else BLUE
        ax.scatter(d["t"], d["f1"], s=34, color=c, zorder=3, edgecolor="white", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Training time (s, log scale)")
    ax.set_ylabel("Weighted F1-score")
    ts = [d["t"] for d in pts]
    ax.set_xlim(min(ts) / 3.2, max(ts) * 5)            # room for the end labels
    lo, hi = min(d["f1"] for d in pts), max(d["f1"] for d in pts)
    pad = max((hi - lo) * 0.28, 0.006)
    ax.set_ylim(lo - pad, min(1.004, hi + pad))

    # Side labels, vertically centred on their own dot. Right of the dot by
    # default, left for dots in the right part of the axis; flip sides if
    # another dot sits in the label's path. Beside-the-dot placement cannot be
    # misread as belonging to a neighbour, which above/below placement could.
    FS, GAP = 6.3, 6                                    # font pt, offset pt
    px = fig.dpi / 72.0
    disp = [ax.transData.transform((d["t"], d["f1"])) for d in pts]
    x0, x1 = ax.transData.transform((ax.get_xlim()[0], lo))[0], ax.transData.transform((ax.get_xlim()[1], lo))[0]
    for k, d in enumerate(pts):
        label = " / ".join(d["names"])
        w_px = len(label) * FS * 0.55 * px
        (dx, dy) = disp[k]

        def blocked(side):
            a, b = (dx + GAP * px, dx + GAP * px + w_px) if side > 0 else (dx - GAP * px - w_px, dx - GAP * px)
            if a < x0 or b > x1:
                return True
            return any(j != k and a - 4 * px <= disp[j][0] <= b + 4 * px
                       and abs(disp[j][1] - dy) < FS * px for j in range(len(pts)))

        side = -1 if (dx - x0) / (x1 - x0) > 0.6 else 1
        if blocked(side) and not blocked(-side):
            side = -side
        ax.annotate(label, (d["t"], d["f1"]), textcoords="offset points",
                    xytext=(GAP * side, 0), ha="left" if side > 0 else "right",
                    va="center", fontsize=FS, color=INK2)
    bare(ax)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    save(fig, "fig4_efficiency_tradeoff")


# ---- Fig 4: McNemar significance matrix ----------------------------------
def fig_significance():
    p = os.path.join(S, "out_mcnemar_sanitised.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    # order by accuracy so tied models form n.s. blocks on the diagonal -> tiers are visible
    acc = {}
    for _, r in df.iterrows():
        acc[r.model_a], acc[r.model_b] = r.acc_a, r.acc_b
    names = sorted(acc, key=lambda k: -acc[k])
    n = len(names)
    M = np.full((n, n), np.nan)
    for _, r in df.iterrows():
        i, j = names.index(r.model_a), names.index(r.model_b)
        v = -np.log10(max(r.p_holm, 1e-160))
        M[i, j] = M[j, i] = v
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=60)
    thr = -np.log10(0.05)
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=7, color=GRID)
            elif not np.isnan(M[i, j]):
                sig = M[i, j] > thr
                ax.text(j, i, "*" if sig else "n.s.", ha="center", va="center",
                        fontsize=8 if sig else 6,
                        color="white" if M[i, j] > 30 else INK,
                        fontweight="bold" if sig else "normal")
    short = [x.replace(" (seed 42)", "").replace("Logistic Regression", "LogReg")
              .replace("Random Forest", "RF").replace("Naive Bayes", "NB") for x in names]
    ax.set_xticks(range(n)); ax.set_xticklabels(short, rotation=40, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(short)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.4)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("$-\\log_{10}$ (Holm-adjusted $p$)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5); cb.outline.set_visible(False)
    ax.set_title("* significant at $\\alpha$=0.05  ·  n.s. = not significant",
                 fontsize=7, color=INK2, pad=6)
    save(fig, "fig5_significance")


# ---- Fig 5: feature importance (diverging) -------------------------------
def fig_features():
    if not dl2:
        return
    a = dl2["A_aggressive_all_sources"]
    h = a["top_hoax_features"][:10][::-1]
    f = a["top_factual_features"][:10]
    terms = [t for t, _ in f] + [t for t, _ in h]
    vals = [v for _, v in f] + [v for _, v in h]
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    yy = np.arange(len(terms))
    cols = [BLUE if v < 0 else ORANGE for v in vals]
    ax.barh(yy, vals, height=0.66, color=cols)
    ax.set_yticks(yy); ax.set_yticklabels(terms, fontsize=7)
    ax.axvline(0, color=INK2, lw=0.7)
    ax.set_xlabel("Logistic Regression coefficient")
    bare(ax)
    ax.text(0.02, 0.98, "← factual", transform=ax.transAxes, fontsize=7,
            color=BLUE, va="top", fontweight="bold")
    ax.text(0.98, 0.02, "hoax →", transform=ax.transAxes, fontsize=7,
            color=ORANGE, ha="right", fontweight="bold")
    save(fig, "fig7_feature_importance")


# ---- Fig 6: why sanitisation was necessary -------------------------------
def fig_leakage():
    if not (lk and dl2 and san and cross):
        return
    svm_o = cross["A_random"]["f1_weighted"]
    rows = [
        ("Verdict-tag regex\n(no training)", lk["f1_weighted"], ORANGE),
        ("SVM, original corpus", svm_o, ORANGE),
        ("SVM, sanitised", san["results"]["SVM"]["test_f1_weighted"], BLUE),
        ("SVM, sanitised +\nunseen outlets", dl2["A_aggressive_all_sources"]["cross_source"]["f1_weighted"], BLUE),
        ("Length only, original", lenb["original"]["f1_weighted"], AQUA),
        ("Length only, sanitised", lenb["sanitised"]["f1_weighted"], AQUA),
    ][::-1]
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    yy = np.arange(len(rows))
    lo, hi = 0.55, 1.06
    for i, (lab, v, c) in enumerate(rows):
        ax.plot([lo, v], [i, i], color=GRID, lw=0.7, zorder=1)
        ax.scatter(v, i, s=42, color=c, zorder=3, edgecolor="white", lw=0.8)
        ax.text(v + 0.012, i, "{:.3f}".format(v), va="center", ha="left",
                fontsize=7, color=INK)
    ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.set_xlim(lo, hi); ax.set_xlabel("Weighted F1-score")
    bare(ax)
    save(fig, "fig2_leakage_ladder")


# ---- Fig 7: leave-one-source-out -----------------------------------------
def fig_loso():
    if not (crs and "C_leave_one_source_out" in crs):
        return
    d = crs["C_leave_one_source_out"]
    items = sorted(d.items(), key=lambda kv: kv[1]["recall_when_unseen"])
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    yy = np.arange(len(items))
    for i, (s, r) in enumerate(items):
        a, b = r["recall_when_unseen"], r["recall_in_distribution"]
        ax.plot([a, b], [i, i], color=GRID, lw=1.6, zorder=1, solid_capstyle="round")
        ax.scatter(b, i, s=26, color=BLUE, zorder=3, edgecolor="white", lw=0.7)
        ax.scatter(a, i, s=26, color=ORANGE, zorder=3, edgecolor="white", lw=0.7)
    labels = [s.replace("_cleaned", "").replace("_hoaks", "").replace("_", " ") for s, _ in items]
    ax.set_yticks(yy); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Recall on the held-out outlet")
    ax.set_xlim(0.5, 1.02)
    bare(ax)
    ax.scatter([], [], s=26, color=BLUE, label="outlet seen in training")
    ax.scatter([], [], s=26, color=ORANGE, label="outlet unseen")
    # upper left is the only region free of marks (top rows all sit near 1.0)
    ax.legend(frameon=False, loc="upper left", fontsize=6.5, handletextpad=0.3,
              borderaxespad=0.4)
    save(fig, "fig6_leave_one_source_out")


if __name__ == "__main__":
    print("figures ->", OUT)
    for fn in (fig_pipeline, fig_models, fig_efficiency, fig_significance,
               fig_features, fig_leakage, fig_loso):
        try:
            fn()
        except Exception as e:
            print("  SKIP {}: {}".format(fn.__name__, e))
    print("done")
