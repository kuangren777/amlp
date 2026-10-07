"""Method figure (Figure 1) of the AMLP paper, drawn in code.
Usage: python3 analysis/make_method_fig.py  ->  outputs/figs/method.pdf (+ method.png preview)
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans",
                     "mathtext.fontset": "dejavusans"})
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT.parent / "outputs" / "figs"
FS = 8
W, H = 6.5, 4.55

C_HOLD, C_SEL, C_CAL = "#E8A33D", "#5FAE7B", "#5B8DD6"
C_BOX, C_EDGE, C_TXT = "#F4F5F7", "#6B7280", "#1F2937"
C_PANEL = "#FFFFFF"

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


def box(x, y, w, h, text="", fc=C_BOX, ec=C_EDGE, fs=FS, lw=0.8, r=0.04, weight="normal", color=C_TXT, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=color, weight=weight, zorder=3, linespacing=1.15)


def arrow(p, q, color=C_EDGE, lw=1.0, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls, zorder=4,
                                 shrinkA=0, shrinkB=0))


def panel(x, y, w, h, num, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=C_PANEL, ec="#9CA3AF", lw=0.9, zorder=1))
    ax.text(x + 0.08, y + h - 0.13, f"{num}  {title}", ha="left", va="center", fontsize=FS + 0.5,
            weight="bold", color=C_TXT, zorder=3)


def chip(x, y, text, c, w=0.62, h=0.19):
    box(x, y, w, h, text, fc=c, ec=c, fs=FS - 0.5, r=0.05, color="white", weight="bold")


# ---------------- panel positions (snake: 1 2 / 4 3) ----------------
TOP_Y, TOP_H = 2.38, 2.1
BOT_Y, BOT_H = 0.05, 2.18
X1, W1 = 0.05, 2.45
X2, W2 = 2.65, 3.8

# ---------------- 1 Task folds ----------------
panel(X1, TOP_Y, W1, TOP_H, "①", "Task folds")
ax.text(X1 + W1 / 2, TOP_Y + 1.68, "tasks of each suite", ha="center", va="center", fontsize=FS, color=C_TXT)
bx, bw, by, bh = X1 + 0.15, 2.15, TOP_Y + 1.18, 0.38
segs = [("holdout", 0.5, C_HOLD), ("selection", 0.25, C_SEL), ("calibration", 0.25, C_CAL)]
# draw: calibration and selection left, holdout right so non-holdout is contiguous
order = [("holdout", 0.5, C_HOLD), ("selection", 0.25, C_SEL), ("calibration", 0.25, C_CAL)]
cx = bx
centers = {}
for name, f, c in order:
    ww = bw * f
    ax.add_patch(Rectangle((cx, by), ww, bh, fc=c, ec="white", lw=1.2, zorder=2))
    centers[name] = (cx + ww / 2, cx, cx + ww)
    cx += ww
ax.text(centers["selection"][0], by + bh / 2, "sel.", ha="center", va="center", fontsize=FS, color="white", weight="bold", zorder=3)
ax.text(centers["calibration"][0], by + bh / 2, "cal.", ha="center", va="center", fontsize=FS, color="white", weight="bold", zorder=3)
ax.text(centers["holdout"][0], by + bh / 2, "holdout", ha="center", va="center", fontsize=FS, color="white", weight="bold", zorder=3)
ax.text(centers["holdout"][0], by - 0.14, "novel tasks", ha="center", va="center", fontsize=FS, color=C_TXT)
ax.text((centers["selection"][1] + centers["calibration"][2]) / 2, by - 0.14, "non-holdout", ha="center", va="center", fontsize=FS, color=C_TXT)
ax.annotate("", xy=(centers["selection"][1] + 0.02, by - 0.04), xytext=(centers["calibration"][2] - 0.02, by - 0.04),
            arrowprops=dict(arrowstyle="<->", color=C_EDGE, lw=0.8), zorder=4)
ax.text(X1 + W1 / 2, TOP_Y + 0.62, "task-disjoint folds,\nseed fixed before any run", ha="center", va="center",
        fontsize=FS, color=C_TXT, linespacing=1.15)
ax.text(X1 + W1 / 2, TOP_Y + 0.22, r"no holdout run in any policy", ha="center", va="center", fontsize=FS, color=C_TXT)

# ---------------- 2 Allowlist family ----------------
panel(X2, TOP_Y, W2, TOP_H, "②", "Allowlist family")
rx, rw = X2 + 0.12, 1.2
box(rx, TOP_Y + 1.18, rw, 0.55, "benign runs\nof non-holdout\ntasks", fs=FS)
box(rx, TOP_Y + 0.28, rw, 0.45, "mined pool", fs=FS)
arrow((rx + rw / 2, TOP_Y + 1.18), (rx + rw / 2, TOP_Y + 0.73))
# feed from folds
arrow((X1 + W1 - 0.15, TOP_Y + 1.38), (rx, TOP_Y + 1.45), color="#4B5563", lw=1.4)
tx, tw = X2 + 1.62, 2.06
box(tx, TOP_Y + 1.0, tw, 0.76, "", fc="#EEF3FB", ec=C_CAL)
ax.text(tx + tw / 2, TOP_Y + 1.62, "tool layer  $T(p)$", ha="center", va="center", fontsize=FS, weight="bold", color=C_TXT, zorder=3)
ax.text(tx + tw / 2, TOP_Y + 1.28, "LLM-predicted tools\n$\\cup$ tools of $r$ nearest\nmined tasks", ha="center", va="center", fontsize=FS, color=C_TXT, zorder=3, linespacing=1.1)
box(tx, TOP_Y + 0.1, tw, 0.76, "", fc="#EEF3FB", ec=C_CAL)
ax.text(tx + tw / 2, TOP_Y + 0.72, "value layer  $V_\\ell$", ha="center", va="center", fontsize=FS, weight="bold", color=C_TXT, zorder=3)
ax.text(tx + tw / 2, TOP_Y + 0.36, "level $\\ell\\in$ {exact, email,\nclass, any}", ha="center", va="center", fontsize=FS, color=C_TXT, zorder=3, linespacing=1.1)
arrow((rx + rw, TOP_Y + 0.5), (tx, TOP_Y + 0.5))
arrow((rx + rw, TOP_Y + 0.62), (tx, TOP_Y + 1.25), rad=-0.15)

# ---------------- 3 Calibration (bottom right) ----------------
panel(X2, BOT_Y, W2, BOT_H, "③", "Calibration")
arrow((X2 + W2 / 2, TOP_Y), (X2 + W2 / 2, BOT_Y + BOT_H))
chain = [("exact", "$r{=}0$"), ("…", ""), ("exact", "all"), ("email", "all"), ("class", "all"), ("any", "all")]
n = len(chain); cw, gap = 0.5, 0.1
cx0 = X2 + (W2 - (n * cw + (n - 1) * gap)) / 2
cy, ch = BOT_Y + 1.14, 0.44
ax.text(cx0 - 0.0, cy + ch + 0.16, "nested chain  $\\Lambda$", ha="left", va="center", fontsize=FS, weight="bold", color=C_TXT)
for i, (a, b) in enumerate(chain):
    x = cx0 + i * (cw + gap)
    if a == "…":
        ax.text(x + cw / 2, cy + ch / 2, "…", ha="center", va="center", fontsize=FS + 2, color=C_TXT)
    else:
        box(x, cy, cw, ch, f"{a}\n{b}", fs=FS, fc="#EEF3FB" if i < 3 else "#E3EAF6", ec=C_CAL, lw=0.9)
    if i < n - 1:
        arrow((x + cw + 0.005, cy + ch / 2), (x + cw + gap - 0.005, cy + ch / 2), lw=0.8)
# brackets
def bracket(x0, x1, y, text):
    ax.plot([x0, x0, x1, x1], [y + 0.05, y, y, y + 0.05], color=C_EDGE, lw=0.8, zorder=3)
    ax.text((x0 + x1) / 2, y - 0.11, text, ha="center", va="center", fontsize=FS, color=C_TXT)
bracket(cx0, cx0 + 3 * cw + 2 * gap, cy - 0.04, "$r$ grows")
bracket(cx0 + 3 * (cw + gap), cx0 + n * cw + (n - 1) * gap, cy - 0.04, "$\\ell$ coarsens")
# CRC
box(X2 + 0.12, BOT_Y + 0.1, 2.45, 0.66, "", fc="#F4F5F7")
chip(X2 + 0.18, BOT_Y + 0.60, "cal.", C_CAL, w=0.36, h=0.16)
ax.text(X2 + 0.58, BOT_Y + 0.68, "CRC: first $\\lambda\\in\\Lambda$ with", ha="left", va="center", fontsize=FS, color=C_TXT)
ax.text(X2 + 1.345, BOT_Y + 0.33, "$\\frac{n}{n+1}\\,\\bar{L}_n(\\lambda)+\\frac{1}{n+1}\\leq\\varepsilon$", ha="center", va="center", fontsize=FS + 3, color=C_TXT)
box(X2 + 2.68, BOT_Y + 0.1, 1.0, 0.66, "", fc="#F4F5F7")
chip(X2 + 2.74, BOT_Y + 0.60, "sel.", C_SEL, w=0.36, h=0.16)
ax.text(X2 + 3.18, BOT_Y + 0.35, "tool\npredictor", ha="center", va="center", fontsize=FS, color=C_TXT, linespacing=1.1)

# ---------------- 4 Measurement (bottom left) ----------------
panel(X1, BOT_Y, W1, BOT_H, "④", "Measurement")
# arrows: holdout down into 4, policy from 3 to 4
arrow((centers["holdout"][0], TOP_Y), (centers["holdout"][0], BOT_Y + BOT_H - 0.0), color=C_HOLD, lw=1.4)
arrow((X2, BOT_Y + 0.43), (X1 + W1, BOT_Y + 0.43), lw=1.0)
ax.text((X2 + X1 + W1) / 2, BOT_Y + 0.55, "$\\hat\\lambda$", ha="center", va="center", fontsize=FS, color=C_TXT)
chip(X1 + 0.12, BOT_Y + 1.72 - 0.06, "holdout", C_HOLD, w=0.62, h=0.17)
box(X1 + 0.12, BOT_Y + 0.88, W1 - 0.24, 0.68, "", fc="#F4F5F7")
ax.text(X1 + W1 / 2, BOT_Y + 1.34, "every defense", ha="center", va="center", fontsize=FS, weight="bold", color=C_TXT, zorder=3)
ax.text(X1 + W1 / 2, BOT_Y + 1.06, "AMLP, Progent, CaMeL,\nAgent-Sentry, MELON, …", ha="center", va="center", fontsize=FS, color=C_TXT, zorder=3, linespacing=1.1)
box(X1 + 0.12, BOT_Y + 0.12, 1.0, 0.54, "benign cost\ninterception", fs=FS)
box(X1 + 1.2, BOT_Y + 0.12, 1.13, 0.54, "McNemar +\ntask-cluster\nbootstrap", fs=FS)
arrow((X1 + W1 / 2 - 0.3, BOT_Y + 0.88), (X1 + 0.62, BOT_Y + 0.66))
arrow((X1 + 1.12, BOT_Y + 0.39), (X1 + 1.2, BOT_Y + 0.39), lw=0.8)

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "method.pdf")
prev = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("method_preview.png")
fig.savefig(prev, dpi=200)
print("wrote", OUT / "method.pdf", prev)
