"""Method figure (Figure 1) of the AMLP paper, drawn in code, single-column width.
Offline path: task folds -> allowlist family -> calibration. Online path: enforcement and measurement on the holdout.
Layout is relative: every row is placed below the previous one and every box right of its neighbour (no hand-placed coordinates).
Usage: python3 analysis/make_method_fig.py [preview.png]  ->  outputs/figs/method.pdf (+ preview PNG in the current directory)
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})
ROOT = Path(__file__).resolve().parent.parent.parent   # release root (code/ and outputs/ are siblings)
OUT = ROOT / "outputs" / "figs"

FS = 6.6                      # body labels at single-column width
W = 3.33                      # \columnwidth of usenix-2020-09, inches
STRIP = 0.17                  # offline / online label strip
PX, PW = STRIP + 0.03, W - STRIP - 0.05   # panel x and width
PAD, GAP, VGAP, TITLE = 0.06, 0.09, 0.07, 0.17

C_HOLD, C_SEL, C_CAL = "#E8A33D", "#5FAE7B", "#5B8DD6"
C_BOX, C_EDGE, C_TXT, C_LAYER = "#F4F5F7", "#6B7280", "#1F2937", "#EEF3FB"
C_OK, C_NO = "#5FAE7B", "#D9534F"

draw = []                     # deferred draw calls: final height is known only after layout


def box(x, y, w, h, text="", fc=C_BOX, ec=C_EDGE, fs=FS, weight="normal", color=C_TXT, r=0.03, lw=0.6):
    draw.append(lambda ax: (
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                    fc=fc, ec=ec, lw=lw, zorder=2)),
        text and ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                         color=color, weight=weight, zorder=3, linespacing=1.12)))
    return (x, y, w, h)


def text(x, y, s, fs=FS, ha="center", weight="normal", color=C_TXT):
    draw.append(lambda ax: ax.text(x, y, s, ha=ha, va="center", fontsize=fs, weight=weight, color=color,
                                   zorder=3, linespacing=1.12))


def arrow(p, q, color=C_EDGE, lw=0.7, rad=0.0):
    draw.append(lambda ax: ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, color=color,
                                                        lw=lw, connectionstyle=f"arc3,rad={rad}", zorder=4,
                                                        shrinkA=0, shrinkB=0)))


def hrow(x0, widths, gap=GAP):
    """x positions of boxes placed left to right, each right of its neighbour."""
    xs = []
    for w in widths:
        xs.append(x0)
        x0 += w + gap
    return xs


def right(b): return (b[0] + b[2], b[1] + b[3] / 2)
def left(b): return (b[0], b[1] + b[3] / 2)
def top(b): return (b[0] + b[2] / 2, b[1] + b[3])
def bottom(b): return (b[0] + b[2] / 2, b[1])


class Panel:
    """Panel whose top edge sits VGAP below the previous panel; rows stack downwards inside it."""
    def __init__(self, y_top, num, title):
        self.top, self.y = y_top, y_top - TITLE - PAD
        text(PX + 0.06, y_top - TITLE / 2 - 0.02, f"{num}  {title}", fs=FS + 0.4, ha="left", weight="bold")

    def row(self, h):
        self.y -= h
        y = self.y
        self.y -= PAD
        return y

    def close(self):
        top_, bot = self.top, self.y
        draw.insert(0, lambda ax: ax.add_patch(FancyBboxPatch((PX, bot), PW, top_ - bot,
                    boxstyle="round,pad=0,rounding_size=0.04", fc="white", ec="#9CA3AF", lw=0.7, zorder=1)))
        return bot - VGAP


IX = PX + PAD                 # inner x
IW = PW - 2 * PAD             # inner width

# ---------------- ① Task folds ----------------
y = 0.0
p1 = Panel(y, "①", "Task folds")
by = p1.row(0.17)
segs = [("holdout", 0.5, C_HOLD), ("sel.", 0.25, C_SEL), ("cal.", 0.25, C_CAL)]
xs = hrow(IX, [IW * f for _, f, _ in segs], gap=0)
fold = {}
for (name, f, c), x in zip(segs, xs):
    draw.append(lambda ax, x=x, f=f, c=c: ax.add_patch(Rectangle((x, by), IW * f, 0.17, fc=c, ec="white", lw=0.8, zorder=2)))
    text(x + IW * f / 2, by + 0.085, name, weight="bold", color="white")
    fold[name] = (x, by, IW * f, 0.17)
ly = pool_y = p1.row(0.12) + 0.06
text(fold["holdout"][0] + fold["holdout"][2] / 2, ly, "novel tasks, never mined")
text(fold["sel."][0] + IW * 0.25, ly, "benign runs $\\rightarrow$ pool $B$")
y = p1.close()

# ---------------- ② Allowlist family ----------------
p2 = Panel(y, "②", "Allowlist family")
lh = 0.50
ly = p2.row(lh)
tw = IW * 0.37
xs = hrow(IX, [tw, IW - tw - GAP])
tool = box(xs[0], ly, tw, lh, "tool layer $T$\n$g(p)\\ \\cup$ tools of $r$\nnearest tasks in $B$", fc=C_LAYER, ec=C_CAL)
val = box(xs[1], ly, IW - tw - GAP, lh,
          "value layer $V_\\ell$, per control arg\nmined $\\cup$ env entities $\\cup$ literals of $p$\n"
          "$\\ell$: exact $\\subset$ email $\\subset$ class $\\subset$ any", fc=C_LAYER, ec=C_CAL)
y = p2.close()
arrow((fold["sel."][0] + IW * 0.25 - 0.25, pool_y - 0.06), (tool[0] + tool[2] * 0.85, tool[1] + tool[3]), color=C_EDGE)
arrow((fold["sel."][0] + IW * 0.25 + 0.05, pool_y - 0.06), (val[0] + val[2] * 0.6, val[1] + val[3]), color=C_EDGE)

# ---------------- ③ Calibration ----------------
p3 = Panel(y, "③", "Calibration   (target $\\varepsilon$)")
ch = 0.28
cy = p3.row(ch)
chain = ["exact\n$r{=}0$", "$\\cdots$", "exact\n$r_{\\max}$", "email", "class", "any"]
cw = (IW - 5 * GAP) / 6
xs = hrow(IX, [cw] * 6)
cb = []
for i, (lab, x) in enumerate(zip(chain, xs)):
    cb.append(box(x, cy, cw, ch, lab, fc=C_LAYER if lab != "$\\cdots$" else "white",
                  ec=C_CAL if lab != "$\\cdots$" else "white"))
    if i:
        arrow(right(cb[i - 1]), left(cb[i]))
bry = p3.row(0.09) + 0.07


def bracket(x0, x1, s):
    draw.append(lambda ax: ax.plot([x0, x0, x1, x1], [bry + 0.03, bry, bry, bry + 0.03], color=C_EDGE, lw=0.5, zorder=3))
    text((x0 + x1) / 2, bry - 0.06, s)


bracket(cb[0][0], cb[2][0] + cw, "$r$ grows")
bracket(cb[3][0], cb[5][0] + cw, "$\\ell$ coarsens")
p3.row(0.03)
rh = 0.30
ry = p3.row(rh)
ww = [IW * 0.66, IW - IW * 0.66 - GAP]
xs = hrow(IX, ww)
crc = box(xs[0], ry, ww[0], rh, "cal.: first $\\lambda \\in \\Lambda$ with\n"
          "$\\frac{n}{n+1}\\bar{L}_n(\\lambda) + \\frac{1}{n+1} \\leq \\varepsilon \\;\\rightarrow\\; \\hat\\lambda$",
          ec=C_CAL)
sel = box(xs[1], ry, ww[1], rh, "sel.: predictor\n$g^\\ast$ with most flags", ec=C_SEL)
y = p3.close()
arrow((IX + IW / 2, p3.top + VGAP), (IX + IW / 2, p3.top))
offline_bot = p3.y

# ---------------- ④ Enforcement and measurement (online) ----------------
p4 = Panel(y, "④", "Enforcement and measurement on holdout")
fh = 0.30
fy = p4.row(fh)
fw = [IW * 0.21, IW * 0.16, IW * 0.21]
fw.append(IW - sum(fw) - 3 * GAP)
xs = hrow(IX, fw)
run = box(xs[0], fy, fw[0], fh, "holdout run\n$\\pm$ injection", ec=C_HOLD, fc="#FDF3E3")
call = box(xs[1], fy, fw[1], fh, "call $(t, v)$")
chk = box(xs[2], fy, fw[2], fh, "$t \\in T$ and\n$v \\in V_{\\hat\\ell}$ ?", ec=C_CAL, weight="bold")
oh = (fh - 0.04) / 2
ok = box(xs[3], fy + oh + 0.04, fw[3], oh, "yes: execute", fc="#E8F4EC", ec=C_OK)
no = box(xs[3], fy, fw[3], oh, "no: error to agent", fc="#FBEAEA", ec=C_NO)
arrow(right(run), left(call)); arrow(right(call), left(chk))
arrow(right(chk), left(ok)); arrow(right(chk), left(no))
mh = 0.24
my = p4.row(mh)
mw = [IW * 0.30, IW * 0.33]
mw.append(IW - sum(mw) - 2 * GAP)
xs = hrow(IX, mw)
m1 = box(xs[0], my, mw[0], mh, "no attack:\nfalse block, cost")
m2 = box(xs[1], my, mw[1], mh, "attack:\ninterception, flag")
m3 = box(xs[2], my, mw[2], mh, "paired: McNemar,\ntask bootstrap")
arrow(right(m1), left(m2)); arrow(right(m2), left(m3))
y = p4.close()
arrow((IX + IW / 2, p4.top + VGAP), (IX + IW / 2, p4.top))
text(IX + IW / 2 + 0.08, p4.top + VGAP / 2 + 0.01, "$\\hat\\lambda,\\ g^\\ast$", ha="left", fs=FS - 0.4)

# ---------------- offline / online strip ----------------
H = -y + 0.02
top_y = 0.0


def strip(y0, y1, s, c):
    draw.append(lambda ax: ax.add_patch(Rectangle((0.02, y1), STRIP - 0.03, y0 - y1, fc=c, ec="none", zorder=1)))
    draw.append(lambda ax: ax.text(0.02 + (STRIP - 0.03) / 2, (y0 + y1) / 2, s, rotation=90, ha="center",
                                   va="center", fontsize=FS + 0.4, weight="bold", color="white", zorder=3))


strip(top_y, offline_bot - PAD, "offline", "#6B7280")
strip(p4.top, p4.y + PAD - 0.0, "online", C_HOLD)

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(-H + 0.01, 0.01); ax.axis("off")
for f in draw:
    f(ax)
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "method.pdf")
prev = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("method_preview.png")
fig.savefig(prev, dpi=150)
print("wrote", OUT / "method.pdf", prev, f"{W}x{H:.2f}in")
