r"""Figures and tables from data/analysis_out/<set>/*.json (analysis/holdout_set.py), plan.md §21.
Figures show the primary set all48, the pre-registered 48 holdout tasks. Every main table is built for all48 and for
clean27, the 27 holdout tasks no pilot run touched, and printed with the clean27 value after each differing all48 cell
as \untouched{...} (defined in main.tex). Appendix tables (rq3_ci, rq4_transfer_flag) and the setup table use all48.
Usage: python3 analysis/make_figs.py"""
import builtins
import re
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(D)   # release root (code/ and data/ are siblings)
sys.path.insert(0, f"{D}/analysis")
from fmt1 import F  # noqa: E402  one rounding rule: ROUND_HALF_UP on exact values (project lead)
FIG, TAB = f"{ROOT}/outputs/figs", f"{ROOT}/outputs/tables"
SET = "all48"                                       # switched by main() for each set
PRIMARY = True
OUTD = f"{ROOT}/data/analysis_out/{SET}"
SUF = ""
TABS = {}                                            # set -> table name -> LaTeX text
MERGE = ("rq2_models", "rq1_eps_grid", "rq1_chain", "rq3_main", "rq3_replay", "rq4_shift", "rq4_poison",
         "rq_transfer", "tests_utility", "rq3_ci", "rq4_transfer_flag", "rq3_bounds", "rq3_oracle")   # cold read fb025b2 E2


class _Tab:
    """File stand-in: table writers call open(f"{TAB}/<name>.tex", "w").write(text); the text is kept per set."""
    def __init__(self, path):
        self.name = os.path.basename(path)[:-4]

    def write(self, text):
        TABS.setdefault(SET, {})[self.name] = text


def open(path, mode="r", *a, **k):                   # noqa: A001
    import builtins
    if mode == "w" and path.startswith(TAB):
        return _Tab(path)
    return builtins.open(path, mode, *a, **k)


def savefig(fig, name):
    if PRIMARY:                                     # figures show the primary set only
        fig.savefig(f"{FIG}/{name}", bbox_inches="tight")
    plt.close(fig)


def tests():
    return json.load(open(f"{OUTD}/tests.json"))   # analysis/holdout_tests.py (C3' tests, recall side, C1)
NAME = {"qwen3-8b-local": "Qwen3-8B", "llama31-8b-local": "Llama-3.1-8B", "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
        "gpt-4.1-mini-2025-04-14": "gpt-4.1-mini"}
COL = {"qwen3-8b-local": "#4C72B0", "llama31-8b-local": "#8C8C8C", "gpt-4o-mini-2024-07-18": "#DD8452",
       "gpt-4.1-mini-2025-04-14": "#55A868"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def chain_label(c):
    r = "all" if c["r"] >= 1000 else str(c["r"])
    return f"exact\nr={r}" if c["level"] == "exact" else f"{c['level']}\nr=all"


def fig_rq1(rq):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for m, x in rq["rq1"].items():
        ch = x["chain"]
        xs = range(len(ch))
        axes[0].plot(xs, [c["cal_mean_loss"] for c in ch], marker="o", ms=3, color=COL[m], label=NAME[m])
        if ch[0]["holdout_recall"] is not None and ch[0]["holdout_n_viol"] > 0:
            axes[1].plot(xs, [c["holdout_recall"] for c in ch], marker="o", ms=3, color=COL[m], label=NAME[m])
    eps = float(rq["meta"]["primary_eps"])
    axes[0].axhline(eps, ls="--", lw=1, color="black")
    axes[0].text(0.1, eps + 0.01, r"$\varepsilon$", fontsize=9)
    labels = [chain_label(c) for c in next(iter(rq["rq1"].values()))["chain"]]
    for ax, yl in zip(axes, ("calibration loss ↓", "holdout flag rate, all refusals ↑")):
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel(yl)
        ax.set_ylim(0, 1.0 if "flag" in yl else 0.4)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    savefig(fig, "rq1_chain.pdf")


def fig_rq2(rq):
    pooled = rq["rq2"]["pooled_per_attacker_tool"]
    items = sorted(pooled.items(), key=lambda kv: kv[1]["n"])
    names = [k.replace("_", " ") for k, _ in items]
    inter = [v["intercepted"] for _, v in items]
    invis = [v["invisible"] for _, v in items]
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.barh(names, inter, color="#55A868", hatch="//", edgecolor="black", linewidth=0.4, label="flagged")
    ax.barh(names, invis, left=inter, color="#C44E52", edgecolor="black", linewidth=0.4, label="inside the tool layer")
    ax.set_xlabel("violations, pooled over the three models with successful attacks")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    savefig(fig, "rq2_attacker_tool.pdf")


def tab_rq2(rq):
    rows = []
    for m in ("qwen3-8b-local", "gpt-4o-mini-2024-07-18", "gpt-4.1-mini-2025-04-14"):
        x = rq["rq2"][m]
        rows.append(f"{NAME[m]} & {x['n_viol']} & {P(100 * x['intercepted'] / x['n_viol'])} & "
                    f"{P(100 * x['invisible'] / x['n_viol'])} \\\\")
    body = "\n".join(rows)
    open(f"{TAB}/rq2_models{SUF}.tex", "w").write(
        "% generated by analysis/make_figs.py\n\\begin{tabular}{lrrr}\n\\toprule\n"
        "Model & Violations & Flagged, all (\\%) $\\uparrow$ & Inside the tool layer (\\%) $\\downarrow$ \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n")


def tab_rq1(rq):
    def lab(c):
        r = "all" if c["r"] >= 1000 else str(c["r"])
        return f"{c['level']}, $r$={r}"
    rows = []
    for m, x in rq["rq1"].items():
        cells = []
        for eps in ("0.05", "0.1", "0.15"):
            c = x["chosen_by_eps"][eps]
            ok = x["feasible_by_eps"][eps]
            cells.append(lab(c) + ("" if ok else r"$^\dagger$"))
        rows.append(f"{NAME[m]} & " + " & ".join(cells) + r" \\")
    open(f"{TAB}/rq1_eps_grid{SUF}.tex", "w").write(
        "% generated by analysis/make_figs.py\n\\begin{tabular}{llll}\n\\toprule\n"
        "Model & $\\varepsilon=0.05$ & $\\varepsilon=0.10$ & $\\varepsilon=0.15$ \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


ARM_NAME = {"amlp": "calibrated allowlist (ours)", "block_all": "block-all", "spotlighting": "spotlighting", "sandwich": "sandwich",
            "tool_filter": "tool filter", "pi_detector": "PI detector", "melon": "MELON", "tripwire": "TripWire",
            "progent": "Progent", "camel": "CaMeL", "agentsentry": "Agent-Sentry"}
SEC3 = ["qwen3-8b-local", "gpt-4o-mini-2024-07-18", "gpt-4.1-mini-2025-04-14"]


def fig_rq3(q):
    r3 = q["rq3"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
    pts = {}
    for ax, m in zip(axes, SEC3):
        pts[m] = []
        for arm, e in r3[m].items():
            if arm == "amlp_vs_tripwire" or e.get("block_rate") is None:
                continue
            x, y = 100 * e["benign_cost"], 100 * e["block_rate"]
            ours = arm == "amlp"
            ax.scatter(x, y, s=28 if ours else 16, color="#C44E52" if ours else "#4C72B0", zorder=3)
            pts[m].append((x, y, ARM_NAME[arm].replace(" (ours)", "")))
        xs = [x for x, _, _ in pts[m]]
        span = max(xs) - min(xs)
        ax.set_xlim(min(xs) - 0.22 * span - 4, max(xs) + 0.45 * span + 2)
        ax.axvline(0, lw=0.6, color="grey")
        ax.set_title(NAME[m], fontsize=8)
        ax.set_xlabel("benign cost (pp) ↓")
    axes[0].set_ylabel("interception (%) ↑")
    axes[0].set_ylim(-30, 108)
    fig.tight_layout()
    fig.canvas.draw()
    for ax, m in zip(axes, SEC3):                        # greedy de-overlap in display points, highest point first
        disp = [(tuple(ax.transData.transform((x, y)) * 72 / fig.dpi), lab, x, y) for x, y, lab in pts[m]]
        boxes = [(px - 3, py - 3, px + 3, py + 3) for (px, py), _, _, _ in disp]     # markers
        hit = lambda r: any(r[0] < q[2] and q[0] < r[2] and r[1] < q[3] and q[1] < r[3] for q in boxes)
        for (px, py), lab, x, y in sorted(disp, key=lambda t: (-t[0][1], t[0][0])):
            w, h = 3.0 * len(lab) + 1, 6.5
            cands = [(dx, dy) for dy in (2, -h - 2, 9, -h - 9, 16, -h - 16, 23, -h - 23) for dx in (4, -w - 4)]
            dx, dy = next((c for c in cands if not hit((px + c[0], py + c[1], px + c[0] + w, py + c[1] + h))), cands[0])
            boxes.append((px + dx, py + dy, px + dx + w, py + dy + h))
            far = abs(dy) > 8 or dx < 0 and abs(dy) > 3
            ax.annotate(lab, (x, y), fontsize=6, xytext=(dx, dy), textcoords="offset points", va="bottom", ha="left",
                        arrowprops=dict(arrowstyle="-", lw=0.4, color="grey", shrinkA=0, shrinkB=2) if far else None)
    savefig(fig, "rq3_cost_vs_interception.pdf")


def rq3_head(llama):
    """Header of the RQ3 tables; the Llama-3.1-8B cost column only when the outputs hold Llama (four-model run)."""
    return (r"\begin{tabular}{l" + "rr" * 3 + ("r" if llama else "") + "}\n\\toprule\n"
            r" & \multicolumn{2}{c}{Qwen3-8B} & \multicolumn{2}{c}{gpt-4o-mini} & \multicolumn{2}{c}{gpt-4.1-mini}" + (r" & Llama-3.1-8B" if llama else "") + r" \\" + "\n"
            r"Defense & Int.\,$\uparrow$ & Cost\,$\downarrow$ & Int.\,$\uparrow$ & Cost\,$\downarrow$ & Int.\,$\uparrow$ & Cost\,$\downarrow$" + (r" & Cost\,$\downarrow$" if llama else "") + r" \\" + "\n\\midrule\n")


def tab_rq3(q):
    r3 = q["rq3"]
    order = ["amlp", "progent", "agentsentry", "camel", "melon", "tripwire", "tool_filter", "sandwich", "spotlighting",
             "pi_detector", "block_all"]
    llama = "llama31-8b-local" in r3          # absent from the main results (project lead): appendix table app_llama
    mods = SEC3 + (["llama31-8b-local"] if llama else [])
    rows = []
    for arm in order:
        cells = []
        for m in mods:
            e = r3[m].get(arm)
            if e is None:
                cells += ["--"] if m == "llama31-8b-local" else ["--", "--"]
                continue
            if m != "llama31-8b-local":
                cells.append(f"{P(100 * e['block_rate'])}" if e.get("block_rate") is not None else "--")
            cells.append(f"{P(100 * e['benign_cost'], 1)}")
        name = ARM_NAME[arm]
        if arm == "amlp":
            name = r"\textbf{Calibrated allowlist} (ours)"
            cells = [r"\textbf{" + c + "}" for c in cells]
        rows.append(name + " & " + " & ".join(cells) + r" \\")
    head = rq3_head(llama)
    open(f"{TAB}/rq3_main{SUF}.tex", "w").write("% generated by analysis/make_figs.py\n" + head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def fig_rq4(q):
    po = q["rq4"]["poisoning"]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for m in SEC3:
        xs = [0.0, 0.01, 0.05, 0.10]
        ax.plot([100 * x for x in xs], [100 * (po[m][str(x)]["recall"] if x == 0.0 else po[m][str(x)]["recall_mean"]) for x in xs], marker="o", ms=3, color=COL[m], label=NAME[m])
    ax.set_xlabel("poisoned mining runs (%)")
    ax.set_ylabel("flag rate, all refusals (%) ↑")
    ax.set_ylim(0, 45)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    savefig(fig, "rq4_poisoning.pdf")


def tab_rq4(q):
    r4 = q["rq4"]
    rows = []
    for m, x in r4["paraphrase"].items():
        for arm in ("amlp", "progent"):
            e = x[arm]
            nm = r"\textbf{Calibrated allowlist} (ours)" if arm == "amlp" else "Progent"
            rows.append(f"Paraphrase, {NAME[m]} & {nm} & {P(100 * e['cost_orig'], 1)} & {P(100 * e['cost_para'], 1)} & {P(100 * e['shift'], 1)} \\\\")
    rq12 = json.load(open(f"{OUTD}/rq12.json"))["rq1"]
    own_fb = {m: next(c["holdout_fb"] for c in x["chain"] if c["level"] == "any" and c["r"] >= 1000) for m, x in rq12.items()}
    flag_rows = []
    for k, v in r4["transfer"].items():           # errata E3: transfer reports the false-block rate only
        a, b = k.split("->")
        rows.append(f"Mined on {NAME[a]} & on {NAME[b]} & {P(100 * own_fb[b])} & {P(100 * v['fb'])} & "
                    f"{P(100 * (v['fb'] - own_fb[b]), 1)} \\\\")
        base, rec = q["rq4"]["poisoning"][b]["0.0"]["recall"], v["recall"]
        flag_rows.append(f"{NAME[a]} & {NAME[b]} & " + (f"{P(100 * base)}" if base is not None else "--") + " & "
                         + (f"{P(100 * rec)}" if rec is not None else "--") + " \\\\")
    open(f"{TAB}/rq4_shift{SUF}.tex", "w").write(
        "% generated by analysis/make_figs.py\n\\begin{tabular}{llrrr}\n\\toprule\n"
        "Condition & Defense or target & Original & Shifted & Shift \\\\\n\\midrule\n" + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n")
    open(f"{TAB}/rq4_transfer_flag{SUF}.tex", "w").write(
        "% generated by analysis/make_figs.py (exploratory, appendix)\n\\begin{tabular}{llrr}\n\\toprule\n"
        "Mined on & Applied to & Own allowlist, all $\\uparrow$ & Transferred, all $\\uparrow$ \\\\\n\\midrule\n" + "\n".join(flag_rows)
        + "\n\\bottomrule\n\\end{tabular}\n")


def fig_transfer():
    sn = json.load(open(f"{OUTD}/seen_novel.json"))
    cv = json.load(open(f"{OUTD}/seen_novel_curve.json"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    ax = axes[0]
    sp = json.load(open(f"{OUTD}/seen_novel_pure.json"))
    pooled = dict(sn["pooled"]); pooled.update(sp["pooled"])
    meths = [("M2", "automaton\nmined"), ("M1a-pure", "arguments\nmined"), ("M1a", "arguments\n+request"),
             ("M1b-pure", "tools\nmined"), ("M1b", "tools\n+predicted")]
    for i, (mk, lab) in enumerate(meths):
        for off, st, col, hatch in ((-0.18, "seen", "#4C72B0", ""), (0.18, "novel", "#C44E52", "//")):
            ax.bar(i + off, 100 * pooled[f"{mk}_{st}_fb"], width=0.34, color=col, hatch=hatch, edgecolor="black",
                   linewidth=0.4, label=st if i == 0 else None)
    ax.set_xticks(range(len(meths))); ax.set_xticklabels([l for _, l in meths], fontsize=6)
    ax.set_ylabel("false-block rate (%) ↓"); ax.set_ylim(0, 100); ax.legend(frameon=False, fontsize=7)
    ax = axes[1]
    for mk, lab, col in (("M2", "trajectory automaton", "#C44E52"), ("M1a", "AMLP exact", "#4C72B0")):
        ks = [1, 2, 4, 6, 7]; js = [2, 4, 6, 8]
        ax.plot(ks, [100 * cv["fb"][f"{mk}_seen{k}"]["point"] for k in ks], marker="o", ms=3, color=col, label=f"{lab}, seen: own traces")
        ax.plot(js, [100 * cv["fb"][f"{mk}_novel{j}"]["point"] for j in js], marker="s", ms=3, ls="--", color=col,
                label=f"{lab}, novel: traces per mined task")
    ax.set_xlabel("traces"); ax.set_ylabel("false-block rate (%) ↓"); ax.set_ylim(0, 100)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=6, loc="lower center", bbox_to_anchor=(0.72, 0.0), ncol=2)
    savefig(fig, "rq_transfer.pdf")
    names = {"qwen3-8b-local": "Qwen3-8B", "llama31-8b-local": "Llama-3.1-8B", "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
             "gpt-4.1-mini-2025-04-14": "gpt-4.1-mini"}
    spm = sp["per_model"]                               # cold read 95d6ba9 H2: per-model mined-only columns
    rows = []
    for mod, x in sn["per_model"].items():
        cells = [f"{P(100 * x[f'M2_{st}_fb'])}" for st in ("seen", "novel")]
        cells += [f"{P(100 * spm[mod][f'M1a-pure_{st}_fb'])}" for st in ("seen", "novel")]
        cells += [f"{P(100 * x[f'M1a_{st}_fb'])}" for st in ("seen", "novel")]
        cells += [f"{P(100 * spm[mod][f'M1b-pure_{st}_fb'])}" for st in ("seen", "novel")]
        cells += [f"{P(100 * x[f'M1b_{st}_fb'])}" for st in ("seen", "novel")]
        rows.append(f"{names[mod]} & " + " & ".join(cells) + r" \\")
    sv = r"Seen\,$\downarrow$ & Novel\,$\downarrow$"
    open(f"{TAB}/rq_transfer{SUF}.tex", "w").write(
        "% generated by analysis/make_figs.py\n\\begin{tabular}{l" + "rr" * 5 + "}\n\\toprule\n"
        " & \\multicolumn{2}{c}{Automaton} & \\multicolumn{2}{c}{Mined arguments} & \\multicolumn{2}{c}{Hybrid arguments (exact)}"
        " & \\multicolumn{2}{c}{Mined tools} & \\multicolumn{2}{c}{Hybrid tools (calibrated allowlist)} \\\\\n"
        "Model & " + " & ".join([sv] * 5) + " \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

def tab_rq3_replay():
    """Replay comparison of tool-level policies on identical runs (plan.md §16/§18 replay, C3' judge rows), with the
    flag rate counting all refusals and counting refusals before the first harmful call (plan.md §22.2)."""
    sn = json.load(open(f"{OUTD}/seen_novel.json"))["pooled"]
    sp = json.load(open(f"{OUTD}/seen_novel_pure.json"))["pooled"]
    jt = tests()
    fbh = json.load(open(f"{ROOT}/data/analysis_out/flag_before_harm.json"))["replay"]
    fbh = fbh if SET == "all48" else fbh["clean27"]
    rows = [("Calibrated allowlist (ours), mined and predicted tools", sn["M1b_novel_fb"], sn["M1b_novel_flag"], fbh["calibrated"]["flag_before_harm"]),
            ("Mined tools only", sp["M1b-pure_novel_fb"], sp["M1b-pure_novel_flag"], fbh["mined_tools_only"]["flag_before_harm"]),
            ("Predicted tools only", sp["M1b-pred_novel_fb"], sp["M1b-pred_novel_flag"], fbh["predicted_tools_only"]["flag_before_harm"]),
            ("Progent", jt["c3prime_fb"]["fb_progent"], jt["recall_A2rev"]["recall_progent"], fbh["progent"]["flag_before_harm"])]
    body = "\n".join(f"{n} & {P(100 * fb)} & {P(100 * fl)} & {P(100 * bh)} \\\\" for n, fb, fl, bh in rows)
    open(f"{TAB}/rq3_replay{SUF}.tex", "w").write(
        "% generated by analysis/make_figs.py\n\\begin{tabular}{lrrr}\n\\toprule\n"
        "Policy & False-block rate (\\%) $\\downarrow$ & Flagged, all (\\%) $\\uparrow$ & Flagged before harm (\\%) $\\uparrow$ \\\\\n\\midrule\n" + body
        + "\n\\bottomrule\n\\end{tabular}\n")

def tab_rq3_ci(q):
    r3 = q["rq3"]
    order = ["amlp", "progent", "agentsentry", "camel", "melon", "tripwire", "tool_filter", "sandwich", "spotlighting",
             "pi_detector", "block_all"]
    ci = lambda e, k: f"[{P(100 * e[k][0])}, {P(100 * e[k][1])}]" if e and e.get(k) else "--"   # noqa: E731
    llama = "llama31-8b-local" in r3
    rows = []
    for arm in order:
        cells = []
        for m in SEC3:
            e = r3[m].get(arm)
            cells += [ci(e, "block_rate_ci"), ci(e, "benign_cost_ci")]
        if llama:
            cells.append(ci(r3["llama31-8b-local"].get(arm), "benign_cost_ci"))
        nm = r"\textbf{Calibrated allowlist} (ours)" if arm == "amlp" else ARM_NAME[arm]
        rows.append(nm + " & " + " & ".join(cells) + r" \\")
    head = rq3_head(llama)
    open(f"{TAB}/rq3_ci{SUF}.tex", "w").write("% generated by analysis/make_figs.py\n" + head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def tab_rq1_chain(rq):
    rows = []
    for m, x in rq["rq1"].items():
        for c in x["chain"]:
            lv = c["level"]
            r = "all" if c["r"] >= 1000 else str(c["r"])
            rec = f"{P(100 * c['holdout_recall'])}" if c["holdout_recall"] is not None else "--"
            ch = [("$" + {"0.05": ".05", "0.1": ".10", "0.15": ".15"}[e] + "$") + ("" if x["feasible_by_eps"][e] else r"$^\ddagger$")
                  for e in ("0.05", "0.1", "0.15") if (x["chosen_by_eps"][e]["level"], x["chosen_by_eps"][e]["r"]) == (c["level"], c["r"])]
            rows.append(f"{NAME[m]} & {lv} & {r} & {P(100 * c['cal_mean_loss'])} & {P(100 * c['cal_bound'])} & "
                        f"{P(100 * c['holdout_fb'])} & {rec} & {', '.join(ch) or '--'} \\\\")
        rows.append("\\midrule")
    rows = rows[:-1]
    open(f"{TAB}/rq1_chain{SUF}.tex", "w").write(
        "% generated by analysis/make_figs.py\n\\begin{tabular}{llrrrrrl}\n\\toprule\n"
        "Model & Level & $r$ & Cal. loss $\\downarrow$ & Bound $\\downarrow$ & Holdout FB $\\downarrow$ & Flagged, all $\\uparrow$ & Chosen at $\\varepsilon$ \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def tab_rq4_poison(q):
    po = q["rq4"]["poisoning"]
    rows = []
    for m in SEC3:
        cells = [f"{P(100 * po[m]['0.0']['recall'])}"]
        for rho in ("0.01", "0.05", "0.1"):
            e = po[m][rho]
            cells.append(f"{P(100 * e['recall_mean'])} ({P(100 * e['recall_min'])}--{P(100 * e['recall_max'])})")   # no bare brackets (author rule 10-19)
        rows.append(f"{NAME[m]} & " + " & ".join(cells) + " \\\\")
    open(f"{TAB}/rq4_poison{SUF}.tex", "w").write(
        "% generated by analysis/make_figs.py\n\\begin{tabular}{lrrrr}\n\\toprule\n"
        "Model & $\\rho=0$ & $\\rho=0.01$ & $\\rho=0.05$ & $\\rho=0.10$ \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def tab_tests():
    c1 = tests()["C1"]
    rows = []
    for m in [x for x in NAME if x in c1["per_model"]]:
        e = c1["per_model"][m]
        rows.append(f"{NAME[m]} & {e['n']} & {P(100 * e['drop'], 1)} & [{P(100 * e['ci95'][0], 1)}, {P(100 * e['ci95'][1], 1)}] \\\\")
    rows.append("\\midrule")
    rows.append(f"Pooled & {c1['n_pairs']:,}".replace(",", "{,}") + f" & {P(100 * c1['drop_pooled'], 1)} & [{P(100 * c1['ci95'][0], 1)}, {P(100 * c1['ci95'][1], 1)}] \\\\")
    open(f"{TAB}/tests_utility{SUF}.tex", "w").write(
        "% generated by analysis/make_figs.py\n\\begin{tabular}{lrrr}\n\\toprule\n"
        "Model & Pairs & Utility drop (pp) $\\downarrow$ & 95\\% interval \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def tab_setup():
    import sys
    sys.path.insert(0, D)
    import m3_common as C
    import holdout_set as HS
    clean = HS.CLEAN
    rd = C.read_jsonl.raw                           # unfiltered: the table counts both sets

    def count(rows, unique=False):                  # (all 48 holdout tasks, 27 primary tasks)
        if unique:
            rows = list({r["row_id"]: r for r in rows}.values())
        return len(rows), sum((r["suite"], r["ut"]) in clean for r in rows)
    models = [m for m in C.MODELS if m in NAME]     # the paper's models; C.MODELS also holds the exp-ss E1 models (internal review)
    mining = sum(len(rd(C.rows_path(False, "mine", m))) for m in models)
    ben = [r for m in models for r in rd(C.rows_path(False, "monitor", m))]
    on = []
    for m in models:
        for g in ("main", "camel", "agentsentry"):
            p = C.rows_path(False, "online", m, g)
            if os.path.exists(p):
                on += list({r["row_id"]: r for r in rd(p)}.values())
    para = []
    for p in [f"{ROOT}/data/m3/paraphrase_{m}.jsonl" for m in ("qwen3-8b-local", "gpt-4.1-mini-2025-04-14")]:
        para += list({r["row_id"]: r for r in rd(p)}.values())
    pc = lambda rows: "{1:,} of {0:,}".format(*count(rows))   # noqa: E731
    rows = [f"Holdout / selection / calibration tasks & {len(C.HOLDOUT)} / {len(C.SELECTION)} / {len(C.CALIBRATION)} \\\\",
            f"Holdout tasks untouched by pilot runs & {len(clean)} of {len(C.HOLDOUT)} \\\\",
            f"Models & {', '.join(NAME[m] for m in NAME if m in C.MODELS)} \\\\",
            f"Mining runs & {mining:,} \\\\",
            f"Undefended benign holdout runs, untouched of all & {pc(ben)} \\\\",
            f"Online holdout runs, all defenses and no defense, untouched of all & {pc(on)} \\\\",
            f"Paraphrase runs, untouched of all & {pc(para)} \\\\"]
    open(f"{TAB}/setup{SUF}.tex", "w").write(
        "% generated by analysis/make_figs.py\n\\begin{tabular}{ll}\n\\toprule\n" + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n")


def _plain(cell):
    return re.sub(r"\\textbf\{(.*)\}", r"\1", _strip(cell).strip())


TAG = re.compile("\x01([^\x02]*)\x02")


def P(v, sign=False):
    """Printed point value with its full-precision value attached (stripped before writing), so that the dagger
    rule compares unrounded values (project lead)."""
    s = f"{F(v, '+.1f')}" if sign else f"{F(v, '.1f')}"
    return s + f"\x01{v!r}\x02"


def _strip(text):
    return TAG.sub("", text)


MARKS = []                                    # codex round 2 C2: cells whose two task sets differ in sign or by >5 points


def _mark(p, q):
    """Dagger for point values only: both cells must carry exactly one full-precision tag. Counts, intervals,
    L / M / U triples and mean [min, max] cells are skipped explicitly (project lead option A)."""
    tp, tq = TAG.findall(p), TAG.findall(q)
    if len(tp) != 1 or len(tq) != 1:
        SKIPPED.append({"all48": _strip(p), "clean27": _strip(q), "tags": [len(tp), len(tq)]})
        return ""
    x, y = float(tp[0]), float(tq[0])
    if x * y < 0 or abs(x - y) > 5 + 1e-9:          # tolerance for float noise on exact fifths, e.g. -5.000000000000004
        MARKS.append({"all48": x, "clean27": y, "flip": x * y < 0})
        return "$^\\dagger$"
    return ""


SKIPPED = []                                  # cells the dagger rule does not judge (composite or count)


def merge(a, b):
    """Print the clean27 value after every all48 cell that differs from it."""
    la, lb = a.split("\n"), b.split("\n")
    assert len(la) == len(lb), (len(la), len(lb))
    out = []
    for x, y in zip(la, lb):
        if x == y or "&" not in x:
            out.append(x)
            continue
        end = " \\\\"
        assert x.endswith(end) and y.endswith(end), (x, y)
        cx, cy = x[:-len(end)].split(" & "), y[:-len(end)].split(" & ")
        assert len(cx) == len(cy) and cx[0] == cy[0], (x, y)
        cells = [cx[0]] + [p if _plain(p) == _plain(q) else f"{p}\\untouched{{{_plain(q)}}}{_mark(p, q)}" for p, q in zip(cx[1:], cy[1:])]
        out.append(" & ".join(cells) + end)
    return "\n".join(out)


def tab_oracle():
    """Codex round 1 C10: interception of every arm under both attack oracles (appendix app:oracle), all48 only."""
    ex = json.load(open(f"{OUTD}/rq34.json"))["rq3"]
    ad = json.load(open(f"{OUTD}/rq34_agentdojo.json"))["rq3"]
    order = ["amlp", "progent", "agentsentry", "camel", "melon", "tripwire", "tool_filter", "sandwich", "spotlighting",
             "pi_detector", "block_all"]
    rows = []
    for arm in order:
        cells = []
        for m in SEC3:
            a, e = ad[m].get(arm), ex[m].get(arm)
            cells += [f"{P(100 * a['block_rate'])}" if a and a.get("block_rate") is not None else "--",
                      f"{P(100 * e['block_rate'])}" if e and e.get("block_rate") is not None else "--"]
        nm = r"\textbf{Calibrated allowlist} (ours)" if arm == "amlp" else ARM_NAME[arm]
        rows.append(nm + " & " + " & ".join(cells) + r" \\")
    head = (r"\begin{tabular}{l" + "rr" * 3 + "}\n\\toprule\n"
            r" & \multicolumn{2}{c}{Qwen3-8B} & \multicolumn{2}{c}{gpt-4o-mini} & \multicolumn{2}{c}{gpt-4.1-mini} \\" + "\n"
            r"Defense & AgentDojo & Replay & AgentDojo & Replay & AgentDojo & Replay \\" + "\n\\midrule\n")
    open(f"{TAB}/rq3_oracle.tex", "w").write("% generated by analysis/make_figs.py\n" + head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def tab_bounds():
    """plan.md §22.9: benign cost L / M / U per refusing defense and model, all48 (appendix app:bounds)."""
    rb = json.load(open(f"{OUTD}/refusal_bounds.json"))["models"]
    order = ["amlp", "progent", "agentsentry", "camel", "melon", "tripwire", "pi_detector", "block_all"]
    mods = [x for x in NAME if x in rb]
    rows = []
    for arm in order:
        cells = []
        for m in mods:
            e = rb[m].get(arm)
            cells.append("--" if e is None else f"{P(100 * e['cost_L'], 1)} / {P(100 * e['cost_M'], 1)} / {P(100 * e['cost_U'], 1)}")
        nm = r"\textbf{Calibrated allowlist} (ours)" if arm == "amlp" else ARM_NAME[arm]
        rows.append(nm + " & " + " & ".join(cells) + r" \\")
    head = (r"\begin{tabular}{l" + "r" * len(mods) + "}" + "\n\\toprule\n"
            r"Defense & " + " & ".join(NAME[m] for m in mods) + r" \\" + "\n\\midrule\n")
    open(f"{TAB}/rq3_bounds.tex", "w").write(
        "% generated by analysis/make_figs.py\n" + head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def tab_app_llama():
    """Appendix only (project lead): Llama-3.1-8B fails the tool-call parse gate (analysis/s3_parse_rate.py), so its
    numbers leave the main tables. Per pipeline stage the share of runs whose final answer holds an unexecuted tool-call
    JSON object, then its benign cost per defense and its utility drop, from the four-model run in analysis/out_all4."""
    pr = json.load(builtins.open(f"{ROOT}/data/analysis_out/s3_parse_rate.json"))["files"]
    stage = {"mine": "Mining", "monitor": "Holdout monitor", "online_llama31-8b-local__main": "Online, main defenses",
             "online_llama31-8b-local__agentsentry": "Online, Agent-Sentry"}
    rows = []
    for f, x in pr.items():
        if "llama31-8b-local" not in f:
            continue
        base = os.path.basename(f).split(".")[0]
        nm = next((v for k, v in stage.items() if base.startswith(k)), base)
        nm = "AgentDyn " + nm[0].lower() + nm[1:] if "/a2/" in f else nm
        runs = f"{x['runs']:,}".replace(",", "{,}")
        rows.append(f"{nm} & {runs} & {P(100 * x['call_rate'])} & {P(100 * x['unparsed_rate'])} \\\\")
    r3 = json.load(builtins.open(f"{ROOT}/data/analysis_out_all4/all48/rq34.json"))["rq3"]["llama31-8b-local"]
    c1 = json.load(builtins.open(f"{ROOT}/data/analysis_out_all4/all48/tests.json"))["C1"]["per_model"]["llama31-8b-local"]
    order = ["amlp", "progent", "agentsentry", "melon", "tripwire", "tool_filter", "sandwich", "spotlighting", "pi_detector", "block_all"]
    cost = [(r"\textbf{Calibrated allowlist} (ours)" if a == "amlp" else ARM_NAME[a]) + f" & {P(100 * r3[a]['benign_cost'], 1)} \\\\"
            for a in order if r3.get(a)]
    builtins.open(f"{TAB}/app_llama.tex", "w").write(_strip(
        "% generated by analysis/make_figs.py (tab_app_llama), four-model run analysis/out_all4, all 48 holdout tasks\n"
        "\\resizebox{\\linewidth}{!}{\\begin{tabular}{lrrr}\n\\toprule\nStage & Runs & Parsed (\\%) $\\uparrow$ & Unparsed (\\%) $\\downarrow$ \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\n\\medskip\n\n"
        "\\resizebox{\\linewidth}{!}{\\begin{tabular}{lr}\n\\toprule\nDefense on Llama-3.1-8B & Benign cost (pp) $\\downarrow$ \\\\\n\\midrule\n"
        + "\n".join(cost) + f"\n\\midrule\nUtility drop of the calibrated allowlist (pp) & {P(100 * c1['drop'], 1)} \\\\\n\\bottomrule\n\\end{{tabular}}}}\n"))


def tab_oracle_s5():
    """S5 (12eacf8a8, all 48 holdout tasks): accuracy of each oracle against blind labels and the rank of every
    defense under the execution oracle of the main table and under the repaired oracle (project lead)."""
    S5 = f"{ROOT}/data/analysis_out/s5/all48"
    L = json.load(builtins.open(f"{S5}/label_eval.json"))["oracles"]["all"]
    R = json.load(builtins.open(f"{S5}/rankings.json"))
    names = {"exec": "Execution", "agentdojo": "AgentDojo recorded", "replay_any": "Replay, any point (main text)"}
    acc = [f"{names[o]} & {P(100 * L[o]['accuracy'][0])} ({P(100 * L[o]['accuracy'][1])}--{P(100 * L[o]['accuracy'][2])}) & {P(100 * L[o]['accuracy_population_weighted'])} \\\\"
           for o in ("exec", "agentdojo", "replay_any")]
    mods = [m for m in NAME if m in R["exec"]]
    rank = {o: {mm: {d: i + 1 for i, d in enumerate(sorted(R[o][mm], key=lambda d: -R[o][mm][d]["block_rate"]))} for mm in mods} for o in ("exec", "replay_any")}
    alld = {d for mm in mods for d in R["exec"][mm]}      # CaMeL runs on the two hosted models only
    defs = sorted(alld, key=lambda d: sum(rank["exec"][mm][d] for mm in mods if d in rank["exec"][mm]) / sum(d in rank["exec"][mm] for mm in mods))
    rows = []
    for d in defs:        # project lead 2a: interception in percent under both oracles instead of ranks
        nm = r"\textbf{Calibrated allowlist} (ours)" if d == "amlp" else ARM_NAME.get(d, d)
        cells = []
        for mm in mods:
            if d not in R["exec"][mm]:
                cells.append("--")
                continue
            a, b = 100 * R["exec"][mm][d]["block_rate"], 100 * R["replay_any"][mm][d]["block_rate"]
            cells.append(f"{P(a)} / {P(b)}")
        rows.append(nm + " & " + " & ".join(cells) + r" \\")
    # [t] tabulars: both parts start at the top of the float (project lead)
    builtins.open(f"{TAB}/oracle_s5.tex", "w").write(_strip(
        "% generated by analysis/make_figs.py (tab_oracle_s5) from analysis/out/s5/all48\n"
        "\\resizebox{0.5\\textwidth}{!}{\\begin{tabular}[t]{lrr}\n\\toprule\nOracle & Sample accuracy, \\% (95\\% CI) $\\uparrow$ & Weighted accuracy, \\% $\\uparrow$ \\\\\n\\midrule\n"
        + "\n".join(acc) + "\n\\bottomrule\n\\end{tabular}}\\hfill\n"
        "\\resizebox{0.46\\textwidth}{!}{\\begin{tabular}[t]{l" + "r" * len(mods) + "}\n\\toprule\nInterception, \\% $\\uparrow$ & " + " & ".join(NAME[m] for m in mods)
        + " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n"))


def s4_fidelity():
    """S4 (exp_s4/AMENDMENT.md e535038d1): per persona and model, the seen-task benign block rate of the automaton mined
    on about 400 traces and on the 8-runs-per-task corpus, with 95% Wilson intervals, and the mining-size curve."""
    import builtins
    d = json.load(builtins.open(f"{ROOT}/data/analysis_out/s4/fidelity.json"))
    W = lambda x: f"{P(100 * x['rate'])} ({P(100 * x['wilson95'][0])}--{P(100 * x['wilson95'][1])})"
    rows = []
    for mm, x in d["models"].items():
        for ps, y in x["personas"].items():
            rows.append(f"{NAME[mm]} & {ps} & {y['tasks']} & {y['mining_traces']} & {y['n']} & {W(y)} & {y['paper_corpus']['mining_traces']} & {W(y['paper_corpus'])} \\\\")
        rows.append(f"{NAME[mm]} & all four & & & {x['pooled']['n']} & {W(x['pooled'])} & & {W(x['paper_corpus_pooled'])} \\\\")
    builtins.open(f"{TAB}/s4_fidelity.tex", "w").write(_strip(
        "% generated by analysis/make_figs.py (s4_fidelity) from analysis/out/s4/fidelity.json\n"
        "\\resizebox{\\textwidth}{!}{\\begin{tabular}{llrrrrrr}\n\\toprule\n"
        "Model & Persona & Tasks & Mining traces & Test runs & Block rate, \\% $\\downarrow$ (95\\% Wilson CI) & "
        "Mining traces, 8 per task & Block rate, \\% $\\downarrow$ (95\\% Wilson CI) \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n"))
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for mm, x in d["models"].items():
        xs, ys, lo, hi = [], [], [], []
        for b in ("50", "100", "200", "all"):
            k = sum(p["curve"][b]["blocked"] for p in x["personas"].values())
            n = sum(p["curve"][b]["n"] for p in x["personas"].values())
            mt = int(b) if b != "all" else round(sum(p["mining_traces"] for p in x["personas"].values()) / len(x["personas"]))
            xs.append(mt); ys.append(100 * k / n)
        ax.plot(xs, ys, marker="o", ms=3, color=COL[mm], label=NAME[mm])
    ax.set_xlabel("mining traces per persona")
    ax.set_ylabel("benign block rate, seen tasks (%) ↓")
    ax.set_ylim(0, 80)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{FIG}/s4_curve.pdf", bbox_inches="tight")   # not via savefig(): runs after the per-set loop
    plt.close(fig)


def main():
    global SET, PRIMARY, OUTD
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)
    for SET in ("all48", "clean27"):
        PRIMARY, OUTD = SET == "all48", f"{ROOT}/data/analysis_out/{SET}"
        rq = json.load(open(f"{OUTD}/rq12.json"))
        fig_rq1(rq)
        fig_rq2(rq)
        tab_rq2(rq)
        tab_rq1(rq)
        q = json.load(open(f"{OUTD}/rq34.json"))
        fig_rq3(q); tab_rq3(q); tab_rq3_ci(q); fig_rq4(q); tab_rq4(q); fig_transfer(); tab_rq3_replay()
        tab_rq1_chain(rq); tab_rq4_poison(q); tab_tests(); tab_setup(); tab_oracle(); tab_bounds()
    import builtins
    tab_app_llama()
    tab_oracle_s5()
    s4_fidelity()
    for name, text in TABS["all48"].items():
        if name in MERGE:
            n0 = len(MARKS)
            text = merge(text, TABS["clean27"][name])
            for mk in MARKS[n0:]:
                mk["table"] = name
        builtins.open(f"{TAB}/{name}.tex", "w").write(_strip(text))
    json.dump(MARKS, builtins.open(f"{ROOT}/data/analysis_out/table_marks.json", "w"), indent=1)
    print(f"dagger rule: {len(MARKS)} marked, {len(SKIPPED)} differing cells skipped as count or composite")
    print("figs:", sorted(os.listdir(FIG)), "tables:", sorted(TABS["all48"]))


if __name__ == "__main__":
    main()
