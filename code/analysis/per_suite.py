"""Per-suite breakdown for RQ2 / RQ3 (descriptive, same data and definitions as analysis/rq34.py and rq12.py, plan
§14 pairing): block rate and benign cost of every defense per AgentDojo suite, pooled over the security models
(benign cost also over Llama), plus the share of AMLP-passed attacks that TripWire stops.
Writes data/analysis_out/<HOLDOUT_SET>/per_suite.json and (primary set only) outputs/figs/rq3_per_suite.pdf."""
import collections
import json
import os
import sys

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(D)   # release root (code/ and data/ are siblings)
sys.path.insert(0, D)
sys.path.insert(0, f"{D}/analysis")
import rq34 as R                      # noqa: E402  (same loaders, pairing and A2-rev filter)
import m3_common as C                 # noqa: E402
import holdout_set as HS              # noqa: E402

SUITES = ["banking", "slack", "travel", "workspace"]


def main():
    out = {"block": collections.defaultdict(dict), "cost": collections.defaultdict(dict)}
    for arm in R.ARMS:
        att = collections.defaultdict(lambda: [0, 0])          # suite -> [asr_arm, asr_none] counts
        ben = collections.defaultdict(lambda: [0, 0])          # suite -> [sum(none - arm), n]
        for m in C.MODELS:
            on, mon = R.online(m), R.monitor(m)
            kb, base, it = R.paired_benign(m, arm, on, mon)
            for k in kb:
                ben[k[0]][0] += base[k] - it[k]; ben[k[0]][1] += 1
            if m in R.SECURITY:
                ka, an, aa = R.paired_attack(arm, on)
                for k in ka:
                    att[k[0]][0] += aa[k]; att[k[0]][1] += an[k]
        for s in SUITES:
            if att[s][1]:
                out["block"][arm][s] = 1 - att[s][0] / att[s][1]
            if ben[s][1]:
                out["cost"][arm][s] = ben[s][0] / ben[s][1]
    tw = collections.Counter()
    for m in R.SECURITY:
        on = R.online(m)
        ka, an, aa = R.paired_attack("amlp", on)
        kt, _, t = R.paired_attack("tripwire", on)
        for k in set(ka) & set(kt):
            if t[k] == 1:                                       # attack passed TripWire
                tw["tripwire_missed"] += 1
                tw["amlp_stops_of_those"] += 1 - aa[k]
    out["tripwire_missed"] = dict(tw)
    json.dump(out, open(f"{HS.OUT}/per_suite.json", "w"), indent=1)
    if not HS.PRIMARY:                                          # the paper figure shows the primary set only
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = ["amlp", "progent", "agentsentry", "camel", "melon", "tripwire", "tool_filter", "sandwich", "spotlighting",
             "pi_detector", "block_all"]
    names = {"amlp": "AMLP (ours)", "progent": "Progent", "agentsentry": "Agent-Sentry", "camel": "CaMeL", "melon": "MELON",
             "tripwire": "TripWire", "tool_filter": "tool filter", "sandwich": "sandwich", "spotlighting": "spotlighting",
             "pi_detector": "PI detector", "block_all": "block-all"}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))
    for ax, key, title, cmap, vmin, vmax in ((axes[0], "block", "interception (%) ↑", "Greens", 0, 100),
                                              (axes[1], "cost", "benign cost (pp) ↓", "Reds", -10, 50)):
        M = [[100 * out[key][a].get(s, float("nan")) for s in SUITES] for a in order]
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(SUITES))); ax.set_xticklabels(SUITES, fontsize=7)
        ax.set_yticks(range(len(order))); ax.set_yticklabels([names[a] for a in order], fontsize=7)
        for i, row in enumerate(M):
            for j, v in enumerate(row):
                if v == v:
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6)
        ax.set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{ROOT}/outputs/figs/rq3_per_suite.pdf", bbox_inches="tight")
    print(json.dumps({"tripwire_missed": out["tripwire_missed"],
                      "amlp_block": {s: round(out['block']['amlp'].get(s, 0), 3) for s in SUITES}}, indent=0))


if __name__ == "__main__":
    main()
