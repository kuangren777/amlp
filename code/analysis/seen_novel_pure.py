"""Pure-mined ablation of the AMLP parts (plan.md §18, pre-registered 345c635e before computing).
Same corpora, settings, metrics and bootstrap as analysis/seen_novel.py (§16).
  M1a-pure : exact value check with mined values only (no request literals, no environment entities), tool check off
  M1a-req  : exact value check with request literals and environment entities only, tool check off
  M1b-pure : tool layer = side-effecting tools of the mined runs only (no LLM predictor), value layer off
  M1b-pred : tool layer = LLM-predicted tools only, value layer off (corpus-independent, reported once)
Writes data/analysis_out/seen_novel_pure.json."""
from __future__ import annotations

import collections
import json
import os
import sys

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(D)   # release root (code/ and data/ are siblings)
sys.path.insert(0, f"{D}/analysis")
import seen_novel as S                        # noqa: E402
from seen_novel import C, ME, MS, SIDE, EB, sel_entry, holdout_benign, holdout_violations, SECURITY  # noqa: E402


def variant(name, mined, suite, ut, pred):
    if name in ("M1a-pure", "M1a-req"):
        e = S.amlp_env(mined if name == "M1a-pure" else {}, suite, ut, pred, "exact")
        e.tools = set(SIDE[suite])                              # tool check off
        if name == "M1a-pure":
            e.prompt, e.env_text = "", ""                      # mined values only
        return e
    if name == "M1b-pure":
        e = S.amlp_env(mined, suite, ut, pred, "any")
        e.tools = {c["f"] for rs in mined.values() for o in rs for c in o["trace"] if c["f"] in SIDE[suite]}
        return e
    if name == "M1b-pred":
        e = S.amlp_env({}, suite, ut, pred, "any")
        e.tools = set(EB.PRED[pred][f"{suite}|{ut}"])
        return e
    raise ValueError(name)


VARS = ("M1a-pure", "M1a-req", "M1b-pure", "M1b-pred")


def main():
    tot = collections.defaultdict(lambda: [0, 0])
    blk = collections.defaultdict(lambda: collections.defaultdict(list))
    for m in C.MODELS:
        e = sel_entry(m, MS.PRIMARY_EPS)
        pred = e["chosen"]["predictor"]
        pool = ME.Pool(m, False, k=e["k"])
        C.assert_pool_clean(pool.tasks(), "analysis/seen_novel_pure")
        ben, viol = holdout_benign(m), (holdout_violations(m) if m in SECURITY else [])
        own = collections.defaultdict(list)
        for b in ben:
            own[(b["suite"], b["ut"])].append(b)
        by_suite = {s: {k: v for k, v in pool.ben.items() if k[0] == s} for s in SIDE}

        def judge(kind, run, seen_runs):
            t = (run["suite"], run["ut"])
            novel = by_suite[t[0]]
            seen = dict(novel); seen[t] = seen_runs
            for v in VARS:
                bn = ME.first_block(variant(v, novel, *t, pred), run["trace"]) is not None
                bs = ME.first_block(variant(v, seen, *t, pred), run["trace"]) is not None
                tot[(v, "novel", kind)][0] += bn; tot[(v, "novel", kind)][1] += 1
                tot[(v, "seen", kind)][0] += bs; tot[(v, "seen", kind)][1] += 1
                blk[f"{v}_{kind}"][t].append(int(bn) - int(bs) if kind == "fb" else int(bs) - int(bn))
        for b in ben:
            judge("fb", b, [r for r in own[(b["suite"], b["ut"])] if r["rep"] != b["rep"]])
        for v in viol:
            judge("flag", v, own[(v["suite"], v["ut"])])
        print(m, "done", flush=True)
    res = {"pooled": {f"{a}_{b}_{c}": x[0] / x[1] for (a, b, c), x in tot.items()}, "delta": {}}
    for k, vals in blk.items():
        p, lo, hi = C.cluster_bootstrap(vals)
        res["delta"][k] = {"point": p, "ci95": [lo, hi]}
    res["H_gap_descriptive"] = {v: res["delta"][f"{v}_fb"]["point"] >= 0.05 and res["delta"][f"{v}_fb"]["ci95"][0] > 0 for v in VARS}
    json.dump(res, open(f"{ROOT}/data/analysis_out/seen_novel_pure.json", "w"), indent=1)
    print(json.dumps({"pooled": {k: round(100 * v, 1) for k, v in res["pooled"].items()},
                      "delta": {k: [round(100 * v["point"], 1)] + [round(100 * x, 1) for x in v["ci95"]] for k, v in res["delta"].items()},
                      "H": res["H_gap_descriptive"]}, indent=1))


if __name__ == "__main__":
    main()
