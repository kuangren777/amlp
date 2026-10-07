"""Data-volume learning curve for the seen-vs-novel gap (plan.md §17, pre-registered 43bbccf8 before computing).
SEEN-k: full pool + first k other benign reps of the evaluated holdout task (k in 1,2,4,6,7).
NOVEL-j: pool with the first j reps per mined task (j in 2,4,6,8), no own runs.
M2 (Praetor-style pDFA) and M1a (AMLP exact, r = all), same code as analysis/seen_novel.py.
Writes data/analysis_out/seen_novel_curve.json."""
from __future__ import annotations

import collections
import json
import os
import sys

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(D)   # release root (code/ and data/ are siblings)
sys.path.insert(0, f"{D}/analysis")
import seen_novel as S                     # noqa: E402  (same pool, embedder, pDFA, AMLP construction)
from seen_novel import C, ME, MS, SIDE, sel_entry, holdout_benign  # noqa: E402

KS, JS = (1, 2, 4, 6, 7), (2, 4, 6, 8)


def main():
    emb = S.Embedder()
    blk = collections.defaultdict(lambda: collections.defaultdict(list))   # setting -> task -> [0/1]
    for m in C.MODELS:
        e = sel_entry(m, MS.PRIMARY_EPS)
        pred = e["chosen"]["predictor"]
        pool = ME.Pool(m, False, k=e["k"])
        C.assert_pool_clean(pool.tasks(), "analysis/seen_novel_curve")
        ben = holdout_benign(m)
        own = collections.defaultdict(list)
        for b in ben:
            own[(b["suite"], b["ut"])].append(b)
        for v in own.values():
            v.sort(key=lambda r: r["rep"])
        emb.fill(S.all_strings([r for rs in pool.ben.values() for r in rs] + ben))
        by_suite = {s: {k: v for k, v in pool.ben.items() if k[0] == s} for s in SIDE}
        novel = {j: {s: {t: rs[:j] for t, rs in by_suite[s].items()} for s in SIDE} for j in JS}
        novel_pdfa = {j: {s: S.PDFA([r["trace"] for rs in novel[j][s].values() for r in rs], emb) for s in SIDE} for j in JS}
        cache = {}
        for b in ben:
            t = (b["suite"], b["ut"])
            for j in JS:
                blk[f"M2_novel{j}"][t].append(int(novel_pdfa[j][t[0]].first_block(b["trace"], emb) is not None))
                blk[f"M1a_novel{j}"][t].append(int(ME.first_block(S.amlp_env(novel[j][t[0]], *t, pred, "exact"), b["trace"]) is not None))
            others = [r for r in own[t] if r["rep"] != b["rep"]]
            for k in KS:
                mined = dict(by_suite[t[0]]); mined[t] = others[:k]
                ck = (m, t, tuple(r["rep"] for r in others[:k]))
                if ck not in cache:
                    cache[ck] = S.PDFA([r["trace"] for rs in mined.values() for r in rs], emb)
                blk[f"M2_seen{k}"][t].append(int(cache[ck].first_block(b["trace"], emb) is not None))
                blk[f"M1a_seen{k}"][t].append(int(ME.first_block(S.amlp_env(mined, *t, pred, "exact"), b["trace"]) is not None))
        print(m, "done", flush=True)
    res = {"fb": {}}
    for k_, vals in blk.items():
        p, lo, hi = C.cluster_bootstrap(vals)
        res["fb"][k_] = {"point": p, "ci95": [lo, hi]}
    for meth in ("M2", "M1a"):
        d = {t: [a - b for a, b in zip(blk[f"{meth}_novel8"][t], blk[f"{meth}_seen1"][t])] for t in blk[f"{meth}_novel8"]}
        p, lo, hi = C.cluster_bootstrap(d)
        res[f"{meth}_novel8_minus_seen1"] = {"point": p, "ci95": [lo, hi]}
        res[f"{meth}_novel4_minus_novel8"] = res["fb"][f"{meth}_novel4"]["point"] - res["fb"][f"{meth}_novel8"]["point"]
    r = res
    r["M2_confound_ruled_out"] = (r["M2_novel8_minus_seen1"]["point"] >= 0.05 and r["M2_novel8_minus_seen1"]["ci95"][0] > 0
                                  and r["M2_novel4_minus_novel8"] < 0.05)
    json.dump(res, open(f"{ROOT}/data/analysis_out/seen_novel_curve.json", "w"), indent=1)
    print(json.dumps({k: (v if not isinstance(v, dict) or "point" in v else {kk: round(100 * vv["point"], 1) for kk, vv in v.items()})
                      for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
