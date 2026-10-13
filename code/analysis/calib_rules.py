"""A1 (plan.md §23, pre-registered e8fdb5951 before computing): the CRC selection rule against simpler calibration
rules on the §22.1 re-splits. Per split, model and chain element: calibration mean loss and CRC bound, mean loss over
selection + calibration tasks, holdout false-block rate and flag rate. Rules: (i) crc bound <= eps, (ii) mean
calibration loss <= eps, (iii) fixed (exact, r = all), (iv) mean loss over all non-holdout tasks <= eps.
Writes data/analysis_out/calib_rules.json.  Usage: python3 analysis/calib_rules.py [n_splits]"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import resplit as R            # noqa: E402  (sets HOLDOUT_SET and the import paths)
from resplit import C, ME, MS  # noqa: E402

EPS = R.EPS
RULES = ("crc", "empirical", "fixed", "seen")


def element_stats(full, H, S, Cal, pred, viol):
    pool = R.Split(full, H)
    vh = [v for v in viol if (v["suite"], v["ut"]) in H]
    hb_n = sum(len(full.ben[t]) for t in H)
    out = []
    for level, r in R.CHAIN:
        Lc = R.losses(pool, Cal, full.ben, H, pred, level, r)
        Ls = R.losses(pool, S | Cal, full.ben, H, pred, level, r)
        fb = fl = flbh = 0
        for t in C.sorted_tasks(H):
            env = ME.build(pool, *t, exclude_tasks=H, predictor=pred, r=r, level=level)
            fb += sum(ME.first_block(env, o["trace"]) is not None for o in full.ben[t])
            for v in vh:
                if (v["suite"], v["ut"]) == t:
                    f = ME.first_block(env, v["trace"])
                    fl += f is not None
                    flbh += f is not None and (R.HARM[v["row_id"]] is None or f <= R.HARM[v["row_id"]])
        out.append({"level": level, "r": r, "cal_mean": float(np.mean(Lc)), "cal_bound": MS.crc_bound(Lc),
                    "seen_mean": float(np.mean(Ls)), "holdout_fb": fb / hb_n,
                    "flag": fl / len(vh) if vh else None, "flag_bh": flbh / len(vh) if vh else None, "flag_n": len(vh)})
    return out


def pick(chain, rule, eps):
    if rule == "fixed":
        return next(i for i, c in enumerate(chain) if c["level"] == "exact" and c["r"] == 1000)
    key = {"crc": "cal_bound", "empirical": "cal_mean", "seen": "seen_mean"}[rule]
    return next((i for i, c in enumerate(chain) if c[key] <= eps), len(chain) - 1)


def summarize(rows, eps):
    out = {}
    for rule in RULES:
        sel = [ch[pick(ch, rule, eps)] for ch in rows]
        kept = [c["level"] != "any" for c in sel]
        k = sum(kept)
        fl = [c["flag"] for c in sel if c["flag"] is not None]
        out[rule] = {"kept": k, "n": len(rows), "kept_ci95": R.wilson(k, len(rows)),
                     "fb_mean": float(np.mean([c["holdout_fb"] for c in sel])),
                     "fb_over_eps": sum(c["holdout_fb"] > eps for c in sel),
                     "flag_mean": float(np.mean(fl)) if fl else None}
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else R.N_DEFAULT
    F = R.folds()
    rng = np.random.default_rng(R.SEED)
    splits = [(C.HOLDOUT, C.SELECTION, C.CALIBRATION)] + [R.draw(rng, F) for _ in range(n)]
    import flag_before_harm as FB
    gt = FB.ground_truth()
    res = {"meta": {"seed": R.SEED, "n_splits": n, "eps": EPS, "chain": R.CHAIN, "rules": RULES,
                    "plan": "§23 e8fdb5951"}, "models": {}}
    for m in C.MODELS:
        pred = R.sel_entry(m, MS.PRIMARY_EPS)["chosen"]["predictor"]
        full = R.pool_for(m)
        viol = R.holdout_violations(m) if m in R.SECURITY else []
        for v in viol:
            R.HARM[v["row_id"]] = FB.harm_index([(c["f"], c["a"]) for c in v["trace"]], gt[(v["suite"], v["it"])])[0]
        orig = element_stats(full, *[set(x) for x in splits[0]], pred, viol)
        rows = []
        for i, (H, S, Cal) in enumerate(splits[1:], 1):
            rows.append(element_stats(full, set(H), set(S), set(Cal), pred, viol))
            if i % 10 == 0:
                print(m, i, flush=True)
        res["models"][m] = {"predictor": pred, "original_split": orig,
                            "original_pick": {str(e): {r: R.CHAIN[pick(orig, r, e)] for r in RULES} for e in EPS},
                            "summary": {str(e): summarize(rows, e) for e in EPS}, "splits": rows}
        print(m, json.dumps({r: (s["kept"], s["fb_over_eps"]) for r, s in res["models"][m]["summary"]["0.1"].items()}),
              flush=True)
    json.dump(res, open(f"{R.ROOT}/data/analysis_out/calib_rules.json", "w"), indent=1)


if __name__ == "__main__":
    main()
