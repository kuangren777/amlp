"""Calibration-split robustness (plan.md §22.1, pre-registered a703606e4 before computing).
100 stratified re-splits of the 97 tasks with the original per-suite fold sizes (numpy seed 20261010, split 0 = the
original split). Per split and model: CRC along m3_select.CHAIN on the split's calibration tasks with the model's
originally selected predictor (selection fold not re-run), the value-layer verdict at eps in {0.05, 0.10, 0.15}, and
the holdout false-block and flag rate at lambda-hat. Writes data/analysis_out/resplit.json.
Usage: python3 analysis/resplit.py [n_splits]"""
from __future__ import annotations

import collections
import json
import math
import os
import sys

os.environ["HOLDOUT_SET"] = "all48"                 # no row filtering; folds are re-drawn below
import numpy as np                                  # noqa: E402

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(D)   # release root (code/ and data/ are siblings)
sys.path.insert(0, D)
sys.path.insert(0, f"{D}/analysis")
import m3_common as C          # noqa: E402
import m3_envelope as ME       # noqa: E402
import m3_select as MS         # noqa: E402
from rq12 import sel_entry, holdout_violations, SECURITY  # noqa: E402

SEED, N_DEFAULT, K = 20261010, 100, 8
EPS = (0.05, 0.10, 0.15)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def folds():
    """Original folds per suite as sorted task lists."""
    h = json.load(open(f"{D}/holdout_split.json"))
    return {s: {f: sorted(h[f][s], key=lambda u: int(u.split("_")[-1])) for f in ("holdout", "selection", "calibration")}
            for s in h["holdout"]}


def draw(rng, F):
    H, S, Cal = set(), set(), set()
    for s, f in sorted(F.items()):
        tasks = sorted(f["holdout"] + f["selection"] + f["calibration"], key=lambda u: int(u.split("_")[-1]))
        perm = [tasks[i] for i in rng.permutation(len(tasks))]
        a, b = len(f["holdout"]), len(f["holdout"]) + len(f["selection"])
        H |= {(s, u) for u in perm[:a]}
        S |= {(s, u) for u in perm[a:b]}
        Cal |= {(s, u) for u in perm[b:]}
    return frozenset(H), frozenset(S), frozenset(Cal)


def pool_for(model):
    """Benign runs of every task: mining runs for original non-holdout tasks, monitor runs for original holdout."""
    mine = [r for r in C.read_jsonl(C.rows_path(False, "mine", model)) if (r["suite"], r["ut"]) not in C.HOLDOUT]
    mon = [r for r in C.read_jsonl(C.rows_path(False, "monitor", model)) if (r["suite"], r["ut"]) in C.HOLDOUT]
    full = ME.Pool(model, k=K, rows=mine + mon)
    assert len(full.tasks()) == 97, len(full.tasks())
    return full


class Split:
    """Pool restricted to the split's non-holdout tasks (benign runs of holdout tasks never mined)."""

    def __init__(self, full, H):
        self.ben = {t: v for t, v in full.ben.items() if t not in H}
        self.k = full.k


def losses(pool, tasks, runs_of, H, pred, level, r):
    out = []
    for t in C.sorted_tasks(tasks):
        env = ME.build(pool, *t, exclude_tasks=H, predictor=pred, r=r, level=level)
        runs = runs_of[t]
        if runs:
            out.append(sum(ME.first_block(env, o["trace"]) is not None for o in runs) / len(runs))
    return out


def run_split(full, H, Cal, pred, viol):
    pool = Split(full, H)
    chain = []
    for level, r in MS.CHAIN:
        L = losses(pool, Cal, full.ben, H, pred, level, r)
        chain.append({"level": level, "r": r, "mean_loss": sum(L) / len(L), "bound": MS.crc_bound(L), "n": len(L)})
    res = {"chain": chain}
    for eps in EPS:
        feas = [c for c in chain if c["bound"] <= eps]
        lv, r = (feas[0]["level"], feas[0]["r"]) if feas else MS.CHAIN[-1]
        hb = [o for t in H for o in full.ben[t]]
        fb_k = 0
        cache = {}
        for t in C.sorted_tasks(H):
            cache[t] = ME.build(pool, *t, exclude_tasks=H, predictor=pred, r=r, level=lv)
            fb_k += sum(ME.first_block(cache[t], o["trace"]) is not None for o in full.ben[t])
        vh = [v for v in viol if (v["suite"], v["ut"]) in H]
        fl = sum(ME.first_block(cache[(v["suite"], v["ut"])], v["trace"]) is not None for v in vh)
        res[str(eps)] = {"feasible": bool(feas), "level": lv, "r": r, "keeps_value_layer": bool(feas) and lv != "any",
                         "holdout_fb": fb_k / len(hb), "holdout_fb_n": len(hb),
                         "flag": fl / len(vh) if vh else None, "flag_n": len(vh)}
    return res


def diagnostic(full, pred):
    """Mean exact-level (r = all) loss by run source on the original split's non-holdout pool."""
    pool = Split(full, C.HOLDOUT)
    out = {}
    for name, tasks in (("monitor_runs_original_holdout", C.HOLDOUT), ("mining_runs_original_calibration", C.CALIBRATION)):
        L = losses(pool, tasks, full.ben, C.HOLDOUT, pred, "exact", 1000)
        out[name] = {"mean_loss": sum(L) / len(L), "n_tasks": len(L)}
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_DEFAULT
    F = folds()
    rng = np.random.default_rng(SEED)
    splits = [(C.HOLDOUT, C.SELECTION, C.CALIBRATION)] + [draw(rng, F) for _ in range(n)]
    res = {"meta": {"seed": SEED, "n_splits": n, "k": K, "eps": EPS, "chain": MS.CHAIN, "plan": "§22.1 a703606e4"},
           "models": {}}
    for m in C.MODELS:
        e = sel_entry(m, MS.PRIMARY_EPS)
        pred = e["chosen"]["predictor"]
        full = pool_for(m)
        viol = holdout_violations(m) if m in SECURITY else []
        orig = run_split(full, *splits[0][:1], splits[0][2], pred, viol)
        rows = []
        for i, (H, _, Cal) in enumerate(splits[1:], 1):
            rows.append(run_split(full, H, Cal, pred, viol))
            if i % 10 == 0:
                print(m, i, flush=True)
        summ = {}
        for eps in EPS:
            k = sum(x[str(eps)]["keeps_value_layer"] for x in rows)
            kept = [x[str(eps)] for x in rows if x[str(eps)]["keeps_value_layer"]]
            summ[str(eps)] = {
                "kept": k, "n": len(rows), "kept_share": k / len(rows), "kept_ci95": wilson(k, len(rows)),
                "lambda_hist": dict(collections.Counter(f"{x[str(eps)]['level']},{x[str(eps)]['r']}" for x in rows)),
                "fb_mean_all": float(np.mean([x[str(eps)]["holdout_fb"] for x in rows])),
                "fb_mean_kept": float(np.mean([x["holdout_fb"] for x in kept])) if kept else None,
                "flag_mean_kept": float(np.mean([x["flag"] for x in kept if x["flag"] is not None])) if any(
                    x["flag"] is not None for x in kept) else None,
                "flag_mean_dropped": float(np.mean([x[str(eps)]["flag"] for x in rows if not x[str(eps)]["keeps_value_layer"]
                                                     and x[str(eps)]["flag"] is not None])) if m in SECURITY else None}
        res["models"][m] = {"predictor": pred, "original_split": orig, "published_chosen": e["chosen"],
                            "summary": summ, "diagnostic_run_source": diagnostic(full, pred), "splits": rows}
        print(m, json.dumps({k: (v["kept"], v["n"]) for k, v in summ.items()}), flush=True)
    json.dump(res, open(f"{ROOT}/data/analysis_out/resplit.json", "w"), indent=1)


if __name__ == "__main__":
    main()
