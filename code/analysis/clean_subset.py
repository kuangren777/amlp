"""Pilot-clean robustness check (plan.md §15, pre-registered before computing).
Recomputes C3' (false-block test + recall side), C1 and the RQ3 per-arm interception / benign cost on the holdout
tasks that no pilot run touched, with the same estimators as the main analysis. Frozen M3 data, no new runs.
Writes data/analysis_out/clean_subset.json; macros via analysis/make_numbers.py."""
import json
import os
import sys

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(D)   # release root (code/ and data/ are siblings)
sys.path.insert(0, D)
sys.path.insert(0, f"{D}/analysis")
import m3_common as C       # noqa: E402
import m3_judge as MJ       # noqa: E402
import m3_analyze as MA     # noqa: E402
import rq34 as R            # noqa: E402
from extra_checks import pilot_test_tasks   # noqa: E402
import holdout_set as HS    # noqa: E402

CLEAN = C.HOLDOUT - pilot_test_tasks()


def keep(r):
    return (r["suite"], r["ut"]) in CLEAN


def main():
    mon, onl, judg = MA.load(False)
    mon, onl = [r for r in mon if keep(r)], [r for r in onl if keep(r)]
    judg = [j for j in judg if keep(j)]
    fb = MJ.c3_tests(judg, None)
    rec = MJ.recall_side(judg)
    res = {"n_clean_tasks": len(CLEAN), "c3prime_fb": fb, "recall": rec, "C1": MA.c1(mon, onl, None)}
    on0, mon0 = R.online, R.monitor
    R.online = lambda m: [r for r in on0(m) if keep(r)]
    R.monitor = lambda m: [r for r in mon0(m) if keep(r)]
    rq3, _ = R.rq3(False)
    res["rq3"] = {m: {a: {k: v[k] for k in ("benign_cost", "benign_cost_ci", "block_rate", "block_rate_ci", "n_benign", "n_attack")
                          if k in v} for a, v in x.items() if a != "amlp_vs_tripwire"} for m, x in rq3.items()}
    json.dump(res, open(f"{HS.OUT}/clean_subset.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("n_clean_tasks", "c3prime_fb", "recall", "C1")}, indent=1))


if __name__ == "__main__":
    main()
