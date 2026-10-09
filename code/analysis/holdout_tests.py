"""Pre-registered tests on the active holdout set (HOLDOUT_SET, analysis/holdout_set.py), same functions as the
frozen run: C3' false-block test and recall side (m3_judge.c3_tests / recall_side, plan.md A3, §11.3) and C1
(m3_analyze.c1, plan.md A4(a)). The pair-count guards of the frozen run (48-task n) are not applied to clean27.
With HOLDOUT_SET=all48 the result must reproduce data/m3/judge_tests.json and results_m3_analyze.txt exactly; the
script asserts this (provenance check). Writes data/analysis_out/<HOLDOUT_SET>/tests.json.
Usage: HOLDOUT_SET=clean27|all48 python3 analysis/holdout_tests.py"""
import json
import os
import sys

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(D)   # release root (code/ and data/ are siblings)
sys.path.insert(0, D)
sys.path.insert(0, f"{D}/analysis")
import holdout_set as HS    # noqa: E402  (patches m3_common.read_jsonl before the loaders run)
import m3_common as C       # noqa: E402
import m3_judge as MJ       # noqa: E402
import m3_analyze as MA     # noqa: E402


def frozen_c1():
    txt = open(f"{ROOT}/data/results_m3_analyze.txt").read()
    return json.loads(txt[txt.index("{"): txt.rindex("}") + 1][: txt[txt.index("{"):].index("\n}") + 2])["C1"]


def main():
    mon, onl, judg = MA.load(False)
    tasks = {(r["suite"], r["ut"]) for r in mon}
    assert tasks == set(HS.ACTIVE), (len(tasks), len(HS.ACTIVE))
    fb = MJ.c3_tests(judg, None)
    rec = MJ.recall_side(judg)
    res = {"holdout_set": HS.NAME, "n_tasks": len(tasks), "c3prime_fb": fb, "recall_A2rev": rec,
           "recall_all35": MJ.recall_side(judg, True), "C3prime": MJ.c3prime_decision(fb, rec),
           "C1": MA.c1(mon, onl, None)}
    if not HS.PRIMARY:                                   # provenance: the frozen 48-task run is reproduced
        fr = json.load(open(f"{ROOT}/data/m3/judge_tests.json"))
        for k in ("c3prime_fb", "recall_A2rev", "recall_all35"):
            assert json.dumps(res[k], sort_keys=True) == json.dumps(fr[k], sort_keys=True), k
        c1 = frozen_c1()
        for k in ("n_pairs", "drop_pooled", "upper95"):
            assert abs(res["C1"][k] - c1[k]) < 1e-12, (k, res["C1"][k], c1[k])
        res["reproduces_frozen"] = True
    json.dump(res, open(f"{HS.OUT}/tests.json", "w"), indent=1, default=str)
    print(json.dumps({k: res[k] for k in ("holdout_set", "n_tasks", "C3prime")}, default=str))
    print("fb", {k: fb[k] for k in ("n_pairs", "fb_amlp", "fb_progent", "mcnemar_p", "d_ci95")})
    print("rec", {k: rec.get(k) for k in ("n_violations", "recall_amlp", "recall_progent", "recall_diff_ci95")})
    print("C1", {k: res["C1"].get(k) for k in ("n_pairs", "drop_pooled", "upper95", "holds")})


if __name__ == "__main__":
    main()
