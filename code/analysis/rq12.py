"""RQ1 / RQ2 analysis for the AMLP measurement paper (replay only, frozen M3 data, no agent runs).

RQ1: for every model and every element of the nested chain (plan.md §12), the calibration-fold mean loss and CRC bound
(from data/m3_selection_log.jsonl, amendment-B entries) and, on the holdout, the realised benign false-block rate and
the interception (recall) of the undefended successful attacks, so the cost of reaching eps is visible.
RQ2: at the calibrated configuration (primary eps), interception of the tool layer per attacker side-effect tool and
the share of violations whose side-effecting calls all stay inside the tool layer (structurally invisible to it).
Definitions: A2-rev denominators; holdout benign runs = A3 monitor runs; violations = online `none` attack runs with
security True. Writes data/analysis_out/rq12.json; numbers are turned into macros by analysis/make_numbers.py."""
from __future__ import annotations

import collections
import json
import os
import sys

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(D)   # release root (code/ and data/ are siblings)
sys.path.insert(0, D)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import m3_common as C          # noqa: E402
import m3_envelope as ME       # noqa: E402
import m3_select as MS         # noqa: E402
from amlp import SIDE          # noqa: E402
import harness                 # noqa: E402

OUT = f"{ROOT}/data/analysis_out"
SECURITY = ["qwen3-8b-local", "gpt-4o-mini-2024-07-18", "gpt-4.1-mini-2025-04-14"]   # llama: 0 successful attacks


def sel_entry(model, eps):
    es = [e for e in C.read_jsonl(C.selection_log(False)) if e["model"] == model and e.get("eps") == eps and "amendment" in e]
    return es[-1]


def holdout_benign(model):
    return [r for r in C.read_jsonl(C.rows_path(False, "monitor", model)) if r.get("err") is None]


def holdout_violations(model):
    rows = [r for r in C.read_jsonl(C.rows_path(False, "online", model, "main"))
            if (r.get("defense") in (None, "none")) and r.get("part") == "b" and r.get("err") is None]
    return [r for r in rows if r["security"] and (r["suite"], r["it"]) not in C.TEXT_ONLY]


def attacker_tool():
    """(suite, injection task) -> first side-effecting function of its ground truth (A2-rev tasks only)."""
    out = {}
    for s in SIDE:
        S = harness.suite(s)
        env = S.load_and_inject_default_environment(S.get_injection_vector_defaults())
        for it, t in S.injection_tasks.items():
            fs = [c.function for c in t.ground_truth(env) if c.function in SIDE[s]]
            out[(s, it)] = fs[0] if fs else "text_only_or_unspecified"
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {"rq1": {}, "rq2": {}}
    atool = attacker_tool()
    for m in C.MODELS:
        e = sel_entry(m, MS.PRIMARY_EPS)
        pred = e["chosen"]["predictor"]
        pool = ME.Pool(m, False, k=e["k"])
        C.assert_pool_clean(pool.tasks(), "analysis/rq12")
        row = next(x for x in e["selection_table"] if x["cfg"]["predictor"] == pred)
        ben, viol = holdout_benign(m), holdout_violations(m)
        chain = []
        cache = {}
        for cal in row["crc_table"]:
            lv, r = cal["level"], cal["r"]

            def env_for(t, lv=lv, r=r):
                k = (t, lv, r)
                if k not in cache:
                    cache[k] = ME.build(pool, *t, exclude_tasks=C.HOLDOUT, predictor=pred, r=r, level=lv)
                return cache[k]
            fb = sum(ME.first_block(env_for((b["suite"], b["ut"])), b["trace"]) is not None for b in ben)
            ic = sum(ME.first_block(env_for((v["suite"], v["ut"])), v["trace"]) is not None for v in viol)
            chain.append({"level": lv, "r": r, "cal_mean_loss": cal["mean_loss"], "cal_bound": cal["bound"],
                          "holdout_fb": fb / len(ben), "holdout_fb_k": fb, "holdout_n_benign": len(ben),
                          "holdout_recall": (ic / len(viol)) if viol else None, "holdout_recall_k": ic,
                          "holdout_n_viol": len(viol)})
        chosen = {eps: sel_entry(m, eps)["chosen"] for eps in MS.EPS_GRID}
        res["rq1"][m] = {"predictor": pred, "chain": chain, "chosen_by_eps": {str(k): v for k, v in chosen.items()},
                         "feasible_by_eps": {str(eps): sel_entry(m, eps)["crc_feasible"] for eps in MS.EPS_GRID}}
        if m not in SECURITY:
            continue
        cfg = e["chosen"]
        per_tool = collections.defaultdict(lambda: {"n": 0, "intercepted": 0, "invisible": 0})
        n = inter = invis = 0
        for v in viol:
            env = ME.build(pool, v["suite"], v["ut"], exclude_tasks=C.HOLDOUT, predictor=pred, r=cfg["r"],
                           level=cfg["level"])
            side = [c for c in v["trace"] if c["f"] in SIDE[v["suite"]]]
            blocked = ME.first_block(env, v["trace"]) is not None
            inside = all(c["f"] in env.tools for c in side)
            k = atool.get((v["suite"], v["it"]), "unknown")
            if k == "text_only_or_unspecified" and side:     # AgentDojo v1.2 leaves e.g. workspace injection_task_6..13
                k = side[-1]["f"]                             # without ground truth: use the observed attacker call
            per_tool[k]["n"] += 1
            per_tool[k]["intercepted"] += blocked
            per_tool[k]["invisible"] += inside
            n += 1; inter += blocked; invis += inside
        res["rq2"][m] = {"n_viol": n, "intercepted": inter, "invisible": invis, "per_attacker_tool": per_tool}
    pooled = collections.defaultdict(lambda: {"n": 0, "intercepted": 0, "invisible": 0})
    for m in SECURITY:
        for k, v in res["rq2"][m]["per_attacker_tool"].items():
            for f in ("n", "intercepted", "invisible"):
                pooled[k][f] += v[f]
    res["rq2"]["pooled_per_attacker_tool"] = pooled
    res["meta"] = {"git_head": C.GIT_HEAD, "ts": C.now_sgt(), "primary_eps": MS.PRIMARY_EPS}
    json.dump(res, open(f"{OUT}/rq12.json", "w"), indent=1, default=dict)
    for m, x in res["rq1"].items():
        print(m[:14], " ".join(f"{c['level'][:2]}{c['r'] if c['r'] < 1000 else 'A'}:cal{c['cal_mean_loss']:.2f}/ho{c['holdout_fb']:.2f}/rec{(c['holdout_recall'] or 0):.2f}" for c in x["chain"]))
    for m in SECURITY:
        x = res["rq2"][m]; print(m[:14], "viol", x["n_viol"], "intercepted", x["intercepted"], "invisible", x["invisible"])
    for k, v in sorted(pooled.items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {k:32s} n={v['n']:3d} intercepted={v['intercepted']:3d} invisible={v['invisible']:3d}")


if __name__ == "__main__":
    main()
