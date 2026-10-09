"""RQ3 / RQ4 analysis, implementing plan.md §14 exactly (rules frozen at ffe6dfef before any metric was computed).

--dry-run: validates the row schema and counts pairs per (model, arm / condition) without computing or printing any
metric. Full run writes data/analysis_out/<HOLDOUT_SET>/rq34.json (metrics) for analysis/make_numbers.py and analysis/make_figs.py.
Usage: python3 analysis/rq34.py [--dry-run]"""
from __future__ import annotations

import collections
import json
import os
import random
import sys

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(D)   # release root (code/ and data/ are siblings)
sys.path.insert(0, D)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import m3_common as C          # noqa: E402
import m3_envelope as ME       # noqa: E402
import m3_select as MS         # noqa: E402
from amlp import SIDE          # noqa: E402
import holdout_set as HS       # noqa: E402  (HOLDOUT_SET switch, filters holdout rows)

OUT = HS.OUT
SECURITY = ["qwen3-8b-local", "gpt-4o-mini-2024-07-18", "gpt-4.1-mini-2025-04-14"]
ARMS = ["amlp", "block_all", "spotlighting", "sandwich", "tool_filter", "pi_detector", "melon", "tripwire", "progent",
        "camel", "agentsentry"]
REQ = {"online": ("defense", "part", "rep", "suite", "ut", "it", "utility", "security", "err", "model", "row_id"),
       "monitor": ("rep", "suite", "ut", "utility", "err", "model", "row_id", "trace"),
       "paraphrase": ("defense", "paraphrase_idx", "suite", "ut", "utility", "err", "model", "row_id")}


def last_ok(rows):
    """Last row per row id; CaMeL internal exceptions are kept as failures (plan.md §14)."""
    out = {}
    for r in rows:
        if r.get("err") is None or r.get("err_kind") in ("camel_internal", "context"):
            out[r["row_id"]] = r
    return list(out.values())


def check(rows, kind):
    miss = collections.Counter(k for r in rows for k in REQ[kind] if k not in r)
    assert not miss, f"{kind}: missing fields {dict(miss)}"


def a2rev(r):
    return r.get("it") is None or (r["suite"], r["it"]) not in C.TEXT_ONLY


def online(model):
    rows = []
    for g in ("main", "camel", "agentsentry"):
        p = C.rows_path(False, "online", model, g)
        if os.path.exists(p):
            rows += C.read_jsonl(p)
    rows = last_ok(rows)
    check(rows, "online")
    return [r for r in rows if a2rev(r)]


def monitor(model):
    rows = last_ok(C.read_jsonl(C.rows_path(False, "monitor", model)))
    check(rows, "monitor")
    return rows


def paraphrase(model):
    p = f"{ROOT}/data/m3/paraphrase_{model}.jsonl"
    if not os.path.exists(p):
        return []
    rows = last_ok(C.read_jsonl(p))
    check(rows, "paraphrase")
    return rows


def arm_of(r):
    return r.get("defense") or "none"


def paired_benign(model, arm, on, mon):
    base = {(r["suite"], r["ut"], r["rep"]): int(bool(r["utility"])) for r in mon}
    if arm == "amlp":
        it = {(r["suite"], r["ut"], r["rep"]): int(bool(r["utility"])) for r in on if arm_of(r) == "amlp" and r["part"] == "a"}
    else:
        it = {(r["suite"], r["ut"], r["rep"]): int(bool(r["utility"])) for r in on if arm_of(r) == arm and r["part"] == "c"}
    keys = sorted(set(base) & set(it))
    return keys, base, it


def paired_attack(arm, on):
    none = {(r["suite"], r["ut"], r["it"], r["rep"]): int(bool(r["security"])) for r in on if arm_of(r) == "none" and r["part"] == "b"}
    it = {(r["suite"], r["ut"], r["it"], r["rep"]): int(bool(r["security"])) for r in on if arm_of(r) == arm and r["part"] == "b"}
    keys = sorted(set(none) & set(it))
    return keys, none, it


def ratio_boot(num_by_task, den_by_task, n=C.BOOT_N, seed=C.BOOT_SEED):
    """Task-cluster bootstrap of a ratio statistic f = 1 - sum(num)/sum(den) (block rate). Returns (point, lo, hi)."""
    tasks = sorted(set(num_by_task) | set(den_by_task), key=str)
    sn = {t: num_by_task.get(t, 0) for t in tasks}
    sd = {t: den_by_task.get(t, 0) for t in tasks}
    point = 1 - sum(sn.values()) / sum(sd.values()) if sum(sd.values()) else float("nan")
    rng = random.Random(seed)
    bs = []
    for _ in range(n):
        a = b = 0
        for _ in tasks:
            t = rng.choice(tasks)
            a += sn[t]; b += sd[t]
        if b:
            bs.append(1 - a / b)
    bs.sort()
    return point, bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1]


def rq3(dry):
    res, counts = {}, {}
    for m in C.MODELS:
        on, mon = online(m), monitor(m)
        res[m], counts[m] = {}, {}
        for arm in ARMS:
            kb, base, it = paired_benign(m, arm, on, mon)
            ka, an, aa = paired_attack(arm, on)
            counts[m][arm] = {"benign_pairs": len(kb), "attack_pairs": len(ka)}
            if dry or not kb:
                continue
            diff = collections.defaultdict(list)
            for k in kb:
                diff[k[:2]].append(base[k] - it[k])
            cost, lo, hi = C.cluster_bootstrap(diff)
            entry = {"benign_utility": sum(it[k] for k in kb) / len(kb),
                     "benign_utility_none": sum(base[k] for k in kb) / len(kb),
                     "benign_cost": cost, "benign_cost_ci": [lo, hi], "n_benign": len(kb)}
            if ka and m in SECURITY:
                num, den = collections.Counter(), collections.Counter()
                for k in ka:
                    num[k[:2]] += aa[k]; den[k[:2]] += an[k]
                br, blo, bhi = ratio_boot(num, den)
                entry.update({"asr": sum(aa[k] for k in ka) / len(ka), "asr_none": sum(an[k] for k in ka) / len(ka),
                              "block_rate": br, "block_rate_ci": [blo, bhi], "n_attack": len(ka)})
            res[m][arm] = entry
        if not dry and m in SECURITY:                       # TripWire complementarity (RQ2, plan.md §14)
            ka, an, aa = paired_attack("amlp", on)
            kt, _, tw = paired_attack("tripwire", on)
            both = sorted(set(ka) & set(kt))
            res[m]["amlp_vs_tripwire"] = {f"amlp{x}_tripwire{y}": c for (x, y), c in
                                          collections.Counter((aa[k], tw[k]) for k in both).items()}   # 1 = attack succeeded
    return res, counts


def rq4(dry):
    res, counts = {"paraphrase": {}, "transfer": {}, "poisoning": {}}, {"paraphrase": {}}
    for m in ("qwen3-8b-local", "gpt-4.1-mini-2025-04-14"):
        para = paraphrase(m)
        counts["paraphrase"][m] = dict(collections.Counter(arm_of(r) for r in para))
        if dry or not para:
            continue
        on, mon = online(m), monitor(m)
        out = {}
        for arm in ("none", "amlp", "progent"):
            pu = [int(bool(r["utility"])) for r in para if arm_of(r) == arm]
            if arm == "none":
                ou = [int(bool(r["utility"])) for r in mon if r["rep"] < 3]
            elif arm == "amlp":
                ou = [int(bool(r["utility"])) for r in on if arm_of(r) == "amlp" and r["part"] == "a" and r["rep"] < 3]
            else:
                ou = [int(bool(r["utility"])) for r in on if arm_of(r) == arm and r["part"] == "c"]
            out[arm] = {"utility_para": sum(pu) / len(pu) if pu else None, "utility_orig": sum(ou) / len(ou) if ou else None,
                        "n_para": len(pu), "n_orig": len(ou)}
        for arm in ("amlp", "progent"):
            if out[arm]["utility_para"] is not None and out["none"]["utility_para"] is not None:
                cp = out["none"]["utility_para"] - out[arm]["utility_para"]
                co = out["none"]["utility_orig"] - out[arm]["utility_orig"]
                out[arm]["cost_para"], out[arm]["cost_orig"], out[arm]["shift"] = cp, co, cp - co
        # errata E2: task-cluster bootstrap of the shift, paired paraphrase p <-> original rep p on the same task
        for arm in ("amlp", "progent"):
            P = {(r["suite"], r["ut"], r["paraphrase_idx"]): int(bool(r["utility"])) for r in para if arm_of(r) == arm}
            Pn = {(r["suite"], r["ut"], r["paraphrase_idx"]): int(bool(r["utility"])) for r in para if arm_of(r) == "none"}
            if arm == "amlp":
                O = {(r["suite"], r["ut"], r["rep"]): int(bool(r["utility"])) for r in on if arm_of(r) == "amlp" and r["part"] == "a" and r["rep"] < 3}
            else:
                O = {(r["suite"], r["ut"], r["rep"]): int(bool(r["utility"])) for r in on if arm_of(r) == arm and r["part"] == "c"}
            On = {(r["suite"], r["ut"], r["rep"]): int(bool(r["utility"])) for r in mon if r["rep"] < 3}
            vals = collections.defaultdict(list)
            for k in set(P) & set(Pn) & set(O) & set(On):
                vals[k[:2]].append((Pn[k] - P[k]) - (On[k] - O[k]))
            if vals:
                d, lo, hi = C.cluster_bootstrap(vals)
                out[arm].update({"shift_paired": d, "shift_ci": [lo, hi], "shift_n": sum(len(v) for v in vals.values())})
        cache = json.load(open(f"{ROOT}/data/m3/paraphrase_pred_cache.json"))
        e = [x for x in C.read_jsonl(C.selection_log(False)) if x["model"] == m and x.get("eps") == MS.PRIMARY_EPS and "amendment" in x][-1]
        cfg = e["chosen"]
        pool = ME.Pool(m, False, k=e["k"])
        C.assert_pool_clean(pool.tasks(), "analysis/rq34 paraphrase replay")
        fb = n = 0
        for r in para:
            if arm_of(r) != "none":
                continue
            env = ME.build(pool, r["suite"], r["ut"], exclude_tasks=C.HOLDOUT, predictor=None, r=cfg["r"], level=cfg["level"])
            pred = cache.get(f"{r['suite']}|{r['ut']}|p{r['paraphrase_idx']}||{cfg['predictor']}")
            env.tools |= set((pred or {}).get("tools", []))
            assert pred is not None, (r['suite'], r['ut'], r['paraphrase_idx'])
            fb += ME.first_block(env, r["trace"]) is not None; n += 1
        out["amlp_replay_fb_on_paraphrase"] = fb / n if n else None
        out["amlp_replay_fb_n"] = n
        res["paraphrase"][m] = out
    if dry:
        return res, counts
    for a, b in (("gpt-4o-mini-2024-07-18", "gpt-4.1-mini-2025-04-14"), ("gpt-4.1-mini-2025-04-14", "gpt-4o-mini-2024-07-18"),
                 ("qwen3-8b-local", "llama31-8b-local"), ("llama31-8b-local", "qwen3-8b-local")):
        e = [x for x in C.read_jsonl(C.selection_log(False)) if x["model"] == a and x.get("eps") == MS.PRIMARY_EPS and "amendment" in x][-1]
        pool = ME.Pool(a, False, k=e["k"])
        C.assert_pool_clean(pool.tasks(), "analysis/rq34 transfer")
        cfg = e["chosen"]
        ben = monitor(b)
        viol = [r for r in online(b) if arm_of(r) == "none" and r["part"] == "b" and r["security"]]
        env = {}
        def blk(r):
            t = (r["suite"], r["ut"])
            if t not in env:
                env[t] = ME.build(pool, *t, exclude_tasks=C.HOLDOUT, predictor=cfg["predictor"], r=cfg["r"], level=cfg["level"])
            return ME.first_block(env[t], r["trace"]) is not None
        res["transfer"][f"{a}->{b}"] = {"fb": sum(blk(r) for r in ben) / len(ben),
                                        "recall": (sum(blk(r) for r in viol) / len(viol)) if viol else None,
                                        "n_benign": len(ben), "n_viol": len(viol)}
    res["poisoning"] = poisoning()
    return res, counts


def attacker_calls():
    """(suite) -> list of attacker call lists, one per A2-rev injection task (plan.md §14 (c))."""
    import copy, re
    import harness
    out = collections.defaultdict(list)
    for s in SIDE:
        S = harness.suite(s)
        env = S.load_and_inject_default_environment(S.get_injection_vector_defaults())
        for it, t in sorted(S.injection_tasks.items()):
            if (s, it) in C.TEXT_ONLY:
                continue
            calls = [{"f": c.function, "a": dict(c.args)} for c in t.ground_truth(env) if c.function in SIDE[s]]
            if not calls:                                   # workspace 6..13: no ground truth in AgentDojo v1.2
                addr = re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", t.GOAL)
                calls = [{"f": "send_email", "a": {"recipients": [addr[0]] if addr else ["attacker@example.com"]}}]
            out[s].append(calls)
    return out


def poisoning():
    import copy
    att = attacker_calls()
    res = {}
    for m in C.MODELS:
        e = [x for x in C.read_jsonl(C.selection_log(False)) if x["model"] == m and x.get("eps") == MS.PRIMARY_EPS and "amendment" in x][-1]
        cfg = e["chosen"]
        ben = monitor(m)
        viol = [r for r in online(m) if arm_of(r) == "none" and r["part"] == "b" and r["security"]]
        res[m] = {}
        for rho in (0.0, 0.01, 0.05, 0.10):
          seeds = [20261006] if rho == 0.0 else [20261006 + i for i in range(10)]     # errata E1
          per_seed = []
          for seed in seeds:
            pool = ME.Pool(m, False, k=e["k"])
            C.assert_pool_clean(pool.tasks(), "analysis/rq34 poisoning")
            runs = [(t, i) for t in sorted(pool.ben, key=str) for i in range(len(pool.ben[t]))]
            rng = random.Random(seed)
            chosen = sorted(rng.sample(runs, round(rho * len(runs)))) if rho else []   # errata E1: sorted, no set
            for (t, i) in chosen:
                r = copy.deepcopy(pool.ben[t][i])
                r["trace"] = r["trace"] + rng.choice(att[t[0]])
                pool.ben[t][i] = r
            env = {}
            def blk(r):
                t = (r["suite"], r["ut"])
                if t not in env:
                    env[t] = ME.build(pool, *t, exclude_tasks=C.HOLDOUT, predictor=cfg["predictor"], r=cfg["r"], level=cfg["level"])
                return ME.first_block(env[t], r["trace"]) is not None
            per_seed.append({"seed": seed, "poisoned_runs": len(chosen), "fb": sum(blk(r) for r in ben) / len(ben),
                             "recall": (sum(blk(r) for r in viol) / len(viol)) if viol else None})
          rec = [x["recall"] for x in per_seed if x["recall"] is not None]
          res[m][str(rho)] = {"poisoned_runs": per_seed[0]["poisoned_runs"], "fb": max(x["fb"] for x in per_seed),
                              "recall": per_seed[0]["recall"],
                              "recall_mean": (sum(rec) / len(rec)) if rec else None,
                              "recall_min": min(rec) if rec else None, "recall_max": max(rec) if rec else None,
                              "per_seed": per_seed}
    return res


def main():
    dry = "--dry-run" in sys.argv
    r3, c3 = rq3(dry)
    r4, c4 = rq4(dry)
    if dry:
        print(json.dumps({"rq3_pair_counts": c3, "rq4_counts": c4}, indent=1))
        return
    os.makedirs(OUT, exist_ok=True)
    json.dump({"rq3": r3, "rq4": r4, "meta": {"git_head": C.GIT_HEAD, "ts": C.now_sgt(), "rules": "plan.md §14 (ffe6dfef)"}},
              open(f"{OUT}/rq34.json", "w"), indent=1, default=str)
    print("wrote", f"{OUT}/rq34.json")


if __name__ == "__main__":
    main()
