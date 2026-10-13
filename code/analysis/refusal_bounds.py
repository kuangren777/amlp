"""Benign utility under refusal: lower and upper bounds for every refusing defense (plan.md §22.9, pre-registered
747bead79 before computing). Paired benign runs of rq34.paired_benign (same tasks, reps, denominators as Table 5).
M = measured utility (native semantics). L = a run in which the defense refused or intervened at least once counts as
failed. U = such a run takes the paired undefended run's utility (estimate); CaMeL's U is measured by runs without
security policies (analysis/run_camel_nosecpol.py). Intervention markers are each defense's own refusal text, since
ordinary tool errors also occur in arms that never refuse.
Usage: HOLDOUT_SET=all48|clean27 python3 analysis/refusal_bounds.py -> data/analysis_out/<set>/refusal_bounds.json"""
import collections
import json
import os
import sys

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(D)   # release root (code/ and data/ are siblings)
sys.path.insert(0, D)
sys.path.insert(0, f"{D}/analysis")
import rq34 as Q               # noqa: E402
import m3_common as C          # noqa: E402
import m3_judge as MJ          # noqa: E402

MARK = {"progent": ("is not allowed", "Failed validating"), "tripwire": "Blocked by TripWire", "agentsentry": "Not permitted for this task",
        "pi_detector": "Data omitted", "melon": "stopping execution"}
# Progent refuses a tool by name ("is not allowed") and an argument through its JSON-schema policy ("Failed validating");
# pydantic "Input schema" errors are the tool's own argument checks and occur in every arm.
ARMS = ["amlp", "block_all", "progent", "tripwire", "agentsentry", "pi_detector", "melon", "camel"]


def steps_all(m):
    out = {}
    for g in ("main", "camel", "agentsentry"):
        out.update(MJ.steps_index(C.steps_path(False, "online", m, g)))
    return out


def intervened(arm, r, st):
    if arm in ("amlp", "block_all"):
        return any(f.get("blocked") for f in (r.get("amlp_flags") or []))
    if arm == "camel":
        return bool(r.get("camel_denied"))
    text = " ".join(f"{s.get('output') or ''} {s.get('error') or ''}" for s in st.get(r["row_id"], []))
    text += " " + str(r.get("final") or "")
    marks = MARK[arm] if isinstance(MARK[arm], tuple) else (MARK[arm],)
    return any(k in text for k in marks)


def main():
    res = {"plan": "§22.9 747bead79", "set": os.environ.get("HOLDOUT_SET", "clean27"), "models": {}}
    for m in C.MODELS:
        on, mon, st = Q.online(m), Q.monitor(m), steps_all(m)
        nosec = {}
        p = f"{ROOT}/data/m3/online_{m}__camel_nosecpol.jsonl"
        for r in C.read_jsonl(p):
            if r.get("err") is None and (r["suite"], r["ut"]) in Q.HS.ACTIVE:
                nosec[(r["suite"], r["ut"], r["rep"])] = int(bool(r["utility"]))
        res["models"][m] = {}
        for arm in ARMS:
            kb, base, it = Q.paired_benign(m, arm, on, mon)
            if not kb:
                continue
            part = "a" if arm == "amlp" else "c"
            rows = {(r["suite"], r["ut"], r["rep"]): r for r in on if Q.arm_of(r) == arm and r["part"] == part}
            L, U, dM, dL, dU = {}, {}, collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
            n_int = 0
            for k in kb:
                iv = intervened(arm, rows[k], st)
                n_int += iv
                L[k] = 0 if iv else it[k]
                if arm == "camel":
                    if k not in nosec:
                        continue
                    U[k] = nosec[k]
                else:
                    U[k] = base[k] if iv else it[k]
            for k in kb:
                dM[k[:2]].append(base[k] - it[k]); dL[k[:2]].append(base[k] - L[k])
                if k in U:
                    dU[k[:2]].append(base[k] - U[k])
            e = {"n": len(kb), "n_intervened": n_int,
                 "util_M": sum(it[k] for k in kb) / len(kb), "util_L": sum(L.values()) / len(kb),
                 "util_U": sum(U.values()) / len(U) if U else None, "n_U": len(U)}
            if arm == "camel":                                  # §22.12: paired policy increment, cost_M - cost_U
                dP = collections.defaultdict(list)
                for k in kb:
                    if k in U:
                        dP[k[:2]].append(U[k] - it[k])
                pnt, lo, hi = C.cluster_bootstrap(dP)
                e["policy_add"], e["policy_add_ci"] = pnt, [lo, hi]
            for tag, d in (("M", dM), ("L", dL), ("U", dU)):
                if d:
                    pnt, lo, hi = C.cluster_bootstrap(d)
                    e[f"cost_{tag}"], e[f"cost_{tag}_ci"] = pnt, [lo, hi]
            res["models"][m][arm] = e
    json.dump(res, open(f"{Q.OUT}/refusal_bounds.json", "w"), indent=1)
    for m, x in res["models"].items():
        print(m[:14], {a: (v["n_intervened"], round(100 * v["cost_L"], 1), round(100 * v["cost_M"], 1),
                          None if v.get("cost_U") is None else round(100 * v["cost_U"], 1)) for a, v in x.items()})


if __name__ == "__main__":
    main()
