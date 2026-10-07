"""Seen-vs-novel transfer of mined least privilege (plan.md §16, pre-registered e56e60b5 + baa0a093 before computing).

Replay only, frozen M3 data. Two mining methods judged under two corpora that differ only in whether the holdout
task's own benign runs are mined:
  M1 AMLP (m3_envelope logic; a = exact level r = all, b = calibrated element any / all)
  M2 Praetor-style pDFA re-implemented from arXiv 2604.26274 §4.1 / Alg. 1 (w = 3, theta = 3, eps = 0.05)
NOVEL: suite pool only. SEEN: + the task's own other benign reps (leave-one-run-out for benign runs; all reps for
violation runs). Writes data/analysis_out/seen_novel.json; macros via analysis/make_numbers.py."""
from __future__ import annotations

import collections
import json
import math
import os
import sys
import time
import urllib.request

import numpy as np

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(D)   # release root (code/ and data/ are siblings)
sys.path.insert(0, D)
sys.path.insert(0, f"{D}/analysis")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import m3_common as C          # noqa: E402
import m3_envelope as ME       # noqa: E402
import envbuild as EB          # noqa: E402
from amlp import SIDE, CTRL, control_values, lift  # noqa: E402
from rq12 import sel_entry, holdout_benign, holdout_violations, SECURITY  # noqa: E402
import m3_select as MS         # noqa: E402

OUT = f"{ROOT}/data/analysis_out"
EMB_CACHE = f"{OUT}/seen_novel_emb.json"   # released as seen_novel_emb.json.gz (read transparently)
W, THETA, EPS_NUM, EPS_STR = 3, 3, 0.05, 0.05


# ---------------------------------------------------------------- M1 AMLP with an explicit mined corpus
def amlp_env(mined: dict, suite: str, ut: str, predictor: str, level: str) -> ME.LevelEnvelope:
    """Same construction as m3_envelope.build at r = all, but over an explicit corpus {task: runs} that may contain the
    evaluated task itself (SEEN). Holdout tasks other than ut never appear (callers build `mined` from the pool)."""
    t = (suite, ut)
    tools = set(EB.PRED[predictor][f"{suite}|{ut}"])
    tools |= {c["f"] for rs in mined.values() for o in rs for c in o["trace"] if c["f"] in SIDE[suite]}
    vals, lifted = collections.defaultdict(set), collections.defaultdict(set)
    for rs in mined.values():
        for o in rs:
            for c in o["trace"]:
                if c["f"] in SIDE[suite]:
                    for a, v in control_values(c["f"], c["a"]):
                        vals[(c["f"], a)].add(v)
                        lifted[(c["f"], a)].add(lift(v))
    la = {k for k in lifted if any("@" in v for v in vals[k])} if level in ("email", "class", "any") else set()
    e = ME.LevelEnvelope(suite, tools, dict(vals), dict(lifted), la, EB.ENV_TYPED[suite], EB.PROMPTS[t])
    e.level, e.check_values = level, level != "any"
    e.class_vals = {k: {ME.lift_class(v) for v in vs} for k, vs in vals.items()}
    return e


# ---------------------------------------------------------------- M2 Praetor pDFA
class Embedder:
    def __init__(self):
        if os.path.exists(EMB_CACHE):
            self.cache = json.load(open(EMB_CACHE))
        elif os.path.exists(EMB_CACHE + ".gz"):
            import gzip
            self.cache = json.load(gzip.open(EMB_CACHE + ".gz", "rt"))
        else:
            self.cache = {}
        # network access is only needed for texts missing from the cache (env LLM_API_BASE / LLM_API_KEY)
        u = os.environ.get("LLM_API_BASE", "").rstrip("/")
        self.url = u + ("/embeddings" if u.endswith("/v1") else "/v1/embeddings")
        self.key = os.environ.get("LLM_API_KEY", "")

    def fill(self, texts):
        todo = sorted({t for t in texts if t not in self.cache})
        for i in range(0, len(todo), 64):
            batch = todo[i:i + 64]
            for att in range(5):
                try:
                    req = urllib.request.Request(self.url, data=json.dumps({"model": "bge-m3", "input": batch}).encode(),
                                                 headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
                    d = json.load(urllib.request.urlopen(req, timeout=120))
                    for t, x in zip(batch, d["data"]):
                        v = np.asarray(x["embedding"], dtype=float)
                        self.cache[t] = (v / (np.linalg.norm(v) or 1.0)).round(6).tolist()
                    json.dump(self.cache, open(EMB_CACHE, "w"))
                    break
                except Exception as ex:  # noqa: BLE001
                    sys.stderr.write(f"embed retry {att}: {ex}\n"); time.sleep(10)
            else:
                raise SystemExit("embedding failed")
        json.dump(self.cache, open(EMB_CACHE, "w"))

    def vec(self, t):
        return np.asarray(self.cache[t])


def kind(name, v):
    if name in CTRL or isinstance(v, bool) or v is None or isinstance(v, dict):
        return "cat"
    if isinstance(v, (int, float)):
        return "num"
    if isinstance(v, str):
        return "str" if v.strip() else "cat"         # plan.md §16.3(v): empty strings exact-match
    return "cat"


def elems(v):
    return list(v) if isinstance(v, (list, tuple)) else [v]


def key(v):
    return json.dumps(v, sort_keys=True, default=str)


class Schema:
    """Per-edge parameter schema (Praetor §4.1 Step 3) under the pre-registered type rules (plan.md §16.3)."""

    def __init__(self, obs: list[dict], emb: Embedder):
        self.names = set().union(*[set(p) for p in obs]) if obs else set()
        self.s = {}
        for n in self.names:
            vs = [x for p in obs if n in p for x in elems(p[n])]
            ks = {kind(n, x) for x in vs}
            k = "num" if ks == {"num"} else ("str" if ks == {"str"} else "cat")
            if k == "num":
                lo, hi = min(vs), max(vs)
                self.s[n] = ("num", lo - EPS_NUM * abs(lo), hi + EPS_NUM * abs(hi))
            elif k == "str":
                E = np.stack([emb.vec(x) for x in vs])
                c = E.mean(0); c = c / (np.linalg.norm(c) or 1.0)
                r = float(max(1 - E @ c)) + EPS_STR
                self.s[n] = ("str", c, r)
            else:
                self.s[n] = ("cat", {key(x) for x in vs})

    def ok(self, p: dict, emb: Embedder) -> bool:
        if set(p) - self.names:                       # parameter never observed on this edge
            return False
        for n, v in p.items():
            sc = self.s[n]
            for x in elems(v):
                if sc[0] == "num":
                    if not (isinstance(x, (int, float)) and not isinstance(x, bool) and sc[1] <= x <= sc[2]):
                        return False
                elif sc[0] == "str":
                    if kind(n, x) != "str" or float(1 - emb.vec(x) @ sc[1]) > sc[2]:   # §16.3(v)
                        return False
                elif key(x) not in sc[1]:
                    return False
        return True


S0 = ("<bot>", ())
END = ("<end>", ())


def states(trace):
    names = [c["f"] for c in trace]
    return [(names[j], tuple(names[max(0, j - W):j])) for j in range(len(names))]


class PDFA:
    def __init__(self, traces: list[list[dict]], emb: Embedder):
        edges = collections.defaultdict(list)                  # (s, s') -> [params]
        for tr in traces:
            prev = S0
            for s, c in zip(states(tr), tr):
                edges[(prev, s)].append(dict(c["a"]))
                prev = s
            edges[(prev, END)].append({})                      # §16.3(iv): termination is an outgoing instance
        alive = set(edges)
        while True:                                            # prune to a fixpoint (Praetor §4.1 Step 4)
            out = collections.Counter()
            for (a, b) in alive:
                out[a] += len(edges[(a, b)])
            dead = {a for (a, b) in alive if a != S0 and out[a] < THETA}
            dead |= {b for (a, b) in alive if b not in (END,) and b != S0 and out[b] < THETA}
            new = {(a, b) for (a, b) in alive if a not in dead and b not in dead}
            if new == alive:
                break
            alive = new
        reach, frontier = {S0}, [S0]                           # forward reachability from s0
        succ = collections.defaultdict(list)
        for (a, b) in alive:
            succ[a].append(b)
        while frontier:
            a = frontier.pop()
            for b in succ[a]:
                if b not in reach:
                    reach.add(b); frontier.append(b)
        self.edges = {e: Schema(edges[e], emb) for e in alive if e[0] in reach and e[1] != END}

    def first_block(self, trace, emb):
        prev = S0
        for j, (s, c) in enumerate(zip(states(trace), trace)):
            sc = self.edges.get((prev, s))
            if sc is None or not sc.ok(dict(c["a"]), emb):
                return j
            prev = s
        return None


def all_strings(runs):
    out = set()
    for r in runs:
        for c in r["trace"]:
            for n, v in c["a"].items():
                for x in elems(v):
                    if kind(n, x) == "str":
                        out.add(x)
    return out


# ---------------------------------------------------------------- driver
def main():
    os.makedirs(OUT, exist_ok=True)
    emb = Embedder()
    res = {"per_model": {}, "rows": collections.defaultdict(list)}
    blk = collections.defaultdict(lambda: collections.defaultdict(list))   # metric -> task -> per-run diffs
    tot = collections.defaultdict(lambda: [0, 0])                             # (method, setting, kind) -> [k, n]
    for m in C.MODELS:
        e = sel_entry(m, MS.PRIMARY_EPS)
        pred = e["chosen"]["predictor"]
        pool = ME.Pool(m, False, k=e["k"])
        C.assert_pool_clean(pool.tasks(), "analysis/seen_novel")
        ben, viol = holdout_benign(m), (holdout_violations(m) if m in SECURITY else [])
        own = collections.defaultdict(list)
        for b in ben:
            own[(b["suite"], b["ut"])].append(b)
        emb.fill(all_strings([r for rs in pool.ben.values() for r in rs] + ben + viol))
        by_suite = {s: {k: v for k, v in pool.ben.items() if k[0] == s} for s in SIDE}
        novel_pdfa = {s: PDFA([r["trace"] for rs in by_suite[s].values() for r in rs], emb) for s in SIDE}
        pm = collections.defaultdict(lambda: [0, 0])
        cache_seen_task = {}

        def judge(kind_, run, seen_runs):
            t = (run["suite"], run["ut"])
            mined_novel = by_suite[t[0]]
            mined_seen = dict(mined_novel); mined_seen[t] = seen_runs
            out = {}
            for lv, tag in (("exact", "M1a"), ("any", "M1b")):
                out[(tag, "novel")] = ME.first_block(amlp_env(mined_novel, *t, pred, lv), run["trace"]) is not None
                out[(tag, "seen")] = ME.first_block(amlp_env(mined_seen, *t, pred, lv), run["trace"]) is not None
            out[("M2", "novel")] = novel_pdfa[t[0]].first_block(run["trace"], emb) is not None
            key_ = (t, tuple(sorted(r["rep"] for r in seen_runs)))
            if key_ not in cache_seen_task:
                cache_seen_task[key_] = PDFA([r["trace"] for rs in mined_seen.values() for r in rs], emb)
            out[("M2", "seen")] = cache_seen_task[key_].first_block(run["trace"], emb) is not None
            for (meth, st), b in out.items():
                tot[(meth, st, kind_)][0] += b; tot[(meth, st, kind_)][1] += 1
                pm[(meth, st, kind_)][0] += b; pm[(meth, st, kind_)][1] += 1
            for meth in ("M1a", "M1b", "M2"):
                d = (out[(meth, "novel")] - out[(meth, "seen")]) if kind_ == "fb" else (out[(meth, "seen")] - out[(meth, "novel")])
                blk[f"{meth}_{kind_}"][t].append(int(d))
        for b in ben:                                           # leave-one-run-out
            judge("fb", b, [r for r in own[(b["suite"], b["ut"])] if r["rep"] != b["rep"]])
        for v in viol:                                          # all benign reps of the task
            judge("flag", v, own[(v["suite"], v["ut"])])
        res["per_model"][m] = {f"{a}_{b}_{c}": x[0] / x[1] for (a, b, c), x in pm.items() if x[1]}
        # §16.5 sanity: M2 leave-one-run-out FB inside the mining pool (Praetor-style seen-workflow BTFR analogue)
        k = n = 0
        for s, tasks in by_suite.items():
            for t, rs in tasks.items():
                for r in rs:
                    corpus = [x["trace"] for tt, xs in tasks.items() for x in xs if not (tt == t and x["rep"] == r["rep"])]
                    k += PDFA(corpus, emb).first_block(r["trace"], emb) is not None; n += 1
        res["per_model"][m]["M2_pool_loo_fb"] = k / n
        print(m, json.dumps(res["per_model"][m]), flush=True)
    res["pooled"] = {f"{a}_{b}_{c}": x[0] / x[1] for (a, b, c), x in tot.items()}
    res["n"] = {f"{a}_{b}_{c}": x[1] for (a, b, c), x in tot.items()}
    res["delta"] = {}
    for k_, vals in blk.items():
        p, lo, hi = C.cluster_bootstrap(vals)
        res["delta"][k_] = {"point": p, "ci95": [lo, hi]}
    res["H_gap"] = {meth: res["delta"][f"{meth}_fb"]["point"] >= 0.05 and res["delta"][f"{meth}_fb"]["ci95"][0] > 0
                    for meth in ("M1a", "M2")}
    res["general_claim_allowed"] = all(res["H_gap"].values())
    res.pop("rows")
    json.dump(res, open(f"{OUT}/seen_novel.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("pooled", "delta", "H_gap", "general_claim_allowed")}, indent=1))


if __name__ == "__main__":
    main()
