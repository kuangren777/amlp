"""Automaton component split and clean-subset replication of RQ1 (plan.md §19, pre-registered a49dc195 before computing).
(a) M2 full / M2-seq (argument guards off) / M2-arg (w = 0, per-tool argument schemas), seen vs novel, all holdout tasks.
(b) every §16 / §18 part restricted to the 27 holdout tasks no pilot run touched (exploratory replication).
Same corpora, settings, FB / flag metrics and task-cluster bootstrap as analysis/seen_novel.py.
Writes data/analysis_out/seen_novel_split.json."""
from __future__ import annotations

import collections
import json
import os
import sys

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(D)   # release root (code/ and data/ are siblings)
sys.path.insert(0, f"{D}/analysis")
import seen_novel as S                     # noqa: E402
import seen_novel_pure as SP               # noqa: E402
from seen_novel import C, ME, MS, SIDE, sel_entry, holdout_benign, holdout_violations, SECURITY  # noqa: E402
from extra_checks import pilot_test_tasks  # noqa: E402

CLEAN = C.HOLDOUT - pilot_test_tasks()


class SeqPDFA(S.PDFA):
    """M2-seq: structural transitions only, argument guards always pass."""
    def first_block(self, trace, emb):
        prev = S.S0
        for j, s in enumerate(S.states(trace)):
            if (prev, s) not in self.edges:
                return j
            prev = s
        return None


def build_pdfa(kind, traces, emb):
    if kind == "M2-arg":
        w0 = S.W
        S.W = 0
        try:
            return S.PDFA(traces, emb)
        finally:
            S.W = w0
    return (SeqPDFA if kind == "M2-seq" else S.PDFA)(traces, emb)


def blocks_pdfa(kind, p, trace, emb):
    if kind == "M2-arg":
        w0 = S.W
        S.W = 0
        try:
            return p.first_block(trace, emb) is not None
        finally:
            S.W = w0
    return p.first_block(trace, emb) is not None


AUTOS = ("M2", "M2-seq", "M2-arg")


def main():
    emb = S.Embedder()
    rec = []                                         # (part, kind, task, novel_block, seen_block)
    for m in C.MODELS:
        e = sel_entry(m, MS.PRIMARY_EPS)
        pred = e["chosen"]["predictor"]
        pool = ME.Pool(m, False, k=e["k"])
        C.assert_pool_clean(pool.tasks(), "analysis/seen_novel_split")
        ben, viol = holdout_benign(m), (holdout_violations(m) if m in SECURITY else [])
        own = collections.defaultdict(list)
        for b in ben:
            own[(b["suite"], b["ut"])].append(b)
        emb.fill(S.all_strings([r for rs in pool.ben.values() for r in rs] + ben + viol))
        by_suite = {s: {k: v for k, v in pool.ben.items() if k[0] == s} for s in SIDE}
        novel_p = {(a, s): build_pdfa(a, [r["trace"] for rs in by_suite[s].values() for r in rs], emb) for a in AUTOS for s in SIDE}
        cache = {}

        def judge(kind, run, seen_runs):
            t = (run["suite"], run["ut"])
            novel = by_suite[t[0]]
            seen = dict(novel); seen[t] = seen_runs
            key = (t, tuple(sorted(r["rep"] for r in seen_runs)))
            for a in AUTOS:
                if (a, key) not in cache:
                    cache[(a, key)] = build_pdfa(a, [r["trace"] for rs in seen.values() for r in rs], emb)
                rec.append((a, kind, t, blocks_pdfa(a, novel_p[(a, t[0])], run["trace"], emb),
                            blocks_pdfa(a, cache[(a, key)], run["trace"], emb)))
            for lv, tag in (("exact", "M1a"), ("any", "M1b")):
                rec.append((tag, kind, t, ME.first_block(S.amlp_env(novel, *t, pred, lv), run["trace"]) is not None,
                            ME.first_block(S.amlp_env(seen, *t, pred, lv), run["trace"]) is not None))
            for v in SP.VARS:
                rec.append((v, kind, t, ME.first_block(SP.variant(v, novel, *t, pred), run["trace"]) is not None,
                            ME.first_block(SP.variant(v, seen, *t, pred), run["trace"]) is not None))
        for b in ben:
            judge("fb", b, [r for r in own[(b["suite"], b["ut"])] if r["rep"] != b["rep"]])
        for v in viol:
            judge("flag", v, own[(v["suite"], v["ut"])])
        print(m, "done", flush=True)
    res = {}
    for scope, keep in (("all", lambda t: True), ("clean", lambda t: t in CLEAN)):
        out = {}
        for part in sorted({r[0] for r in rec}):
            for kind in ("fb", "flag"):
                rows = [r for r in rec if r[0] == part and r[1] == kind and keep(r[2])]
                if not rows:
                    continue
                d = collections.defaultdict(list)
                for _, _, t, bn, bs in rows:
                    d[t].append(int(bn) - int(bs) if kind == "fb" else int(bs) - int(bn))
                p, lo, hi = C.cluster_bootstrap(d)
                out[f"{part}_{kind}"] = {"novel": sum(r[3] for r in rows) / len(rows), "seen": sum(r[4] for r in rows) / len(rows),
                                         "n": len(rows), "delta": p, "ci95": [lo, hi],
                                         "gap_rule": kind == "fb" and p >= 0.05 and lo > 0}
        res[scope] = out
    res["n_clean_tasks"] = len(CLEAN)
    json.dump(res, open(f"{ROOT}/data/analysis_out/seen_novel_split.json", "w"), indent=1)
    for scope in ("all", "clean"):
        print(scope)
        for k, v in res[scope].items():
            print(f"  {k:16s} novel {100*v['novel']:5.1f} seen {100*v['seen']:5.1f} d {100*v['delta']:5.1f} [{100*v['ci95'][0]:.1f},{100*v['ci95'][1]:.1f}] rule={v['gap_rule']}")


if __name__ == "__main__":
    main()
