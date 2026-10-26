"""Heuristic sensitivity of the automaton transfer gap (plan.md §24, pre-registered a5d21eaba before computing).

Same corpora, FB / flag metrics and task-cluster bootstrap as analysis/seen_novel.py (M2 only). The automaton of
seen_novel.PDFA / Schema is copied here with three switches for the heuristics our re-implementation adds:
  E encoder      bge-m3 (ours, cached hub embeddings) | minilm (all-MiniLM-L6-v2, local CPU, mean pooling, L2 norm)
  T control args exact (ours, categorical) | string (non-empty string control values use the embedding schema)
  X trace end    end (ours, END is one outgoing observation) | exempt (END not counted, trace-final states never pruned)
                 | literal (END not counted, no exemption; descriptive row of §24.1)
plus Praetor's own parameters w, theta, eps_str for the secondary one-at-a-time rows.
The reference cell (bge-m3, exact, end, w 3, theta 3, eps 0.05) must equal seen_novel.json of the same set.
Writes analysis/out/<set>/seen_novel_sens.json. Usage: HOLDOUT_SET=all48|clean27 python3 analysis/seen_novel_sens.py
"""
from __future__ import annotations

import collections
import glob
import json
import os
import sys

import numpy as np

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(D)   # release root (code/ and data/ are siblings)
sys.path.insert(0, f"{D}/analysis")
import seen_novel as S                     # noqa: E402
from seen_novel import C, ME, MS, SIDE, CTRL, sel_entry, holdout_benign, holdout_violations, SECURITY  # noqa: E402

OUT = S.OUT
MINILM_CACHE = f"{ROOT}/data/analysis_out/seen_novel_emb_minilm.json"
REF = dict(E="bge", T="exact", X="end", w=3, theta=3, eps=0.05)
PRIMARY = [dict(REF, E=e, T=t, X=x) for e in ("bge", "minilm") for t in ("exact", "string") for x in ("end", "exempt")]
SECONDARY = [dict(REF, theta=2), dict(REF, theta=5), dict(REF, w=2), dict(REF, eps=0.10), dict(REF, X="literal")]   # §24.1


def cell_name(c):
    return f"E={c['E']},T={c['T']},X={c['X']},w={c['w']},theta={c['theta']},eps={c['eps']}"


class MiniLM:
    """all-MiniLM-L6-v2 from the local HF cache, sentence-transformers pooling (mean over tokens, L2 norm, 256 tokens)."""

    def __init__(self):
        self.cache = json.load(open(MINILM_CACHE)) if os.path.exists(MINILM_CACHE) else {}
        self.model = None

    def _load(self):
        import torch
        from transformers import AutoModel, AutoTokenizer
        snap = glob.glob(os.path.expanduser(
            "~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/*/"))[0]
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(snap)
        self.model = AutoModel.from_pretrained(snap).eval()

    def fill(self, texts):
        todo = sorted({t for t in texts if t not in self.cache})
        if not todo:
            return
        if self.model is None:
            self._load()
        for i in range(0, len(todo), 128):
            batch = todo[i:i + 128]
            x = self.tok(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
            with self.torch.no_grad():
                h = self.model(**x).last_hidden_state
            m = x["attention_mask"].unsqueeze(-1).float()
            v = ((h * m).sum(1) / m.sum(1).clamp(min=1e-9)).numpy()
            for t, e in zip(batch, v):
                self.cache[t] = (e / (np.linalg.norm(e) or 1.0)).round(6).tolist()
        json.dump(self.cache, open(MINILM_CACHE, "w"))

    def vec(self, t):
        return np.asarray(self.cache[t])


def kind(cfg, name, v):
    if cfg["T"] == "string" and name in CTRL and isinstance(v, str) and v.strip():
        return "str"
    return S.kind(name, v)


class Schema:
    """seen_novel.Schema with the typing switch T and slack eps."""

    def __init__(self, obs, emb, cfg):
        self.cfg = cfg
        self.names = set().union(*[set(p) for p in obs]) if obs else set()
        self.s = {}
        for n in self.names:
            vs = [x for p in obs if n in p for x in S.elems(p[n])]
            ks = {kind(cfg, n, x) for x in vs}
            k = "num" if ks == {"num"} else ("str" if ks == {"str"} else "cat")
            if k == "num":
                lo, hi = min(vs), max(vs)
                self.s[n] = ("num", lo - S.EPS_NUM * abs(lo), hi + S.EPS_NUM * abs(hi))
            elif k == "str":
                E = np.stack([emb.vec(x) for x in vs])
                c = E.mean(0); c = c / (np.linalg.norm(c) or 1.0)
                self.s[n] = ("str", c, float(max(1 - E @ c)) + cfg["eps"])
            else:
                self.s[n] = ("cat", {S.key(x) for x in vs})

    def ok(self, p, emb):
        if set(p) - self.names:
            return False
        for n, v in p.items():
            sc = self.s[n]
            for x in S.elems(v):
                if sc[0] == "num":
                    if not (isinstance(x, (int, float)) and not isinstance(x, bool) and sc[1] <= x <= sc[2]):
                        return False
                elif sc[0] == "str":
                    if kind(self.cfg, n, x) != "str" or float(1 - emb.vec(x) @ sc[1]) > sc[2]:
                        return False
                elif S.key(x) not in sc[1]:
                    return False
        return True


def states(trace, w):
    names = [c["f"] for c in trace]
    return [(names[j], tuple(names[max(0, j - w):j])) for j in range(len(names))]


class PDFA:
    """seen_novel.PDFA with the trace-end switch X and parameters w, theta."""

    def __init__(self, traces, emb, cfg):
        self.cfg = cfg
        w, theta, exempt = cfg["w"], cfg["theta"], cfg["X"] == "exempt"
        edges = collections.defaultdict(list)
        final = set()
        for tr in traces:
            prev = S.S0
            for s, c in zip(states(tr, w), tr):
                edges[(prev, s)].append(dict(c["a"]))
                prev = s
            if exempt:
                final.add(prev)
            elif cfg["X"] == "end":
                edges[(prev, S.END)].append({})
        alive = set(edges)
        while True:
            out = collections.Counter()
            for (a, b) in alive:
                out[a] += len(edges[(a, b)])
            dead = {a for (a, b) in alive if a != S.S0 and out[a] < theta and a not in final}
            dead |= {b for (a, b) in alive if b not in (S.END,) and b != S.S0 and out[b] < theta and b not in final}
            new = {(a, b) for (a, b) in alive if a not in dead and b not in dead}
            if new == alive:
                break
            alive = new
        reach, frontier = {S.S0}, [S.S0]
        succ = collections.defaultdict(list)
        for (a, b) in alive:
            succ[a].append(b)
        while frontier:
            a = frontier.pop()
            for b in succ[a]:
                if b not in reach:
                    reach.add(b); frontier.append(b)
        self.edges = {e: Schema(edges[e], emb, cfg) for e in alive if e[0] in reach and e[1] != S.END}

    def first_block(self, trace, emb):
        prev = S.S0
        for j, (s, c) in enumerate(zip(states(trace, self.cfg["w"]), trace)):
            sc = self.edges.get((prev, s))
            if sc is None or not sc.ok(dict(c["a"]), emb):
                return j
            prev = s
        return None


def all_strings(runs, cfg):
    out = set()
    for r in runs:
        for c in r["trace"]:
            for n, v in c["a"].items():
                for x in S.elems(v):
                    if kind(cfg, n, x) == "str":
                        out.add(x)
    return out


def run_cell(cfg, data, embs):
    emb = embs[cfg["E"]]
    blk = {"fb": collections.defaultdict(list), "flag": collections.defaultdict(list)}
    tot = collections.defaultdict(lambda: [0, 0])
    for m, (pool_ben, ben, viol, own) in data.items():
        by_suite = {s: {k: v for k, v in pool_ben.items() if k[0] == s} for s in SIDE}
        novel = {s: PDFA([r["trace"] for rs in by_suite[s].values() for r in rs], emb, cfg) for s in SIDE}
        cache = {}

        def judge(kind_, run, seen_runs):
            t = (run["suite"], run["ut"])
            k = (t, tuple(sorted(r["rep"] for r in seen_runs)))
            if k not in cache:
                mined = dict(by_suite[t[0]]); mined[t] = seen_runs
                cache[k] = PDFA([r["trace"] for rs in mined.values() for r in rs], emb, cfg)
            bn = novel[t[0]].first_block(run["trace"], emb) is not None
            bs = cache[k].first_block(run["trace"], emb) is not None
            tot[("novel", kind_)][0] += bn; tot[("novel", kind_)][1] += 1
            tot[("seen", kind_)][0] += bs; tot[("seen", kind_)][1] += 1
            blk[kind_][t].append(int(bn - bs) if kind_ == "fb" else int(bs - bn))
        for b in ben:
            judge("fb", b, [r for r in own[(b["suite"], b["ut"])] if r["rep"] != b["rep"]])
        for v in viol:
            judge("flag", v, own[(v["suite"], v["ut"])])
    res = {f"{st}_{k}": x[0] / x[1] for (st, k), x in tot.items() if x[1]}
    res["n"] = {f"{st}_{k}": x[1] for (st, k), x in tot.items()}
    for k in ("fb", "flag"):
        p, lo, hi = C.cluster_bootstrap(blk[k])
        res[f"delta_{k}"] = {"point": p, "ci95": [lo, hi]}
    return res


def load_data():
    data = {}
    for m in C.MODELS:
        e = sel_entry(m, MS.PRIMARY_EPS)
        pool = ME.Pool(m, False, k=e["k"])
        C.assert_pool_clean(pool.tasks(), "analysis/seen_novel_sens")
        ben, viol = holdout_benign(m), (holdout_violations(m) if m in SECURITY else [])
        own = collections.defaultdict(list)
        for b in ben:
            own[(b["suite"], b["ut"])].append(b)
        data[m] = (pool.ben, ben, viol, own)
    return data


ARG_GAP = {"all48": 0.275, "clean27": 0.219}     # mined argument sets, plan §20 table (§18 / §19 values)


def verdict(cells, set_):
    ok = {name: r["delta_fb"]["ci95"][0] >= 0.20 and r["delta_fb"]["point"] > ARG_GAP[set_] for name, r in cells.items()}
    return {"per_cell": ok, "holds": all(ok.values())}


def main():
    set_ = os.environ.get("HOLDOUT_SET", "clean27")
    data = load_data()
    embs = {"bge": S.Embedder(), "minilm": MiniLM()}
    runs = [r for m in data.values() for rs in m[0].values() for r in rs] + \
        [r for m in data.values() for r in m[1] + m[2]]
    for cfg in PRIMARY + SECONDARY:
        embs[cfg["E"]].fill(all_strings(runs, cfg))
    res = {"set": set_, "plan": "plan.md §24 (a5d21eaba)", "primary": {}, "secondary": {}}
    for grp, cfgs in (("primary", PRIMARY), ("secondary", SECONDARY)):
        for cfg in cfgs:
            r = run_cell(cfg, data, embs)
            res[grp][cell_name(cfg)] = r
            print(set_, grp, cell_name(cfg), {k: (round(v["point"] * 100, 1), [round(x * 100, 1) for x in v["ci95"]])
                                              for k, v in r.items() if k.startswith("delta")},
                  f"FB seen {100 * r['seen_fb']:.1f} novel {100 * r['novel_fb']:.1f}", flush=True)
    ref = json.load(open(f"{OUT}/seen_novel.json"))
    mine = res["primary"][cell_name(REF)]
    res["reference_reproduces"] = (mine["delta_fb"] == ref["delta"]["M2_fb"]
                                   and mine["seen_fb"] == ref["pooled"]["M2_seen_fb"]
                                   and mine["novel_fb"] == ref["pooled"]["M2_novel_fb"])
    res["verdict"] = verdict(res["primary"], set_)
    json.dump(res, open(f"{OUT}/seen_novel_sens.json", "w"), indent=1)
    print("reference reproduces", res["reference_reproduces"], "verdict", res["verdict"]["holds"])


if __name__ == "__main__":
    main()
