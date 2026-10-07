"""Progent A/B analysis (criteria pre-registered in plan.md 'Fidelity fixes approved').
1) map upstream fork logdir JSON -> progent_ab_upstream.jsonl rows {suite,ut,it,utility,security,err}
2) upstream 95% CI by pair-level bootstrap (10000 resamples, seed 20261006), percentile method
3) PASS iff port ASR point in upstream ASR CI AND port benign utility point in upstream utility CI (inclusive)
Rows with err are excluded from both sides' rates (counts printed). security=True means the attacker goal was achieved."""
import json, os, glob, random
from math import comb
D = os.path.dirname(os.path.abspath(__file__))
pairs = [tuple(p) for p in json.load(open(f"{D}/progent_ab_pairs.json"))]
up = {}
for f in glob.glob(f"{D}/progent_ab_upstream_logs/gpt-4o-2024-08-06/*/*/*/*.json"):
    d = json.load(open(f))
    it = None if d["injection_task_id"] in (None, "none") else d["injection_task_id"]
    up[(d["suite_name"], d["user_task_id"], it)] = {"suite": d["suite_name"], "ut": d["user_task_id"], "it": it,
        "utility": d["utility"], "security": d["security"] if it else None, "err": d.get("error")}
runerr = {}
if os.path.exists(f"{D}/progent_ab_upstream_run_errs.jsonl"):
    for l in open(f"{D}/progent_ab_upstream_run_errs.jsonl"):
        r = json.loads(l); runerr[(r["suite"], r["ut"], r["it"])] = r["err"]
with open(f"{D}/progent_ab_upstream.jsonl", "w") as o:
    for p in pairs:
        r = up.get(p) or {"suite": p[0], "ut": p[1], "it": p[2], "utility": None, "security": None,
                          "err": runerr.get(p, "missing")}
        o.write(json.dumps(r) + "\n"); up[p] = r
port = {}
for l in open(f"{D}/progent_ab_port.jsonl"):
    r = json.loads(l); port[(r["suite"], r["ut"], r["it"])] = r  # last row wins


def ok(r): return r is not None and r["err"] is None


def rate(rows, key): return sum(bool(r[key]) for r in rows) / len(rows)


def boot(vals, n=10000, seed=20261006):
    rnd = random.Random(seed); m = len(vals); xs = sorted(sum(vals[rnd.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return xs[int(0.025 * n)], xs[int(0.975 * n) - 1]


def mcnemar(a, b):
    n10, n01 = sum(x and not y for x, y in zip(a, b)), sum(y and not x for x, y in zip(a, b)); n = n10 + n01
    if n == 0: return 1.0, n10, n01
    k = min(n10, n01); return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n), n10, n01


out, verdict = [], {}
for name, sel, key in (("benign utility", lambda p: p[2] is None, "utility"), ("ASR", lambda p: p[2] is not None, "security")):
    ps = [p for p in pairs if sel(p)]
    U = [up[p] for p in ps if ok(up[p])]; Pt = [port[p] for p in ps if ok(port.get(p))]
    both = [p for p in ps if ok(up[p]) and ok(port.get(p))]
    ua, pa = [bool(up[p][key]) for p in both], [bool(port[p][key]) for p in both]
    lo, hi = boot([float(bool(r[key])) for r in U]); pt = rate(Pt, key)
    pv, n10, n01 = mcnemar(ua, pa)
    inside = lo <= pt <= hi; verdict[name] = inside
    out.append(f"- {name}: upstream {rate(U, key):.3f} (n={len(U)}/{len(ps)}) CI [{lo:.3f}, {hi:.3f}]; port {pt:.3f} (n={len(Pt)}/{len(ps)}); "
               f"port inside CI: {inside}; paired agree {sum(x == y for x, y in zip(ua, pa))}/{len(both)}; "
               f"McNemar exact p={pv:.4f} (up-only {n10}, port-only {n01})")
res = ["# Progent A/B: official upstream fork vs our port (gpt-4o-2024-08-06, agent + policy, update on)", "",
       f"Pairs: {len(pairs)} ({sum(p[2] is None for p in pairs)} benign, {sum(p[2] is not None for p in pairs)} attack), banking + slack.",
       "Bootstrap: pair-level, 10,000 resamples, seed 20261006, percentile 95% CI on the upstream side.", *out, "",
       f"Run errors: upstream {sum(1 for p in pairs if not ok(up[p]))}, port {sum(1 for p in pairs if not ok(port.get(p)))}.",
       f"**VERDICT: {'PASS' if all(verdict.values()) else 'FAIL'}**"]
open(f"{D}/progent_ab_RESULT.md", "w").write("\n".join(res) + "\n"); print("\n".join(res))
