"""S5-fresh evaluation (review/s5_fresh/AMENDMENT.md section 5): accuracy of exec, replay_any and full against the new
blind labels, on the sample and reweighted to the population by the (exec, replay_any) strata, with false positive and
false negative rates, Wilson 95% intervals, and a sensitivity check without the items the labeller called ambiguous.

    python3 analysis/s5_fresh/eval.py [labels.jsonl]   -> analysis/s5_fresh/eval.json
"""
import collections
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
S5 = os.path.join(os.path.dirname(HERE), "out", "s5")
REVIEW = os.path.join(os.path.dirname(HERE), "..", "review", "s5_fresh")
ORACLES = ("exec", "replay_any", "full")


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None, None]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [p, c - h, c + h]


def evaluate(labels, smap, scored):
    pop = collections.Counter((x["exec"], x["replay_any"]) for x in scored.values() if x["has_steps"])
    out = {}
    for subset in ("all", "unambiguous"):
        lab = [l for l in labels if subset == "all" or not l.get("ambiguous")]
        res = {}
        for o in ORACLES:
            rows = [(scored[smap[l["item"]]], int(l["success"])) for l in lab]
            rows = [(s, y) for s, y in rows if s[o] is not None]
            ok = sum(int(s[o] == y) for s, y in rows)
            fp = sum(int(s[o] == 1 and y == 0) for s, y in rows); nf = sum(int(y == 0) for _, y in rows)
            fn = sum(int(s[o] == 0 and y == 1) for s, y in rows); ns = sum(int(y == 1) for _, y in rows)
            strata = collections.defaultdict(lambda: [0, 0])
            for s, y in rows:
                c = (s["exec"], s["replay_any"])
                strata[c][0] += int(s[o] == y); strata[c][1] += 1
            tot = sum(pop[c] for c in strata)
            res[o] = dict(n=len(rows), correct=ok, accuracy=wilson(ok, len(rows)), fp=fp, n_label_fail=nf,
                          fp_rate=wilson(fp, nf), fn=fn, n_label_success=ns, fn_rate=wilson(fn, ns),
                          accuracy_population_weighted=sum(pop[c] * a / n for c, (a, n) in strata.items()) / tot,
                          per_stratum={str(c): dict(correct=a, n=n, accuracy=wilson(a, n)) for c, (a, n) in sorted(strata.items())})
        out[subset] = res
    out["population_strata"] = {str(c): n for c, n in pop.items()}
    return out


def main():
    lab_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REVIEW, "labels.jsonl")
    labels = [json.loads(l) for l in open(lab_path)]
    smap = json.load(open(os.path.join(HERE, "sample_map.json")))["item_to_row"]
    assert sorted(l["item"] for l in labels) == sorted(smap), "labels must cover exactly the sampled items"
    scored = {json.loads(l)["row_id"]: json.loads(l) for l in open(os.path.join(S5, "all48", "scored.jsonl"))}
    res = evaluate(labels, smap, scored)
    res["n_label_success"] = sum(int(l["success"]) for l in labels)
    res["n_ambiguous"] = sum(bool(l.get("ambiguous")) for l in labels)
    json.dump(res, open(os.path.join(HERE, "eval.json"), "w"), indent=1)
    for subset in ("all", "unambiguous"):
        for o, x in res[subset].items():
            a = x["accuracy"]
            print(f"{subset:11s} {o:10s} {x['correct']}/{x['n']} = {100*a[0]:.1f}% (95% Wilson {100*a[1]:.1f} to {100*a[2]:.1f}), "
                  f"FP {x['fp']}/{x['n_label_fail']}, FN {x['fn']}/{x['n_label_success']}, pop-weighted {100*x['accuracy_population_weighted']:.1f}%")


if __name__ == "__main__":
    main()
