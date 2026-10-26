"""S5: agreement of each attack oracle with blind labels on the stratified sample (analysis/out/s5/labels_blind.jsonl).

Reports, per oracle (exec, agentdojo, replay, full): accuracy on the sample, accuracy reweighted to the full population
of attack rows by the (exec, agentdojo) strata, false positives and false negatives, each sample proportion with a
Wilson 95% confidence interval; per-stratum accuracy; and a sensitivity check without the items the labeller called
ambiguous. Writes analysis/out/s5/label_eval.json.
"""
import collections
import json
import math
import os

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "s5")   # label sample
SET = os.environ.get("HOLDOUT_SET", "clean27")                                 # rescoring outputs and population weights
DS = os.path.join(D, SET)
AMBIGUOUS_IDX = {14, 4, 60, 78, 87, 131, 133, 140, 91, 108, 63, 75, 96, 26, 105}   # labeller's list, 0-based lines


def wilson(k, n, z=1.959964):
    if n == 0:
        return [float("nan")] * 3
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [p, c - h, c + h]


scored = {json.loads(l)["row_id"]: json.loads(l) for l in open(os.path.join(DS, "scored.jsonl"))}
labels = [json.loads(l) for l in open(os.path.join(D, "labels_blind.jsonl"))]
meta = json.load(open(os.path.join(D, "label_sample_meta.json")))
assert [x["id"] for x in labels] == meta["sample_ids"], "label order must match the sample"
from s5_labels import keep_labels  # noqa: E402
EXCL = os.path.join(D, "excluded_labels_planA.json")
labels, AMBIGUOUS_IDX = keep_labels(labels, set(json.load(open(EXCL))["label_ids"]) if os.path.exists(EXCL) else set(), AMBIGUOUS_IDX)
pop = dict(collections.Counter(str((x["exec"], x["agentdojo"])) for x in scored.values() if x["has_steps"]))   # strata sizes in this set
res = {"n_sample": len(labels), "n_label_success": sum(x["success"] for x in labels), "population_cells": pop, "oracles": {}}
for subset, idx in (("all", range(len(labels))), ("unambiguous", [i for i in range(len(labels)) if i not in AMBIGUOUS_IDX])):
    for o in ("exec", "agentdojo", "replay", "replay_any", "full"):
        rows = [(scored[labels[i]["id"]], labels[i]["success"]) for i in idx]
        rows = [(s, y) for s, y in rows if s[o] is not None]
        ok = sum(int(s[o] == y) for s, y in rows)
        fp = sum(int(s[o] == 1 and y == 0) for s, y in rows); npos = sum(int(y == 0) for _, y in rows)
        fn = sum(int(s[o] == 0 and y == 1) for s, y in rows); nneg = sum(int(y == 1) for _, y in rows)
        strata = collections.defaultdict(lambda: [0, 0])
        for s, y in rows:
            c = str((s["exec"], s["agentdojo"]))
            strata[c][0] += int(s[o] == y); strata[c][1] += 1
        tot = sum(pop[c] for c in strata)
        reweighted = sum(pop[c] * a / n for c, (a, n) in strata.items()) / tot
        res["oracles"].setdefault(subset, {})[o] = {
            "n": len(rows), "accuracy": wilson(ok, len(rows)), "accuracy_k": ok,
            "false_positive_rate": wilson(fp, npos), "fp_k": fp, "n_label_fail": npos,
            "false_negative_rate": wilson(fn, nneg), "fn_k": fn, "n_label_success": nneg,
            "accuracy_population_weighted": reweighted,
            "per_stratum": {c: {"correct": a, "n": n, "accuracy": wilson(a, n)} for c, (a, n) in sorted(strata.items())}}
res["holdout_set"] = SET
json.dump(res, open(os.path.join(DS, "label_eval.json"), "w"), indent=1)
for subset, d in res["oracles"].items():
    for o, x in d.items():
        a = x["accuracy"]
        print(f"{subset:11s} {o:9s} acc {x['accuracy_k']}/{x['n']} = {100*a[0]:.1f}% (95% Wilson {100*a[1]:.1f}-{100*a[2]:.1f}%), "
              f"FP {x['fp_k']}/{x['n_label_fail']}, FN {x['fn_k']}/{x['n_label_success']}, pop-weighted {100*x['accuracy_population_weighted']:.1f}%")
