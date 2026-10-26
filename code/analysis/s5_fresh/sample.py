"""S5-fresh sample (review/s5_fresh/AMENDMENT.md, internal review): a new stratified blind label sample.

Population: attack rows with recorded steps in analysis/out/s5/all48/scored.jsonl (plan-A data). The 160 rows of the
original S5 sample are excluded. Strata (exec, replay_any), 30 rows each, seed 20261021. The blind file carries only the
attacker goal, the user-task prompt and the executed calls with their errors, under opaque item ids; the id map is
written to a separate file that the labeller does not read.

    python3 analysis/s5_fresh/sample.py   -> review/s5_fresh/blind.jsonl, analysis/s5_fresh/sample_map.json
"""
import collections
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.dirname(HERE)
sys.path.insert(0, AN)
import rq34 as RQ                 # noqa: E402,F401  (sets sys.path for m3_common / harness)
import exec_oracle as EO          # noqa: E402
import m3_common as C             # noqa: E402

SEED, K = 20261021, 30
S5 = os.path.join(AN, "out", "s5")
REVIEW = os.path.join(AN, "..", "review", "s5_fresh")


def draw():
    scored = [json.loads(l) for l in open(os.path.join(S5, "all48", "scored.jsonl"))]
    old = set(json.load(open(os.path.join(S5, "label_sample_meta.json")))["sample_ids"])
    cells = collections.defaultdict(list)
    for x in scored:
        if x["has_steps"] and x["row_id"] not in old:
            cells[(x["exec"], x["replay_any"])].append(x)
    rng = random.Random(SEED)
    pick = []
    for c in sorted(cells, key=str):
        xs = sorted(cells[c], key=lambda z: z["row_id"])
        pick += rng.sample(xs, min(K, len(xs)))
    rng.shuffle(pick)
    return pick, {str(c): len(v) for c, v in cells.items()}, len(old)


def main():
    pick, sizes, n_old = draw()
    st = {m: EO.steps(m) for m in sorted({x["model"] for x in pick})}
    blind, idmap = [], {}
    for n, x in enumerate(pick):
        S = C.harness.suite(x["suite"])
        steps = st[x["model"]][x["row_id"]]
        item = f"item_{n:03d}"
        idmap[item] = x["row_id"]
        blind.append({"item": item, "suite": x["suite"], "attacker_goal": S.injection_tasks[x["it"]].GOAL,
                      "user_task": S.user_tasks[x["ut"]].PROMPT,
                      "executed_calls": [{"function": s["function"], "args": s["args"], "error": s.get("error")}
                                         for s in steps]})
    os.makedirs(REVIEW, exist_ok=True)
    with open(os.path.join(REVIEW, "blind.jsonl"), "w") as f:
        for b in blind:
            f.write(json.dumps(b, default=str) + "\n")
    json.dump({"seed": SEED, "per_stratum": K, "strata": "(exec, replay_any)", "stratum_sizes_after_exclusion": sizes,
               "excluded_original_ids": n_old, "n_sample": len(pick), "item_to_row": idmap},
              open(os.path.join(HERE, "sample_map.json"), "w"), indent=1)
    print("sample", len(pick), "strata", sizes)


if __name__ == "__main__":
    main()
