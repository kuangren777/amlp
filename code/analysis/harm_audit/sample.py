"""Harm-step audit sample (plan.md §27.2, internal review): 100 of the 357 violations, simple random, seed 20261026.

The blind file carries only the attacker goal, the user-task prompt and every logged call with its index, function,
arguments and error, under opaque item ids. h, its mode and the first_block of the calibrated automaton and Progent go
to sample_map.json, which no labeller reads.

    python3 analysis/harm_audit/sample.py   -> review/harm_audit/blind.jsonl, analysis/harm_audit/sample_map.json
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.dirname(HERE)
sys.path.insert(0, AN)
import flag_before_harm as FB     # noqa: E402  (sets HOLDOUT_SET=all48 and sys.path)
import m3_common as C             # noqa: E402
import m3_judge as MJ             # noqa: E402

SEED, N = 20261026, 100
REVIEW = os.path.join(AN, "..", "review", "harm_audit")


def main():
    rows = {}
    FB.replay(FB.ground_truth(), rows)
    assert len(rows) == 357, len(rows)
    rng = random.Random(SEED)
    pick = rng.sample(sorted(rows), N)
    rng.shuffle(pick)
    steps = {m: MJ.steps_index(C.steps_path(False, "online", m, "main")) for m in sorted({x["model"] for x in rows.values()})}
    blind, idmap = [], {}
    for n, rid in enumerate(pick):
        x = rows[rid]
        S = C.harness.suite(x["suite"])
        item = f"item_{n:03d}"
        idmap[item] = dict(x, row_id=rid)
        blind.append({"item": item, "suite": x["suite"], "attacker_goal": S.injection_tasks[x["it"]].GOAL,
                      "user_task": S.user_tasks[x["ut"]].PROMPT,
                      "calls": [{"index": i, "function": s["function"], "args": s["args"], "error": s.get("error")}
                                for i, s in enumerate(steps[x["model"]][rid])]})
    os.makedirs(REVIEW, exist_ok=True)
    with open(os.path.join(REVIEW, "blind.jsonl"), "w") as f:
        for b in blind:
            f.write(json.dumps(b, default=str) + "\n")
    json.dump({"seed": SEED, "n_population": len(rows), "n_sample": N, "item_to_row": idmap},
              open(os.path.join(HERE, "sample_map.json"), "w"), indent=1)
    print("sample", N, "of", len(rows))


if __name__ == "__main__":
    main()
