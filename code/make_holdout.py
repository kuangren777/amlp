"""M3 clean holdout (plan.md A3): per suite, floor(n/2) user tasks drawn uniformly at random with a fixed fresh seed,
drawn once before any M3 run. The rest of each suite is split into selection / calibration (half each, same seed).
No task-level result is looked at; the draw depends only on the seed and the sorted task ids."""
import json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness
SEED = 20261006_2
rng = random.Random(SEED)
out = {"seed": SEED, "holdout": {}, "selection": {}, "calibration": {}}
for s in ("banking", "slack", "travel", "workspace"):
    uts = sorted(harness.suite(s).user_tasks, key=lambda x: int(x.split("_")[-1]))
    rng.shuffle(uts)
    h = len(uts) // 2
    rest = uts[h:]
    out["holdout"][s] = sorted(uts[:h], key=lambda x: int(x.split("_")[-1]))
    out["selection"][s] = sorted(rest[: len(rest) // 2], key=lambda x: int(x.split("_")[-1]))
    out["calibration"][s] = sorted(rest[len(rest) // 2:], key=lambda x: int(x.split("_")[-1]))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "holdout_split.json"), "w"), indent=1)
print({k: {s: len(v) for s, v in out[k].items()} for k in ("holdout", "selection", "calibration")})
