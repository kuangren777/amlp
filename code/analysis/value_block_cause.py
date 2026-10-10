"""Why the value layer never survives calibration for Llama-3.1-8B (plan.md §22.6, descriptive): per model, the cause
of the first exact-level (r = all) refusal on each benign holdout run (tool vs argument value) and the benign utility of
those runs. Writes data/analysis_out/value_block_cause.json."""
import collections
import json
import os
import sys

os.environ["HOLDOUT_SET"] = "all48"
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(D)   # release root (code/ and data/ are siblings)
sys.path.insert(0, D)
sys.path.insert(0, f"{D}/analysis")
import m3_common as C          # noqa: E402
import m3_envelope as ME       # noqa: E402
import m3_select as MS         # noqa: E402
from rq12 import sel_entry, holdout_benign  # noqa: E402

out = {}
for m in C.MODELS:
    e = sel_entry(m, MS.PRIMARY_EPS)
    pool = ME.Pool(m, False, k=e["k"])
    ben = [r for r in holdout_benign(m) if r["rep"] < e["k"]]
    c, blocked_util, cache = collections.Counter(), [], {}
    for r in ben:
        t = (r["suite"], r["ut"])
        if t not in cache:
            cache[t] = ME.build(pool, *t, exclude_tasks=C.HOLDOUT, predictor=e["chosen"]["predictor"], r=1000, level="exact")
        for cl in r["trace"]:
            v = cache[t].violations(cl["f"], dict(cl["a"]))
            if v:
                c["tool" if v == ["<tool>"] else "value"] += 1
                blocked_util.append(bool(r["utility"]))
                break
    out[m] = {"runs": len(ben), "blocked_tool": c["tool"], "blocked_value": c["value"],
              "benign_utility": sum(bool(r["utility"]) for r in ben) / len(ben),
              "utility_of_blocked": sum(blocked_util) / len(blocked_util) if blocked_util else None}
json.dump(out, open(f"{ROOT}/data/analysis_out/value_block_cause.json", "w"), indent=1)
print(json.dumps(out, indent=1))
