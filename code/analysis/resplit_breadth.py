"""Tool breadth in kept re-splits (plan.md §22.10 C8, descriptive, post-hoc, pre-registered c622c64c1 before computing).
Groups the splits whose λ̂ keeps the value layer at ε = 0.10 by λ̂'s r (r ≤ 4 versus r ≥ 8 or all) and reports the
holdout false-block and flag rates of each group. Writes data/analysis_out/resplit_breadth.json."""
import json
import os

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(D)   # release root (code/ and data/ are siblings)
rs = json.load(open(f"{ROOT}/data/analysis_out/resplit.json"))
out = {}
for m, x in rs["models"].items():
    kept = [s["0.1"] for s in x["splits"] if s["0.1"]["keeps_value_layer"]]
    g = {"narrow_r_le4": [k for k in kept if k["r"] <= 4], "wide_r_ge8": [k for k in kept if k["r"] >= 8]}
    out[m] = {name: {"n": len(v), "share_of_kept": len(v) / len(kept) if kept else None,
                     "fb": sum(k["holdout_fb"] for k in v) / len(v) if v else None,
                     "flag": (sum(k["flag"] for k in v if k["flag"] is not None) / sum(k["flag"] is not None for k in v))
                     if any(k["flag"] is not None for k in v) else None}
              for name, v in g.items()}
json.dump(out, open(f"{ROOT}/data/analysis_out/resplit_breadth.json", "w"), indent=1)
for m, x in out.items():
    print(m[:14], {k: (v["n"], v["fb"] and round(100 * v["fb"], 1), v["flag"] and round(100 * v["flag"], 1)) for k, v in x.items()})
