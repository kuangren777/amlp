"""Exact McNemar on the paired benign runs behind the AMLP benign cost of each model (cold read 95d6ba9 M3: our
significance rule needs McNemar and the bootstrap). Writes data/analysis_out/<set>/amlp_cost_mcnemar.json.
Usage: HOLDOUT_SET=all48|clean27 python3 analysis/amlp_cost_mcnemar.py"""
import json
import os
import sys

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, D)
sys.path.insert(0, f"{D}/analysis")
import rq34 as Q               # noqa: E402
import m3_common as C          # noqa: E402

out = {}
for m in C.MODELS:
    on, mon = Q.online(m), Q.monitor(m)
    kb, base, it = Q.paired_benign(m, "amlp", on, mon)
    b = sum(base[k] == 1 and it[k] == 0 for k in kb)
    c = sum(base[k] == 0 and it[k] == 1 for k in kb)
    out[m] = {"pairs": len(kb), "none_only": b, "amlp_only": c, "p": C.mcnemar_exact(b, c)}
json.dump(out, open(f"{Q.OUT}/amlp_cost_mcnemar.json", "w"), indent=1)
print(json.dumps(out))
