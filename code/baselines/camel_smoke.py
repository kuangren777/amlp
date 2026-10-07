"""Smoke: 5 CaMeL runs (secpol on, important_instructions) via hosted-gateway. Run: camel_env/bin/python camel_smoke.py"""
import json, os, sys
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
import camel_run as C

M = "gpt-4o-mini-2024-07-18"
out = f"{D}/camel_smoke.jsonl"
open(out, "w").close()
for suite, ut, it in [("banking", "user_task_0", None), ("banking", "user_task_1", None),
                      ("banking", "user_task_0", "injection_task_0"), ("banking", "user_task_1", "injection_task_0"),
                      ("workspace", "user_task_0", None)]:
    r = C.run_row(M, suite, ut, it, q_model=M, attack="important_instructions", secpol=True, log=out)
    print(suite, ut, it, "utility", r["utility"], "security", r["security"], "err", r["err"],
          "denied", r["camel_denied"], "ncalls", len(r["trace"]), "secs", r["secs"], flush=True)
