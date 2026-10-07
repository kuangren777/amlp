"""Smoke for the ToolFence RE-IMPLEMENTATION (arXiv 2609.37196): banking, agent gpt-4o-mini-2024-07-18 via hosted-gateway.
v2: judge mode (paper's full path, the default) on user_task_0..2 x {benign, injection_task_0} = 6 agent runs.
v1 (block x6 + judge x4, pre prompt fix) is kept in toolfence_smoke_v1.jsonl. Not part of the tests."""
import json
import os
import sys
import threading

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
import toolfence_port as T  # noqa: E402
import harness  # noqa: E402

MODEL = "gpt-4o-mini-2024-07-18"
COMPILER = os.environ.get("TF_COMPILER_MODEL", "gpt-4o-2024-08-06")   # paper's GPT-4o backbone, exact snapshot on hub
_tl = threading.local()
_orig = harness.make_pipeline


def mk(model, temperature=1.0, system_message=harness.SYSTEM_MESSAGE, defense=None):
    if defense not in ("toolfence_block", "toolfence_judge"):
        return _orig(model, temperature, system_message, defense)
    p = T.make_pipeline_toolfence(model, temperature, system_message, compiler_model=COMPILER,
                                  fallback=defense.split("_")[1])
    _tl.ex = p.toolfence
    return p


harness.make_pipeline = mk
out = f"{D}/toolfence_smoke.jsonl"
open(out, "w").close()
T.reset_grant_cache()
# v2 (af4 decision 4): judge mode only (paper's full method), 6 runs. v1 rows: toolfence_smoke_v1.jsonl
plan = [("toolfence_judge", u, i) for u in ("user_task_0", "user_task_1", "user_task_2") for i in (None, "injection_task_0")]
for mode, ut, it in plan:
    r = harness.run_pair(MODEL, "banking", ut, it, harness.default_injection("banking", it) if it else None,
                         defense=mode, tag="toolfence_smoke_REIMPL")
    r.update(_tl.ex.row())
    r["compiler_model"] = r["judge_model"] = COMPILER
    r["impl"] = "re-implementation of arXiv 2609.37196 (no public code)"
    with open(out, "a") as f:
        f.write(json.dumps(r, default=str) + "\n")
    print(mode, ut, it, "utility", r["utility"], "security", r["security"], "err", r["err"],
          "fallbacks", r["toolfence_fallbacks"], "judge", r["toolfence_judge_calls"], "grants", r["toolfence_grants"],
          "calls", [(c["f"], c["path"]) for c in r["toolfence_calls"]], flush=True)
