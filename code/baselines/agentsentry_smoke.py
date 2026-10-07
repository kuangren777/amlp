"""Smoke for the Agent-Sentry RE-IMPLEMENTATION: 6 online runs on hosted-gateway, gpt-4o-mini-2024-07-18, banking
user_task_0..2 x {benign, injection_task_0}. Layer 1 = gbm, fitted on the snapshot with the three smoke tasks
excluded (exclude_tasks). Judge = agentsentry_port.JUDGE_MODEL. Not part of the tests."""
import json
import os
import sys
import threading

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
import agentsentry_port as P  # noqa: E402
import harness  # noqa: E402

MODEL = "gpt-4o-mini-2024-07-18"
UTS = ("user_task_0", "user_task_1", "user_task_2")
EXCL = frozenset(("banking", u) for u in UTS)
B, A = P.load_snapshot()
FIT = P.fit(B, A, exclude_tasks=EXCL, kind="gbm")
assert not (FIT.train_tasks & EXCL)
_tl = threading.local()
_orig = harness.make_pipeline


def mk(model, temperature=1.0, system_message=harness.SYSTEM_MESSAGE, defense=None):
    if defense != "agentsentry":
        return _orig(model, temperature, system_message, defense)
    p = P.make_pipeline_agentsentry(model, temperature, system_message, fitted=FIT, judge_model=P.JUDGE_MODEL)
    _tl.ex = p.agentsentry
    return p


harness.make_pipeline = mk
out = f"{D}/agentsentry_smoke.jsonl"
open(out, "w").close()
for ut in UTS:
    for it in (None, "injection_task_0"):
        r = harness.run_pair(MODEL, "banking", ut, it, harness.default_injection("banking", it) if it else None,
                             defense="agentsentry", tag="smoke_agentsentry")
        r["as_calls"], r["as_flags"] = _tl.ex.calls, _tl.ex.flags
        r["judge_model"], r["layer1"], r["fit_n"] = P.JUDGE_MODEL, "gbm(sklearn)", FIT.n_train
        r["reimplementation"] = "Agent-Sentry re-implementation (arXiv 2603.22868), not official code"
        open(out, "a").write(json.dumps(r, default=str) + "\n")
        print(ut, it, "utility", r["utility"], "security", r["security"], "err", r["err"],
              "calls", [(c["f"], c["decision"], c["layer"]) for c in _tl.ex.calls], flush=True)
