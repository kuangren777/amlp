"""Smoke: banking user_task_0..2 benign + injection_task_0, modes progent / progent_llm, via hosted-gateway. Not part of the tests."""
import json, os, sys, threading
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
import progent_port as P
import harness

MODEL = "gpt-4o-mini-2024-07-18"
POLICY_MODEL = os.environ.get("SMOKE_POLICY_MODEL", "gpt-4.1-mini-2025-04-14")
_tl = threading.local()
_orig = harness.make_pipeline


def mk(model, temperature=1.0, system_message=harness.SYSTEM_MESSAGE, defense=None):
    if defense not in ("progent", "progent_llm"):
        return _orig(model, temperature, system_message, defense)
    p = P.make_pipeline_progent(model, temperature, system_message, defense, policy_model=POLICY_MODEL)
    _tl.ex = p.progent
    return p


harness.make_pipeline = mk
out = f"{D}/smoke_progent.jsonl"
open(out, "w").close()
for mode in ("progent", "progent_llm"):
    for ut in ("user_task_0", "user_task_1", "user_task_2"):
        for it in (None, "injection_task_0"):
            P.TASK.v = ("banking", ut)
            r = harness.run_pair(MODEL, "banking", ut, it, harness.default_injection("banking", it) if it else None,
                                 defense=mode, tag="smoke_progent")
            r["progent_calls"], r["progent_flags"] = _tl.ex.calls, _tl.ex.flags
            r["policy_model"] = POLICY_MODEL
            open(out, "a").write(json.dumps(r, default=str) + "\n")
            print(mode, ut, it, "utility", r["utility"], "security", r["security"], "err", r["err"],
                  "blocked", len(_tl.ex.flags), "calls", len(_tl.ex.calls), flush=True)
