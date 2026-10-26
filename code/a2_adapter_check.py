"""A2 adapter checks (plan.md §25.2), none of which runs an agent or judges an agent outcome.
(i)  autodojo and ISSTA26/agentdojo load identical default environments for the four AgentDojo suites
     (one subprocess per package, sha256 of the JSON dump of the injected default environment);
(ii) on every AgentDyn user task, replaying its ground-truth calls (a2_replay.replay) gives utility True, and the
     injection vectors the harness would use are listed.
usage: python3 a2_adapter_check.py  -> data/a2_checks/adapter_check.json"""
from __future__ import annotations

import json
import os
import subprocess
import sys

D = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(D)   # release root (code/ and data/ are siblings)
OUT = f"{ROOT}/data/a2_checks/adapter_check.json"
ROOTS = {"autodojo": os.environ.get("AUTODOJO_SRC", ""),
         "issta26": os.environ.get("AGENTDOJO_SRC", "")}
_DUMP = r"""
import hashlib, json, sys
sys.path.insert(0, sys.argv[1])
import agentdojo
assert agentdojo.__file__.startswith(sys.argv[1])
from agentdojo.task_suite.load_suites import get_suite
out = {}
for s in ("banking", "slack", "travel", "workspace"):
    S = get_suite("v1.2", s)
    env = S.load_and_inject_default_environment(S.get_injection_vector_defaults())
    out[s] = {"env": hashlib.sha256(json.dumps(env.model_dump(mode="json"), sort_keys=True, default=str).encode()).hexdigest(),
              "user_tasks": len(S.user_tasks), "injection_tasks": len(S.injection_tasks),
              "tools": sorted(t.name for t in S.tools)}
print(json.dumps(out))
"""


def check_parity() -> dict:
    got = {k: json.loads(subprocess.run([sys.executable, "-I", "-c", _DUMP, r], capture_output=True, text=True,
                                        check=True).stdout) for k, r in ROOTS.items()}
    return {s: {"identical": got["autodojo"][s] == got["issta26"][s], **{k: got[k][s]["env"][:16] for k in ROOTS}}
            for s in got["issta26"]}


def check_ground_truth() -> dict:
    import a2_agentdyn as A
    import a2_replay as RP
    out = {}
    for s, ut in A.tasks():
        S = A.suite(s)
        t = S.user_tasks[ut]
        ok, err = RP.user_task_gt_utility(S, t)
        out[f"{s}|{ut}"] = {"utility": ok, "err": err, "vectors": A.harness.injection_candidates(s, ut)}
    return out


def main():
    par = check_parity()
    gt = check_ground_truth()
    res = {"parity": par, "parity_ok": all(v["identical"] for v in par.values()),
           "gt_utility_true": sum(v["utility"] is True for v in gt.values()), "n_tasks": len(gt),
           "no_vector_tasks": sorted(k for k, v in gt.items() if not v["vectors"]), "ground_truth": gt}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print("parity_ok", res["parity_ok"], "| gt utility True", res["gt_utility_true"], "/", res["n_tasks"],
          "| tasks without injection vector", len(res["no_vector_tasks"]))


if __name__ == "__main__":
    main()
