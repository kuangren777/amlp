"""M2a online pilot. Resumable (rows already logged are skipped).

pairs : every user task once without injection (benign) + once with injection (first task of TripWire's rotation)
arms  : none | amlp | llm_env (tool layer only) | block_all (no side-effecting call allowed)
envelopes are task-disjoint: mined from snapshot benign runs of OTHER user tasks (see envbuild.py)
usage : AMLP_CFG='{"predictor": "...", "r": 1, "lift_emails": true, "use_env": true}' python3 exp_pilot.py MODEL WORKERS
"""
import json, os, sys, threading
from concurrent.futures import ThreadPoolExecutor
D = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(D)   # release root (code/ and data/ are siblings)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, D)
import harness
from agentdojo.agent_pipeline import AgentPipeline, InitQuery, SystemMessage, ToolsExecutionLoop
from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM
from amlp import AMLPExecutor, Envelope
import envbuild as B

ARMS = os.environ.get("ARMS", "none,amlp,llm_env,block_all").split(",")
CFG = json.loads(os.environ.get("AMLP_CFG", '{"predictor": "gpt-4.1-mini-2025-04-14", "r": 1, "lift_emails": true, "use_env": true}'))
LOG = f"{ROOT}/data/pilot_m2a.jsonl"
_tl = threading.local()
_orig = harness.make_pipeline


def make_pipeline(model, temperature=1.0, system_message=harness.SYSTEM_MESSAGE, defense=None):
    if defense not in ("amlp", "llm_env", "block_all"):
        return _orig(model, temperature, system_message, defense)
    s, ut = _tl.task
    if defense == "block_all":
        env = Envelope(s, set(), {}, {}, set(), "", B.PROMPTS[(s, ut)])
    else:
        env = B.build(s, ut, **CFG)
        env.check_values = defense == "amlp"
    ex = AMLPExecutor(envelope=env, mode="block")
    _tl.ex = ex
    llm = OpenAILLM(harness.client_for(model), model, temperature=temperature)
    p = AgentPipeline([SystemMessage(system_message), InitQuery(), llm, ToolsExecutionLoop([ex, llm], max_iters=12)])
    p.name = model
    return p


harness.make_pipeline = make_pipeline


def pairs():
    out = []
    for s in B.SIDE:
        S = harness.suite(s)
        its = sorted(S.injection_tasks, key=lambda x: int(x.split("_")[-1]))
        for i, ut in enumerate(sorted(S.user_tasks, key=lambda x: int(x.split("_")[-1]))):
            out += [(s, ut, None), (s, ut, its[i % len(its)])]
    return out


def served_root(model):
    """Served model root from /v1/models for local vLLM; hub models only expose the requested id."""
    if model not in harness.LOCAL_MODELS:
        return f"hub:{model}"
    c = harness.client_for(model)
    c = getattr(c, "inner", c)
    return c.models.list().data[0].root


def main():
    model, workers = sys.argv[1], int(sys.argv[2])
    root = served_root(model)
    os.makedirs(f"{ROOT}/data", exist_ok=True)
    done = set()
    if os.path.exists(LOG):
        for l in open(LOG):
            r = json.loads(l)
            if r["model"] == model and r["err"] is None:
                done.add((r["defense"], r["suite"], r["ut"], r["it"]))
    jobs = [(a, s, ut, it) for a in ARMS for (s, ut, it) in pairs() if (a, s, ut, it) not in done]
    print(model, "root", root, "todo", len(jobs), "cfg", CFG, flush=True)
    lk, n = threading.Lock(), [0]

    def go(j):
        a, s, ut, it = j
        _tl.task, _tl.ex = (s, ut), None
        r = harness.run_pair(model, s, ut, it, harness.default_injection(s, it) if it else None,
                             defense=None if a == "none" else a, tag="m2a_r0")
        if _tl.ex is not None:
            r["amlp_flags"], r["amlp_calls"] = _tl.ex.flags, _tl.ex.calls
        r["amlp_cfg"] = CFG if a in ("amlp", "llm_env") else None
        r["served_root"] = root
        with lk:
            with open(LOG, "a") as f:
                f.write(json.dumps(r, default=str) + "\n")
            n[0] += 1
            if n[0] % 25 == 0:
                print(model, n[0], "/", len(jobs), flush=True)
        return r["err"]
    with ThreadPoolExecutor(workers) as ex:
        errs = [e for e in ex.map(go, jobs) if e]
    print(model, "finished, errors", len(errs), errs[:3], flush=True)


if __name__ == "__main__":
    main()
