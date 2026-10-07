"""Progent A/B, our-port side. Same pairs as progent_ab_pairs.json, mode progent_llm, update=True, agent+policy
gpt-4o-2024-08-06, temperature=None (parameter omitted, exactly what the upstream fork sends: its OpenAILLM does
`temperature or NOT_GIVEN` with default 0.0). Own LOG; resumable (ok rows skipped).
usage: python3 baselines/progent_ab_port.py [WORKERS]"""
import json, os, sys, threading
from concurrent.futures import ThreadPoolExecutor
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import harness
import progent_port as P

MODEL = POLICY = "gpt-4o-2024-08-06"
TEMP = None
LOG = f"{D}/progent_ab_port.jsonl"
_tl = threading.local()
_orig = harness.make_pipeline


def mk(model, temperature=1.0, system_message=harness.SYSTEM_MESSAGE, defense=None):
    if defense != "progent":
        return _orig(model, temperature, system_message, defense)
    p = P.make_pipeline_progent(model, temperature, system_message, "progent_llm", policy_model=POLICY, update=True)
    _tl.ex = p.progent
    return p


harness.make_pipeline = mk


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    pairs = [tuple(p) for p in json.load(open(f"{D}/progent_ab_pairs.json"))]
    done = set()
    if os.path.exists(LOG):
        for l in open(LOG):
            r = json.loads(l)
            if r["err"] is None:
                done.add((r["suite"], r["ut"], r["it"]))
    jobs = [p for p in pairs if p not in done]
    print("todo", len(jobs), flush=True)
    lk = threading.Lock()

    def go(j):
        s, ut, it = j
        P.TASK.v, _tl.ex = (s, ut), None
        r = harness.run_pair(MODEL, s, ut, it, harness.default_injection(s, it) if it else None,
                             temperature=TEMP, defense="progent", tag="progent_ab")
        if _tl.ex is not None:
            r["progent_flags"], r["progent_calls"], r["progent_updates"] = _tl.ex.flags, _tl.ex.calls, _tl.ex.updates
        r["policy_model"], r["served_root"] = POLICY, f"hub:{MODEL}"
        with lk:
            open(LOG, "a").write(json.dumps(r, default=str) + "\n")
        return r["err"]
    with ThreadPoolExecutor(workers) as ex:
        errs = [e for e in ex.map(go, jobs) if e]
    print("finished, errors", len(errs), errs[:3], flush=True)


if __name__ == "__main__":
    main()
