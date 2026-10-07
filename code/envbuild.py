"""Task-disjoint envelope construction shared by replay (pilot/p2b) and online runs (exp_pilot.py).
Mining pool = benign runs in the frozen tw_main snapshot (tw_monitor arm, all models) of OTHER user tasks."""
from __future__ import annotations
import collections, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness
from amlp import SIDE, Envelope, control_values, lift, norm

D = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(D)   # release root (code/ and data/ are siblings)
SNAP = f"{ROOT}/data/pilot/snap/tw_main_20261005.jsonl"
PROMPTS = {(s, ut): t.PROMPT for s in SIDE for ut, t in harness.suite(s).user_tasks.items()}
_emb = json.load(open(f"{ROOT}/data/pilot/snap/prompt_emb_bge-m3.json"))
E = {tuple(k.split("|")): np.array(v) / np.linalg.norm(v) for k, v in _emb.items()}
ENV = json.load(open(f"{ROOT}/data/pilot/snap/clean_env_dump.json"))
ROWS = [r for r in (json.loads(l) for l in open(SNAP)) if r["defense"] == "tw_monitor" and r["err"] is None]
BEN = collections.defaultdict(list)
for r in ROWS:
    if r["it"] is None:
        BEN[(r["suite"], r["ut"])].append(r)
# plan.md A1: typed environment entities = leaf strings whose own field name is an identifier (suite-agnostic)
ID_KEYS = {"email", "emails", "account_email", "iban", "recipient", "recipients", "sender", "cc", "bcc",
           "participants", "users", "channels", "shared_with", "owner", "id", "id_"}


def _typed_entities(o, key=None, out=None):
    out = set() if out is None else out
    if isinstance(o, dict):
        for k, v in o.items():
            _typed_entities(v, k, out)
    elif isinstance(o, (list, tuple)):
        for v in o:
            _typed_entities(v, key, out)
    elif isinstance(o, str) and key in ID_KEYS and o.strip():
        out.add(norm(o))
    return out


ENV_TYPED_CACHE = f"{ROOT}/data/pilot/snap/clean_env_typed.json"
if os.path.exists(ENV_TYPED_CACHE):
    ENV_TYPED = {s: frozenset(v) for s, v in json.load(open(ENV_TYPED_CACHE)).items()}
else:
    ENV_TYPED = {}
    for s in SIDE:
        S = harness.suite(s)
        ENV_TYPED[s] = frozenset(_typed_entities(
            S.load_and_inject_default_environment(S.get_injection_vector_defaults()).model_dump(mode="json")))
    json.dump({s: sorted(v) for s, v in ENV_TYPED.items()}, open(ENV_TYPED_CACHE, "w"), indent=0)
PRED = {}
for m in ("gpt-4.1-mini-2025-04-14", "gpt-5.5"):
    p = f"{ROOT}/data/pilot/snap/llm_envelope_{m}.json"
    if os.path.exists(p):
        PRED[m] = json.load(open(p))


def build(suite: str, ut: str, predictor: str | None = "gpt-4.1-mini-2025-04-14", r: int = 0,
          lift_emails: bool = True, use_env: bool | str = True, exclude_model: str | None = None,
          exclude_tasks: frozenset = frozenset()) -> Envelope:
    """exclude_tasks: (suite, ut) pairs whose benign runs never enter the mining pool (plan.md A4(c)2: holdout)."""
    t = (suite, ut)
    tools = set(PRED[predictor][f"{suite}|{ut}"]) if predictor else set()
    if r:
        nb = sorted((k for k in BEN if k[0] == suite and k != t and k not in exclude_tasks), key=lambda k: -float(E[k] @ E[t]))[:r]
        tools |= {c["f"] for k in nb for o in BEN[k] if o["model"] != exclude_model for c in o["trace"] if c["f"] in SIDE[suite]}
    vals, lifted = collections.defaultdict(set), collections.defaultdict(set)
    for k, rs in BEN.items():
        if k[0] != suite or k == t or k in exclude_tasks:
            continue
        for o in rs:
            if o["model"] == exclude_model:
                continue
            for c in o["trace"]:
                if c["f"] in SIDE[suite]:
                    for a, v in control_values(c["f"], c["a"]):
                        vals[(c["f"], a)].add(v)
                        lifted[(c["f"], a)].add(lift(v))
    la = {k for k in lifted if any("@" in v for v in vals[k])} if lift_emails else set()
    env = ENV_TYPED[suite] if use_env == "typed" else (ENV[suite] if use_env else "")
    return Envelope(suite, tools, dict(vals), dict(lifted), la, env, PROMPTS[t])
