"""P1 (reuse-only, go/no-go): leave-one-task-out envelope with prompt retrieval. No task identity.
Envelope for held-out task t = side-effecting tools used in benign runs (all models except target) of the
r nearest OTHER user tasks of the same suite, by bge-m3 cosine on the user prompt. Test on target model's
benign run of t (false-block) and its tw_monitor attack runs on t (violations flagged). Monitor replay only."""
import collections, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import harness, openai
from amlp_common import SIDE, SNAP, side_calls

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "pilot", "snap", "prompt_emb_bge-m3.json")
prompts = {(s, ut): t.PROMPT for s in SIDE for ut, t in harness.suite(s).user_tasks.items()}
if os.path.exists(CACHE):
    emb = {tuple(k.split("|")): v for k, v in json.load(open(CACHE)).items()}
else:
    keys = list(prompts)
    r = harness.CLIENT.embeddings.create(model="bge-m3", input=[prompts[k] for k in keys])
    emb = {k: d.embedding for k, d in zip(keys, r.data)}
    json.dump({"|".join(k): v for k, v in emb.items()}, open(CACHE, "w"))
E = {k: np.array(v) / np.linalg.norm(v) for k, v in emb.items()}

rows = [json.loads(l) for l in open(SNAP)]
rows = [r for r in rows if r["defense"] == "tw_monitor" and r["err"] is None]
ben = collections.defaultdict(list)
for r in rows:
    if r["it"] is None: ben[(r["suite"], r["ut"])].append(r)
models = sorted({r["model"] for r in rows})

def envelope(t, m, rr):
    s = t[0]
    others = sorted((k for k in ben if k[0] == s and k != t), key=lambda k: -float(E[k] @ E[t]))
    nb = others if rr == "all" else others[:rr]
    return set().union(*[side_calls(s, o["trace"]) for k in nb for o in ben[k] if o["model"] != m])

print("| r | benign false-block | violations flagged | flag rate, attack runs w/o violation |")
print("|---|---|---|---|")
for rr in (0, 1, 2, 3, 5, "all", "oracle"):
    c = collections.Counter(); per_suite = collections.Counter()
    for r in rows:
        t = (r["suite"], r["ut"]); m = r["model"]
        if rr == "oracle":
            env = set().union(*[side_calls(t[0], o["trace"]) for o in ben[t] if o["model"] != m])
        else:
            env = set() if rr == 0 else envelope(t, m, rr)
        flag = bool(side_calls(t[0], r["trace"]) - env)
        if r["it"] is None: c["bn"] += 1; c["bfb"] += flag; per_suite[(t[0], "bn")] += 1; per_suite[(t[0], "fb")] += flag
        elif r["security"]: c["v"] += 1; c["tp"] += flag
        else: c["nv"] += 1; c["fa"] += flag
    ps = " ".join(f"{s}:{per_suite[(s,'fb')]/max(1,per_suite[(s,'bn')]):.2f}" for s in SIDE)
    print(f"| {rr} | {c['bfb']}/{c['bn']} = {c['bfb']/c['bn']:.3f} ({ps}) | {c['tp']}/{c['v']} = {c['tp']/max(1,c['v']):.2f} | {c['fa']/c['nv']:.3f} |")
print("models:", models)
