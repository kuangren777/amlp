"""Control for the A2 residual (project lead): resend today, through the same gateway token, the exact first-call
prompt (system message, user request, tool schemas) of the A2 single-call rows and compare billed prompt_tokens with
the value billed during the run. Equal values mean the run calls carried no prefix beyond today's channel, whose
fixed offset the analyst probe measured at 7 tokens. usage: python3 analysis/injection_replay_control.py"""
import json, os, sys
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(D)   # release root (code/ and data/ are siblings)
sys.path.insert(0, D)
import a2_agentdyn as A  # noqa: E402
from agentdojo.agent_pipeline.llms.openai_llm import _function_to_openai  # noqa: E402
cl = A.harness.CLIENT
seen, out = set(), []
for p in ("monitor", "attack"):
    for m in ("gpt-4o-mini-2024-07-18", "gpt-4.1-mini-2025-04-14"):
        for r in map(json.loads, open(f"{A.DATA}/{p}_{m}.jsonl")):
            k = (m, r["suite"], r["ut"])
            if r["usage"]["calls"] != 1 or r["it"] is not None or k in seen:
                continue
            seen.add(k)
            S = A.suite(r["suite"])
            msgs = [{"role": "system", "content": A.harness.SYSTEM_MESSAGE}, {"role": "user", "content": S.user_tasks[r["ut"]].PROMPT}]
            x = cl.chat.completions.create(model=m, messages=msgs, tools=[_function_to_openai(t) for t in S.tools], max_tokens=1)
            out.append({"model": m, "suite": r["suite"], "ut": r["ut"], "billed_run": r["usage"]["prompt_tokens"],
                        "billed_today": x.usage.prompt_tokens, "served_today": x.model})
            print(out[-1])
json.dump({"ts": A.C.now_sgt(), "rows": out}, open(f"{ROOT}/data/analysis_out/injection/injection_replay_control.json", "w"), indent=1)
