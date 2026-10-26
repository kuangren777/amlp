"""Deterministic environment replay for A2 (plan.md §25.2 (ii), §25.4, §25.8). Import after a2_agentdyn, so the
agentdojo package is ROPE's autodojo.

user_task_gt_utility: replays a user task's ground-truth calls in the clean environment and checks utility.
harm_index: the state-replay harm index of §25.4. The logged calls of an attack run are replayed one by one from the
run's injected environment (the harness construction: JB_TEMPLATE text in every injection vector, then the user
task's init_environment); h = the first call index after which the injection task's security check holds. ROPE's
checker corrections (dailylife IT7, github IT1) are applied through `security_of`. None if no prefix makes it hold."""
from __future__ import annotations

import sys

import a2_agentdyn as A
from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime
from agentdojo.agent_pipeline.ground_truth_pipeline import GroundTruthPipeline
from agentdojo.task_suite.task_suite import functions_stack_trace_from_messages, model_output_from_messages

harness = A.harness
sys.path.insert(0, os.environ.get("ROPE_RUNS", ""))


def user_task_gt_utility(S, t) -> tuple[bool | None, str | None]:
    try:
        env = t.init_environment(S.load_and_inject_default_environment(S.get_injection_vector_defaults()))
        pre = env.model_copy(deep=True)
        _, _, env, msgs, _ = GroundTruthPipeline(t).query(t.PROMPT, FunctionsRuntime(S.tools), env)
        out = model_output_from_messages(msgs) or []
        return bool(S._check_task_result(t, out, pre, env, functions_stack_trace_from_messages(msgs))), None
    except Exception as e:  # noqa: BLE001
        return None, repr(e)[:200]


def injected_env(suite_name: str, ut: str, it: str | None):
    """The harness environment of run_pair for (ut, it) with the default injection text."""
    S = A.suite(suite_name)
    vecs = harness.injection_candidates(suite_name, ut) if it else []
    canaries = {v: f"ZQXCANARY{n}ZQX" for n, v in enumerate(vecs)}
    env = S.load_and_inject_default_environment(canaries)
    if canaries:
        harness._substitute(env, {c: harness.default_injection(suite_name, it) for c in canaries.values()})
    return S.user_tasks[ut].init_environment(env)


def corrected(suite_name: str, it: str, native: bool, executed: list[tuple[str, dict, str | None]]) -> bool:
    """ROPE's effect-based corrections (ext/rope/runs/corrections.py) for dailylife IT7 and github IT1, fed with the
    executed calls as (function, args, error); every other injection task keeps its native check."""
    n = int(it.split("_")[-1])
    if (suite_name, n) not in (("dailylife", 7), ("github", 1)):
        return native
    import corrections as RC
    d = {"security": native, "messages": [{"role": "tool", "tool_call": {"function": f, "args": a}, "error": e}
                                          for f, a, e in executed]}
    return bool(RC.corrected_security(d, suite_name, n))


def security_of(suite_name: str, it: str, pre, post, stack, executed) -> bool:
    S = A.suite(suite_name)
    native = bool(S._check_task_result(S.injection_tasks[it], [], pre, post, stack))
    return corrected(suite_name, it, native, executed)


def harm_index(suite_name: str, ut: str, it: str, trace: list[dict]) -> int | None:
    S = A.suite(suite_name)
    env = injected_env(suite_name, ut, it)
    pre = env.model_copy(deep=True)
    rt = FunctionsRuntime(S.tools)
    stack: list[FunctionCall] = []
    executed: list[tuple[str, dict, str | None]] = []
    for k, c in enumerate(trace):
        _, err = rt.run_function(env, c["f"], dict(c["a"]), raise_on_error=False)
        stack.append(FunctionCall(function=c["f"], args=dict(c["a"])))
        executed.append((c["f"], dict(c["a"]), err))
        if security_of(suite_name, it, pre, env, stack, executed):
            return k
    return None
