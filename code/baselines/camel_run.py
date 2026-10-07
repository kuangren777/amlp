"""CaMeL (arXiv 2503.18813, google-research/camel-prompt-injection, Apache-2.0) as a defense arm in the agent-fuzz
AgentDojo v1.2 harness. Route A: CaMeL's `PrivilegedLLM` pipeline element is imported unchanged and wrapped so that
`harness.run_pair` produces our usual row schema. Run with `baselines/camel_env/bin/python` (see camel_PORT_NOTES.md).

    CAMEL_REPO=<clone>  (default: the scratchpad clone)  provides nothing at runtime when camel is pip-installed in
    the venv; it is only a fallback sys.path entry.

Public API: configure_hub, register_models, make_hub_client, make_camel_pipeline, run_row, row_from_log.
The hub key is read from the environment (LLM_API_KEY) and is never written anywhere.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time

D = os.path.dirname(os.path.abspath(__file__))
HUB_BASE_URL = os.environ.get("LLM_API_BASE", "")   # full base URL incl. /v1
_CLONE = os.environ.get(
    "CAMEL_REPO", os.path.join(D, "third_party", "camel-prompt-injection"))   # checkout of the upstream CaMeL repo
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openai  # noqa: E402

# agentdojo.task_suite must be imported before agentdojo.default_suites (circular import in 0.1.35, which camel's
# `security_policies` triggers). Importing it here first is the whole fix.
from agentdojo.task_suite import get_suite  # noqa: E402,F401
from agentdojo.agent_pipeline import AgentPipeline, InitQuery  # noqa: E402
from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM  # noqa: E402
from agentdojo.models import MODEL_NAMES  # noqa: E402

DEFAULT_MODELS = {"gpt-4o-2024-08-06": "GPT-4", "gpt-4o-mini-2024-07-18": "GPT-4"}
SUFFIXES = ["", "+camel", "+camel+secpol", "+camel+secpol+strict"]
_tl = threading.local()  # per-thread run config consumed by the patched harness.make_pipeline


def configure_hub(base_url: str = HUB_BASE_URL, key: str | None = None) -> str:
    """Route every OpenAI-compatible client CaMeL creates (P-LLM client here, Q-LLM via pydantic_ai's
    OpenAIProvider, which reads OPENAI_BASE_URL / OPENAI_API_KEY) to hosted-gateway. Process-local env only."""
    key = key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_BASE_URL"] = base_url
    if key:
        os.environ["OPENAI_API_KEY"] = key
    return base_url


def register_models(ids=tuple(DEFAULT_MODELS)) -> None:
    """Replacement for editing `_supported_model_names` in camel/models.py: make agentdojo's MODEL_NAMES know the
    ids (and their +camel suffixes) so attacks can resolve the {model} name. Unknown ids get 'AI model'."""
    for m in ids:
        for s in SUFFIXES:
            MODEL_NAMES.setdefault(f"{m}{s}", DEFAULT_MODELS.get(m, "AI model"))
    try:  # also keep camel's own whitelist consistent when camel is importable
        import camel.models as cm
        for m in ids:
            cm._supported_model_names.setdefault(m, DEFAULT_MODELS.get(m, "AI model"))
    except Exception:
        pass


def make_hub_client(base_url: str = HUB_BASE_URL, key: str | None = None) -> openai.OpenAI:
    key = key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "none"
    return openai.OpenAI(base_url=base_url, api_key=key, timeout=120, max_retries=4)


def make_camel_pipeline(model: str, q_model: str | None = None, suite_name: str = "banking", secpol: bool = True,
                        strict: bool = False, temperature: float = 0.0, client=None) -> AgentPipeline:
    """[InitQuery, PrivilegedLLM] exactly as camel.models.make_tools_pipeline builds it, except (i) the suite's
    security-policy engine is applied online when secpol=True (official: no-policy run + offline replay with
    policies), (ii) the OpenAI client points at hosted-gateway, (iii) P-LLM temperature is explicit (official: 0.0)."""
    register_models([model, q_model or model])
    from camel.interpreter.interpreter import MetadataEvalMode
    from camel.models import _SECURITY_POLICY_ENGINES
    from camel.pipeline_elements.privileged_llm import PrivilegedLLM
    from camel.pipeline_elements.security_policies import ADNoSecurityPolicyEngine
    from camel.security_policy import SecurityPolicyDeniedError
    from agentdojo import types as ad_types

    class _PLLM(PrivilegedLLM):
        """PrivilegedLLM + the denial semantics of camel's replay_task: a SecurityPolicyDeniedError ends the run
        (user-role error message, then an assistant message with the output printed so far); env keeps whatever
        tool calls already executed. Records the denial text in self.denied."""
        denied = None

        def _generate_and_interpret_code(self, query, runtime, namespace, env, messages, plm_msgs, sp, prev, deps):
            try:
                return super()._generate_and_interpret_code(query, runtime, namespace, env, messages, plm_msgs, sp,
                                                            prev, deps)
            except SecurityPolicyDeniedError as e:
                self.denied = str(e)
                err = ad_types.ChatUserMessage(role="user", content=[
                    ad_types.text_content_block_from_string(f"SecurityPolicyDeniedError: {e}")])
                fin = ad_types.ChatAssistantMessage(role="assistant", tool_calls=None, content=[
                    ad_types.text_content_block_from_string(prev)])
                return prev, [], None, [*messages, err, fin], list(plm_msgs), namespace, deps

    if is_oai_reasoning_model(model):
        # upstream camel.models: reasoning models get reasoning_effort and NO temperature; the paper's o-series rows
        # use "high" (fidelity fix 2026-10-06, earlier rows superseded)
        llm = OpenAILLM(client or make_hub_client(), model, REASONING_EFFORT, None)
    else:
        llm = OpenAILLM(client or make_hub_client(), model, None, temperature)
    llm.name = model
    engine = _SECURITY_POLICY_ENGINES[suite_name] if secpol else ADNoSecurityPolicyEngine
    mode = MetadataEvalMode.STRICT if strict else MetadataEvalMode.NORMAL
    p = AgentPipeline([InitQuery(), _PLLM(llm, engine, f"openai:{q_model or model}", eval_mode=mode)])
    p.camel = p.elements[-1]
    p.name = f"{model}+camel" + ("+secpol" if secpol else "") + ("+strict" if strict else "")
    return p


def patch_format_spec_py310() -> None:
    """Port fix (plan.md, CaMeL fidelity): upstream turns an f-string ValueError into a CaMeL interpreter error only if
    str(e) == "Invalid format specifier", the Python 3.10 message. Python 3.12 appends details, so the error escaped
    the interpreter instead of going back to the P-LLM. Re-define the function with a 3.10-equivalent test."""
    import inspect, textwrap
    import camel.interpreter.interpreter as I
    if getattr(I, "_af4_fmt_patched", False):
        return
    src = textwrap.dedent(inspect.getsource(I._eval_formatted_value))
    old = 'if str(e) == "Invalid format specifier":'
    assert src.count(old) == 1, "upstream changed; re-check the format-spec port fix"
    exec(compile(src.replace(old, 'if "format specifier" in str(e):'), I.__file__, "exec"), I.__dict__)
    I._af4_fmt_patched = True


REASONING_EFFORT = os.environ.get("CAMEL_REASONING_EFFORT", "high")


def is_oai_reasoning_model(model: str) -> bool:
    return model.split(":")[-1].startswith(("o1", "o3", "o4"))


def _install_harness_patch():
    patch_format_spec_py310()
    import harness
    if getattr(harness, "_camel_patched", False):
        return harness
    orig = harness.make_pipeline

    def mk(model, temperature=1.0, system_message=harness.SYSTEM_MESSAGE, defense=None):
        if defense not in ("camel", "camel_nosecpol"):
            return orig(model, temperature, system_message, defense)
        c = _tl.cfg
        _tl.pipe = make_camel_pipeline(model, c["q_model"], c["suite"], secpol=defense == "camel",
                                       strict=c["strict"], temperature=c["temp"])
        return _tl.pipe

    harness.make_pipeline, harness._camel_patched = mk, True
    return harness


def injection_text(attack: str, suite_name: str, user_task_id: str, injection_task_id: str, model: str) -> str:
    """attack='important_instructions': AgentDojo's own attack object (the one CaMeL's paper uses; {model} is the
    display name of the model). attack='jb_template': harness.JB_TEMPLATE (same text, {model} -> 'AI assistant')."""
    import harness
    if attack == "jb_template":
        return harness.default_injection(suite_name, injection_task_id)
    from agentdojo.attacks import load_attack
    s = harness.suite(suite_name)

    class _P:  # attack only needs .name to resolve the model display name
        name = f"{model}+camel"
    register_models([model])
    atk = load_attack("important_instructions", s, _P())
    inj = atk.attack(s.user_tasks[user_task_id], s.injection_tasks[injection_task_id])
    return next(iter(inj.values()))


def run_row(model: str, suite_name: str, user_task_id: str, injection_task_id: str | None, q_model: str | None = None,
            attack: str = "important_instructions", secpol: bool = True, strict: bool = False, temp: float = 0.0,
            log: str | None = None, tag: str = "camel") -> dict:
    """One CaMeL run in our row schema (harness.run_pair row + attack/q_model/secpol fields)."""
    configure_hub()
    harness = _install_harness_patch()
    _tl.cfg = {"q_model": q_model or model, "suite": suite_name, "strict": strict, "temp": temp}
    text = injection_text(attack, suite_name, user_task_id, injection_task_id, model) if injection_task_id else None
    defense = "camel" if secpol else "camel_nosecpol"
    r = harness.run_pair(model, suite_name, user_task_id, injection_task_id, text, temperature=temp,
                         tag=tag, defense=defense, keep_msgs=False)
    r.update(attack=attack if injection_task_id else None, q_model=q_model or model, secpol=secpol,
             camel_strict=strict, camel_denied=getattr(_tl.pipe.camel, "denied", None))
    r.setdefault("served_root", None)  # hub-served (not a local vLLM root)
    if log:
        with open(log, "a") as f:
            f.write(json.dumps(r, default=str) + "\n")
    return r


def row_from_log(d: dict, model: str, attack: str | None = None, q_model: str | None = None) -> dict:
    """Map one AgentDojo log JSON (what CaMeL's own `main.py` writes under ./logs/<pipeline>/<suite>/<ut>/<attack>/
    <it>.json) into our row schema. Used for the cross-check / fidelity route that runs the official main.py."""
    msgs = d.get("messages") or []
    trace = []
    for m in msgs:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                trace.append({"f": tc.get("function"), "a": dict(tc.get("args") or {})})
    last = msgs[-1] if msgs else {}
    c = last.get("content")
    final = " ".join(b.get("content", "") for b in c if isinstance(b, dict)) if isinstance(c, list) else (c or "")
    inj = d.get("injections") or {}
    return {
        "tag": "camel_official", "model": model, "suite": d.get("suite_name"), "ut": d.get("user_task_id"),
        "it": d.get("injection_task_id") or None, "defense": "camel", "attack": attack or d.get("attack_type"),
        "q_model": q_model or model, "utility": d.get("utility"),
        "security": d.get("security") if d.get("injection_task_id") else None,
        "err": d.get("error"), "trace": trace, "exposed": bool(inj), "final": str(final)[:600],
        "n_msgs": len(msgs), "secs": round(float(d.get("duration") or 0.0), 1), "ts": time.time(),
        "served_root": None,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    ap.add_argument("--q-model")
    ap.add_argument("--suite", default="banking")
    ap.add_argument("--ut", default="user_task_0")
    ap.add_argument("--it")
    ap.add_argument("--attack", default="important_instructions", choices=["important_instructions", "jb_template"])
    ap.add_argument("--no-secpol", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    r = run_row(a.model, a.suite, a.ut, a.it, a.q_model, a.attack, not a.no_secpol, log=a.out)
    print(json.dumps({k: r[k] for k in ("utility", "security", "err", "secs", "n_msgs", "trace")}, default=str)[:1500])
