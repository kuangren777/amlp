"""pytest for camel_run (no network). Run: camel_env/bin/python -m pytest camel_test.py -q"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pytest
import camel_run as C

HUB = "https://gateway.example.invalid/v1"   # dummy URL, no request is sent


def test_row_from_log_maps_fake_camel_result():
    fake = {
        "suite_name": "banking", "pipeline_name": "gpt-4o-mini-2024-07-18+camel+secpol", "user_task_id": "user_task_1",
        "injection_task_id": "injection_task_0", "attack_type": "important_instructions", "injections": {"v": "x"},
        "messages": [
            {"role": "user", "content": [{"type": "text", "content": "q"}]},
            {"role": "assistant", "content": [{"type": "text", "content": "code"}],
             "tool_calls": [{"function": "get_balance", "args": {"a": 1}, "id": None}]},
            {"role": "assistant", "content": [{"type": "text", "content": "1050.0"}], "tool_calls": None},
        ],
        "error": None, "utility": True, "security": False, "duration": 7.123,
    }
    r = C.row_from_log(fake, model="gpt-4o-mini-2024-07-18")
    assert (r["model"], r["suite"], r["ut"], r["it"], r["defense"]) == (
        "gpt-4o-mini-2024-07-18", "banking", "user_task_1", "injection_task_0", "camel")
    assert (r["utility"], r["security"], r["err"], r["n_msgs"], r["secs"]) == (True, False, None, 3, 7.1)
    assert r["trace"] == [{"f": "get_balance", "a": {"a": 1}}] and r["final"] == "1050.0"
    assert r["attack"] == "important_instructions" and "served_root" in r
    fake["injection_task_id"] = None
    assert C.row_from_log(fake, "m")["security"] is None  # benign rows carry no security value


def test_hub_routing_p_and_q_llm(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    C.configure_hub(HUB, key="sk-fake")
    assert os.environ["OPENAI_BASE_URL"] == HUB
    cl = C.make_hub_client(HUB, key="sk-fake")
    assert str(cl.base_url).rstrip("/") == HUB
    # P-LLM: our pipeline's OpenAILLM uses the hub client for arbitrary ids
    p = C.make_camel_pipeline("gpt-4o-mini-2024-07-18", "gpt-4o-2024-08-06", "banking", client=cl)
    plm = p.camel
    assert str(plm.llm.client.base_url).rstrip("/") == HUB and plm.llm.model == "gpt-4o-mini-2024-07-18"
    assert plm.quarantined_llm_model == "openai:gpt-4o-2024-08-06" and p.name == "gpt-4o-mini-2024-07-18+camel+secpol"
    # Q-LLM: pydantic_ai resolves "openai:<id>" via OPENAI_BASE_URL (no request is sent)
    from pydantic_ai.models import infer_model
    qm = infer_model(plm.quarantined_llm_model)
    assert str(qm.client.base_url).rstrip("/") == HUB and qm.model_name == "gpt-4o-2024-08-06"


def test_register_models_arbitrary_id_and_attack_name():
    from agentdojo.models import MODEL_NAMES
    C.register_models(["gpt-4o-2024-08-06", "my-new-model-1"])
    assert MODEL_NAMES["gpt-4o-mini-2024-07-18+camel"] == "GPT-4"
    assert MODEL_NAMES["my-new-model-1+camel+secpol"] == "AI model"
    t = C.injection_text("important_instructions", "banking", "user_task_0", "injection_task_0", "gpt-4o-mini-2024-07-18")
    j = C.injection_text("jb_template", "banking", "user_task_0", "injection_task_0", "gpt-4o-mini-2024-07-18")
    assert "to you, GPT-4." in t and "to you, AI assistant." in j and t.replace("GPT-4", "AI assistant") == j


def test_pllm_denial_hook_installed():
    p = C.make_camel_pipeline("m", None, "banking", client=C.make_hub_client(HUB, key="k"))
    assert type(p.camel).__name__ == "_PLLM" and p.camel.denied is None


def test_format_spec_error_returns_to_interpreter():
    import ast
    import camel_run as C
    import camel.interpreter.interpreter as I
    C.patch_format_spec_py310()
    consts = I._eval_formatted_value.__code__.co_consts
    assert "format specifier" in consts and "Invalid format specifier" not in consts
    try:
        f"{'x':{' 1200'}}"
    except ValueError as e:
        assert "format specifier" in str(e)


def test_reasoning_model_routing_matches_upstream():
    import camel_run as C
    assert C.is_oai_reasoning_model("o4-mini-2025-04-16") and not C.is_oai_reasoning_model("gpt-4o-2024-08-06")
    import openai
    cl = openai.OpenAI(base_url="http://gateway.example.invalid/v1", api_key="x")
    o = C.make_camel_pipeline("o4-mini-2025-04-16", client=cl).camel.llm
    assert o.reasoning_effort == "high" and o.temperature is None
    g = C.make_camel_pipeline("gpt-4o-2024-08-06", client=cl).camel.llm
    assert g.reasoning_effort is None and g.temperature == 0.0
