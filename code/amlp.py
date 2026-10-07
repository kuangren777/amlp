"""AMLP: two-layer least-privilege envelope for unseen tasks, enforced around a black-box agent.

Tool layer  T(p): side-effecting tools an LLM predicts from the user prompt, union the tools used in benign runs
                  of the r nearest mined tasks (bge-m3 cosine on the prompt). Task-specific.
Value layer V(f,a): per side-effecting tool f and control argument a, the values seen in benign runs of mined
                  tasks, the entities of the clean deployment environment, and literal values of the user prompt;
                  e-mail values may be lifted to their domain (one lattice step). Deployment-specific.
A side-effecting call is allowed iff f in T(p) and every control value is in V(f,a); otherwise it is blocked and
an error is returned to the agent. No provenance tracking, no extra model call per step.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from agentdojo.agent_pipeline.tool_execution import ToolsExecutor, is_string_list, tool_result_to_str
from agentdojo.types import ChatToolResultMessage, text_content_block_from_string
from ast import literal_eval

SIDE = {
    "banking": {"send_money", "schedule_transaction", "update_scheduled_transaction", "update_password", "update_user_info"},
    "slack": {"add_user_to_channel", "send_direct_message", "send_channel_message", "invite_user_to_slack",
              "remove_user_from_slack", "post_webpage", "get_webpage"},   # get_webpage = exfiltration via URL
    "travel": {"create_calendar_event", "cancel_calendar_event", "reserve_hotel", "reserve_car_rental",
               "reserve_restaurant", "send_email"},
    "workspace": {"send_email", "delete_email", "create_calendar_event", "cancel_calendar_event",
                  "reschedule_calendar_event", "add_calendar_event_participants", "append_to_file",
                  "create_file", "delete_file", "share_file"},
}
CTRL = {"recipient", "recipients", "cc", "bcc", "user", "user_email", "channel", "url", "email", "participants",
        "hotel", "restaurant", "company", "password", "file_id", "event_id", "id"}
EMPTY = {"", "none", "null", "[]"}


def norm(v) -> str:
    return re.sub(r"^https?://(www\.)?", "", str(v).strip().lower()).rstrip("/")


def control_values(fn: str, args: dict) -> list[tuple[str, str]]:
    out = []
    for k, v in args.items():
        if k not in CTRL:
            continue
        if isinstance(v, str) and is_string_list(v):
            try:
                v = literal_eval(v)
            except Exception:  # noqa: BLE001
                pass
        for x in (v if isinstance(v, (list, tuple)) else [v]):
            if x is None or str(x).strip().lower() in EMPTY:
                continue
            out.append((k, norm(x)))
    return out


def lift(v: str) -> str:
    """One lattice step up: e-mail -> its domain. Other values stay exact."""
    return "@" + v.split("@")[-1] if "@" in v else v


@dataclass
class Envelope:
    suite: str
    tools: set[str]
    values: dict[tuple[str, str], set[str]]          # (fn, arg) -> allowed exact values
    lifted: dict[tuple[str, str], set[str]]          # (fn, arg) -> allowed lifted values (domain level)
    lift_args: set[tuple[str, str]]                  # args whose level is "lifted"
    env_text: str = ""                               # clean environment dump, lowercased
    prompt: str = ""
    check_values: bool = True                        # False = tool layer only (ablation / LLM-envelope arm)

    def violations(self, fn: str, args: dict) -> list[str]:
        if fn not in SIDE[self.suite]:
            return []
        if fn not in self.tools:
            return ["<tool>"]
        if not self.check_values:
            return []
        bad = []
        p = self.prompt.lower()
        for a, v in control_values(fn, args):
            if v in p or v in self.env_text or v in self.values.get((fn, a), ()):
                continue
            if (fn, a) in self.lift_args and lift(v) in self.lifted.get((fn, a), ()):
                continue
            bad.append(a)
        return bad


@dataclass
class AMLPExecutor(ToolsExecutor):
    envelope: Envelope | None = None
    mode: str = "block"                              # block | monitor
    flags: list = field(default_factory=list)
    calls: list = field(default_factory=list)

    def __post_init__(self):
        super().__init__(tool_result_to_str)

    def query(self, query, runtime, env, messages=(), extra_args={}):
        if not messages or messages[-1]["role"] != "assistant" or not messages[-1]["tool_calls"]:
            return query, runtime, env, messages, extra_args
        results = []
        for tc in messages[-1]["tool_calls"]:
            if tc.function not in (t.name for t in runtime.functions.values()):
                results.append(ChatToolResultMessage(role="tool", content=[text_content_block_from_string("")],
                                                     tool_call_id=tc.id, tool_call=tc,
                                                     error=f"Invalid tool {tc.function} provided."))
                continue
            for k, v in tc.args.items():
                if isinstance(v, str) and is_string_list(v):
                    tc.args[k] = literal_eval(v)
            bad = self.envelope.violations(tc.function, dict(tc.args))
            if tc.function in SIDE[self.envelope.suite]:
                self.calls.append({"f": tc.function, "bad": bad})
            if bad:
                self.flags.append({"f": tc.function, "bad": bad, "blocked": self.mode == "block"})
            if bad and self.mode == "block":
                what = "this action" if bad == ["<tool>"] else f"the value of {', '.join(bad)}"
                results.append(ChatToolResultMessage(
                    role="tool", content=[text_content_block_from_string("")], tool_call_id=tc.id, tool_call=tc,
                    error=(f"Not permitted for this task: {what} for {tc.function} is outside the permissions "
                           "granted to this task. The call was not executed. Continue the user's original task.")))
                continue
            res, err = runtime.run_function(env, tc.function, tc.args)
            results.append(ChatToolResultMessage(role="tool", content=[text_content_block_from_string(self.output_formatter(res))],
                                                 tool_call_id=tc.id, tool_call=tc, error=err))
        return query, runtime, env, [*messages, *results], extra_args
