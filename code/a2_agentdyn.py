"""A2 runner (plan.md §25, §25.8): undefended benign and attack runs on the 60 AgentDyn user tasks.

The shared harness is imported, never modified. agentdojo is imported from ROPE's vendored autodojo before the harness
loads, so every harness module resolves to autodojo; harness.suite, the data directory and the model client are
patched in this process only. Everything else is testbed/harness.run_pair and m3_common.Runner unchanged
(temperature 1.0, max_iters 12, JB_TEMPLATE in every injection vector, resumable rows, per-step logs, outcome-blind
reruns). Every model call records its API usage, priced by §25.7, and the run stops itself at the hard cap.
usage: python3 a2_agentdyn.py --model M --part benign|attack [--workers N] [--dry-run] [--smoke --limit N]"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import threading

AUTODOJO = os.environ.get("AUTODOJO_SRC", "")
sys.path.insert(0, AUTODOJO)
import agentdojo  # noqa: E402
import agentdojo.agent_pipeline  # noqa: E402,F401
import agentdojo.agent_pipeline.ground_truth_pipeline  # noqa: E402,F401
import agentdojo.agent_pipeline.llms.openai_llm  # noqa: E402,F401
import agentdojo.functions_runtime  # noqa: E402,F401
import agentdojo.task_suite.load_suites  # noqa: E402,F401
import agentdojo.task_suite.task_suite  # noqa: E402,F401

assert agentdojo.__file__.startswith(AUTODOJO), agentdojo.__file__

D = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(D)   # release root (code/ and data/ are siblings)
sys.path.insert(0, D)
import m3_common as C  # noqa: E402
harness = C.harness
assert harness.get_suite.__module__ == "agentdojo.task_suite.load_suites"
assert sys.modules["agentdojo.task_suite.load_suites"].__file__.startswith(AUTODOJO)

VERSION = "v1.2.2"
SUITES = ("shopping", "github", "dailylife")
BENIGN_REPS, ATTACK_REPS = 8, 3
SECURITY_MODELS = ("qwen3-8b-local", "gpt-4o-mini-2024-07-18", "gpt-4.1-mini-2025-04-14")
DATA, SMOKE = f"{ROOT}/data/a2", f"{ROOT}/data/a2_smoke"
# §25.7: USD per 1M input / output tokens (OpenAI list prices); local models cost nothing
PRICE = {"gpt-4o-mini-2024-07-18": (0.15, 0.60), "gpt-4.1-mini-2025-04-14": (0.40, 1.60)}
HARD_STOP, CAP = 110.0, 120.0

_SUITE_CACHE: dict = {}

# §25.3, fixed before any run: side-effecting tools and control arguments from signatures only
sys.path.insert(1, os.environ.get("TRIPWIRE_DIR", ""))
import tripwire_agentdyn as TA  # noqa: E402
FETCH_TOOLS = ("browse_webpage", "download_file_through_url")
CTRL_A2 = frozenset(a for args in TA.CONTROL_ARGS_AGENTDYN.values() for a in args) | {"url"}


def side_tools(s: str) -> frozenset:
    named = set(TA.CONTROL_ARGS_AGENTDYN) | set(TA.CONTENT_WRITES_AGENTDYN) | set(FETCH_TOOLS)
    return frozenset(t.name for t in suite(s).tools if t.name in named)


def suite(name: str):
    assert name in SUITES, name
    if name not in _SUITE_CACHE:
        _SUITE_CACHE[name] = harness.get_suite(VERSION, name)
    return _SUITE_CACHE[name]


harness.suite = suite
C.data_dir = lambda smoke, model=None: SMOKE if smoke else DATA  # m3_common.data_dir took a model argument on 10-19 (exp-ss E1)


def task_order(s: str) -> list[str]:
    return sorted(suite(s).user_tasks, key=lambda x: int(x.split("_")[-1]))


def tasks() -> list[tuple[str, str]]:
    return [(s, ut) for s in SUITES for ut in task_order(s)]


# ---------------------------------------------------------------- usage capture and cost ledger
_tl = threading.local()


def _usage() -> dict:
    if not hasattr(_tl, "u"):
        _tl.u = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    return _tl.u


class _UsageClient:
    """Forwards to the real client; adds the usage of every chat completion to the calling thread's counter."""

    def __init__(self, inner):
        self._inner = inner
        outer = self

        class _Comp:
            def create(self, **kw):
                out = outer._inner.chat.completions.create(**kw)
                u, us = _usage(), getattr(out, "usage", None)
                u["calls"] += 1
                if us is not None:
                    u["prompt_tokens"] += int(us.prompt_tokens or 0)
                    u["completion_tokens"] += int(us.completion_tokens or 0)
                return out

        class _Chat:
            completions = _Comp()
        self.chat = _Chat()

    def __getattr__(self, name):
        return getattr(self._inner, name)


_prev_client_for = harness.client_for
_WRAPPED: dict = {}


def client_for(model: str):
    if model not in _WRAPPED:
        _WRAPPED[model] = _UsageClient(_prev_client_for(model))
    return _WRAPPED[model]


harness.client_for = client_for


def cost_of(model: str, u: dict) -> float:
    pin, pout = PRICE.get(model, (0.0, 0.0))
    return (u.get("prompt_tokens", 0) * pin + u.get("completion_tokens", 0) * pout) / 1e6


def ledger(dirs=(DATA, SMOKE)) -> dict:
    """Logged spend per file over every A2 row file (real and smoke: both are paid)."""
    out = {}
    for d in dirs:
        for p in sorted(glob.glob(f"{d}/*.jsonl")):
            out[p] = sum(cost_of(r["model"], r.get("usage") or {}) for r in C.read_jsonl(p))
    return out


class BudgetRunner(C.Runner):
    """m3_common.Runner plus usage per row and the §25.7 hard stop (no new job starts once spend reaches $110)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.spent = sum(ledger().values())
        self.stopped = 0

    def one(self, j, run_fn):
        with self.lock:
            if self.spent >= HARD_STOP:
                self.stopped += 1
                return {"err": "budget_stop"}
        _tl.u = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        r = super().one(j, run_fn)
        return r

    def _write(self, r, steps):
        u = dict(_usage())
        r["usage"], r["cost_usd"] = u, round(cost_of(self.model, u), 6)
        self.spent += r["cost_usd"]
        super()._write(r, steps)


def jobs(model: str, part: str) -> list[dict]:
    if part == "benign":
        return [{"defense": None, "suite": s, "ut": ut, "it": None, "rep": rep}
                for rep in range(BENIGN_REPS) for (s, ut) in tasks()]
    assert model in SECURITY_MODELS, f"{model} is utility only (§25.2)"
    return [{"defense": None, "suite": s, "ut": ut, "it": it, "rep": rep}
            for rep in range(ATTACK_REPS) for (s, ut) in tasks() for it in C.rotation_its(s, ut)]


def run_fn(model: str, part: str):
    def f(j):
        if part == "benign":
            return harness.run_pair(model, j["suite"], j["ut"], None, None, defense=None, tag="a2_monitor")
        return harness.run_pair(model, j["suite"], j["ut"], j["it"], harness.default_injection(j["suite"], j["it"]),
                                defense=None, tag="a2_attack")
    return f


def main(argv=None):
    ap = C.common_args(argparse.ArgumentParser())
    ap.add_argument("--part", required=True, choices=("benign", "attack"))
    a = ap.parse_args(argv)
    runner = {"benign": "monitor", "attack": "attack"}[a.part]
    J = jobs(a.model, a.part)
    if a.smoke:                                     # §25.2 (iii): one run per suite, rows under data/a2_smoke/
        J = [next(j for j in J if j["suite"] == s) for s in SUITES]
    R = BudgetRunner(runner, a.model, a.smoke)
    if a.dry_run:
        print(f"a2 {runner} {a.model}: jobs {len(J)}, pending {len(R.todo(J))}, logged spend ${R.spent:.2f}")
        return
    if R.spent >= HARD_STOP:
        raise SystemExit(f"hard stop: logged spend ${R.spent:.2f} >= ${HARD_STOP}")
    os.makedirs(C.data_dir(a.smoke), exist_ok=True)
    C.install_capture()
    w = C.effective_workers(a.model, a.workers)
    C.start_guard(a.model, w)
    todo = R.todo(J)[: a.limit] if a.limit else J
    R.run(todo, run_fn(a.model, a.part), w)
    print(f"{C.now_sgt()} spend now ${sum(ledger().values()):.2f} (cap ${CAP}, hard stop ${HARD_STOP}); "
          f"jobs not started for budget {R.stopped}", flush=True)


if __name__ == "__main__":
    main()
