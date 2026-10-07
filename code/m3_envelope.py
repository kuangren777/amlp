"""M3 envelope builder (R2): mirrors envbuild.build (two layers; A1 typed env entities; LLM tool predictor ∪ tools of
the r nearest mined tasks by bge-m3 prompt cosine; value layer = mined values ∪ typed env ∪ prompt literals; e-mail
domain lift) but mines the NEW M3 benign runs (data/m3/mine_<model>.jsonl), never the old snapshot.

Pool = benign runs of the same security model (one deployment = one agent), first k reps per task. Every call must
pass exclude_tasks explicitly (no default); M3 callers pass m3_common.HOLDOUT."""
from __future__ import annotations

import collections
import os
import sys

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
import m3_common as C  # noqa: E402
import envbuild as EB  # noqa: E402  (read-only reuse: embeddings, prompts, typed env, tool predictions)
from amlp import SIDE, Envelope, control_values, lift  # noqa: E402

K_DEFAULT = 8


class Pool:
    """M3 mining pool for one model: (suite, ut) -> list of benign rows (err_kind None, rep < k)."""

    def __init__(self, model: str, smoke: bool = False, k: int = K_DEFAULT, rows=None):
        self.model, self.k = model, k
        rows = rows if rows is not None else C.read_jsonl(C.rows_path(smoke, "mine", model))
        self.ben = collections.defaultdict(list)
        for r in rows:
            if r.get("err") is None and r.get("err_kind") is None and r.get("it") is None and r["model"] == model \
                    and r["rep"] < k:
                self.ben[(r["suite"], r["ut"])].append(r)
        for v in self.ben.values():
            v.sort(key=lambda r: r["rep"])

    def tasks(self) -> frozenset:
        return frozenset(self.ben)


LEVELS = ("exact", "email", "class", "any")          # plan.md §12: nested value levels, exact ⊂ email ⊂ class ⊂ any


def lift_class(v: str) -> str:
    """class level (plan.md §12): e-mail -> '@domain'; URL -> host; other values unchanged. Values arrive normalised by
    amlp.norm (lower case, scheme and leading www. stripped)."""
    if "@" in v:
        return "@" + v.split("@")[-1]
    if "." in v.split("/")[0] and " " not in v:
        return "host:" + v.split("/")[0]
    return v


class LevelEnvelope(Envelope):
    """amlp.Envelope with a selectable value level. 'exact' / 'email' behave exactly as amlp.Envelope with
    lift_emails False / True; 'class' also lifts URLs to their host; 'any' turns the value layer off."""
    level: str = "exact"

    def violations(self, fn: str, args: dict) -> list[str]:
        if self.level != "class":
            return super().violations(fn, args)
        if fn not in SIDE[self.suite]:
            return []
        if fn not in self.tools:
            return ["<tool>"]
        bad, p = [], self.prompt.lower()
        for a, v in control_values(fn, args):
            if v in p or v in self.env_text or v in self.values.get((fn, a), ()):
                continue
            if lift_class(v) in self.class_vals.get((fn, a), ()):
                continue
            bad.append(a)
        return bad


def build(pool: Pool, suite: str, ut: str, *, exclude_tasks: frozenset, predictor: str | None = "gpt-4.1-mini-2025-04-14",
          r: int = 1, level: str | None = None, lift_emails: bool | None = None, use_env="typed") -> Envelope:
    """Leave-task-out envelope for (suite, ut). exclude_tasks is keyword-only and required (R1).
    level ∈ LEVELS (plan.md §12); the legacy lift_emails flag maps to 'email' / 'exact' when level is not given."""
    if level is None:
        level = "email" if (lift_emails is None or lift_emails) else "exact"
    assert level in LEVELS, level
    t = (suite, ut)
    ex = frozenset(exclude_tasks)
    mined = {k: v for k, v in pool.ben.items() if k[0] == suite and k != t and k not in ex}
    assert not (set(mined) & ex) and t not in mined, "envelope mined an excluded task"
    tools = set(EB.PRED[predictor][f"{suite}|{ut}"]) if predictor else set()
    if r:
        nb = sorted(mined, key=lambda k: (-float(EB.E[k] @ EB.E[t]), k))[:r]
        tools |= {c["f"] for k in nb for o in mined[k] for c in o["trace"] if c["f"] in SIDE[suite]}
    vals, lifted = collections.defaultdict(set), collections.defaultdict(set)
    for k, rs in mined.items():
        for o in rs:
            for c in o["trace"]:
                if c["f"] in SIDE[suite]:
                    for a, v in control_values(c["f"], c["a"]):
                        vals[(c["f"], a)].add(v)
                        lifted[(c["f"], a)].add(lift(v))
    la = {k for k in lifted if any("@" in v for v in vals[k])} if level in ("email", "class", "any") else set()
    env = EB.ENV_TYPED[suite] if use_env == "typed" else (EB.ENV[suite] if use_env else "")
    e = LevelEnvelope(suite, tools, dict(vals), dict(lifted), la, env, EB.PROMPTS[t])
    e.level = level
    e.check_values = level != "any"
    cv = collections.defaultdict(set)
    for k, vs in vals.items():
        cv[k] |= {lift_class(v) for v in vs}
    e.class_vals = dict(cv)
    e.mined_tasks = frozenset(mined)          # provenance for the leak assert
    return e


def first_block(env: Envelope, trace: list[dict]) -> int | None:
    """Index of the first call the envelope blocks (monitor-mode judging), else None."""
    for i, c in enumerate(trace):
        if env.violations(c["f"], dict(c["a"])):
            return i
    return None
