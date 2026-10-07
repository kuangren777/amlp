"""M3 online arms on the 48 holdout tasks (plan.md A4(a), A4(d); ROADMAP E1).
 (a) AMLP benign, 8 reps                                   48 × 8 per model (pairs with the A3 monitor runs, C1)
 (b) attack pairs = holdout × TripWire 2-injection rotation (96 pairs): none, amlp at 3 reps; block_all,
     spotlighting, sandwich, tool_filter, pi_detector, melon, tripwire, progent, camel, agentsentry at 1 rep
 (c) benign utility of the baseline arms, 1 rep
Interpreter groups (one rows/steps file each): main = python3; camel = baselines/camel_env/bin/python;
agentsentry = baselines/agentsentry_env/bin/python (XGBoost layer 1). CaMeL only on hub models; an arm that cannot
run on a model is written to online_skips.jsonl with the reason, never as a number. All injections =
harness.default_injection (JB_TEMPLATE), including CaMeL (attack="jb_template").
usage: python3 m3_online.py --model M --group main [--parts a,b,c] [--arms ...] [--workers N] [--dry-run] [--smoke]"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
import m3_common as C  # noqa: E402

RUNNER = "online"
GROUPS = {
    "main": ["none", "amlp", "block_all", "spotlighting", "sandwich", "tool_filter", "pi_detector", "melon",
             "tripwire", "progent"],
    "camel": ["camel"],
    "agentsentry": ["agentsentry"],
}
BASELINES = ["block_all", "spotlighting", "sandwich", "tool_filter", "pi_detector", "melon", "tripwire", "progent",
             "camel", "agentsentry"]
HARNESS_NAME = {"none": None, "spotlighting": "spotlighting", "sandwich": "repeat_user_prompt",
                "tool_filter": "tool_filter", "pi_detector": "pi_detector", "melon": "melon", "tripwire": "tripwire"}
REPS_AMLP_BENIGN, REPS_ATTACK_MAIN = 8, 3
PROGENT_POLICY_MODEL = "gpt-4o-2024-08-06"
CAMEL_TEMP = 0.0                       # CaMeL's official P-LLM temperature (camel_PORT_NOTES deviation 4)


def cannot_run(arm: str, model: str) -> str | None:
    if arm == "camel" and model not in C.HUB_MODELS:
        return "CaMeL runs only on hosted-gateway models (M3 brief); its P-LLM/Q-LLM routing targets the hub"
    return None


def jobs(model: str, arms=None, parts="abc", baseline_reps: int = 1) -> tuple[list[dict], list[dict]]:
    """baseline_reps: reps for baseline arms in parts b and c (v5: 3, aligned with none / AMLP); existing rep-0 rows are
    skipped on resume because the row id carries the rep."""
    arms = set(arms or [a for g in GROUPS.values() for a in g])
    J, skips = [], []
    pairs = C.attack_pairs(C.HOLDOUT)
    for a in sorted(arms):
        why = cannot_run(a, model)
        if why:
            skips.append({"model": model, "arm": a, "reason": why})
            continue
        if "a" in parts and a == "amlp":
            J += [{"defense": "amlp", "suite": s, "ut": ut, "it": None, "rep": r, "part": "a"}
                  for r in range(REPS_AMLP_BENIGN) for (s, ut) in C.sorted_tasks(C.HOLDOUT)]
        if "b" in parts:
            reps = REPS_ATTACK_MAIN if a in ("none", "amlp") else baseline_reps
            J += [{"defense": a, "suite": s, "ut": ut, "it": it, "rep": r, "part": "b"}
                  for r in range(reps) for (s, ut, it) in pairs]
        if "c" in parts and a in BASELINES:
            J += [{"defense": a, "suite": s, "ut": ut, "it": None, "rep": r, "part": "c"}
                  for r in range(baseline_reps) for (s, ut) in C.sorted_tasks(C.HOLDOUT)]
    return J, skips


# ---------------------------------------------------------------- arm builders (registered into C.ARM_BUILDERS)
_x = threading.local()


class Arms:
    def __init__(self, model: str, group: str, smoke: bool, arms: set):
        self.model, self.group, self.smoke, self.arms = model, group, smoke, arms
        self.extra_static: dict = {}
        self._env_cache, self._lk = {}, threading.Lock()
        if "amlp" in arms or "block_all" in arms:
            from agentdojo.agent_pipeline import AgentPipeline, InitQuery, SystemMessage, ToolsExecutionLoop  # noqa
            from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM  # noqa
            import m3_envelope as ME
            import m3_select as MS
            self.ME = ME
            if "amlp" in arms:
                self.sel = MS.load_selected(model, smoke)
                self.pool = ME.Pool(model, smoke, k=self.sel["k"])
                C.assert_pool_clean(self.pool.tasks(), "m3_online")             # R1
                self.cfg = dict(self.sel["chosen"])
            C.ARM_BUILDERS["amlp"] = self._amlp
            C.ARM_BUILDERS["block_all"] = self._amlp
        if "progent" in arms:
            import progent_port as P
            self.P = P
            self.port_commit = C.port_commit()                                    # R5: asserts >= 05d44ebb
            cache = json.load(open(P._cache_path(PROGENT_POLICY_MODEL)))
            for (s, ut) in C.HOLDOUT:                                             # read-only cache: no baselines/ write
                k = f"{s}|{ut}"
                assert k in cache and cache[k]["prompt"] == C.harness.suite(s).user_tasks[ut].PROMPT, k
            C.ARM_BUILDERS["progent"] = self._progent
        if "agentsentry" in arms:
            import agentsentry_port as AS
            import m3_envelope as ME
            self.AS = AS
            mined = [r for rs in ME.Pool(model, smoke).ben.values() for r in rs]
            _, adv = AS.load_snapshot()
            self.fit = AS.fit(mined, adv, exclude_tasks=C.HOLDOUT, kind="gbm")   # R1: explicit holdout exclusion
            C.assert_pool_clean(self.fit.train_tasks, "m3_online/agentsentry")
            C.ARM_BUILDERS["agentsentry"] = self._agentsentry
        if "camel" in arms:
            import camel_run as CR
            self.CR = CR
            CR._install_harness_patch()

    def _amlp(self, model, temperature, system_message):
        from agentdojo.agent_pipeline import AgentPipeline, InitQuery, SystemMessage, ToolsExecutionLoop
        from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM
        from amlp import AMLPExecutor, Envelope
        s, ut = _x.task
        if _x.arm == "block_all":
            env = Envelope(s, set(), {}, {}, set(), "", self.ME.EB.PROMPTS[(s, ut)])
        else:
            with self._lk:
                if (s, ut) not in self._env_cache:
                    self._env_cache[(s, ut)] = self.ME.build(self.pool, s, ut, exclude_tasks=C.HOLDOUT, **self.cfg)
                env = self._env_cache[(s, ut)]
            assert not (env.mined_tasks & C.HOLDOUT)
        ex = AMLPExecutor(envelope=env, mode="block")
        _x.ex = ex
        llm = OpenAILLM(C.harness.client_for(model), model, temperature=temperature)
        p = AgentPipeline([SystemMessage(system_message), InitQuery(), llm, ToolsExecutionLoop([ex, llm], max_iters=12)])
        p.name = model
        return p

    def _progent(self, model, temperature, system_message):
        self.P.TASK.v = _x.task
        p = self.P.make_pipeline_progent(model, temperature, system_message, "progent_llm",
                                         policy_model=PROGENT_POLICY_MODEL, update=True)
        _x.ex = p.progent
        return p

    def _agentsentry(self, model, temperature, system_message):
        p = self.AS.make_pipeline_agentsentry(model, temperature, system_message, fitted=self.fit,
                                              judge_model=self.AS.JUDGE_MODEL)
        _x.ex = p.agentsentry
        return p

    def run_fn(self, j):
        s, ut, it, arm = j["suite"], j["ut"], j["it"], j["defense"]
        _x.task, _x.arm, _x.ex = (s, ut), arm, None
        if arm == "camel":
            r = self.CR.run_row(self.model, s, ut, it, q_model=self.model, attack="jb_template", secpol=True,
                                temp=CAMEL_TEMP, tag="m3_online")
            r.update(effort=self.CR.REASONING_EFFORT if self.CR.is_oai_reasoning_model(self.model) else None,
                     secpol=True, camel_temp=CAMEL_TEMP)
            return r
        inj = C.harness.default_injection(s, it) if it else None
        hdef = arm if arm in ("amlp", "block_all", "progent", "agentsentry") else HARNESS_NAME[arm]
        r = C.harness.run_pair(self.model, s, ut, it, inj, defense=hdef, tag="m3_online")
        ex = _x.ex
        if arm in ("amlp", "block_all") and ex is not None:
            r["amlp_flags"], r["amlp_calls"] = ex.flags, ex.calls
            r["amlp_cfg"] = self.cfg if arm == "amlp" else None
            r["amlp_selection_ts"] = self.sel["timestamp"] if arm == "amlp" else None
        if arm == "progent" and ex is not None:
            r.update(progent_flags=ex.flags, progent_updates=ex.updates,
                     progent_llm_calls=getattr(ex.pp, "llm_calls", None), port_commit=self.port_commit,
                     policy_model=PROGENT_POLICY_MODEL, progent_update=True)
        if arm == "agentsentry" and ex is not None:
            r.update(as_flags=ex.flags, as_calls=ex.calls, judge_model=self.AS.JUDGE_MODEL,
                     layer1=getattr(self.fit, "layer1_backend", None), fit_n=self.fit.n_train,
                     reimplementation="Agent-Sentry re-implementation (arXiv 2603.22868)")
        return r


def check_interpreter(group: str) -> None:
    if group == "camel":
        import camel  # noqa: F401  (fails outside baselines/camel_env)
    if group == "agentsentry":
        import xgboost  # noqa: F401  (fails outside baselines/agentsentry_env)


def main(argv=None):
    ap = C.common_args(argparse.ArgumentParser())
    ap.add_argument("--group", required=True, choices=sorted(GROUPS))
    ap.add_argument("--parts", default="abc")
    ap.add_argument("--arms", default="", help="comma list, subset of the group's arms")
    ap.add_argument("--baseline-reps", type=int, default=1, help="v5 (plan.md §13): 3")
    a = ap.parse_args(argv)
    arms = [x for x in (a.arms.split(",") if a.arms else GROUPS[a.group])]
    assert set(arms) <= set(GROUPS[a.group]), f"arms {arms} not in group {a.group}"
    J, skips = jobs(a.model, arms, a.parts, a.baseline_reps)
    R = C.Runner(RUNNER, a.model, a.smoke, group=a.group)
    C.assert_pool_clean({(r["suite"], r["ut"]) for r in C.read_jsonl(C.rows_path(a.smoke, "mine", a.model))}, RUNNER)
    if a.dry_run:
        by = {}
        for j in J:
            by[(j["part"], j["defense"])] = by.get((j["part"], j["defense"]), 0) + 1
        print(f"{RUNNER} {a.model} group={a.group}: jobs {len(J)}, pending {len(R.todo(J))}; "
              + ", ".join(f"{p}:{d}={n}" for (p, d), n in sorted(by.items()))
              + ("; skipped " + "; ".join(f"{s['arm']} ({s['reason']})" for s in skips) if skips else ""))
        return
    check_interpreter(a.group)
    os.makedirs(C.data_dir(a.smoke), exist_ok=True)
    if skips:
        with open(f"{C.data_dir(a.smoke)}/online_skips.jsonl", "a") as f:
            for s in skips:
                f.write(json.dumps({**s, "ts_sgt": C.now_sgt(), "git_head": C.GIT_HEAD}) + "\n")
    arms_obj = Arms(a.model, a.group, a.smoke, set(arms))
    C.install_capture()                       # after camel's patch, so CaMeL pipelines are captured too
    w = C.effective_workers(a.model, a.workers)
    C.start_guard(a.model, w)
    todo = R.todo(J)
    if a.limit:
        todo = todo[: a.limit]
    R.extra = lambda j: {"part": j["part"]}
    R.run(todo, arms_obj.run_fn, w)


if __name__ == "__main__":
    main()
