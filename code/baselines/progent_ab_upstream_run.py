"""Drive the OFFICIAL progent AgentDojo fork on the A/B pairs. Run with baselines/progent_up_env/venv/bin/python.
usage: progent_ab_upstream_run.py SUITE SHARD NSHARDS
Mirrors `python -m agentdojo.scripts.benchmark -s SUITE --model gpt-4o-2024-08-06 [--attack important_instructions]`
(run.sh env: SECAGENT_UPDATE/IGNORE_UPDATE_ERROR/POLICY_MODEL/SUITE) restricted to the pair list, with two differences:
 (1) the CLI's attack path first runs every selected injection task as a benign user task (extra, un-budgeted runs)
     -> skipped here; the per-user-task reset_security_policy() and run_task_with_injection_tasks() are the CLI's own calls.
 (2) benchmark version v1.2 (CLI default v1.1.2) so task definitions equal our harness (get_suite("v1.2")).
Benign pass and attack pass are separate loops, like the two separate run.sh invocations. Env OPENAI_BASE_URL/OPENAI_API_KEY from caller."""
import json, os, sys
suite_name, shard, nsh = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
os.environ.update(SECAGENT_POLICY_MODEL="gpt-4o-2024-08-06", SECAGENT_UPDATE="True", SECAGENT_IGNORE_UPDATE_ERROR="True",
                  SECAGENT_SUITE=suite_name, COLUMNS="300")
D = os.path.dirname(os.path.abspath(__file__))
from pathlib import Path
from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.benchmark import run_task_with_injection_tasks, run_task_without_injection_tasks
from agentdojo.logging import OutputLogger
from agentdojo.task_suite.load_suites import get_suite
from secagent import reset_security_policy

logdir = Path(f"{D}/progent_ab_upstream_logs")
suite = get_suite("v1.2", suite_name)
pipe = AgentPipeline.from_config(PipelineConfig(llm="gpt-4o-2024-08-06", defense=None, system_message_name=None, system_message=None))
atk = load_attack("important_instructions", suite, pipe)
pairs = [p for p in json.load(open(f"{D}/progent_ab_pairs.json")) if p[0] == suite_name]
uts = sorted({p[1] for p in pairs}, key=lambda x: int(x.split("_")[-1]))[shard::nsh]
errf = open(f"{D}/progent_ab_upstream_run_errs.jsonl", "a")


def guard(ut, it, fn):
    try:
        return fn()
    except Exception as e:
        errf.write(json.dumps({"suite": suite_name, "ut": ut, "it": it, "err": repr(e)[:300]}) + "\n"); errf.flush()


with OutputLogger(str(logdir), live=None):
    for ut in uts:
        guard(ut, None, lambda: run_task_without_injection_tasks(suite, pipe, suite.get_user_task_by_id(ut), logdir, False))
    for ut in uts:
        reset_security_policy()
        for it in [p[2] for p in pairs if p[1] == ut and p[2]]:
            guard(ut, it, lambda: run_task_with_injection_tasks(suite, pipe, suite.get_user_task_by_id(ut), atk, logdir, False, [it]))
print("done", suite_name, shard)
