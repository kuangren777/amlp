"""M3 A3 monitor runs: undefended benign runs of the 48 holdout tasks, 8 reps per model (48 × 4 × 8 = 1,536), with
per-step logs (function, args, tool output, error) so AMLP and Progent can both judge every call (m3_judge.py).
These rows are also the `none` side of the C1 test (plan.md A4(a)). Resumable; errored runs are rerun outcome-blind.
usage: python3 m3_holdout_monitor.py --model M [--workers N] [--dry-run] [--smoke --limit N]"""
from __future__ import annotations

import argparse
import os
import sys

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
import m3_common as C  # noqa: E402

RUNNER, REPS = "monitor", 8


def jobs(reps: int = REPS) -> list[dict]:
    assert len(C.HOLDOUT) == 48
    return [{"defense": None, "suite": s, "ut": ut, "it": None, "rep": rep}
            for rep in range(reps) for (s, ut) in C.sorted_tasks(C.HOLDOUT)]


def run_fn(model):
    def f(j):
        return C.harness.run_pair(model, j["suite"], j["ut"], None, None, defense=None, tag="m3_monitor")
    return f


def main(argv=None):
    a = C.common_args(argparse.ArgumentParser()).parse_args(argv)
    # R1 start-up assert: whatever mining pool exists for this model holds no holdout task
    C.assert_pool_clean({(r["suite"], r["ut"]) for r in C.read_jsonl(C.rows_path(a.smoke, "mine", a.model))}, RUNNER)
    J = jobs()
    R = C.Runner(RUNNER, a.model, a.smoke)
    if a.dry_run:
        print(f"{RUNNER} {a.model}: jobs {len(J)} (48 holdout tasks × {REPS} reps), pending {len(R.todo(J))}")
        return
    os.makedirs(C.data_dir(a.smoke), exist_ok=True)
    C.install_capture()
    w = C.effective_workers(a.model, a.workers)
    C.start_guard(a.model, w)
    todo = R.todo(J)[: a.limit] if a.limit else J
    R.run(todo, run_fn(a.model), w)


if __name__ == "__main__":
    main()
