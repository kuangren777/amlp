"""M3 mining runs (R2): NEW undefended benign runs of the non-holdout user tasks, k = 8 reps per task per model.
Rows data/m3/mine_<model>.jsonl, step logs data/m3/steps/mine_<model>.jsonl.gz. Resumable.
usage: python3 m3_mine.py --model M [--workers N] [--dry-run] [--smoke --limit N] [--k 8]"""
from __future__ import annotations

import argparse
import os
import sys

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
import m3_common as C  # noqa: E402

RUNNER = "mine"


def jobs(k: int = 8, tasks=None) -> list[dict]:
    tasks = C.NON_HOLDOUT if tasks is None else frozenset(tasks)
    C.assert_pool_clean(tasks, RUNNER)                       # R1: the mining job list never touches the holdout
    return [{"defense": None, "suite": s, "ut": ut, "it": None, "rep": rep}
            for rep in range(k) for (s, ut) in C.sorted_tasks(tasks)]


def run_fn(model):
    def f(j):
        return C.harness.run_pair(model, j["suite"], j["ut"], None, None, defense=None, tag="m3_mine")
    return f


def main(argv=None):
    ap = C.common_args(argparse.ArgumentParser())
    ap.add_argument("--k", type=int, default=8)
    a = ap.parse_args(argv)
    J = jobs(a.k)
    R = C.Runner(RUNNER, a.model, a.smoke)
    C.assert_pool_clean({(r["suite"], r["ut"]) for r in C.read_jsonl(R.rows)}, RUNNER)
    if a.dry_run:
        print(f"{RUNNER} {a.model}: jobs {len(J)} ({len(C.NON_HOLDOUT)} tasks × k={a.k}), pending {len(R.todo(J))}")
        return
    os.makedirs(C.data_dir(a.smoke), exist_ok=True)
    C.install_capture()
    w = C.effective_workers(a.model, a.workers)
    C.start_guard(a.model, w)
    todo = R.todo(J)[: a.limit] if a.limit else J
    R.run(todo, run_fn(a.model), w)


if __name__ == "__main__":
    main()
