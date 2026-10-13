"""U bound for CaMeL (plan.md §22.9, pre-registered 747bead79): CaMeL WITHOUT security policies on the benign holdout
(part c, 3 reps) for the two hub models of the main evaluation. Same port, P/Q model, temperature and harness as the
main camel arm (m3_online, CAMEL_TEMP 0.0); only secpol=False. Resume-safe by row id.
Usage: baselines/camel_env/bin/python analysis/run_camel_nosecpol.py [workers]"""
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(D)   # release root (code/ and data/ are siblings)
sys.path.insert(0, D)
sys.path.insert(0, f"{D}/baselines")
import m3_common as C          # noqa: E402
import camel_run as CR         # noqa: E402

MODELS = ("gpt-4o-mini-2024-07-18", "gpt-4.1-mini-2025-04-14")
REPS, TEMP = 3, 0.0
lock = threading.Lock()


def out_path(m):
    return f"{ROOT}/data/m3/online_{m}__camel_nosecpol.jsonl"


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    CR._install_harness_patch()
    jobs = []
    for m in MODELS:
        done = {r["row_id"] for r in C.read_jsonl(out_path(m)) if r.get("err") is None}
        for rep in range(REPS):
            for (s, ut) in C.sorted_tasks(C.HOLDOUT):
                rid = f"online|{m}|camel_nosecpol|{s}|{ut}|-|{rep}"
                if rid not in done:
                    jobs.append((m, s, ut, rep, rid))
    print(f"{len(jobs)} runs pending", flush=True)

    def go(j):
        m, s, ut, rep, rid = j
        r = CR.run_row(m, s, ut, None, secpol=False, temp=TEMP, tag="m3_camel_nosecpol")
        r.update(row_id=rid, rep=rep, part="c", camel_temp=TEMP, git_head=C.GIT_HEAD, ts_sgt=C.now_sgt())
        with lock, open(out_path(m), "a") as f:
            f.write(json.dumps(r, default=str) + "\n")
        return r.get("err") is None
    with ThreadPoolExecutor(workers) as ex:
        ok = sum(ex.map(go, jobs))
    print(f"done {ok}/{len(jobs)} ok", flush=True)


if __name__ == "__main__":
    main()
