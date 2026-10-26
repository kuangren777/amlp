"""S5 add-on (review round 9 C6, project lead): per-task decomposition of the gap between the calibration loss on the
calibration fold and the false-block rate on the benign holdout runs, for every model and chain element of RQ1.

Calibration loss = mean over calibration tasks of (blocked / total) of the model's own mining runs (m3_select.task_losses,
same call as the CRC chain). Holdout false-block rate = blocked / total over the benign holdout monitor runs (rq12).
Both use ME.build(..., exclude_tasks=C.HOLDOUT) with the selected predictor. Writes analysis/out/<set>/s5_gap_by_task.json.
"""
from __future__ import annotations

import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rq12 as R12                # noqa: E402  (sets sys.path)
import m3_common as C             # noqa: E402
import m3_envelope as ME          # noqa: E402
import m3_select as MS            # noqa: E402


def main():
    res = {}
    for m in C.MODELS:
        try:
            e = R12.sel_entry(m, MS.PRIMARY_EPS)
        except IndexError:                     # model without an amendment-B selection entry (not in RQ1)
            print(m, "no selection entry, skipped", flush=True); continue
        pred = e["chosen"]["predictor"]
        pool = ME.Pool(m, False, k=e["k"])
        cal_tasks = [tuple(t.split("|")) for t in e["calibration_task_ids"]]
        row = next(x for x in e["selection_table"] if x["cfg"]["predictor"] == pred)
        cfg = {k: v for k, v in row["cfg"].items() if k not in ("level", "r")}
        ben = R12.holdout_benign(m)
        ho_by_task = collections.defaultdict(list)
        for b in ben:
            ho_by_task[(b["suite"], b["ut"])].append(b)
        out = []
        for cal in row["crc_table"]:
            lv, r = cal["level"], cal["r"]
            L = MS.task_losses(pool, cal_tasks, {**cfg, "level": lv}, r)
            cal_rows = [{"task": "/".join(t), "blocked": a, "runs": b, "loss": a / b} for t, (a, b) in L.items() if b]
            ho_rows = []
            for t, runs in sorted(ho_by_task.items()):
                env = ME.build(pool, *t, exclude_tasks=C.HOLDOUT, predictor=pred, r=r, level=lv)
                k = sum(ME.first_block(env, x["trace"]) is not None for x in runs)
                ho_rows.append({"task": "/".join(t), "blocked": k, "runs": len(runs), "loss": k / len(runs)})
            cal_mean = sum(x["loss"] for x in cal_rows) / len(cal_rows)
            ho_pooled = sum(x["blocked"] for x in ho_rows) / sum(x["runs"] for x in ho_rows)
            ho_mean = sum(x["loss"] for x in ho_rows) / len(ho_rows)
            cal_pooled = sum(x["blocked"] for x in cal_rows) / sum(x["runs"] for x in cal_rows)
            contrib = sorted(cal_rows, key=lambda x: -x["loss"])
            out.append({"level": lv, "r": r, "cal_mean_loss_logged": cal["mean_loss"], "cal_task_mean": cal_mean,
                        "cal_run_pooled": cal_pooled, "holdout_run_pooled": ho_pooled, "holdout_task_mean": ho_mean,
                        "n_cal_tasks": len(cal_rows), "n_cal_tasks_blocked": sum(x["blocked"] > 0 for x in cal_rows),
                        "n_holdout_tasks": len(ho_rows), "n_holdout_tasks_blocked": sum(x["blocked"] > 0 for x in ho_rows),
                        "cal_runs_per_task": sorted({x["runs"] for x in cal_rows}),
                        "holdout_runs_per_task": sorted({x["runs"] for x in ho_rows}),
                        "cal_top_tasks": contrib[:6], "holdout_blocked_tasks": [x for x in ho_rows if x["blocked"]],
                        "cal_loss_share_top3": (sum(x["loss"] for x in contrib[:3]) / sum(x["loss"] for x in contrib)) if cal_mean else None})
        res[m] = {"predictor": pred, "chain": out}
        print(m, " ".join(f"{c['level'][:2]}{c['r'] if c['r'] < 1000 else 'A'}:cal{c['cal_task_mean']:.3f}(log{c['cal_mean_loss_logged']:.3f})/ho{c['holdout_run_pooled']:.3f}"
                          f" blockedT {c['n_cal_tasks_blocked']}/{c['n_cal_tasks']} vs {c['n_holdout_tasks_blocked']}/{c['n_holdout_tasks']}" for c in out), flush=True)
    os.makedirs(R12.OUT, exist_ok=True)
    json.dump(res, open(os.path.join(R12.OUT, "s5_gap_by_task.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
