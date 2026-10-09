"""Review round-1 checks (replay only, frozen M3 data).
P0-2: RQ1 / RQ2 numbers restricted to the holdout tasks that the pilot p3 test split never contained.
P1-5: on the calibration fold at the class level with all mined tasks, the share of blocked benign mining runs whose
first refused call is refused for a value (not for its tool).
Writes data/analysis_out/extra.json; macros via analysis/make_numbers.py."""
import json
import os
import sys

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(D)   # release root (code/ and data/ are siblings)
sys.path.insert(0, D)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import m3_common as C          # noqa: E402
import m3_envelope as ME       # noqa: E402
import m3_select as MS         # noqa: E402
import holdout_set as HS       # noqa: E402

SECURITY = ["qwen3-8b-local", "gpt-4o-mini-2024-07-18", "gpt-4.1-mini-2025-04-14"]


def pilot_test_tasks():                      # same rule as holdout_set.pilot_test_tasks
    h = json.load(open(f"{D}/holdout_split.json"))
    out = set()
    for s, uts in h["holdout"].items():
        order = sorted(uts + h["selection"][s] + h["calibration"][s], key=lambda u: int(u.split("_")[-1]))
        out |= {(s, u) for k, u in enumerate(order) if k % 2 == 1}
    return out


def main():
    overlap = pilot_test_tasks() & C.HOLDOUT
    clean = C.HOLDOUT - overlap
    res = {"n_clean_tasks": len(clean), "n_overlap": len(overlap), "per_model": {}}
    tot = {"viol": 0, "inter": 0}
    val_blocks = tool_blocks = 0
    for m in C.MODELS:
        e = [x for x in C.read_jsonl(C.selection_log(False)) if x["model"] == m and x.get("eps") == MS.PRIMARY_EPS and "amendment" in x][-1]
        cfg, pool = e["chosen"], ME.Pool(m, False, k=e["k"])
        C.assert_pool_clean(pool.tasks(), "analysis/extra_checks")
        ben = [r for r in C.read_jsonl(C.rows_path(False, "monitor", m)) if r.get("err") is None and (r["suite"], r["ut"]) in clean]
        viol = [r for r in C.read_jsonl(C.rows_path(False, "online", m, "main"))
                if r.get("defense") in (None, "none") and r.get("part") == "b" and r.get("err") is None and r["security"]
                and (r["suite"], r["it"]) not in C.TEXT_ONLY and (r["suite"], r["ut"]) in clean]
        envs = {}
        def env(t, lv, r):
            k = (t, lv, r)
            if k not in envs:
                envs[k] = ME.build(pool, *t, exclude_tasks=C.HOLDOUT, predictor=cfg["predictor"], r=r, level=lv)
            return envs[k]
        fb_exact = sum(ME.first_block(env((b["suite"], b["ut"]), "exact", 1000), b["trace"]) is not None for b in ben)
        inter = sum(ME.first_block(env((v["suite"], v["ut"]), cfg["level"], cfg["r"]), v["trace"]) is not None for v in viol)
        res["per_model"][m] = {"n_benign": len(ben), "fb_exact_all": fb_exact / len(ben) if ben else None,
                               "n_viol": len(viol), "intercepted": inter}
        if m in SECURITY:
            tot["viol"] += len(viol); tot["inter"] += inter
        for t in C.sorted_tasks(C.CALIBRATION):                  # P1-5
            e_cls = env(t, "class", 1000)
            for o in pool.ben.get(t, []):
                for c in o["trace"]:
                    bad = e_cls.violations(c["f"], dict(c["a"]))
                    if bad:
                        if bad == ["<tool>"]:
                            tool_blocks += 1
                        else:
                            val_blocks += 1
                        break
    res["clean_pooled_recall"] = tot["inter"] / tot["viol"] if tot["viol"] else None
    res["clean_pooled_viol"] = tot["viol"]
    res["class_level_blocks"] = {"value": val_blocks, "tool": tool_blocks,
                                 "value_share": val_blocks / (val_blocks + tool_blocks) if val_blocks + tool_blocks else None}
    json.dump(res, open(f"{HS.OUT}/extra.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
