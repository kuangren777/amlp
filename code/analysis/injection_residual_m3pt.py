"""Same per-row residual as review/injection_residual_a2.py on the M3 persist_taint rows, the only M3 hub rows that
store usage (summed per run). Suites from the AgentDojo harness. usage: python3 analysis/injection_residual_m3pt.py"""
import collections, gzip, json, os, statistics, sys
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(D)   # release root (code/ and data/ are siblings)
sys.path.insert(0, D)
import m3_common as C  # noqa: E402
import tiktoken  # noqa: E402
enc = tiktoken.get_encoding("o200k_base")
tk = lambda s: len(enc.encode(s or ""))
SYS = tk(C.harness.SYSTEM_MESSAGE)
out = {}
for m in ("gpt-4o-mini-2024-07-18", "gpt-4.1-mini-2025-04-14"):
    for tag, sp in (("persist_taint", f"{ROOT}/data/m3/steps/online_{m}__persist_taint.jsonl.gz"),
                    ("persist_taint_t0", f"{ROOT}/data/m3/online_{m}__persist_taint_t0.steps.jsonl.gz")):
        rows = [json.loads(l) for l in open(f"{ROOT}/data/m3/online_{m}__{tag}.jsonl")]
        st = {s["row_id"]: s["steps"] for s in map(json.loads, gzip.open(sp, "rt"))} if os.path.exists(sp) else {}
        res = []
        for r in rows:
            if not r.get("usage") or not r["usage"].get("calls"):
                continue
            base = SYS + tk(C.harness.suite(r["suite"]).user_tasks[r["ut"]].PROMPT)
            turns = collections.OrderedDict()
            for x in st.get(r["row_id"], []):
                turns.setdefault(x["t"], []).append(tk(x["function"]) + tk(json.dumps(x["args"])) + tk(str(x.get("output"))) + tk(str(x.get("error") or "")))
            hist, h = [0], 0
            for v in turns.values():
                h += sum(v); hist.append(h)
            n = r["usage"]["calls"]
            exp = sum(base + hist[min(k, len(hist) - 1)] for k in range(n))
            res.append({"row_id": r["row_id"], "blocked": bool(r.get("pt_flags")), "suite": r["suite"], "calls": n, "per_call": (r["usage"]["prompt_tokens"] - exp) / n})
        fl = 0
        for s in {x["suite"] for x in res}:
            v = [x for x in res if x["suite"] == s]; med = statistics.median(x["per_call"] for x in v)
            for x in v: x["excess"] = x["per_call"] - med
            fl += sum(x["excess"] > 200 for x in v)
        ex = sorted(x["excess"] for x in res)
        out[f"{tag}_{m}"] = {"rows": len(res), "calls": sum(x["calls"] for x in res), "rows_flagged": fl, "steps": bool(st),
                             "excess_p50": ex[len(ex) // 2], "excess_p99": ex[int(len(ex) * .99)], "excess_max": ex[-1]}
        print(tag, m, {k: (round(v, 1) if isinstance(v, float) else v) for k, v in out[f"{tag}_{m}"].items()})
        out[f"{tag}_{m}"]["flagged_blocked"] = sum(x["excess"] > 200 and x["blocked"] for x in res); out[f"{tag}_{m}"]["blocked_rows"] = sum(x["blocked"] for x in res); out[f"{tag}_{m}"]["flagged_calls_mean"] = statistics.mean([x["calls"] for x in res if x["excess"] > 200] or [0]); out[f"{tag}_{m}"]["calls_mean"] = statistics.mean(x["calls"] for x in res)
json.dump(out, open(f"{ROOT}/data/analysis_out/injection/injection_residual_m3pt.json", "w"), indent=1)
