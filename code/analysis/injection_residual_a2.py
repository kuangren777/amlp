"""Per-row hidden-prefix residual on the A2 hub agent rows (project lead). A2 rows store usage summed over the
calls of a run. Expected prompt tokens of call k = tiktoken(system + user request) + history before k (tool-call
names and JSON arguments, tool outputs from the step log). Per (model, suite) the unknown constant (tool schemas,
message framing) is the median of (billed - expected) / calls. A row is flagged when its mean per-call residual
exceeds that median by more than 200 tokens, which a prefix of more than 200 tokens on every call would cause.
usage: python3 analysis/injection_residual_a2.py"""
import collections, gzip, json, os, statistics, sys
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(D)   # release root (code/ and data/ are siblings)
sys.path.insert(0, D)
import a2_agentdyn as A  # noqa: E402
import tiktoken  # noqa: E402
enc = tiktoken.get_encoding("o200k_base")
tk = lambda s: len(enc.encode(s or ""))
SYS = tk(A.harness.SYSTEM_MESSAGE)
out = {"method": __doc__.split("\n")[1:6], "files": {}}
for m in ("gpt-4o-mini-2024-07-18", "gpt-4.1-mini-2025-04-14"):
    for part in ("monitor", "attack"):
        rows = [json.loads(l) for l in open(f"{A.DATA}/{part}_{m}.jsonl")]
        st = {s["row_id"]: s["steps"] for s in map(json.loads, gzip.open(f"{A.DATA}/steps/{part}_{m}.jsonl.gz", "rt"))}
        res = []
        for r in rows:
            base = SYS + tk(A.suite(r["suite"]).user_tasks[r["ut"]].PROMPT)
            turns = collections.OrderedDict()
            for x in st.get(r["row_id"], []):
                turns.setdefault(x["t"], []).append(tk(x["function"]) + tk(json.dumps(x["args"])) + tk(str(x.get("output"))) + tk(str(x.get("error") or "")))
            hist, h, exp = [0], 0, 0
            for t, v in turns.items():
                h += sum(v); hist.append(h)
            n = r["usage"]["calls"]
            exp = sum(base + hist[min(k, len(hist) - 1)] for k in range(n))
            res.append({"row_id": r["row_id"], "suite": r["suite"], "calls": n, "per_call": (r["usage"]["prompt_tokens"] - exp) / n})
        flagged = 0
        for s in A.SUITES:
            v = [x for x in res if x["suite"] == s]
            med = statistics.median(x["per_call"] for x in v)
            for x in v:
                x["excess"] = x["per_call"] - med
            flagged += sum(x["excess"] > 200 for x in v)
        ex = sorted(x["excess"] for x in res)
        out["files"][f"{part}_{m}"] = {"rows": len(res), "calls": sum(x["calls"] for x in res), "rows_flagged": flagged,
                                       "excess_p50": ex[len(ex) // 2], "excess_p99": ex[int(len(ex) * 0.99)], "excess_max": ex[-1],
                                       "excess_min": ex[0]}
        print(part, m, {k: (round(v, 1) if isinstance(v, float) else v) for k, v in out["files"][f"{part}_{m}"].items()})
json.dump(out, open(f"{ROOT}/data/analysis_out/injection/injection_residual_a2.json", "w"), indent=1)
