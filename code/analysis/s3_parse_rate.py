"""S3 offline tool-call parse check for the open-weight models (project lead), with the definitions of EngageFuzz
AMENDMENT_s1a_toolparser.md: tool-call rate = share of runs with at least one parsed tool call, threshold >= 0.90;
unparsed = share of runs whose final answer still holds a tool-call JSON object (s1b_reparse.calls_in_text),
threshold <= 0.02. Reads the step logs only, no model call.

    python3 analysis/s3_parse_rate.py   -> analysis/out/s3_parse_rate.json
"""
import glob
import gzip
import json
import os
import sys

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(D)   # release root (code/ and data/ are siblings)
sys.path.append(os.path.join(D, "..", "paper-engagefuzz", "analysis"))  # appended: its make_numbers.py must not shadow ours
from s1b_reparse import calls_in_text  # noqa: E402

OPEN = ("qwen3-8b-local", "llama31-8b-local")
RATE_MIN, UNPARSED_MAX = 0.90, 0.02


def check(path):
    n = called = unparsed = 0
    for line in gzip.open(path, "rt"):
        r = json.loads(line)
        n += 1
        called += bool(r.get("steps"))
        unparsed += bool(calls_in_text(r.get("final") or ""))
    return {"runs": n, "with_call": called, "unparsed_text_call": unparsed,
            "call_rate": called / n if n else None, "unparsed_rate": unparsed / n if n else None}


def main():
    out = {"thresholds": {"call_rate_min": RATE_MIN, "unparsed_rate_max": UNPARSED_MAX}, "files": {}}
    for m in OPEN:
        for p in sorted(glob.glob(f"{ROOT}/data/m3/steps/*_{m}*.jsonl.gz") + glob.glob(f"{ROOT}/data/a2/steps/*_{m}*.jsonl.gz")):
            r = check(p)
            r["pass"] = r["runs"] > 0 and r["call_rate"] >= RATE_MIN and r["unparsed_rate"] <= UNPARSED_MAX
            out["files"][os.path.relpath(p, D)] = r
            print(f"{os.path.relpath(p, D):60s} runs {r['runs']:5d} call {r['call_rate']:.3f} unparsed {r['unparsed_rate']:.3f} {'PASS' if r['pass'] else 'FAIL'}")
    json.dump(out, open(f"{ROOT}/data/analysis_out/s3_parse_rate.json", "w"), indent=1)


if __name__ == "__main__":
    main()
