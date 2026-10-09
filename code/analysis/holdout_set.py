"""Single switch for the holdout task set every analysis script evaluates on.

HOLDOUT_SET=clean27 (default, primary): the 27 holdout tasks that no pilot run touched (holdout minus the pilot p3
test split, extra_checks.pilot_test_tasks). The choice follows from the pilot overlap alone, not from any result.
HOLDOUT_SET=all48: all 48 holdout tasks (secondary; the pre-2026-10-09 primary set).

Importing this module patches m3_common.read_jsonl so that every row whose (suite, ut) is a holdout task outside the
active set is dropped at load time. Mining, selection and calibration rows never carry a holdout task (the pools exclude
C.HOLDOUT), so they pass unchanged; envelopes are still built with exclude_tasks=C.HOLDOUT (all 48), so the policy
under evaluation is identical in both modes. Estimators, seeds, bootstrap settings and tests are unchanged.
Outputs go to data/analysis_out/<set>/ (OUT)."""
from __future__ import annotations

import json
import os
import sys

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(D)   # release root (code/ and data/ are siblings)
sys.path.insert(0, D)
import m3_common as C          # noqa: E402

SETS = ("clean27", "all48")
NAME = os.environ.get("HOLDOUT_SET", "clean27")
assert NAME in SETS, f"HOLDOUT_SET must be one of {SETS}, got {NAME!r}"
PRIMARY = NAME == "clean27"


def pilot_test_tasks() -> set:
    """Pilot p3 test split: every second task (by task index) of each suite over all folds."""
    h = json.load(open(f"{D}/holdout_split.json"))
    out = set()
    for s, uts in h["holdout"].items():
        order = sorted(uts + h["selection"][s] + h["calibration"][s], key=lambda u: int(u.split("_")[-1]))
        out |= {(s, u) for k, u in enumerate(order) if k % 2 == 1}
    return out


OVERLAP = frozenset(pilot_test_tasks() & C.HOLDOUT)
CLEAN = frozenset(C.HOLDOUT - OVERLAP)
ACTIVE = CLEAN if PRIMARY else frozenset(C.HOLDOUT)
DROP = frozenset(C.HOLDOUT - ACTIVE)
assert (len(C.HOLDOUT), len(OVERLAP), len(CLEAN)) == (48, 21, 27), (len(C.HOLDOUT), len(OVERLAP), len(CLEAN))
OUT = f"{ROOT}/data/analysis_out/{NAME}"
os.makedirs(OUT, exist_ok=True)


def keep(r: dict) -> bool:
    return not ("suite" in r and "ut" in r and (r["suite"], r["ut"]) in DROP)


if not getattr(C.read_jsonl, "_holdout_set", None):
    _orig = C.read_jsonl

    def read_jsonl(path: str) -> list[dict]:
        return [r for r in _orig(path) if keep(r)]
    read_jsonl._holdout_set = NAME
    read_jsonl.raw = _orig                         # unfiltered reader (run counts of the setup table)
    C.read_jsonl = read_jsonl
