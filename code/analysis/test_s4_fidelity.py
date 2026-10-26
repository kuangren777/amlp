"""python3 -m pytest analysis/test_s4_fidelity.py -q"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s4_fidelity as F  # noqa: E402


def test_split_is_stratified_and_deterministic():
    runs = [{"rep": i, "trace": []} for i in range(25)]
    a1, b1 = F.split("m", ("banking", "user_task_0"), runs)
    a2, b2 = F.split("m", ("banking", "user_task_0"), runs)
    assert [r["rep"] for r in b1] == [r["rep"] for r in b2] and len(b1) == 5 and len(a1) == 20
    assert not {r["rep"] for r in a1} & {r["rep"] for r in b1}


def test_wilson():
    lo, hi = F.wilson(2, 100)
    assert 0.005 < lo < 0.006 and 0.070 < hi < 0.071
    assert F.wilson(0, 0) is None
