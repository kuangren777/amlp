"""Tests for analysis/seen_novel_sens.py on synthetic traces. Run: python3 -m pytest analysis/test_seen_novel_sens.py -q"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seen_novel as S          # noqa: E402
import seen_novel_sens as SS    # noqa: E402


class FakeEmb:
    """Deterministic unit vectors per string."""
    def __init__(self):
        self.c = {}

    def fill(self, texts):
        for t in texts:
            if t not in self.c:
                v = np.random.default_rng(abs(hash(t)) % 2**32).normal(size=8)
                self.c[t] = v / np.linalg.norm(v)

    def vec(self, t):
        return self.c[t]


def tr(*calls):
    return [{"f": f, "a": a} for f, a in calls]


TRACES = [tr(("a", {"x": 1}), ("b", {"msg": "hi"}), ("c", {})) for _ in range(4)] + \
         [tr(("a", {"x": 2}), ("b", {"msg": "hello"})) for _ in range(3)]
PROBES = TRACES + [tr(("a", {"x": 1}), ("c", {})), tr(("a", {"x": 9}),), tr(("a", {"x": 1}), ("b", {"msg": "hey"}))]


def emb_for(cfg):
    e = FakeEmb()
    e.fill(SS.all_strings([{"trace": t} for t in TRACES + PROBES], cfg) | {"hi", "hello", "hey"})
    return e


def test_reference_cell_equals_seen_novel_pdfa():
    e = emb_for(SS.REF)
    ref, mine = S.PDFA(TRACES, e), SS.PDFA(TRACES, e, SS.REF)
    assert set(ref.edges) == set(mine.edges)
    assert [ref.first_block(t, e) for t in PROBES] == [mine.first_block(t, e) for t in PROBES]


def test_literal_end_prunes_final_states_and_exempt_keeps_them():
    e = emb_for(SS.REF)
    lit = SS.PDFA(TRACES, e, dict(SS.REF, X="literal"))
    ex = SS.PDFA(TRACES, e, dict(SS.REF, X="exempt"))
    # every trace-final state has no outgoing non-END observation, so the literal rule blocks the last call
    assert all(lit.first_block(t, e) is not None for t in TRACES)
    assert all(ex.first_block(t, e) is None for t in TRACES)


def test_string_typing_moves_control_args_to_embedding():
    ctrl = sorted(S.CTRL)[0]
    assert SS.kind(SS.REF, ctrl, "bob@x.com") == "cat"
    assert SS.kind(dict(SS.REF, T="string"), ctrl, "bob@x.com") == "str"
    assert SS.kind(dict(SS.REF, T="string"), ctrl, " ") == "cat"     # empty strings stay categorical
    assert SS.kind(dict(SS.REF, T="string"), "msg", "hi") == "str"


def test_grid_shape():
    assert len(SS.PRIMARY) == 8 and SS.REF in SS.PRIMARY
    assert dict(SS.REF, X="literal") in SS.SECONDARY and len(SS.SECONDARY) == 5


def test_verdict_rule():
    good = {"delta_fb": {"point": 0.40, "ci95": [0.25, 0.55]}}
    low = {"delta_fb": {"point": 0.40, "ci95": [0.15, 0.55]}}
    small = {"delta_fb": {"point": 0.26, "ci95": [0.21, 0.30]}}
    assert SS.verdict({"a": good}, "all48")["holds"]
    assert not SS.verdict({"a": good, "b": low}, "all48")["holds"]
    assert not SS.verdict({"a": small}, "all48")["holds"] and SS.verdict({"a": small}, "clean27")["holds"]
