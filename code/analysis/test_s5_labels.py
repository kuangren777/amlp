"""python3 -m pytest analysis/test_s5_labels.py -q"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s5_labels import keep_labels  # noqa: E402


def test_keep_labels_maps_ambiguous_indices():
    labels = [{"id": c} for c in "abcde"]
    kept, amb = keep_labels(labels, {"b"}, {0, 2, 4})
    assert [x["id"] for x in kept] == ["a", "c", "d", "e"]
    assert {kept[j]["id"] for j in amb} == {"a", "c", "e"}


def test_keep_labels_without_exclusions_is_identity():
    labels = [{"id": c} for c in "abc"]
    kept, amb = keep_labels(labels, set(), {1})
    assert kept == labels and amb == {1}


def test_committed_exclusion_list_hits_the_sample():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "s5")
    ex = set(json.load(open(os.path.join(d, "excluded_labels_planA.json")))["label_ids"])
    ids = json.load(open(os.path.join(d, "label_sample_meta.json")))["sample_ids"]
    assert len(ex) == 10 and ex <= set(ids)
