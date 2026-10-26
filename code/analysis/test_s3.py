"""Tests for the S3 recompute helpers.  python3 -m pytest analysis/test_s3.py -q"""
import gzip
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)


def test_amlp_models_filter():
    env = dict(os.environ, AMLP_MODELS="qwen3-8b-local,gpt-4o-mini-2024-07-18")
    out = subprocess.run([sys.executable, "-c", "import m3_common as C; print(','.join(C.MODELS))"], cwd=ROOT, env=env,
                         capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
    assert out == "qwen3-8b-local,gpt-4o-mini-2024-07-18"


def test_parse_rate_counts(tmp_path):
    import s3_parse_rate as P
    p = tmp_path / "x.jsonl.gz"
    rows = [{"steps": [{"f": "a"}], "final": "done"},
            {"steps": [], "final": 'call {"name": "send_money", "parameters": {"amount": 1}}'},
            {"steps": [], "final": "no call"}]
    with gzip.open(p, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    r = P.check(str(p))
    assert (r["runs"], r["with_call"], r["unparsed_text_call"]) == (3, 1, 1)


def test_nmodels_is_four():
    import importlib.util
    spec = importlib.util.spec_from_file_location("amlp_make_numbers", os.path.join(HERE, "make_numbers.py"))
    MN = importlib.util.module_from_spec(spec)  # by path: paper-engagefuzz/analysis also has a make_numbers.py
    spec.loader.exec_module(MN)
    main = [x for x in MN.PAPER_MODELS if x not in MN.APPENDIX_ONLY]
    assert MN.EXPECT == tuple(main) and MN.n_models(main) == str(len(main))   # main results: the run models minus Llama
    assert "llama31-8b-local" not in MN.EXPECT
    import pytest
    with pytest.raises(AssertionError):
        MN.n_models(MN.PAPER_MODELS)                                      # the appendix model must not enter the main count
