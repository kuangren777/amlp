"""S5: calibration robustness across splits and selection rules under the repaired attack oracle.

resplit.py and calib_rules.py define holdout violations as undefended attack rows whose recorded AgentDojo security()
is True. This wrapper reruns both unchanged except that a violation is an undefended attack row that succeeds under the
replay_any oracle of s5_rescore.py (AgentDojo's check after each replayed executed call). Outputs go to new files and
never overwrite the paper's resplit.json / calib_rules.json:
  analysis/out/s5_resplit_replayany.json, analysis/out/s5_calib_rules_replayany.json
Usage: python3 analysis/s5_calib_robust.py [n_splits]
"""
from __future__ import annotations

import json as _json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import resplit as R               # noqa: E402  (sets HOLDOUT_SET=all48 and the import paths)
import calib_rules as CR          # noqa: E402
import m3_common as C             # noqa: E402
import exec_oracle as EO          # noqa: E402
import s5_rescore as S5           # noqa: E402

_cache = {}


def violations_replay_any(model):
    if model not in _cache:
        st = EO.steps(model)
        rows = [r for r in C.read_jsonl(C.rows_path(False, "online", model, "main"))
                if r.get("defense") in (None, "none") and r.get("part") == "b" and r.get("err") is None
                and (r["suite"], r["it"]) not in C.TEXT_ONLY]
        out, n_rec = [], 0
        for r in rows:
            n_rec += bool(r["security"])
            s = st.get(r["row_id"])
            ok = S5.replay_security(r, s, any_point=True) if s is not None else int(bool(r["security"]))
            if ok:
                out.append(r)
        print(model, "violations recorded", n_rec, "replay_any", len(out), flush=True)
        _cache[model] = out
    return _cache[model]


class _Json:  # redirect calib_rules' fixed output path
    def __getattr__(self, k):
        return getattr(_json, k)

    @staticmethod
    def dump(obj, fp, **kw):
        path = fp.name.replace("calib_rules.json", "s5_calib_rules_replayany.json")
        fp.close()                                  # the original is restored from bytes in __main__
        with open(path, "w") as f:
            _json.dump(obj, f, **kw)


if __name__ == "__main__":
    import hashlib
    orig = os.path.join(R.D, "analysis", "out", "calib_rules.json")
    keep = open(orig, "rb").read() if os.path.exists(orig) else None
    C.MODELS = {m: C.MODELS[m] for m in R.SECURITY}   # same models as the paper's resplit.json / calib_rules.json
    R.holdout_violations = violations_replay_any
    R.OUTNAME = "s5_resplit_replayany.json"
    CR.json = _Json()
    try:
        R.main()
        CR.main()
    finally:
        if keep is not None:
            open(orig, "wb").write(keep)
            assert hashlib.sha256(open(orig, "rb").read()).digest() == hashlib.sha256(keep).digest()
            print("calib_rules.json restored byte-identical", flush=True)
