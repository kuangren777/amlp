"""Benign-side fidelity of Agent-Sentry and CaMeL (plan.md §22.3, pre-registered a703606e4 / §22.5 23767e221).
Agent-Sentry: benign utility of the five-fold replay (main|gbm|trace5|seed*|L123, the runs of the blocked-share check)
against the published Overall Util. of the XGBoost layer on Agent-Sentry Bench (96.4, arXiv 2603.22868 l.623).
CaMeL: benign utility of camel vs undefended on the fidelity rows (it None), o4-mini; the authors report 77% vs 84%
for their headline model (indicative only, different model). Writes data/analysis_out/fidelity_benign.json."""
import json
import os
import statistics as st

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(D)   # release root (code/ and data/ are siblings)
a = json.load(open(f"{ROOT}/data/agentsentry_fidelity_results.json"))
tr = [v for k, v in a["runs"].items() if k.startswith("main|gbm|trace5|") and k.endswith("|L123")]
sentry = {"util_ours": st.mean(v["util"] for v in tr), "util_published": 96.4, "n_seeds": len(tr), "n_benign": tr[0]["n_benign"]}
sentry["gap_pp"] = sentry["util_published"] - sentry["util_ours"]
sentry["within_10pp"] = abs(sentry["gap_pp"]) <= 10
rows = {}
for l in open(f"{ROOT}/data/fidelity_camel.jsonl"):
    r = json.loads(l)
    if r.get("err") is None and r["it"] is None:
        rows[(r.get("defense") or "none", r["suite"], r["ut"])] = bool(r["utility"])
u = {d: [v for (dd, *_), v in rows.items() if dd == d] for d in ("none", "camel")}
camel = {"util_none": 100 * sum(u["none"]) / len(u["none"]), "util_camel": 100 * sum(u["camel"]) / len(u["camel"]),
         "n": len(u["camel"]), "model": "o4-mini-2025-04-16", "published_none": 84.0, "published_camel": 77.0,
         "comparison": "indicative only (different model)"}
camel["drop_pp"] = camel["util_none"] - camel["util_camel"]
camel["published_drop_pp"] = camel["published_none"] - camel["published_camel"]
json.dump({"plan": "§22.3", "agentsentry": sentry, "camel": camel}, open(f"{ROOT}/data/analysis_out/fidelity_benign.json", "w"), indent=1)
print(json.dumps({"agentsentry": sentry, "camel": camel}, indent=1))
