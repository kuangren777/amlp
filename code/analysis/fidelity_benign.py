"""Benign-side fidelity of Agent-Sentry and CaMeL (plan.md §22.3, pre-registered a703606e4 / §22.5 23767e221).
Agent-Sentry: benign utility of the five-fold replay (main|gbm|trace5|seed*|L123, the runs of the blocked-share check)
against the published 5-fold CV of the XGBoost layer on AgentDojo + AgentDyn (Util 69.1, ABR 91.2; arXiv 2603.22868 Tab. 6,
baselines/agentsentry_PAPER_NOTES.md), the setting the blocked-share replay reproduces. Erratum plan.md §22.7: the first
version compared against 96.4, the Agent-Sentry Bench headline, which is a different dataset.
CaMeL: benign utility of camel vs undefended on the fidelity rows (it None), o4-mini; the authors report 77% vs 84%
for their headline model (indicative only, different model). Writes data/analysis_out/fidelity_benign.json."""
import json
import os
import statistics as st

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(D)   # release root (code/ and data/ are siblings)
a = json.load(open(f"{ROOT}/data/agentsentry_fidelity_results.json"))
tr = [v for k, v in a["runs"].items() if k.startswith("main|gbm|trace5|") and k.endswith("|L123")]
sentry = {"util_ours": st.mean(v["util"] for v in tr), "util_published": 69.1, "blocked_ours": st.mean(v["abr"] for v in tr),
          "blocked_published": 91.2, "n_seeds": len(tr), "n_benign": tr[0]["n_benign"]}
sentry["gap_pp"] = sentry["util_published"] - sentry["util_ours"]
sentry["within_10pp"] = abs(sentry["gap_pp"]) <= 10
rows = {}
for l in open(f"{ROOT}/data/fidelity_camel.jsonl"):
    r = json.loads(l)
    if r.get("err") is None and r["it"] is None:
        rows[(r.get("defense") or "none", r["suite"], r["ut"])] = bool(r["utility"])
u = {d: [v for (dd, *_), v in rows.items() if dd == d] for d in ("none", "camel")}
den = {}
for l in open(f"{ROOT}/data/fidelity_camel.jsonl"):
    r = json.loads(l)
    if r.get("err") is None and r["it"] is None and r.get("defense") == "camel":
        den[(r["suite"], r["ut"])] = bool(r["utility"]) and not r.get("camel_denied")
nop = {}
for l in open(f"{ROOT}/data/fidelity_camel_nosecpol.jsonl"):
    r = json.loads(l)
    if r.get("err") is None and r["it"] is None:
        nop[(r["suite"], r["ut"])] = bool(r["utility"])
camel = {"util_none": 100 * sum(u["none"]) / len(u["none"]), "util_camel": 100 * sum(u["camel"]) / len(u["camel"]),
         "util_nopolicy": 100 * sum(nop.values()) / len(nop), "util_camel_L": 100 * sum(den.values()) / len(den), "n": len(u["camel"]), "model": "o4-mini-2025-04-16",
         "published_none": 79.4, "published_camel": 76.3,       # arXiv 2503.18813 Tab. 2, o4-mini (plan.md §22.8)
         "comparison": "same model (o4-mini); authors' utility is their no-policy run"}
camel["drop_pp"] = camel["util_none"] - camel["util_camel"]
camel["published_drop_pp"] = camel["published_none"] - camel["published_camel"]
json.dump({"plan": "§22.3", "agentsentry": sentry, "camel": camel}, open(f"{ROOT}/data/analysis_out/fidelity_benign.json", "w"), indent=1)
print(json.dumps({"agentsentry": sentry, "camel": camel}, indent=1))
