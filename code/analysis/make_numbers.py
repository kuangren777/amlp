"""Generate outputs/numbers.tex: every number the prose uses, as a LaTeX macro computed from frozen data.
Sources: data/analysis_out/<set>/*.json for both holdout sets of analysis/holdout_set.py (rq12, rq34, tests = C3' tests and
C1 via analysis/holdout_tests.py, extra, clean_subset, per_suite, seen_novel*). Primary macros come from all48, the
pre-registered holdout (plan.md §21); every macro is also emitted from clean27, the 27 holdout tasks no pilot run
touched, with the suffix Untouched. Usage: python3 analysis/make_numbers.py"""
import collections
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # fmt1 under python3 -I
from fmt1 import F  # one rounding rule: ROUND_HALF_UP on exact values (project lead)

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(D)   # release root (code/ and data/ are siblings)
OUT = f"{ROOT}/outputs/numbers.tex"
ALIAS = {"qwen3-8b-local": "Qwen", "llama31-8b-local": "Llama", "gpt-4o-mini-2024-07-18": "FourOMini",
         "gpt-4.1-mini-2025-04-14": "FourOneMini"}
PAPER_MODELS = tuple(ALIAS)  # the four agent models that were run; m3_common.MODELS also lists the S&S E1 models
# project lead: Llama-3.1-8B fails the tool-call parse gate (analysis/s3_parse_rate.py) and leaves the main results;
# its numbers stay in the appendix, from the four-model outputs in analysis/out_all4.
APPENDIX_ONLY = ("llama31-8b-local",)
FAMILY = {"qwen3-8b-local": "Qwen", "llama31-8b-local": "Llama", "gpt-4o-mini-2024-07-18": "GPT", "gpt-4.1-mini-2025-04-14": "GPT"}
MAIN_MODELS = tuple(m for m in PAPER_MODELS if m not in APPENDIX_ONLY)
OUTROOT = f"{ROOT}/data/analysis_out"            # main results (MAIN_MODELS); OUTROOT_ALL4 for the appendix-only models
OUTROOT_ALL4 = f"{ROOT}/data/analysis_out_all4"
EXPECT = MAIN_MODELS


def write_table(name, lines):
    """Tables come from the main pass only; the four-model pass for the appendix macros must not overwrite them."""
    if EXPECT == MAIN_MODELS:
        open(f"{ROOT}/outputs/tables/{name}.tex", "w").write("\n".join(lines) + "\n")


def n_models(models):
    """NModels from the models present in the outputs; fails if they are not the expected set (project lead)."""
    assert tuple(sorted(models)) == tuple(sorted(EXPECT)), f"outputs hold {sorted(models)}, expected {sorted(EXPECT)}"
    return str(len(EXPECT))


def pct(x, nd=1):
    return F(100 * x, f".{nd}f")


def build(S):
    OD = f"{OUTROOT}/{S}"
    m = {}
    rq = json.load(open(f"{OD}/rq12.json"))
    # ---- RQ1: chain endpoints per model (calibration fold and holdout)
    lo_rec, hi_rec, hi_cal, lvl_cal = [], [], [], []
    for model, x in rq["rq1"].items():
        a = ALIAS[model]
        ch = {(c["level"], c["r"]): c for c in x["chain"]}
        exact_all = ch[("exact", 1000)]
        email_all = ch[("email", 1000)]
        off = ch[("any", 1000)]
        m[f"RqOneCalExact{a}"] = pct(exact_all["cal_mean_loss"])
        m[f"RqOneCalEmail{a}"] = pct(email_all["cal_mean_loss"])
        m[f"RqOneBoundEmail{a}"] = pct(email_all["cal_bound"])
        m[f"RqOneCalOff{a}"] = pct(off["cal_mean_loss"])
        m[f"RqOneHoFbExact{a}"] = pct(exact_all["holdout_fb"])
        if exact_all["holdout_recall"] is not None:
            m[f"RqOneRecExact{a}"] = pct(exact_all["holdout_recall"])
            m[f"RqOneRecOff{a}"] = pct(off["holdout_recall"])
            hi_rec.append(exact_all["holdout_recall"]); lo_rec.append(off["holdout_recall"])
        hi_cal.append(email_all["cal_mean_loss"])
        lvl_cal += [exact_all["cal_mean_loss"], email_all["cal_mean_loss"]]
    m["RqOneCalValueMin"] = pct(min(hi_cal)); m["RqOneCalValueMax"] = pct(max(hi_cal))
    m["RqOneCalLevelMax"] = pct(max(lvl_cal))      # largest calibration loss at the exact or email level, all mined tasks
    m["RqOneRecExactMin"] = pct(min(hi_rec)); m["RqOneRecExactMax"] = pct(max(hi_rec))
    m["RqOneRecOffMin"] = pct(min(lo_rec)); m["RqOneRecOffMax"] = pct(max(lo_rec))
    # ---- RQ2: interception and structural invisibility at the calibrated config
    pw = rq["rq2"]["pooled_per_attacker_tool"]["post_webpage"]      # attacks that post a web page (rq2 prose)
    m["RqTwoPostFlagged"], m["RqTwoPostN"] = str(pw["intercepted"]), str(pw["n"])
    n = sum(rq["rq2"][k]["n_viol"] for k in rq["rq2"] if k != "pooled_per_attacker_tool")
    i = sum(rq["rq2"][k]["intercepted"] for k in rq["rq2"] if k != "pooled_per_attacker_tool")
    v = sum(rq["rq2"][k]["invisible"] for k in rq["rq2"] if k != "pooled_per_attacker_tool")
    m["RqTwoNViol"] = str(n); m["RqTwoIntercepted"] = str(i); m["RqTwoInvisible"] = str(v)
    m["RqTwoInterceptPct"] = pct(i / n); m["RqTwoInvisiblePct"] = pct(v / n)
    for model in ("qwen3-8b-local", "gpt-4o-mini-2024-07-18", "gpt-4.1-mini-2025-04-14"):
        x = rq["rq2"][model]; a = ALIAS[model]
        m[f"RqTwoIntercept{a}"] = pct(x["intercepted"] / x["n_viol"])
        m[f"RqTwoNViol{a}"] = str(x["n_viol"])
    # ---- pre-registered tests (plan.md A3, A4, §11.3)
    t = json.load(open(f"{OD}/tests.json"))                     # analysis/holdout_tests.py
    assert t["holdout_set"] == S
    fb, rec = t["c3prime_fb"], t["recall_A2rev"]
    m["CThreeNPairs"] = f"{fb['n_pairs']:,}"
    m["CThreeFbProgent"] = pct(fb["fb_progent"]); m["CThreeFbAmlp"] = pct(fb["fb_amlp"])
    m["CThreeDLo"] = pct(fb["d_ci95"][0]); m["CThreeDHi"] = pct(fb["d_ci95"][1])
    m["CThreeRecAmlp"] = pct(rec["recall_amlp"]); m["CThreeRecProgent"] = pct(rec["recall_progent"])
    m["CThreeRecNViol"] = str(rec["n_violations"])
    m["CThreeRecDiff"] = pct(rec["recall_diff_amlp_minus_progent"])
    m["CThreeRecDiffLo"] = pct(rec["recall_diff_ci95"][0]); m["CThreeRecDiffHi"] = pct(rec["recall_diff_ci95"][1])
    m["CThreeRecMargin"] = "5"                      # plan.md §11.3(b): lower bound >= -0.05
    m["COneTarget"] = "80"                          # ROADMAP C1 premise: >= 80% of violations blocked
    m["CThreeP"] = r"p < 0.001" if fb["mcnemar_p"] < 0.001 else f"p = {F(fb['mcnemar_p'], '.3f')}"
    c1 = t["C1"]
    m["COneDrop"] = pct(c1["drop_pooled"]); m["COneUpper"] = pct(c1["upper95"])
    m["CTwoEps"] = "0.10"
    # ---- setup constants, derived from the frozen artefacts where possible
    import sys
    sys.path.insert(0, D)
    import m3_common as C, m3_select as MS, harness
    h = json.load(open(f"{D}/holdout_split.json"))
    suites = sorted(h["holdout"])
    m["NSuites"] = str(len(suites))
    m["NUserTasks"] = str(sum(len(harness.suite(x).user_tasks) for x in suites))
    m["NHoldout"] = str(len(C.HOLDOUT))
    over = 0
    for x, uts in h["holdout"].items():
        order = sorted(uts + h["selection"][x] + h["calibration"][x], key=lambda u: int(u.split("_")[-1]))
        test = {u for k, u in enumerate(order) if k % 2 == 1}          # pilot p3 test split (alternate tasks)
        over += len(set(uts) & test)
    m["NPilotOverlap"] = str(over)
    m["NModels"] = n_models(rq["rq1"]); m["NFamilies"] = str(len({FAMILY[x] for x in EXPECT}))   # Qwen, OpenAI GPT (+ Llama in the four-model run)
    m["NTemp"] = "1.0"
    m["NEps"] = f"{F(MS.PRIMARY_EPS, '.2f')}"; m["NEpsLow"] = f"{F(MS.EPS_GRID[0], '.2f')}"; m["NEpsHigh"] = f"{F(MS.EPS_GRID[-1], '.2f')}"
    m["NRepsBase"] = "3"; m["NRepsBenign"] = "8"
    m["NBoot"] = f"{C.BOOT_N:,}"; m["NPairs"] = m["CThreeNPairs"]; m["NDefenses"] = "10"
    # ---- abstract / conclusion aliases of the pre-registered numbers
    m["FBAMLP"] = m["CThreeFbAmlp"]; m["RecAMLP"] = m["CThreeRecAmlp"]; m["RecProgent"] = m["CThreeRecProgent"]
    # ---- review round 1 (analysis/extra_checks.py): pilot-free subset, value share of class-level blocks, n
    x = json.load(open(f"{OD}/extra.json"))
    m["NClean"] = str(x["n_clean_tasks"])
    m["ClassValueShare"] = pct(x["class_level_blocks"]["value_share"])
    # §15 Cs* macros removed (G1 triage 03bea17): superseded by the Untouched macros of the clean27 set
    m["NCal"] = str(len(C.CALIBRATION)); m["NSel"] = str(len(C.SELECTION))
    m["CorrPct"] = f"{F(100 / (len(C.CALIBRATION) + 1), '.1f')}"   # conformal correction 1/(n+1)
    # ---- plan.md §16 seen vs novel (analysis/seen_novel.py) and §17 learning curve (analysis/seen_novel_curve.py)
    sn = json.load(open(f"{OD}/seen_novel.json"))
    for meth, nm in (("M1a", "Val"), ("M1b", "Tool"), ("M2", "Seq")):
        m[f"Sn{nm}Novel"] = pct(sn["pooled"][f"{meth}_novel_fb"]); m[f"Sn{nm}Seen"] = pct(sn["pooled"][f"{meth}_seen_fb"])
        dd = sn["delta"][f"{meth}_fb"]
        m[f"Sn{nm}Delta"] = pct(dd["point"]); m[f"Sn{nm}DeltaLo"] = pct(dd["ci95"][0]); m[f"Sn{nm}DeltaHi"] = pct(dd["ci95"][1])
        m[f"Sn{nm}Flag"] = pct(sn["pooled"][f"{meth}_novel_flag"])
    for model, x in sn["per_model"].items():                         # cold read 31caece item 5: per-model hybrid gap
        m[f"SnValDelta{ALIAS[model]}"] = pct(x["M1a_novel_fb"] - x["M1a_seen_fb"])
    m["SnSeqNovelMin"] = pct(min(x["M2_novel_fb"] for x in sn["per_model"].values()))
    m["SnSeqNovelMax"] = pct(max(x["M2_novel_fb"] for x in sn["per_model"].values()))
    m["SnSeqPoolLooMin"] = pct(min(x["M2_pool_loo_fb"] for x in sn["per_model"].values()))
    m["SnSeqPoolLooMax"] = pct(max(x["M2_pool_loo_fb"] for x in sn["per_model"].values()))
    m["SnGapMargin"] = "5"
    m["PraetorBtfr"] = "2.0"                         # Praetor arXiv 2604.26274 §7.4.1 (owner-verified, POSITIONING §1b)
    m["PraetorTraces"] = "400"
    m["SentryInBench"] = "96.4"; m["SentryTransferLo"] = "64.4"; m["SentryTransferHi"] = "79.1"   # Agent-Sentry 2603.22868 l.623: XGBoost 64.4 (rules 79.1) vs 96.4 (owner-verified)
    m["RsCrcOverInt"] = "30.6"                       # role-stratified CRC arXiv 2607.24343 Tab. 2, l.398 (owner-verified)
    sp = json.load(open(f"{OD}/seen_novel_pure.json"))          # plan.md §18
    for v, nm in (("M1a-pure", "ValPure"), ("M1b-pure", "ToolPure"), ("M1a-req", "ValReq"), ("M1b-pred", "ToolPred")):
        m[f"Sp{nm}Novel"] = pct(sp["pooled"][f"{v}_novel_fb"]); m[f"Sp{nm}Seen"] = pct(sp["pooled"][f"{v}_seen_fb"])
        dd = sp["delta"][f"{v}_fb"]
        m[f"Sp{nm}Delta"] = pct(dd["point"]); m[f"Sp{nm}DeltaLo"] = pct(dd["ci95"][0]); m[f"Sp{nm}DeltaHi"] = pct(dd["ci95"][1])
        m[f"Sp{nm}Flag"] = pct(sp["pooled"][f"{v}_novel_flag"])
    ss = json.load(open(f"{OD}/seen_novel_split.json"))         # plan.md §19
    for part, nm in (("M2-seq", "SeqOnly"), ("M2-arg", "ArgOnly")):
        x = ss["all"][f"{part}_fb"]
        m[f"Ss{nm}Seen"], m[f"Ss{nm}Novel"] = pct(x["seen"]), pct(x["novel"])
        m[f"Ss{nm}Delta"], m[f"Ss{nm}DeltaLo"], m[f"Ss{nm}DeltaHi"] = pct(x["delta"]), pct(x["ci95"][0]), pct(x["ci95"][1])
    # §19 Cl* macros removed (G1 triage 03bea17): superseded by the Untouched macros of the clean27 set
    cpath = f"{OD}/seen_novel_curve.json"
    if os.path.exists(cpath):
        cv = json.load(open(cpath))
        NUMW = {1: "One", 2: "Two", 4: "Four", 6: "Six", 7: "Seven", 8: "Eight"}
        for meth, nm in (("M1a", "Val"), ("M2", "Seq")):
            for k in (1, 2, 4, 6, 7):
                m[f"Cv{nm}Seen{NUMW[k]}"] = pct(cv["fb"][f"{meth}_seen{k}"]["point"])
            for j in (2, 4, 6, 8):
                m[f"Cv{nm}Novel{NUMW[j]}"] = pct(cv["fb"][f"{meth}_novel{j}"]["point"])
            dd = cv[f"{meth}_novel8_minus_seen1"]
            m[f"Cv{nm}OneTrace"] = pct(dd["point"]); m[f"Cv{nm}OneTraceLo"] = pct(dd["ci95"][0]); m[f"Cv{nm}OneTraceHi"] = pct(dd["ci95"][1])
            m[f"Cv{nm}Doubling"] = pct(cv[f"{meth}_novel4_minus_novel8"])
    # ---- baseline fidelity (appendix), recomputed from the raw fidelity rows
    def lastok(path):
        R = {}
        for l in open(path):
            r = json.loads(l)
            if r.get("err") is None:
                R[(r.get("defense") or "none", r["suite"], r["ut"], r["it"])] = r
        return R
    ab_up = [json.loads(l) for l in open(f"{ROOT}/data/progent_ab_upstream.jsonl")]
    ab_po = list(lastok(f"{ROOT}/data/progent_ab_port.jsonl").values())
    def u(rows): b = [r for r in rows if r["it"] is None]; return sum(bool(r["utility"]) for r in b) / len(b)
    def asr(rows): a = [r for r in rows if r["it"]]; return sum(bool(r["security"]) for r in a) / len(a)
    m["FidProgentUpUtil"] = pct(u(ab_up)); m["FidProgentPortUtil"] = pct(u(ab_po))
    m["FidProgentUpAsr"] = pct(asr(ab_up)); m["FidProgentPortAsr"] = pct(asr(ab_po)); m["FidProgentPairs"] = str(len(ab_po))
    def tboot(rows, key, ben):                      # task-cluster bootstrap 95% interval of a rate (C.BOOT_N, C.BOOT_SEED)
        import random
        by = {}
        for r in rows:
            if (r["it"] is None) == ben:
                by.setdefault((r["suite"], r["ut"]), []).append(int(bool(r[key])))
        T, rng, bs = list(by), random.Random(C.BOOT_SEED), []
        for _ in range(C.BOOT_N):
            a = b = 0
            for _ in T:
                t = rng.choice(T); a += sum(by[t]); b += len(by[t])
            bs.append(a / b)
        bs.sort()
        return pct(bs[int(0.025 * len(bs))]), pct(bs[int(0.975 * len(bs)) - 1])
    # codex round 1 C14: paired difference port - upstream on the same (suite, ut, it), task-cluster bootstrap
    upk = {(r["suite"], r["ut"], r["it"]): r for r in ab_up if r.get("err") is None}
    for key, ben, tag in (("utility", True, "Util"), ("security", False, "Asr")):
        diff = {}
        for r in ab_po:
            k = (r["suite"], r["ut"], r["it"])
            if k in upk and (r["it"] is None) == ben:
                diff.setdefault(k[:2], []).append(int(bool(r[key])) - int(bool(upk[k][key])))
        d, lo, hi = C.cluster_bootstrap(diff)
        m[f"FidProgent{tag}Diff"], m[f"FidProgent{tag}DiffLo"], m[f"FidProgent{tag}DiffHi"] = pct(d), pct(lo), pct(hi)
    m["FidProgentUpUtilLo"], m["FidProgentUpUtilHi"] = tboot(ab_up, "utility", True)
    m["FidProgentUpAsrLo"], m["FidProgentUpAsrHi"] = tboot(ab_up, "security", False)
    cn = lastok(f"{ROOT}/data/fidelity_camel_nosecpol.jsonl"); cs = lastok(f"{ROOT}/data/fidelity_camel.jsonl")
    att = [r for k, r in cn.items() if k[0] == "camel_nosecpol" and r["it"]]
    m["FidCamelUua"] = pct(sum(bool(r["utility"]) for r in att) / len(att))
    m["FidCamelAsr"] = pct(asr([r for k, r in cs.items() if k[0] == "camel"]))
    asf = json.load(open(f"{ROOT}/data/agentsentry_fidelity_results.json"))
    import statistics as st
    tr = [v for k, v in asf["runs"].items() if k.startswith("main|gbm|trace5|") and k.endswith("|L123")]
    m["FidSentryBlocked"] = f"{F(st.mean(x['abr'] for x in tr), '.1f')}"          # abr is already in percent
    tf = lastok(f"{ROOT}/data/fidelity_toolfence.jsonl"); tn = lastok(f"{ROOT}/data/fidelity_toolfence_nocache.jsonl")
    none = [r for k, r in tf.items() if k[0] == "none"]
    m["FidTfNoneUtil"] = pct(u(none))
    m["FidTfCacheUtil"] = pct(u([r for k, r in tf.items() if k[0] == "toolfence"]))
    m["FidTfCacheAsr"] = pct(asr([r for k, r in tf.items() if k[0] == "toolfence"]))
    m["FidTfNoCacheUtil"] = pct(u([r for k, r in tn.items() if k[0] == "toolfence"]))
    m["FidTfNoCacheAsr"] = pct(asr([r for k, r in tn.items() if k[0] == "toolfence"]))
    # ---- RQ3 / RQ4 (analysis/rq34.py, rules plan.md §14)
    q = json.load(open(f"{OD}/rq34.json"))
    r3, r4 = q["rq3"], q["rq4"]
    SEC = ["qwen3-8b-local", "gpt-4o-mini-2024-07-18", "gpt-4.1-mini-2025-04-14"]
    def rng(arm, key, models=SEC, scale=100):
        v = [r3[mm][arm][key] for mm in models if arm in r3[mm] and r3[mm][arm].get(key) is not None]
        return f"{F(scale * min(v), '.1f')}", f"{F(scale * max(v), '.1f')}"
    for arm, tag in (("progent", "Progent"), ("camel", "Camel"), ("pi_detector", "Pi"), ("melon", "Melon"),
                     ("agentsentry", "Sentry"), ("amlp", "Amlp"), ("tripwire", "Tripwire"), ("sandwich", "Sandwich"),
                     ("tool_filter", "ToolFilter"), ("spotlighting", "Spot"), ("block_all", "BlockAll")):
        lo, hi = rng(arm, "block_rate"); m[f"RqThree{tag}BlockMin"], m[f"RqThree{tag}BlockMax"] = lo, hi
        lo, hi = rng(arm, "benign_cost", models=[mm for mm in r3 if arm in r3[mm]])
        m[f"RqThree{tag}CostMin"], m[f"RqThree{tag}CostMax"] = lo, hi
    ga = r3["gpt-4.1-mini-2025-04-14"]["amlp"]                      # AMLP online benign cost, gpt-4.1-mini (Result 3)
    mc = json.load(open(f"{OD}/amlp_cost_mcnemar.json"))["gpt-4.1-mini-2025-04-14"]   # cold read 95d6ba9 M3
    m["AmlpCostPFourOneMini"] = f"{F(mc['p'], '.2f')}"
    m["AmlpCostFourOneMini"] = pct(ga["benign_cost"])
    m["AmlpCostLoFourOneMini"], m["AmlpCostHiFourOneMini"] = pct(ga["benign_cost_ci"][0]), pct(ga["benign_cost_ci"][1])
    # largest upper CI end of Progent benign cost across models (Result 3), same source as tables/rq3_ci.tex
    m["RqThreeProgentCostHiMax"] = f"{F(100 * max(r3[mm]['progent']['benign_cost_ci'][1] for mm in r3 if r3[mm].get('progent', {}).get('benign_cost_ci')), '.1f')}"
    cu = [r3[mm]["camel"]["benign_utility"] for mm in r3 if "camel" in r3[mm]]
    m["CamelUtilLow"], m["CamelUtilHigh"] = pct(min(cu)), pct(max(cu))
    nu = [r3[mm]["camel"]["benign_utility_none"] for mm in r3 if "camel" in r3[mm]]
    m["NoneUtilLow"], m["NoneUtilHigh"] = pct(min(nu)), pct(max(nu))
    tw = {}
    for mm in SEC:
        for k, v in r3[mm]["amlp_vs_tripwire"].items():
            tw[k] = tw.get(k, 0) + v
    m["TwOnlyAmlp"] = str(tw.get("amlp0_tripwire1", 0)); m["TwOnlyTripwire"] = str(tw.get("amlp1_tripwire0", 0))
    m["TwNeither"] = str(tw.get("amlp1_tripwire1", 0)); m["TwPairs"] = str(sum(tw.values()))
    para = r4["paraphrase"]
    m["ParaShiftAmlpMax"] = f"{F(100 * max(abs(para[mm]['amlp']['shift']) for mm in para), '.1f')}"
    m["ParaShiftProgentQwen"] = f"{F(100 * para['qwen3-8b-local']['progent']['shift'], '.1f')}"
    m["ParaShiftProgentFourOneMini"] = f"{F(100 * para['gpt-4.1-mini-2025-04-14']['progent']['shift'], '.1f')}"
    tr = r4["transfer"]
    m["TransferFbMax"] = pct(max(v["fb"] for v in tr.values()))
    m["TransferRecMin"] = pct(min(v["recall"] for v in tr.values() if v.get("recall") is not None))   # review round 3: flag-rate counterpart
    m["TransferRecFourToFourOne"] = pct(tr["gpt-4o-mini-2024-07-18->gpt-4.1-mini-2025-04-14"]["recall"])
    m["TransferRecFourOneToFour"] = pct(tr["gpt-4.1-mini-2025-04-14->gpt-4o-mini-2024-07-18"]["recall"])
    if "llama31-8b-local->qwen3-8b-local" in tr:
        m["TransferRecLlamaToQwen"] = pct(tr["llama31-8b-local->qwen3-8b-local"]["recall"])
    po = r4["poisoning"]
    for mm in SEC:
        a = ALIAS[mm]
        m[f"PoisonRecZero{a}"] = pct(po[mm]["0.0"]["recall"])     # One/Five/Ten removed: unused, other source than the table (project lead)
    for mm in SEC:
        a = ALIAS[mm]
        for rho, tag in (("0.05", "Five"), ("0.1", "Ten")):
            m[f"PoisonMean{tag}{a}"] = pct(po[mm][rho]["recall_mean"])
            m[f"PoisonMin{tag}{a}"] = pct(po[mm][rho]["recall_min"]); m[f"PoisonMax{tag}{a}"] = pct(po[mm][rho]["recall_max"])
    m["PoisonSeeds"] = str(len(po["qwen3-8b-local"]["0.05"]["per_seed"]))
    for mm, a in (("qwen3-8b-local", "Qwen"), ("gpt-4.1-mini-2025-04-14", "FourOneMini")):
        x = para[mm]
        m[f"ParaShiftAmlp{a}"] = f"{F(100 * x['amlp']['shift_paired'], '.1f')}"
        m[f"ParaShiftAmlpLo{a}"] = f"{F(100 * x['amlp']['shift_ci'][0], '.1f')}"; m[f"ParaShiftAmlpHi{a}"] = f"{F(100 * x['amlp']['shift_ci'][1], '.1f')}"
        m[f"ParaShiftProgentLo{a}"] = f"{F(100 * x['progent']['shift_ci'][0], '.1f')}"; m[f"ParaShiftProgentHi{a}"] = f"{F(100 * x['progent']['shift_ci'][1], '.1f')}"
        m[f"ParaReplayFb{a}"] = pct(x["amlp_replay_fb_on_paraphrase"])
    m["PoisonFbMax"] = pct(max(v["fb"] for mm in po for v in po[mm].values()))
    m["PoisonRunsFive"] = str(po["qwen3-8b-local"]["0.05"]["poisoned_runs"])
    # ---- per-suite breakdown and stacking with TripWire (analysis/per_suite.py)
    ps = json.load(open(f"{OD}/per_suite.json"))
    for suite in ("banking", "slack", "travel", "workspace"):
        m[f"SuiteAmlp{suite.capitalize()}"] = pct(ps["block"]["amlp"][suite])
        m[f"SuiteProgent{suite.capitalize()}"] = pct(ps["block"]["progent"][suite])
    m["SuiteProgentMin"] = pct(min(ps["block"]["progent"].values())); m["SuiteProgentMax"] = pct(max(ps["block"]["progent"].values()))
    m["TwMissed"] = str(ps["tripwire_missed"]["tripwire_missed"])
    m["TwMissedAmlpStops"] = str(ps["tripwire_missed"]["amlp_stops_of_those"])
    m["RecAllAmlp"] = pct(t["recall_all35"]["recall_amlp"]); m["RecAllProgent"] = pct(t["recall_all35"]["recall_progent"])
    m["RecAllN"] = str(t["recall_all35"]["n_violations"])
    m["NTasksEval"] = str(t["n_tasks"])            # tasks in the evaluated holdout set (48 primary, 27 untouched)
    return m


def section22():
    """plan.md §22 (pre-registered a703606e4, claims §22.5 23767e221), all 48 holdout tasks only.
    Headline replay flag rates (RecAMLP, RecProgent) switch to flag-before-harm per §22.5; the pre-registered
    hypothesis tests (CThree*) and the RQ2 split keep the plain flag rate."""
    OD = f"{OUTROOT}"
    m = {}
    rs = json.load(open(f"{OD}/resplit.json"))
    m["RsN"] = str(rs["meta"]["n_splits"])
    kept_fb, kept_flag, drop_flag = [], [], []
    for model, x in rs["models"].items():
        s = x["summary"]["0.1"]
        m[f"RsKept{ALIAS[model]}"] = f"{F(100 * s['kept_share'], '.0f')}"
        m[f"RsKeptCount{ALIAS[model]}"] = str(s["kept"])
        if s["fb_mean_kept"] is not None:
            kept_fb.append(s["fb_mean_kept"])
        if s["flag_mean_kept"] is not None:
            kept_flag.append(s["flag_mean_kept"]); drop_flag.append(s["flag_mean_dropped"])
    m["RsFbKeptMin"], m["RsFbKeptMax"] = pct(min(kept_fb)), pct(max(kept_fb))
    m["RsFlagKeptMin"], m["RsFlagKeptMax"] = pct(min(kept_flag)), pct(max(kept_flag))
    m["RsFlagDroppedMin"], m["RsFlagDroppedMax"] = pct(min(drop_flag)), pct(max(drop_flag))
    fb_all, fl_all = [], []                                   # codex round 1 C6: all-split means
    for model, x in rs["models"].items():
        if model == "llama31-8b-local":
            continue
        fb_all.append(x["summary"]["0.1"]["fb_mean_all"])
        fl = [sp["0.1"]["flag"] for sp in x["splits"] if sp["0.1"]["flag"] is not None]
        fl_all.append(sum(fl) / len(fl))
    m["RsFbAllMin"], m["RsFbAllMax"] = pct(min(fb_all)), pct(max(fb_all))
    m["RsFlagAllMin"], m["RsFlagAllMax"] = pct(min(fl_all)), pct(max(fl_all))
    S = [x["summary"]["0.1"] for x in rs["models"].values() if x["summary"]["0.1"]["flag_bh_mean_kept"] is not None]
    m["RsFlagBhKeptMin"], m["RsFlagBhKeptMax"] = pct(min(x["flag_bh_mean_kept"] for x in S)), pct(max(x["flag_bh_mean_kept"] for x in S))
    m["RsFlagKeptLo"], m["RsFlagKeptHi"] = pct(min(x["flag_range_kept"][0] for x in S)), pct(max(x["flag_range_kept"][1] for x in S))
    m["RsFbKeptLo"], m["RsFbKeptHi"] = pct(min(x["fb_range_kept"][0] for x in S)), pct(max(x["fb_range_kept"][1] for x in S))
    fb = json.load(open(f"{OD}/flag_before_harm.json"))["replay"]
    m["RecAMLP"], m["RecProgent"] = pct(fb["calibrated"]["flag_before_harm"]), pct(fb["progent"]["flag_before_harm"])
    f27 = fb["clean27"]                                     # before-harm on the 27 untouched tasks (cold read fb025b2 E1)
    m["RecAMLPUntouched"], m["RecProgentUntouched"] = pct(f27["calibrated"]["flag_before_harm"]), pct(f27["progent"]["flag_before_harm"])
    m["FbhToolPureUntouched"] = pct(f27["mined_tools_only"]["flag_before_harm"])
    m["FbhToolPure"], m["FbhToolPred"] = pct(fb["mined_tools_only"]["flag_before_harm"]), pct(fb["predicted_tools_only"]["flag_before_harm"])
    m["FbhAmlpDrop"] = f"{F(fb['calibrated']['diff_pp'], '.1f')}"
    hm = json.load(open(f"{OD}/flag_before_harm.json"))["harm_index_modes"]   # harm-step fallbacks (project lead internal review)
    m["FbhModeValue"] = str(hm.get("value", 0))                 # plan.md §26.5: executed value match first
    m["FbhModeReplay"] = str(hm.get("replay", 0) + hm.get("replay_nomatch", 0))   # replay-oracle step
    m["FbhModeN"] = str(sum(hm.values()))
    fbj = json.load(open(f"{OD}/flag_before_harm.json"))
    m["RqTwoNViolAdojo"], m["RqTwoNViolAdojoUntouched"] = str(fbj["replay"]["n_agentdojo"]["all48"]), str(fbj["replay"]["n_agentdojo"]["clean27"])
    g = fbj["replay"]["goal_step_26_2"]                       # plan.md §26.2 first definition, superseded by §26.5
    m["FbhGoalAmlp"], m["FbhGoalLater"] = pct(g["calibrated_flag_before_harm"]), str(g["rows_goal_later"])
    m["FbhProgentN"], m["FbhProgentNUntouched"] = str(fb["progent"]["n"]), str(f27["progent"]["n"])
    ad = json.load(open(f"{OD}/all48/rq34_agentdojo.json"))["rq3"]          # AgentDojo oracle (appendix)
    ex48 = json.load(open(f"{OD}/all48/rq34.json"))["rq3"]
    sec_models = [mm for mm in ad if ad[mm].get("amlp", {}).get("block_rate") is not None]
    for arm, tag in (("amlp", "Amlp"), ("progent", "Progent"), ("block_all", "BlockAll")):
        v = [100 * ad[mm][arm]["block_rate"] for mm in sec_models]
        m[f"Ad{tag}Min"], m[f"Ad{tag}Max"] = f"{F(min(v), '.1f')}", f"{F(max(v), '.1f')}"
    m["ExecMaxShift"] = f"{F(max(abs(100 * (ex48[mm][a]['block_rate'] - ad[mm][a]['block_rate'])) for mm in sec_models for a in ('amlp', 'progent')), '.1f')}"
    other = [abs(100 * (ex48[mm][a]["block_rate"] - ad[mm][a]["block_rate"])) for mm in sec_models for a in ex48[mm]
             if a not in ("amlp", "progent") and isinstance(ex48[mm][a], dict) and a in ad[mm]
             and ex48[mm][a].get("block_rate") is not None and ad[mm][a].get("block_rate") is not None]
    m["ExecOtherMaxShift"] = F(max(other), ".1f")             # internal review: every other defense, same comparison
    vbs = json.load(open(f"{OD}/value_block_cause.json")).get("llama31-8b-local")   # appendix only: four-model pass
    for scope, suf in ((("all48", ""), ("clean27", "Untouched")) if vbs else ()):
        vb = vbs[scope]
        m["LlamaBenignUtil" + suf] = pct(vb["benign_utility"]); m["LlamaBlockedUtil" + suf] = pct(vb["utility_of_blocked"])
        m["LlamaValueBlocks" + suf] = str(vb["blocked_value"]); m["LlamaToolBlocks" + suf] = str(vb["blocked_tool"])
        m["LlamaRuns" + suf] = str(vb["runs"])
    r1 = lambda x: float(F(100 * x, '.1f'))                         # printed table values; differences use full precision (project lead)
    tags = {"amlp": "Amlp", "block_all": "BlockAll", "progent": "Progent", "tripwire": "Tripwire",
            "agentsentry": "Sentry", "pi_detector": "Pi", "melon": "Melon", "camel": "Camel"}
    below = {}
    for scope, suf in (("all48", ""), ("clean27", "Untouched")):      # plan.md §22.9 L/M/U
        rb = json.load(open(f"{OD}/{scope}/refusal_bounds.json"))["models"]
        below[scope] = sum(e["cost_M"] < e["cost_U"] for x in rb.values() for e in x.values()
                           if isinstance(e, dict) and e.get("cost_M") is not None and e.get("cost_U") is not None)
        for arm, tg in tags.items():
            es = [x[arm] for x in rb.values() if arm in x]
            # range over all three values per model (substitution U, measured M, worst case L): the substitution
            # estimate is no bound, so a measured cost can fall below it (project lead)
            m[f"Bnd{tg}Lo{suf}"] = f"{F(min(min(r1(e['cost_U']), r1(e['cost_M']), r1(e['cost_L'])) for e in es), '.1f')}"
            m[f"Bnd{tg}Hi{suf}"] = f"{F(max(max(r1(e['cost_U']), r1(e['cost_M']), r1(e['cost_L'])) for e in es), '.1f')}"
            m[f"Bnd{tg}Intervened{suf}"] = str(sum(e["n_intervened"] for e in es)); m[f"Bnd{tg}Runs{suf}"] = str(sum(e["n"] for e in es))
            m[f"Bnd{tg}GapMin{suf}"] = f"{F(min(100 * (e['cost_L'] - e['cost_M']) for e in es), '.1f')}"
            m[f"Bnd{tg}GapMax{suf}"] = f"{F(max(100 * (e['cost_L'] - e['cost_M']) for e in es), '.1f')}"
    m["BndBelowSubAllN"], m["BndBelowSubUntouchedN"] = str(below["all48"]), str(below["clean27"])   # measured cost below the substitution estimate
    m["BndBelowSubN"] = str(below["all48"] + below["clean27"])
    for scope, suf in (("all48", ""), ("clean27", "Untouched")):
        rb = json.load(open(f"{OD}/{scope}/refusal_bounds.json"))["models"]
        for mm, al in (("gpt-4o-mini-2024-07-18", "FourOMini"), ("gpt-4.1-mini-2025-04-14", "FourOneMini")):
            e = rb[mm]["camel"]
            m[f"CamelNoPolCost{al}{suf}"] = f"{F(r1(e['cost_U']), '.1f')}"; m[f"CamelPolAdds{al}{suf}"] = f"{F(100 * e['policy_add'], '.1f')}"   # §22.12, same field as its interval
            m[f"CamelCostM{al}{suf}"] = f"{F(r1(e['cost_M']), '.1f')}"
            m[f"CamelPolAddsLo{al}{suf}"], m[f"CamelPolAddsHi{al}{suf}"] = (f"{F(100 * v, '.1f')}" for v in e["policy_add_ci"])  # §22.12
    alt = json.load(open(f"{OD}/resplit_altchain.json"))["models"]          # §22.10 R2: coarsen values before widening tools
    m["AltMaxShift"] = f"{F(max(abs(100 * (alt[mm]['summary']['0.1'][k] - rs['models'][mm]['summary']['0.1'][k])) for mm in alt for k in ('fb_mean_kept', 'flag_mean_kept', 'flag_bh_mean_kept', 'fb_mean_all') if alt[mm]['summary']['0.1'][k] is not None), '.1f')}"
    br = {mm: v for mm, v in json.load(open(f"{OD}/resplit_breadth.json")).items() if v["narrow_r_le4"]["n"]}  # §22.10 C8
    for part, tg in (("narrow_r_le4", "Narrow"), ("wide_r_ge8", "Wide")):
        m[f"Br{tg}N"] = str(sum(br[mm][part]["n"] for mm in br))
        for k, kk in (("flag", "Flag"), ("fb", "Fb")):
            v = [br[mm][part][k] for mm in br]
            m[f"Br{tg}{kk}Min"], m[f"Br{tg}{kk}Max"] = pct(min(v)), pct(max(v))
    pa = json.load(open(f"{OD}/praetor_asb.json"))                          # §22.11 P: close replication on ASB
    m["PraetorAsbBtfr"], m["PraetorAsbTest"], m["PraetorAsbTrain"] = pct(pa["btfr"]), str(pa["n_test"]), str(pa["n_train"])
    m["PraetorAsbNonEmpty"], m["PraetorAsbTools"] = str(pa["n_test_nonempty"]), str(pa["n_tools"])
    m["PraetorAsbSeqs"] = str(pa["distinct_name_sequences"])
    tm = [x for x in json.load(open(f"{OD}/table_marks.json")) if x["table"] == "rq3_main"]   # review round 2 C2 (make_figs first)
    m["MarkMainN"] = str(len(tm)); m["MarkMainFlips"] = str(sum(x["flip"] for x in tm))
    m["MarkMainFlipMax"] = f"{F(max(max(abs(x['all48']), abs(x['clean27'])) for x in tm if x['flip']), '.1f')}"
    mf = json.load(open(f"{OD}/melon_fidelity.json"))                       # §22.11 M: upstream MELON vs our port
    m["FidMelonPairs"] = str(mf["utility_diff_upstream_minus_port"]["n_pairs"])
    m["FidMelonUtilDiff"] = f"{F(mf['utility_diff_upstream_minus_port']['point_pp'], '.1f')}"
    m["FidMelonUtilDiffLo"], m["FidMelonUtilDiffHi"] = (f"{F(x, '.1f')}" for x in mf["utility_diff_upstream_minus_port"]["ci95_pp"])
    m["FidMelonExcl"] = str(math.ceil(max(abs(x) for x in mf["utility_diff_upstream_minus_port"]["ci95_pp"])))
    m["FidMelonAtk"] = str(mf["per_pipeline"]["melon_upstream"]["n_attack"])
    m["FidMelonAtkSucc"] = str(round(mf["per_pipeline"]["melon_upstream"]["asr"] * mf["per_pipeline"]["melon_upstream"]["n_attack"]))
    m["FidMelonUtilUp"], m["FidMelonUtilPort"] = pct(mf["per_pipeline"]["melon_upstream"]["utility"]), pct(mf["per_pipeline"]["melon_port"]["utility"])
    fi = json.load(open(f"{OD}/fidelity_benign.json"))
    m["FidSentryUtil"], m["FidSentryUtilPub"] = f"{F(fi['agentsentry']['util_ours'], '.1f')}", f"{F(fi['agentsentry']['util_published'], '.1f')}"
    m["FidSentryBlockedPub"] = f"{F(fi['agentsentry']['blocked_published'], '.1f')}"
    m["FidCamelUtilNone"], m["FidCamelUtilCamel"] = f"{F(fi['camel']['util_none'], '.1f')}", f"{F(fi['camel']['util_camel'], '.1f')}"
    m["FidCamelUtilNoPol"] = f"{F(fi['camel']['util_nopolicy'], '.1f')}"
    m["FidCamelUtilL"] = f"{F(fi['camel']['util_camel_L'], '.1f')}"
    m["FidCamelPubCamel"], m["FidCamelPubNone"] = f"{F(fi['camel']['published_camel'], '.1f')}", f"{F(fi['camel']['published_none'], '.1f')}"
    m["FidCamelPubGap"] = f"{F(fi['camel']['published_camel'] - fi['camel']['util_nopolicy'], '.1f')}"
    return m


def section23():
    """plan.md §23 (A1, pre-registered e8fdb5951): the CRC rule against simpler calibration rules on the §22.1
    re-splits at eps 0.10. Writes macros and tables/rq1_rules.tex."""
    cr = json.load(open(f"{OUTROOT}/calib_rules.json"))
    RT = {"crc": "Crc", "empirical": "Emp", "fixed": "Fixed", "seen": "Seen"}
    RL = {"crc": "bound, ours", "empirical": "empirical", "fixed": "fixed", "seen": "seen-task"}
    ML = {"qwen3-8b-local": "Qwen3-8B", "llama31-8b-local": "Llama-3.1-8B", "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
          "gpt-4.1-mini-2025-04-14": "gpt-4.1-mini"}
    m, L = {}, ["% generated by analysis/make_numbers.py section23", "\\begin{tabular}{@{}llcccc@{}}", "\\toprule",
                "model & rule & kept $\\uparrow$ & over $\\varepsilon$ $\\downarrow$ & holdout FB $\\downarrow$ & flagged $\\uparrow$ \\\\",
                "\\midrule"]
    models = list(cr["models"])
    for i, model in enumerate(models):
        s = cr["models"][model]["summary"]["0.1"]
        for j, r in enumerate(("crc", "empirical", "fixed", "seen")):
            x = s[r]
            m[f"CrKept{RT[r]}{ALIAS[model]}"] = str(x["kept"])
            m[f"CrOver{RT[r]}{ALIAS[model]}"] = str(x["fb_over_eps"])
            m[f"CrFlag{RT[r]}{ALIAS[model]}"] = "--" if x["flag_mean"] is None else f"{F(100 * x['flag_mean'], '.1f')}"
            fl = "--" if x["flag_mean"] is None else f"{F(100 * x['flag_mean'], '.1f')}"
            name = ("\\textbf{" + RL[r] + "}") if r == "crc" else RL[r]
            L.append(f"{ML[model] if j == 0 else ''} & {name} & {x['kept']} & {x['fb_over_eps']} & {F(100 * x['fb_mean'], '.1f')} & {fl} \\\\")
        L.append("\\midrule" if i < len(models) - 1 else "\\bottomrule")
    L.append("\\end{tabular}")
    write_table("rq1_rules", L)
    used = {"CrKeptEmpQwen", "CrKeptEmpFourOMini", "CrKeptCrcQwen", "CrKeptCrcFourOMini", "CrOverEmpQwen",
            "CrOverEmpFourOMini", "CrOverCrcQwen", "CrOverCrcFourOMini", "CrFlagCrcQwen", "CrFlagEmpQwen",
            "CrOverCrcFourOneMini", "CrOverFixedLlama"}   # the rest is printed in tables/rq1_rules.tex only (G1)
    assert m["CrOverCrcFourOneMini"] == m["CrOverEmpFourOneMini"]   # the text states the tie
    return {k: v for k, v in m.items() if k in used}


def section24():
    """plan.md §24 (pre-registered a5d21eaba, d77cb0809): heuristic sensitivity of the automaton transfer gap.
    Writes macros and tables/rq1_sens.tex (8 primary cells, untouched value in small type)."""
    R = {st: json.load(open(f"{OUTROOT}/{st}/seen_novel_sens.json")) for st in ("all48", "clean27")}
    assert all(R[st]["reference_reproduces"] and R[st]["verdict"]["holds"] for st in R)
    EL = {"bge": "bge-m3", "minilm": "MiniLM-L6"}
    TL = {"exact": "exact set", "string": "string"}
    XL = {"end": "end observation", "exempt": "final exempt"}
    ut = lambda a, b: a if a == b else f"{a}\\untouched{{{b}}}"
    p1 = lambda x: F(100 * x, ".1f")
    L = ["% generated by analysis/make_numbers.py section24", "\\begin{tabular}{@{}lllcccc@{}}", "\\toprule",
         "encoder & control args & trace end & seen FB $\\downarrow$ & novel FB $\\downarrow$ & gap & 95\\% interval \\\\",
         "\\midrule"]
    names = list(R["all48"]["primary"])
    for nm in names:
        a, b = R["all48"]["primary"][nm], R["clean27"]["primary"][nm]
        kv = dict(x.split("=") for x in nm.split(","))
        ci = lambda r: f"[{p1(r['delta_fb']['ci95'][0])}, {p1(r['delta_fb']['ci95'][1])}]"
        L.append(f"{EL[kv['E']]} & {TL[kv['T']]} & {XL[kv['X']]} & {ut(p1(a['seen_fb']), p1(b['seen_fb']))} & "
                 f"{ut(p1(a['novel_fb']), p1(b['novel_fb']))} & {ut(p1(a['delta_fb']['point']), p1(b['delta_fb']['point']))} & "
                 f"{ut(ci(a), ci(b))} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write_table("rq1_sens", L)
    m = {}
    for st, suf in (("all48", ""), ("clean27", "Untouched")):
        g = [r["delta_fb"]["point"] for r in R[st]["primary"].values()]
        m[f"SensGapMin{suf}"], m[f"SensGapMax{suf}"] = p1(min(g)), p1(max(g))
        lit = R[st]["secondary"]["E=bge,T=exact,X=literal,w=3,theta=3,eps=0.05"]
        m[f"SensLitSeen{suf}"], m[f"SensLitNovel{suf}"] = p1(lit["seen_fb"]), p1(lit["novel_fb"])
    ref = "E=bge,T=exact,X=end,w=3,theta=3,eps=0.05"
    m["SensShiftMax"] = p1(max(abs(r["delta_fb"]["point"] - R[st]["primary"][ref]["delta_fb"]["point"])
                               for st in R for r in R[st]["primary"].values()))
    m["SensLoMin"] = p1(min(r["delta_fb"]["ci95"][0] for st in R for r in R[st]["primary"].values()))
    return m


def section25():
    """plan.md §25 (33a89946a, 640c3aa05, 937703674; run log §25.10): RQ1 transfer gaps on AgentDyn, one shot,
    analysis/out/a2/a2_agentdyn.json and a2_prep.json; predictor token check analysis/out/injection/injection_check_pred.json."""
    R = json.load(open(f"{OUTROOT}/a2/a2_agentdyn.json"))
    P = json.load(open(f"{OUTROOT}/a2/a2_prep.json"))
    T = json.load(open(f"{OUTROOT}/injection/injection_check_pred.json"))
    p1 = lambda x: F(100 * x, ".1f")
    m = {"AdynSuites": str(len(P["side"])), "AdynTasks": str(len(P["pred"])), "AdynBenign": str(R["n_benign"]),
         "AdynViol": str(R["n_violations"]), "AdynPredCalls": str(T["n"]), "AdynPredOffsetMax": str(T["diff_max"])}
    assert m["AdynTasks"] == "60" and T["n_diff_gt_200"] == 0
    m["AdynCritLo"] = "20"   # §25.5 (a): the automaton interval must start at 20 points or more
    assert R["delta_fb"]["M2"]["ci95"][0] >= 0.20 and R["delta_fb"]["M1a-pure"]["ci95"][0] >= 0.20
    for key, nm in (("M2", "Seq"), ("M1a-pure", "Val"), ("M1b-pure", "Tool")):
        d = R["delta_fb"][key]
        m[f"Adyn{nm}Seen"] = p1(R["rates"][f"{key}|seen|all"]["rate"])
        m[f"Adyn{nm}Novel"] = p1(R["rates"][f"{key}|novel|all"]["rate"])
        m[f"Adyn{nm}Delta"], m[f"Adyn{nm}DeltaLo"], m[f"Adyn{nm}DeltaHi"] = p1(d["point"]), p1(d["ci95"][0]), p1(d["ci95"][1])
    return m


def section26():
    """Gateway prefix check for the SI (project lead reporting rule): one row per model and role. Writes
    tables/si_injection.tex. Inputs: analysis/out/injection/*.json, data/m3, data/a2, baselines/fidelity_*."""
    J = lambda f: json.load(open(f"{OUTROOT}/injection/{f}.json"))
    pred, a2, pt = J("injection_check_pred"), J("injection_residual_a2")["files"], J("injection_residual_m3pt")
    pc = lambda k, n: F(100 * k / n, ".1f")
    rows = [("gpt-4.1-mini", "AgentDyn tool predictor", f"{pred['n']} calls", f"{pred['n_diff_gt_200']} ({pc(pred['n_diff_gt_200'], pred['n'])}\\%)", "measured per call")]
    for m, nm in (("gpt-4o-mini-2024-07-18", "gpt-4o-mini"), ("gpt-4.1-mini-2025-04-14", "gpt-4.1-mini")):
        n = sum(a2[f"{p}_{m}"]["rows"] for p in ("monitor", "attack")); k = sum(a2[f"{p}_{m}"]["rows_flagged"] for p in ("monitor", "attack"))
        rows.append((nm, "agent on AgentDyn", f"{n} runs", f"{k} ({pc(k, n)}\\%)", "measured per run"))
    for m, nm in (("gpt-4o-mini-2024-07-18", "gpt-4o-mini"), ("gpt-4.1-mini-2025-04-14", "gpt-4.1-mini")):
        n = sum(pt[f"{t}_{m}"]["rows"] for t in ("persist_taint", "persist_taint_t0")); k = sum(pt[f"{t}_{m}"]["rows_flagged"] for t in ("persist_taint", "persist_taint_t0"))
        rows.append((nm, "agent under the taint defense", f"{n} runs", f"at most {k} ({pc(k, n)}\\%)", "measured per run, upper bound"))
    other = 0
    for q in sorted(__import__("glob").glob(f"{ROOT}/data/m3/*.jsonl")) + [f"{ROOT}/data/praetor_asb_runs.jsonl"] + \
            sorted(__import__("glob").glob(f"{ROOT}/data/fidelity_toolfence*.jsonl")) + [f"{ROOT}/data/progent_ab_port.jsonl"]:
        b = os.path.basename(q)
        if "SUPERSEDED" in b or "round0" in b or b.startswith(("judge_", "paraphrase_pred")) or "persist_taint" in b or "local" in b:
            continue
        other += sum(1 for _ in open(q))
    rows.append(("gpt-4o-mini, gpt-4.1-mini, gpt-4o", "other agent, policy and fidelity runs", f"{other} runs", "cannot be attributed", "shared credential, overlapping windows"))
    ph = json.load(open(f"{ROOT}/data/m3/paraphrases_holdout.json"))
    gen = ph["report"]["tasks"] + ph["report"]["regenerations"]
    ncand = len(json.load(open(f"{ROOT}/data/pilot/snap/llm_envelope_gpt-5.5.json")))
    rows.append(("gpt-5.5", "paraphrase generator, predictor candidate", f"{gen} + {ncand} calls", "0 to 100\\% inferred", "depends on the channel"))
    nj = sum(1 for _ in open(f"{ROOT}/data/agentsentry_judge_cache.jsonl"))
    rows.append(("qwen3.5-35b-a3b", "Agent-Sentry judge", f"{nj} calls", "cannot be verified", "attributable, no usage stored"))
    L = ["% generated by analysis/make_numbers.py section26", "\\begin{tabular}{@{}lllll@{}}", "\\toprule",
         "model & role & calls or runs & with a prefix $\\downarrow$ & basis \\\\", "\\midrule"]
    L += [" & ".join(r) + " \\\\" for r in rows]
    L += ["\\bottomrule", "\\end{tabular}"]
    # tables/si_injection.tex is no longer in the paper (S8, project lead): the gateway check stays internal
    return {"InjOffsetLimit": "200", "InjGenCalls": str(gen)}


def section27():
    """S5 (internal review, agentdojo-invariance-1, 12eacf8a8): repaired attack oracle on all 48 holdout tasks (project lead,
    ). The main table keeps the execution oracle; accuracies are population-weighted, the sample one is stratified."""
    S5 = f"{ROOT}/data/analysis_out/s5/all48"
    L = json.load(open(f"{S5}/label_eval.json"))
    R = json.load(open(f"{S5}/rankings.json"))
    m = {}
    for o, tg in (("exec", "Exec"), ("agentdojo", "Adojo"), ("replay_any", "Rep")):
        x = L["oracles"]["all"][o]
        m[f"Ora{tg}AccW"] = pct(x["accuracy_population_weighted"])
        m[f"Ora{tg}AccS"], m[f"Ora{tg}AccSLo"], m[f"Ora{tg}AccSHi"] = (pct(v) for v in x["accuracy"])
    m["OraLabelN"] = str(L["n_sample"])
    m["OraRows"] = f"{sum(L['population_cells'].values()):,}".replace(",", "{,}")
    m["OraDisagree"] = str(len(json.load(open(f"{S5}/disagreements_exec_vs_repaired.json"))))
    shift = 0
    for mm in R["exec"]:
        rk = {o: {d: i for i, d in enumerate(sorted(R[o][mm], key=lambda d: -R[o][mm][d]["block_rate"]))} for o in ("exec", "replay_any")}
        shift = max(shift, abs(rk["exec"]["amlp"] - rk["replay_any"]["amlp"]))
    m["OraAmlpMaxRankShift"] = str(shift)
    # project lead 2b: interception of the calibrated allowlist under both oracles and the largest cell difference
    tag = {"qwen3-8b-local": "Qwen", "gpt-4o-mini-2024-07-18": "FourOMini", "gpt-4.1-mini-2025-04-14": "FourOneMini"}
    name = {"qwen3-8b-local": "Qwen3-8B", "gpt-4o-mini-2024-07-18": "gpt-4o-mini", "gpt-4.1-mini-2025-04-14": "gpt-4.1-mini"}
    for mm, t in tag.items():
        m[f"OraAmlpGain{t}"] = F(100 * (R["replay_any"][mm]["amlp"]["block_rate"] - R["exec"][mm]["amlp"]["block_rate"]), ".1f")
    cells = [(abs(R["replay_any"][mm][d]["block_rate"] - R["exec"][mm][d]["block_rate"]), mm, d) for mm in R["exec"] for d in R["exec"][mm]]
    gap, mm, d = max(cells)
    m["OraMaxCellDiff"], m["OraMaxCellModel"] = F(100 * gap, ".1f"), name[mm]
    m["OraMaxCellDefense"] = {"spotlighting": "spotlighting", "sandwich": "sandwich", "tool_filter": "tool filter", "melon": "MELON",
                              "pi_detector": "PI detector", "tripwire": "TripWire", "progent": "Progent", "agentsentry": "Agent-Sentry",
                              "amlp": "the calibrated allowlist"}.get(d, d)
    return m


def section28():
    """S4 (internal review, exp_s4/AMENDMENT.md e535038d1, decision 2): seen-task benign block rate of the automaton mined per
    persona (one AgentDojo suite) on about 400 traces, with 95% Wilson intervals (project lead)."""
    d = json.load(open(f"{ROOT}/data/analysis_out/s4/fidelity.json"))
    tag = {"qwen3-8b-local": "Qwen", "gpt-4o-mini-2024-07-18": "FourOMini"}
    m = {}
    def put(pre, x):
        m[pre], m[pre + "Lo"], m[pre + "Hi"] = pct(x["rate"]), pct(x["wilson95"][0]), pct(x["wilson95"][1])
    put("SfourPooled", d["pooled"])
    m["SfourPooledK"], m["SfourPooledN"] = str(d["pooled"]["blocked"]), str(d["pooled"]["n"])
    mining = []
    for mm, t in tag.items():
        x = d["models"][mm]
        put(f"Sfour{t}", x["pooled"])
        put(f"SfourPaper{t}", x["paper_corpus_pooled"])
        m[f"Sfour{t}K"], m[f"Sfour{t}N"] = str(x["pooled"]["blocked"]), str(x["pooled"]["n"])
        mining += [p["mining_traces"] for p in x["personas"].values()]
        for b, w in (("50", "Fifty"), ("all", "All")):
            k = sum(p["curve"][b]["blocked"] for p in x["personas"].values())
            n = sum(p["curve"][b]["n"] for p in x["personas"].values())
            m[f"SfourCurve{w}{t}"] = pct(k / n)
    m["SfourMiningMin"], m["SfourMiningMax"] = str(min(mining)), str(max(mining))
    m["SfourPersonas"] = str(len(next(iter(d["models"].values()))["personas"]))
    m["SfourPraetor"] = pct(d["praetor_reported"])
    return m


def section29():
    """review round 11 (project lead): where the benign cost of the calibrated allowlist on the untouched tasks of gpt-4.1-mini
    comes from. Paired holdout runs (suite, task, rep) of the allowlist (online part a) against the undefended monitor
    runs; a refusal is a step whose error starts with the AMLP refusal text."""
    import gzip
    import holdout_set as HS
    mm = "gpt-4.1-mini-2025-04-14"
    clean = set(HS.CLEAN)
    rows = [json.loads(l) for l in open(f"{ROOT}/data/m3/online_{mm}__main.jsonl")]
    mon = {(r["suite"], r["ut"], r["rep"]): r for r in map(json.loads, open(f"{ROOT}/data/m3/monitor_{mm}.jsonl"))}
    a = [r for r in rows if r["defense"] == "amlp" and r.get("part") == "a" and (r["suite"], r["ut"]) in clean
         and (r["suite"], r["ut"], r["rep"]) in mon]
    lost = [r for r in a if not r["utility"] and mon[(r["suite"], r["ut"], r["rep"])]["utility"]]
    gain = [r for r in a if r["utility"] and not mon[(r["suite"], r["ut"], r["rep"])]["utility"]]
    ids = {r["row_id"] for r in lost}
    refused = set()
    for line in gzip.open(f"{ROOT}/data/m3/steps/online_{mm}__main.jsonl.gz", "rt"):
        st = json.loads(line)
        if st["row_id"] in ids and any(str(x.get("error") or "").startswith("Not permitted for this task") for x in st["steps"]):
            refused.add(st["row_id"])
    per = collections.Counter((r["suite"], r["ut"]) for r in lost)
    return {"UdropLost": str(len(lost)), "UdropGain": str(len(gain)), "UdropTasks": str(len(per)),
            "UdropMaxPerTask": str(max(per.values()) if per else 0), "UdropRefused": str(len(refused))}


def section30():
    """S5-fresh (internal review, b98e1f3e4; internal review): oracle accuracy on a new stratified blind sample that excludes the 160
    original rows. Reads only analysis/s5_fresh/eval.json and the label files in review/s5_fresh/ (read only)."""
    e = json.load(open(f"{ROOT}/data/analysis_out/s5_fresh/eval.json"))
    m = {}
    for o, t in (("exec", "Exec"), ("replay_any", "Rep"), ("full", "Full")):
        x = e["all"][o]
        m[f"Fresh{t}Acc"], m[f"Fresh{t}AccLo"], m[f"Fresh{t}AccHi"] = (pct(v) for v in x["accuracy"])
        m[f"Fresh{t}AccW"] = pct(x["accuracy_population_weighted"])
    w = [e["all"][o]["accuracy_population_weighted"] for o in ("exec", "replay_any", "full")]
    m["FreshWSpread"] = pct(max(w) - min(w))
    m["FreshN"] = str(e["all"]["exec"]["n"])
    m["FreshPerStratum"] = str(next(iter(e["all"]["exec"]["per_stratum"].values()))["n"])
    m["FreshStrata"] = str(len(e["all"]["exec"]["per_stratum"]))
    m["FreshAmbiguous"] = str(e["n_ambiguous"])
    m["FreshRepAccUnamb"] = pct(e["unambiguous"]["replay_any"]["accuracy"][0])
    a = {r["item"]: r["success"] for r in map(json.loads, open(f"{ROOT}/data/s5_fresh/labels_a.jsonl"))}
    b = {r["item"]: r["success"] for r in map(json.loads, open(f"{ROOT}/data/s5_fresh/labels_b.jsonl"))}
    ks = sorted(set(a) & set(b))
    agree = sum(a[k] == b[k] for k in ks)
    po = agree / len(ks)
    pa, pb = sum(a[k] for k in ks) / len(ks), sum(b[k] for k in ks) / len(ks)
    pe = pa * pb + (1 - pa) * (1 - pb)
    m["FreshAgree"], m["FreshKappa"] = str(agree), F((po - pe) / (1 - pe), ".2f")
    m["FreshAdjudicated"] = str(sum(1 for _ in open(f"{ROOT}/data/s5_fresh/labels_c.jsonl")))
    # internal review (project lead): undefended Llama-3.1-8B attack runs on all 48 tasks under both oracles
    ll = [json.loads(l) for l in open(f"{ROOT}/data/analysis_out/s5/all48/scored_llama.jsonl")]
    nn = [r for r in ll if r["defense"] == "none"]
    m["LlamaNoneN"] = str(len(nn))
    m["LlamaNoneExecK"] = str(sum(r["exec"] for r in nn))
    m["LlamaNoneRepK"] = str(sum(r["replay_any"] for r in nn))
    return m


def harm_audit():
    """Blind audit of the harm step (plan.md §27, f52a3c14d; internal review). Reads only analysis/harm_audit/eval.json."""
    e = json.load(open(f"{ROOT}/data/analysis_out/harm_audit/eval.json"))["all"]
    m = {"HaN": str(e["n"]), "HaAgreeN": str(e["agree"]), "HaEarlier": str(e["direction"].get("earlier", 0))}
    m["HaAgree"], m["HaAgreeLo"], m["HaAgreeHi"] = (pct(v) for v in e["agreement"])
    m["HaAmlpDiff"], m["HaProgentDiff"] = F(abs(e["calibrated"]["diff_pp"]), ".1f"), F(abs(e["progent"]["diff_pp"]), ".1f")
    assert e["unresolved"] == 0 and e["direction"].get("later", 0) == 0 and e["direction"].get("none", 0) == 0
    return m


def parse_macros():
    """Tool-call parse check (analysis/s3_parse_rate.py, project lead): Llama-3.1-8B unparsed range over its stages and
    the Qwen3-8B stages with model-made malformed calls, in percent of runs."""
    pr = json.load(open(f"{ROOT}/data/analysis_out/s3_parse_rate.json"))["files"]
    ll = [x["unparsed_rate"] for f, x in pr.items() if "llama31-8b-local" in f]
    q = {k: pr[k]["unparsed_rate"] for k in pr if "qwen3-8b-local" in k}
    return {"LlamaUnparsedMin": pct(min(ll)), "LlamaUnparsedMax": pct(max(ll)),
            "QwenUnparsedPara": pct(q["data/m3/steps/paraphrase_qwen3-8b-local.jsonl.gz"]),
            "QwenUnparsedAdynAttack": pct(q["data/a2/steps/attack_qwen3-8b-local.jsonl.gz"]),
            "QwenUnparsedAdynMonitor": pct(q["data/a2/steps/monitor_qwen3-8b-local.jsonl.gz"]),
            "QwenUnparsedMainMax": pct(max(v for k, v in q.items() if "/m3/" in k and "paraphrase" not in k))}


def macros():
    prim, sec = build("all48"), build("clean27")                  # plan.md §21: primary = pre-registered 48 tasks
    assert prim["NTasksEval"] == "48" and sec["NTasksEval"] == "27" and prim["NPilotOverlap"] == "21"
    assert prim["NClean"] == "27"
    m = dict(prim)
    m.update({k + "Untouched": v for k, v in sec.items()})
    m.update(section22())
    m.update(section23())
    m.update(section24())
    m.update(section25())
    m.update(section26())
    gaps = [m[k] for k in ("SnSeqDelta", "SpValPureDelta", "AdynSeqDelta", "AdynValDelta")]   # project lead: both benchmarks
    m["GapPartsMin"], m["GapPartsMax"] = min(gaps, key=float), max(gaps, key=float)
    tools = [m[k] for k in ("SpToolPureDelta", "AdynToolDelta")]
    m["GapToolMin"], m["GapToolMax"] = min(tools, key=float), max(tools, key=float)
    m["ParaShiftAmlpMaxBoth"] = f"{F(max(float(prim['ParaShiftAmlpMax']), float(sec['ParaShiftAmlpMax'])), '.1f')}"   # cold read 49cd6f5 item 5
    return m


def main():
    global OUTROOT, EXPECT
    m = macros()                                               # main results without the appendix-only models
    OUTROOT, EXPECT = OUTROOT_ALL4, PAPER_MODELS
    m4 = macros()
    OUTROOT, EXPECT = f"{ROOT}/data/analysis_out", MAIN_MODELS
    tags = [ALIAS[x] for x in APPENDIX_ONLY]
    app = {k: v for k, v in m4.items() if k not in m and any(t in k for t in tags)}
    m.update(app)
    m.update(parse_macros())
    m.update(section27())
    m.update(section28())
    m.update(section29())
    m.update(section30())
    m.update(harm_audit())
    lines = ["% generated by analysis/make_numbers.py; do not edit by hand",
             "% primary macros: all 48 holdout tasks (pre-registered); suffix Untouched: the 27 holdout tasks no pilot run touched",
             f"% main results over {len(MAIN_MODELS)} models; {len(app)} macros of {', '.join(APPENDIX_ONLY)} (appendix only) from analysis/out_all4"]
    for k, val in m.items():
        assert re.fullmatch(r"[A-Za-z]+", k), k
        if re.fullmatch(r"-[0-9][0-9.{},]*", val):   # keep the sign with the number: TeX may break a line after "-"
            val = f"\\mbox{{{val}}}"
        lines.append(f"\\newcommand{{\\{k}}}{{{val}\\xspace}}")
    open(OUT, "w").write("\n".join(lines) + "\n")
    print(f"{len(m)} macros -> {OUT}")


if __name__ == "__main__":
    main()
