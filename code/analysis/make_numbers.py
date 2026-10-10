"""Generate outputs/numbers.tex: every number the prose uses, as a LaTeX macro computed from frozen data.
Sources: data/analysis_out/<set>/*.json for both holdout sets of analysis/holdout_set.py (rq12, rq34, tests = C3' tests and
C1 via analysis/holdout_tests.py, extra, clean_subset, per_suite, seen_novel*). Primary macros come from all48, the
pre-registered holdout (plan.md §21); every macro is also emitted from clean27, the 27 holdout tasks no pilot run
touched, with the suffix Untouched. Usage: python3 analysis/make_numbers.py"""
import json
import os
import re

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(D)   # release root (code/ and data/ are siblings)
OUT = f"{ROOT}/outputs/numbers.tex"
ALIAS = {"qwen3-8b-local": "Qwen", "llama31-8b-local": "Llama", "gpt-4o-mini-2024-07-18": "FourOMini",
         "gpt-4.1-mini-2025-04-14": "FourOneMini"}


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}"


def build(S):
    OD = f"{ROOT}/data/analysis_out/{S}"
    m = {}
    rq = json.load(open(f"{OD}/rq12.json"))
    # ---- RQ1: chain endpoints per model (calibration fold and holdout)
    lo_rec, hi_rec, hi_cal = [], [], []
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
    m["RqOneCalValueMin"] = pct(min(hi_cal)); m["RqOneCalValueMax"] = pct(max(hi_cal))
    m["RqOneRecExactMin"] = pct(min(hi_rec)); m["RqOneRecExactMax"] = pct(max(hi_rec))
    m["RqOneRecOffMin"] = pct(min(lo_rec)); m["RqOneRecOffMax"] = pct(max(lo_rec))
    # ---- RQ2: interception and structural invisibility at the calibrated config
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
    m["CThreeP"] = r"p < 0.001" if fb["mcnemar_p"] < 0.001 else f"p = {fb['mcnemar_p']:.3f}"
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
    m["NModels"] = str(len(C.MODELS)); m["NFamilies"] = "3"           # Qwen, Llama, OpenAI GPT
    m["NTemp"] = "1.0"
    m["NEps"] = f"{MS.PRIMARY_EPS:.2f}"; m["NEpsLow"] = f"{MS.EPS_GRID[0]:.2f}"; m["NEpsHigh"] = f"{MS.EPS_GRID[-1]:.2f}"
    m["NRepsBase"] = "3"; m["NRepsBenign"] = "8"
    m["NBoot"] = f"{C.BOOT_N:,}"; m["NPairs"] = m["CThreeNPairs"]; m["NDefenses"] = "10"
    # ---- abstract / conclusion aliases of the pre-registered numbers
    m["FBAMLP"] = m["CThreeFbAmlp"]; m["RecAMLP"] = m["CThreeRecAmlp"]; m["RecProgent"] = m["CThreeRecProgent"]
    # ---- review round 1 (analysis/extra_checks.py): pilot-free subset, value share of class-level blocks, n
    x = json.load(open(f"{OD}/extra.json"))
    m["NClean"] = str(x["n_clean_tasks"]); m["CleanRecall"] = pct(x["clean_pooled_recall"])
    m["CleanNViol"] = str(x["clean_pooled_viol"])
    for model, a in ALIAS.items():
        m[f"CleanFbExact{a}"] = pct(x["per_model"][model]["fb_exact_all"])
    m["ClassValueShare"] = pct(x["class_level_blocks"]["value_share"])
    # ---- plan.md §15 pilot-clean robustness check (analysis/clean_subset.py)
    cs = json.load(open(f"{OD}/clean_subset.json"))
    f3, rc, c1c = cs["c3prime_fb"], cs["recall"], cs["C1"]
    m["CsFbAmlp"] = pct(f3["fb_amlp"]); m["CsFbProgent"] = pct(f3["fb_progent"])
    m["CsP"] = r"p < 0.001" if f3["mcnemar_p"] < 0.001 else f"p = {f3['mcnemar_p']:.3f}"
    m["CsDLo"] = pct(f3["d_ci95"][0]); m["CsDHi"] = pct(f3["d_ci95"][1])
    m["CsRecAmlp"] = pct(rc["recall_amlp"]); m["CsRecProgent"] = pct(rc["recall_progent"])
    m["CsCOneDrop"] = pct(c1c["drop_pooled"]); m["CsCOneUpper"] = pct(c1c["upper95"])
    m["CsRecDiff"] = pct(rc["recall_diff_amlp_minus_progent"]); m["CsRecDiffLo"], m["CsRecDiffHi"] = pct(rc["recall_diff_ci95"][0]), pct(rc["recall_diff_ci95"][1])
    sec = ["qwen3-8b-local", "gpt-4o-mini-2024-07-18", "gpt-4.1-mini-2025-04-14"]
    for arm, nm in (("amlp", "Amlp"), ("progent", "Progent")):
        bl = [100 * cs["rq3"][x][arm]["block_rate"] for x in sec]
        co = [100 * cs["rq3"][x][arm]["benign_cost"] for x in cs["rq3"]]
        m[f"Cs{nm}BlockMin"], m[f"Cs{nm}BlockMax"] = f"{min(bl):.1f}", f"{max(bl):.1f}"
        m[f"Cs{nm}CostMin"], m[f"Cs{nm}CostMax"] = f"{min(co):.1f}", f"{max(co):.1f}"
    g = cs["rq3"]["gpt-4.1-mini-2025-04-14"]["amlp"]
    m["CsAmlpCostFourOneMini"] = pct(g["benign_cost"])
    m["CsAmlpCostLoFourOneMini"], m["CsAmlpCostHiFourOneMini"] = pct(g["benign_cost_ci"][0]), pct(g["benign_cost_ci"][1])
    m["NCal"] = str(len(C.CALIBRATION)); m["NSel"] = str(len(C.SELECTION))
    m["CorrPct"] = f"{100 / (len(C.CALIBRATION) + 1):.1f}"   # conformal correction 1/(n+1)
    # ---- plan.md §16 seen vs novel (analysis/seen_novel.py) and §17 learning curve (analysis/seen_novel_curve.py)
    sn = json.load(open(f"{OD}/seen_novel.json"))
    for meth, nm in (("M1a", "Val"), ("M1b", "Tool"), ("M2", "Seq")):
        m[f"Sn{nm}Novel"] = pct(sn["pooled"][f"{meth}_novel_fb"]); m[f"Sn{nm}Seen"] = pct(sn["pooled"][f"{meth}_seen_fb"])
        dd = sn["delta"][f"{meth}_fb"]
        m[f"Sn{nm}Delta"] = pct(dd["point"]); m[f"Sn{nm}DeltaLo"] = pct(dd["ci95"][0]); m[f"Sn{nm}DeltaHi"] = pct(dd["ci95"][1])
        m[f"Sn{nm}Flag"] = pct(sn["pooled"][f"{meth}_novel_flag"])
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
    for part, nm in (("M2", "Seq"), ("M1a-pure", "ValPure"), ("M1b-pure", "ToolPure"), ("M2-seq", "SeqOnly")):
        x = ss["clean"][f"{part}_fb"]
        m[f"Cl{nm}Delta"], m[f"Cl{nm}DeltaLo"], m[f"Cl{nm}DeltaHi"] = pct(x["delta"]), pct(x["ci95"][0]), pct(x["ci95"][1])
    m["ClHybridValDelta"] = pct(ss["clean"]["M1a_fb"]["delta"])
    m["ClHybridValDeltaLo"], m["ClHybridValDeltaHi"] = pct(ss["clean"]["M1a_fb"]["ci95"][0]), pct(ss["clean"]["M1a_fb"]["ci95"][1])
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
    cn = lastok(f"{ROOT}/data/fidelity_camel_nosecpol.jsonl"); cs = lastok(f"{ROOT}/data/fidelity_camel.jsonl")
    att = [r for k, r in cn.items() if k[0] == "camel_nosecpol" and r["it"]]
    m["FidCamelUua"] = pct(sum(bool(r["utility"]) for r in att) / len(att))
    m["FidCamelAsr"] = pct(asr([r for k, r in cs.items() if k[0] == "camel"]))
    asf = json.load(open(f"{ROOT}/data/agentsentry_fidelity_results.json"))
    import statistics as st
    tr = [v for k, v in asf["runs"].items() if k.startswith("main|gbm|trace5|") and k.endswith("|L123")]
    m["FidSentryBlocked"] = f"{st.mean(x['abr'] for x in tr):.1f}"          # abr is already in percent
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
        return f"{scale * min(v):.1f}", f"{scale * max(v):.1f}"
    for arm, tag in (("progent", "Progent"), ("camel", "Camel"), ("pi_detector", "Pi"), ("melon", "Melon"),
                     ("agentsentry", "Sentry"), ("amlp", "Amlp"), ("tripwire", "Tripwire"), ("sandwich", "Sandwich"),
                     ("tool_filter", "ToolFilter"), ("spotlighting", "Spot"), ("block_all", "BlockAll")):
        lo, hi = rng(arm, "block_rate"); m[f"RqThree{tag}BlockMin"], m[f"RqThree{tag}BlockMax"] = lo, hi
        lo, hi = rng(arm, "benign_cost", models=[mm for mm in r3 if arm in r3[mm]])
        m[f"RqThree{tag}CostMin"], m[f"RqThree{tag}CostMax"] = lo, hi
    ga = r3["gpt-4.1-mini-2025-04-14"]["amlp"]                      # AMLP online benign cost, gpt-4.1-mini (Result 3)
    m["AmlpCostFourOneMini"] = pct(ga["benign_cost"])
    m["AmlpCostLoFourOneMini"], m["AmlpCostHiFourOneMini"] = pct(ga["benign_cost_ci"][0]), pct(ga["benign_cost_ci"][1])
    # largest upper CI end of Progent benign cost across models (Result 3), same source as tables/rq3_ci.tex
    m["RqThreeProgentCostHiMax"] = f"{100 * max(r3[mm]['progent']['benign_cost_ci'][1] for mm in r3 if r3[mm].get('progent', {}).get('benign_cost_ci')):.1f}"
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
    m["ParaShiftAmlpMax"] = f"{100 * max(abs(para[mm]['amlp']['shift']) for mm in para):.1f}"
    m["ParaShiftProgentQwen"] = f"{100 * para['qwen3-8b-local']['progent']['shift']:.1f}"
    m["ParaShiftProgentFourOneMini"] = f"{100 * para['gpt-4.1-mini-2025-04-14']['progent']['shift']:.1f}"
    tr = r4["transfer"]
    m["TransferFbMax"] = pct(max(v["fb"] for v in tr.values()))
    m["TransferRecFourToFourOne"] = pct(tr["gpt-4o-mini-2024-07-18->gpt-4.1-mini-2025-04-14"]["recall"])
    m["TransferRecFourOneToFour"] = pct(tr["gpt-4.1-mini-2025-04-14->gpt-4o-mini-2024-07-18"]["recall"])
    m["TransferRecLlamaToQwen"] = pct(tr["llama31-8b-local->qwen3-8b-local"]["recall"])
    po = r4["poisoning"]
    for mm in SEC:
        a = ALIAS[mm]
        m[f"PoisonRecZero{a}"] = pct(po[mm]["0.0"]["recall"]); m[f"PoisonRecOne{a}"] = pct(po[mm]["0.01"]["recall"])
        m[f"PoisonRecFive{a}"] = pct(po[mm]["0.05"]["recall"]); m[f"PoisonRecTen{a}"] = pct(po[mm]["0.1"]["recall"])
    for mm in SEC:
        a = ALIAS[mm]
        for rho, tag in (("0.05", "Five"), ("0.1", "Ten")):
            m[f"PoisonMean{tag}{a}"] = pct(po[mm][rho]["recall_mean"])
            m[f"PoisonMin{tag}{a}"] = pct(po[mm][rho]["recall_min"]); m[f"PoisonMax{tag}{a}"] = pct(po[mm][rho]["recall_max"])
    m["PoisonSeeds"] = str(len(po["qwen3-8b-local"]["0.05"]["per_seed"]))
    for mm, a in (("qwen3-8b-local", "Qwen"), ("gpt-4.1-mini-2025-04-14", "FourOneMini")):
        x = para[mm]
        m[f"ParaShiftAmlp{a}"] = f"{100 * x['amlp']['shift_paired']:.1f}"
        m[f"ParaShiftAmlpLo{a}"] = f"{100 * x['amlp']['shift_ci'][0]:.1f}"; m[f"ParaShiftAmlpHi{a}"] = f"{100 * x['amlp']['shift_ci'][1]:.1f}"
        m[f"ParaShiftProgentLo{a}"] = f"{100 * x['progent']['shift_ci'][0]:.1f}"; m[f"ParaShiftProgentHi{a}"] = f"{100 * x['progent']['shift_ci'][1]:.1f}"
        m[f"ParaReplayFb{a}"] = pct(x["amlp_replay_fb_on_paraphrase"])
    # plan §14 headline: max absolute shift over (a) and (b); its condition is the Llama -> Qwen transfer
    shifts = [abs(para[mm][arm]["shift"]) for mm in para for arm in ("amlp", "progent")]
    base = {k.split("->")[1]: po[k.split("->")[1]]["0.0"]["recall"] for k in tr}
    shifts += [abs(v["recall"] - po[k.split("->")[1]]["0.0"]["recall"]) for k, v in tr.items() if v["recall"] is not None]
    m["RQFourMaxShift"] = f"{100 * max(shifts):.1f}"
    m["PoisonFbMax"] = pct(max(v["fb"] for mm in po for v in po[mm].values()))
    m["PoisonRunsFive"] = str(po["qwen3-8b-local"]["0.05"]["poisoned_runs"])
    # ---- per-suite breakdown and stacking with TripWire (analysis/per_suite.py)
    ps = json.load(open(f"{OD}/per_suite.json"))
    for suite in ("banking", "slack", "travel", "workspace"):
        m[f"SuiteAmlp{suite.capitalize()}"] = pct(ps["block"]["amlp"][suite])
        m[f"SuiteProgent{suite.capitalize()}"] = pct(ps["block"]["progent"][suite])
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
    OD = f"{ROOT}/data/analysis_out"
    m = {}
    rs = json.load(open(f"{OD}/resplit.json"))
    m["RsN"] = str(rs["meta"]["n_splits"])
    kept_fb, kept_flag, drop_flag = [], [], []
    for model, x in rs["models"].items():
        s = x["summary"]["0.1"]
        m[f"RsKept{ALIAS[model]}"] = f"{100 * s['kept_share']:.0f}"
        m[f"RsKeptCount{ALIAS[model]}"] = str(s["kept"])
        if s["fb_mean_kept"] is not None:
            kept_fb.append(s["fb_mean_kept"])
        if s["flag_mean_kept"] is not None:
            kept_flag.append(s["flag_mean_kept"]); drop_flag.append(s["flag_mean_dropped"])
    m["RsFbKeptMin"], m["RsFbKeptMax"] = pct(min(kept_fb)), pct(max(kept_fb))
    m["RsFlagKeptMin"], m["RsFlagKeptMax"] = pct(min(kept_flag)), pct(max(kept_flag))
    m["RsFlagDroppedMin"], m["RsFlagDroppedMax"] = pct(min(drop_flag)), pct(max(drop_flag))
    fb = json.load(open(f"{OD}/flag_before_harm.json"))["replay"]
    m["RecAMLP"], m["RecProgent"] = pct(fb["calibrated"]["flag_before_harm"]), pct(fb["progent"]["flag_before_harm"])
    m["FbhToolPure"], m["FbhToolPred"] = pct(fb["mined_tools_only"]["flag_before_harm"]), pct(fb["predicted_tools_only"]["flag_before_harm"])
    m["FbhAmlpDrop"] = f"{fb['calibrated']['diff_pp']:.1f}"
    ex = json.load(open(f"{OD}/flag_before_harm_online_valueonly.json"))["online"]
    for arm, tag in (("amlp", "Amlp"), ("progent", "Progent"), ("block_all", "BlockAll")):
        v = [100 * ex[mm][arm]["interception_exec"] for mm in ex if arm in ex[mm]]
        m[f"Exec{tag}Min"], m[f"Exec{tag}Max"] = f"{min(v):.1f}", f"{max(v):.1f}"
    ad = json.load(open(f"{OD}/all48/rq34_agentdojo.json"))["rq3"]          # AgentDojo oracle (appendix)
    ex48 = json.load(open(f"{OD}/all48/rq34.json"))["rq3"]
    sec_models = [mm for mm in ad if ad[mm].get("amlp", {}).get("block_rate") is not None]
    for arm, tag in (("amlp", "Amlp"), ("progent", "Progent"), ("block_all", "BlockAll")):
        v = [100 * ad[mm][arm]["block_rate"] for mm in sec_models]
        m[f"Ad{tag}Min"], m[f"Ad{tag}Max"] = f"{min(v):.1f}", f"{max(v):.1f}"
    m["ExecMaxShift"] = f"{max(abs(100 * (ex48[mm][a]['block_rate'] - ad[mm][a]['block_rate'])) for mm in sec_models for a in ('amlp', 'progent')):.1f}"
    vb = json.load(open(f"{OD}/value_block_cause.json"))["llama31-8b-local"]
    m["LlamaBenignUtil"] = pct(vb["benign_utility"]); m["LlamaBlockedUtil"] = pct(vb["utility_of_blocked"])
    m["LlamaValueBlocks"] = str(vb["blocked_value"]); m["LlamaToolBlocks"] = str(vb["blocked_tool"]); m["LlamaRuns"] = str(vb["runs"])
    fi = json.load(open(f"{OD}/fidelity_benign.json"))
    m["FidSentryUtil"], m["FidSentryUtilPub"] = f"{fi['agentsentry']['util_ours']:.1f}", f"{fi['agentsentry']['util_published']:.1f}"
    m["FidCamelUtilNone"], m["FidCamelUtilCamel"] = f"{fi['camel']['util_none']:.1f}", f"{fi['camel']['util_camel']:.1f}"
    return m


def main():
    prim, sec = build("all48"), build("clean27")                  # plan.md §21: primary = pre-registered 48 tasks
    assert prim["NTasksEval"] == "48" and sec["NTasksEval"] == "27" and prim["NPilotOverlap"] == "21"
    assert prim["NClean"] == "27"
    m = dict(prim)
    m.update({k + "Untouched": v for k, v in sec.items()})
    m.update(section22())
    lines = ["% generated by analysis/make_numbers.py; do not edit by hand",
             "% primary macros: all 48 holdout tasks (pre-registered); suffix Untouched: the 27 holdout tasks no pilot run touched"]
    for k, val in m.items():
        assert re.fullmatch(r"[A-Za-z]+", k), k
        lines.append(f"\\newcommand{{\\{k}}}{{{val}\\xspace}}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print(f"{len(m)} macros -> {OUT}")


if __name__ == "__main__":
    main()
