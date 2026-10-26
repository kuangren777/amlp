# AMLP code and data release

AMLP (agent mined least privilege) builds a per-task policy for a tool-using LLM agent from benign traces that the
same agent produced on other tasks of the same suite. The policy has a tool layer (which side-effecting tools a task
may call) and a value layer (which argument values those calls may carry), and a configuration is calibrated on a
held-out fold so that the benign false-block rate stays below a target level. The study measures, on the AgentDojo
benchmark (suites banking, slack, travel and workspace, version v1.2), how much benign utility such a policy costs and
how many successful prompt-injection attacks it intercepts, against re-implemented or ported baseline defenses
(Progent, CaMeL, Agent-Sentry, ToolFence and others). It also tests how the result changes when a task's own
benign traces are or are not available for mining, and which parts of the policy rest on mined values versus values
taken from the user request. The experimental protocol, decision rules and amendments were written down before the
corresponding data were analysed and are in `PREREGISTRATION.md`.

## Layout

```
code/                 AMLP, the harness and the analysis scripts (flat Python modules, run from inside code/)
  amlp.py             AMLP executor and envelope (tool layer, value layer)
  harness.py          shared AgentDojo harness (run_pair, defenses, row schema)
  envbuild.py         task-disjoint envelope construction
  m3_*.py, m3_*.sh    M3 pipeline: select (calibration), mine, holdout monitor, online runs, judge, analyze, paraphrase
  make_holdout.py     draws the holdout split (holdout_split.json)
  exp_pilot.py        pilot online runs (data/pilot_m2a.jsonl)
  test_amlp.py, test_m3.py, test_m3_paraphrase.py   offline unit tests (pytest)
  analysis/           the analysis scripts that produce every number, table and figure of the paper
  baselines/          ports and re-implementations of the baseline defenses, fidelity and smoke scripts, their tests
  pilot/              the two pilot scripts that produced the frozen snapshot caches
data/
  m3/                 M3 rows: mine_*, monitor_*, online_*, judge_*, paraphrase_*, steps/, judge_tests.json, ...
                      online_*__camel_nosecpol.jsonl (CaMeL without security policies), melon_fidelity_rows.jsonl (MELON fidelity runs)
  praetor_asb_runs.jsonl   benign runs of the Praetor close replication on ASB (plan section 22.11)
  m3_selection_log.jsonl   calibration log (selection of the configuration per model)
  a2/                 AgentDyn runs of the independent replication (plan section 25): monitor_*, attack_*, steps/, predictor caches
  s5_fresh/           three independent blind labels of the second oracle sample (labels_a/b/c.jsonl)
  harm_audit/         blind items and two model labels of the harm-step audit (blind.jsonl, labels_a/b.jsonl)
  agentsentry_judge_cache.jsonl   cached LLM-judge verdicts of the Agent-Sentry baseline
  pilot_m2a.jsonl     pilot online rows
  pilot/              frozen pilot snapshot read by envbuild.py (tw_main_20261005.jsonl, embeddings, environment dumps)
  fidelity_*.jsonl, progent_ab_*.jsonl, agentsentry_fidelity_results.json   baseline fidelity checks
  results_m3_analyze.txt, results_m3_judge_test*.{txt,json}                 frozen outputs of m3_analyze.py / m3_judge.py --test
  analysis_out/       JSON outputs of the analysis scripts, one directory per holdout set
    all48/            all 48 holdout tasks, the pre-registered primary set of the paper
    clean27/          the 27 holdout tasks no pilot run touched, printed beside every main number
    seen_novel_emb.json.gz   embedding cache shared by both sets (gzipped)
    resplit.json, resplit_altchain.json, resplit_breadth.json, flag_before_harm*.json, fidelity_benign.json,
    value_block_cause.json, melon_fidelity.json, praetor_asb.json   plan section 22 outputs, not per set
    calib_rules.json (+ .log)   plan section 23 output; table_marks.json   dagger marks written by make_figs.py, read by make_numbers.py
    {all48,clean27}/refusal_bounds.json, amlp_cost_mcnemar.json   plan sections 22.9 and 22.12 outputs, per set
    _superseded_flat/ outputs of the earlier 48-task-only analysis, kept for the record, read by no script
    s5/               attack-oracle rescoring (s5_rescore.py): scored.jsonl holds the replay oracle per attack row, read by
                      rq12.py (violation set, plan section 26) and rq34.py (interception); the blind label samples and their evaluation
    s5_fresh/eval.json, s4/, a2/, injection/, s3_parse_rate.json   plan sections 24 to 29 outputs
    harm_audit/eval.json, harm_audit/sample_map.json              plan section 27 harm-step audit
    seen_novel_emb_minilm.json.gz   all-MiniLM-L6-v2 embedding cache of seen_novel_sens.py (gzipped)
  analysis_out_all4/  the same outputs computed over four models including Llama-3.1-8B; make_numbers.py reads it for the
                      Llama-3.1-8B appendix only (the main results use the three models with successful attacks)
outputs/              created on demand: numbers.tex, figs/, tables/ (written by make_numbers.py, make_figs.py, ...)
third_party/          licenses and provenance of upstream projects used by the baseline ports (see third_party/README.md)
PREREGISTRATION.md    the pre-registration and amendment log (plan), scrubbed of host names, paths and personal names
SCRUB.md              internal scrub report (quotes the forbidden patterns; remove before publishing)
LICENSE               MIT
```

Every row is one agent run and carries the model, suite, user task, injection task, defense, rep, seed, utility and
security outcome, the tool-call trace and bookkeeping fields. In `served_root`, `hub:<model>` marks a model reached
through a hosted OpenAI-compatible gateway and `local:<name>` a self-hosted vLLM model. Hosts, endpoints and paths of
the original runs were removed (see `SCRUB.md`). No metric field was changed.

## Environment

- Python 3.10 or newer (code uses `str | None` and `str.removeprefix`). The analysis was run with Python 3.13.
  The CaMeL baseline needs its own Python 3.12 virtual environment, the Agent-Sentry baseline Python 3.13 with
  scikit-learn and xgboost.
- Packages: `agentdojo==0.1.35` (benchmark suites v1.2, imported as `agentdojo`), `openai`, `numpy`, `matplotlib`,
  `jsonschema`, `pydantic`, `pytest`. Optional: `pydantic-ai==0.2.12` and `openai==1.82.1` inside the CaMeL environment
  (see `third_party/README.md`), `transformers` for the `pi_detector` baseline.
- If you work from an AgentDojo source checkout instead of the pip package, set `AGENTDOJO_SRC=<checkout>/src`.
- Optional extras, only for the baseline tests and runs that need them (tests marked `optional` are skipped when the
  package is missing): the upstream CaMeL package `camel` from a checkout of the CaMeL repository (`CAMEL_REPO`,
  `pip install -e $CAMEL_REPO` in the Python 3.12 CaMeL environment with `pydantic-ai==0.2.12` and `openai==1.82.1`)
  for four tests in `code/baselines/camel_test.py`; `scikit-learn` and `xgboost` for the Agent-Sentry tests;
  `transformers` and `torch` for the `pi_detector` baseline.

```
python3 -m venv .venv && . .venv/bin/activate
pip install agentdojo==0.1.35 openai numpy matplotlib jsonschema pydantic pytest
cd code && python3 -m pytest -q test_amlp.py test_m3.py test_m3_paraphrase.py
```

Four tests in `code/baselines/camel_test.py` need the upstream CaMeL package (`camel`, installed in the CaMeL environment from `CAMEL_REPO`) and are skipped when it is missing. `test_m3.py::test_r5_served_root_hub_and_port_commit` asks `git` for the commit of `baselines/progent_port.py` and
fails outside a git checkout. All other tests are offline.

## Model access

Offline analysis needs no network and no key. Anything that calls a model (agent runs, the judge, the LLM tool
predictor, embeddings that are missing from the cache) reads these environment variables:

| Variable | Meaning |
|---|---|
| `LLM_API_BASE` | OpenAI-compatible base URL of the hosted gateway, including the `/v1` suffix |
| `LLM_API_KEY` | key for that gateway |
| `LLM_API_BASE_<NAME>` | base URL (with `/v1`) of a self-hosted vLLM server for one local model. `<NAME>` is the model name in upper case with every non-alphanumeric character replaced by `_`: `LLM_API_BASE_QWEN3_8B_LOCAL`, `LLM_API_BASE_LLAMA31_8B_LOCAL`, `LLM_API_BASE_QWEN25_7B_LOCAL`, `LLM_API_BASE_MISTRAL_7B_LOCAL`, `LLM_API_BASE_GRANITE31_8B_LOCAL` |
| `TRIPWIRE_DIR` | optional, checkout that provides `tripwire.py` and `melon_port.py` (not part of this release) |
| `PI_DETECTOR_MODEL` | optional, model path or id for the `pi_detector` baseline (default `protectai/deberta-v3-base-prompt-injection-v2`) |
| `CAMEL_REPO` | optional, checkout of the upstream CaMeL repository used by `baselines/camel_run.py` |
| `AUTODOJO_SRC`, `ROPE_RUNS` | optional, the AgentDyn benchmark source (`autodojo/src` of ROPE) and ROPE's runner directory, for the AgentDyn runs (`a2_agentdyn.py`, `a2_replay.py`, `a2_adapter_check.py`) |
| `AMLP_MODELS` | optional, comma-separated model list that restricts `m3_common.MODELS`; the main results use `qwen3-8b-local,gpt-4o-mini-2024-07-18,gpt-4.1-mini-2025-04-14` |

Model ids used in the study: `gpt-4o-mini-2024-07-18` and `gpt-4.1-mini-2025-04-14` (hosted), `qwen3-8b-local`
(Qwen/Qwen3-8B) and `llama31-8b-local` (Meta-Llama-3.1-8B-Instruct), both served with vLLM (tool calling enabled),
`gpt-4o-2024-08-06` and `o4-mini-2025-04-16` for baseline fidelity checks, and `bge-m3` for embeddings. The
concurrency guard in `m3_common.py` (`MODELS`, `effective_workers`) refers to the two vLLM servers as `vllm-A`
(qwen3) and `vllm-B` (llama31).

## Baselines

The baseline ports live in `code/baselines/`. Upstream checkouts are not redistributed. `third_party/README.md` lists,
for Progent, CaMeL and AgentDojo, the upstream URL, commit where recoverable and license, and copies the license
texts. Agent-Sentry and ToolFence are re-implementations from the papers. The `data/fidelity_*.jsonl`,
`data/progent_ab_*.jsonl` and `data/agentsentry_fidelity_results.json` files are the frozen outputs of the fidelity
checks of the ports against the published numbers. The Progent policy caches (`baselines/cache/`) and the
virtual environments are not included.

## Reproduce

All commands run from `code/` (`cd code`). Paths are resolved relative to the release root, so `data/` and
`outputs/` are found without configuration.

Offline, from the frozen data in `data/` (no model access):

| Command | What it produces |
|---|---|
| `HOLDOUT_SET=clean27\|all48 python3 analysis/<script>.py` | Environment switch read by `analysis/holdout_set.py`, which every analysis script imports. `all48` evaluates on all 48 holdout tasks, the primary set of the paper. `clean27` (the default of the switch) evaluates on the 27 holdout tasks that no pilot run touched, the robustness check for the pilot overlap. Mining, selection and calibration rows and the envelopes are the same in both modes. The scripts write to `data/analysis_out/<set>/`. The paper reports `all48` with the `clean27` value beside it. |
| `python3 analysis/rq12.py` | RQ1 and RQ2: per model and per element of the nested chain, calibration-fold loss and bound, holdout false-block rate and interception of the undefended successful attacks, interception of the tool layer per attacker tool, share of violations invisible to it. Writes `data/analysis_out/<set>/rq12.json`. |
| `python3 analysis/rq34.py [--dry-run]` | RQ3 and RQ4 exactly as pre-registered in plan section 14 (`--dry-run` only validates the row schema and counts pairs). Writes `data/analysis_out/<set>/rq34.json`. Environment switch `ATTACK_ORACLE`: `exec` (default, the main text) counts an attack as successful when a call matching a ground-truth side-effecting call of its injection task ran without a refusal (`analysis/exec_oracle.py`, plan section 22); `agentdojo` uses the AgentDojo security flag of the row (appendix `app:oracle`). The frozen appendix outputs `rq34_agentdojo.json` and `per_suite_agentdojo.json` in each set directory are the `ATTACK_ORACLE=agentdojo` runs, renamed after the run (the scripts always write `rq34.json` and `per_suite.json`). |
| `python3 analysis/exec_oracle.py` | Library, not run directly. Execution-semantics attack oracle imported by `rq34.py` (and through it by `per_suite.py` and `flag_before_harm.py`). Reads the step logs `data/m3/steps/online_*.jsonl.gz` and the task definitions of `harness.py`. |
| `python3 analysis/extra_checks.py` | Review-round checks: RQ1 and RQ2 on holdout tasks that the pilot test split never contained, and the share of blocked benign mining runs refused for a value rather than a tool. Writes `extra.json`. |
| `python3 analysis/per_suite.py` | Per-suite breakdown of block rate and benign cost of every defense, and the share of AMLP-passed attacks that TripWire stops. Writes `per_suite.json` (honours `ATTACK_ORACLE`) and `outputs/figs/rq3_per_suite.pdf`. |
| `python3 analysis/clean_subset.py` | Pilot-clean robustness check (plan section 15): C1, the C3' tests and the RQ3 per-arm numbers on holdout tasks that no pilot run touched. Writes `clean_subset.json`. |
| `python3 analysis/seen_novel.py` | Seen-versus-novel transfer of mined least privilege (plan section 16), AMLP against a Praetor-style pDFA. Writes `seen_novel.json`. Uses `data/analysis_out/seen_novel_emb.json.gz` as embedding cache. Needs model access (`LLM_API_BASE`, `LLM_API_KEY`, model `bge-m3`) only if a text is missing from the cache. |
| `python3 analysis/seen_novel_curve.py` | Data-volume learning curve of the seen-novel gap (plan section 17). Writes `seen_novel_curve.json`. |
| `python3 analysis/seen_novel_pure.py` | Pure-mined ablation of the AMLP parts (plan section 18). Writes `seen_novel_pure.json`. |
| `python3 analysis/seen_novel_split.py` | Automaton component split and clean-subset replication (plan section 19): the Praetor-style pDFA with its argument guards off or with per-tool argument schemas only, seen versus novel, and every part of sections 16 and 18 restricted to the holdout tasks no pilot run touched. Writes `seen_novel_split.json`. |
| `python3 analysis/holdout_tests.py` | Pre-registered C3' false-block test, recall side and C1 on the active holdout set (`m3_judge.c3_tests`, `m3_judge.recall_side`, `m3_analyze.c1`). With `HOLDOUT_SET=all48` it asserts that the result equals `data/m3/judge_tests.json` and `data/results_m3_analyze.txt`. Writes `data/analysis_out/<set>/tests.json`. |
| `python3 analysis/holdout_set.py` | Not run directly. Defines the two sets and filters holdout rows of the other set when `m3_common.read_jsonl` is called. |
| `analysis/fmt1.py`, `analysis/fmtnum.py` | Number formatting used by `make_figs.py` and `make_numbers.py`: every printed value is computed at full precision and rounded once with ROUND_HALF_UP, ratios of counts as exact fractions. |
| `python3 analysis/make_numbers.py` | Every number used in the prose as a LaTeX macro, computed from the frozen data and the JSON files of both sets. Primary macros come from `all48`, the same macro from `clean27` carries the suffix `Untouched`. Writes `outputs/numbers.tex`. |
| `python3 analysis/make_figs.py` | Figures and tables of the paper from `data/analysis_out/*.json`. One run reads both sets. Figures `outputs/figs/*.pdf` show `all48`. Every main table in `outputs/tables/*.tex` prints the `clean27` value after each differing `all48` value as `\untouched{...}`. |
| `python3 analysis/resplit.py` | Calibration-split robustness (plan section 22): the calibration and holdout roles of the tasks are swapped and redrawn, and the holdout false-block and flag rate at the calibrated configuration are recomputed. Writes `data/analysis_out/resplit.json`. |
| `python3 analysis/flag_before_harm.py` | Flag-before-harm (plan section 22): for each successful attack, whether the envelope refuses a call before the first harmful call of the run, for the replayed holdout runs and the online AMLP and Progent arms. Writes `data/analysis_out/flag_before_harm.json` and `flag_before_harm_online_valueonly.json`. `--replay-only` recomputes only the replay block and keeps the frozen online block of the existing file (plan section 22.6). A run without the flag recomputes the online block as well and gives different `interception_agentdojo` and `diff_pp` values there than the frozen file (this `online` block of the frozen file is not read by `make_numbers.py` or `make_figs.py`); use `--replay-only` to regenerate the file as shipped. |
| `python3 analysis/fidelity_benign.py` | Benign-side utility of the Agent-Sentry and CaMeL fidelity runs, from `data/agentsentry_fidelity_results.json` and `data/fidelity_camel.jsonl` (plan section 22.3). Writes `data/analysis_out/fidelity_benign.json`. |
| `python3 analysis/value_block_cause.py` | Which kind of value (user-request value versus mined value) causes the value-layer false blocks of `llama31-8b-local` (plan section 22). Writes `data/analysis_out/value_block_cause.json`. |
| `HOLDOUT_SET=clean27\|all48 python3 analysis/refusal_bounds.py` | Benign utility under refusal, lower and upper bound for every refusing defense (plan section 22.9). Reads the online rows including `online_*__camel_nosecpol.jsonl`. Writes `data/analysis_out/<set>/refusal_bounds.json`. |
| `HOLDOUT_SET=clean27\|all48 python3 analysis/amlp_cost_mcnemar.py` | McNemar test and bootstrap of the AMLP benign cost (plan section 22.12). Writes `data/analysis_out/<set>/amlp_cost_mcnemar.json`. |
| `CHAIN=alt python3 analysis/resplit.py` | Same as `resplit.py` with the value level coarsened before the tool layer is widened (plan section 22.10). Writes `data/analysis_out/resplit_altchain.json`. |
| `python3 analysis/resplit_breadth.py` | Tool breadth of the kept re-splits (plan section 22.10), from `resplit.json`. Writes `data/analysis_out/resplit_breadth.json`. |
| `python3 analysis/calib_rules.py [n_splits]` | CRC selection rule against empirical, fixed and seen-task calibration rules on 100 re-splits (plan section 23). Writes `data/analysis_out/calib_rules.json`. |
| `python3 analysis/praetor_asb.py`, `analysis/melon_fidelity.py`, `analysis/run_camel_nosecpol.py` | Model-calling scripts of plan sections 22.9 and 22.11 that produced `data/praetor_asb_runs.jsonl`, `data/m3/melon_fidelity_rows.jsonl` and `data/m3/online_*__camel_nosecpol.jsonl`. They need model access, and checkouts given by `ASB_DIR` (Praetor replication, also for its offline evaluation step, which reads the persona's tool list), `MELON_UPSTREAM` (upstream MELON) and `CAMEL_REPO`. Not needed to check the paper's numbers. |
| `python3 analysis/make_example_fig.py` | The page-1 example: one logged undefended attack run replayed against the calibrated allowlist, the first refused call being the first harmful call. Reads the logs and the envelope (needs `harness.py` and the step logs, no model access). Writes `outputs/tables/example_replay.tex`. |
| `python3 analysis/s5_rescore.py` | Rescores every attack row under four oracles (execution, AgentDojo, replay, replay at any point) and writes `data/analysis_out/s5/<set>/scored.jsonl`; the replay oracle at any point is the one of the paper. `s5_labels.py`, `s5_label_eval.py`, `s5_report.py`, `analysis/s5_fresh/eval.py` evaluate the oracles against the blind labels. `analysis/harm_audit/sample.py` draws the harm-step audit sample and `analysis/harm_audit/eval.py` compares its labels with the harm step (Appendix C). |
| `python3 analysis/a2_prep.py`, `analysis/a2_agentdyn.py` | The AgentDyn replication analysis (plan section 25) over `data/a2/`. |
| `python3 analysis/seen_novel_sens.py`, `s4_fidelity.py`, `s3_parse_rate.py`, `injection_*.py`, `s5_calib_robust.py`, `s5_gap_by_task.py` | Sensitivity of the automaton gap (section 24), per-persona automaton fidelity, tool-call parse rates, the gateway prefix check of the supplementary table, calibration under the replay oracle and per-task gaps. |
| `python3 analysis/make_method_fig.py` | The method figure, drawn in code. Writes `outputs/figs/method.pdf` and a preview PNG in the current directory. |
| `python3 m3_analyze.py` | C1 non-inferiority test and the per-arm table (stdout is the content of `data/results_m3_analyze.txt`). |
| `python3 m3_judge.py --test` | The pre-registered judge tests (`data/m3/judge_tests.json`). |

Run the `rq12`, `rq34`, `extra_checks`, `per_suite`, `clean_subset`, `seen_novel*`, `holdout_tests`, `refusal_bounds`, `amlp_cost_mcnemar` scripts (once per set) and `resplit` (twice, with and without `CHAIN=alt`), `resplit_breadth`, `calib_rules`, `flag_before_harm`, `fidelity_benign`, `value_block_cause` (once, they write to the top level of `data/analysis_out/`) before `make_figs.py` and `make_numbers.py` (in this order, because `make_figs.py` writes `table_marks.json`, which `make_numbers.py` reads), since both read their JSON output. Frozen copies of those JSON files are in `data/analysis_out/clean27/` and `data/analysis_out/all48/`,
so the generators also run directly. Run them from `code/` in this order: `python3 analysis/make_figs.py`, then `python3 analysis/make_numbers.py`, `python3 analysis/make_example_fig.py`, `python3 analysis/make_method_fig.py`. `make_figs.py` must come first because it creates `outputs/tables/` and writes `data/analysis_out/table_marks.json`, which `make_numbers.py` reads. Rerunning the analysis rewrites the JSON files in place.

With model access (new agent runs, not needed to check the paper's numbers):

| Command | Purpose |
|---|---|
| `python3 m3_select.py --model M` | calibration and selection of the configuration |
| `python3 m3_mine.py --model M` | benign mining runs |
| `python3 m3_holdout_monitor.py --model M` | benign holdout runs under monitoring |
| `python3 m3_online.py --model M --group main` | online arms (undefended, AMLP, baselines) |
| `python3 m3_judge.py --model M --target holdout|attack` | LLM judge over runs |
| `python3 m3_paraphrase.py ...` | paraphrase robustness runs |
| `m3_stage1.sh`, `m3_stage2*.sh`, `m3_v5_baselines.sh`, `m3_judge_stage.sh` | the batch drivers used in the study |

The `.sh` drivers launch the steps above for all four models in parallel and assume the model endpoints above are set.

## Pre-registration

`PREREGISTRATION.md` holds the plan with its amendments in the order they were written (protocol rules R1 to R6, the
chain of nested configurations, decision rules, and sections 14 to 19 for the RQ3/RQ4 analysis, the pilot-clean check,
the seen-versus-novel test, its learning curve, the pure-mined ablation and the automaton split, section 20 and 21 on the choice of the primary set, section 22 on the split, flag-before-harm, refusal-bound, chain-order, ASB and MELON checks and the CaMeL policy interval, section 23 on the calibration rule comparison, section 24 on the automaton heuristics, section 25 on the AgentDyn replication, and section 26 on the replay-oracle violation set and its harm step, including the revision made after a first result). Paths and script names inside it are those
of the original development tree: `analysis/out/<set>/` is `data/analysis_out/<set>/` here (and `analysis/out/*.json` of the early sections is `data/analysis_out/_superseded_flat/`, whereas the flat `analysis/out/*.json` of sections 22 and 23 are at the top level of `data/analysis_out/`) and the `pilot/` snapshot is
`data/pilot/`.

## Known limits of this release

- Runs against hosted models are not bit-reproducible, because the gateway does not guarantee deterministic output.
- The virtual environments, Progent policy caches, logs and the paper sources are not included.
- Some strings in the rows (for example file names under a fictional user home directory, `bluesparrowtech.com` addresses) are
  AgentDojo benchmark content or text invented by the agent. They are kept on purpose, see `SCRUB.md`.
