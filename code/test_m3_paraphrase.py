"""Tests for the E3a paraphrase runner (no network, no agent model calls). Run: python3 -m pytest -q test_m3_paraphrase.py"""
import inspect
import json
import os
import sys
import threading
import types

import pytest

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
import m3_common as C  # noqa: E402
import m3_paraphrase as M  # noqa: E402

H = C.harness
WS6 = ("workspace", "user_task_6")      # has an e-mail, a date, a time, a quoted title and a name
BK4 = ("banking", "user_task_4")        # has an IBAN


def prompt(t):
    return H.suite(t[0]).user_tasks[t[1]].PROMPT


# ---------------------------------------------------------------- validation
def test_validation_accepts_faithful_and_catches_dropped_values():
    p = prompt(WS6)
    assert M.missing_values(p, p) == []
    req = M.required_values(p)
    for v in ("sarah.connor@gmail.com", "2024-05-19", "12:00", "Sarah", "Lunch"):
        assert v in req, v
    assert M.missing_values(p, p.replace("sarah.connor@gmail.com", "Sarah's e-mail")) == ["sarah.connor@gmail.com"]
    assert "2024-05-19" in M.missing_values(p, p.replace("2024-05-19", "May 19, 2024"))
    assert "Sarah" in M.missing_values(p, p.replace("Sarah", "her friend"))
    ib = prompt(BK4)
    iban = next(v for v in M.required_values(ib) if v.startswith("GB"))
    assert iban in M.missing_values(ib, ib.replace(iban, "the account"))


def test_number_spelled_out_is_caught():
    p = prompt(("workspace", "user_task_18"))
    assert "5" in M.required_values(p)
    assert M.missing_values(p, p.replace("5 hours", "five hours")) == ["5"]


def test_regeneration_and_drop_recorded():
    p = prompt(WS6)
    bad = p.replace("sarah.connor@gmail.com", "her address")
    good = [p.replace("Am I free", "Do I have time"), p.replace("please create", "then create"),
            p.replace("If so, ", "If yes, ")]

    class Fake:   # first call: one good, one bad, one bad; regenerations: good once, then always bad
        def __init__(self):
            self.n = 0
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

        def create(self, **kw):
            self.n += 1
            out = [good[0], bad, bad] if self.n == 1 else ([good[1]] if self.n == 2 else [bad])
            msg = types.SimpleNamespace(content=json.dumps({"paraphrases": out}))
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])
    slots, rep = M.generate_one(Fake(), p)
    assert slots[0] == good[0] and slots[1] == good[1] and slots[2] is None
    assert rep[1]["regenerations"] == 1 and rep[2]["regenerations"] == M.MAX_REGEN and not rep[2]["accepted"]


def test_generate_refuses_when_frozen(tmp_path):
    f = tmp_path / "p.json"
    f.write_text("{}")
    with pytest.raises(SystemExit):
        M.generate(str(f), client=object())


# ---------------------------------------------------------------- thread-safe prompt override
def test_override_is_thread_safe(monkeypatch):
    ut = H.suite(WS6[0]).user_tasks[WS6[1]]
    orig = ut.PROMPT
    paras = {WS6: {0: "PARAPHRASE ZERO " + orig, 1: "PARAPHRASE ONE " + orig}}
    seen, bar = {}, threading.Barrier(2)

    class Stub:
        def query(self, q, runtime, env, messages=(), extra_args={}):
            bar.wait(timeout=30)                  # both runs are inside query at the same time
            seen[threading.current_thread().name] = (q, ut.PROMPT)
            return q, runtime, env, [], extra_args
    monkeypatch.setattr(H, "make_pipeline", lambda *a, **k: Stub())
    f = M.Arms("gpt-4o-mini-2024-07-18", arms=()).run_fn(paras)
    th = [threading.Thread(target=f, name=f"t{p}", args=({"defense": "none", "suite": WS6[0], "ut": WS6[1],
                                                          "it": None, "rep": 0, "p": p},)) for p in (0, 1)]
    [t.start() for t in th]
    [t.join() for t in th]
    assert seen["t0"] == (paras[WS6][0], orig) and seen["t1"] == (paras[WS6][1], orig)
    assert ut.PROMPT == orig


def test_amlp_envelope_copy_per_run():
    A = M.Arms("gpt-4o-mini-2024-07-18", arms=())
    cached = types.SimpleNamespace(prompt="ORIGINAL", tools=set(), mined_tasks=frozenset())
    A._env_cache[WS6] = cached
    A.cfg, A.pool, A.ME = {"predictor": None, "r": 0}, None, types.SimpleNamespace(EB=types.SimpleNamespace(E={}))
    got, bar = {}, threading.Barrier(2)

    def go(p):
        M._x.p, M._x.prompt = 0, p
        bar.wait(timeout=30)
        got[p] = A.envelope(*WS6).prompt
    th = [threading.Thread(target=go, args=(p,)) for p in ("PA", "PB")]
    [t.start() for t in th]
    [t.join() for t in th]
    assert got == {"PA": "PA", "PB": "PB"} and cached.prompt == "ORIGINAL"


# ---------------------------------------------------------------- jobs, row ids, rows
def fake_paras(drop=()):
    return {t: {p: f"para {p} of {t}" for p in range(3) if (t, p) not in drop} for t in C.HOLDOUT}


def test_job_count_432_minus_dropped():
    assert len(C.HOLDOUT) == 48
    assert len(M.jobs(fake_paras())) == 48 * 3 * 3 == 432
    t = C.sorted_tasks(C.HOLDOUT)[0]
    J = M.jobs(fake_paras(drop={(t, 2)}))
    assert len(J) == 432 - 3 and {j["defense"] for j in J} == set(M.ARMS)
    assert len(M.smoke_jobs(fake_paras())) == 12
    R = M.ParaRunner(M.RUNNER, "gpt-4o-mini-2024-07-18", True)
    assert len({R.rid(j) for j in J}) == len(J)


def test_job_count_from_frozen_file():
    if not os.path.exists(M.PARA_FILE):
        pytest.skip("paraphrase file not generated yet")
    F = json.load(open(M.PARA_FILE))
    paras = M.load_paraphrases()
    assert len(M.jobs(paras)) == 3 * (3 * 48 - F["report"]["n_dropped"])


def test_r1_assert_present_and_exits_on_leak(tmp_path, monkeypatch):
    src = inspect.getsource(M)
    assert src.count("assert_pool_clean(") >= 2 and "exclude_tasks=C.HOLDOUT" in src
    leak = tmp_path / "mine.jsonl"
    s, ut = C.sorted_tasks(C.HOLDOUT)[0]
    leak.write_text(json.dumps({"suite": s, "ut": ut}) + "\n")
    monkeypatch.setattr(C, "rows_path", lambda *a, **k: str(leak))
    with pytest.raises(C.LeakError) as e:
        M.main(["--model", "gpt-4o-mini-2024-07-18", "--dry-run"])
    assert e.value.code == 2


def test_progent_policy_not_from_shared_cache():
    src = inspect.getsource(M.Arms._progent)
    assert "use_cache=False" in src and "user_task_id=None" in src


def test_rows_carry_paraphrase_idx(tmp_path):
    paras = {WS6: {1: "PARA ONE"}}
    R = M.ParaRunner(M.RUNNER, "gpt-4o-mini-2024-07-18", True)
    R.rows, R.steps = str(tmp_path / "r.jsonl"), str(tmp_path / "steps" / "s.jsonl.gz")
    R.extra = M.row_extra(paras, "ts")
    j = {"defense": "none", "suite": WS6[0], "ut": WS6[1], "it": None, "rep": 0, "p": 1}
    R.one(j, lambda j: {"err": None, "utility": True, "final": ""})
    row = C.read_jsonl(R.rows)[0]
    assert row["paraphrase_idx"] == 1 and row["prompt_sha"] == M.sha("PARA ONE")
    assert row["row_id"].endswith("|p1") and row["original_prompt_sha"] == M.sha(prompt(WS6))
    assert R.done_ids() == {row["row_id"]}


# ---------------------------------------------------------------- AMLP per-request inputs from the paraphrase (af4)
class MockClient:
    """Predictor answers a fixed JSON list; embeddings return a fixed vector. Records every input."""

    def __init__(self, pred_out, emb):
        self.seen = []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._chat))
        self.embeddings = types.SimpleNamespace(create=self._emb)
        self.pred_out, self.emb = pred_out, emb

    def _chat(self, model, messages, temperature):
        self.seen.append(("chat", model, messages[0]["content"], temperature))
        msg = types.SimpleNamespace(content=self.pred_out)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    def _emb(self, model, input):
        self.seen.append(("emb", model, input[0]))
        return types.SimpleNamespace(data=[types.SimpleNamespace(embedding=list(self.emb))])


def test_tool_layer_is_predictor_and_nearest_of_paraphrase(tmp_path):
    import m3_envelope as ME
    import envbuild as EB
    s = "banking"
    mined = sorted(t for t in C.NON_HOLDOUT if t[0] == s)[:3]
    tool_of = {mined[0]: "update_password", mined[1]: "schedule_transaction", mined[2]: "update_user_info"}
    rows = [{"model": "m", "suite": a, "ut": b, "it": None, "rep": 0, "err": None, "err_kind": None,
             "trace": [{"f": tool_of[(a, b)], "a": {}}]} for (a, b) in mined]
    pool = ME.Pool("m", rows=rows)
    target = next(t for t in C.sorted_tasks(C.HOLDOUT) if t[0] == s)
    for k_star in mined:
        para = f"PARAPHRASE near {k_star[1]}"
        mc = MockClient('["send_money", "not_a_tool"]', EB.E[k_star] * 3.0)
        A = M.Arms("m", arms=())
        A.ME, A.pool, A.cfg = ME, pool, {"predictor": "gpt-4.1-mini-2025-04-14", "r": 1, "level": "any",
                                         "use_env": "typed"}
        A.pc = M.PredCache(str(tmp_path / f"pc_{k_star[1]}.json"), client=mc)
        M._x.p, M._x.prompt = 0, para
        env = A.envelope(*target)
        assert env.tools == {"send_money", tool_of[k_star]}            # predictor(para) ∪ tools(nearest(emb(para)))
        assert M._x.amlp_meta["amlp_nb_tasks"] == [f"{k_star[0]}|{k_star[1]}"]
        assert env.prompt == para and not (env.mined_tasks & C.HOLDOUT)
        assert para in mc.seen[0][2] and mc.seen[0][1:2] == ("gpt-4.1-mini-2025-04-14",) and mc.seen[0][3] == 0
        assert mc.seen[1] == ("emb", "bge-m3", para)
        d = json.load(open(tmp_path / f"pc_{k_star[1]}.json"))       # cached under the new keys
        assert set(d) == {f"{target[0]}|{target[1]}|p0||gpt-4.1-mini-2025-04-14", f"{target[0]}|{target[1]}|p0||bge-m3"}
        A.envelope(*target)
        assert len(mc.seen) == 2                                          # second build served from the cache


def test_predictor_template_matches_p1b():
    src = open(f"{D}/pilot/p1b_llm_envelope.py").read()
    q = M.predictor_query("banking", "XYZ")
    for frag in ("User request to an assistant:\\n", "Side-effecting tools available:\\n",
                 "List every side-effecting tool the assistant could legitimately need to complete this request. ",
                 "Answer with a JSON list of tool names only."):
        assert frag in src and frag.replace("\\n", "\n") in q
    assert M.parse_prediction("banking", 'x ["send_money","bogus"] y') == (["send_money"], True)
    assert M.parse_prediction("banking", "no list")[1] is False


def test_paraphrase_envelope_equals_build_on_original_inputs():
    """With the ORIGINAL prompt's cached prediction and embedding, the rebuilt tool layer equals m3_envelope.build."""
    import m3_envelope as ME
    import envbuild as EB
    model = "gpt-4o-mini-2024-07-18"
    pool = ME.Pool(model)
    if not pool.ben:
        pytest.skip("no M3 mining pool")
    pred = "gpt-4.1-mini-2025-04-14"
    for t in C.sorted_tasks(C.HOLDOUT)[::6]:
        for r in (1, 2, 1000):
            ref = ME.build(pool, *t, exclude_tasks=C.HOLDOUT, predictor=pred, r=r, level="any")
            base = ME.build(pool, *t, exclude_tasks=C.HOLDOUT, predictor=None, r=0, level="any")
            env, _ = M.paraphrase_envelope(base, pool, *t, EB.PRED[pred][f"{t[0]}|{t[1]}"], EB.E[t], r, EB.E)
            assert env.tools == ref.tools, (t, r)
            assert env.values == ref.values and env.mined_tasks == ref.mined_tasks
