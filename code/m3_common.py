"""M3 shared runner core (plan.md A3/A4, "M3 runner hard requirements"; ROADMAP §5-§6).

Split (R1), row provenance (R5), step logging (R4), concurrency guard (R6), error policy (R7), resumable job loop,
and the stdlib statistics used by m3_judge / m3_analyze. Deliberately light: no numpy / envbuild import, so the
CaMeL venv (baselines/camel_env) and the Agent-Sentry venv (baselines/agentsentry_env) can import it too.
Shared harness and baselines are imported, never modified.
"""
from __future__ import annotations

import datetime as _dt
import gzip
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from math import comb

D = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(D)   # release root (code/ and data/ are siblings)
TESTBED = D   # harness.py sits next to this file in the release
sys.path.insert(0, D)
sys.path.insert(0, f"{D}/baselines")
import harness  # noqa: E402

SUITES = ("banking", "slack", "travel", "workspace")
SGT = _dt.timezone(_dt.timedelta(hours=8))

# ---------------------------------------------------------------- R1: split from holdout_split.json
_SPLIT = json.load(open(f"{D}/holdout_split.json"))


def _fold(name: str) -> frozenset:
    return frozenset((s, ut) for s, uts in _SPLIT[name].items() for ut in uts)


HOLDOUT, SELECTION, CALIBRATION = _fold("holdout"), _fold("selection"), _fold("calibration")
NON_HOLDOUT = SELECTION | CALIBRATION
assert not (HOLDOUT & SELECTION) and not (HOLDOUT & CALIBRATION) and not (SELECTION & CALIBRATION), "split overlap"


def all_tasks() -> frozenset:
    return frozenset((s, ut) for s in SUITES for ut in harness.suite(s).user_tasks)


def task_order(s: str) -> list[str]:
    return sorted(harness.suite(s).user_tasks, key=lambda x: int(x.split("_")[-1]))


def sorted_tasks(tasks) -> list[tuple[str, str]]:
    return sorted(tasks, key=lambda t: (SUITES.index(t[0]), int(t[1].split("_")[-1])))


class LeakError(SystemExit):
    pass


def assert_pool_clean(pool_tasks, where: str) -> None:
    """R1: mining-pool tasks ∩ HOLDOUT = ∅, else exit non-zero (code 2)."""
    bad = sorted_tasks(set(pool_tasks) & HOLDOUT)
    if bad:
        print(f"[{where}] LEAK: mining pool contains holdout tasks {bad[:5]} (n={len(bad)})", file=sys.stderr, flush=True)
        raise LeakError(2)


# ---------------------------------------------------------------- models and endpoints (R2, R6)
MODELS = {   # model -> (endpoint, hard client cap)
    "qwen3-8b-local": ("vllm-A", 16),
    "llama31-8b-local": ("vllm-B", 8),
    "gpt-4o-mini-2024-07-18": ("hosted-gateway", None),
    "gpt-4.1-mini-2025-04-14": ("hosted-gateway", None),
}
HUB_MODELS = {m for m, (e, _) in MODELS.items() if e == "hosted-gateway"}
VLLM_A_BASE = harness.LOCAL_BASE   # env LLM_API_BASE_QWEN3_8B_LOCAL; its /metrics endpoint feeds vllm_ok16


def served_root(model: str) -> str:
    """R5: local vLLM -> live root from /v1/models; hub -> 'hub:<id>'."""
    if model not in harness.LOCAL_MODELS:
        return f"hub:{model}"
    c = harness.client_for(model)
    return c.models.list().data[0].root


def vllm_ok16(base: str | None = None, timeout: float = 10.0) -> bool:
    """pilot/qwen_guard.sh ok16 in Python: True iff num_requests_waiting == 0 and KV-cache usage < 0.7."""
    import urllib.request
    try:
        root = (base or VLLM_A_BASE or "").rstrip("/")
        root = root[:-3] if root.endswith("/v1") else root
        txt = urllib.request.urlopen(f"{root}/metrics", timeout=timeout).read().decode()
    except Exception:  # noqa: BLE001  unreachable metrics = condition not established
        return False
    return parse_ok16(txt)


def parse_ok16(txt: str) -> bool:
    w = k = None
    for line in txt.splitlines():
        if line.startswith("vllm:num_requests_waiting"):
            w = float(line.split()[-1])
        elif re.match(r"^vllm:(gpu|kv)_cache_usage_perc", line):
            k = float(line.split()[-1])
    return w is not None and k is not None and w == 0 and k < 0.7


def effective_workers(model: str, requested: int, ok16=vllm_ok16) -> int:
    """R6: vllm-A <= 16 only while waiting = 0 and KV < 0.7, else 8; vllm-B <= 8; hub as requested."""
    ep, cap = MODELS[model]
    if ep == "vllm-A":
        if requested <= 8:
            return requested
        return min(requested, 16) if ok16() else 8
    if ep == "vllm-B":
        return min(requested, 8)
    return requested


WRITE_LOCK = threading.Lock()   # held for the row write + gzip step write; the guard takes it before os.execv


def start_guard(model: str, workers: int, on_breach=None, period: float = 60.0, ok16=vllm_ok16):
    """qwen_guard.sh loop as a daemon thread: if vllm-A runs with > 8 clients and the quota condition breaks, the
    process re-execs itself with --workers 8 (the runner is resumable; in-flight rows are lost, logged rows kept)."""
    if MODELS[model][0] != "vllm-A" or workers <= 8:
        return None

    def breach():
        print(f"{now_sgt()} quota condition broken on vllm-A, downgrade to 8 workers", flush=True)
        argv = list(sys.argv)
        if "--workers" in argv:
            argv[argv.index("--workers") + 1] = "8"
        else:
            argv += ["--workers", "8"]
        WRITE_LOCK.acquire()            # wait for any in-flight row + steps write, never release: execv replaces us
        os.execv(sys.executable, [sys.executable, *argv])

    def loop():
        while True:
            time.sleep(period)
            if not ok16():
                (on_breach or breach)()
                return
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------- provenance (R5)
def _git(*a) -> str:
    return subprocess.run(["git", "-C", D, *a], capture_output=True, text=True, check=False).stdout.strip()


def git_head() -> str:
    return _git("rev-parse", "HEAD")


def code_sha() -> str:
    """Hash of the code actually executed (the M3 files are not committed by this worker): m3_*.py + amlp.py +
    envbuild.py contents. Recorded next to git_head."""
    h = hashlib.sha256()
    for f in sorted(os.listdir(D)):
        if (f.startswith("m3_") and f.endswith(".py")) or f in ("amlp.py", "envbuild.py"):
            h.update(f.encode() + b"\0" + open(f"{D}/{f}", "rb").read())
    return h.hexdigest()[:16]


PROGENT_MIN_COMMIT = "05d44ebb"


def port_commit() -> str:
    """R5: git hash of baselines/progent_port.py; asserts >= 05d44ebb and no uncommitted edits to the port."""
    c = _git("log", "-1", "--format=%H", "--", "baselines/progent_port.py")
    assert c, "baselines/progent_port.py has no commit"
    anc = subprocess.run(["git", "-C", D, "merge-base", "--is-ancestor", PROGENT_MIN_COMMIT, c], check=False)
    assert anc.returncode == 0, f"progent_port.py commit {c[:8]} is not {PROGENT_MIN_COMMIT} or later"
    dirty = subprocess.run(["git", "-C", D, "diff", "--quiet", "HEAD", "--", "baselines/progent_port.py"], check=False)
    assert dirty.returncode == 0, "baselines/progent_port.py has uncommitted edits; port_commit would be wrong"
    return c


def now_sgt() -> str:
    return _dt.datetime.now(SGT).strftime("%Y-%m-%d %H:%M:%S SGT")


def row_id(runner: str, model: str, defense: str | None, suite: str, ut: str, it: str | None, rep: int) -> str:
    return f"{runner}|{model}|{defense or 'none'}|{suite}|{ut}|{it or '-'}|{rep}"


def seed_of(rid: str) -> int:
    """Deterministic per-row seed (recorded; the harness does not pass a seed to the model API)."""
    return int(hashlib.sha256(rid.encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------- paths
def data_dir(smoke: bool) -> str:
    return f"{ROOT}/data/m3_smoke" if smoke else f"{ROOT}/data/m3"


def selection_log(smoke: bool) -> str:
    return f"{ROOT}/data/m3_smoke/m3_selection_log.jsonl" if smoke else f"{ROOT}/data/m3_selection_log.jsonl"


def rows_path(smoke: bool, runner: str, model: str, group: str = "") -> str:
    return f"{data_dir(smoke)}/{runner}_{model}{'__' + group if group else ''}.jsonl"


def steps_path(smoke: bool, runner: str, model: str, group: str = "") -> str:
    """R4: data/m3/steps/<runner>_<model>.jsonl.gz (one file per interpreter group for the online runner)."""
    return f"{data_dir(smoke)}/steps/{runner}_{model}{'__' + group if group else ''}.jsonl.gz"


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    op = gzip.open if path.endswith(".gz") else open
    out = []
    with op(path, "rt") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


# ---------------------------------------------------------------- R7: error classes
_CTX = re.compile(r"context length|context_length|maximum context|context window|too many tokens|prompt is too long",
                  re.I)
_API = re.compile(r"^(APIConnectionError|APITimeoutError|RateLimitError|InternalServerError|APIStatusError|APIError|"
                  r"ServiceUnavailableError|AuthenticationError|PermissionDeniedError|NotFoundError|ConflictError|"
                  r"UnprocessableEntityError|BadRequestError|APIResponseValidationError|ModelHTTPError|"
                  r"ConnectError|ReadTimeout|ConnectTimeout|RemoteProtocolError|TimeoutError|ConnectionError)\b")
TERMINAL = {None, "context", "camel_internal"}


def classify_err(err: str | None, defense: str | None) -> str | None:
    """None (ok) | context | api | camel_internal | check | other. Context-length is tested first (it arrives as a
    400 BadRequestError). For CaMeL every non-API, non-context exception is a CaMeL-internal task failure."""
    if err is None:
        return None
    if _CTX.search(err):
        return "context"
    if _API.match(err) or "Connection error" in err or "timed out" in err.lower():
        return "api"
    if err.startswith("check:"):
        return "check"
    if defense in ("camel", "camel_nosecpol"):
        return "camel_internal"
    return "other"


def finalize_failure(r: dict, kind: str) -> dict:
    """Terminal failure row (R7): utility False, security False (attack rows; benign rows have no security)."""
    r["err_raw"], r["err"], r["err_kind"] = r["err"], None, kind
    r["utility"] = False
    r["security"] = False if r.get("it") else None
    return r


# ---------------------------------------------------------------- R4: step capture around any pipeline
_tl = threading.local()


class _Capture:
    """Transparent pipeline wrapper: stores the final message list of the run (also on AbortAgentError)."""

    def __init__(self, p):
        self._p = p

    def query(self, *a, **k):
        from agentdojo.agent_pipeline.errors import AbortAgentError
        try:
            out = self._p.query(*a, **k)
            _tl.msgs = out[3]
            return out
        except AbortAgentError as e:
            _tl.msgs = e.messages
            raise

    def __getattr__(self, n):
        return getattr(self._p, n)


ARM_BUILDERS: dict = {}   # defense -> fn(model, temperature, system_message) -> pipeline (filled by runners)
_installed = [False]


def install_capture() -> None:
    """Patch harness.make_pipeline once (after any other patch, e.g. camel_run's): M3 arms from ARM_BUILDERS, every
    other defense delegated to the previous make_pipeline; the result is wrapped for step capture."""
    if _installed[0]:
        return
    prev = harness.make_pipeline

    def mk(model, temperature=1.0, system_message=harness.SYSTEM_MESSAGE, defense=None):
        p = ARM_BUILDERS[defense](model, temperature, system_message) if defense in ARM_BUILDERS else \
            prev(model, temperature, system_message, defense)
        return _Capture(p)
    harness.make_pipeline = mk
    _installed[0] = True


def _text(m) -> str:
    c = m.get("content")
    if isinstance(c, list):
        return "".join(str(b.get("content", "")) if isinstance(b, dict) else str(b) for b in c)
    return "" if c is None else str(c)


def steps_from_messages(msgs) -> list[dict]:
    """Per tool call, in order: {t (assistant turn), function, args, output (tool message text), error}."""
    if not msgs:
        return []
    res = {m.get("tool_call_id"): m for m in msgs if m.get("role") == "tool" and m.get("tool_call_id") is not None}
    out, t = [], -1
    for m in msgs:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            t += 1
            for tc in m["tool_calls"]:
                r = res.get(tc.id)
                out.append({"t": t, "function": tc.function, "args": dict(tc.args),
                            "output": _text(r) if r is not None else None,
                            "error": (r.get("error") if r is not None else "no tool result")})
    return out


# ---------------------------------------------------------------- job loop: resume (R6), retries (R7), rows (R5)
class Runner:
    """jobs: list of dicts {defense, suite, ut, it, rep}. run_fn(job) -> harness-style row (err set on failure).
    Resumable: a row id with a terminal row (err_kind in TERMINAL) is skipped."""

    API_TRIES, CTX_RERUNS = 3, 2

    def __init__(self, runner: str, model: str, smoke: bool, group: str = "", extra=None):
        self.runner, self.model, self.smoke, self.group = runner, model, smoke, group
        self.rows = rows_path(smoke, runner, model, group)
        self.steps = steps_path(smoke, runner, model, group)
        self.extra = extra or (lambda job: {})
        self.lock = WRITE_LOCK                 # shared with the guard: no re-exec in the middle of a row + steps write
        self.root = None

    def done_ids(self) -> set:
        return {r["row_id"] for r in read_jsonl(self.rows) if r.get("err_kind") in TERMINAL and r.get("err") is None}

    def todo(self, jobs: list[dict]) -> list[dict]:
        d = self.done_ids()
        return [j for j in jobs if self.rid(j) not in d]

    def rid(self, j) -> str:
        return row_id(self.runner, self.model, j["defense"], j["suite"], j["ut"], j["it"], j["rep"])

    def _write(self, r: dict, steps: list) -> None:
        with self.lock:
            os.makedirs(os.path.dirname(self.steps), exist_ok=True)
            with open(self.rows, "a") as f:
                f.write(json.dumps(r, default=str) + "\n")
                f.flush()
            with gzip.open(self.steps, "at") as g:
                g.write(json.dumps({"row_id": r["row_id"], "steps": steps, "final": r.get("final")}, default=str) + "\n")

    def one(self, j: dict, run_fn) -> dict:
        rid = self.rid(j)
        history, ctx_runs, api_tries = [], 0, 0
        while True:
            _tl.msgs = None
            r = run_fn(j)
            steps = steps_from_messages(_tl.msgs)
            kind = classify_err(r.get("err"), j["defense"])
            history.append(kind)
            if kind == "context" and ctx_runs < self.CTX_RERUNS:      # outcome-blind rerun, twice
                ctx_runs += 1
                continue
            if kind in ("api", "other", "check") and api_tries < self.API_TRIES - 1:
                api_tries += 1
                time.sleep(5 * api_tries)
                continue
            break
        if kind == "context":
            r = finalize_failure(r, "context")
        elif kind == "camel_internal":
            r = finalize_failure(r, "camel_internal")
        else:
            r["err_kind"] = kind                                        # api/other/check stay non-terminal
        r.update({"row_id": rid, "runner": self.runner, "rep": j["rep"], "seed": seed_of(rid),
                  "seed_passed_to_api": False, "served_root": self.root, "defense": j["defense"] or None,
                  "git_head": GIT_HEAD, "code_sha": CODE_SHA, "attempts": history, "ts_sgt": now_sgt()})
        r.update(self.extra(j))
        self._write(r, steps)
        return r

    def run(self, jobs: list[dict], run_fn, workers: int) -> list:
        from concurrent.futures import ThreadPoolExecutor
        self.root = served_root(self.model)
        todo = self.todo(jobs)
        print(f"{now_sgt()} {self.runner} {self.model} root={self.root} todo={len(todo)}/{len(jobs)} "
              f"workers={workers}", flush=True)
        n = [0]

        def go(j):
            r = self.one(j, run_fn)
            with self.lock:
                n[0] += 1
                if n[0] % 25 == 0:
                    print(f"{now_sgt()} {self.model} {n[0]}/{len(todo)}", flush=True)
            return r
        with ThreadPoolExecutor(max(1, workers)) as ex:
            out = list(ex.map(go, todo))
        bad = [r for r in out if r.get("err") is not None]
        print(f"{now_sgt()} {self.runner} {self.model} finished, non-terminal errors {len(bad)}", flush=True)
        return out


GIT_HEAD = git_head()
CODE_SHA = code_sha()


# ---------------------------------------------------------------- pairs (TripWire 2-injection rotation) and A2-rev
def rotation_its(s: str, ut: str) -> list[str]:
    """Same rotation as baselines/progent_fidelity.pairs(): index of ut among ALL user tasks of the suite."""
    S = harness.suite(s)
    uts = task_order(s)
    its = sorted(S.injection_tasks, key=lambda x: int(x.split("_")[-1]))
    i = uts.index(ut)
    return list(dict.fromkeys([its[i % len(its)], its[(i + len(its) // 2) % len(its)]]))


def attack_pairs(tasks) -> list[tuple[str, str, str]]:
    return [(s, ut, it) for (s, ut) in sorted_tasks(tasks) for it in rotation_its(s, ut)]


TEXT_ONLY = frozenset((s, it) for s, its in json.load(open(f"{ROOT}/data/pilot/text_only_injection_tasks.json"))
                      ["A2rev_oracle"].items() for it in its)


# ---------------------------------------------------------------- statistics (stdlib; pre-registered in plan.md A3/A4)
BOOT_SEED, BOOT_N = 20261006, 10000


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar: 2 * P(X <= min(b, c)), X ~ Bin(b + c, 1/2), capped at 1."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def cluster_bootstrap(values_by_task: dict, n: int = BOOT_N, seed: int = BOOT_SEED) -> tuple[float, float, float]:
    """Task-cluster bootstrap of a pooled mean. values_by_task: task -> list of per-run values (all models and reps of
    a task move together). Resample tasks with replacement (random.Random(seed).choice, as pilot/m2a_analyze.py),
    pooled mean over all runs of the drawn tasks. Returns (point, 2.5th pct, 97.5th pct)."""
    tasks = sorted(values_by_task, key=str)
    sums = {t: float(sum(values_by_task[t])) for t in tasks}
    cnts = {t: len(values_by_task[t]) for t in tasks}
    tot = sum(cnts.values())
    point = sum(sums.values()) / tot if tot else float("nan")
    rng = random.Random(seed)
    bs = []
    for _ in range(n):
        s = c = 0.0
        for _ in tasks:
            t = rng.choice(tasks)
            s += sums[t]
            c += cnts[t]
        bs.append(s / c if c else 0.0)
    bs.sort()
    return point, bs[int(0.025 * n)], bs[int(0.975 * n) - 1]


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return c - h, c + h


def common_args(ap):
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="print the job count and exit")
    ap.add_argument("--smoke", action="store_true", help="write to data/m3_smoke/ (never the real M3 files)")
    ap.add_argument("--limit", type=int, default=0, help="run only the first N pending jobs (smoke)")
    return ap
