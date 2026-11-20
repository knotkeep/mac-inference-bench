"""Benchmark one OpenAI-compatible endpoint (an engine with one model loaded).

    speed        3 repetitions at about 1k, 4k and 16k prompt tokens, 512 output tokens,
                 tool_choice "none"; every request starts with a unique line so no prompt cache
                 is reused
    concurrency  4 simultaneous 4k-token requests: aggregate output tokens per second
    tool calls   the 80 golden-set scenario requests: did the first turn call a tool, with JSON
                 arguments that match the tool's schema, and the right tool
    quality      the 100-case golden set through the PayerBench agent, `trials` times; needs the
                 `payerbench` package, which is not public, and is skipped without it

Results go to results/runs/<engine>__<model>.json, one file per pair, or to another runs
directory. `tools_sha` identifies the tool definitions the model was sent, so runs made with
different versions of the agent's tools can be told apart.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from servebench.measure import stream_chat
from servebench.workloads import check_tool_call, speed_messages, tool_cases, tools

RUNS = Path("results") / "runs"
SIZES = (1000, 4000, 16000)


def _speed_request(endpoint: dict[str, Any], size: int, max_tokens: int = 512) -> dict[str, Any]:
    msgs = speed_messages(size, nonce=uuid.uuid4().hex)
    kw = dict(api_key=endpoint["api_key"], extra_body=endpoint.get("extra_body"))
    r = stream_chat(
        endpoint["base_url"],
        endpoint["model"],
        msgs,
        tools=tools(),
        tool_choice="none",
        max_tokens=max_tokens,
        **kw,
    )
    if r.get("error") and "HTTP 4" in r["error"]:
        # Engines that reject tool_choice "none" get the same conversation without tools.
        r = stream_chat(endpoint["base_url"], endpoint["model"], msgs, max_tokens=max_tokens, **kw)
        r["tools_sent"] = False
    r.pop("text", None)
    return r


def _median(xs: list[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return statistics.median(vals) if vals else None


def bench(
    engine: str,
    endpoint: dict[str, Any],
    *,
    reps: int = 3,
    tool_limit: int | None = None,
    golden: bool = True,
    trials: int = 1,
    speed: bool = True,
    runs_dir: Path = RUNS,
    meta: dict[str, Any] | None = None,
    log: Any = print,
) -> dict[str, Any]:
    """Run every workload against `endpoint` ({base_url, model, api_key, extra_body}).

    `speed=False` skips the speed and concurrency workloads. `meta` carries what the engine
    adapter knows (versions, load time, memory)."""
    result: dict[str, Any] = {
        "engine": engine,
        "model": endpoint["label"],
        "served_as": endpoint["model"],
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tools_sha": tools_sha(),
        **(meta or {}),
    }
    _speed_request(endpoint, 1000, max_tokens=16)  # warm-up: compile kernels, fill the weights
    if speed:
        _speed(result, endpoint, reps, log)
    _tool_calls(result, endpoint, tool_limit, log)
    if golden and not golden_available():
        log("  golden set skipped: the payerbench package is not installed")
    elif golden:
        result["golden"] = _golden(engine, endpoint, log, trials=trials, runs_dir=runs_dir)
    result["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{engine}__{endpoint['label']}.json"
    path.write_text(json.dumps(result, indent=1) + "\n")
    log(f"  -> {path}")
    return result


def tools_sha() -> str:
    """Short hash of the tool definitions the speed and tool-call workloads send."""
    return hashlib.sha256(json.dumps(tools(), sort_keys=True).encode()).hexdigest()[:12]


def _speed(result: dict[str, Any], endpoint: dict[str, Any], reps: int, log: Any) -> None:
    speed = {}
    for size in SIZES:
        runs = [_speed_request(endpoint, size) for _ in range(reps)]
        speed[str(size)] = {
            "runs": runs,
            "prompt_tokens": _median([r.get("prompt_tokens") for r in runs]),
            "ttft_s": _median([r.get("ttft_s") for r in runs]),
            "prefill_tps": _median([r.get("prefill_tps") for r in runs]),
            # Decode speed only from answers long enough to measure (100+ output tokens).
            "decode_tps": _median(
                [r.get("decode_tps") for r in runs if (r.get("output_tokens") or 0) >= 100]
            ),
            "output_tokens": _median([r.get("output_tokens") for r in runs]),
            "errors": sum(1 for r in runs if r.get("error")),
        }
        log(
            f"  {size:>6} tokens: ttft {speed[str(size)]['ttft_s']}, "
            f"decode {speed[str(size)]['decode_tps']} t/s"
        )
    result["speed"] = speed

    t0 = time.perf_counter()
    with ThreadPoolExecutor(4) as pool:
        par = list(pool.map(lambda _: _speed_request(endpoint, 4000), range(4)))
    wall = time.perf_counter() - t0
    result["concurrency"] = {
        "requests": 4,
        "wall_s": wall,
        "aggregate_tps": sum(r.get("output_tokens") or 0 for r in par) / wall,
        "errors": sum(1 for r in par if r.get("error")),
    }
    log(f"  4 parallel: {result['concurrency']['aggregate_tps']:.1f} t/s aggregate")


def _tool_calls(
    result: dict[str, Any], endpoint: dict[str, Any], tool_limit: int | None, log: Any
) -> None:
    checks = []
    for case in tool_cases(tool_limit):
        r = stream_chat(
            endpoint["base_url"],
            endpoint["model"],
            case["messages"],
            tools=tools(),
            max_tokens=512,
            api_key=endpoint["api_key"],
            extra_body=endpoint.get("extra_body"),
        )
        c = check_tool_call(case, r.get("tool_calls") or [])
        checks.append(
            {"case_id": case["case_id"], "error": r.get("error"), **c, "ttft_s": r.get("ttft_s")}
        )
    n = len(checks)
    result["tool_calls"] = {
        "cases": n,
        **{
            k: sum(c[k] for c in checks) / n
            for k in ("made_call", "valid_json", "schema_valid", "right_tool")
        },
        "errors": sum(1 for c in checks if c["error"]),
        "checks": checks,
    }
    log(
        f"  tool calls: {result['tool_calls']['schema_valid']:.0%} schema-valid, "
        f"{result['tool_calls']['right_tool']:.0%} right tool"
    )


def golden_available() -> bool:
    """The golden set runs through the PayerBench agent, a separate package."""
    return importlib.util.find_spec("payerbench") is not None


def golden_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pass rate over every scored (case, trial), by slice, and per trial."""
    scored = [r for r in rows if not r["error"]]
    by_slice: dict[str, list[bool]] = {}
    by_trial: dict[int, list[bool]] = {}
    for r in scored:
        by_slice.setdefault(r["slice"], []).append(r["passed"])
        by_trial.setdefault(r.get("trial", 0), []).append(r["passed"])
    return {
        "cases": len(rows),
        "errors": len(rows) - len(scored),
        "pass_rate": sum(r["passed"] for r in scored) / len(scored) if scored else None,
        "by_slice": {k: sum(v) / len(v) for k, v in sorted(by_slice.items())},
        "trials": [sum(v) / len(v) for _, v in sorted(by_trial.items())],
    }


def _golden(
    engine: str, endpoint: dict[str, Any], log: Any, trials: int = 1, runs_dir: Path = RUNS
) -> dict[str, Any]:
    """The golden set through the PayerBench agent on this endpoint, `trials` times."""
    from payerbench.gate.collect import collect
    from payerbench.gate.golden import build

    runs_dir.mkdir(parents=True, exist_ok=True)
    out = runs_dir / f"{engine}__{endpoint['label']}.golden.jsonl"
    out.unlink(missing_ok=True)
    keys = (
        "PAYERBENCH_MODEL",
        "OPENAI_BASE_URL",
        "PAYERBENCH_API_KEY",
        "PAYERBENCH_PLAIN_OPTIONAL",
    )
    saved = {k: os.environ.get(k) for k in keys}
    os.environ.update(
        PAYERBENCH_MODEL=endpoint["model"],
        OPENAI_BASE_URL=endpoint["base_url"],
        PAYERBENCH_API_KEY=endpoint["api_key"],
        # The agent binds the same tool definitions the other workloads send.
        PAYERBENCH_PLAIN_OPTIONAL=os.environ.get("SERVEBENCH_PLAIN_OPTIONAL", "0"),
    )
    t0 = time.perf_counter()
    try:
        collect(build(), out, trials=trials)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    g = {
        **golden_summary(rows),
        "minutes": (time.perf_counter() - t0) / 60,
        "records": str(out),
    }
    log(
        f"  golden set: {g['pass_rate']} (trials {[round(t, 2) for t in g['trials']]}) "
        f"in {g['minutes']:.1f} min, {g['errors']} errors"
    )
    return g


def run_model(
    label: str,
    engines: list[str] | None = None,
    *,
    spec: bool = True,
    golden: bool = True,
    reps: int = 3,
    keep_weights: bool = False,
    restore: bool = True,
    template: bool = False,
    trials: int = 1,
    speed: bool = True,
    runs_dir: Path = RUNS,
    log: Any = print,
) -> list[dict[str, Any]]:
    """Benchmark one model on every engine (and speculative variant) that can serve it.

    With `template`, only the `+tpl` variants run: engines of the template's format that accept
    a chat template at start, serving the model's corrected template. A model server named in
    the environment (see host.py) is stopped for the run and restarted afterwards; downloads are
    deleted unless `keep_weights`. Unsupported pairs and failures are recorded in `runs_dir`,
    not raised."""
    from servebench import host
    from servebench.engines import ENGINES
    from servebench.models import MODELS

    model = MODELS[label]
    names = engines or list(ENGINES)
    if template:
        if model.template is None:
            raise ValueError(f"{label} has no corrected chat template")
        names = [n for n in names if ENGINES[n].templates and ENGINES[n].fmt == model.template[2]]
    was_running = host.stop_production()
    results = []
    try:
        formats = tuple(sorted({ENGINES[n].fmt for n in names}))
        weights = host.fetch(model, formats=formats, log=log)
        tpl = host.fetch_template(model) if template else None
        variants = [(n, False) for n in names] + ([(n, True) for n in names] if spec else [])
        for name, is_spec in variants:
            engine = ENGINES[name]
            tag = f"{name}{'+spec' if is_spec else ''}{'+tpl' if tpl else ''}"
            reason = engine.unsupported(model, weights, is_spec)
            if reason:
                if not is_spec:
                    results.append(
                        _record(tag, model, runs_dir, status="unsupported", reason=reason)
                    )
                continue
            log(f"[{label}] {tag}")
            running = None
            try:
                extra = {"template": tpl} if tpl else {}
                running = engine.start(model, weights, spec=is_spec, **extra)
                ep = {"label": model.label, **running.endpoint}
                r = bench(
                    tag,
                    ep,
                    reps=reps,
                    golden=golden and not is_spec,
                    trials=trials,
                    speed=speed,
                    runs_dir=runs_dir,
                    meta={
                        "status": "ok",
                        "model_name": model.name,
                        "params": model.params,
                        **running.meta,
                    },
                    log=log,
                )
                results.append(r)
            except Exception as e:  # noqa: BLE001 - one engine failing must not end the run
                log(f"  failed: {type(e).__name__}: {str(e)[:200]}")
                results.append(
                    _record(
                        tag,
                        model,
                        runs_dir,
                        status="failed",
                        error=f"{type(e).__name__}: {str(e)[:500]}",
                    )
                )
            finally:
                if running is not None:
                    engine.stop(running)
    finally:
        if not keep_weights:
            log(f"  deleted: {host.delete(model)}")
        if was_running and restore:
            host.start_production()
    return results


def _record(engine: str, model: Any, runs_dir: Path = RUNS, **fields: Any) -> dict[str, Any]:
    rec = {
        "engine": engine,
        "model": model.label,
        "model_name": model.name,
        **fields,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / f"{engine}__{model.label}.json").write_text(json.dumps(rec, indent=1) + "\n")
    return rec
