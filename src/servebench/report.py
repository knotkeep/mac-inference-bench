"""Tables from data/servebench/runs/*.json: one per model, plus each model's best engine.

Columns, per engine and variant:
    load         seconds until the first completion succeeded
    mem          memory in use after loading minus before (GB)
    ttft         seconds to the first token at about 1k / 16k prompt tokens
    prefill      prompt tokens per second at about 16k
    decode       output tokens per second at about 4k (median of repetitions with 100+ output
                 tokens; at 1k some models answer in under 100)
    x4           aggregate output tokens per second with 4 simultaneous 4k-token requests
    tools        schema-valid first tool calls / right first tool, of 80 golden-set requests
    golden       pass rate on the 100-case golden set, graded by rules; with several trials,
                 the mean over all trials and the lowest and highest single trial
    notes        speed cells left out because the engine processed under half of the 4k or 16k
                 prompt (its usage report shows the conversation was dropped while rendering)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from servebench.run import RUNS


def load_runs(path: Path = RUNS) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(path.glob("*.json"))]


def _f(x: float | None, fmt: str) -> str:
    return "-" if x is None else format(x, fmt)


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x:.0%}"


def golden_cell(g: dict[str, Any]) -> str:
    trials = g.get("trials") or []
    if len(trials) > 1:
        return f"{_pct(g.get('pass_rate'))} ({_pct(min(trials))}–{_pct(max(trials))})"
    return _pct(g.get("pass_rate"))


def dropped(speed: dict[str, Any]) -> list[str]:
    """Prompt sizes (of 4k and 16k) where the engine processed under half the prompt."""
    return [
        k
        for k in ("4000", "16000")
        if (speed.get(k) or {}).get("prompt_tokens") is not None
        and speed[k]["prompt_tokens"] < int(k) // 2
    ]


def row(r: dict[str, Any]) -> list[str]:
    if r.get("status") not in (None, "ok"):
        why = r.get("reason") or r.get("error") or ""
        return [r["engine"], f"{r['status']}: {why[:70]}"] + [""] * 10
    s = r.get("speed", {})
    cut = dropped(s)
    small = s.get("1000", {})
    mid = {} if "4000" in cut else s.get("4000", {})
    big = {} if "16000" in cut else s.get("16000", {})
    tc = r.get("tool_calls", {})
    g = r.get("golden") or {}
    notes = ", ".join(
        f"{s[k]['prompt_tokens']} of ~{int(k) // 1000}k prompt tokens processed" for k in cut
    )
    return [
        r["engine"],
        _f(r.get("load_s"), ".0f"),
        _f(r.get("memory_gb"), ".0f"),
        _f(small.get("ttft_s"), ".2f"),
        _f(big.get("ttft_s"), ".1f"),
        _f(big.get("prefill_tps"), ".0f"),
        _f(mid.get("decode_tps"), ".1f"),
        _f((r.get("concurrency") or {}).get("aggregate_tps"), ".1f"),
        f"{_pct(tc.get('schema_valid'))} / {_pct(tc.get('right_tool'))}" if tc else "-",
        golden_cell(g),
        str(tc.get("errors", 0) + sum(v.get("errors", 0) for v in s.values())),
        notes,
    ]


HEADER = [
    "engine",
    "load s",
    "mem GB",
    "ttft 1k",
    "ttft 16k",
    "prefill 16k",
    "decode 4k",
    "x4 t/s",
    "tools valid / right",
    "golden",
    "errors",
    "notes",
]


def markdown(runs: list[dict[str, Any]]) -> str:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in runs:
        by_model[r["model"]].append(r)
    out = []
    for model, rs in by_model.items():
        name = next((r.get("model_name") for r in rs if r.get("model_name")), model)
        params = next((r.get("params") for r in rs if r.get("params")), "")
        out += [
            f"### {name} ({params})",
            "",
            "| " + " | ".join(HEADER) + " |",
            "|" + "---|" * len(HEADER),
        ]
        ok = sorted((r for r in rs if r.get("status") in (None, "ok")), key=lambda r: r["engine"])
        rest = sorted(
            (r for r in rs if r.get("status") not in (None, "ok")), key=lambda r: r["engine"]
        )
        out += ["| " + " | ".join(row(r)) + " |" for r in ok + rest]
        out.append("")
    return "\n".join(out) + "\n"


MATRIX_ENGINES = [
    "omlx",
    "mlx-lm",
    "vllm-mlx",
    "vllm-metal",
    "lmstudio-mlx",
    "llama.cpp",
    "ollama",
    "lmstudio",
]


def _cell(r: dict[str, Any] | None, field: str) -> str:
    if r is None:
        return ""
    if r.get("status") == "unsupported":
        return "n/a"
    if r.get("status") not in (None, "ok"):
        return "fail"
    if field == "golden":
        return golden_cell(r.get("golden") or {})
    if "4000" in dropped(r.get("speed", {})):
        return "-"
    return _f(((r.get("speed") or {}).get("4000") or {}).get("decode_tps"), ".0f")


def matrix(runs: list[dict[str, Any]], field: str) -> str:
    """Model × engine table of golden pass rate (`golden`) or decode t/s at 4k (`decode`) for
    the base variants; n/a is unsupported, fail is a failed load or start."""
    cells: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    names: dict[str, str] = {}
    for r in runs:
        if r["engine"] in MATRIX_ENGINES:
            cells[r["model"]][r["engine"]] = r
            names[r["model"]] = r.get("model_name") or names.get(r["model"], r["model"])
    out = [
        "| model | " + " | ".join(MATRIX_ENGINES) + " |",
        "|---|" + "---|" * len(MATRIX_ENGINES),
    ]
    for model in sorted(cells, key=lambda m: names[m].lower()):
        row_ = [_cell(cells[model].get(e), field) for e in MATRIX_ENGINES]
        out.append(f"| {names[model]} | " + " | ".join(row_) + " |")
    return "\n".join(out) + "\n"


def best(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per model, the engine with the highest decode speed at 4k among those that ran."""
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in runs:
        if r.get("status") in (None, "ok") and r.get("speed") and not dropped(r["speed"]):
            by_model[r["model"]].append(r)
    out = []
    for model, rs in by_model.items():
        top = max(rs, key=lambda r: r["speed"]["4000"].get("decode_tps") or 0)
        golden = [r for r in rs if (r.get("golden") or {}).get("pass_rate") is not None]
        out.append(
            {
                "model": top.get("model_name", model),
                "fastest_engine": top["engine"],
                "decode_tps_4k": top["speed"]["4000"].get("decode_tps"),
                "prefill_tps_16k": top["speed"]["16000"].get("prefill_tps"),
                "golden_best": max((r["golden"]["pass_rate"] for r in golden), default=None),
                "engines_ran": len(rs),
            }
        )
    return out
