"""Workloads, tool-call checks, engine support rules and the report; no servers are started."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from servebench.engines import ENGINES, Weights
from servebench.models import MODELS
from servebench.report import best, markdown, matrix
from servebench.workloads import check_tool_call, speed_messages, tool_cases, tools

CASE = {"expected": ["find_providers"]}


def _call(name: str, args: object) -> list[dict]:
    return [{"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}]


def test_tool_call_checks_json_schema_and_choice():
    ok = check_tool_call(
        CASE, _call("find_providers", {"specialty": "cardiology", "zip_code": "84111"})
    )
    assert ok["schema_valid"] and ok["right_tool"] and ok["problem"] is None
    number = check_tool_call(
        CASE, _call("find_providers", {"specialty": "cardiology", "zip_code": 84111})
    )
    assert number["valid_json"] and not number["schema_valid"]
    broken = check_tool_call(CASE, _call("find_providers", '{"specialty": '))
    assert not broken["valid_json"] and "not JSON" in broken["problem"]
    wrong = check_tool_call(CASE, _call("list_claims", {"member_id": "M-SYNTH-000001"}))
    assert wrong["schema_valid"] and not wrong["right_tool"]
    assert check_tool_call(CASE, [])["problem"] == "no tool call"


def test_speed_prompts_grow_and_tool_cases_cover_the_set():
    sizes = [
        len(json.dumps(speed_messages(n))) + len(json.dumps(tools())) for n in (1000, 4000, 16000)
    ]
    assert sizes[0] < sizes[1] < sizes[2]
    tokens = [s // 4 for s in sizes]
    assert 800 <= tokens[0] <= 1600 and 3000 <= tokens[1] <= 5500 and tokens[2] >= 12000
    a, b = speed_messages(1000, "n1"), speed_messages(1000, "n2")
    assert a[0]["content"] != b[0]["content"] and a[1:] == b[1:]
    cases = tool_cases()
    assert len(cases) == 80 and all(c["expected"] for c in cases)


def test_engine_support_rules():
    mlx_only = Weights(mlx_dir=Path("/x"))
    both = Weights(mlx_dir=Path("/x"), gguf=Path("/y.gguf"), draft=Path("/d.gguf"))
    minimax, deepseek = MODELS["minimax-m3.1-flash"], MODELS["deepseek-v4-flash"]
    assert ENGINES["llama.cpp"].unsupported(minimax, mlx_only, False) == "no GGUF weights"
    assert "vllm-metal" in ENGINES["vllm-metal"].unsupported(deepseek, both, False)
    assert ENGINES["omlx"].unsupported(MODELS["gemma-4-31b"], both, True)  # no MTP head
    assert ENGINES["omlx"].unsupported(MODELS["glm-5.3-flash"], both, True) is None
    assert ENGINES["llama.cpp"].unsupported(MODELS["gemma-4-31b"], both, True) is None
    split = Weights(gguf=Path("/m-00001-of-00004.gguf"))
    assert "split" in ENGINES["ollama"].unsupported(MODELS["glm-4.7"], split, False)
    # Split shards named by the pattern are known before anything is downloaded.
    assert "split" in ENGINES["lmstudio"].unsupported(MODELS["mimo-v2.6-flash"], Weights(), False)
    assert ENGINES["ollama"].unsupported(MODELS["gemma-4-31b"], Weights(), False) == (
        "no GGUF weights"
    )


def test_template_variants_and_fetch(monkeypatch, tmp_path):
    from servebench import host

    def tpl_engines(label):
        fmt = MODELS[label].template[2]
        return [n for n, e in ENGINES.items() if e.templates and e.fmt == fmt]

    assert tpl_engines("glm-5.3-flash") == ["llama.cpp"]
    assert tpl_engines("gemma-4-26b-a4b") == ["mlx-lm"]
    cfg = tmp_path / "tokenizer_config.json"
    cfg.write_text(json.dumps({"chat_template": "{{ messages }}"}))
    # A stand-in module, so the test runs without the huggingface_hub package installed.
    hub = types.SimpleNamespace(hf_hub_download=lambda repo, file: str(cfg))
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setattr(host, "BENCH", tmp_path)
    assert host.fetch_template(MODELS["gemma-4-26b-a4b"]).read_text() == "{{ messages }}"


def test_ready_stops_on_a_failed_omlx_load(monkeypatch):
    import httpx
    import pytest

    from servebench import engines

    calls = []

    def post(*a, **k):
        calls.append(1)
        return httpx.Response(409, text="Model 'm' is unavailable after a previous load failure")

    monkeypatch.setattr(engines.httpx, "post", post)
    with pytest.raises(RuntimeError, match="load failure"):
        engines._ready("http://x/v1", "m", "k", 60, [])
    assert len(calls) == 1


def test_report_tables_and_best_engine():
    def run(engine, decode, golden):
        speed = {
            k: {"ttft_s": 1.0, "prefill_tps": 900.0, "decode_tps": decode, "errors": 0}
            for k in ("1000", "4000", "16000")
        }
        return {
            "engine": engine,
            "model": "m",
            "model_name": "M",
            "params": "1B",
            "status": "ok",
            "load_s": 5,
            "memory_gb": 10,
            "speed": speed,
            "concurrency": {"aggregate_tps": 2 * decode},
            "tool_calls": {"schema_valid": 1.0, "right_tool": 0.9, "errors": 0},
            "golden": {"pass_rate": golden},
        }

    runs = [
        run("a", 50.0, 0.9),
        run("b", 80.0, 0.95),
        {"engine": "c", "model": "m", "status": "unsupported", "reason": "no GGUF weights"},
    ]
    md = markdown(runs)
    assert "### M (1B)" in md and "unsupported: no GGUF weights" in md and "100% / 90%" in md
    top = best(runs)[0]
    assert top["fastest_engine"] == "b" and top["golden_best"] == 0.95 and top["engines_ran"] == 2

    grid = [
        {**run("omlx", 50.0, 0.9), "model_name": "M"},
        {"engine": "ollama", "model": "m", "status": "unsupported", "reason": "split GGUF"},
        {"engine": "llama.cpp", "model": "m", "status": "failed", "error": "exit 1"},
    ]
    assert "| M | 90% |" in matrix(grid, "golden") and "| n/a |" in matrix(grid, "golden")
    assert "| M | 50 |" in matrix(grid, "decode") and "| fail |" in matrix(grid, "decode")

    short = run("d", 99.0, 0.2)
    short["speed"]["16000"]["prompt_tokens"] = 991
    assert "991 of ~16k prompt tokens processed" in markdown([short])
    assert best(runs + [short])[0]["fastest_engine"] == "b"


def test_golden_trials_summary_and_cell():
    from servebench.report import golden_cell
    from servebench.run import golden_summary

    rows = [
        {"slice": "happy", "trial": t, "passed": p, "error": None}
        for t, p in [(0, True), (0, True), (1, True), (1, False), (2, False), (2, False)]
    ] + [{"slice": "happy", "trial": 2, "passed": False, "error": "timeout"}]
    g = golden_summary(rows)
    assert g["trials"] == [1.0, 0.5, 0.0] and g["pass_rate"] == 0.5 and g["errors"] == 1
    assert golden_cell(g) == "50% (0%–100%)"
    assert golden_cell({"pass_rate": 0.9, "trials": [0.9]}) == "90%"


def test_fixtures_are_the_benchmarked_requests(monkeypatch):
    from servebench.run import golden_available, tools_sha

    monkeypatch.delenv("SERVEBENCH_PLAIN_OPTIONAL", raising=False)
    # The hashes recorded as tools_sha in results/runs-trials and results/runs-plain-optional.
    assert tools_sha() == "e65df970eed1"
    zip_code = next(t for t in tools() if t["function"]["name"] == "find_providers")
    assert "anyOf" in zip_code["function"]["parameters"]["properties"]["zip_code"]
    monkeypatch.setenv("SERVEBENCH_PLAIN_OPTIONAL", "1")
    assert tools_sha() == "75089d8c73f4"
    plain = next(t for t in tools() if t["function"]["name"] == "find_providers")
    assert plain["function"]["parameters"]["properties"]["zip_code"] == {"type": "string"}
    # Calls are validated against the original definitions whichever form was sent.
    listed = check_tool_call(CASE, _call("find_providers", {"specialty": "x", "zip_code": [84111]}))
    assert not listed["schema_valid"] and "zip_code" in listed["problem"]
    # A boolean sent as text is accepted, as the agent's own validation accepts it.
    text_bool = check_tool_call(
        CASE, _call("find_providers", {"specialty": "x", "in_network_only": "true"})
    )
    assert text_bool["schema_valid"]
    assert isinstance(golden_available(), bool)
