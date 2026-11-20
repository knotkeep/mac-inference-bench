"""Benchmark workloads: the requests a health-plan member-services agent sends its model.

The requests are fixtures exported from the PayerBench agent (a LangGraph agent with 8 tools over
synthetic members, claims and a plan document) at the version that was benchmarked, so this
package needs none of the agent's code. One string differs from the benchmarked requests: the
marker the agent's system prompt carries to detect prompt leaks is replaced by `PB-0000-CANARY`.

speed prompts   the agent's system prompt and 8 tool definitions, grown to about 1k, 4k and
                16k tokens with the plan document and earlier turns carrying tool results, as an
                agent's context grows; the last user turn asks for a long plain-text answer so
                decode speed is measured over hundreds of tokens
tool cases      80 requests whose correct first move is a tool call, with the tools a correct
                agent may call first

SERVEBENCH_PLAIN_OPTIONAL=1 sends the tool definitions with optional parameters as their plain
type (`{"type": "string"}`) instead of `anyOf: [string, null]`. Tool calls are always validated
against the original definitions, with one allowance the agent's own validation (Pydantic) makes:
a boolean sent as the string "true" or "false" counts as valid.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from importlib import resources
from typing import Any


def _fixture(name: str) -> Any:
    return json.loads(resources.files("servebench").joinpath("fixtures", name).read_text())


@lru_cache(maxsize=2)
def _tools(plain_optional: bool) -> list[dict[str, Any]]:
    return _fixture("tools_plain_optional.json" if plain_optional else "tools.json")


def tools() -> list[dict[str, Any]]:
    """OpenAI tool definitions for the agent's 8 tools, as sent to the model."""
    return _tools(os.getenv("SERVEBENCH_PLAIN_OPTIONAL") == "1")


@lru_cache(maxsize=1)
def _schemas() -> dict[str, dict[str, Any]]:
    """Each tool's parameter schema, as the agent defines it."""
    return {t["function"]["name"]: t["function"]["parameters"] for t in _tools(False)}


@lru_cache(maxsize=3)
def _speed(target_tokens: int) -> dict[str, Any]:
    return _fixture(f"speed_{target_tokens}.json")


def speed_messages(target_tokens: int, nonce: str = "") -> list[dict[str, Any]]:
    """The conversation of about `target_tokens` prompt tokens (1000, 4000 or 16000), ending in
    a request for a long plain-text answer. A `nonce` line first makes every request's prefix
    unique, so no engine can answer from a prompt cache filled by an earlier repetition."""
    fx = _speed(target_tokens)
    msgs = [dict(m) for m in fx["messages"]]
    if nonce:
        msgs[0]["content"] = fx["nonce_line"].format(nonce=nonce) + msgs[0]["content"]
    return msgs


def tool_cases(limit: int | None = None) -> list[dict[str, Any]]:
    """Requests whose first move should be a tool call: case_id, category, expected, messages."""
    return _fixture("tool_cases.json")[:limit]


def check_tool_call(case: dict[str, Any], tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    """made_call, valid_json, schema_valid and right_tool for the first tool call."""
    from jsonschema import Draft202012Validator

    out = {
        "made_call": bool(tool_calls),
        "valid_json": False,
        "schema_valid": False,
        "right_tool": False,
        "first_tool": None,
        "problem": None,
    }
    if not tool_calls:
        out["problem"] = "no tool call"
        return out
    first = tool_calls[0]
    name = first.get("name")
    out["first_tool"] = name
    out["arguments"] = (first.get("arguments") or "")[:200]
    out["right_tool"] = name in case["expected"]
    try:
        args = json.loads(first.get("arguments") or "{}")
        out["valid_json"] = isinstance(args, dict)
    except json.JSONDecodeError as e:
        out["problem"] = f"arguments are not JSON: {e.msg}"
        return out
    schemas = _schemas()
    if name not in schemas:
        out["problem"] = f"unknown tool {name!r}"
        return out
    booleans = {
        k for k, v in schemas[name].get("properties", {}).items() if v.get("type") == "boolean"
    }
    args = {
        k: v.lower() == "true"
        if k in booleans and isinstance(v, str) and v.lower() in ("true", "false")
        else v
        for k, v in args.items()
    }
    error = next(iter(Draft202012Validator(schemas[name]).iter_errors(args)), None)
    if error is None:
        out["schema_valid"] = True
    else:
        where = ".".join(str(p) for p in error.absolute_path) or "arguments"
        out["problem"] = f"schema: {where}: {error.message[:100]}"
    return out
