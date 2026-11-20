"""One streamed chat completion against an OpenAI-compatible endpoint, timed.

    ttft_s          request sent to first content or tool-call token
    prefill_tps     prompt tokens / ttft_s (prompt processing speed)
    decode_tps      (output tokens - 1) / (last token time - first token time)

Reasoning text that a server streams in a separate field (`reasoning_content` or `reasoning`)
counts as output for timing and is reported as `reasoning_chars`, since it is generated tokens.
Token counts come from the server's usage report when it sends one (stream_options include_usage);
otherwise output tokens are counted as streamed chunks and prompt tokens are estimated at four
characters per token, and the record says so.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx


def stream_chat(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    max_tokens: int = 512,
    api_key: str = "local",
    extra_body: dict[str, Any] | None = None,
    timeout: float = 900.0,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        **(extra_body or {}),
    }
    if tools:
        body["tools"] = tools
        if tool_choice:
            body["tool_choice"] = tool_choice
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    t0 = time.perf_counter()
    first = last = None
    chunks = 0
    text: list[str] = []
    reasoning = 0
    calls: dict[int, dict[str, Any]] = {}
    usage: dict[str, Any] | None = None
    finish = None
    error = None
    try:
        with httpx.stream("POST", url, json=body, headers=headers, timeout=timeout) as r:
            if r.status_code != 200:
                return {"error": f"HTTP {r.status_code}: {r.read().decode()[:300]}"}
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    got = False
                    if delta.get("content"):
                        text.append(delta["content"])
                        got = True
                    thought = delta.get("reasoning_content") or delta.get("reasoning")
                    if thought:
                        reasoning += len(thought)
                        got = True
                    for tc in delta.get("tool_calls") or []:
                        slot = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                        fn = tc.get("function") or {}
                        slot["name"] += fn.get("name") or ""
                        slot["arguments"] += fn.get("arguments") or ""
                        got = True
                    if got:
                        now = time.perf_counter()
                        first = first or now
                        last = now
                        chunks += 1
                    finish = choice.get("finish_reason") or finish
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        error = f"{type(e).__name__}: {e}"[:300]
    end = time.perf_counter()
    prompt_chars = len(json.dumps(messages)) + (len(json.dumps(tools)) if tools else 0)
    prompt_tokens = (usage or {}).get("prompt_tokens") or prompt_chars // 4
    output_tokens = (usage or {}).get("completion_tokens") or chunks
    ttft = (first - t0) if first else None
    decode = (output_tokens - 1) / (last - first) if first and last and last > first else None
    return {
        "error": error,
        "ttft_s": ttft,
        "total_s": end - t0,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "prefill_tps": prompt_tokens / ttft if ttft else None,
        "decode_tps": decode,
        "usage_reported": usage is not None,
        "finish_reason": finish,
        "text": "".join(text),
        "reasoning_chars": reasoning,
        "tool_calls": [calls[k] for k in sorted(calls)],
    }
