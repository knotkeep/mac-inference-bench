# mac-inference-bench

Eleven open-weight models on eight local inference engines on one Mac Studio (M5 Ultra, 256 GB),
measured for speed and for how well an agent works on them. October 2026.

Most Mac inference benchmarks report tokens per second. This one also asks whether the engine
serves the model correctly for an agent: every model and engine pair answers the same requests a
health-plan member-services agent sends (system prompt, eight tool definitions, tool results), and
the agent's 100-case golden set runs through each pair.

The full write-up is [docs/WRITEUP.md](docs/WRITEUP.md), also at
[knotkeep.ai/research/inference-engines-on-a-mac](https://knotkeep.ai/research/inference-engines-on-a-mac).

## What it found

- **The engine changed the agent's quality far more than its speed.** The same weights scored
  88% on one engine and 52% on another. Every large gap was in the layer between the weights and
  the API: the chat template that lays out the conversation, or the parser that reads tool calls.
- **A score near 20% means tool use is broken**, not that the model is weak: an agent that never
  calls a tool still passes the refusal and historical cases.
- **MLX engines process long prompts 1.5 to 3.4 times faster than llama.cpp**; generation speed
  is mostly the same.
- **llama.cpp's published draft models gave 1.4 to 2.0 times faster decoding** on five of the
  eight models that have one.
- **llama.cpp loaded all 10 models that have a GGUF**; no MLX engine loaded more than 9 of 11.

Golden-set pass rate (n/a: the engine cannot take that model's weights; fail: it would not load):

| model | omlx | mlx-lm | vllm-mlx | vllm-metal | lmstudio-mlx | llama.cpp | ollama | lmstudio |
|---|---|---|---|---|---|---|---|---|
| DeepSeek-V4-Flash-0731 | 88% | 20% | 20% | n/a | 52% | 95% | n/a | n/a |
| Gemma 4 26B-A4B | 86% | 90% | 90% | 90% | 87% | 88% | 91% | 88% |
| Gemma 4 31B | 99% | 98% | 97% | 98% | 98% | 94% | 95% | 98% |
| GLM-4.7 | 83% | 81% | 83% | 87% | 85% | 89% | n/a | n/a |
| GLM-5.3-Flash | 90% | fail | fail | n/a | fail | 20% | n/a | n/a |
| gpt-oss-120B | 85% | 20% | 18% | 88% | 94% | 84% | 85% | 93% |
| MiMo-V2.6-Flash-MOPD | fail | fail | fail | fail | fail | 93% | n/a | n/a |
| MiniMax-M3.1-Flash | fail | 20% | fail | fail | fail | n/a | n/a | n/a |
| Qwen3-VL-235B-A22B | 84% | 83% | 83% | 84% | 74% | 83% | n/a | n/a |
| Qwen3.8 27B | 96% | 97% | 96% | 94% | 86% | 90% | 96% | 92% |
| Qwen3.8-Flash-Next | 90% | fail | 95% | fail | fail | 91% | n/a | n/a |

The table is each engine as shipped. Served with a corrected chat template, GLM-5.3-Flash on
llama.cpp scores 90%.

Decode tokens per second at a 4k-token context:

| model | omlx | mlx-lm | vllm-mlx | vllm-metal | lmstudio-mlx | llama.cpp | ollama | lmstudio |
|---|---|---|---|---|---|---|---|---|
| DeepSeek-V4-Flash-0731 | 37 | - | - | n/a | 39 | 39 | n/a | n/a |
| Gemma 4 26B-A4B | 137 | 121 | 136 | 89 | 146 | 141 | 139 | 138 |
| Gemma 4 31B | 40 | 39 | 42 | 35 | 43 | 40 | 38 | 40 |
| GLM-4.7 | 34 | 31 | 32 | 26 | 32 | 32 | n/a | n/a |
| GLM-5.3-Flash | 84 | fail | fail | n/a | fail | - | n/a | n/a |
| gpt-oss-120B | 113 | 96 | 106 | 89 | 110 | 133 | 131 | 128 |
| MiMo-V2.6-Flash-MOPD | fail | fail | fail | fail | fail | 69 | n/a | n/a |
| MiniMax-M3.1-Flash | fail | 32 | fail | fail | fail | n/a | n/a | n/a |
| Qwen3-VL-235B-A22B | 40 | 24 | 24 | 29 | 39 | 46 | n/a | n/a |
| Qwen3.8 27B | 45 | 47 | 38 | 41 | 53 | 43 | 43 | 43 |
| Qwen3.8-Flash-Next | 146 | fail | 51 | fail | fail | 56 | n/a | n/a |

Per-model tables with load time, time to first token, prompt processing speed, concurrency and
tool-call validity are in [results/report.md](results/report.md). Three-trial reruns of the
contested pairs are in [results/report-trials.md](results/report-trials.md).

## Run it on your Mac

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and whichever engines you want to
measure, installed and on your PATH: `omlx`, `mlx_lm.server` (and `mlx_vlm.server`), `vllm`
(vllm-metal), `vllm-mlx`, `llama-server`, `ollama`, and LM Studio with its `lms` CLI. An engine
that is not installed is recorded as failed and the run continues.

```bash
uv sync
uv run servebench models                                  # the 11 models and 8 engines
uv run servebench run gemma-4-26b-a4b --engines llama.cpp,mlx-lm --no-golden
uv run servebench report                                  # results/report.md
```

`run` downloads the weights a pair needs from Hugging Face, starts the engine, waits for the
first successful completion, runs the workloads, stops the engine and deletes the weights
(`--keep-weights` keeps them). The large models need most of 256 GB. To free the machine first:

| variable | effect |
|---|---|
| `SERVEBENCH_STOP_LAUNCHD` | label of a launchd agent (another model server) to stop for the run and restart afterwards |
| `SERVEBENCH_KEEP_REPOS` | comma-separated Hugging Face repos a run never deletes |
| `SERVEBENCH_HOME` | scratch directory for engines that need their own model folder (default `~/.servebench`) |

## What is measured

| workload | what it does |
|---|---|
| speed | the agent's conversation at about 1k, 4k and 16k prompt tokens, 3 repetitions each, 512 output tokens: time to first token, prompt tokens per second, output tokens per second. Each request starts with a unique line, so no prompt cache is reused |
| concurrency | four 4k-token requests at once: aggregate output tokens per second |
| first tool call | 80 requests whose correct first step is a tool call: did the model call a tool, is it valid JSON, does it match the tool's schema, is it a tool a correct agent would call first |
| golden set | 100 cases through the PayerBench agent with all defenses on, graded by rules |

The first three run from the fixtures in `src/servebench/fixtures/`, exported from the agent at
the version that was benchmarked (with the system prompt's leak-detection marker replaced by a
placeholder). The golden set needs the PayerBench agent, which is not public:
its results are in the run files, and `servebench run` skips it when the package is absent.

Variants: `+spec` adds the engine's speculative decoding (a draft model on llama.cpp, an MTP head
on MLX engines); `+tpl` (`--template`) serves a corrected chat template where the shipped one
breaks tool use; `--plain-optional` sends optional tool parameters as their plain type instead of
`anyOf: [string, null]`.

## Layout

```
src/servebench/
  models.py     the 11 models: where each format comes from, draft models, parsers
  engines.py    one adapter per engine: start a server, wait until it answers, stop it
  workloads.py  the requests (fixtures) and the tool-call check
  measure.py    one streamed chat completion, timed
  run.py        the workloads against one endpoint; one model across engines
  report.py     tables from the run files
results/
  runs/                  one JSON file per engine and model: every request's timing, every
                         first tool call, golden-set pass rates by slice
  runs-trials/           three golden-set trials on the pairs whose gaps needed retesting
  runs-plain-optional/   the same pairs with plain-type optional parameters
docs/WRITEUP.md          method, findings, caveats
```

## Limits

- One machine, October 2026 versions of projects that release weekly. Several failures recorded
  here are known upstream and may already be fixed.
- MLX and GGUF weights are different quantizations, so a gap across formats mixes engine and
  weights.
- The first pass ran the golden set once per pair; a single run sat up to 4.3 points from the
  mean of three.
- Engines differ in what they do with `tool_choice: none`, which the speed requests send: some
  leave the tool definitions out of the prompt. Prompt processing speed divides by each engine's
  own token count.

## License

MIT. See [LICENSE](LICENSE).
