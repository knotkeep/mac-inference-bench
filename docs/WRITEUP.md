# Same model, different engine: eight Mac inference engines measured on an agent

*Eleven open-weight models on eight inference engines on one Mac Studio, measured for speed on
the PayerBench agent's own prompts and for quality on its 100-case golden set. October 2026.*

Also published at [knotkeep.ai/research/inference-engines-on-a-mac](https://knotkeep.ai/research/inference-engines-on-a-mac).
P1 to P5 below are the earlier studies in that series: P2 is the
[judge calibration study](https://knotkeep.ai/research/judge-calibration), P3 the
[red-team suite](https://knotkeep.ai/research/red-teaming-a-health-plan-agent) and P5 the
[statistical CI gate](https://knotkeep.ai/research/statistical-ci-gate). Every run behind the
tables is a JSON file under [`results/`](../results).

## TL;DR

- **The engine changed the agent's quality far more than its speed.** The same weights scored
  88% on one engine and 52% on another (DeepSeek-V4-Flash on oMLX and on LM Studio's MLX
  engine), and gpt-oss-120B scored 84 to 94% on six engines and 18 to 20% on two. None of these
  failures were numerical. Each one was in the layer between the weights and the API: the chat
  template that lays out the conversation, or the parser that reads tool calls back out.
- **A score of 20% means tool use is broken.** An agent that never calls a tool still passes the
  10 refusal cases and the 10 historical-failure cases. Six runs landed at 18 to 20%, and five of
  them passed every speed test without an error.
- **Where the integration works, engines mostly agree, and three trials tell which gaps are
  real.** A single golden-set run landed up to 4.3 points from the mean of three, so the first
  pass's single-run gaps of 5 to 10 points needed retesting. With three trials and the gate's
  paired test, Qwen3.8-Flash-Next's 90-against-95 and most of gpt-oss's 84-to-94 spread were
  noise. Two gaps were real: Gemma 4 31B scored 99.0% on oMLX and 93.3% on llama.cpp (p < 0.001,
  every lost case a denial explanation without the appeal window), and Qwen3.8 27B 96.7% on
  mlx-lm and 87.3% on LM Studio's MLX engine with the same weights folder (p < 0.001).
- **Two template bugs, each fixed by serving a different template file.** Gemma 4 26B's MLX
  download renders an optional `str | None` tool parameter as `type:[]`, so the model sends
  `"zip_code": [84111]`. With the template from the same repo's `tokenizer_config.json`,
  schema-valid first calls went from 88% to 100% and the agent stopped retrying every provider
  search (20 calls over 10 cases became 10); the pass rate did not move (90% and 88%), because
  the tool's validation error let the model correct itself on the next turn. GLM-5.3-Flash's
  GGUF template uses Jinja filters that llama.cpp cannot run (HTTP 400 on any request with tool
  turns, 20% on the golden set); with the MLX repo's template it scored 90%, the same as on
  oMLX.
- **Every schema-invalid tool call in the benchmark was the same field.** All 114 were the
  optional ZIP code, sent as a number or a list, and all were on MLX engines. llama.cpp
  constrains tool-call arguments with a grammar built from the schema, so it cannot send a number
  for a string.
- **The obvious agent-side fix failed the CI gate, and the gate was right.** Describing the
  three optional parameters as plain strings instead of `anyOf: [string, null]` removed the
  schema errors on every engine but one and lifted Qwen3.8 27B on LM Studio MLX from 87.3% to
  93.0% (p = 0.002). It also dropped Gemma 4 31B, the production model, from 99.0% to 96.0%
  (p < 0.001): denial explanations left out the 180-day appeal window 12 times in 30 instead of
  3. The rendered prompts differ in three type tokens. The P5 gate blocked the change on
  its first run; the plain form now ships off by default and is measured as an opt-in.
- **MLX engines process long prompts 1.5 to 3.4 times faster than llama.cpp; generation speed
  is mostly the same.** At a 16k-token prompt the best MLX engine beat llama.cpp on every model
  that ran on both. Decode at a 4k context is bound by memory bandwidth, and on six of the nine
  models that ran on several engines the fastest engine was within 1.4 times the slowest. The
  exceptions are oMLX on its own oQ4e quantizations: Qwen3.8-Flash-Next at 146 tokens/s against
  51 to 56 elsewhere, and GLM-5.3 at 84 against 39 on llama.cpp.
- **llama.cpp's published draft models are the speculative decoding that pays.** They raised
  decode 1.4 to 2.0 times on five of the eight models that have one (DeepSeek 39 → 70 tokens/s
  with DSpark, Qwen3.8 27B 43 → 84 with DFlash). MTP heads in MLX engines did nothing on vllm-mlx
  and LM Studio; oMLX gained 1.6 times on DeepSeek and 1.1 on Qwen3.8-Flash-Next and lost 8% on
  GLM-5.3.
- **llama.cpp loaded all 10 models that have a GGUF; no MLX engine loaded more than 9 of 11.**
  The newest MLX conversions (MiMo-V2.6, MiniMax-M3.1, GLM-5.3) failed to load on most MLX
  engines, and Ollama and LM Studio's GGUF engine cannot import split GGUFs, which every model
  here above about 120B parameters ships as.

## Setup

**Machine.** Mac Studio, Apple M5 Ultra, 256 GB unified memory, macOS 27.0.1. The oMLX server
that normally runs on the machine was stopped for every run, one engine ran at
a time, and the next engine started only after memory returned to where it was before the last
one loaded.

**Engines.** Each serves an OpenAI-compatible API; the harness starts it, waits for the first
successful one-token completion, runs everything, and stops it.

| engine | version | weights | what it is |
|---|---|---|---|
| oMLX | 0.7.2 | MLX | MLX server with its own mixed-precision quantization (oQ4e) and MTP support |
| mlx-lm | 0.32.2 | MLX | Apple's MLX reference server (`mlx_lm.server`); mlx-vlm 0.7.10 for architectures mlx-lm lacks |
| vllm-mlx | 0.6.0 | MLX | community vLLM-style server on MLX with continuous batching |
| vllm-metal | 0.32.0 | MLX | the official vLLM plugin for Apple silicon |
| LM Studio (MLX) | 0.4.26 | MLX | desktop app's MLX engine, driven through its `lms` CLI |
| llama.cpp | b11669 | GGUF | `llama-server` with Metal |
| Ollama | 0.40.8 | GGUF | a separate Ollama instance, model created from the GGUF |
| LM Studio (GGUF) | 0.4.26 | GGUF | the same app's llama.cpp engine |

Shared settings: 4 parallel slots; a 64k context with one shared cache on llama.cpp, 32k on
Ollama and LM Studio, and the server's default on the MLX servers; temperature 0 for the speed
and tool-call tests; thinking turned off where the model has it; and the engine's tool-call
parser for the model family.

**Models.** Weights were chosen per format to sit at the same bit width where one was
published; quantization methods differ between formats, which is one reason scores across
formats are not a pure engine comparison.

| model | parameters | MLX weights | GGUF weights | llama.cpp draft |
|---|---|---|---|---|
| Gemma 4 26B-A4B | 25B, 3.8B active | QAT 4-bit (lmstudio-community) | Q4_0 (ggml-org) | DFlash |
| Gemma 4 31B | 31B dense | 4-bit (mlx-community) | Q4_0 | DFlash |
| Qwen3.8 27B | 27B dense | 4-bit (mlx-community) | Q4_K_M | DFlash |
| Qwen3.8-Flash-Next | MoE | oQ4e with MTP (Jundot) | IQ4_NL | DFlash |
| gpt-oss-120B | 117B, 5.1B active | MXFP4-Q8 (mlx-community) | MXFP4 | EAGLE3 |
| Qwen3-VL-235B-A22B | 235B, 22B active | 4-bit (mlx-community) | UD-Q4_K_XL (unsloth) | none |
| GLM-4.7 | 358B, 32B active | 3-bit | UD-Q3_K_XL (unsloth) | none |
| DeepSeek-V4-Flash-0731 | 284B, 13B active | oQ4e with MTP (Jundot) | MXFP4 | DSpark |
| GLM-5.3-Flash | 320B, 18B active | oQ4e with MTP (Jundot) | Q4_K | MTP |
| MiMo-V2.6-Flash-MOPD | 309B, 15B active | 4-bit with MTP (EigenLabs) | MXFP4 | DFlash |
| MiniMax-M3.1-Flash | 428B, 23B active | 3-bit text-only (mlx-community) | none published | none |

GLM-4.7 and MiniMax run at 3 bits because their 4-bit weights leave too little of 256 GB for
the KV cache.

**What is measured.** Everything is built from the PayerBench agent (P2), not from a
generic benchmark prompt.

- *Speed.* The agent's system prompt and its eight tool definitions, grown to about 1k, 4k and
  16k tokens with the plan document and earlier turns carrying real tool results, as an agent's
  context grows. The last turn asks for a long plain-text summary so generation is timed over
  hundreds of tokens. Each request starts with a unique line so no engine can reuse a cached
  prefix, and `tool_choice: none` keeps the model from answering with a tool call. Three
  repetitions per size; decode speed only from answers of 100 or more tokens.
- *Concurrency.* Four 4k-token requests at once; aggregate output tokens per second.
- *First tool call.* The 80 golden-set requests whose correct first step is a tool call, one
  completion each: did the model call a tool, is the call valid JSON, does it match the tool's
  schema, and is it a tool a correct agent would call first.
- *Golden set.* The full 100-case golden set from P5 (60 typical requests, 20 edge
  cases, 10 that must be refused, 10 past failures), run through the real agent with all
  defenses on and only the model endpoint swapped, graded by the P2 and P3 rules. The agent
  samples at temperature 0.2, so this is one draw.

## Quality

Golden-set pass rate. n/a: the engine cannot take that model's weights; fail: it would not load.

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

Read the 20% cells as "no tool calls". In each, the happy and edge slices scored 0 and the
refusal and historical slices scored 100%. The table is each engine as shipped; with a corrected
chat template (below) GLM-5.3 on llama.cpp scored 90% and Gemma 4 26B on mlx-lm 88%.

### Why the same weights score differently

Every large gap traced to one of five integration failures. None needed a change to the weights.

1. **The engine cannot read the model's tool-call format.** gpt-oss writes tool calls in its
   "harmony" channel format; mlx-lm and vllm-mlx do not parse it, so the agent never sees a call
   (20% and 18%). MiniMax-M3.1 writes XML-style calls; mlx-lm's parser expects JSON and logs
   "Failed to parse tool call" for each one (20%).
2. **The download has no chat template.** oMLX's own DeepSeek-V4 repo ships without one, since
   oMLX has a built-in formatter for the model. mlx-lm and vllm-mlx fall back to a generic
   layout that drops most of the conversation: for the 16k-token request they report processing
   991 tokens. No tool calls, 20%, and their speed numbers at 4k and 16k are not measurements of
   the real prompt (the report leaves them out).
3. **The template lays out tool results wrongly.** LM Studio's MLX engine calls the right first
   tool on DeepSeek 100% of the time, the best of any engine, and then fails 41 cases that oMLX
   passes. The tool result does not reach the model as a tool result. Asked about a deductible
   whose lookup returned $3,071 of $4,000, it answered "You've met $1,250 of your $3,000
   deductible" followed by the model's internal markers (`</think>`, `</｜DSML｜>`).
4. **The engine cannot run the template.** GLM-5.3-Flash's GGUF template uses `unique` and
   `sort(attribute=...)` filters. llama.cpp's Jinja engine fails on it with "Unable to generate
   parser for this template", so every request with tool turns gets HTTP 400 and only short
   plain prompts work. Reproduced with a 135M-parameter model: the GGUF template fails the same
   way, and the template from the MLX repo, which does the same reordering with loops, loads.
   Served with that template, GLM-5.3 on llama.cpp made 100% schema-valid first calls and scored
   90% on the golden set, from 20% and the same as on oMLX.
5. **The template describes a parameter wrongly.** The agent's `find_providers(zip_code: str |
   None)` becomes the JSON schema `{"anyOf": [{"type": "string"}, {"type": "null"}]}`, which has
   no top-level type. Gemma 4 26B's MLX repo has two templates. `chat_template.jinja`, which MLX
   engines load first, renders the parameter as `type:[]`, and the model sends a list. The one
   embedded in `tokenizer_config.json` is byte-identical to the GGUF's (Google's canonical
   template of 2026-07-09) and renders an empty string type, and the model sends text. Served
   with the second on mlx-lm, schema-valid first calls went from 88% to 100%. The golden set did
   not move (90% and 88%): in the base run the tool rejected every list, the model resent the
   ZIP code as text, and each provider search cost two calls instead of one.

   Every one of the 114 schema-invalid tool calls in the benchmark was this ZIP code: Gemma 26B
   (all five MLX engines), GLM-4.7 (four MLX engines), GLM-5.3 (oMLX), Qwen3.8 27B (LM Studio MLX
   only) and four calls from gpt-oss sent it as a number or a list. None came from a GGUF engine.
   llama.cpp builds a grammar from each tool's JSON schema and constrains the arguments once a
   call starts, so it cannot emit a number where the schema says string; LM Studio's GGUF engine
   is llama.cpp, and Ollama also produced none. The agent-side fix, describing optional
   parameters with their plain type, is in the follow-up below: it worked, and it was blocked.

Speculative variants made the same first tool call, with the same arguments, as their base runs
in 1,118 of 1,120 cases; the other two differ only in the wording of a search query. Drafting
changes speed, not output.

## Speed

Decode tokens per second at a 4k-token context (base variants, median of three).

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

A dash is a run whose 4k requests failed or were truncated: GLM-5.3 on llama.cpp with its
shipped template (39 tokens/s with the corrected one), and DeepSeek on mlx-lm and vllm-mlx.

**Long prompts.** Prompt processing at about 16k tokens, best MLX engine against llama.cpp:

| model | llama.cpp t/s | best MLX t/s | engine | ratio |
|---|---|---|---|---|
| Gemma 4 31B | 902 | 1346 | oMLX | 1.5× |
| Gemma 4 26B-A4B | 4448 | 7946 | oMLX | 1.8× |
| Qwen3.8 27B | 1048 | 1935 | oMLX | 1.8× |
| DeepSeek-V4-Flash | 690 | 1480 | oMLX | 2.1× |
| Qwen3-VL-235B | 558 | 1286 | oMLX | 2.3× |
| gpt-oss-120B | 2495 | 6666 | vllm-metal | 2.7× |
| GLM-4.7 | 375 | 1009 | vllm-metal | 2.7× |
| Qwen3.8-Flash-Next | 1284 | 3951 | oMLX | 3.1× |
| GLM-5.3-Flash | 563 (corrected template) | 1905 | oMLX | 3.4× |

For an agent this is the number that matters most: every turn resends the system prompt, the
tools and the conversation so far. At 16k tokens Qwen3.8 27B starts answering in 9.7 seconds on
oMLX and 18.8 on llama.cpp (18.8k and 19.7k prompt tokens as each engine counts them; see the
caveats).

**Generation.** Decode is bound by memory bandwidth, so engines land close together: on the
dense 27 to 31B models every engine was between 35 and 53 tokens/s, and on six of the nine
models that ran on several engines the fastest was within 1.4 times the slowest. vllm-metal was
the slowest on most models (89 against 137 to 146 on Gemma 26B), mlx-vlm-served models were slow
on mlx-lm and vllm-mlx (Qwen3-VL at 24 against 39 to 46), and oMLX ran its own oQ4e
quantizations far faster than any other engine: Qwen3.8-Flash-Next 2.6 times and GLM-5.3 2.2
times (84 against 39 on llama.cpp), the largest engine effects in the benchmark.

**Concurrency.** With four simultaneous requests the batching MLX servers led: Gemma 26B
aggregated 257 to 268 tokens/s on oMLX, LM Studio MLX and vllm-mlx against 192 on llama.cpp, and
Qwen3.8 27B 81 to 85 against 52 on llama.cpp and 32 on Ollama.

**Speculative decoding.** Decode at 4k, base → with drafting:

| engine | model | draft | base | with draft | gain |
|---|---|---|---|---|---|
| llama.cpp | Qwen3.8 27B | DFlash | 43.0 | 84.2 | 1.96× |
| llama.cpp | DeepSeek-V4-Flash | DSpark | 38.7 | 69.8 | 1.80× |
| llama.cpp | Gemma 4 31B | DFlash | 39.8 | 60.0 | 1.51× |
| llama.cpp | MiMo-V2.6-Flash | DFlash | 68.6 | 97.2 | 1.42× |
| llama.cpp | Gemma 4 26B-A4B | DFlash | 140.9 | 167.0 | 1.19× |
| llama.cpp | gpt-oss-120B | EAGLE3 | 133.1 | 145.8 | 1.10× |
| llama.cpp | GLM-5.3-Flash | MTP | 39.1 (1k) | 54.2 (1k) | 1.39× |
| llama.cpp | Qwen3.8-Flash-Next | DFlash | 55.5 | crashed (abort at start) | |
| oMLX | DeepSeek-V4-Flash | MTP | 37.2 | 61.0 | 1.64× |
| oMLX | Qwen3.8-Flash-Next | MTP | 145.8 | 160.1 | 1.10× |
| oMLX | GLM-5.3-Flash | MTP | 83.9 | 77.0 | 0.92× |
| vllm-mlx | Qwen3.8-Flash-Next | MTP | 51.3 | 51.4 | 1.00× |
| LM Studio MLX | DeepSeek-V4-Flash | MTP | 39.2 | 39.1 | 1.00× |

GLM-5.3's draft run used the shipped template, so only the 1k prompt, which has no tool turns,
could be timed. Drafting helps a single stream and can hurt a batch: gpt-oss on llama.cpp with
EAGLE3 aggregated 98 tokens/s across four requests against 152 without it.

**Load and memory.** With the weights on local disk, every engine but vllm-metal answered its
first request within 50 seconds (LM Studio's figure starts after its `lms load` step returns, so
it reads near zero); vllm-metal took 24 to 69 seconds and reserved 200 to 220 GB
whatever the model's size (Gemma 26B's weights are 15 GB), because it preallocates a KV-cache
pool as a fraction of memory. The other engines' memory figures are not comparable with each
other: the harness measures system memory in use, and engines that memory-map their weights
(Ollama most of all) show only part of them.

## Coverage

What each engine could run, of the 11 models:

| engine | ran | would not load | cannot take the weights |
|---|---|---|---|
| llama.cpp | 10 | 0 | 1 (no GGUF for MiniMax) |
| oMLX | 9 | 2 (MiMo, MiniMax: weight names the loader does not expect) | 0 |
| mlx-lm (and mlx-vlm) | 8 | 3 (MiMo, GLM-5.3, Qwen3.8-Flash-Next) | 0 |
| vllm-mlx | 8 | 3 (MiMo, GLM-5.3, MiniMax) | 0 |
| LM Studio MLX | 7 | 4 (its memory guard refused GLM-5.3; MiMo and MiniMax never appeared in its model list; its bundled mlx-vlm failed on Qwen3.8-Flash-Next) | 0 |
| vllm-metal | 6 | 3 (MiMo, MiniMax, Qwen3.8-Flash-Next) | 2 (architecture not supported) |
| Ollama | 4 | 0 | 7 (split GGUF or no GGUF) |
| LM Studio GGUF | 4 | 0 | 7 (split GGUF or no GGUF) |

"Ran" counts a server that loaded and answered, including the runs that then scored 20%.

Three patterns. The newest MLX conversions fail in mlx-vlm, vllm-mlx and vllm-metal for format
reasons (parameters the model class does not have, shape mismatches, an image processor the
text model lacks, config validation) while their GGUFs already work in llama.cpp. Models whose
only MLX weights are in one server's own quantization are tied to that server among MLX
engines: GLM-5.3 in oMLX's oQ4e loaded on no other MLX engine. And split GGUFs, which Hugging
Face's 50 GB file limit forces on large models, shut Ollama and LM Studio out of every model
here larger than about 120B parameters.

## Follow-up: three trials, and the fix the gate blocked

*October 29, 2026. Three golden-set trials per pair, no speed tests. Comparisons use the P5
gate's statistics: a
within-case randomization test on each case's pass rate across trials and a paired bootstrap
interval.*

### Which gaps were noise

| model | engines (first, second) | first pass, 1 trial | three trials (each trial) | second minus first, 95% CI | p |
|---|---|---|---|---|---|
| Gemma 4 31B | oMLX, llama.cpp | 99, 94 | 99.0 (99, 99, 99), 93.3 (94, 94, 92) | −5.7 [−10.3, −1.7] | < 0.001 |
| Qwen3.8 27B | mlx-lm, LM Studio MLX | 97, 86 | 96.7 (98, 98, 94), 87.3 (87, 87, 88) | −9.3 [−14.7, −4.7] | < 0.001 |
| gpt-oss-120B | LM Studio MLX, oMLX | 94, 85 | 90.0 (91, 86, 93), 86.3 (88, 84, 87) | −3.7 [−7.7, 0.0] | 0.035 |
| gpt-oss-120B | LM Studio MLX, llama.cpp | 94, 84 | 90.0, 88.3 (89, 89, 87) | −1.7 [−5.7, +2.3] | 0.22 |
| Qwen3.8-Flash-Next | oMLX, vllm-mlx | 90, 95 | 92.0 (90, 93, 93), 91.7 (93, 91, 91) | −0.3 [−3.0, +2.0] | 0.50 |
| Gemma 4 26B-A4B | llama.cpp, oMLX | 88, 86 | 88.3 (88, 89, 88), 87.3 (88, 88, 86) | −1.0 [−3.3, +0.7] | 0.17 |

A pair's three trials spread 0 to 7 points, 2 at the median, and a first-pass single run sat
up to 4.3 points from its pair's three-trial mean. The 90-against-95 on Qwen3.8-Flash-Next and
the 10-point spread on gpt-oss were mostly sampling. Two gaps held:

- **Gemma 4 31B on llama.cpp** lost only denial explanations: 10 of 30 passed against 27 of 30
  on oMLX. Every failure skipped `search_plan_document`, the only place the 180-day appeal
  window is written, and every pass called it. The weights differ (MLX 4-bit against GGUF
  Q4_0), so this is the engine or the quantization, not separable here.
- **Qwen3.8 27B on LM Studio MLX** runs the same weights folder as mlx-lm, so the gap is the
  engine. Provider searches passed 11 of 30 against 25 of 30: LM Studio's rendering of the
  `anyOf` ZIP code made the model send it as a number, and the failed call led to answers that
  named no provider or the wrong one.

### The plain-type fix

| model | engine | schema-valid first calls | golden set, three trials | change, 95% CI | p |
|---|---|---|---|---|---|
| Gemma 4 31B | oMLX | 100% → 100% | 99.0 → 96.0 (96, 96, 96) | −3.0 [−7.0, 0.0] | < 0.001 |
| Qwen3.8 27B | LM Studio MLX | 88% → 100% | 87.3 → 93.0 (92, 93, 94) | +5.7 [+1.3, +10.3] | 0.002 |
| Gemma 4 26B-A4B | oMLX | 88% → 100% | 87.3 → 87.0 | −0.3 [−1.0, 0.0] | 0.50 |
| Gemma 4 26B-A4B | llama.cpp | 100% → 100% | 88.3 → 88.0 | −0.3 [−2.0, +1.0] | 0.50 |
| gpt-oss-120B | oMLX | 99% → 100% | 86.3 → 86.7 | +0.3 [−3.0, +3.7] | 0.66 |

`anyOf` on the left of each arrow, plain types on the right. With plain types, Gemma 4 26B also
made 100% schema-valid first calls on mlx-lm, vllm-mlx, vllm-metal and LM Studio MLX (88% in the
first pass) and scored 87 to 88% on all six engines. GLM-4.7, one trial each, sent no invalid
arguments on oMLX, LM Studio MLX and mlx-lm (86 to 88% schema-valid in the first pass; mlx-lm's
one miss was a case with no tool call), but on vllm-mlx it still sent `84111` as a number in all
10 provider searches: GLM writes each argument as bare text between tags, and vllm-mlx's parser
evidently turns it into a number whatever the schema declares.

The change failed the gate the first time it was pushed: 99.0% to 96.0% against the nightly
pool (p = 0.014), three denial explanations without the appeal window, each of which had passed
in every baseline run. An A/B on the 10 denial cases, five trials each against Gemma 4 31B,
confirmed it: 45 of 50 passed with the `anyOf` definitions and 30 of 50 with plain types. The
prompts Gemma sees differ in three tokens, `zip_code`, `address` and `phone` going from an empty
type to `STRING`, and nothing about denials. Yet the model looked up the plan document in 18 of
30 denial explanations instead of 27, and every one that did not look it up left out the window.

Two conclusions. The fix is model-specific: +5.7 points for one pair, −3.0 for the production
model, nothing for the rest, so it ships off by default. And the agent's denial explanations depend on the model choosing a second tool call, which
three unrelated tokens, a different engine or a different quantization can each talk it out of.
Returning the appeal rule with every denied claim removes that dependency. That change was
made on November 9 and passed the gate: the agent stopped making the lookup and stated the window
in 45 of 50 denial explanations, and with one more sentence in the prompt for a member with two
denied claims, the golden set has scored 100 of 100 on each of eight nights since. The engine and
quantization comparisons above predate the change.

## What we would run

For this agent on this machine:

- **Highest quality:** Gemma 4 31B on oMLX, 99% in each of three trials at 40 tokens/s with the
  fastest long-prompt handling. This is what production oMLX already serves. The same model on
  llama.cpp scored 93% over three trials.
- **Fast and good:** Qwen3.8-Flash-Next on oMLX, 90% at 146 tokens/s (160 with MTP) and a
  16k-token prompt in under 5 seconds. Qwen3.8 27B on llama.cpp with its DFlash draft, 90% at 84
  tokens/s, if a GGUF stack is preferred.
- **The large models** (DeepSeek-V4-Flash, MiMo-V2.6, GLM-5.3) reach 88 to 95% but need 160 to
  200 GB and decode at 37 to 84 tokens/s. DeepSeek on llama.cpp with DSpark, 95% at 70 tokens/s,
  is the best of them.
- **Whatever the choice,** run the agent's own evaluation on the exact engine, weights and
  template before switching. The failures above were invisible to a speed benchmark and would
  have reached members as wrong deductible amounts.

## Caveats

- The first pass ran the golden set once per pair at the agent's temperature of 0.2; a single
  run sat up to 4.3 points from the mean of three. The follow-up retested the pairs whose gaps
  the conclusions rest on, with three trials each; the rest of the quality table is one trial.
- MLX and GGUF weights are different quantizations, so a cross-format gap mixes engine and
  weights. Same-format comparisons (oMLX against LM Studio MLX on the same folder) do not.
- Engines do not process the same prompt for the same speed request. The speed tests send
  `tool_choice: none`, and oMLX then leaves the tool definitions out: it counts 441 prompt
  tokens for the 1k request with that setting, 437 with no tools at all and 1,231 with
  `tool_choice: auto`. LM Studio and vllm-mlx report the same smaller counts; llama.cpp, mlx-lm,
  vllm-metal and Ollama report counts that include the tools. The gap is about 790 tokens, over
  half of the 1k prompt and about 5% of the 16k one.
  Prefill speed divides by each engine's own count, so it stays comparable; time to first token
  at 1k favors the engines that drop the tools.
- Grading is the rules from P2 and P3; the judge cascade from P5 was not run.
- These are October 2026 versions of projects that release weekly. Some failures here are known
  upstream (gpt-oss tool calls in mlx-lm, mlx-lm issue 875; MiniMax-M3 support in llama.cpp) and
  any of them may be fixed by the time this is read.

## Reproduce

```bash
uv sync
uv run servebench models                           # the roster and where weights come from
uv run servebench run gemma-4-31b --no-golden      # every engine, with speculative variants
uv run servebench run glm-5.3-flash --engines llama.cpp --template --no-golden   # corrected template
uv run servebench run gemma-4-31b --engines omlx --plain-optional --no-golden    # plain-type parameters
uv run servebench report                           # results/report.md
```

A run downloads only the formats its engines need and deletes them afterwards. The speed,
concurrency and first-tool-call workloads run from the fixtures in this repository. The golden
set runs through the PayerBench agent, which is not public; its results are in the run files.
