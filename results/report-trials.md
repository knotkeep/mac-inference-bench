## Golden-set pass rate

| model | omlx | mlx-lm | vllm-mlx | vllm-metal | lmstudio-mlx | llama.cpp | ollama | lmstudio |
|---|---|---|---|---|---|---|---|---|
| Gemma 4 26B-A4B | 87% (86%–88%) |  |  |  |  | 88% (88%–89%) |  |  |
| Gemma 4 31B | 99% (99%–99%) |  |  |  |  | 93% (92%–94%) |  |  |
| gpt-oss-120B | 86% (84%–88%) |  |  |  | 90% (86%–93%) | 88% (87%–89%) |  |  |
| Qwen3.8 27B |  | 97% (94%–98%) |  |  | 87% (87%–88%) |  |  |  |
| Qwen3.8-Flash-Next | 92% (90%–93%) |  | 92% (91%–93%) |  |  |  |  |  |

## Decode tokens per second at 4k

| model | omlx | mlx-lm | vllm-mlx | vllm-metal | lmstudio-mlx | llama.cpp | ollama | lmstudio |
|---|---|---|---|---|---|---|---|---|
| Gemma 4 26B-A4B | - |  |  |  |  | - |  |  |
| Gemma 4 31B | - |  |  |  |  | - |  |  |
| gpt-oss-120B | - |  |  |  | - | - |  |  |
| Qwen3.8 27B |  | - |  |  | - |  |  |  |
| Qwen3.8-Flash-Next | - |  | - |  |  |  |  |  |

## Per model

### Gemma 4 26B-A4B (25B total / 3.8B active)

| engine | load s | mem GB | ttft 1k | ttft 16k | prefill 16k | decode 4k | x4 t/s | tools valid / right | golden | errors | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llama.cpp | 3 | 18 | - | - | - | - | - | 100% / 100% | 88% (88%–89%) | 0 |  |
| omlx | 7 | 16 | - | - | - | - | - | 88% / 100% | 87% (86%–88%) | 0 |  |

### Gemma 4 31B (31B dense)

| engine | load s | mem GB | ttft 1k | ttft 16k | prefill 16k | decode 4k | x4 t/s | tools valid / right | golden | errors | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llama.cpp | 3 | 28 | - | - | - | - | - | 100% / 98% | 93% (92%–94%) | 0 |  |
| omlx | 7 | 20 | - | - | - | - | - | 100% / 99% | 99% (99%–99%) | 0 |  |

### gpt-oss-120B (117B total / 5.1B active)

| engine | load s | mem GB | ttft 1k | ttft 16k | prefill 16k | decode 4k | x4 t/s | tools valid / right | golden | errors | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llama.cpp | 3 | 66 | - | - | - | - | - | 100% / 100% | 88% (87%–89%) | 0 |  |
| lmstudio-mlx | 0 | 66 | - | - | - | - | - | 100% / 99% | 90% (86%–93%) | 0 |  |
| omlx | 8 | 65 | - | - | - | - | - | 99% / 100% | 86% (84%–88%) | 0 |  |

### Qwen3.8 27B (27B dense)

| engine | load s | mem GB | ttft 1k | ttft 16k | prefill 16k | decode 4k | x4 t/s | tools valid / right | golden | errors | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lmstudio-mlx | 0 | 18 | - | - | - | - | - | 88% / 100% | 87% (87%–88%) | 0 |  |
| mlx-lm | 3 | 16 | - | - | - | - | - | 100% / 100% | 97% (94%–98%) | 0 |  |

### Qwen3.8-Flash-Next (MoE, oQ4e)

| engine | load s | mem GB | ttft 1k | ttft 16k | prefill 16k | decode 4k | x4 t/s | tools valid / right | golden | errors | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| omlx | 13 | 109 | - | - | - | - | - | 100% / 99% | 92% (90%–93%) | 0 |  |
| vllm-mlx | 10 | 109 | - | - | - | - | - | 100% / 99% | 92% (91%–93%) | 0 |  |

