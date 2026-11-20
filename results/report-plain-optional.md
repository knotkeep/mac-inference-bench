## Golden-set pass rate

| model | omlx | mlx-lm | vllm-mlx | vllm-metal | lmstudio-mlx | llama.cpp | ollama | lmstudio |
|---|---|---|---|---|---|---|---|---|
| Gemma 4 26B-A4B | 87% (86%–88%) | 88% (87%–89%) | 88% (87%–90%) | 88% (87%–89%) | 87% (86%–88%) | 88% (88%–88%) |  |  |
| Gemma 4 31B | 96% (96%–96%) |  |  |  |  |  |  |  |
| GLM-4.7 | 86% | 88% | 84% |  | 86% |  |  |  |
| gpt-oss-120B | 87% (85%–89%) |  |  | 86% (85%–87%) |  |  |  |  |
| Qwen3.8 27B |  |  |  |  | 93% (92%–94%) |  |  |  |

## Decode tokens per second at 4k

| model | omlx | mlx-lm | vllm-mlx | vllm-metal | lmstudio-mlx | llama.cpp | ollama | lmstudio |
|---|---|---|---|---|---|---|---|---|
| Gemma 4 26B-A4B | - | - | - | - | - | - |  |  |
| Gemma 4 31B | - |  |  |  |  |  |  |  |
| GLM-4.7 | - | - | - |  | - |  |  |  |
| gpt-oss-120B | - |  |  | - |  |  |  |  |
| Qwen3.8 27B |  |  |  |  | - |  |  |  |

## Per model

### Gemma 4 26B-A4B (25B total / 3.8B active)

| engine | load s | mem GB | ttft 1k | ttft 16k | prefill 16k | decode 4k | x4 t/s | tools valid / right | golden | errors | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llama.cpp | 3 | 18 | - | - | - | - | - | 100% / 100% | 88% (88%–88%) | 0 |  |
| lmstudio-mlx | 0 | 18 | - | - | - | - | - | 100% / 100% | 87% (86%–88%) | 0 |  |
| mlx-lm | 3 | 15 | - | - | - | - | - | 100% / 100% | 88% (87%–89%) | 0 |  |
| omlx | 7 | 17 | - | - | - | - | - | 100% / 100% | 87% (86%–88%) | 0 |  |
| vllm-metal | 34 | 194 | - | - | - | - | - | 100% / 100% | 88% (87%–89%) | 0 |  |
| vllm-mlx | 6 | 15 | - | - | - | - | - | 100% / 100% | 88% (87%–90%) | 0 |  |

### GLM-4.7 (358B total / 32B active)

| engine | load s | mem GB | ttft 1k | ttft 16k | prefill 16k | decode 4k | x4 t/s | tools valid / right | golden | errors | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lmstudio-mlx | 4 | 157 | - | - | - | - | - | 100% / 98% | 86% | 0 |  |
| mlx-lm | 20 | 155 | - | - | - | - | - | 99% / 96% | 88% | 0 |  |
| omlx | 23 | 164 | - | - | - | - | - | 100% / 98% | 86% | 0 |  |
| vllm-mlx | 27 | 155 | - | - | - | - | - | 88% / 98% | 84% | 0 |  |

### Qwen3.8 27B (27B dense)

| engine | load s | mem GB | ttft 1k | ttft 16k | prefill 16k | decode 4k | x4 t/s | tools valid / right | golden | errors | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lmstudio-mlx | 0 | 18 | - | - | - | - | - | 100% / 100% | 93% (92%–94%) | 0 |  |

### Gemma 4 31B (31B dense)

| engine | load s | mem GB | ttft 1k | ttft 16k | prefill 16k | decode 4k | x4 t/s | tools valid / right | golden | errors | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| omlx | 7 | 20 | - | - | - | - | - | 100% / 99% | 96% (96%–96%) | 0 |  |

### gpt-oss-120B (117B total / 5.1B active)

| engine | load s | mem GB | ttft 1k | ttft 16k | prefill 16k | decode 4k | x4 t/s | tools valid / right | golden | errors | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| omlx | 7 | 64 | - | - | - | - | - | 100% / 100% | 87% (85%–89%) | 0 |  |
| vllm-metal | 24 | 197 | - | - | - | - | - | 100% / 99% | 86% (85%–87%) | 0 |  |

