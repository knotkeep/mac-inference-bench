## Golden-set pass rate

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

## Decode tokens per second at 4k

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

## Per model

### DeepSeek-V4-Flash-0731 (284B total / 13B active)

| engine | load s | mem GB | ttft 1k | ttft 16k | prefill 16k | decode 4k | x4 t/s | tools valid / right | golden | errors | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llama.cpp | 21 | 158 | 1.94 | 26.5 | 690 | 38.7 | 28.0 | 100% / 78% | 95% | 0 |  |
| llama.cpp+spec | 24 | 170 | 2.02 | 27.8 | 656 | 69.8 | 45.6 | 100% / 78% | - | 0 |  |
| lmstudio-mlx | 5 | 144 | 0.57 | 18.3 | 936 | 39.2 | 53.9 | 100% / 100% | 52% | 0 |  |
| lmstudio-mlx+spec | 5 | 126 | 0.69 | 18.3 | 935 | 39.1 | 55.0 | 100% / 100% | - | 0 |  |
| mlx-lm | 24 | 155 | 0.41 | - | - | - | 60.8 | 0% / 0% | 20% | 0 | 878 of ~4k prompt tokens processed, 992 of ~16k prompt tokens processed |
| mlx-lm+spec | 24 | 156 | 0.41 | - | - | - | 60.0 | 0% / 0% | - | 0 | 878 of ~4k prompt tokens processed, 991 of ~16k prompt tokens processed |
| omlx | 24 | 165 | 0.39 | 11.8 | 1480 | 37.2 | 51.0 | 100% / 76% | 88% | 0 |  |
| omlx+spec | 27 | 167 | 0.42 | 12.2 | 1427 | 61.0 | 50.8 | 100% / 76% | - | 0 |  |
| vllm-mlx | 26 | 156 | 0.40 | - | - | - | 56.8 | 0% / 0% | 20% | 0 | 879 of ~4k prompt tokens processed, 991 of ~16k prompt tokens processed |
| vllm-mlx+spec | 26 | 156 | 0.40 | - | - | - | 51.4 | 0% / 0% | - | 0 | 879 of ~4k prompt tokens processed, 990 of ~16k prompt tokens processed |
| lmstudio | unsupported: split GGUF; not imported |  |  |  |  |  |  |  |  |  |  |
| ollama | unsupported: split GGUF; ollama create takes a single file |  |  |  |  |  |  |  |  |  |  |
| vllm-metal | unsupported: architecture not supported by vllm-metal |  |  |  |  |  |  |  |  |  |  |

### Gemma 4 26B-A4B (25B total / 3.8B active)

| engine | load s | mem GB | ttft 1k | ttft 16k | prefill 16k | decode 4k | x4 t/s | tools valid / right | golden | errors | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llama.cpp | 3 | 19 | 0.32 | 4.2 | 4448 | 140.9 | 191.6 | 100% / 100% | 88% | 0 |  |
| llama.cpp+spec | 3 | 20 | 0.37 | 4.6 | 4040 | 167.0 | 187.6 | 100% / 100% | - | 0 |  |
| lmstudio | 0 | 17 | 0.22 | 4.1 | 4396 | 137.6 | 190.3 | 100% / 100% | 88% | 0 |  |
| lmstudio-mlx | 1 | 19 | 0.12 | 2.3 | 7400 | 146.3 | 260.5 | 88% / 100% | 87% | 0 |  |
| mlx-lm | 3 | 15 | 0.19 | 2.5 | 7420 | 120.9 | 233.7 | 88% / 100% | 90% | 0 |  |
| mlx-lm+tpl | 3 | 15 | 0.19 | 2.5 | 7424 | 119.7 | 236.5 | 100% / 100% | 88% | 0 |  |
| ollama | 1 | 7 | 0.20 | 3.4 | 5578 | 138.6 | 217.8 | 100% / 100% | 91% | 0 |  |
| omlx | 8 | 17 | 0.09 | 2.1 | 7946 | 136.9 | 257.2 | 88% / 100% | 86% | 0 |  |
| vllm-metal | 24 | 200 | 0.15 | 2.7 | 6890 | 89.1 | 196.3 | 88% / 100% | 90% | 0 |  |
| vllm-mlx | 6 | 16 | 0.10 | 2.4 | 7688 | 135.7 | 267.7 | 88% / 100% | 90% | 0 |  |

### Gemma 4 31B (31B dense)

| engine | load s | mem GB | ttft 1k | ttft 16k | prefill 16k | decode 4k | x4 t/s | tools valid / right | golden | errors | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llama.cpp | 6 | 29 | 1.50 | 20.7 | 902 | 39.8 | 49.2 | 100% / 98% | 94% | 0 |  |
| llama.cpp+spec | 3 | 17 | 1.52 | 20.9 | 895 | 60.0 | 57.2 | 100% / 98% | - | 0 |  |
| lmstudio | 0 | 26 | 0.87 | 20.0 | 908 | 39.7 | 52.8 | 100% / 99% | 98% | 0 |  |
| lmstudio-mlx | 0 | 21 | 0.46 | 12.8 | 1320 | 43.2 | 73.8 | 100% / 99% | 98% | 0 |  |
| mlx-lm | 3 | 18 | 0.87 | 14.7 | 1274 | 38.7 | 65.6 | 100% / 99% | 98% | 0 |  |
| ollama | 2 | 25 | 1.07 | 17.9 | 1046 | 38.2 | 54.7 | 100% / 98% | 95% | 0 |  |
| omlx | 7 | 20 | 0.41 | 12.6 | 1346 | 40.5 | 64.8 | 100% / 99% | 99% | 0 |  |
| vllm-metal | 44 | 210 | 0.76 | 15.5 | 1204 | 35.4 | 55.9 | 100% / 99% | 98% | 0 |  |
| vllm-mlx | 6 | 20 | 0.41 | 13.6 | 1331 | 41.6 | 73.5 | 100% / 99% | 97% | 0 |  |

### GLM-5.3-Flash (320B total / 18B active)

| engine | load s | mem GB | ttft 1k | ttft 16k | prefill 16k | decode 4k | x4 t/s | tools valid / right | golden | errors | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llama.cpp | 24 | 181 | 1.14 | - | - | - | 0.0 | 0% / 0% | 20% | 6 |  |
| llama.cpp+spec | 27 | 187 | 1.17 | - | - | - | 0.0 | 0% / 0% | - | 6 |  |
| llama.cpp+tpl | 9 | 184 | 1.11 | 28.2 | 563 | 38.5 | 35.8 | 100% / 95% | 90% | 0 |  |
| omlx | 49 | 200 | 0.40 | 7.9 | 1905 | 83.9 | 64.8 | 88% / 98% | 90% | 0 |  |
| omlx+spec | 40 | 166 | 0.40 | 8.0 | 1900 | 77.0 | 69.7 | 88% / 98% | - | 0 |  |
| lmstudio | unsupported: split GGUF; not imported |  |  |  |  |  |  |  |  |  |  |
| lmstudio-mlx | failed: RuntimeError: lms load failed: Error: Model loading was stopped due to |  |  |  |  |  |  |  |  |  |  |
| lmstudio-mlx+spec | failed: RuntimeError: lms load failed: Error: Model loading was stopped due to |  |  |  |  |  |  |  |  |  |  |
| mlx-lm | failed: RuntimeError: server exited with code 3: HTTP 404: {"error": "generati |  |  |  |  |  |  |  |  |  |  |
| mlx-lm+spec | failed: RuntimeError: server exited with code 3: HTTP 404: {"error": "generati |  |  |  |  |  |  |  |  |  |  |
| ollama | unsupported: split GGUF; ollama create takes a single file |  |  |  |  |  |  |  |  |  |  |
| vllm-metal | unsupported: architecture not supported by vllm-metal |  |  |  |  |  |  |  |  |  |  |
| vllm-mlx | failed: RuntimeError: server exited with code 3: ConnectError |  |  |  |  |  |  |  |  |  |  |
| vllm-mlx+spec | failed: RuntimeError: server exited with code 3: ConnectError |  |  |  |  |  |  |  |  |  |  |

### gpt-oss-120B (117B total / 5.1B active)

| engine | load s | mem GB | ttft 1k | ttft 16k | prefill 16k | decode 4k | x4 t/s | tools valid / right | golden | errors | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llama.cpp | 12 | 67 | 0.47 | 6.6 | 2495 | 133.1 | 152.4 | 100% / 100% | 84% | 0 |  |
| llama.cpp+spec | 6 | 56 | 0.50 | 7.2 | 2291 | 145.8 | 98.2 | 100% / 100% | - | 0 |  |
| lmstudio | 0 | 65 | 0.26 | 6.4 | 2527 | 128.5 | 155.2 | 100% / 100% | 93% | 0 |  |
| lmstudio-mlx | 1 | 65 | 0.28 | 2.8 | 5550 | 110.2 | 190.4 | 100% / 100% | 94% | 0 |  |
| mlx-lm | 5 | 64 | 0.30 | 3.1 | 5295 | 96.3 | 186.6 | 0% / 0% | 20% | 0 |  |
| ollama | 3 | 61 | 0.40 | 5.0 | 3278 | 131.4 | 165.3 | 100% / 100% | 85% | 0 |  |
| omlx | 9 | 62 | 0.26 | 2.6 | 6004 | 113.5 | 190.7 | 96% / 100% | 85% | 0 |  |
| vllm-metal | 27 | 207 | 0.25 | 2.4 | 6666 | 88.7 | 177.9 | 99% / 99% | 88% | 0 |  |
| vllm-mlx | 10 | 65 | 0.17 | 2.9 | 5445 | 105.5 | 208.5 | 100% / 100% | 18% | 0 |  |

### MiMo-V2.6-Flash-MOPD (309B total / 15B active)

| engine | load s | mem GB | ttft 1k | ttft 16k | prefill 16k | decode 4k | x4 t/s | tools valid / right | golden | errors | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llama.cpp | 9 | 170 | 0.63 | 18.1 | 1019 | 68.6 | 62.7 | 100% / 100% | 93% | 0 |  |
| llama.cpp+spec | 9 | 158 | 0.65 | 18.7 | 981 | 97.2 | 66.2 | 100% / 100% | - | 0 |  |
| lmstudio | unsupported: split GGUF; not imported |  |  |  |  |  |  |  |  |  |  |
| lmstudio-mlx | failed: RuntimeError: lms import did not register servebench/mimo-v2.6-flash-m |  |  |  |  |  |  |  |  |  |  |
| lmstudio-mlx+spec | failed: RuntimeError: lms import did not register servebench/mimo-v2.6-flash-m |  |  |  |  |  |  |  |  |  |  |
| mlx-lm | failed: RuntimeError: server exited with code 3: HTTP 404: {"error": "generati |  |  |  |  |  |  |  |  |  |  |
| mlx-lm+spec | failed: RuntimeError: server exited with code 3: HTTP 404: {"error": "generati |  |  |  |  |  |  |  |  |  |  |
| ollama | unsupported: split GGUF; ollama create takes a single file |  |  |  |  |  |  |  |  |  |  |
| omlx | failed: TimeoutError: not ready after 1800s: HTTP 409: {"error":{"message":"Mo |  |  |  |  |  |  |  |  |  |  |
| omlx+spec | failed: TimeoutError: not ready after 1800s: HTTP 409: {"error":{"message":"Mo |  |  |  |  |  |  |  |  |  |  |
| vllm-metal | failed: RuntimeError: server exited with code 1: ConnectError |  |  |  |  |  |  |  |  |  |  |
| vllm-mlx | failed: RuntimeError: server exited with code 3: ConnectError |  |  |  |  |  |  |  |  |  |  |
| vllm-mlx+spec | failed: RuntimeError: server exited with code 3: ConnectError |  |  |  |  |  |  |  |  |  |  |

### Qwen3.8 27B (27B dense)

| engine | load s | mem GB | ttft 1k | ttft 16k | prefill 16k | decode 4k | x4 t/s | tools valid / right | golden | errors | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llama.cpp | 6 | 25 | 0.62 | 18.8 | 1048 | 43.0 | 52.0 | 100% / 100% | 90% | 0 |  |
| llama.cpp+spec | 3 | 12 | 0.65 | 20.3 | 973 | 84.2 | 63.2 | 100% / 100% | - | 0 |  |
| lmstudio | 0 | 23 | 0.75 | 18.1 | 1040 | 42.5 | 58.9 | 100% / 99% | 92% | 0 |  |
| lmstudio-mlx | 1 | 18 | 0.42 | 11.1 | 1700 | 53.4 | 81.2 | 88% / 100% | 86% | 0 |  |
| mlx-lm | 4 | 17 | 0.92 | 11.8 | 1668 | 46.8 | 81.6 | 100% / 100% | 97% | 0 |  |
| ollama | 2 | 7 | 0.98 | 16.9 | 1166 | 42.9 | 32.3 | 100% / 100% | 96% | 0 |  |
| omlx | 7 | 18 | 0.32 | 9.7 | 1935 | 44.8 | 74.5 | 100% / 100% | 96% | 0 |  |
| vllm-metal | 46 | 220 | 0.45 | 12.2 | 1609 | 40.5 | 73.1 | 100% / 100% | 94% | 0 |  |
| vllm-mlx | 7 | 17 | 0.36 | 10.7 | 1764 | 38.4 | 85.3 | 100% / 100% | 96% | 0 |  |

### Qwen3.8-Flash-Next (MoE, oQ4e)

| engine | load s | mem GB | ttft 1k | ttft 16k | prefill 16k | decode 4k | x4 t/s | tools valid / right | golden | errors | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llama.cpp | 7 | 76 | 0.57 | 15.3 | 1284 | 55.5 | 38.0 | 100% / 95% | 91% | 0 |  |
| omlx | 17 | 108 | 0.21 | 4.8 | 3951 | 145.8 | 119.1 | 100% / 96% | 90% | 0 |  |
| omlx+spec | 9 | 109 | 0.21 | 4.7 | 3972 | 160.1 | 135.6 | 100% / 96% | - | 0 |  |
| vllm-mlx | 13 | 106 | 0.33 | 8.3 | 2255 | 51.3 | 74.4 | 100% / 96% | 95% | 0 |  |
| vllm-mlx+spec | 10 | 106 | 0.29 | 8.3 | 2259 | 51.4 | 77.0 | 100% / 96% | - | 0 |  |
| llama.cpp+spec | failed: RuntimeError: server exited with code -6: RemoteProtocolError |  |  |  |  |  |  |  |  |  |  |
| lmstudio | unsupported: split GGUF; not imported |  |  |  |  |  |  |  |  |  |  |
| lmstudio-mlx | failed: RuntimeError: lms load failed: python3.11/site-packages/mlx_vlm/models |  |  |  |  |  |  |  |  |  |  |
| lmstudio-mlx+spec | failed: RuntimeError: lms load failed: python3.11/site-packages/mlx_vlm/models |  |  |  |  |  |  |  |  |  |  |
| mlx-lm | failed: TimeoutError: not ready after 1800s: HTTP 404: {"error": "generation t |  |  |  |  |  |  |  |  |  |  |
| ollama | unsupported: split GGUF; ollama create takes a single file |  |  |  |  |  |  |  |  |  |  |
| vllm-metal | failed: RuntimeError: server exited with code 1: ConnectError |  |  |  |  |  |  |  |  |  |  |

### GLM-4.7 (358B total / 32B active)

| engine | load s | mem GB | ttft 1k | ttft 16k | prefill 16k | decode 4k | x4 t/s | tools valid / right | golden | errors | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llama.cpp | 24 | 184 | 0.87 | 42.3 | 375 | 31.9 | 27.7 | 100% / 96% | 89% | 0 |  |
| lmstudio-mlx | 4 | 86 | 0.79 | 19.0 | 792 | 32.4 | 42.4 | 88% / 99% | 85% | 0 |  |
| mlx-lm | 17 | 164 | 0.72 | 21.0 | 755 | 31.0 | 42.2 | 86% / 96% | 81% | 0 |  |
| omlx | 25 | 163 | 0.66 | 18.2 | 828 | 33.7 | 33.2 | 88% / 98% | 83% | 0 |  |
| vllm-metal | 69 | 95 | 0.82 | 15.7 | 1009 | 25.9 | 39.3 | 99% / 96% | 87% | 0 |  |
| vllm-mlx | 22 | 160 | 0.71 | 19.7 | 764 | 32.2 | 45.8 | 86% / 96% | 83% | 0 |  |
| lmstudio | unsupported: split GGUF; not imported |  |  |  |  |  |  |  |  |  |  |
| ollama | unsupported: split GGUF; ollama create takes a single file |  |  |  |  |  |  |  |  |  |  |

### MiniMax-M3.1-Flash (428B total / 23B active)

| engine | load s | mem GB | ttft 1k | ttft 16k | prefill 16k | decode 4k | x4 t/s | tools valid / right | golden | errors | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mlx-lm | 26 | 193 | 1.43 | 18.0 | 971 | 32.1 | 45.8 | 0% / 0% | 20% | 0 |  |
| llama.cpp | unsupported: no GGUF weights |  |  |  |  |  |  |  |  |  |  |
| lmstudio | unsupported: no GGUF weights |  |  |  |  |  |  |  |  |  |  |
| lmstudio-mlx | failed: RuntimeError: lms import did not register servebench/minimax-m3.1-flas |  |  |  |  |  |  |  |  |  |  |
| ollama | unsupported: no GGUF weights |  |  |  |  |  |  |  |  |  |  |
| omlx | failed: TimeoutError: not ready after 1800s: HTTP 409: {"error":{"message":"Mo |  |  |  |  |  |  |  |  |  |  |
| vllm-metal | failed: RuntimeError: server exited with code 1: ConnectError |  |  |  |  |  |  |  |  |  |  |
| vllm-mlx | failed: RuntimeError: server exited with code 3: ConnectError |  |  |  |  |  |  |  |  |  |  |

### Qwen3-VL-235B-A22B (235B total / 22B active)

| engine | load s | mem GB | ttft 1k | ttft 16k | prefill 16k | decode 4k | x4 t/s | tools valid / right | golden | errors | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llama.cpp | 18 | 148 | 1.50 | 33.7 | 558 | 45.7 | 38.2 | 94% / 94% | 83% | 0 |  |
| lmstudio-mlx | 2 | 74 | 0.50 | 15.3 | 1177 | 38.8 | 41.4 | 94% / 92% | 74% | 0 |  |
| mlx-lm | 17 | 132 | 0.52 | 16.5 | 1097 | 23.9 | 44.4 | 94% / 94% | 83% | 0 |  |
| omlx | 20 | 142 | 0.43 | 14.0 | 1286 | 40.4 | 39.8 | 95% / 95% | 84% | 0 |  |
| vllm-metal | 38 | 208 | 1.01 | 15.2 | 1239 | 29.0 | 41.1 | 92% / 91% | 84% | 0 |  |
| vllm-mlx | 20 | 140 | 0.47 | 14.8 | 1221 | 23.9 | 44.9 | 94% / 92% | 83% | 0 |  |
| lmstudio | unsupported: split GGUF; not imported |  |  |  |  |  |  |  |  |  |  |
| ollama | unsupported: split GGUF; ollama create takes a single file |  |  |  |  |  |  |  |  |  |  |

