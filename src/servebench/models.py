"""The benchmark's models: where each format comes from and what each engine needs to serve it.

Repos and file sizes were checked against the Hugging Face API on 2026-10-23. MLX weights come
from one repo per model; GGUF weights are one quantization of one repo, chosen to sit near the
MLX bit width. `draft` is the speculative-decoding file llama.cpp pairs with the model, of the kind
the GGUF publisher ships (dflash, mtp, eagle3 or dspark). Parsers are the tool-call parsers for
vLLM (vllm-metal) and vllm-mlx. `template` is a corrected chat template for the `+tpl` variants,
used where the template shipped with one format breaks tool use on an engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Model:
    label: str
    name: str
    params: str
    mlx: str | None
    gguf: tuple[str, str] | None  # (repo, file or folder/ prefix)
    draft: tuple[str, str, str] | None = None  # (repo, file, llama.cpp --spec-type)
    vllm_parser: str | None = None
    vllm_mlx_parser: str | None = None
    mlx_mtp: bool = False  # MLX weights carry an MTP head (oMLX, mlx-vlm, vllm-mlx can use it)
    mlx_lm_arch: bool = True  # mlx-lm supports the architecture; otherwise mlx-vlm serves it
    vllm_metal: bool = True  # architecture supported by vllm-metal
    # (repo, file, format): chat template for `+tpl` runs on engines of that format; a file of
    # tokenizer_config.json means its chat_template key
    template: tuple[str, str, str] | None = None
    notes: list[str] = field(default_factory=list)


MODELS: dict[str, Model] = {
    m.label: m
    for m in [
        Model(
            "gemma-4-26b-a4b",
            "Gemma 4 26B-A4B",
            "25B total / 3.8B active",
            mlx="lmstudio-community/gemma-4-26B-A4B-it-QAT-MLX-4bit",
            gguf=("ggml-org/gemma-4-26B-A4B-it-GGUF", "gemma-4-26B-A4B-it-Q4_0.gguf"),
            draft=(
                "ggml-org/gemma-4-26B-A4B-it-GGUF",
                "dflash-gemma-4-26B-A4B-it-Q8_0.gguf",
                "draft-dflash",
            ),
            vllm_parser="gemma4",
            vllm_mlx_parser="gemma4",
            template=(
                "lmstudio-community/gemma-4-26B-A4B-it-QAT-MLX-4bit",
                "tokenizer_config.json",
                "mlx",
            ),
            notes=[
                "the MLX repo's chat_template.jinja renders a parameter without a top-level type "
                "(anyOf string/null) as type:[]; its tokenizer_config.json template, identical to "
                "the GGUF's, renders it as an empty string type",
            ],
        ),
        Model(
            "gemma-4-31b",
            "Gemma 4 31B",
            "31B dense",
            mlx="mlx-community/gemma-4-31b-it-4bit",
            gguf=("ggml-org/gemma-4-31B-it-GGUF", "gemma-4-31B-it-Q4_0.gguf"),
            draft=(
                "ggml-org/gemma-4-31B-it-GGUF",
                "dflash-gemma-4-31B-it-Q8_0.gguf",
                "draft-dflash",
            ),
            vllm_parser="gemma4",
            vllm_mlx_parser="gemma4",
        ),
        Model(
            "qwen3.8-27b",
            "Qwen3.8 27B",
            "27B dense",
            mlx="mlx-community/Qwen3.8-27B-4bit",
            gguf=("ggml-org/Qwen3.8-27B-GGUF", "Qwen3.8-27B-Q4_K_M.gguf"),
            draft=("ggml-org/Qwen3.8-27B-GGUF", "dflash-Qwen3.8-27B-Q4_0.gguf", "draft-dflash"),
            vllm_parser="qwen3_xml",
            vllm_mlx_parser="qwen3.5",
        ),
        Model(
            "qwen3.8-flash-next",
            "Qwen3.8-Flash-Next",
            "MoE, oQ4e",
            mlx="Jundot/Qwen3.8-Flash-Next-oQ4e-mtp",
            gguf=("ggml-org/Qwen3.8-Flash-Next-GGUF", "Qwen3.8-Flash-Next-IQ4_NL-*.gguf"),
            draft=(
                "ggml-org/Qwen3.8-Flash-Next-GGUF",
                "dflash-Qwen3.8-Flash-Next-Q4_0.gguf",
                "draft-dflash",
            ),
            vllm_parser="qwen3_xml",
            vllm_mlx_parser="qwen3.5",
            mlx_mtp=True,
            notes=["MLX weights only in oMLX's oQ4e quantization (mixed 4/5-bit affine)"],
        ),
        Model(
            "gpt-oss-120b",
            "gpt-oss-120B",
            "117B total / 5.1B active",
            mlx="mlx-community/gpt-oss-120b-MXFP4-Q8",
            gguf=("ggml-org/gpt-oss-120b-GGUF", "gpt-oss-120b-MXFP4.gguf"),
            draft=("ggml-org/gpt-oss-120b-GGUF", "eagle3-gpt-oss-120b-Q8_0.gguf", "draft-eagle3"),
            vllm_parser="openai",
            vllm_mlx_parser="harmony",
            notes=["mlx_lm.server does not parse gpt-oss (harmony) tool calls (mlx-lm #875)"],
        ),
        Model(
            "qwen3-vl-235b",
            "Qwen3-VL-235B-A22B",
            "235B total / 22B active",
            mlx="mlx-community/Qwen3-VL-235B-A22B-Instruct-4bit",
            gguf=("unsloth/Qwen3-VL-235B-A22B-Instruct-GGUF", "UD-Q4_K_XL/"),
            vllm_parser="hermes",
            vllm_mlx_parser="qwen3",
            mlx_lm_arch=False,
            notes=["vision-language model, measured on text"],
        ),
        Model(
            "glm-4.7",
            "GLM-4.7",
            "358B total / 32B active",
            mlx="mrtoots/unsloth-GLM-4.7-mlx-3Bit",
            gguf=("unsloth/GLM-4.7-GGUF", "UD-Q3_K_XL/"),
            vllm_parser="glm47",
            vllm_mlx_parser="glm47",
            notes=["3-bit: the 4-bit (199-217 GB) leaves too little memory for the KV cache"],
        ),
        Model(
            "deepseek-v4-flash",
            "DeepSeek-V4-Flash-0731",
            "284B total / 13B active",
            mlx="Jundot/DeepSeek-V4-Flash-0731-oQ4e-mtp",
            gguf=("ggml-org/DeepSeek-V4-Flash-0731-GGUF", "DeepSeek-V4-Flash-0731-MXFP4-*.gguf"),
            draft=(
                "ggml-org/DeepSeek-V4-Flash-0731-GGUF",
                "dspark-DeepSeek-V4-Flash-0731-MXFP4.gguf",
                "draft-dspark",
            ),
            vllm_parser="deepseek_v4",
            vllm_mlx_parser="deepseek_v4",
            mlx_lm_arch=False,
            vllm_metal=False,
            mlx_mtp=True,
            notes=[
                "MLX weights in oMLX's oQ4e quantization; the third-party conversion "
                "inferencerlabs/DeepSeek-V4-Flash-0731-MLX loaded in none of the five MLX engines",
            ],
        ),
        Model(
            "glm-5.3-flash",
            "GLM-5.3-Flash",
            "320B total / 18B active",
            mlx="Jundot/GLM-5.3-Flash-oQ4e-mtp",
            gguf=("ggml-org/GLM-5.3-Flash-GGUF", "GLM-5.3-Flash-Q4_K-*.gguf"),
            draft=("ggml-org/GLM-5.3-Flash-GGUF", "mtp-GLM-5.3-Flash-Q4_0.gguf", "draft-mtp"),
            vllm_parser="glm47",
            vllm_mlx_parser="glm47",
            mlx_mtp=True,
            mlx_lm_arch=False,
            vllm_metal=False,
            template=("Jundot/GLM-5.3-Flash-oQ4e-mtp", "chat_template.jinja", "gguf"),
            notes=[
                "MLX weights only in oMLX's oQ4e quantization",
                "llama.cpp b11669 cannot build a tool-call parser from the GGUF's template "
                "(HTTP 400 on any request with tool turns); the MLX repo's template loads",
            ],
        ),
        Model(
            "mimo-v2.6-flash",
            "MiMo-V2.6-Flash-MOPD",
            "309B total / 15B active",
            mlx="EigenLabs/MiMo-V2.6-Flash-MOPD-MLX-4bit-mtp",
            gguf=("ggml-org/MiMo-V2.6-Flash-MOPD-GGUF", "MiMo-V2.6-Flash-MOPD-MXFP4-*.gguf"),
            draft=(
                "ggml-org/MiMo-V2.6-Flash-MOPD-GGUF",
                "dflash-MiMo-V2.6-Flash-MOPD-Q8_0.gguf",
                "draft-dflash",
            ),
            vllm_parser="mimo",
            vllm_mlx_parser="auto",
            mlx_mtp=True,
            mlx_lm_arch=False,
        ),
        Model(
            "minimax-m3.1-flash",
            "MiniMax-M3.1-Flash",
            "428B total / 23B active",
            mlx="mlx-community/MiniMax-M3.1-Flash-Text-3bit",
            gguf=None,
            vllm_parser="minimax_m3",
            vllm_mlx_parser="minimax",
            notes=["no GGUF published yet; 3-bit, since the 4-bit (249 GB) does not fit"],
        ),
    ]
}
