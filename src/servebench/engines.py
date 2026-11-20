"""Engine adapters: start a server for one model, report when it can answer, stop it.

Every engine serves an OpenAI-compatible API on its own port on 127.0.0.1. `start` returns the
endpoint the benchmark talks to plus what the adapter measured (version, seconds until the first
successful completion, memory in use). Variants with speculative decoding end in `+spec`.

    engine       format  served by
    omlx         MLX     `omlx serve` on a private model dir (oMLX's own quant formats, MTP)
    mlx-lm       MLX     `mlx_lm.server`, or `mlx_vlm.server` for architectures mlx-lm lacks
    vllm-metal   MLX     `vllm serve` (official vLLM Apple-silicon plugin)
    vllm-mlx     MLX     `vllm-mlx serve` (community, continuous batching)
    llama.cpp    GGUF    `llama-server`
    ollama       GGUF    `ollama serve` on a private models dir, model created from the GGUF
    lmstudio     GGUF    LM Studio's llama.cpp engine through `lms`
    lmstudio-mlx MLX     LM Studio's MLX engine through `lms`
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from servebench.models import Model

# Scratch space for engines that need their own model directory, and the servers' logs.
BENCH = Path(os.environ.get("SERVEBENCH_HOME", Path.home() / ".servebench"))
LOGS = Path(os.environ.get("SERVEBENCH_LOGS", "logs"))
LMS = "/Applications/LM Studio.app/Contents/Resources/app/.webpack/lms"
NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}
CONTEXT = 65536


@dataclass
class Weights:
    """Local paths of one model's downloaded files."""

    mlx_dir: Path | None = None
    gguf: Path | None = None  # first shard
    draft: Path | None = None


@dataclass
class Running:
    endpoint: dict[str, Any]
    meta: dict[str, Any]
    procs: list[subprocess.Popen] = field(default_factory=list)
    cleanup: list[list[str]] = field(default_factory=list)


def _used_gb() -> float:
    import psutil

    vm = psutil.virtual_memory()
    return (vm.total - vm.available) / 1e9


def _version(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        text = (out.stdout + "\n" + out.stderr).strip().splitlines()
        tagged = [t for t in text if "version" in t.lower() and "warning" not in t.lower()]
        return (tagged or text or ["?"])[0].strip()[:120]
    except Exception as e:  # noqa: BLE001 - a missing version is recorded, not fatal
        return f"? ({type(e).__name__})"


def _launch(cmd: list[str], log_name: str, env: dict[str, str] | None = None) -> subprocess.Popen:
    port = next((cmd[i + 1] for i, c in enumerate(cmd) if c == "--port"), None)
    if port:
        _free_port(int(port))
    LOGS.mkdir(parents=True, exist_ok=True)
    log = open(LOGS / f"{log_name}.log", "w")  # noqa: SIM115 - closed when the process ends
    return subprocess.Popen(
        cmd,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env={**os.environ, **(env or {})},
    )


def _ready(
    base_url: str, model: str, api_key: str, timeout: float, procs: list[subprocess.Popen]
) -> float:
    """Seconds until a one-token completion succeeds (server up and model loaded)."""
    t0 = time.perf_counter()
    last = ""
    while time.perf_counter() - t0 < timeout:
        for p in procs:
            if p.poll() is not None:
                raise RuntimeError(f"server exited with code {p.returncode}: {last}")
        try:
            r = httpx.post(
                base_url + "/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 1,
                    "temperature": 0,
                },
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=600,
            )
            if r.status_code == 200:
                return time.perf_counter() - t0
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code == 409 and "load failure" in r.text:
                # oMLX keeps the server up after a failed load and refuses every request.
                raise RuntimeError(last)
        except httpx.HTTPError as e:
            last = f"{type(e).__name__}"
        time.sleep(3)
    raise TimeoutError(f"not ready after {timeout:.0f}s: {last}")


def _free_port(port: int) -> None:
    """Stop whatever still listens on the engine's port (a server left by an earlier run)."""
    out = subprocess.run(
        ["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"], capture_output=True, text=True
    )
    for pid in out.stdout.split():
        try:
            os.kill(int(pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
    if out.stdout.strip():
        time.sleep(5)


def _settle(baseline: float | None, timeout: float = 180) -> None:
    """Wait until memory in use is back near where it was before the engine started, so the
    next engine's load starts from the same state (vllm-metal releases its pool slowly)."""
    if baseline is None:
        return
    deadline = time.time() + timeout
    while time.time() < deadline and _used_gb() > baseline + 3:
        time.sleep(3)


def _stop(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                continue
    deadline = time.time() + 60
    for p in procs:
        try:
            p.wait(timeout=max(1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)


def _split(model: Model, weights: Weights) -> bool:
    """The model's GGUF is several files: a downloaded first shard, or a pattern or folder that
    names shards (known without downloading)."""
    if weights.gguf is not None:
        return "-00001-of-" in weights.gguf.name
    return model.gguf is not None and (model.gguf[1].endswith("/") or "*" in model.gguf[1])


class Engine:
    name: str = ""
    fmt: str = "mlx"
    port: int = 0
    load_timeout: float = 1800
    templates: bool = False  # can serve with a chat template given at start (`+tpl` variants)

    def unsupported(self, model: Model, weights: Weights, spec: bool) -> str | None:
        """Why this engine cannot serve the model (or variant); None when it can."""
        if self.fmt == "mlx" and weights.mlx_dir is None:
            return "no MLX weights"
        if self.fmt == "gguf" and weights.gguf is None:
            return "no GGUF weights"
        return None

    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def start(
        self, model: Model, weights: Weights, spec: bool = False, template: Path | None = None
    ) -> Running:
        raise NotImplementedError

    def stop(self, running: Running) -> None:
        _stop(running.procs)
        for cmd in running.cleanup:
            subprocess.run(cmd, capture_output=True, timeout=600)
        _settle(running.meta.get("mem_baseline"))

    def _finish(
        self,
        procs: list[subprocess.Popen],
        model_id: str,
        version: str,
        extra_body: dict | None = None,
        api_key: str = "local",
        **meta: Any,
    ) -> Running:
        before = meta.pop("mem_before", _used_gb())
        try:
            load_s = _ready(self.base(), model_id, api_key, self.load_timeout, procs)
        except Exception:
            # A server that never became ready must not keep its port and memory.
            _stop(procs)
            _settle(before)
            raise
        endpoint = {
            "base_url": self.base(),
            "model": model_id,
            "api_key": api_key,
            "extra_body": extra_body,
        }
        info = {
            "version": version,
            "load_s": round(load_s, 1),
            "memory_gb": round(_used_gb() - before, 1),
            "mem_baseline": round(before, 1),
            **meta,
        }
        return Running(endpoint, info, procs)


class OMLX(Engine):
    name, fmt, port = "omlx", "mlx", 8201

    def unsupported(self, model, weights, spec):
        if spec and not model.mlx_mtp:
            return "no MTP head in the MLX weights"
        return super().unsupported(model, weights, spec)

    def start(self, model, weights, spec=False):
        mdir = BENCH / "omlx"
        mdir.mkdir(parents=True, exist_ok=True)
        for old in mdir.iterdir():
            old.unlink()
        (mdir / model.label).symlink_to(weights.mlx_dir)
        before = _used_gb()
        p = _launch(
            [
                "omlx",
                "serve",
                "--model-dir",
                str(mdir),
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--no-cache",
                "--headless",
                "--max-concurrent-requests",
                "4",
                "--memory-guard",
                "off",
            ],
            f"omlx__{model.label}{'+spec' if spec else ''}",
        )
        if spec:
            # oMLX turns MTP on per model through its admin API (Lightning MTP: same greedy output).
            for _ in range(60):
                try:
                    httpx.put(
                        f"http://127.0.0.1:{self.port}/admin/api/models/{model.label}/settings",
                        json={"mtp_enabled": True},
                        timeout=10,
                    ).raise_for_status()
                    break
                except httpx.HTTPError:
                    time.sleep(2)
        return self._finish(
            [p],
            model.label,
            _version(["omlx", "--version"]),
            NO_THINKING,
            mem_before=before,
            mtp=spec,
        )


class MLXLM(Engine):
    """mlx_lm.server, or mlx_vlm.server for architectures mlx-lm does not have."""

    name, fmt, port = "mlx-lm", "mlx", 8202
    templates = True  # mlx_lm.server only

    def unsupported(self, model, weights, spec):
        if spec and not model.mlx_mtp:
            return "no MTP head in the MLX weights"
        if spec and model.mlx_lm_arch:
            return "mlx_lm.server has no native MTP (mlx-lm PR #990)"
        return super().unsupported(model, weights, spec)

    def start(self, model, weights, spec=False, template=None):
        before = _used_gb()
        path = str(weights.mlx_dir)
        if template is not None and not (model.mlx_lm_arch and not spec):
            raise RuntimeError("chat template override needs mlx_lm.server")
        if model.mlx_lm_arch and not spec:
            cmd = [
                "mlx_lm.server",
                "--model",
                path,
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--chat-template-args",
                json.dumps({"enable_thinking": False}),
            ]
            if template is not None:
                cmd += ["--chat-template", template.read_text()]
            version = _tool_version("mlx-lm")
        else:
            cmd = [
                "mlx_vlm.server",
                "--model",
                path,
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
            ]
            if spec:
                cmd += ["--draft-kind", "mtp"]
            version = _tool_version("mlx-vlm")
        tag = ("+spec" if spec else "") + ("+tpl" if template is not None else "")
        p = _launch(cmd, f"mlx-lm__{model.label}{tag}")
        return self._finish(
            [p], path, version, NO_THINKING, mem_before=before, server=cmd[0], mtp=spec
        )


class VLLMMetal(Engine):
    name, fmt, port = "vllm-metal", "mlx", 8204

    def unsupported(self, model, weights, spec):
        if spec:
            return "no speculative variant run"
        if not model.vllm_metal:
            return "architecture not supported by vllm-metal"
        return super().unsupported(model, weights, spec)

    def start(self, model, weights, spec=False):
        before = _used_gb()
        cmd = [
            "vllm",
            "serve",
            str(weights.mlx_dir),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--served-model-name",
            model.label,
            "--max-model-len",
            str(CONTEXT // 2),
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            model.vllm_parser or "hermes",
        ]
        p = _launch(cmd, f"vllm-metal__{model.label}")
        return self._finish(
            [p], model.label, _version(["vllm", "--version"]), NO_THINKING, mem_before=before
        )


class VLLMMLX(Engine):
    name, fmt, port = "vllm-mlx", "mlx", 8205

    def unsupported(self, model, weights, spec):
        if spec and not model.mlx_mtp:
            return "no MTP head in the MLX weights"
        return super().unsupported(model, weights, spec)

    def start(self, model, weights, spec=False):
        before = _used_gb()
        cmd = [
            "vllm-mlx",
            "serve",
            str(weights.mlx_dir),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--served-model-name",
            model.label,
            "--continuous-batching",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            model.vllm_mlx_parser or "auto",
        ]
        if not model.mlx_lm_arch:
            cmd.append("--mllm")
        if spec:
            cmd.append("--enable-mtp")
        p = _launch(cmd, f"vllm-mlx__{model.label}{'+spec' if spec else ''}")
        return self._finish(
            [p], model.label, _tool_version("vllm-mlx"), NO_THINKING, mem_before=before, mtp=spec
        )


class LlamaCpp(Engine):
    name, fmt, port = "llama.cpp", "gguf", 8203
    templates = True

    def unsupported(self, model, weights, spec):
        if spec and (model.draft is None or weights.draft is None):
            return "no draft/MTP file published for this GGUF"
        return super().unsupported(model, weights, spec)

    def start(self, model, weights, spec=False, template=None):
        before = _used_gb()
        cmd = [
            "llama-server",
            "-m",
            str(weights.gguf),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "-a",
            model.label,
            "-c",
            str(CONTEXT),
            "-np",
            "4",
            "-kvu",
            "-ngl",
            "999",
            "-fa",
            "on",
            "--jinja",
            "--no-webui",
        ]
        if spec:
            cmd += ["--spec-type", model.draft[2], "-md", str(weights.draft)]
            if model.draft[2] == "draft-mtp":
                cmd += ["--spec-draft-n-max", "2"]
        if template is not None:
            cmd += ["--chat-template-file", str(template)]
        tag = ("+spec" if spec else "") + ("+tpl" if template is not None else "")
        p = _launch(cmd, f"llama.cpp__{model.label}{tag}")
        return self._finish(
            [p],
            model.label,
            _version(["llama-server", "--version"]),
            NO_THINKING,
            mem_before=before,
            spec_type=model.draft[2] if spec else None,
        )


class Ollama(Engine):
    name, fmt, port = "ollama", "gguf", 8206

    def unsupported(self, model, weights, spec):
        if spec:
            return "no speculative variant run"
        if _split(model, weights):
            return "split GGUF; ollama create takes a single file"
        return super().unsupported(model, weights, spec)

    def start(self, model, weights, spec=False):
        env = {
            "OLLAMA_HOST": f"127.0.0.1:{self.port}",
            "OLLAMA_MODELS": str(BENCH / "ollama-models"),
            "OLLAMA_NUM_PARALLEL": "4",
            "OLLAMA_CONTEXT_LENGTH": str(CONTEXT // 2),
            "OLLAMA_FLASH_ATTENTION": "1",
        }
        before = _used_gb()
        p = _launch(["ollama", "serve"], f"ollama__{model.label}", env)
        name = f"bench-{model.label}"
        modelfile = BENCH / "ollama.Modelfile"
        modelfile.write_text(f"FROM {weights.gguf}\nPARAMETER num_ctx {CONTEXT // 2}\n")
        for _ in range(60):
            if (
                subprocess.run(
                    ["ollama", "list"], env={**os.environ, **env}, capture_output=True
                ).returncode
                == 0
            ):
                break
            time.sleep(1)
        subprocess.run(
            ["ollama", "create", name, "-f", str(modelfile)],
            env={**os.environ, **env},
            capture_output=True,
            timeout=3600,
            check=True,
        )
        # Ollama ignores chat_template_kwargs; reasoning_effort "none" turns thinking off.
        running = self._finish(
            [p],
            name,
            _version(["ollama", "--version"]),
            {**NO_THINKING, "reasoning_effort": "none"},
            mem_before=before,
        )
        running.cleanup.append(["env", f"OLLAMA_HOST=127.0.0.1:{self.port}", "ollama", "rm", name])
        return running

    def stop(self, running):
        # Remove the created model while the server still runs, then stop it.
        for cmd in running.cleanup:
            subprocess.run(cmd, capture_output=True, timeout=600)
        running.cleanup = []
        _stop(running.procs)


class LMStudio(Engine):
    """LM Studio through `lms`: the model is imported as a symbolic link and loaded under a fixed
    identifier; `fmt` picks the llama.cpp (GGUF) or MLX engine."""

    name, fmt, port = "lmstudio", "gguf", 8207

    def __init__(self, fmt: str = "gguf"):
        self.fmt = fmt
        self.name = "lmstudio" if fmt == "gguf" else "lmstudio-mlx"

    def unsupported(self, model, weights, spec):
        if spec and not model.mlx_mtp and self.fmt == "mlx":
            return "no MTP head in the MLX weights"
        if spec and self.fmt == "gguf":
            return "no speculative variant run"
        if self.fmt == "gguf" and _split(model, weights):
            return "split GGUF; not imported"
        return super().unsupported(model, weights, spec)

    def start(self, model, weights, spec=False):
        status = subprocess.run([LMS, "daemon", "status"], capture_output=True, text=True)
        if "is running" not in status.stdout:
            # The headless daemon is not installed here; the app runs the same server.
            subprocess.run(["open", "-a", "LM Studio"], check=True)
            time.sleep(20)
        ident = f"bench-{model.label}"
        user_repo = f"servebench/{model.label}-{self.fmt}"
        src = weights.gguf if self.fmt == "gguf" else weights.mlx_dir
        subprocess.run(
            [LMS, "server", "start", "--port", str(self.port)], capture_output=True, timeout=300
        )
        target = Path.home() / ".lmstudio" / "models" / user_repo
        if self.fmt == "gguf":
            # A folder left by an earlier run holds a link LM Studio indexed before its target
            # was deleted; `lms import` then skips the file and it is never listed again.
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            subprocess.run(
                [LMS, "import", str(src), "-l", "-y", "--user-repo", user_repo],
                capture_output=True,
                timeout=600,
            )
        else:
            # `lms import` takes single files; an MLX model is a folder, linked where LM Studio
            # looks for models (<publisher>/<model>).
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink() or target.exists():
                target.unlink()
            target.symlink_to(src)
            time.sleep(5)
        key = None
        for _ in range(36):
            # LM Studio indexes new files asynchronously; large folders take minutes.
            listed = subprocess.run(
                [LMS, "ls", "--json"], capture_output=True, text=True, timeout=120
            )
            key = next(
                (
                    m["modelKey"]
                    for m in json.loads(listed.stdout or "[]")
                    if m.get("path", "").startswith(user_repo + "/") or m.get("path") == user_repo
                ),
                None,
            )
            if key:
                break
            time.sleep(5)
        if key is None:
            subprocess.run(["rm", "-rf", str(target)], capture_output=True)
            raise RuntimeError(f"lms import did not register {user_repo}")
        before = _used_gb()
        load = [
            LMS,
            "load",
            key,
            "--identifier",
            ident,
            "--context-length",
            str(CONTEXT // 2),
            "--parallel",
            "4",
            "--gpu",
            "max",
            "-y",
        ]
        if spec:
            load.append("--speculative-draft-mtp")
        try:
            out = subprocess.run(load, capture_output=True, text=True, timeout=3600)
            if out.returncode != 0:
                raise RuntimeError(f"lms load failed: {(out.stderr or out.stdout)[-300:]}")
            running = self._finish(
                [], ident, _lmstudio_version(), NO_THINKING, mem_before=before, mtp=spec
            )
        except Exception:
            subprocess.run([LMS, "unload", ident], capture_output=True, timeout=600)
            subprocess.run(["rm", "-rf", str(target)], capture_output=True)
            raise
        running.cleanup += [[LMS, "unload", ident], [LMS, "server", "stop"]]
        running.cleanup.append(["rm", "-rf", str(target)])
        return running


def _lmstudio_version() -> str:
    """The app's version (`lms version` prints a banner before it)."""
    plist = Path(LMS).parents[3] / "Info.plist"
    out = _version(["defaults", "read", str(plist), "CFBundleShortVersionString"])
    return f"LM Studio {out}"


def _tool_version(tool: str) -> str:
    out = subprocess.run(["uv", "tool", "list"], capture_output=True, text=True, timeout=60)
    for line in out.stdout.splitlines():
        if line.startswith(tool + " "):
            return line.strip()
    return f"{tool} ?"


ENGINES: dict[str, Engine] = {
    e.name: e
    for e in [
        OMLX(),
        MLXLM(),
        VLLMMetal(),
        VLLMMLX(),
        LlamaCpp(),
        Ollama(),
        LMStudio("gguf"),
        LMStudio("mlx"),
    ]
}
