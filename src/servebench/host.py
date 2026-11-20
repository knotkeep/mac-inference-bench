"""The Mac around a benchmark run: other model servers out of the way, weights on disk.

A clean measurement needs the memory and GPU free. If a model server runs on the machine as a
launchd agent, name it and a run boots it out and bootstraps it back afterwards:

    SERVEBENCH_STOP_LAUNCHD   label of a launchd agent in ~/Library/LaunchAgents to stop
                              for the run (unset: nothing is stopped)
    SERVEBENCH_KEEP_REPOS     comma-separated Hugging Face repos a run never deletes, such
                              as the weights that server uses
"""

from __future__ import annotations

import fnmatch
import os
import shutil
import subprocess
import time
from pathlib import Path

from servebench.engines import BENCH, Weights
from servebench.models import Model


def _label() -> str | None:
    return os.environ.get("SERVEBENCH_STOP_LAUNCHD") or None


def _plist() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{_label()}.plist"


def _protected() -> set[str]:
    return {r.strip() for r in os.environ.get("SERVEBENCH_KEEP_REPOS", "").split(",") if r.strip()}


def _domain() -> str:
    return f"gui/{os.getuid()}"


def production_running() -> bool:
    if _label() is None:
        return False
    out = subprocess.run(
        ["launchctl", "print", f"{_domain()}/{_label()}"], capture_output=True, text=True
    )
    return out.returncode == 0


def stop_production() -> bool:
    """Stop the named launchd agent. Returns whether it was running."""
    if not production_running():
        return False
    subprocess.run(["launchctl", "bootout", f"{_domain()}/{_label()}"], capture_output=True)
    for _ in range(60):
        if not production_running():
            break
        time.sleep(2)
    time.sleep(10)  # the stopped server's memory is released over a few seconds
    return True


def start_production() -> None:
    if _label() is not None and not production_running():
        subprocess.run(["launchctl", "bootstrap", _domain(), str(_plist())], capture_output=True)


def fetch(model: Model, formats: tuple[str, ...] = ("mlx", "gguf"), log=print) -> Weights:
    """Download what the model needs into the Hugging Face cache and return local paths."""
    from huggingface_hub import snapshot_download

    w = Weights()
    if "mlx" in formats and model.mlx:
        log(f"  fetching {model.mlx}")
        w.mlx_dir = Path(snapshot_download(model.mlx))
    if "gguf" in formats and model.gguf:
        repo, pattern = model.gguf
        allow = f"{pattern}*" if pattern.endswith("/") else pattern
        log(f"  fetching {repo} {allow}")
        root = Path(snapshot_download(repo, allow_patterns=[allow]))
        files = sorted(
            p for p in root.rglob("*.gguf") if fnmatch.fnmatch(str(p.relative_to(root)), allow)
        )
        w.gguf = files[0] if files else None
        if model.draft:
            drepo, dfile, _ = model.draft
            log(f"  fetching {drepo} {dfile}")
            w.draft = Path(snapshot_download(drepo, allow_patterns=[dfile])) / dfile
    return w


def fetch_template(model: Model) -> Path:
    """Write the model's corrected chat template to a local .jinja file and return its path."""
    import json

    from huggingface_hub import hf_hub_download

    repo, file, _ = model.template
    src = Path(hf_hub_download(repo, file))
    text = json.loads(src.read_text())["chat_template"] if file.endswith(".json") else None
    out = BENCH / "templates" / f"{model.label}.jinja"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text if text is not None else src.read_text())
    return out


def disk_free_gb() -> float:
    return shutil.disk_usage(Path.home()).free / 1e9


def delete(model: Model, keep: set[str] | None = None) -> list[str]:
    """Delete this model's downloads from the Hugging Face cache (repos in `keep` stay)."""
    removed = []
    repos = (
        {
            model.mlx,
            model.gguf[0] if model.gguf else None,
            model.draft[0] if model.draft else None,
            model.template[0] if model.template else None,
        }
        - {None}
        - (keep or set())
        - _protected()
    )
    for repo in repos:
        out = subprocess.run(
            ["hf", "cache", "rm", f"model/{repo}", "-y", "--format", "quiet"],
            capture_output=True,
            text=True,
        )
        if out.returncode == 0:
            removed.append(repo)
    return removed
