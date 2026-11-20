"""Command line: servebench run | run-all | report | models."""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer

app = typer.Typer(
    help="Speed and agent-quality benchmark for local inference engines on Apple silicon.",
    no_args_is_help=True,
)


@app.command("run")
def run(
    model: str = typer.Argument(..., help="Model label (see `servebench models`)"),
    engines: str | None = typer.Option(None, help="Comma-separated engines (default: all)"),
    spec: bool = typer.Option(True, help="Also run speculative-decoding variants"),
    golden: bool = typer.Option(True, help="Run the golden set (needs the payerbench package)"),
    reps: int = typer.Option(3, help="Repetitions per prompt size"),
    keep_weights: bool = typer.Option(False, help="Keep downloads after the run"),
    template: bool = typer.Option(
        False, help="Run only the +tpl variants, serving the model's corrected chat template"
    ),
    trials: int = typer.Option(1, help="Golden-set trials per pair"),
    speed: bool = typer.Option(True, help="Run the speed and concurrency workloads"),
    plain_optional: bool = typer.Option(
        False, help="Send optional tool parameters as their plain type instead of anyOf"
    ),
    runs_dir: str = typer.Option("results/runs", help="Where run results are written"),
) -> None:
    """Benchmark one model on every engine that can serve it."""
    from servebench.run import run_model

    if plain_optional:
        os.environ["SERVEBENCH_PLAIN_OPTIONAL"] = "1"
    results = run_model(
        model,
        engines.split(",") if engines else None,
        spec=spec,
        golden=golden,
        reps=reps,
        keep_weights=keep_weights,
        template=template,
        trials=trials,
        speed=speed,
        runs_dir=Path(runs_dir),
        log=typer.echo,
    )
    for r in results:
        typer.echo(f"{r['engine']:18} {r.get('status', 'ok')}")


@app.command("run-all")
def run_all(
    models: str = typer.Option(..., help="Comma-separated model labels, run in this order"),
    golden: bool = typer.Option(True, help="Run the golden set (needs the payerbench package)"),
    runs_dir: str = typer.Option("results/runs", help="Where run results are written"),
) -> None:
    """Benchmark several models in sequence; a stopped model server stays stopped until the end."""
    from servebench import host
    from servebench.run import run_model

    was_running = host.stop_production()
    try:
        for label in models.split(","):
            typer.echo(f"=== {label} (free disk {host.disk_free_gb():.0f} GB)")
            try:
                run_model(
                    label, golden=golden, restore=False, runs_dir=Path(runs_dir), log=typer.echo
                )
            except Exception as e:  # noqa: BLE001 - the next model still runs
                typer.echo(f"{label} failed: {type(e).__name__}: {e}")
    finally:
        if was_running:
            host.start_production()


@app.command("report")
def report(
    out: str = typer.Option("results/report", help="Writes <out>.md and <out>.json"),
    runs_dir: str = typer.Option("results/runs", help="Run results to tabulate"),
) -> None:
    """Tables from every recorded run (one per model) and each model's fastest engine."""
    from servebench.report import best, load_runs, markdown, matrix

    runs = load_runs(Path(runs_dir))
    md = (
        "## Golden-set pass rate\n\n"
        + matrix(runs, "golden")
        + "\n## Decode tokens per second at 4k\n\n"
        + matrix(runs, "decode")
        + "\n## Per model\n\n"
        + markdown(runs)
    )
    Path(f"{out}.md").write_text(md)
    Path(f"{out}.json").write_text(json.dumps({"best": best(runs), "runs": len(runs)}, indent=1))
    typer.echo(md)


@app.command("models")
def models() -> None:
    """The benchmark's models and engines."""
    from servebench.engines import ENGINES
    from servebench.models import MODELS

    for m in MODELS.values():
        typer.echo(f"{m.label:20} {m.name:24} {m.params}")
    typer.echo("engines: " + ", ".join(ENGINES))
