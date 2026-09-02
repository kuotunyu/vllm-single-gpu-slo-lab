"""`slo-lab` command line (typer).

W0: `attainment`, `capacity` and `cost` compute for real on the canonical record/CSV formats;
`quiet-gpu`, `power-sample` and `shim` are the W1 entry points for the WSL2 host;
`reproduce-lite` validates configs and rebuilds whatever evidence exists (none yet).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer
import yaml

from slo_lab import __version__
from slo_lab.slo import (
    ATTAINMENT_TARGET,
    Slo,
    SweepCell,
    SweepRun,
    filter_window,
    grid_markdown,
    r_slo,
    read_records_jsonl,
    sensitivity_grid,
    summarise,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Single-GPU vLLM SLO lab: attainment, capacity, admission shim, power and cost.",
)


def _echo_json(obj: object) -> None:
    typer.echo(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


@app.callback(invoke_without_command=True)
def _root(
    version: Annotated[bool, typer.Option("--version", help="Print version and exit.")] = False,
) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()


@app.command()
def attainment(
    records: Annotated[Path, typer.Argument(help="JSONL of canonical RequestRecord rows.")],
    ttft_s: Annotated[float, typer.Option(help="TTFT threshold, seconds.")] = 1.0,
    tpot_s: Annotated[float, typer.Option(help="TPOT threshold, seconds.")] = 0.05,
    window_start_s: Annotated[
        float, typer.Option(help="Drop requests offered before this.")
    ] = 60.0,
    window_s: Annotated[float | None, typer.Option(help="Window length for goodput.")] = None,
) -> None:
    """SLO attainment (offered denominator), admitted-only attainment, rejection rate, goodput."""
    recs = filter_window(read_records_jsonl(records), window_start_s)
    if not recs:
        typer.echo("no records in the measurement window", err=True)
        raise typer.Exit(code=1)
    summary = summarise(recs, Slo(ttft_s=ttft_s, tpot_s=tpot_s), window_s=window_s)
    typer.echo(summary.model_dump_json(indent=2))


@app.command()
def capacity(
    sweep: Annotated[
        Path,
        typer.Argument(
            help="JSONL: {offered_rate, seed, records: <path relative to this file>} per run, "
            "or {offered_rate, seed, attainment} if already reduced."
        ),
    ],
    ttft_s: float = 1.0,
    tpot_s: float = 0.05,
    target: float = ATTAINMENT_TARGET,
    window_start_s: float = 60.0,
    grid: Annotated[bool, typer.Option(help="Also print the SLO-sensitivity grid.")] = True,
) -> None:
    """Capacity r_SLO across seeds, plus the zero-cost SLO-sensitivity grid from raw records."""
    slo = Slo(ttft_s=ttft_s, tpot_s=tpot_s)
    cells: list[SweepCell] = []
    runs: list[SweepRun] = []
    for line in sweep.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "attainment" in row:
            cells.append(SweepCell.model_validate(row))
            continue
        recs = read_records_jsonl(sweep.parent / row["records"])
        run = SweepRun(offered_rate=row["offered_rate"], seed=row["seed"], records=recs)
        runs.append(run)
        windowed = filter_window(recs, window_start_s)
        if windowed:
            cells.append(
                SweepCell(
                    offered_rate=run.offered_rate,
                    seed=run.seed,
                    attainment=summarise(windowed, slo).attainment_offered,
                )
            )
    if not cells:
        typer.echo("sweep file has no usable runs", err=True)
        raise typer.Exit(code=1)
    typer.echo(r_slo(cells, target=target).model_dump_json(indent=2))
    if grid and runs:
        typer.echo("")
        typer.echo(
            grid_markdown(sensitivity_grid(runs, target=target, window_start_s=window_start_s))
        )


@app.command()
def cost(
    output_tok_per_s: Annotated[float, typer.Option(help="Output tok/s at r_SLO.")],
    config: Annotated[Path, typer.Option(help="config/cost.yaml")] = Path("config/cost.yaml"),
    power_csv: Annotated[Path | None, typer.Option(help="power.csv from the sampler.")] = None,
    phase: Annotated[str, typer.Option(help="Phase label of the measurement window.")] = "measure",
    p_avg_w: Annotated[float | None, typer.Option(help="Mean board power if no CSV.")] = None,
    input_tok_per_s: float | None = None,
    peak_output_tok_per_s: Annotated[float | None, typer.Option(help="Closed-loop peak.")] = None,
    peak_p_avg_w: float | None = None,
) -> None:
    """$/M output token at r_SLO (measured) and utilisation-naive, Wh/M token, with caveats."""
    from slo_lab.cost import (
        OperatingPoint,
        cost_report,
        idle_baseline_w,
        integrate,
        load_cost_config,
        read_power_csv,
        select_window,
    )

    cfg = load_cost_config(config)
    extra: dict[str, object] = {}
    if power_csv is not None:
        samples = read_power_csv(power_csv)
        window = integrate(select_window(samples, phase=phase))
        p_avg_w = window.p_avg_w
        extra["power_window"] = window.model_dump()
        idle = select_window(samples, phase="idle")
        if idle:
            baseline = idle_baseline_w(idle)
            extra["idle_baseline_w"] = baseline
            extra["p_avg_net_w"] = window.p_avg_w - baseline
    if p_avg_w is None:
        typer.echo("need --power-csv or --p-avg-w", err=True)
        raise typer.Exit(code=1)
    at = OperatingPoint(
        output_tok_per_s=output_tok_per_s, p_avg_w=p_avg_w, input_tok_per_s=input_tok_per_s
    )
    peak = None
    if peak_output_tok_per_s is not None:
        peak = OperatingPoint(
            output_tok_per_s=peak_output_tok_per_s,
            p_avg_w=p_avg_w if peak_p_avg_w is None else peak_p_avg_w,
            label="closed-loop peak",
        )
    report = cost_report(cfg, at, peak)
    _echo_json({**report.model_dump(), **extra})


@app.command("quiet-gpu")
def quiet_gpu_cmd(
    out: Annotated[Path | None, typer.Option(help="Write snapshot JSON here.")] = None,
    gpu_index: int = 0,
    memory_threshold_mib: float | None = None,
    allow_pid: Annotated[list[int] | None, typer.Option(help="PIDs allowed on the GPU.")] = None,
) -> None:
    """Refuse (exit 1) unless the GPU has no other compute process; write the snapshot JSON."""
    from slo_lab.nvml import NvmlUnavailableError
    from slo_lab.quiet_gpu import DEFAULT_MEMORY_THRESHOLD_MIB, decide_from_snapshot, snapshot
    from slo_lab.quiet_gpu import write_snapshot as _write

    try:
        snap = snapshot(index=gpu_index)
    except NvmlUnavailableError as exc:
        typer.echo(f"quiet-gpu: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    decision = decide_from_snapshot(
        snap,
        memory_threshold_mib=memory_threshold_mib or DEFAULT_MEMORY_THRESHOLD_MIB,
        allow_pids=tuple(allow_pid or ()),
    )
    if out is not None:
        _write(out, snap, decision)
    _echo_json({"ok": decision.ok, "reasons": decision.reasons, "snapshot": str(out or "-")})
    if not decision.ok:
        raise typer.Exit(code=1)


@app.command("power-sample")
def power_sample(
    out: Annotated[Path, typer.Argument(help="CSV to write/append.")],
    interval_s: float = 1.0,
    duration_s: Annotated[float | None, typer.Option(help="Stop after this many seconds.")] = None,
    phase: Annotated[str, typer.Option(help="idle | warmup | measure")] = "",
    gpu_index: int = 0,
    append: bool = False,
) -> None:
    """1 s NVML power sampler (board power, util, memory, SM clock, temperature)."""
    from slo_lab.nvml import NvmlUnavailableError
    from slo_lab.power.sampler import NvmlReader, run_sampler

    try:
        reader = NvmlReader(gpu_index)
    except NvmlUnavailableError as exc:
        typer.echo(f"power-sample: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    max_samples = None if duration_s is None else int(duration_s / interval_s) + 1
    out.parent.mkdir(parents=True, exist_ok=True)
    fresh = not (append and out.exists() and out.stat().st_size > 0)
    try:
        with out.open("a" if append else "w", encoding="utf-8", newline="") as fh:
            n = run_sampler(
                reader,
                fh,
                interval_s=interval_s,
                max_samples=max_samples,
                phase=phase,
                write_header=fresh,
            )
    except KeyboardInterrupt:  # pragma: no cover - interactive stop
        n = -1
    finally:
        reader.close()
    typer.echo(f"wrote {n} samples to {out}")


@app.command()
def shim(
    upstream: Annotated[str, typer.Option(help="OpenAI-compatible base URL, e.g. vLLM.")],
    host: str = "127.0.0.1",
    port: int = 8001,
    policy: Annotated[
        str, typer.Option(help="passthrough | hard_cap | bounded_queue")
    ] = "passthrough",
    capacity: int | None = None,
    queue_limit: int | None = None,
    timeout_s: float = 1.0,
    retry_after_s: float = 1.0,
) -> None:
    """Run the admission shim in front of an OpenAI-compatible server (blocking)."""
    from slo_lab.admission import policy_from_config
    from slo_lab.admission.shim import run

    cfg = {
        "policy": policy,
        "capacity": capacity,
        "queue_limit": queue_limit,
        "timeout_s": timeout_s,
        "retry_after_s": retry_after_s,
    }
    if policy != "passthrough" and capacity is None:
        typer.echo("--capacity is required for hard_cap / bounded_queue", err=True)
        raise typer.Exit(code=1)
    run(upstream, policy_from_config(cfg), host=host, port=port)


@app.command("reproduce-lite")
def reproduce_lite(
    root: Annotated[Path, typer.Option(help="Repository root.")] = Path("."),
) -> None:
    """CPU-only rebuild: validate configs, recompute attainment for every run with records.

    W0: there is no evidence yet, so this validates the skeleton and reports zero runs.
    """
    problems: list[str] = []
    config_dir = root / "config"
    yaml_files = sorted(config_dir.rglob("*.yaml")) + sorted(config_dir.rglob("*.example"))
    for path in yaml_files:
        try:
            yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            problems.append(f"{path}: {exc}")
    typer.echo(f"configs parsed: {len(yaml_files)}")

    from slo_lab.cost import load_cost_config

    cost_path = root / "config" / "cost.yaml"
    if cost_path.exists():
        cfg = load_cost_config(cost_path)
        typer.echo(f"cost config status: {cfg.status}")
    else:
        problems.append("config/cost.yaml missing")

    for name in ("runs.csv", "cost.csv", "spend.csv"):
        if not (root / "analysis" / "ledger" / name).exists():
            problems.append(f"analysis/ledger/{name} missing")

    n_runs = 0
    for area in ("raw", "runpod"):
        base = root / "evidence" / area
        if not base.exists():
            continue
        for run_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            records = run_dir / "records.jsonl"
            if not records.exists():
                continue
            n_runs += 1
            recs = filter_window(read_records_jsonl(records), 60.0)
            if not recs:
                problems.append(f"{records}: no records after the 60 s warm-up window")
                continue
            s = summarise(recs)
            typer.echo(
                f"{area}/{run_dir.name}: offered={s.offered} attainment={s.attainment_offered:.3f} "
                f"[{s.attainment_offered_ci95[0]:.3f}, {s.attainment_offered_ci95[1]:.3f}] "
                f"rejection={s.rejection_rate:.3f}"
            )
    typer.echo(f"evidence runs with records: {n_runs}")
    if n_runs == 0:
        typer.echo("nothing to rebuild yet (W0 skeleton) - tables/plots/ledgers unchanged")
    if problems:
        for p in problems:
            typer.echo(f"problem: {p}", err=True)
        raise typer.Exit(code=1)


def main() -> None:  # pragma: no cover - console entry
    app(prog_name="slo-lab")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
