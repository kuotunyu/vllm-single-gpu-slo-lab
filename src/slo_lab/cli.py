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
    evidence_path,
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
    utilization_threshold_percent: float | None = None,
    allow_pid: Annotated[list[int] | None, typer.Option(help="PIDs allowed on the GPU.")] = None,
) -> None:
    """Refuse (exit 1) unless the GPU is quiet (no foreign process, low memory, low utilization)."""
    from slo_lab.nvml import NvmlUnavailableError
    from slo_lab.quiet_gpu import (
        DEFAULT_MEMORY_THRESHOLD_MIB,
        DEFAULT_UTILIZATION_THRESHOLD_PERCENT,
        decide_from_snapshot,
        snapshot,
    )
    from slo_lab.quiet_gpu import write_snapshot as _write

    try:
        snap = snapshot(index=gpu_index)
    except NvmlUnavailableError as exc:
        typer.echo(f"quiet-gpu: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    decision = decide_from_snapshot(
        snap,
        memory_threshold_mib=memory_threshold_mib or DEFAULT_MEMORY_THRESHOLD_MIB,
        utilization_threshold_percent=(
            DEFAULT_UTILIZATION_THRESHOLD_PERCENT
            if utilization_threshold_percent is None
            else utilization_threshold_percent
        ),
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


@app.command("make-trace")
def make_trace(
    rate_ref: Annotated[float, typer.Option(help="r that the profile's multipliers scale (rps).")],
    seed: Annotated[int, typer.Option(help="Arrival seed; every policy of a seed replays it.")],
    out: Annotated[Path, typer.Option(help="Azure-format trace CSV for inference-perf.")],
    profile: Annotated[
        Path, typer.Option(help="Multistage traffic YAML (stages of rate_multiplier, duration_s).")
    ] = Path("config/traffic/burst25.yaml"),
) -> None:
    """Write a seeded piecewise-Poisson trace for inference-perf trace replay (ADR 0012)."""
    from slo_lab.harness.trace import burst_arrivals, load_profile, phase_bounds, write_azure_trace

    prof = load_profile(profile)
    arrivals = burst_arrivals(rate_ref, prof, seed)
    digest = write_azure_trace(out, arrivals)
    phases = [
        {
            "phase": name,
            "start_s": lo,
            "end_s": hi,
            "rate_rps": round(mult * rate_ref, 4),
            "arrivals": sum(1 for t in arrivals if lo <= t < hi),
        }
        for (name, lo, hi), (mult, _) in zip(phase_bounds(prof), prof, strict=True)
    ]
    _echo_json(
        {
            "arrivals": len(arrivals),
            "sha256": digest,
            "rate_ref": rate_ref,
            "seed": seed,
            "duration_s": sum(d for _, d in prof),
            "phases": phases,
        }
    )


@app.command("run-stage")
def run_stage_cmd(
    run_dir: Annotated[Path, typer.Option(help="Output directory for this stage.")],
    cell: Annotated[str, typer.Option(help="Engine cell label, e.g. fp8.")],
    model: Annotated[str, typer.Option(help="Model id served by vLLM.")],
    kind: Annotated[str, typer.Option(help="open_loop | closed_loop | trace")],
    base_url: str = "http://127.0.0.1:8013",
    metrics_url: str = "http://127.0.0.1:8013/metrics",
    seed: int = 1,
    rate_rps: float | None = None,
    duration_s: int = 300,
    concurrency: int | None = None,
    num_requests: int | None = None,
    warmup_requests: int = 100,
    discard_first_s: float = 60.0,
    inference_perf_bin: str = "inference-perf",
    workers: int = 4,
    engine_flags: Annotated[
        str | None, typer.Option(help="JSON of the server flags, copied into the manifest.")
    ] = None,
    trace_file: Annotated[
        Path | None, typer.Option(help="trace: Azure-format arrivals from make-trace.")
    ] = None,
    profile: Annotated[
        Path | None, typer.Option(help="trace: the traffic YAML the trace was made from.")
    ] = None,
    policy: Annotated[str | None, typer.Option(help="trace: admission policy label.")] = None,
    shim_stats_url: Annotated[
        str | None, typer.Option(help="trace: the shim's /_shim/stats URL, scraped to shim.csv.")
    ] = None,
    shim_pid: Annotated[
        int | None, typer.Option(help="trace: shim process id, for its CPU time.")
    ] = None,
    specdec: Annotated[
        str | None,
        typer.Option(
            help="W4: speculative-decoding label (none | ngram | eagle3) for the manifest."
        ),
    ] = None,
    family: Annotated[
        str | None,
        typer.Option(help="W4: model family of the cell (fp8 | q4b), pairs it with its none cell."),
    ] = None,
) -> None:
    """Warm up, sample power and /metrics, run inference-perf, adapt records, write manifest.json."""
    import json as _json

    from slo_lab.harness.stage import run_stage

    result = run_stage(
        run_dir=run_dir,
        cell=cell,
        model=model,
        base_url=base_url,
        metrics_url=metrics_url,
        seed=seed,
        kind=kind,
        rate_rps=rate_rps,
        duration_s=duration_s,
        concurrency=concurrency,
        num_requests=num_requests,
        warmup_requests=warmup_requests,
        discard_first_s=discard_first_s,
        inference_perf_bin=inference_perf_bin,
        engine_flags=_json.loads(engine_flags) if engine_flags else None,
        workers=workers,
        trace_file=trace_file,
        profile_path=profile,
        policy=policy,
        shim_stats_url=shim_stats_url,
        shim_pid=shim_pid,
        specdec=specdec,
        family=family,
    )
    if kind == "trace":
        phases = result.get("phase_summaries") or {}
        typer.echo(
            _json.dumps(
                {
                    "cell": cell,
                    "policy": policy,
                    "seed": seed,
                    "records": result.get("records"),
                    "arrivals": (result.get("trace") or {}).get("arrivals"),
                    "first_request_after_launch_s": result.get("first_request_after_launch_s"),
                    "time_to_recover_s": result.get("time_to_recover_s"),
                    "time_to_recover_attainment_s": result.get("time_to_recover_attainment_s"),
                    "shim_cpu_s": result.get("shim_cpu_s"),
                    "load_wall_s": result.get("load_wall_s"),
                    "inference_perf_returncode": result.get("inference_perf_returncode"),
                    **{
                        f"{name}_attainment": (p or {}).get("attainment")
                        for name, p in phases.items()
                    },
                    **{
                        f"{name}_rejection": (p or {}).get("rejection_rate")
                        for name, p in phases.items()
                    },
                }
            )
        )
        raise typer.Exit(code=0 if result.get("records") else 1)
    keys = (
        "cell",
        "kind",
        "rate_rps",
        "concurrency",
        "records",
        "window_records",
        "achieved_rps",
        "ttft_p50_s",
        "ttft_p95_s",
        "tpot_p50_s",
        "tpot_p95_s",
        "output_tok_per_s",
        "warmup_ttft_median_s",
        "inference_perf_returncode",
    )
    typer.echo(_json.dumps({k: result.get(k) for k in keys}))
    summary = result.get("summary") or {}
    if summary:
        typer.echo(
            _json.dumps(
                {
                    k: summary.get(k)
                    for k in (
                        "offered",
                        "met",
                        "attainment_offered",
                        "attainment_offered_ci95",
                        "rejection_rate",
                        "goodput_rps",
                    )
                }
            )
        )
    raise typer.Exit(code=0 if result.get("records") else 1)


@app.command("specdec-flags")
def specdec_flags(
    config: Annotated[Path, typer.Argument(help="config/specdec/<name>.yaml")],
) -> None:
    """Print the `--speculative-config` server argument for a spec-decode YAML ('' for none)."""
    from slo_lab.harness.specdec import speculative_config_flag

    typer.echo(speculative_config_flag(config))


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
        # Committed records may be gzip-compressed (ADR 0011); name each stage by its logical
        # records.jsonl so plain and compressed copies of the same stage are counted once.
        logical = {p.with_name("records.jsonl") for p in base.rglob("records.jsonl*")}
        for records in sorted(p for p in logical if evidence_path(p) is not None):
            n_runs += 1
            recs = filter_window(read_records_jsonl(records), 60.0)
            if not recs:
                # An exploratory stage shorter than the discard period is still evidence (it
                # carries its own manifest); it just contributes nothing to the windowed tables.
                typer.echo(
                    f"warning: {records.parent.relative_to(base).as_posix()}: "
                    "no records after the 60 s discard window"
                )
                continue
            s = summarise(recs)
            typer.echo(
                f"{area}/{records.parent.relative_to(base).as_posix()}: offered={s.offered} "
                f"attainment={s.attainment_offered:.3f} "
                f"[{s.attainment_offered_ci95[0]:.3f}, {s.attainment_offered_ci95[1]:.3f}] "
                f"rejection={s.rejection_rate:.3f}"
            )
    typer.echo(f"evidence runs with records: {n_runs}")
    if n_runs == 0:
        typer.echo("nothing to rebuild yet (W0 skeleton) - tables/plots/ledgers unchanged")

    index_path = root / "analysis" / "tables" / "index.json"
    if index_path.exists():
        from slo_lab.batch_analysis import rebuild_from_index

        rebuilt = rebuild_from_index(root, index_path)
        typer.echo(f"tables rebuilt from evidence: {', '.join(rebuilt) or '(none)'}")

    from slo_lab.quality import rebuild_paired_tables

    paired = rebuild_paired_tables(root, root / "analysis" / "tables" / "w2-quality-paired")
    if paired:
        typer.echo(f"paired quality table rebuilt: {', '.join(paired)} vs baseline")
    if problems:
        for p in problems:
            typer.echo(f"problem: {p}", err=True)
        raise typer.Exit(code=1)


def main() -> None:  # pragma: no cover - console entry
    app(prog_name="slo-lab")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
