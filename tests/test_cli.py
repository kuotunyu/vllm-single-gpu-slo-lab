import json
from pathlib import Path

from typer.testing import CliRunner

from slo_lab import __version__, quiet_gpu
from slo_lab.cli import app
from slo_lab.nvml import NvmlUnavailableError
from slo_lab.slo import RequestRecord, write_records_jsonl

REPO = Path(__file__).resolve().parents[1]
runner = CliRunner()


def _records(n=10, ttft=0.2, tpot=0.02, rejected=0):
    recs = [
        RequestRecord(
            request_id=f"r{i}",
            offered_at_s=float(i),
            outcome="ok",
            ttft_s=ttft,
            e2e_s=ttft + tpot * 10,
            output_tokens=11,
        )
        for i in range(n)
    ]
    recs += [
        RequestRecord(request_id=f"x{i}", offered_at_s=float(n + i), outcome="rejected_429")
        for i in range(rejected)
    ]
    return recs


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == __version__


def test_attainment_command(tmp_path):
    path = tmp_path / "records.jsonl"
    write_records_jsonl(path, _records(n=8, rejected=2))
    result = runner.invoke(
        app, ["attainment", str(path), "--window-start-s", "0", "--window-s", "10"]
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["offered"] == 10 and data["met"] == 8
    assert data["attainment_offered"] == 0.8
    assert data["goodput_rps"] == 0.8


def test_attainment_command_empty_window(tmp_path):
    path = tmp_path / "records.jsonl"
    write_records_jsonl(path, _records(n=3))
    result = runner.invoke(app, ["attainment", str(path), "--window-start-s", "60"])
    assert result.exit_code == 1


def test_capacity_command_from_reduced_cells(tmp_path):
    sweep = tmp_path / "sweep.jsonl"
    rows = [
        {"offered_rate": 1.0, "seed": 0, "attainment": 0.99},
        {"offered_rate": 1.0, "seed": 1, "attainment": 0.97},
        {"offered_rate": 2.0, "seed": 0, "attainment": 0.80},
        {"offered_rate": 2.0, "seed": 1, "attainment": 0.99},
    ]
    sweep.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    result = runner.invoke(app, ["capacity", str(sweep)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["r_slo"] == 1.0


def test_capacity_command_from_raw_records_prints_grid(tmp_path):
    for rate, ttft in ((1.0, 0.2), (2.0, 1.5)):
        for seed in (0, 1):
            write_records_jsonl(tmp_path / f"r{rate}-s{seed}.jsonl", _records(ttft=ttft))
    sweep = tmp_path / "sweep.jsonl"
    rows = [
        {"offered_rate": rate, "seed": seed, "records": f"r{rate}-s{seed}.jsonl"}
        for rate in (1.0, 2.0)
        for seed in (0, 1)
    ]
    sweep.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    result = runner.invoke(app, ["capacity", str(sweep), "--window-start-s", "0"])
    assert result.exit_code == 0, result.output
    head, grid = result.output.split("\n\n", 1)
    assert json.loads(head)["r_slo"] == 1.0
    assert "| TTFT \\ TPOT |" in grid
    assert "| 2 s | 2 req/s | 2 req/s | 2 req/s |" in grid  # TTFT 2 s admits the 1.5 s cell


def test_cost_command_with_explicit_power():
    result = runner.invoke(
        app,
        [
            "cost",
            "--config",
            str(REPO / "config" / "cost.yaml.example"),
            "--output-tok-per-s",
            "100",
            "--p-avg-w",
            "300",
            "--peak-output-tok-per-s",
            "400",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["config_status"] == "owner_input_pending"
    assert data["utilisation_u"] == 0.25
    assert data["wh_per_m_output_tok_at_r_slo"] > 0


def test_cost_command_with_power_csv(tmp_path):
    csv_path = tmp_path / "power.csv"
    lines = ["timestamp_iso,t_s,power_w,util_gpu_pct,mem_used_mib,clocks_sm_mhz,temp_c,phase"]
    lines += [f"t,{i},20,,,,,idle" for i in range(3)]
    lines += [f"t,{60 + i},300,,,,,measure" for i in range(5)]
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "cost",
            "--config",
            str(REPO / "config" / "cost.yaml.example"),
            "--output-tok-per-s",
            "100",
            "--power-csv",
            str(csv_path),
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["power_window"]["p_avg_w"] == 300.0
    assert data["idle_baseline_w"] == 20.0
    assert data["p_avg_net_w"] == 280.0


def test_cost_command_needs_power():
    result = runner.invoke(
        app,
        ["cost", "--output-tok-per-s", "1", "--config", str(REPO / "config" / "cost.yaml.example")],
    )
    assert result.exit_code == 1


def test_quiet_gpu_command_refuses_when_busy(monkeypatch, tmp_path):
    fake = {
        "gpu_name": "fake",
        "memory_used_mib": 50.0,
        "compute_processes": [{"pid": 99, "name": "other", "used_memory_mib": 40.0}],
    }
    monkeypatch.setattr(quiet_gpu, "snapshot", lambda index=0: fake)
    out = tmp_path / "quiet_gpu.json"
    result = runner.invoke(app, ["quiet-gpu", "--out", str(out)])
    assert result.exit_code == 1
    assert json.loads(out.read_text(encoding="utf-8"))["decision"]["ok"] is False
    allowed = runner.invoke(app, ["quiet-gpu", "--allow-pid", "99"])
    assert allowed.exit_code == 0


def test_quiet_gpu_command_exit_2_without_nvml(monkeypatch):
    def boom(index=0):
        raise NvmlUnavailableError("NVML is unavailable")

    monkeypatch.setattr(quiet_gpu, "snapshot", boom)
    result = runner.invoke(app, ["quiet-gpu"])
    assert result.exit_code == 2
    assert "NVML is unavailable" in result.output


def test_shim_command_requires_capacity_for_cap_policies():
    result = runner.invoke(
        app, ["shim", "--upstream", "http://localhost:8000", "--policy", "hard_cap"]
    )
    assert result.exit_code == 1


def test_reproduce_lite_on_the_repo_rebuilds_the_indexed_tables():
    result = runner.invoke(app, ["reproduce-lite", "--root", str(REPO)])
    assert result.exit_code == 0, result.output
    # W2 evidence is nested (raw/w2/<cell>/<batch>/seed-N/<stage>/records.jsonl)
    assert "raw/w2/fp8/closed-loop/seed-1/cl-conc-256: offered=" in result.output
    assert "evidence runs with records: 0" not in result.output
    assert "tables rebuilt from evidence: w2-fp8-closed-loop, w2-fp8-closed-loop-exploratory" in (
        result.output
    )


def test_reproduce_lite_rebuilds_attainment_for_a_run(tmp_path):
    root = tmp_path
    (root / "config").mkdir()
    (root / "analysis" / "ledger").mkdir(parents=True)
    (root / "analysis" / "ledger" / "runs.csv").write_text("run_id\n", encoding="utf-8")
    run_dir = root / "evidence" / "raw" / "2026-09-03-fp8-native-r1-s0"
    write_records_jsonl(run_dir / "records.jsonl", _records(n=70, rejected=5))
    result = runner.invoke(app, ["reproduce-lite", "--root", str(root)])
    assert result.exit_code == 0, result.output
    assert "raw/2026-09-03-fp8-native-r1-s0: offered=15" in result.output
    assert "evidence runs with records: 1" in result.output
    assert "tables rebuilt" not in result.output  # no analysis/tables/index.json in this tree


def test_make_trace_writes_a_seeded_burst25_trace(tmp_path):
    out = tmp_path / "trace-seed-1.csv"
    args = ["make-trace", "--rate-ref", "11.4", "--seed", "1", "--out", str(out)]
    args += ["--profile", str(REPO / "config" / "traffic" / "burst25.yaml")]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    info = json.loads(result.output)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert info["arrivals"] == len(lines) - 1
    assert [p["phase"] for p in info["phases"]] == ["pre", "burst", "recovery"]
    assert sum(p["arrivals"] for p in info["phases"]) == info["arrivals"]
    other = [a if a != str(out) else str(tmp_path / "b.csv") for a in args]
    again = runner.invoke(app, other)
    assert json.loads(again.output)["sha256"] == info["sha256"]
