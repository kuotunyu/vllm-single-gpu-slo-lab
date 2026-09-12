"""Every shipped YAML parses and the traffic/admission files say what the spec says."""

from pathlib import Path

import pytest
import yaml

from slo_lab.admission import BoundedQueue, HardCap, Passthrough, policy_from_config

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config"


def load(rel: str):
    return yaml.safe_load((CONFIG / rel).read_text(encoding="utf-8"))


def test_all_yaml_files_parse():
    files = [*sorted(CONFIG.rglob("*.yaml")), CONFIG / "cost.yaml.example"]
    assert len(files) >= 17
    for path in files:
        assert isinstance(yaml.safe_load(path.read_text(encoding="utf-8")), dict), path


def test_burst25_headline_trace_matches_spec_proposal():
    cfg = load("traffic/burst25.yaml")
    assert cfg["status"] == "proposal"
    assert cfg["rate_reference"] == "r_sat"
    stages = [(s["rate_multiplier"], s["duration_s"]) for s in cfg["stages"]]
    assert stages == [(0.5, 300), (1.5, 300), (0.5, 900)]
    assert sum(d for _, d in stages) == 25 * 60
    assert cfg["prompt"] == {
        "input_tokens": 108,
        "output_tokens": 132,
        "nonce_prefix": True,
        "ignore_eos": True,
        "thinking": False,
    }
    assert cfg["client_timeout_s"] == 300
    assert cfg["warmup_requests"] == 100
    assert cfg["seeds"] is None  # frozen in preregistration at W1


def test_open_and_closed_loop_grids():
    open_loop = load("traffic/open_loop_sweep.yaml")
    # 7 preregistered multipliers plus the 0.55-0.70 refinement of ADR 0008
    assert open_loop["rate_multipliers"] == [
        0.25,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        1.0,
        1.25,
        1.5,
        2.0,
    ]
    assert open_loop["status"] == "frozen"
    assert open_loop["stage_duration_s"] == 300 and open_loop["discard_first_s"] == 60
    closed = load("traffic/closed_loop.yaml")
    # frozen 2026-09-09 (ADR 0006): grid extended to the FP8 --max-num-seqs of 256
    assert closed["concurrency"] == [1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256]
    assert closed["stage_duration_s"] == 180 and closed["discard_first_s"] == 60
    assert closed["status"] == "frozen"


@pytest.mark.parametrize(
    ("name", "kind"),
    [("native", Passthrough), ("cap429", HardCap), ("bounded", BoundedQueue)],
)
def test_admission_configs_build_policies(name, kind):
    cfg = load(f"admission/{name}.yaml")
    if cfg.get("capacity") is None and name != "native":
        cfg["capacity"] = 8  # C is only known after the W2 closed-loop sweep
    policy = policy_from_config(cfg)
    assert isinstance(policy, kind)
    if isinstance(policy, BoundedQueue):
        assert policy.queue_limit == policy.capacity and policy.timeout_s == 1.0


def common_env_has_wsl2_switches() -> bool:
    env = load("engine/common.yaml")["env"]
    return (
        env.get("VLLM_WSL2_ENABLE_PIN_MEMORY") == "1"
        and env.get("VLLM_USE_FLASHINFER_SAMPLER") == "0"
    )


def test_engine_configs_name_the_spec_models():
    assert load("engine/bf16.yaml")["model"] == "Qwen/Qwen3-8B"
    assert load("engine/fp8.yaml")["model"] == "Qwen/Qwen3-8B-FP8"
    assert load("engine/awq.yaml")["model"] == "Qwen/Qwen3-8B-AWQ"
    gptq = load("engine/gptq_int4.yaml")
    assert gptq["model"] == "JunHowie/Qwen3-8B-GPTQ-Int4"  # W1 decision, ADR 0003
    assert gptq["optional"] is False
    assert common_env_has_wsl2_switches()
    assert load("engine/qwen3_4b.yaml")["model"] == "Qwen/Qwen3-4B"
    common = load("engine/common.yaml")
    assert common["max_num_batched_tokens"] == 2048
    assert common["gpu_memory_utilization"] == 0.85
    assert common["kv_cache_dtype"] == "auto"
