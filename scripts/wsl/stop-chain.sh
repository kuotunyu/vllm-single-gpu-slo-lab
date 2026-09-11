#!/usr/bin/env bash
# Stop every W2 / W3 / W4 driver and everything it started (chain -> cell chain -> batch driver ->
# run-stage / inference-perf -> shim -> vllm server or fake engine -> samplers). Safe to run when
# nothing is running.
pkill -f "w2-night.sh" 2>/dev/null; pkill -f "w2-cell-chain.sh" 2>/dev/null
pkill -f "w2-chain.sh" 2>/dev/null; pkill -f "w2-refine.sh" 2>/dev/null; pkill -f "w2-followup.sh" 2>/dev/null
pkill -f "w3-night.sh" 2>/dev/null; pkill -f "w3-trace-chain.sh" 2>/dev/null
pkill -f "w4-night.sh" 2>/dev/null; pkill -f "w4-cell-chain.sh" 2>/dev/null
pkill -f "scripts/wsl/batch.sh" 2>/dev/null; pkill -f "slo-lab run-stage" 2>/dev/null
pkill -f "inference-perf" 2>/dev/null; pkill -f "slo-lab shim" 2>/dev/null
pkill -f ".venv/bin/vllm serve" 2>/dev/null; pkill -f "fake_vllm.py" 2>/dev/null
pkill -f "EngineCore" 2>/dev/null; pkill -f "io-sampler.sh" 2>/dev/null
sleep 4
echo "remaining:"
pgrep -af "w2-|w3-|w4-|wsl/batch|run-stage|inference-perf|slo-lab shim|vllm serve|fake_vllm|EngineCore|io-sampler" | grep -v pgrep || true
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
