#!/usr/bin/env bash
# Stop every W2 driver and everything it started (chain -> cell chain -> batch driver -> run-stage /
# inference-perf -> vllm server -> samplers). Safe to run when nothing is running.
pkill -f "w2-night.sh" 2>/dev/null; pkill -f "w2-cell-chain.sh" 2>/dev/null
pkill -f "w2-chain.sh" 2>/dev/null; pkill -f "w2-refine.sh" 2>/dev/null; pkill -f "w2-followup.sh" 2>/dev/null
pkill -f "scripts/wsl/batch.sh" 2>/dev/null; pkill -f "slo-lab run-stage" 2>/dev/null
pkill -f "inference-perf" 2>/dev/null; pkill -f ".venv/bin/vllm serve" 2>/dev/null
pkill -f "EngineCore" 2>/dev/null; pkill -f "io-sampler.sh" 2>/dev/null
sleep 4
echo "remaining:"
pgrep -af "w2-|wsl/batch|run-stage|inference-perf|vllm serve|EngineCore|io-sampler" | grep -v pgrep || true
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
