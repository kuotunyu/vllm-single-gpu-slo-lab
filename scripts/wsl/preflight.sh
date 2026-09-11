#!/usr/bin/env bash
# Pre-flight for an unattended chain (plan 2026-09-10 Task 0): WSL up, GPU idle, weights cached,
# venvs importable, disk free, no leftovers. Prints only; the go/no-go rules live in the plan.
set -uo pipefail
export HF_HUB_OFFLINE=1
REPO="/mnt/d/AI-Portfolio/CC_github部隊/vllm-single-gpu-slo-lab"
echo "== time =="; date
echo "== gpu =="; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader
echo "== quiet-gpu =="; "$HOME/vllm-slo-lab/.venv-slolab/bin/slo-lab" quiet-gpu --out /tmp/preflight-quiet.json | head -4
echo "== weights in HF cache =="
for m in models--Qwen--Qwen3-8B-AWQ models--JunHowie--Qwen3-8B-GPTQ-Int4 models--Qwen--Qwen3-8B models--Qwen--Qwen3-8B-FP8 models--Qwen--Qwen3-4B models--AngelSlim--Qwen3-4B_eagle3; do
  d="$HOME/.cache/huggingface/hub/$m/snapshots"
  if [ -d "$d" ]; then echo "ok  $m ($(du -sh "$d" | cut -f1))"; else echo "MISSING $m"; fi
done
echo "== venvs =="
"$HOME/vllm-slo-lab/.venv/bin/python" -c "import vllm; print('vllm', vllm.__version__)"
"$HOME/vllm-slo-lab/.venv-slolab/bin/python" -c "import slo_lab.batch_analysis, slo_lab.harness.stage; print('slo_lab ok')"
"$HOME/vllm-slo-lab/.venv-loadgen/bin/inference-perf" --help >/dev/null 2>&1 && echo "inference-perf ok"
echo "== disk =="; df -h /home | tail -1; du -sh "$HOME/vllm-slo-lab/runs-w2" 2>/dev/null
echo "== eval set =="; wc -l "$REPO/eval/tmmluplus/full.jsonl"
echo "== leftovers =="; pgrep -af "vllm serve|inference-perf|w2-|w3-|w4-|wsl/batch|slo-lab shim|fake_vllm" | grep -v pgrep || echo "none"
echo "== io pressure =="; head -1 /proc/pressure/io
