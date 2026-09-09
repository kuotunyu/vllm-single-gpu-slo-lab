#!/usr/bin/env bash
# Create a Linux environment for the lab repo itself (shim, eval scripts) without touching the
# Windows .venv inside the checkout: the venv lives under ~/vllm-slo-lab/.venv-slolab.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
REPO="/mnt/d/AI-Portfolio/CC_github部隊/vllm-single-gpu-slo-lab"
export UV_PROJECT_ENVIRONMENT="$HOME/vllm-slo-lab/.venv-slolab"
cd "$REPO"
uv sync --frozen --extra gpu --python 3.12 2>&1 | tail -2
"$UV_PROJECT_ENVIRONMENT/bin/python" -c "import slo_lab, aiohttp; print('slo_lab importable; aiohttp', aiohttp.__version__)"
"$UV_PROJECT_ENVIRONMENT/bin/slo-lab" --help 2>&1 | head -3
