# Runbook — WSL2 量測主機（W0 草稿）

## 已驗證環境（2026-09-03）

- 宿主：Windows 11、RTX 4090 24 GB、driver 591.86（4090 在 WSL2 內可見，24564 MiB）
- WSL2 Ubuntu：Python 3.12.3、uv
- 獨立 vLLM 環境（**不在本 repo 的 pyproject 依賴中**）：vLLM 0.28.0、torch 2.13.0+cu130；已載入模型並完成一次 completion
- 此 driver 下 `nvidia-smi --query-gpu=...` 不支援；功耗以 NVML 直讀（`uv sync --extra gpu` 安裝 `nvidia-ml-py`）
- 未驗證：NVML 在 WSL2 是否列得出 compute process（`quiet-gpu` 的核心假設；W1 清單第 9 項）

## 每次 run 的順序（`harness/run.py` 尚未寫；W1 先手動）

1. 08:00 之後才開始（00:00–08:00 是另一個 cron 工作的時段）。
2. `uv run slo-lab quiet-gpu --out evidence/raw/<run_id>/quiet_gpu.json` — 有任何其他 compute process 或既有記憶體占用 > 門檻即退出 1，不得繼續。
3. 在 vLLM 環境啟動 server（旗標見 `config/engine/common.yaml` + cell 檔；`HF_HUB_OFFLINE=1`、`VLLM_CACHE_ROOT` 在 ext4）。
4. 若 policy 為 (ii)／(iii)：`uv run slo-lab shim --upstream http://localhost:8000 --policy hard_cap --capacity <C>`；policy (i) 亦走 `--policy passthrough` 以保持 shim 開銷一致。
5. `uv run slo-lab power-sample evidence/raw/<run_id>/power.csv --phase idle --duration-s 60`，之後 `--append --phase warmup`、`--append --phase measure`。
6. Warm-up 100 sequential request；inference-perf 依 `config/traffic/*.yaml` 跑；scrape `/metrics`。
7. 收集 → `records.jsonl`（adapter 未寫）→ `scripts/redact.py redact` 去敏 log → `manifest.json`。
8. `make reproduce`；`make audit-secrets`。

## 未完成

- `harness/run.py` 編排、inference-perf adapter、metrics scraper、`vllm bench serve` 交叉驗證、nonce prompt 產生器。
