# Runbook — WSL2 量測主機（W1 驗證版）

## 已驗證環境（2026-09-08）

- 宿主：Windows 11、RTX 4090 24 GB、driver 591.86；WSL2 kernel 6.6 系列（microsoft-standard-WSL2）；4090 在 WSL2 內可見（24564 MiB）
- WSL2 Ubuntu：Python 3.12.3、uv；vLLM 環境在 `~/vllm-slo-lab/.venv`（**不在本 repo 的 pyproject 依賴中**）：vLLM 0.28.0、torch 2.13.0+cu130
- **必要環境變數（W1 發現，缺一則 engine 起不來）**：
  - `VLLM_WSL2_ENABLE_PIN_MEMORY=1`：vLLM 在 WSL2 預設關閉 pinned memory（舊的效能保守設定），而 0.28 的 GPU model runner 以 UVA host-mapped buffer 存 token 表，`is_uva_available()` 因此回 False → `RuntimeError: UVA is not available`。這是 vLLM 自己提供的開關（`envs.py`），不是 patch；vLLM 註解稱 WSL2 上 pinned memory 有「小幅效能退化」，**本 lab 的所有數字都在此設定下量測，寫進 claim ceiling**。0.27.1 同樣需要此開關（已驗證）。
  - `VLLM_USE_FLASHINFER_SAMPLER=0`：FlashInfer 的 top-k/top-p sampler 在暖機時 JIT 編譯，WSL2 沒有 `nvcc` 與 `ninja` → `FileNotFoundError: 'ninja'`。關閉後走 torch 原生 sampler。
  - `HF_HUB_OFFLINE=1`：權重已在 `~/.cache/huggingface`（Qwen3-8B-FP8 兩個 shard，8.9 GB）。
- 一次 `vllm serve Qwen/Qwen3-8B-FP8 --max-model-len 4096 --gpu-memory-utilization 0.90 --max-num-seqs 64` 的實測：attention backend `FLASH_ATTN`；`Loading weights took 64.32 s`、`Model loading took 8.8 GiB / 74.7 s`；`GPU KV cache size: 83,024 tokens`（4,096 tokens/request 時最大並發 20.27x）；CUDA graph capture 3 s / 0.21 GiB；`/v1/chat/completions` 一次請求回覆正常（21 prompt + 35 completion tokens）；`/metrics` 露出 `vllm:num_requests_running`、`vllm:num_requests_waiting`（另有 `_by_reason{capacity|deferred}`）、`vllm:kv_cache_usage_perc`、`vllm:prompt_tokens_total`。
- FP8 警告：`Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal!`（4090 沒有 tuned config）。這會影響 FP8 cell 的吞吐，屬於**可調參數而非缺陷**；W2 決定是否為 4090 產生 kernel config，並在證據中標明用的是預設或調過的 config。
- 功耗與記憶體：`nvidia-smi --query-gpu` 在 WSL2 與 Windows 兩側都不支援（driver 591.86）；NVML（`nvidia-ml-py`）可讀：閒置 19.7 W／2,609 MiB（Windows 桌面本身占 VRAM），server 閒置待命 143 W／23,436 MiB。
- **NVML 在 WSL2 列不出 compute process**（server 占 23 GiB 時 `nvmlDeviceGetComputeRunningProcesses` 仍回空）。`quiet-gpu` 因此在此主機只能靠記憶體與 utilization 兩個準則，快照會記 `process_list_trustworthy: false`；預設記憶體門檻依實測閒置基線改為 3,072 MiB、utilization 門檻 5%。

## 每次 run 的順序（`harness/run.py` 尚未寫；W1 先手動）

1. 08:00 之後才開始（00:00–08:00 是另一個 cron 工作的時段）。
2. `uv run slo-lab quiet-gpu --out evidence/raw/<run_id>/quiet_gpu.json` — 有其他 compute process、既有記憶體占用 > 3,072 MiB、或 utilization > 5% 即退出 1，不得繼續（WSL2 上 process 準則無效，快照會標明）。
3. 在 vLLM 環境啟動 server（旗標見 `config/engine/common.yaml` + cell 檔；`HF_HUB_OFFLINE=1`、`VLLM_WSL2_ENABLE_PIN_MEMORY=1`、`VLLM_USE_FLASHINFER_SAMPLER=0`、`VLLM_CACHE_ROOT` 在 ext4）。
4. 若 policy 為 (ii)／(iii)：`uv run slo-lab shim --upstream http://localhost:8000 --policy hard_cap --capacity <C>`；policy (i) 亦走 `--policy passthrough` 以保持 shim 開銷一致。
5. `uv run slo-lab power-sample evidence/raw/<run_id>/power.csv --phase idle --duration-s 60`，之後 `--append --phase warmup`、`--append --phase measure`。
6. Warm-up 100 sequential request；inference-perf 依 `config/traffic/*.yaml` 跑；scrape `/metrics`。
7. 收集 → `records.jsonl`（adapter 未寫）→ `scripts/redact.py redact` 去敏 log → `manifest.json`。
8. `make reproduce`；`make audit-secrets`。

## 未完成

- `harness/run.py` 編排、inference-perf adapter、metrics scraper、`vllm bench serve` 交叉驗證、nonce prompt 產生器。
