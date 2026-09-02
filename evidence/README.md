# Evidence contract（設計規格 §9）

`evidence/raw/**`、`evidence/runpod/**`、`evidence/quant/**` **全部提交**；`make reproduce` 只讀這裡與 `config/cost.yaml`，在無 GPU、無網路的 CI 上重算 ledger、CI、表與圖，diff 必須為空。

每個 run 一個目錄 `evidence/raw/<run_id>/`（RunPod 同 contract 放 `evidence/runpod/`）：

| 檔案 | 內容 | 狀態 |
|---|---|---|
| `manifest.json` | vLLM 版本、模型 revision、旗標全文、GPU／driver／CUDA／WSL2 kernel、seed、trace、時間窗 | W1 定案欄位 |
| `records.jsonl` | 本 package 的 canonical per-request 記錄（`slo_lab.slo.RequestRecord`）；由 inference-perf raw JSON 轉出（adapter 為 W1 工作） | 格式已定，adapter 未寫 |
| `inference-perf/*.json` | loadgen 原始輸出 | W1 |
| `power.csv` | `slo-lab power-sample` 的 1 s NVML 取樣（idle／warmup／measure 三段） | 格式已定 |
| `quiet_gpu.json` | `slo-lab quiet-gpu` 快照：NVML 裝置狀態、compute process 清單、best-effort `nvidia-smi` 全文 | 格式已定 |
| `metrics/*.prom` | `/metrics` scrape（名稱見 `metrics-names.txt`） | W1 |
| `vllm.log` | 去敏後的 server log（`scripts/redact.py redact`） | W1 |

排除：權重、`VLLM_CACHE_ROOT`、`.env`、SSH 金鑰、未去敏 log、任何含 IP／pod id／hostname 的原始輸出。`make audit-secrets` 在 CI 與 pre-commit 掃描。

目前狀態：**W0，沒有任何 run**。
