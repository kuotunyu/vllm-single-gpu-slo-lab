# ADR 0005 — W1 收尾：量測工具鏈在 WSL2 + 4090 上全部可用，A3 取消

- 日期：2026-09-09 凌晨
- 狀態：已採納；W1 驗證清單（規格 §4.3）全部結案
- 原始紀錄：`evidence/raw/w1/`（`matrix-load-check.jsonl`、`vllm-bench-serve-smoke.json`、`shim-overhead/*.json`、`inference-perf-smoke/*.json`、`tmmluplus-dryrun-20.json`）與 `evidence/metrics-names.txt`

## 清單結案

| # | 項目 | 結果 |
|---|---|---|
| 1 | vLLM 0.28 載入 Qwen3-8B-FP8 | 通過；需 `VLLM_WSL2_ENABLE_PIN_MEMORY=1` 與 `VLLM_USE_FLASHINFER_SAMPLER=0`（ADR 0002） |
| 2 | 功耗取樣 | NVML 可用；`nvidia-smi --query-gpu` 在此 driver 不可用（ADR 0002） |
| 3 | inference-perf 裸機模式 | 0.6.1 可跑；synthetic datagen 只支援 `api.type: completion`（chat 直接拋 Unsupported API type）；20 個 Poisson 請求：TTFT mean 112 ms／p95 303 ms，TPOT mean 16.7 ms，goodput 95%（ttft 1 s／tpot 50 ms 約束）；tokenizer 對合成語料印出「sequence length 1,306,158 > 131,072」警告，屬語料生成階段，不影響請求 |
| 4 | TMMLU+ 可取得且授權允許切片 | MIT；三個 200 題不重疊切片已凍結（`eval/tmmluplus/slices.json`，各 66 科） |
| 5 | 四種精度載入 | 通過（ADR 0004） |
| 6 | EAGLE-3 4B head 載入 | 通過（ADR 0004） |
| 7 | shim 開銷 | 見下節；passthrough shim 在 2 rps 下中位數 TTFT 與直連差在 ±2 ms 內 |
| 8 | `vllm bench serve` | 可跑（50 prompts @ 1 rps：TTFT mean 54.8 ms／p99 270 ms，TPOT mean 16.2 ms，126.5 tok/s） |
| 9 | quiet-GPU 閘門 | 改為記憶體＋utilization 準則；WSL2 上 process 準則無效並記錄（ADR 0002） |
| 10 | metrics 名稱凍結 | 從 live `/metrics` 抓 96 個名稱寫入 `evidence/metrics-names.txt`；規格列的 `num_requests_running`、`num_requests_waiting`、`kv_cache_usage_perc`、TTFT／ITL／E2E histogram、`prefix_cache_*`、`request_success_total` 全部在場，另有 `num_requests_waiting_by_reason{capacity|deferred}` 與 `external_prefix_cache_*` |
| 11 | Colab 20 題 dry run（A3） | **取消**，見下節 |

## Shim 開銷（`vllm bench serve`，40 prompts @ 2 rps，seed 7，兩輪交錯）

| 回合 | 路徑 | TTFT mean | TTFT median | TTFT p99 | TPOT mean | 成功 |
|---|---|---|---|---|---|---|
| r1 | 直連 8013 | 738.8 ms | 52.3 ms | 5,726.7 ms | 25.0 ms | 40/40 |
| r1 | shim 8021 | 59.8 ms | 47.4 ms | 304.2 ms | 16.9 ms | 40/40 |
| r2 | 直連 8013 | 81.9 ms | 48.0 ms | 659.2 ms | 16.9 ms | 40/40 |
| r2 | shim 8021 | 49.1 ms | 46.2 ms | 137.4 ms | 16.2 ms | 40/40 |

- 中位數 TTFT 直連 48–52 ms、經 shim 46–47 ms：**shim 沒有可量測的中位數開銷**（差異在噪音內，且方向相反）。shim 統計：admitted 91、completed 91、upstream_errors 0。
- 直連 r1 的 mean 739 ms／p99 5.7 s 是 server 剛起、第一批請求的暖機效應（同一 server 三分鐘後直連 r2 的 p99 只剩 659 ms）。這正是規格 §3 暖機規則（100 個 sequential request）存在的理由；**任何沒有暖機的回合都不得進入證據**。
- p99 在四輪間從 137 ms 到 5.7 s，說明 40 個請求太少，尾端統計要靠 W2 的完整 trace 與 n ≥ 3。
- 埠 8001 被本機另一個服務占用（`ss` 顯示 127.0.0.1:8001 LISTEN，非本 lab），shim 預設埠改為 8021；runbook 已更新。

## A3（Colab 全量 TMMLU+）取消

FP8 在 4090 上跑 20 題：14/20 正確（Wilson 95% 0.48–0.85，n 太小僅供 sanity）、0 個無法解析、concurrency 8 下 29.5 題/秒。以此估算：200 題切片約 7 秒，全量 19,680 題約 11 分鐘一個精度，四個精度一小時內完成。Colab 分流沒有必要，且 FP8 本來就不能在 A100 上等價評測（ADR 0003）。**規格 §11 決策 3 定案：A3 取消，全部品質評測在 4090 上做。** 小額付費加值只剩 A1（RunPod L4）。

## 對 W2 的決定

- 品質軌：每精度跑三個切片（600 題）＋全量一次；報 Wilson CI；prompt 與解析規則凍結於 `scripts/tmmluplus_eval.py`（`/no_think`、temperature 0、max_tokens 8）。
- 流量軌：inference-perf 用 completion API；`vllm bench serve` 交叉驗證用 chat API，兩者的 API 差異寫進 manifest。
- 每回合前 100 個 sequential 暖機請求，並把 r1/r2 這種「未暖機」對照保留為負面範例。
