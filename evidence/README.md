# Evidence contract（設計規格 §9）

`evidence/raw/**`、`evidence/runpod/**`、`evidence/quant/**` **全部提交**；`make reproduce` 只讀這裡與 `config/cost.yaml`，在無 GPU、無網路的 CI 上重算 ledger、CI、表與圖，diff 必須為空。

每個 run 一個目錄 `evidence/raw/<run_id>/`（RunPod 同 contract 放 `evidence/runpod/`）：

| 檔案 | 內容 | 狀態 |
|---|---|---|
| `manifest.json` | vLLM 版本、模型 revision、旗標全文、GPU／driver／CUDA／WSL2 kernel、seed、trace、時間窗 | W1 定案欄位 |
| `records.jsonl.gz` | 本 package 的 canonical per-request 記錄（`slo_lab.slo.RequestRecord`），由 inference-perf raw JSON 轉出；以決定性 gzip 提交，`open_evidence_text` 透明讀取（ADR 0011） | 已定 |
| `inference-perf/*.json` | loadgen 原始輸出 | W1 |
| `power.csv` | `slo-lab power-sample` 的 1 s NVML 取樣（idle／warmup／measure 三段） | 格式已定 |
| `quiet_gpu.json` | `slo-lab quiet-gpu` 快照：NVML 裝置狀態、compute process 清單、best-effort `nvidia-smi` 全文 | 格式已定 |
| `metrics/*.prom` | `/metrics` scrape（名稱見 `metrics-names.txt`） | W1 |
| `vllm.log.gz` | 去敏後的 server log（`scripts/redact.py redact`），gzip 提交；同目錄之後的 session 帶 tag，例如 `vllm-refine-0.80.log.gz` | 已定 |

排除：權重、`VLLM_CACHE_ROOT`、`.env`、SSH 金鑰、未去敏 log、任何含 IP／pod id／hostname 的原始輸出。`make audit-secrets` 在 CI 與 pre-commit 掃描。

目前狀態（2026-09-09）：`raw/w1/` 為驗證證據；`raw/w2/fp8/closed-loop-exploratory/seed-1/` 與 `raw/w2/fp8/closed-loop/seed-1/` 為 W2 第一步的兩次 closed-loop 掃描（每 stage 一個目錄：`manifest.json`、`records.jsonl`、`power.csv`／`power-warmup.csv`、`metrics.csv`、`warmup-ttft.json`、`inference-perf.yaml`／`.log`、`ipf/{config,summary,stage_0}`；批次層 `quiet_gpu.json`、`vllm.log`、`io-pressure.log`）。10+ MB 的 `per_request_lifecycle_metrics.json` 不提交，manifest 記其 sha256，`records.jsonl` 是由它轉出的 canonical 形式。正式掃描 c = 1–96 的 stage 被桌面 VRAM 分頁污染，保留但由分析器標記排除（ADR 0006／0007）。之後加入：`closed-loop-rerun-0.90-paging/`（分頁證據）、`closed-loop-v3/`（0.82，全乾淨）、`open-loop/seed-{1,2,3}/`（11 rates）、`tmmluplus/`（三切片逐題輸出）、`win-vram-2026-09-09.log`（Windows 端 VRAM 取樣）。`analysis/tables/index.json` 列出每張表對應的證據目錄，`make reproduce` 逐一重建。

W2 結案（2026-09-11）加入：`raw/w2/{awq,gptq,bf16}/`（每 cell `closed-loop/seed-1/`、`open-loop/seed-{1,2,3}/`、`tmmluplus/`）、`raw/w2/fp8-mbt8192/`（`--max-num-batched-tokens` 8192 對照組）、`raw/w2/fp8/crosscheck/`（`vllm bench serve` 的純量摘要；生成文字不提交，只記原檔 sha256）、`raw/w2/fp8/tmmluplus/full.json`（TMMLU+ 全集）、`raw/w2/win-vram-2026-09-10.log`（W2 全程的 Windows 端 VRAM 取樣）。批次層的 log 以伺服器 session 為單位：主量測是 `vllm.log.gz`／`quiet_gpu.json`，同一目錄之後的 session（膝點補點）帶 tag，例如 `vllm-refine-0.80.log.gz`、`quiet_gpu-refine-0.80.json`；`io-pressure.log` 是整個目錄的連續紀錄。有三個 session 的 server log 在提交前就被覆蓋，清單見 ADR 0009 缺陷 6。

W3 結案（2026-09-11）加入：`raw/w3/{fp8,bf16}/trace/seed-{1,2,3}/`，每個 seed 目錄有重播的到達時間檔 `trace-seed-N.csv.gz`（與 make-trace 的摘要 `trace-seed-N.json`），以及每個策略一個 `trace-<policy>/`（`records.jsonl.gz`、`metrics.csv`、`shim.csv`、`power.csv`、`manifest.json`）；`raw/w3/fp8-smoke/` 是開跑前的 GPU smoke，不進結果表；`raw/w3/win-vram-2026-09-11.log` 是 W3 全程的 Windows 端 VRAM 取樣。協定見 ADR 0012，結果見 ADR 0013。

圖（2026-09-12 起）：`evidence/plots/*.svg` 由 `slo-lab reproduce-lite` 從 `analysis/tables/` 與 `evidence/raw/` 重建（`slo_lab.plots`，純 Python 產生 SVG，無 matplotlib），`make reproduce` 一併 diff：`w2-attainment-vs-rate.svg`（四精度、3 seeds 取最小）、`w3-queue-timeline-{fp8,bf16}.svg`（seed 1 三個策略的 vLLM waiting + shim waiting，以 `t_mono` 對齊）；W4 量測後加 `w4-tpot-vs-rate-<family>.svg` 與 `w4-attainment-vs-rate-<family>.svg`。

W4（2026-09-12）加入：`raw/w4/<cell>/closed-loop/seed-1/` 與 `raw/w4/<cell>/open-loop/seed-{1,2,3}/`（cell 為 `fp8-none`、`fp8-ngram`、`q4b-none`、`q4b-eagle3`、`q4b-ngram`）。open-loop 的三個 seed 共用一個伺服器 session，session 層的 `vllm.log.gz`、`quiet_gpu.json`、`io-pressure.log` 只在 `seed-1`；每段 manifest 多了 `specdec`、`family`、`spec_decode`（草稿與接受計數）、`preemptions` 與 shim 計數，`metrics.csv` 多四欄。被可疑規則排除的段（`fp8-ngram` 的 c = 256 與 30／36 rps：cell 在 256 並行時超出 0.82 預算而分頁）仍提交，由分析器標記排除；被作廢重跑的段不提交。`raw/w4/smoke/` 是開跑前的 GPU smoke，不進結果表；`raw/w4/win-vram-2026-09-12.log` 是全程的 Windows 端顯存取樣。協定 ADR 0015，結果 ADR 0016。

W5（2026-09-12）加入：`raw/w2/reparse-compare-2026-09-12.json`，W2 的 243 段以伺服器 token 數重解析後與提交紀錄的逐段比較（TPOT p95、attainment、計數不同的請求數）與各 cell 的 r_SLO 舊／新；全部不變，W2 紀錄維持原樣（ADR 0017）。

Ledger（2026-09-12）：`analysis/ledger/runs.csv` 由 `reproduce-lite` 從 `analysis/tables/index.json` 列的批次目錄下的每個 `manifest.json` 重建（`slo_lab.ledger`），每段一列：日期、精度、admission 策略、spec-decode、流量、seed、狀態（取自重建後的表的可疑旗標與原因）、證據路徑。`cost.csv` 與 `spend.csv` 維持只有表頭：4090 不計 $（ADR 0010），沒有用過付費算力（ADR 0014）。
