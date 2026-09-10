# Runbook — WSL2 量測主機（W1 驗證版）

## 已驗證環境（2026-09-08）

- 宿主：Windows 11、RTX 4090 24 GB、driver 591.86；WSL2 kernel 6.6 系列（microsoft-standard-WSL2）；4090 在 WSL2 內可見（24564 MiB）
- WSL2 Ubuntu：Python 3.12.3、uv；vLLM 環境在 `~/vllm-slo-lab/.venv`（**不在本 repo 的 pyproject 依賴中**）：vLLM 0.28.0、torch 2.13.0+cu130
- **必要環境變數（W1 發現，缺一則 engine 起不來）**：
  - `VLLM_WSL2_ENABLE_PIN_MEMORY=1`：vLLM 在 WSL2 預設關閉 pinned memory（舊的效能保守設定），而 0.28 的 GPU model runner 以 UVA host-mapped buffer 存 token 表，`is_uva_available()` 因此回 False → `RuntimeError: UVA is not available`。這是 vLLM 自己提供的開關（`envs.py`），不是 patch；vLLM 註解稱 WSL2 上 pinned memory 有「小幅效能退化」，**本 lab 的所有數字都在此設定下量測，寫進 claim ceiling**。0.27.1 同樣需要此開關（已驗證）。
  - `VLLM_USE_FLASHINFER_SAMPLER=0`：FlashInfer 的 top-k/top-p sampler 在暖機時 JIT 編譯，WSL2 沒有 `nvcc` 與 `ninja` → `FileNotFoundError: 'ninja'`。關閉後走 torch 原生 sampler。
  - `HF_HUB_OFFLINE=1`：權重已在 `~/.cache/huggingface`（Qwen3-8B-FP8 兩個 shard，8.9 GB）。
- 一次 `vllm serve Qwen/Qwen3-8B-FP8 --max-model-len 4096 --gpu-memory-utilization 0.90 --max-num-seqs 64` 的實測：attention backend `FLASH_ATTN`；`Loading weights took 64.32 s`、`Model loading took 8.8 GiB / 74.7 s`；`GPU KV cache size: 83,024 tokens`（4,096 tokens/request 時最大並發 20.27x）；CUDA graph capture 3 s / 0.21 GiB；`/v1/chat/completions` 一次請求回覆正常（21 prompt + 35 completion tokens）；`/metrics` 露出 `vllm:num_requests_running`、`vllm:num_requests_waiting`（另有 `_by_reason{capacity|deferred}`）、`vllm:kv_cache_usage_perc`、`vllm:prompt_tokens_total`。
- **載入時間不是證據**：同一份 FP8 權重，page cache 溫熱時 64 s；在 30 GB 下載剛把 cache 沖掉、且 WSL 虛擬磁碟被其他工作（Docker、掃描）競爭時，AWQ 5.8 GB 光讀第一個 shard 就 106 s（io pressure 50–60%，EngineCore 停在 `folio_wait_bit_common`）。冷啟動分段記帳（規格 §5.5）只能用「第二次載入」或明確標示 cache 狀態的量測；每個 run 的 manifest 記錄 `/proc/pressure/io` 與載入前是否預熱。
- FP8 警告：`Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal!`（4090 沒有 tuned config）。這會影響 FP8 cell 的吞吐，屬於**可調參數而非缺陷**；W2 決定是否為 4090 產生 kernel config，並在證據中標明用的是預設或調過的 config。**W2 決定：不產生**，所有 W2 FP8 證據都是預設 config（server log 保留此警告，ADR 0009）。
- 功耗與記憶體：`nvidia-smi --query-gpu` 在 WSL2 與 Windows 兩側都不支援（driver 591.86）；NVML（`nvidia-ml-py`）可讀：閒置 19.7 W／2,609 MiB（Windows 桌面本身占 VRAM），server 閒置待命 143 W／23,436 MiB。
- **NVML 在 WSL2 列不出 compute process**（server 占 23 GiB 時 `nvmlDeviceGetComputeRunningProcesses` 仍回空）。`quiet-gpu` 因此在此主機只能靠記憶體與 utilization 兩個準則，快照會記 `process_list_trustworthy: false`；預設記憶體門檻依實測閒置基線改為 3,072 MiB、utilization 門檻 10%（5 × 1 s 取樣平均；單次讀值在閒置桌面上會出現 9% 的假警報）。

## W1 結案狀態（2026-09-09）

規格 §4.3 的驗證清單全部完成，細節與數字在 ADR 0002–0005。重點：五個 cell 皆可服務（ADR 0004）；inference-perf 0.6.1 裸機可跑但只接受 completion API；`vllm bench serve` 可跑；passthrough shim 在 2 rps 下無可量測中位數開銷，**埠用 8021**（8001 被本機其他服務占用）；TMMLU+ 三切片凍結、FP8 20 題 dry run 29.5 題/秒，因此 A3（Colab）取消；`evidence/metrics-names.txt` 已由 live scrape 凍結（96 個名稱）。repo 本身的 Linux 環境在 `~/vllm-slo-lab/.venv-slolab`（`UV_PROJECT_ENVIRONMENT` 指向它再 `uv sync --frozen`），與 checkout 內的 Windows `.venv` 互不干擾；load generator 在 `~/vllm-slo-lab/.venv-loadgen`。

## W2 狀態（2026-09-09）與 4090 共用租戶

FP8 完成：closed-loop（ADR 0006／0007：r_sat = 41.4 rps at 0.82，下界；C = 256）、open-loop 11 點 × 3 seeds（ADR 0008：r_SLO = 26.2 rps；膝點 26–31 rps，32.75 rps 起 attainment 崩到 0.1–0.6，43.7 rps 起 0）、TMMLU+ 三切片（366/600 = 0.610）。一次 FP8 全套（closed 11 點 + open 11 點 × 3 + TMMLU+）約 6 小時 GPU；open-loop 過載點（≥ 1.25 × r_sat）每點 10–12 分鐘，其中一半是 inference-perf 對 1.5–2.6 萬筆 request 的收尾（純 CPU、GPU 閒置）。`scripts/wsl/w2-chain.sh`、`w2-refine.sh`（只跑缺的 rate，可續跑）、`w2-followup.sh` 是實際用過的驅動。正式掃描 c = 1–96 在 02:22–03:45 被本機另一個 GPU 工作分時占用（NVML util 94–98%、時脈全速、功耗卻只有 170–184 W，吞吐減半且震盪），全部作廢。**`quiet-gpu` 只在 batch 開頭把關，擋不住中途出現的租戶**，因此：

- 每個 stage 的 `manifest.json` 有 `power_window.w_per_util_point`（乾淨 ≥ 2.3 且隨 concurrency 上升；污染 ≈ 1.8）與 re-warm 的單流 TPOT probe（`probe_tpot_median_s`，乾淨 18–19 ms）；`scripts/analyze_batch.py` 把可疑 stage 排除並列出 `suspect_concurrencies_excluded`。可疑 stage 一律重跑。
- **根因（ADR 0007）**：不是別的 compute 工作，是桌面程式把 WDDM 的 total committed VRAM 推過 24,564 MiB 實體——vLLM 拿 0.90 時桌面只剩約 0.5 GiB，dwm／Firefox／Chrome 一活動 VidMm 就分頁，`System` copy engine 就是分頁流量。**預算一律 0.82**（`GPU_MEM_UTIL`），manifest 的 `host_before/after.windows_gpu_memory.committed_mb` 超過實體即污染。Windows 端查租戶用 PowerShell 的 `GPU Engine` / `GPU Adapter Memory` / `GPU Process Memory` 計數器（`nvidia-smi.exe` 只列 C+G 程序名）。
- 長批次前仍問一次「其他 session 有沒有在用 GPU」；規格寫的「00:00–08:00 是 SOP cron」目前並不存在（`crontab -l` 空）。
- 磁碟飽和另有一種症狀（W1／smoke：整批 request 出現相同的 1–3 s TTFT，`/proc/pressure/io` full > 50%），`scripts/wsl/io-sampler.sh` 會每 10 s 記到批次目錄的 `io-pressure.log`。

## 整夜無人值守（`scripts/wsl/w2-night.sh`，2026-09-09 起）

執行計畫（逐步、含失敗處置與收尾）：`docs/superpowers/plans/2026-09-10-w2-overnight-run.md`。起跑前 `scripts/wsl/preflight.sh`，停止一切 `scripts/wsl/stop-chain.sh`。

```bash
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-bench -- bash /mnt/d/.../scripts/wsl/w2-night.sh     # DRY=1 只印計畫
```

`w2-night.sh` 依序呼叫 `w2-cell-chain.sh <cell> <model> <max_num_seqs> "<closed grid>"`：closed-loop → 由分析取 r_sat → open-loop 11 個 rate × seeds 1–3 → 可疑 stage 隔離到 `runs-w2/*/quarantine/` 並重跑（最多兩輪）→ TMMLU+ 三切片與全集（`eval/tmmluplus/full.jsonl`，19,680 題）→ 搬進 `evidence/raw/w2/<cell>/`。已有 manifest 的 stage 一律跳過，所以中斷後重跑同一指令就是續跑。BF16 的 KV 在 0.82 只剩約 1.7 GiB（≈ 11.7k tokens），網格到 40、`--max-num-seqs 40`。FP8 的 `--max-num-batched-tokens 8192` 對照 cell 只跑 open-loop seed 1。整夜約 22 小時 GPU；桌面可照常使用，但其他 GPU 工作會讓 stage 被標可疑而重跑。

## W2 結案（2026-09-11）

四精度全部完成，結果與協定偏離見 ADR 0009。實際耗時：主鏈（AWQ → GPTQ → BF16 → 交叉驗證）2026-09-10 04:35 → 19:00，約 14.5 小時；追加鏈（8192 對照組、FP8 TMMLU+ 全集、三個 cell 的膝點補點）19:01 → 01:00，約 6 小時。全程無可疑 stage、無隔離重跑，Windows committed 最高 24,187 MB。

量測期間新增、之後可沿用的腳本（全部在 `scripts/wsl/`，以腳本檔呼叫，不用 `bash -c`）：

- `run-logged.sh <script>`：把鏈的輸出寫到 ext4 上的日誌並更新 `runs-w2/w2-night-latest.log`；`watch-night.sh` 只挑里程碑與失敗行。Windows 端的 `| tr | grep | tee` 會區塊緩衝、整段遺失，不要再用。
- `refine-cell.sh <cell> <model> <max_num_seqs> <r_sat> "<multipliers>" <seeds...>`：只在既有 open-loop 目錄補指定倍率的 rate，沿用同一套 stage、可疑規則與 promote。
- `tmmlu-only.sh <cell> <model> <max_num_seqs>`：單獨起伺服器跑 TMMLU+ 全集。
- `promote-crosscheck.sh`：`vllm bench serve` 的結果只搬純量摘要與原檔 sha256；`generated_texts` 會重現訓練資料片段，不進 repo。
- `chain-status.sh`、`stage-summary.sh <batch dir>`、`stop-chain.sh`、`preflight.sh`：查進度、摘要、在 stage 邊界停止、起跑前檢查。

`batch.sh` 在 W2 期間修了兩個會整段丟 cell 的缺陷：就緒探測改用 command substitution（`curl | grep -q` 在 `pipefail` 下會因 SIGPIPE 141 誤判失敗），`quiet-gpu` 改為最多重試 5 次、每次間隔 30 s（上一個伺服器剛關閉時 utilization 會殘留數秒）。

**同一批次目錄只保留最後一個伺服器 session 的 `serve.log`／`quiet_gpu.json`。** 在已完成的 seed 目錄再起一個 session（補點、隔離重跑）之前，先確認舊的 log 已經 promote 並提交；補點用 `refine-cell.sh`，它會以 `refine-<第一個倍率>` 為 tag 另存，不覆蓋主量測的 `vllm.log`。隔離重跑尚未分檔（ADR 0009 缺陷 6）。

## 每次 batch 的順序（`scripts/wsl/batch.sh`；`harness/run.py` 的 Python 版尚未寫）

```bash
# 在 Windows 端呼叫（路徑用 /mnt/c、/mnt/d；MSYS_NO_PATHCONV=1 避免 Git Bash 改寫路徑）
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-bench -- env WARMUP=100 MAX_NUM_SEQS=256 RUN_ROOT=/home/<user>/vllm-slo-lab/runs-w2/closed-loop \
  bash /mnt/d/.../vllm-single-gpu-slo-lab/scripts/wsl/batch.sh fp8 Qwen/Qwen3-8B-FP8 1 "" \
  cl:1:90 cl:2:200 cl:4:400 cl:8:780 cl:16:1550 cl:32:2800 cl:64:4700 cl:96:5800 cl:128:6700 cl:192:8000 cl:256:9000
# open-loop 一點：ol:<rate_rps>:<duration_s>；驅動會先 quiet-gpu，再起 server，逐 stage 呼叫 `slo-lab run-stage`，最後關 server
bash scripts/wsl/io-sampler.sh <batch dir>          # 同時在背景跑，記 I/O／CPU 壓力
bash scripts/wsl/promote-w2.sh <batch dir> fp8/closed-loop/seed-1   # 搬進 evidence/raw/w2/（去 home 路徑、redact log、不搬 10+ MB 的 per-request JSON；records 與 server log 最後 gzip，ADR 0011）
uv run python scripts/analyze_batch.py evidence/raw/w2/fp8/closed-loop --out analysis/tables/w2-fp8-closed-loop
```

closed-loop 的 `num_requests` 依上一輪的 rps 取 ≥ 180 s（丟棄前 60 s 後仍有 ≥ 2 min 窗）；每個 stage 的前 60 s 一律丟棄（closed-loop 的起步同步效應會把 c = 128 的 TTFT p95 推到 1 s 以上，60 s 後只剩 0.39 s）。

## 手動順序（對照用）

1. 只在協調過的 GPU 空閒時段開始。
2. `uv run slo-lab quiet-gpu --out evidence/raw/<run_id>/quiet_gpu.json` — 有其他 compute process、既有記憶體占用 > 3,072 MiB、或 utilization（5 × 1 s 平均）> 10% 即退出 1，不得繼續（WSL2 上 process 準則無效，快照會標明）。
3. 在 vLLM 環境啟動 server（旗標見 `config/engine/common.yaml` + cell 檔；`HF_HUB_OFFLINE=1`、`VLLM_WSL2_ENABLE_PIN_MEMORY=1`、`VLLM_USE_FLASHINFER_SAMPLER=0`、`VLLM_CACHE_ROOT` 在 ext4）。
4. 若 policy 為 (ii)／(iii)：`slo-lab shim --upstream http://127.0.0.1:8013 --port 8021 --policy hard_cap --capacity <C>`；policy (i) 亦走 `--policy passthrough` 以保持 shim 開銷一致（W1 量到的中位數開銷在 ±2 ms 內）。
5. `uv run slo-lab power-sample evidence/raw/<run_id>/power.csv --phase idle --duration-s 60`，之後 `--append --phase warmup`、`--append --phase measure`。
6. Warm-up 100 sequential request；inference-perf 依 `config/traffic/*.yaml` 跑；scrape `/metrics`。
7. 收集 → `records.jsonl`（adapter 未寫）→ `scripts/redact.py redact` 去敏 log → `manifest.json`。
8. `make reproduce`；`make audit-secrets`。

## 未完成

- `harness/run.py` 編排、inference-perf adapter、metrics scraper、`vllm bench serve` 交叉驗證、nonce prompt 產生器。
