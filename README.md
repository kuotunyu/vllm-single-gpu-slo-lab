# vllm-single-gpu-slo-lab

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

> **狀態：W2 進行中（2026-09-09）。** W1 驗證清單結案（ADR 0002–0005）；W2 第一步 FP8 closed-loop 掃描完成（ADR 0006）：**r_sat = 43.7 req/s（下界，c = 256 = `--max-num-seqs`）、C = 256**，單 seed、單次；同批 c = 1–96 因 4090 被本機另一個 GPU 工作分時占用而作廢，正在等乾淨時段重測。**尚無 open-loop（r_SLO）、成本、TMMLU+ 數字**；本文其餘數值仍是規格門檻或設定值。

## 一句話

在一張 RTX 4090 上，給定 SLO（TTFT p95 ≤ 1 s、TPOT p95 ≤ 50 ms），量出 Qwen3-8B 四種精度 × 三種 admission 策略 × 兩種 decoding 加速在 open-loop Poisson 流量下的容量（rps at SLO）與每百萬 output token 成本（4090 攤提 + 實測電費），n ≥ 3、附 bootstrap CI，並自跑 TMMLU+ 把品質與成本放進同一張表。

## 30 秒結論（目標讀者：台灣 LLM／AI infra 用人主管）

*（下面是本專案**要證明**的事；W0 尚未完成任何一項。）*

這個人把單張 GPU 上的 vLLM 當成一個必須守 SLO 的服務來量，而不是跑一次 throughput 截圖。同一條 Poisson trace、同一組 seed，報出每個精度在 SLO 下的容量與 $/M token；證明 admission control（原生排隊 vs 硬上限 429 vs 有界佇列）在同一張卡上對 SLO attainment、goodput、拒絕率的三維取捨；成本用實測功耗與實測 utilisation 算，並附 utilisation-naive 值的 1/U 警語；量化品質用自跑的 TMMLU+ 而非過期 leaderboard；全部從 raw JSON 一鍵重建，且明寫哪些結論**不能**外推。

## 與 `local-inference-bench-gateway`（LIBG）的分工

**LIBG 那一側**：LIBG 以共同 workload 比較 llama.cpp／Ollama／LM Studio 三種桌面級 engine，並提供 OpenAI-compatible gateway（alias routing、ordered failover、容量滿時回 HTTP 429 且不建無上限 in-memory queue、SQLite telemetry）。它回答「哪個本機 engine、gateway 該怎麼擋」；不做 vLLM、不做 open-loop 流量、不做量化對照、不畫 SLO 曲線。

**本專案這一側**：只用一個 engine（vLLM），問的是「在 SLO 約束下這張卡能接多少、每個 token 多少錢、精度與 admission 怎麼換」。不重做 engine 比較；429 語意向 LIBG 借用但獨立實作、不 import LIBG。兩邊 README 各放一段互指分工。

## 實驗設計摘要（規格 §3；定值於 `analysis/preregistration.md` 凍結）

| 軸 | 水準 | 主要問題 |
|---|---|---|
| 精度 | BF16、FP8、AWQ、GPTQ-Int4（repo id 與授權 W1 核對） | 每精度的 rps at SLO、$/M token、TMMLU+ |
| Admission | (i) vLLM 原生排隊 (ii) 硬上限 C + HTTP 429 (iii) 有界佇列 Q + 逾時 T | 突發流量下 SLO attainment、goodput、拒絕率的取捨 |
| Decoding 加速 | none、n-gram（8B 與 4B）、EAGLE-3（僅 Qwen3-4B） | 低／高 rate 下 TPOT 與容量的變化 |

- **SLO attainment**：量測窗內同時達 TTFT 與 TPOT 門檻的 request 比例；5xx、timeout、**429 拒絕**皆計未達，分母為 offered requests（另附 admitted-only 供對照）。
- **r_SLO（rps at SLO）**：open-loop 掃描中，所有 seed attainment ≥ 95% 的最高 offered rate（保守值），附 attainment-vs-rate 曲線與 CI；同一份 raw 零成本重算 TTFT ∈ {0.5, 1, 2} s × TPOT ∈ {30, 50, 100} ms 的敏感度網格。
- **流量**：inference-perf 裸機模式、Poisson、108 in／132 out、nonce 前綴、thinking 關閉；admission 用 25 分鐘單峰 trace（`config/traffic/burst25.yaml`，提案）。
- **重複**：每 cell n ≥ 3 個 seed（paired）；分位數 percentile bootstrap、比例 Wilson interval；負結果照登。
- **部分因子**（ADR 0001）：精度軸全做；admission 軸在預設精度與 BF16；decoding 軸在預設精度 8B 與 Qwen3-4B。

## Claim ceilings（規格 §2.2 原文；發佈前 claims audit 逐條核對）

1. 不宣稱多 replica、擴縮、生產可靠度（re-plan §4）。
2. 不宣稱跨 GPU class 推論——只講實際量過的 class：4090，加上 A1 完成後的 L4（或 L40S）；H100／Blackwell 一律不外推（memo §5）。
3. 不宣稱絕對「比 API 便宜」——只報「在我實際驅動的 utilisation 下」的 $/M token，並列 1/U 警語（memo §3(d)：H100 在 1–10 rps 間 $0.21–$15.25）。
4. 不宣稱 EAGLE-3 在 Qwen3-8B 上的效果——memo §2 只找到 AngelSlim 的 4B／14B／32B head；EAGLE-3 數字只屬於 Qwen3-4B，8B 只有 n-gram。
5. 不宣稱 TMMLU+ 分數可與他人 leaderboard 比較——ikala leaderboard README 已 12 個月未更新（memo §4）；只報自跑數字與四個精度間的配對差。
6. 不宣稱電費為整機功耗——`nvidia-smi` 只量 GPU 板卡功耗，主機其餘功耗未量，電費項為下限（提案）。
7. 不宣稱 WSL2 數字等於裸機 Linux——所有 4090 數字都在 `VLLM_WSL2_ENABLE_PIN_MEMORY=1`（vLLM 自述在 WSL2 有小幅效能退化）與 torch 原生 sampler（`VLLM_USE_FLASHINFER_SAMPLER=0`，WSL2 無 nvcc/ninja 可 JIT）下量測；WSL2 額外開銷未分離量測。A1 的 RunPod L4 是唯一的非 WSL2 對照（ADR 0002）。
8. 不宣稱 thinking 模式下的延遲——所有延遲量測關閉 Qwen3 thinking（memo §4 的 thinking toggle；提案）。

## 環境

- 量測全部在 **WSL2 Ubuntu 上的獨立 vLLM 環境**：vLLM 0.28.0、torch 2.13.0+cu130；2026-09-08／09 的 W1 驗證在 RTX 4090 上載入全部五個 cell 並完成 completion（ADR 0004），條件是 `VLLM_WSL2_ENABLE_PIN_MEMORY=1` 與 `VLLM_USE_FLASHINFER_SAMPLER=0`（ADR 0002）。**vLLM 與 torch 不在本 repo 的 `pyproject.toml` 依賴中**：本 package 只做 CPU-side 的計算、shim 與量測工具，`make reproduce` 在無 GPU、無網路的 CI 上跑。
- 此 driver 下 `nvidia-smi --query-gpu` 不支援，功耗改由 **NVML**（`pynvml`，optional extra `gpu`）以 1 s 間隔取樣；`nvidia-smi` 完整輸出仍以 best-effort 方式附進 quiet-GPU 快照。
- 4090 與另一個 00:00–08:00 的 cron 工作共用；本專案只在 08:00 之後量測，且 `slo-lab quiet-gpu` 發現任何其他 compute process 或既有記憶體占用超過門檻即拒跑，快照隨 run 提交。
- 開發：Python 3.12 + uv；Windows 宿主只跑 CPU 測試。

## 目前有什麼／還沒有什麼

### 有（W0）

| 模組 | 內容 | 測試 |
|---|---|---|
| `slo_lab/slo.py` | canonical per-request 記錄、joint SLO 判定、offered-denominator attainment、goodput、拒絕率、r_SLO、SLO 敏感度網格 | 手算小樣本 |
| `slo_lab/stats.py` | percentile bootstrap CI、paired-difference bootstrap、Wilson interval | 手算值 |
| `slo_lab/cost.py` | 規格 §6 公式、`config/cost.yaml` 讀取、power.csv 梯形積分與 idle 基線、實測值 + utilisation-naive 值 + 1/U 警語、Wh/M token | 手算值 |
| `slo_lab/admission/` | 三策略（passthrough／hard cap + 429／bounded FIFO queue + timeout）與 aiohttp reverse-proxy shim | 純 asyncio 語意 + loopback fake upstream |
| `slo_lab/power/sampler.py` | 1 s NVML 取樣寫 CSV；無 NVML 時明確報錯 | 注入 fake reader／clock |
| `slo_lab/quiet_gpu.py` | 拒跑判定 + NVML／nvidia-smi 快照 JSON | 注入 fake process 清單 |
| `slo_lab/redact.py` + `scripts/redact.py` | 去敏與 `make audit-secrets` 掃描（IP、私鑰、SSH 公鑰、RunPod key／host、HF token、email） | 動態組字串 + 掃描本 repo |
| `config/` | cost（placeholder，標 owner input）、engine 五 cell + common、admission 三策略、traffic 四份（含 `burst25` 與保留的 `cloud_2p5x`）、specdec 三份 | YAML 解析與內容 |
| `evidence/metrics-names.txt` | 96 個 vLLM metric 名，2026-09-09 由 live `/metrics` 凍結 | — |
| `slo_lab/tmmluplus.py` + `scripts/tmmluplus_eval.py` | TMMLU+ 分層不重疊切片（3 × 200，seed 20260908，SHA-256 凍結於 `eval/tmmluplus/`）與離線評分（greedy、`/no_think`、Wilson CI） | 合成 CSV：不重疊、比例、決定性、雜湊 |
| `evidence/raw/w1/` | W1 驗證證據：五 cell 載入矩陣、`vllm bench serve` 與 inference-perf smoke、shim 開銷四回合、TMMLU+ 20 題 dry run（ADR 0004、0005） | — |
| `slo_lab/harness/` | `run-stage`：warm-up（TTFT／TPOT probe）→ NVML 功耗 1 s + `/metrics` 5 s 背景取樣 → inference-perf → `records.jsonl` → 量測窗（open／closed-loop 皆丟棄前 60 s）→ manifest（伺服器端 TTFT／queue／TPOT／e2e 直方圖差分、功耗窗與 tok/Wh、主機負載、I/O 壓力、raw sha256） | adapter、直方圖、窗、功耗窗 |
| `scripts/analyze_batch.py` + `scripts/wsl/` | 批次彙整（r_sat 含平台旗標、C、r_SLO）與租戶污染標記（W／util 指紋、probe 漂移）；WSL2 批次驅動、I/O 取樣、證據搬移腳本 | 以真實 manifest 跑過 |
| `evidence/raw/w2/fp8/closed-loop*/` | **W2 第一步**：FP8 closed-loop 探索性（c = 1–128，無丟棄）與正式（c = 1–256，3 min／點）掃描，seed 1；乾淨點給 r_sat = 43.7 rps（下界）、C = 256；正式掃描 c = 1–96 因共用租戶作廢（ADR 0006） | — |
| CI | ruff check、ruff format --check、pytest、audit-secrets、`make reproduce`（空 evidence 通過） | — |

### 還沒有

- **open-loop 量測**：r_SLO、attainment-vs-rate 曲線、n ≥ 3、成本表；ledger 只有表頭；圖、model card、claims audit 皆空。
- FP8 closed-loop c = 1–96 的乾淨重測與 warm-up 充分性檢查（第一次被共用租戶污染，ADR 0006）；AWQ／GPTQ／BF16 的 closed-loop。
- TMMLU+ 三切片與全量的正式評分（W2；A3 Colab 已取消，ADR 0005）。
- `analysis/preregistration.md` 除 warm-up 充分性外已凍結（2026-09-09）；`harness/run.py` 的 Python 編排仍由 `scripts/wsl/batch.sh` 代行。
- `config/cost.yaml` 的 owner 真實數值與來源（目前為標記 placeholder；`slo-lab cost` 會印警語）。
- FP8 block kernel 的 4090 tuned config 決定（W2）；BF16 cell 的 `max-num-seqs`（ADR 0004 提案 16）。
- A1（RunPod L4）尚未開始；任何付費動作前逐筆先問。

## Repository 佈局

```
vllm-single-gpu-slo-lab/
├── README.md · LICENSE (Apache-2.0) · CITATION.cff · Makefile · pyproject.toml (uv)
├── src/slo_lab/            # slo.py · stats.py · cost.py · quiet_gpu.py · nvml.py · redact.py · cli.py
│   ├── admission/          # policies.py（純 asyncio）· shim.py（aiohttp）
│   └── power/sampler.py
├── tests/
├── config/
│   ├── cost.yaml · cost.yaml.example
│   ├── engine/{common,bf16,fp8,awq,gptq_int4,qwen3_4b}.yaml
│   ├── admission/{native,cap429,bounded}.yaml
│   ├── traffic/{closed_loop,open_loop_sweep,burst25,cloud_2p5x}.yaml
│   └── specdec/{none,ngram,eagle3_4b}.yaml
├── analysis/               # preregistration.md · claims_audit.md · ledger/{runs,cost,spend}.csv
├── evidence/               # README.md（contract）· metrics-names.txt · raw/ · runpod/ · quant/ · tables/ · plots/
├── eval/tmmluplus/         # W1／W2
├── docs/                   # decisions/0001 · runbook-wsl2 · runbook-runpod · model-card · licences
├── scripts/                # redact.py · audit_secrets.sh
└── .github/workflows/ci.yml
```

與規格 §9 的差異：`harness/` 與 `analysis/*.py` 合併為 `src/slo_lab/`（對照表見 `docs/decisions/0001`）；加了 `config/engine/common.yaml`。

## 快速開始

```bash
uv sync --all-extras          # dev + gpu extra（gpu extra 只在 WSL2 量測主機需要）
make test                     # pytest
make lint                     # ruff check + ruff format --check
make audit-secrets            # 掃 IP／金鑰／token；有發現即失敗
make reproduce-lite           # CPU-only：驗證 config、重算現有 evidence（W0：0 run）
make reproduce                # reproduce-lite + evidence/ analysis/ 零 diff
```

CLI：

```bash
uv run slo-lab attainment evidence/raw/<run_id>/records.jsonl --window-start-s 60
uv run slo-lab capacity <sweep.jsonl>                   # r_SLO + SLO 敏感度網格
uv run slo-lab cost --output-tok-per-s <x> --power-csv evidence/raw/<run_id>/power.csv --peak-output-tok-per-s <y>
uv run slo-lab quiet-gpu --out evidence/raw/<run_id>/quiet_gpu.json     # 需 NVML；忙碌即 exit 1
uv run slo-lab power-sample power.csv --phase idle --duration-s 60      # 需 NVML
uv run slo-lab shim --upstream http://localhost:8000 --policy bounded_queue --capacity <C>
uv run slo-lab reproduce-lite
```

## 里程碑

W0 骨架（本 commit）→ W1 驗證清單 10 項與基線 → W2 四精度掃描與 TMMLU+ 切片 → W3 admission trace → W4 spec-decode、成本表、A1／A3 → W5 補 n、敏感度、claims audit → W6 誠實寫作與發佈前檢查。細節見設計規格與 `docs/decisions/`。

## 授權

程式碼與文件 Apache-2.0（kuotunyu, 2026）。Qwen3 權重 Apache-2.0；GPTQ-Int4、EAGLE-3 head、TMMLU+ 授權 W1 核對，見 `docs/licences.md`。
