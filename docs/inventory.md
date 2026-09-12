# 模組與證據清單（自 README 移出，2026-09-13）

每個模組做什麼、對應的證據在哪、測試怎麼驗；結案時的狀態。表、圖與 run ledger 都由 `make reproduce` 從 `evidence/` 重建（`evidence/README.md`）。

| 模組 | 內容 | 測試 |
|---|---|---|
| `slo_lab/slo.py` | canonical per-request 記錄、joint SLO 判定、offered-denominator attainment、goodput、拒絕率、r_SLO、SLO 敏感度網格 | 手算小樣本 |
| `slo_lab/stats.py` | percentile bootstrap CI、paired-difference bootstrap、Wilson interval | 手算值 |
| `slo_lab/cost.py` | 規格 §6 公式、`config/cost.yaml` 讀取、power.csv 梯形積分與 idle 基線、實測值 + utilisation-naive 值 + 1/U 警語、Wh/M token；4090 不計 $，保留給租用 GPU（ADR 0010） | 手算值 |
| `slo_lab/admission/` | 三策略（passthrough／hard cap + 429／bounded FIFO queue + timeout）與 aiohttp reverse-proxy shim | 純 asyncio 語意 + loopback fake upstream |
| `slo_lab/power/sampler.py` | 1 s NVML 取樣寫 CSV；無 NVML 時明確報錯 | 注入 fake reader／clock |
| `slo_lab/quiet_gpu.py` | 拒跑判定 + NVML／nvidia-smi 快照 JSON | 注入 fake process 清單 |
| `slo_lab/redact.py` + `scripts/redact.py` | 去敏與 `make audit-secrets` 掃描（IP、私鑰、SSH 公鑰、RunPod key／host、HF token、email），含 gzip 檔與大檔（ADR 0011） | 動態組字串 + 掃描本 repo |
| `scripts/compress_evidence.py` | 逐筆紀錄與 server log 以決定性 gzip 提交，讀取端以 `open_evidence_text` 透明解壓（ADR 0011） | 無損、決定性、壓縮前後讀出相同 |
| `config/` | cost（placeholder；4090 不使用，ADR 0010）、engine 五 cell + common、admission 三策略、traffic 四份（含 `burst25` 與保留的 `cloud_2p5x`）、specdec 三份 | YAML 解析與內容 |
| `evidence/metrics-names.txt` | 104 個 vLLM metric 名：96 個於 2026-09-09 由 live `/metrics` 凍結，8 個 spec-decode counter 於 2026-09-12 的 W4 smoke 補記 | — |
| `evidence/raw/w4/` + `slo_lab/specdec_analysis.py` | W4 五個 cell 的 closed-loop 與 open-loop（3 seeds）證據、smoke、Windows 端顯存取樣；配對分析（同 family 對 none cell）與接受率；表 `analysis/tables/w4-*-specdec/`、圖 `evidence/plots/w4-*.svg` | 配對、缺基準、可疑段、決定性 |
| `slo_lab/tmmluplus.py` + `scripts/tmmluplus_eval.py` | TMMLU+ 分層不重疊切片（3 × 200，seed 20260908，SHA-256 凍結於 `eval/tmmluplus/`）與離線評分（greedy、`/no_think`、Wilson CI） | 合成 CSV：不重疊、比例、決定性、雜湊 |
| `evidence/raw/w1/` | W1 驗證證據：五 cell 載入矩陣、`vllm bench serve` 與 inference-perf smoke、shim 開銷四回合、TMMLU+ 20 題 dry run（ADR 0004、0005） | — |
| `slo_lab/harness/` | `run-stage`：warm-up（TTFT／TPOT probe）→ NVML 功耗 1 s + `/metrics` 5 s 背景取樣 → inference-perf → `records.jsonl` → 量測窗（open／closed-loop 皆丟棄前 60 s）→ manifest（伺服器端 TTFT／queue／TPOT／e2e 直方圖差分、功耗窗與 tok/Wh、主機負載、I/O 壓力、raw sha256） | adapter、直方圖、窗、功耗窗 |
| `scripts/analyze_batch.py` + `scripts/wsl/` | 批次彙整（r_sat 含平台旗標、C、r_SLO）與租戶污染標記（W／util 指紋、probe 漂移）；WSL2 批次驅動、I/O 取樣、證據搬移腳本 | 以真實 manifest 跑過 |
| `evidence/raw/w2/fp8/closed-loop*/` | FP8 closed-loop：夜間 0.90 探索性（c = 1–128）與正式（c = 1–256）掃描（ADR 0006；正式 c = 1–96 被桌面 VRAM 分頁污染）、白天 0.90 分頁證據、**0.82 的 v3（c = 1–256 全乾淨：r_sat 41.4 rps 下界、C = 256）**（ADR 0007） | 表由 `make reproduce` 重建 |
| `evidence/raw/w2/fp8/open-loop/seed-{1,2,3}/` | **FP8 open-loop 11 rates × 3 seeds**（0.25–2.0 × r_sat 加 0.55–0.70 細化，ADR 0008）：r_SLO 26.2 rps、膝點 26–31 rps、SLO 敏感度網格、過載段的佇列與 timeout | 同上 |
| `evidence/raw/w2/fp8/tmmluplus/` | FP8 三切片（366/600）與全集（11,629／19,680 = 0.5909）的分數 | — |
| `evidence/raw/w2/{awq,gptq,bf16}/` | 三精度 closed-loop（AWQ／GPTQ c = 1–256、BF16 c = 1–40）、open-loop 14 rates × 3 seeds（含 0.80–0.90 × r_sat 膝點補點）、TMMLU+ 切片與全集（ADR 0009） | 表由 `make reproduce` 重建 |
| `evidence/raw/w2/fp8-mbt8192/` | `--max-num-batched-tokens` 8192 對照組（seed 1）：C 由 256 降到 128、r_SLO 25.64，2048 維持（ADR 0009） | 同上 |
| `evidence/raw/w2/fp8/crosscheck/` | `vllm bench serve` 交叉驗證：吞吐與 inference-perf 差 5–7%；只存純量摘要與原檔 sha256，不存生成文字 | — |
| `slo_lab/quality.py` | 配對品質：exact McNemar（對數空間）與 paired bootstrap；`reproduce-lite` 重建 `analysis/tables/w2-quality-paired/` | 手算值與常態近似 |
| `evidence/raw/w2/fp8/win-vram-2026-09-09.log`、`evidence/raw/w2/win-vram-2026-09-10.log` | Windows 端 VRAM（dedicated／shared／committed、桌面程序）每 30 s 取樣：FP8 白天量測期間，以及 W2 四精度全程（committed 最高 24,187 MB，未超過實體） | — |
| CI | ruff check、ruff format --check、pytest、audit-secrets、`make reproduce`（從證據重建 `analysis/tables/index.json` 列的每張表、配對品質表、`evidence/plots/` 的圖與 `analysis/ledger/runs.csv`，diff 必須為空） | — |

## 已知未做（都是決定，不是漏做）

- `analysis/ledger/cost.csv` 與 `spend.csv` 只有表頭：4090 不計 $（ADR 0010），也沒有用過任何付費算力（A1 取消，ADR 0014）。`runs.csv` 由 `reproduce-lite` 從證據重建，每段一列，狀態欄取自分析器的可疑旗標（`slo_lab.ledger`）。
- `harness/run.py` 的 Python 編排由 `scripts/wsl/*.sh` 代行。
- FP8 block kernel 的 4090 tuned config 未產生，所有 FP8 數字都用 vLLM 預設 kernel config（server log 有警告，ADR 0009）。
- A1（RunPod L4 雲端對照）已取消（ADR 0014）。
