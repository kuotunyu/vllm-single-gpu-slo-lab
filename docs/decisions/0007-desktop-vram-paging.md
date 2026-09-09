# ADR 0007 — 「共用租戶」的真身：桌面程式把 VRAM 擠爆，`--gpu-memory-utilization` 從 0.90 降到 0.82

- 日期：2026-09-09 中午
- 狀態：已採納；修正 ADR 0006 的歸因，`analysis/preregistration.md` 的引擎旗標隨之更新
- 原始紀錄：`evidence/raw/w2/fp8/closed-loop-rerun/seed-1/`（0.90、白天、污染）、`evidence/raw/w2/fp8/closed-loop-v3/seed-1/`（0.82、白天、乾淨）；Windows 端 VRAM 取樣 `evidence/raw/w2/fp8/closed-loop-v3/win-vram.log`

## 發現

使用者宣告沒有其他 session 在用 GPU 之後重跑，c = 1 仍比夜間探索性掃描慢 17%（0.336 vs 0.406 rps；TPOT p95 24.8 vs 19.1 ms），伺服器端 10 秒吞吐在 22–54 tok/s 間震盪，NVML util 97%、時脈 2,683 MHz（比夜間還高）、無任何 throttle 原因。Windows 端 GPU 引擎計數器只有 `vmwp`（WSL 本身）65% 與 `System` copy engine 2%，沒有別的 3D／compute 使用者。**決定性的證據在記憶體**：

| 量測 | vLLM 預算 | NVML `mem_used`（WSL 內） | Windows `Total Committed` | 實體 | 結果 |
|---|---|---|---|---|---|
| 夜間探索性（桌面閒置） | 0.90 | 23,718–24,342 MiB | 未量 | 24,564 MiB | 乾淨 |
| 正式掃描 c = 1–96（凌晨、桌面有活動） | 0.90 | 24,158–24,419 | 未量 | 24,564 | 污染 |
| 白天重跑 c = 1（使用者在桌面） | 0.90 | 24,285–24,501 | **25,263 MB** | 24,564 | 污染 |
| 白天 v3 c = 1 | **0.82** | 22,320–22,440 | 23,183–23,236 | 24,564 | 乾淨（0.384 rps、probe TPOT 18.9 ms、W／util 2.27） |

0.90 × 24,564 = 22.1 GiB 給 vLLM，加上 CUDA context 只剩約 0.5 GiB 給整個 Windows 桌面；桌面程式（dwm 420–460 MB、Firefox 200–430 MB、Chrome、Explorer、ChatGPT、Claude、Terminal 共 1.1–1.4 GB dedicated）一活動，WDDM 的 total committed 就超過實體 VRAM，VidMm 開始在 PCIe 上分頁——vLLM 的 context 被反覆換出換入，GPU「忙」但每步變慢，`System` copy engine 就是分頁流量。夜間桌面閒置時各程式的 VRAM 被回收，所以同樣的 0.90 是乾淨的；這解釋了為什麼污染在凌晨 02:22–03:45 出現又在 03:46 消失（桌面活動的起訖），以及 ADR 0006 為何找不到任何 compute 租戶。

## 決策

1. **`--gpu-memory-utilization 0.82`**（`scripts/wsl/batch.sh` 的 `GPU_MEM_UTIL` 預設值），所有 cell 一體適用。KV cache 由 90,080 降為 76,112 tokens，c = 256 仍只用 80%；桌面保有 ≥ 1.3 GB 餘裕。這是「桌面 GPU 上量測」的必要條件，不是效能調校；模型卡與 claim ceiling 註明「單卡與 Windows 桌面共用，vLLM 只拿 82%」。
2. **manifest 加 Windows 端 VRAM 帳**：`host_before/after.windows_gpu_memory = {dedicated_mb, shared_mb, committed_mb}`，透過 WSL interop 呼叫 `powershell.exe` 讀 `GPU Adapter Memory` 計數器；`committed_mb` 超過實體即視為污染（分析器規則待補：目前靠 W／util 與 probe）。
3. ADR 0006 的「來源無法斷定」修正為本 ADR 的歸因；「量測只在使用者宣告 GPU 空閒的時段」改為「桌面可以照常使用，但 vLLM 預算 0.82 且 committed 不得超過實體」。
4. Warm-up 充分性（規格 §3.4）在 0.90 白天重跑中以 300 筆量得：101–200 vs 201–300 的 TTFT 中位數 23.6 vs 22.9 ms，差 3.0% ≤ 5% → **warm-up 凍結為 100 sequential**（該批其他數字因分頁作廢，但 warm-up 用的是單流 urllib client，TPOT probe 18.9–19.2 ms 與乾淨值一致，判定有效）。
5. 夜間 0.90 的乾淨點（探索性 c = 1–64、正式 c = 128–256，r_sat 43.7 rps）保留為證據，但 headline 改用 0.82 的 v3 掃描（本 ADR 補記）；兩者差異本身是「KV 預算不影響 r_sat」的檢驗。

## 補記 1：closed-loop v3（0.82，白天、桌面照常使用，2026-09-09 12:45–13:45）

`evidence/raw/w2/fp8/closed-loop-v3/seed-1/`、表 `analysis/tables/w2-fp8-closed-loop-v3/`。11 點全部乾淨（probe TPOT 18.9–19.4 ms；W／util 2.27 → 4.26 隨 concurrency 上升；Windows committed 23.2–23.5 GB < 24.5 GB）。

| c | window | rps | output tok/s | TTFT p95 | TPOT p95 | attainment | mean W | tok/Wh |
|---|---|---|---|---|---|---|---|---|
| 1 | 172 s | 0.384 | 50.6 | 0.026 s | 20.5 ms | 1.000 | 217 | 710 |
| 2 | 164 s | 0.893 | 117.8 | 0.043 | 17.0 | 1.000 | 238 | 1,676 |
| 4 | 159 s | 1.832 | 241.7 | 0.069 | 16.7 | 1.000 | 246 | 3,295 |
| 8 | 155 s | 3.629 | 478.9 | 0.088 | 16.4 | 1.000 | 253 | 6,045 |
| 16 | 161 s | 6.932 | 915 | 0.138 | 17.5 | 1.000 | 255 | 11,700 |
| 32 | 158 s | 12.70 | 1,676 | 0.214 | 18.5 | 1.000 | 261 | 19,980 |
| 64 | 152 s | 22.05 | 2,910 | 0.334 | 21.2 | 1.000 | 264 | 32,680 |
| 96 | 160 s | 25.99 | 3,430 | 0.533 | 26.8 | 0.977 | 297 | 33,490 |
| 128 | 140 s | 33.17 | 4,378 | 0.403 | 27.9 | 1.000 | 302 | 40,030 |
| 192 | 143 s | 38.64 | 5,099 | 0.420 | 35.4 | 0.9998 | 319 | 42,510 |
| 256 | 151 s | **41.35** | 5,457 | 0.461 | 43.8 | 0.9994 | 326 | 43,710 |

- **r_sat（FP8，0.82）= 41.4 req/s**（c = 256 = `--max-num-seqs`；192→256 仍 +7.0%，維持「下界」標記）；**C = 256**（最低 attainment 0.977 在 c = 96）。與夜間 0.90 的 43.7 差 5%，c = 128 兩者只差 0.5%（33.2 vs 33.3）：KV 預算不影響 r_sat，5% 的差在白天桌面本身的 GPU 分時（dwm／瀏覽器合成），是「桌面 GPU 白天量測」的真實條件，不再修正。
- 每百萬 output token 在 c = 256 約 22.9 Wh（43,710 tok/Wh）。
- Open-loop 網格沿用 preregistration 凍結的絕對 rate {10.9 … 87.3} rps（= 0.26–2.11 × 41.4）；不因 r_sat 微調而重排。

## 補記 2：FP8 open-loop × 3 seeds 與 TMMLU+（2026-09-09 13:46–18:49，0.82，桌面照常使用）

`evidence/raw/w2/fp8/open-loop/seed-{1,2,3}/`（每 seed 11 個 rate，每點 5 min、丟棄前 60 s）、表 `analysis/tables/w2-fp8-open-loop/`；33 個 stage 全部乾淨（probe 18.6–19.4 ms、W／util 2.8–4.2、Windows committed 22.6–23.7 GB < 24.5 GB）。

| offered rps | × r_sat | attainment（seed 1 / 2 / 3） | TTFT p95（s） | TPOT p95（ms） | mean W | tok/Wh |
|---|---|---|---|---|---|---|
| 10.92 | 0.25 | 1.000 / 1.000 / 1.000 | 0.06 | 18–19 | 263 | 17,400 |
| 21.84 | 0.50 | 1.000 / 1.000 / 1.000 | 0.09 | 24–25 | 293 | 29,300 |
| 24.02 | 0.55 | 1.000 / 1.000 / 1.000 | 0.09–0.10 | 26 | 307 | 30,500 |
| **26.20** | 0.60 | 1.000 / 1.000 / 1.000 | 0.11 | 29–31 | 316 | 32,100 |
| 28.39 | 0.65 | **0.910** / 1.000 / 1.000 | 0.95 / 0.11 / 0.13 | 51 / 31 / 38 | 322 | 33,900 |
| 30.57 | 0.70 | 0.950 / 1.000 / 0.997 | 0.16–0.17 | 48–50 | 321 | 35,300 |
| 32.75 | 0.75 | 0.103 / 0.209 / 0.608 | 1.0–2.4 | 58–60 | 330 | 36,900 |
| 43.67 | 1.00 | 0 / 0 / 0 | 81–89 | 58 | — | — |
| 54.59–65.5 | 1.25–1.5 | 0 | 161–248 | 53–58 | — | — |
| 87.34 | 2.00 | 0（timeout 23–25%） | 283 | 49–50 | 335–350 | 33,100（以 offer 時間計，見下） |

- **r_SLO（FP8，TTFT p95 ≤ 1 s ∧ TPOT p95 ≤ 50 ms，3 seeds，凍結規則）= 26.2 req/s**，= 0.63 × r_sat(41.4)。上一格 28.39 只有 seed 1 掉到 0.910（TTFT p95 0.95 s、TPOT p95 50.9 ms，其餘兩個 seed 1.000），30.57 三個 seed 都 ≥ 0.95；規則「第一個 < 95% 即停」因此取 26.2，保守但依預註冊。
- **膝點是 TPOT，不是排隊**：26–31 rps 之間 TPOT p95 從 30 ms 爬到 50 ms 而 TTFT p95 仍 < 0.2 s；32.75 rps 起 attainment 崩到 0.1–0.6，43.7 rps 起佇列無界（TTFT p50 50–56 s），87 rps 有 23–25% 的 request 撞到 300 s client timeout。伺服器端直方圖（`server_histograms`）與 client 同桶，排除 loadgen 假象。
- **SLO 敏感度**（同一份 raw 重算，`analysis/tables/w2-fp8-open-loop/tables.md`）：TPOT 30 ms → 24.0 rps；TPOT 50 或 100 ms → 26.2；只有 TTFT 2 s ∧ TPOT 100 ms 才到 30.6。r_SLO 對 TTFT 門檻（0.5–2 s）不敏感，對 TPOT 門檻敏感——這張卡的容量是 decode 步長決定的。
- **能耗**：r_SLO 點 316 W、32,100 output tok/Wh ≈ 31 Wh／百萬 output token；比 closed-loop c = 256 的 43,700 tok/Wh 低 27%（守 SLO 的代價）。
- 注意：open-loop 表裡的 `achieved_rps` 與 `output_tok_per_s` 以「offer 時間落在量測窗」計，過載點（≥ 32.75 rps）的完成時間拖到窗外，這兩欄在過載段不代表服務速率。**2026-09-10 補上 `served_rps`**（以完成時間計，`slo_lab.batch_analysis.served_rps`，回溯適用於本批證據）：過載段的實際完成率是 33.9／34.3／35.7／39.4 rps（offered 43.7／54.6／65.5／87.3），全部**低於 closed-loop 的 r_sat 41.4**——Poisson 突發比等速閉環少了約 15% 的吞吐，且完成率隨佇列變深而微升（排程器批次變大）。這是 admission control（W3）要處理的現象本身。
- **TMMLU+（FP8，三個 200 題切片，greedy、`/no_think`）**：122／119／125 正確 → 0.610／0.595／0.625（Wilson 95%：0.54–0.68、0.53–0.66、0.56–0.69），合計 366/600 = 0.610。`evidence/raw/w2/fp8/tmmluplus/`。這只是 FP8 的絕對值；規格要的是四精度的配對差，等 AWQ／GPTQ／BF16。**租戶註記**：18:44:50 起 Windows 端出現一個 4.0–4.5 GB 的 `python` GPU 程序（本機另一個專案），與 18:47:42–18:48:43 的三切片評分重疊，committed 衝到 27.8–28.2 GB > 24.5 GB（`win-vram-2026-09-09.log`）。greedy 解碼的答案不受速度影響，分數有效；但 8.9–12.9 題/秒的速度不代表 FP8 正常值（W1 乾淨時 29.5）。open-loop 三個 seed 在 18:42:38 已結束，未受影響。另一個 committed 超額點 17:09:03（27.3 GB）發生在 seed 3 伺服器啟動與 warm-up 交界，量測窗外。
