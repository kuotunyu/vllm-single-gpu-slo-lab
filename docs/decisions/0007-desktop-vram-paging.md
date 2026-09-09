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

## 補記 2（open-loop × 3 seeds、TMMLU+）

（chain 完成後填入。）
