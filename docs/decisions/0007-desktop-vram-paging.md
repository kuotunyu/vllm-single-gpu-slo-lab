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

## 補記（v3 結果）

（chain 完成後填入：closed-loop v3 表、r_sat、C、open-loop × 3 seeds 的 r_SLO、TMMLU+ 三切片。）
