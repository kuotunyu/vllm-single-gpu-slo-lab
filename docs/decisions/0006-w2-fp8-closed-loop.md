# ADR 0006 — W2 第一步：FP8 closed-loop 掃描、r_sat 與 C、以及 4090 共用租戶污染

- 日期：2026-09-09
- 狀態：已採納；`analysis/preregistration.md` 據此部分凍結（warm-up 充分性除外）
- 原始紀錄：`evidence/raw/w2/fp8/closed-loop-exploratory/seed-1/`（探索性掃描）、`evidence/raw/w2/fp8/closed-loop/seed-1/`（正式掃描）；彙整表 `analysis/tables/w2-fp8-closed-loop*/`
- 引擎：vLLM 0.28.0，`Qwen/Qwen3-8B-FP8`，`--max-model-len 4096 --gpu-memory-utilization 0.90 --max-num-seqs 256 --max-num-batched-tokens 2048`，FLASH_ATTN，cudagraph PIECEWISE；prompt 108 / output 132 tokens、`ignore_eos`；loadgen inference-perf 0.6.1 concurrent 模式 4 workers；seed 1

## 結果（乾淨的資料點）

| 來源 | concurrency | window | achieved rps | output tok/s | TTFT p95 | TPOT p95 | attainment | mean W | tok/Wh |
|---|---|---|---|---|---|---|---|---|---|
| 探索性 | 1 | 99 s（無丟棄） | 0.406 | 53.6 | 0.051 s | 19.1 ms | 0.975 | 188 | — |
| 探索性 | 2 | 83 s | 0.959 | 126.6 | 0.051 | 16.2 | 1.000 | 238 | — |
| 探索性 | 4 | 81 s | 1.969 | 259.8 | 0.053 | 15.9 | 1.000 | 243 | — |
| 探索性 | 8 | 84 s | 3.801 | 501.7 | 0.084 | 15.8 | 0.978 | 249 | — |
| 探索性 | 16 | 84 s | 7.598 | 1,003 | 0.128 | 16.3 | 1.000 | 250 | — |
| 探索性 | 32 | 92 s | 13.85 | 1,828 | 0.226 | 17.6 | 1.000 | 261 | — |
| 探索性 | 64 | 86 s | 23.29 | 3,074 | 0.388 | 20.5 | 1.000 | 265 | — |
| 探索性 | 128 | 60 s | 33.33 | 4,400 | 1.076 | 28.8 | 0.921 | 263 | — |
| 正式 | 128 | 133 s（丟棄前 60 s） | 34.13 | 4,505 | 0.392 | 26.8 | 1.000 | 294 | 42,010 |
| 正式 | 192 | 135 s | 40.87 | 5,394 | 0.409 | 33.9 | 1.000 | 313 | 45,863 |
| 正式 | 256 | 142 s | **43.67** | 5,763 | 0.440 | 41.5 | 0.9998 | 327 | 46,383 |

- **r_sat（FP8）= 43.7 req/s**，取自 c = 256（= `--max-num-seqs`，引擎上限；KV 使用率峰值 67.5%，無 preemption）。192→256 仍增 6.8%，依「最後一格增幅 < 5% 才算平台」規則這是**下界**；再往上要提高 `--max-num-seqs`，KV 會進入 preemption 區，因此凍結於引擎上限而非再擴網格（分析器輸出 `r_sat_is_lower_bound: true`）。
- **C（closed-loop 仍守 SLO 的最大 concurrency）= 256**：丟棄前 60 s 後，所有乾淨點的 attainment ≥ 0.9998；c = 256 的 TPOT p95 41.5 ms 是最接近 50 ms 門檻的點。探索性掃描 c = 128 的 attainment 0.921（TTFT p95 1.08 s）是**起步同步效應**：closed-loop 開始時 128 個 request 同時到達，13.8k prompt tokens 以 2,048 的 chunk 分七步 prefill；丟棄 60 s 後同一點 TTFT p95 只剩 0.39 s。這是 closed-loop 加丟棄規則的理由，也是 C 從探索性的 64 改為 256 的原因。
- 單流基線：TPOT 15.5–19 ms（53 tok/s at c = 1）；TTFT p50 22–27 ms。
- 能耗：c = 256 時 46,383 output tokens / Wh ≈ 21.6 Wh 每百萬 output token（量測窗平均 327 W；`power_window`）。

## 污染事件：正式掃描 c = 1–96 全部作廢

正式掃描（02:19–04:00 CST）c = 4–96 的吞吐只有探索性掃描的 40–55%，TTFT p95 0.7–3.3 s、TPOT p95 39–114 ms，且在同一 stage 內來回震盪（c = 64 由 1,700 掉到 630 再回 1,450 tok/s）；c = 128 起（03:46 後）數字回到探索性水準以上。GPU 指紋：

| 狀態 | NVML util | SM clock | 功耗 | W／util 點 |
|---|---|---|---|---|
| 乾淨（探索性 c = 2–64；正式 c = 128–256） | 71–95% | 2,080–2,670 MHz | 238–327 W | 2.5–4.3，隨 concurrency 上升 |
| 污染（正式 c = 4–96） | 94–98% | 2,560–2,690 MHz | 169–184 W | 1.76–1.88 |

util 接近滿載、時脈全速、功耗卻只有一半，是**另一個 GPU context 分時共用**的典型形狀（WDDM 分時把「有 kernel 在跑」都算進 util，而 vLLM 自己的工作量減半）。伺服器端直方圖（`server_histograms`，本次新增）顯示 TTFT p95 落在 1–5 s 桶而 scheduler queue time p95 < 0.3 s，即時間耗在 engine step 而非排隊，排除 loadgen 假象。主機 I/O 壓力在量測期間多半 < 5%（`io-pressure.log`），排除 W1／smoke 那種磁碟停頓。WSL 內沒有 cron（`crontab -l` 空；規格所說的 00:00–08:00 SOP 特徵抽取尚未建立），也沒有其他專案目錄在該時段被寫入；Windows 端事後只看得到桌面程序，無法回溯到當時的租戶。**結論：4090 在該時段被本機另一個工作分時占用，來源無法從證據斷定**；`quiet-gpu` 只在 batch 開始時把關，擋不住中途出現的租戶。

- c = 1、2 的 W／util 點為 2.09／2.28，沒有被門檻 2.0 抓到，但同樣受污染：0.349 vs 0.406 rps、TPOT p95 26.8 vs 19.1 ms，300 個 warm-up request 平均 3.58 s（乾淨時 2.6 s）。**這兩點也作廢**（人工排除，分析器只自動排除 4–96）。
- warm-up 充分性檢查（101–200 vs 201–300 的 TTFT 中位數）：25.9 vs 30.4 ms，差 17.5%，但 50 筆一組的中位數在 22–32 ms 間來回，是租戶造成的抖動，**檢查無效，須在乾淨時段重做**；warm-up 暫維持 100（探索性掃描 100 筆後 TTFT 中位數 20–24 ms，各 stage 20 筆 re-warm 中位數 19–23 ms，沒有看到未暖機殘留）。

## 決策

1. **closed-loop 也套用「丟棄前 60 s」**（`stage_window`），量測窗到最後一筆完成為止；每點 `num_requests` 依上一輪的 rps 取 ≥ 180 s。探索性掃描保留為證據但標記「無丟棄、60–100 s」。
2. **網格延伸到 {96, 192, 256}**，256 = `--max-num-seqs`；r_sat 凍結為 43.7 rps（下界）；open-loop 掃描的 offered rate = {0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0} × 43.7 = {10.9, 21.8, 32.7, 43.7, 54.6, 65.5, 87.3} rps。
3. **租戶偵測進 harness 與分析器**：每 stage 的 `power_window.w_per_util_point`（門檻 2.0，本機校準值）與 re-warm 的單流 TPOT probe（`probe_tpot_median_s`，偏離同 batch 最佳值 > 15% 即可疑）；`analyze_batch.py` 把可疑 stage 排除在 r_sat／C 之外並列出 `suspect_concurrencies_excluded`。門檻是 4090 上的經驗值，換主機要重校。
4. **伺服器端直方圖**（TTFT／queue／TPOT／e2e）每 stage 前後快照差分寫進 manifest，作為 loadgen 無關的對照；主機 loadavg／CPU pressure 一併記錄。
5. **量測時段須與本機其他 GPU 工作協調**：W2 其餘量測（c = 1–96 重測、warm-up 充分性、open-loop × 3 seeds、TMMLU+）只在使用者宣告 GPU 空閒的時段執行；被標記可疑的 stage 一律重跑，不得以「平均掉」處理。
6. 探索性 c = 8 前 30 s 的 7 筆 1–2 s TTFT（當時主機 I/O 壓力 5–12%）與 smoke 同型停頓，仍無伺服器端直方圖可對照；丟棄規則已涵蓋，列為待觀察。

## 對規格的偏離

- 規格 §3.3 closed-loop 每點 3 min：正式掃描 c = 128–256 的量測窗為 133–142 s（`num_requests` 依探索性 rps 估算，實際更快），乾淨點仍各有 4.5k–6.2k 筆 request；下次依本 ADR 的 rps 重新取 `num_requests`。
- r_sat 以「引擎上限」而非「吞吐平台」定義，理由如上；open-loop 掃描到 2.0 × r_sat 仍會看到過載段，設計目的不受影響。
