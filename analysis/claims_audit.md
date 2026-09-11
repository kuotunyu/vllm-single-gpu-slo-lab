# Claims audit（W5 填寫；發佈前逐條核對）

README 裡每個數字對應 evidence 路徑、n、CI，以及規格 §2.2 的 claim ceiling 編號。所有表都由 `make reproduce` 從 evidence 重建並 diff。共同條件（每列適用）：`--gpu-memory-utilization 0.82`、WSL2、與 Windows 桌面共用的單張 RTX 4090、Qwen3-8B、108→132 tokens 合成負載、thinking 關閉（ceiling 2、7、8）。

| # | README 句子 | Evidence 路徑 | n | CI | Claim ceiling | 核對 |
|---|---|---|---|---|---|---|
| 1 | r_SLO（TTFT p95 ≤ 1 s ∧ TPOT p95 ≤ 50 ms）：BF16 10.26、FP8 26.2、AWQ 22.61、GPTQ-Int4 22.70 req/s | `evidence/raw/w2/<cell>/open-loop/seed-{1,2,3}/ol-rate-*/records.jsonl.gz`（ADR 0011）；`analysis/tables/w2-<cell>-open-loop/open_loop.json` `per_cell.<cell>.r_slo` | FP8 3 seeds × 11 rates；其他 3 seeds × 14 rates（含 0.80–0.90 × r_sat 補點）；r_SLO 點每 seed 3,078–7,860 筆 | 每點 attainment Wilson 95%（表內）；r_SLO 本身無 CI（凍結規則取所有 seed 的 min），以「下一格」夾擊：+7–11% | 單卡、單一 prompt 形狀；不外推 | 2026-09-11 `reproduce-lite` 重建一致 |
| 2 | 膝點夾到 7–11%（下一格 BF16 11.40、FP8 28.39、AWQ 24.12、GPTQ 24.21 req/s 有 seed 失敗） | 同 #1，`per_cell.<cell>.min_attainment_by_rate` | 同 #1 | 同 #1 | 同 #1 | 同上 |
| 3 | r_sat：BF16 11.40（`--max-num-seqs` 40 封頂）、FP8 41.35（下界）、AWQ 30.15、GPTQ 30.26（平台） | `evidence/raw/w2/{bf16,awq,gptq}/closed-loop/`、`evidence/raw/w2/fp8/closed-loop-v3/`；`analysis/tables/w2-*-closed-loop*/closed_loop.json` `r_sat_rps`、`r_sat_is_lower_bound` | 1 seed；AWQ／GPTQ／FP8 11 點（c = 1–256），BF16 8 點（c = 1–40） | — | 下界不得寫成飽和值 | 同上 |
| 4 | 單流 TPOT：BF16 19.3、FP8 18.9、AWQ 7.8、GPTQ 7.6 ms；4-bit 單流快約 2.5 倍 | closed-loop `cl-conc-1` manifest 的 `probe_tpot_median_s`（warm-up 100 筆 sequential） | 每 cell 100 筆 | — | c = 1、這組 prompt 長度 | 同上 |
| 5 | 4-bit 飽和吞吐比 FP8 低 27% | 由 #3：30.2 / 41.35 | — | — | FP8 為下界，差距只可能更大 | 同上 |
| 6 | 能耗 @ r_SLO：BF16 74.6、FP8 31.2、AWQ 36.2、GPTQ 36.3 Wh／百萬 output token | r_SLO 點三個 seed 的 manifest `power_window.tok_per_wh`（表內 `tok_per_wh`），取 10⁶ ／ 三 seed 平均 | 每 stage 240 s、1 s NVML 取樣 × 3 seeds | — | 只算 GPU 板卡功耗（ceiling 6），不含主機 | 同上 |
| 7 | FP8 對 BF16：SLO 容量 2.55 倍、每 token 能耗 42% | 由 #1、#6：26.2 / 10.26；31.2 / 74.6 | — | — | 同 #1、#6 | 同上 |
| 8 | TMMLU+ 全集：BF16 0.5911、FP8 0.5909、AWQ 0.5797、GPTQ-Int4 0.5703 | `evidence/raw/w2/<cell>/tmmluplus/full.json`（11,632／11,629／11,409／11,223 題答對） | 19,680 題（67 科、snapshot 與 sha256 凍結於 `eval/tmmluplus/`） | Wilson 95% 於 JSON（各約 ±0.007） | 自跑數字，不與 leaderboard 比較（ceiling 5） | 同上；0 unparsed、0 errors |
| 9 | 對 BF16 配對差：FP8 −0.02 pts（p = 0.945）、AWQ −1.13 pts（p = 7.0 × 10⁻⁶）、GPTQ −2.08 pts（p = 1.4 × 10⁻¹⁶） | `analysis/tables/w2-quality-paired/paired.json`（`slo_lab.quality`，由 `reproduce-lite` 重建） | 19,680 對 | paired bootstrap B = 1000 的 95% 區間：[−0.29, +0.27]、[−1.64, −0.63]、[−2.55, −1.56]；exact McNemar | GPTQ 只代表 `JunHowie/Qwen3-8B-GPTQ-Int4` | 同上 |
| 10 | TMMLU+（FP8）三切片 366 / 600 = 0.610 | `evidence/raw/w2/fp8/tmmluplus/slice-{1,2,3}.json` | 600 題（3 × 200，分層、不重疊） | 各切片 Wilson 95% 於 JSON | 已被 #8 全集取代，只作歷史對照 | 同上 |
| 11 | `--max-num-batched-tokens` 8192 對照組：C 由 256 降到 128、r_SLO 25.64 | `evidence/raw/w2/fp8-mbt8192/`；`analysis/tables/w2-fp8-mbt8192-{closed,open}-loop/` | 1 seed（依 W2 執行計畫只跑 seed 1） | 每點 Wilson 95% | 單 seed，只支持「8192 沒有更好」，不支持精確差值 | 同上 |
| 12 | `vllm bench serve` 吞吐與 inference-perf 差 5–7% | `evidence/raw/w2/fp8/crosscheck/{closed-c256,closed-c64,open-r21.83}.json`（純量摘要，附原檔 sha256） | 各一次 | — | 只驗 FP8 | 同上 |
| 13 | Windows committed VRAM 全程最高 24,187 MB，未超過實體 | `evidence/raw/w2/win-vram-2026-09-10.log` | 2,315 筆（每 32 s） | — | 只證明 WDDM 分頁未觸發，不證明桌面零干擾 | 以 awk 取 max 核對 |
| 14 | W3 整段 attainment：FP8 原生排隊 0.19、hard cap 0.57、有界佇列 0.59；BF16 0.47、0.87、0.86 | `evidence/raw/w3/{fp8,bf16}/trace/seed-{1,2,3}/trace-*/records.jsonl.gz`；`analysis/tables/w3-*-admission/admission.json` `per_cell_policy` | 每 cell 3 策略 × 3 seeds；每段 11,894–45,874 筆 | 每段 Wilson 95 %（表內）；配對差三個 seed 全部同號 | 只對這條 1.5 倍 5 分鐘突發與 ADR 0012 的 C、Q、T | 2026-09-11 `reproduce-lite` 重建與量測當下一致 |
| 15 | 限流讓 attainment 提高約 0.4、突發後 10 s 內恢復，拒絕 12–21 %；原生排隊 3.5–14 分鐘恢復、FP8 突發段 TTFT p95 268 s | 同 #14，`time_to_recover_s`、`rejection_rate`、`ttft_p95_burst_s` | 同 #14 | time-to-recover 為 5 s 格點；FP8 原生排隊有 1 個 seed 未恢復 | 同 #14 | 同上 |
| 16 | FP8 的 C = 256 在突發下 TPOT p95 56–58 ms，突發段 attainment 0.01–0.05；BF16 的 C = 40 為 24 ms、0.69 | 同 #14，`tpot_p95_burst_s`（`phase_summaries.burst`） | 同 #14 | — | 只說明 closed-loop 推得的 C 在突發下對 FP8 過於樂觀，不宣稱安全 C 的值（未量） | 同上 |
