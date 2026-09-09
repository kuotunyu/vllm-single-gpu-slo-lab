# Claims audit（W5 填寫；發佈前逐條核對）

README 裡每個數字對應 evidence 路徑、n、CI，以及規格 §2.2 的 claim ceiling 編號。W0 沒有任何 claim。

| # | README 句子 | Evidence 路徑 | n | CI | Claim ceiling | 核對 |
|---|---|---|---|---|---|---|
| 1 | r_SLO（TTFT p95 ≤ 1 s ∧ TPOT p95 ≤ 50 ms）= 26.2 req/s | `evidence/raw/w2/fp8/open-loop/seed-{1,2,3}/ol-rate-*/records.jsonl`；`analysis/tables/w2-fp8-open-loop/open_loop.json` `per_cell.fp8.r_slo` | 3 seeds × 11 rates，每點 5,700–21,000 筆 | 每點 attainment Wilson 95%（表內）；r_SLO 本身無 CI（凍結規則取 min） | FP8、0.82 預算、WSL2、桌面共用；不外推到其他精度／原生 Linux | 2026-09-09 `make reproduce` 重建一致 |
| 2 | r_sat = 41.4 req/s（下界） | `evidence/raw/w2/fp8/closed-loop-v3/seed-1/`；`analysis/tables/w2-fp8-closed-loop-v3/closed_loop.json` | 1 seed，11 點，c = 256 時 6,244 筆 | — | c = 256 = `--max-num-seqs`，192→256 +7%；夜間 0.90 為 43.7 | 同上 |
| 3 | 膝點 26–31 rps，TPOT 先破 50 ms；32.75 rps 起 attainment 0.1–0.6；43.7 rps 起 0 | 同 #1 的 rows | 同 #1 | 同 #1 | 同 #1 | 同上 |
| 4 | 31 Wh／百萬 output token @ r_SLO（316 W、32,100 tok/Wh）；c = 256 時 43,700 tok/Wh | `power_window` 於 `ol-rate-26.20` 三個 seed 與 `cl-conc-256` 的 manifest | 各 stage 240 s、1 s NVML 取樣 | — | 只算 GPU 板卡功耗，不含主機 | 同上 |
| 5 | TMMLU+（FP8）366 / 600 = 0.610 | `evidence/raw/w2/fp8/tmmluplus/slice-{1,2,3}.json` | 600 題（3 × 200，分層、不重疊） | 各切片 Wilson 95% 於 JSON | 單一精度絕對值；配對差待其他精度 | 同上 |
