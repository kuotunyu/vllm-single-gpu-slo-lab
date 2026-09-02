# Preregistration（草稿；W1 結束時凍結，W2 起不得更動）

狀態：**未凍結**。本檔在 W1 清單 10 項全過後填入定值並以 commit 凍結；之後任何變更都要在 `docs/decisions/` 留 ADR。

將凍結的項目（來源：設計規格 §3；標「提案」者為規格自行設計）：

| 項目 | 規格值 | W1 定值 |
|---|---|---|
| SLO | TTFT p95 ≤ 1 s 且 TPOT p95 ≤ 50 ms（re-plan §4） | — |
| Attainment 分母 | offered（429／timeout／5xx 皆計未達；提案） | — |
| r_SLO 規則 | 所有 seed attainment ≥ 95% 的最高 offered rate，自最低 rate 連續向上（提案） | — |
| SLO 敏感度網格 | TTFT ∈ {0.5, 1, 2} s × TPOT ∈ {30, 50, 100} ms（提案） | — |
| Open-loop 網格 | {0.25 … 2.0} × r_sat，每點 5 min，前 60 s 不計 | — |
| Closed-loop 網格 | concurrency {1 … 128}，每點 3 min（提案） | — |
| Admission trace | `config/traffic/burst25.yaml`（提案） | — |
| C、Q、T | C = closed-loop 仍守 SLO 的最大 concurrency；Q = C；T = 1 s（提案） | — |
| Warm-up | 100 sequential；101–200 vs 201–300 TTFT 中位數差 ≤ 5% 否則 200（提案） | — |
| Seeds | 3 個，paired | — |
| Bootstrap | B = 1000，percentile bootstrap，95%（提案） | — |
| Client timeout | 300 s（提案） | — |
| Quiet-GPU 記憶體門檻 | 1024 MiB（提案） | — |
| `--max-num-seqs` | 預設起，preemption 即下調 | — |
| Spec-decode `num_speculative_tokens` | 依安裝版本文件 | — |
