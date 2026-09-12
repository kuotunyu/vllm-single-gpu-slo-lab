# ADR 0010 — 4090 不計成本與電價，成本軸改報實測能耗

- 日期：2026-09-11
- 狀態：已採納（使用者決定：「如果是用4090不用考慮成本和電價」）；取代設計規格 §6 對 4090 的 $／百萬 token 計算，以及 README「一句話」、claim ceiling 3 與 6 的成本表述

## 決定

本機 RTX 4090 是已擁有的硬體。攤提年限與電價都是個人假設，不是量測；把它們填進公式，只會讓 $／百萬 token 看起來精確，實際上完全由假設決定。因此：

1. 4090 上不計算、不報 $／百萬 token；`config/cost.yaml` 不需要使用者填值。
2. 成本軸改報**實測能耗**：GPU 板卡在 r_SLO 點的平均功耗換算成 Wh／百萬 output token（四精度的值見 ADR 0009）。這是量測值，不依賴任何價格假設，也足以比較精度之間的效率（FP8 為 BF16 的 42%）。
3. `slo_lab.cost`、`slo-lab cost` 與 `config/cost.yaml` 保留不刪：若日後在租用 GPU（A1，RunPod L4）上量測，按小時的租金是觀測值，可直接算 $／百萬 token。付費前仍逐筆先問。
4. W4 不再包含成本表，只剩 spec-decode 兩個 cell。

## 影響

- README 的「一句話」、「30 秒結論」、實驗設計表、claim ceiling 3 與 6、「還沒有」、里程碑依此改寫。
- 能耗只含 GPU 板卡、不含主機，這條 ceiling 不變（Wh 數字是下限）。
- 沒有 $ 數字，自然不做「比 API 便宜」的比較。

## 補記（2026-09-13）

A1 取消（ADR 0014）後不會再有租用 GPU 的量測，第 3 點保留的東西整理為：`slo_lab.cost` 與 `slo-lab cost` 指令保留，`config/cost.yaml.example` 保留為欄位範例；`config/cost.yaml`（owner-input 佔位檔）與只有表頭的 `analysis/ledger/cost.csv`、`spend.csv` 移除，`reproduce-lite` 不再要求它們存在。
