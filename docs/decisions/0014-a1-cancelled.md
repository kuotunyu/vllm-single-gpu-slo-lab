# ADR 0014 — 取消 A1（RunPod 雲端第二種 GPU 的對照）

- 日期：2026-09-11
- 狀態：已採納（使用者決定：「預計取消A1」）

## 決定

設計規格 §7 的 A1，是在 RunPod 租一張 L4 做 FP8、AWQ 的 open-loop 掃描與 admission trace，約 8–10 GPU 小時，上限 USD 10。取消，不做。

## 理由

- 需要使用者本人操作雲端帳號與付款，且要花錢。
- 使用者先前已表示雲端實驗室「太多又太麻煩」（2026-09-03 重規劃）。
- 專案的主要結論（W2 精度、W3 admission）都在 4090 上完成，不依賴第二種 GPU。

## 影響

- claim ceiling 2 收窄為「只講 RTX 4090」，不宣稱任何跨 GPU class 的結果。
- 規格 §8 里程碑 W4 的「A1 RunPod 8–10 h」移除。
- `analysis/ledger/spend.csv` 維持零花費。
