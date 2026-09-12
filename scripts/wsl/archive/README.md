# 第一代 W2 driver（歸檔，2026-09-13）

這些腳本產生了 W2 的部分證據（FP8 第一輪 closed-loop／open-loop、ADR 0008 的膝點補點、FP8 的 TMMLU+ 全集），之後被 `w2-night.sh`、`w2-cell-chain.sh`、`refine-cell.sh`、`run-logged.sh` 取代。留在這裡只作為證據來源的紀錄；路徑與參數對應 2026-09-09／10 當時的目錄佈局，不再維護、不建議直接執行。

| 腳本 | 當時的用途 |
|---|---|
| `w2-chain.sh` | FP8 closed-loop 網格與 open-loop 7 點 × 3 seeds 的第一條鏈（ADR 0006–0007） |
| `w2-refine.sh` | open-loop 膝點補點 0.55／0.60／0.65／0.70 × r_sat（ADR 0008） |
| `w2-followup.sh`、`w2-followup-night.sh` | 第一條鏈之後的補點、TMMLU+ 切片與全集、其他精度的膝點補點 |
| `tmmlu-only.sh` | 單獨起伺服器跑一個 cell 的 TMMLU+（FP8 全集） |
