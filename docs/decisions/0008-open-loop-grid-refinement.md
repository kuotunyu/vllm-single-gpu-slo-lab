# ADR 0008 — open-loop 網格在 0.5–0.75 × r_sat 之間加四點

- 日期：2026-09-09 下午
- 狀態：已採納；為 `analysis/preregistration.md`「Open-loop 網格」列的**增補**（原七點不變、不重排）
- 原始紀錄：`evidence/raw/w2/fp8/open-loop/seed-{1,2,3}/`

## 觸發

FP8 open-loop seed 1（0.82 預算，白天）前四點：

| offered rps | × r_sat(43.7) | attainment | TTFT p50 / p95 | TPOT p95 | 說明 |
|---|---|---|---|---|---|
| 10.92 | 0.25 | 1.000 | 0.054 / 0.065 s | 18.7 ms | |
| 21.84 | 0.50 | 1.000 | 0.069 / 0.090 s | 24.6 ms | |
| 32.75 | 0.75 | **0.103** | 0.568 / 2.387 s | **59.8 ms** | TPOT 先破 50 ms：Poisson 突發讓 prefill chunk（2,048 tokens）插進 decode step，closed-loop 的平滑到達看不到這件事（c = 128 時 TPOT p95 只有 27.9 ms） |
| 43.67 | 1.00 | 0.000 | 56.3 / 89.0 s | 58.0 ms | 佇列無界成長；伺服器仍以 43.8 rps 出貨（= r_sat），沒有 timeout（< 300 s） |

r_SLO 規則（所有 seed attainment ≥ 95% 的最高 offered rate，連續向上）在原網格只能給 21.8 rps，而真正的膝點在 21.8–32.75 之間；用 0.5 × r_sat 當 headline 會把容量低估最多 50%。

## 決策

1. 在 {0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0} 之外加 **{0.55, 0.60, 0.65, 0.70} × 43.67 = {24.02, 26.20, 28.39, 30.57} rps**，三個 seed 都跑；每點仍 5 min、丟棄前 60 s。原七點照舊全部跑完（1.25／1.5／2.0 × 的崩潰段是 admission 策略的動機證據）。
2. r_SLO 仍照凍結規則計算，只是網格更密；attainment-vs-rate 曲線與 SLO 敏感度網格（TTFT {0.5, 1, 2} s × TPOT {30, 50, 100} ms）用同一份 raw 重算。
3. 其他精度的 open-loop 直接用 11 點網格（比例同上，乘各自的 r_sat）。
4. 執行方式：`scripts/wsl/w2-refine.sh <seed>...` 只跑該 seed 尚無 manifest 的 rate（可續跑被中斷的批次）。

## 教訓

規格 §3.3 的網格是照「膝點靠近 r_sat」的直覺設計的；在單卡 FP8 上 TPOT p95 才是先破的約束，膝點落在 0.5–0.75 × r_sat。這條記進 claim ceiling：「rps at SLO」與「rps at saturation」差了將近一倍，而且差距來自 decode 與 prefill 的互相干擾，不是佇列。
