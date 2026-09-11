# ADR 0013 — W3 結果：限流讓突發下的 SLO 達成率提高約 0.4，但上限 C 必須在突發下仍然 TPOT 安全

- 日期：2026-09-11（正式量測 04:31–13:04，18 段）
- 狀態：已採納；W3 量測結案
- 協定：ADR 0012（含附錄）；計畫與逐時紀錄：`docs/superpowers/plans/2026-09-11-w3-admission-trace.md`
- 證據：`evidence/raw/w3/{fp8,bf16}/trace/seed-{1,2,3}/`；表 `analysis/tables/w3-{fp8,bf16}-admission/`（`make reproduce` 從證據重建）

## 結果表（3 seeds 平均；各 seed 值在表檔）

| cell | 策略 | 整段 attainment | goodput | 拒絕率 | 突發段 attainment | 恢復段 attainment | 突發段 TTFT p95 | time-to-recover |
|---|---|---|---|---|---|---|---|---|
| FP8 | 原生排隊（passthrough） | 0.192 | 5.85 rps | 0 | 0.001 | 0.112 | 268 s | 745、820 s；1 個 seed 未恢復 |
| FP8 | hard cap + 429（C = 256） | 0.573 | 17.47 rps | 20.9 % | 0.007 | 0.998 | 0.26 s | 0 s |
| FP8 | 有界佇列（Q = 256、T = 1 s） | 0.591 | 18.01 rps | 19.9 % | 0.049 | 0.998 | 1.13 s | 5–10 s |
| BF16 | 原生排隊（passthrough） | 0.471 | 3.76 rps | 0 | 0.012 | 0.753 | 113 s | 210–250 s |
| BF16 | hard cap + 429（C = 40） | 0.867 | 6.92 rps | 13.2 % | 0.691 | 0.999 | 0.09 s | 0 s |
| BF16 | 有界佇列（Q = 40、T = 1 s） | 0.856 | 6.84 rps | 12.0 % | 0.667 | 0.998 | 1.02 s | 5–10 s |

相對原生排隊的配對差，三個 seed 全部同號：

| cell | 策略 | attainment | goodput |
|---|---|---|---|
| FP8 | hard cap | +0.381（+0.349 至 +0.407） | +11.6 rps |
| FP8 | 有界佇列 | +0.399（+0.351 至 +0.460） | +12.2 rps |
| BF16 | hard cap | +0.396（+0.391 至 +0.400） | +3.16 rps |
| BF16 | 有界佇列 | +0.385（+0.380 至 +0.388） | +3.08 rps |

## 發現

1. **原生排隊在 1.5 倍、5 分鐘的突發下幾乎全面失守，而且拖很久。** FP8 的佇列等待讓突發段 TTFT p95 到 268 s，接近 300 s 的用戶端逾時；突發結束後要 12–14 分鐘才恢復，其中一個 seed 在 15 分鐘的恢復段內沒有恢復，整段 attainment 0.19。BF16 的超額較小（17.1 對約 11.4 rps 的容量），3.5–4 分鐘恢復，整段 0.47。
2. **兩種限流都把整段 attainment 提高約 0.4，突發結束後 0–10 s 恢復。** 代價是拒絕 12–21 % 的請求，全部集中在突發段。goodput（每秒守住 SLO 的請求數）是原生排隊的 3 倍（FP8）與 1.8 倍（BF16）。
3. **上限 C 的取法決定限流在突發段有沒有用。** BF16 的 C = 40 在突發下 TPOT p95 24 ms，放行的請求守得住 SLO，突發段 attainment 0.69。FP8 的 C = 256 來自 W2 closed-loop（c = 256 時 TPOT p95 41.5 ms），但 open-loop 突發時新請求的 prefill 插進 decode，同樣 256 並行的 TPOT p95 升到 56–58 ms，放行的請求也超過 50 ms，突發段 attainment 只有 0.007–0.049。預註冊的規則「C = closed-loop 仍守 SLO 的最大並行」對 FP8 過於樂觀；突發下的安全 C 應以 open-loop 量測決定，未量，列為 W5 的可選補點。
4. **hard cap 與有界佇列的差距小，方向依 cell 而異。** FP8 有界佇列略高（+0.018，來自一個 seed 突發段 0.13），BF16 hard cap 略高（+0.011）。T = 1 s 等於 TTFT 門檻，排隊將近 1 s 才放行的請求 TTFT 會超過 1 s（BF16 突發段 TTFT p95 1.02 s），所以有界佇列把一部分「立即拒絕」換成「晚到而未達標」：拒絕率少 1 個百分點左右，attainment 幾乎不變。更短的 T 是否更好未量。
5. **shim 不是瓶頸。** shim 行程最多用到一個核心的 10 %（FP8 原生排隊）；GPU smoke 的同 session 對照顯示經過 shim 的 TPOT p50 多 3 %。

## 量測品質

- 18 段正式量測全部乾淨，沒有任何一段被隔離重跑。probe TPOT 在同 cell 內的差距：FP8 1.3 %，BF16 5.6 %（seed 3 的 session 較慢，當時桌面有使用中的瀏覽器），都低於 15 %。Windows committed VRAM 全程最高 22,604 MB，低於實體 24,564 MiB。全程零逾時。
- 每段的紀錄數都比 trace 列數少 1 筆（例如 45,624 對 45,625），每一段都一樣，是 inference-perf trace replay 的行為，影響 0.002 %。
- 文字全空的串流每段 10–59 筆（0.1–0.5 %），依 ADR 0012 附錄保守計為未達。
- 今天引擎的單流 probe TPOT 為 FP8 19.8–20.0 ms、BF16 18.6–19.6 ms，比 W2 高約 5 %；W3 內部的配對比較不受影響。
- GPU smoke 的證據在 `evidence/raw/w3/fp8-smoke/`，不進結果表；它的 `records.jsonl` 是 adapter 修正前解析的，只作為閘門紀錄。

## Claim ceiling

所有數字只對：這組引擎旗標與 `--gpu-memory-utilization 0.82`、WSL2、與 Windows 桌面共用的單張 RTX 4090、Qwen3-8B 的 FP8 與 BF16、108 → 132 tokens 的隨機 token prompt、這條 1.5 倍 5 分鐘的單峰突發，以及 ADR 0012 的 C、Q、T。不外推到其他突發形狀、其他 T、多 replica 或其他 GPU。

## 尚未量測

- W4：speculative decoding，完整版（8B n-gram；4B none、n-gram、EAGLE-3），使用者 2026-09-11 同意做完整版，GPU 時段由使用者另行安排，約 10 小時。
- 突發下 TPOT 安全的 FP8 上限 C（例如 192），以及更短的 T。
