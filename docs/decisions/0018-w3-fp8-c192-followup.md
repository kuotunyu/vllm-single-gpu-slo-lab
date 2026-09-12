# ADR 0018 — W3 補點：FP8 的 hard cap 上限改 C = 192，突發段 attainment 由 0.01 升到 0.49（非預註冊）

- 日期：2026-09-12（量測 21:34–23:18，3 段，1 小時 44 分 GPU）
- 狀態：已採納；**探索性補點，非預註冊**（`analysis/preregistration.md` 已加註記列）；不改 W3 的表與 ADR 0013 的數字，只補一個對照
- 起因：ADR 0013 發現 3。預註冊規則「C = closed-loop 仍守 SLO 的最大並行」對 FP8 取到 256（W2 closed-loop 在 c = 256 時 TPOT p95 41.5 ms），但突發時新請求的 prefill 插進 decode，256 並行的 TPOT p95 升到 57–58 ms，放行的請求也超標，突發段 attainment 只有 0.005–0.010。這次只換一個數：C = 192（W2 closed-loop 網格裡 256 的前一格），其餘協定與 W3 完全相同
- 證據：`evidence/raw/w3/fp8-c192/trace/seed-{1,2,3}/trace-hard_cap/`；表 `analysis/tables/w3-fp8-c192-admission/`（`make reproduce` 重建）；driver `scripts/wsl/w3-c192.sh`（`w3-trace-chain.sh` 只跑 `hard_cap` 一個策略，`RERUN_MAX=1`）；Windows 端顯存 `evidence/raw/w3/win-vram-2026-09-12-c192.log`

## 做法

同 ADR 0012：每個 seed 一個 vLLM session（旗標同 W3，`--gpu-memory-utilization 0.82`、`--max-num-seqs 256`），shim `hard_cap --capacity 192`，重播 W3 的同一條 seeded trace（`make-trace` 以 rate_ref 43.67 與 seed 1–3 產生；三個 seed 的 `trace-seed-N.json` 與 manifest 內的 trace sha256 都與 `evidence/raw/w3/fp8/` 的相同，所以 C = 192 與 C = 256 在同 seed 下看到的是完全相同的到達序列）。warm-up 100 筆、可疑規則、promote 都與 W3 相同。

## 結果（3 seeds 平均；括號內為各 seed）

| 上限 | 整段 attainment | goodput | 拒絕率（整段／突發段） | 突發段 attainment | 突發段 TPOT p95 | 突發段 TTFT p95 | 恢復段 attainment | time-to-recover |
|---|---|---|---|---|---|---|---|---|
| C = 256（W3 hard cap，ADR 0013） | 0.573（0.5734／0.5743／0.5709） | 17.47 rps | 20.9 %／48.7 % | 0.0075（0.0098／0.0071／0.0054） | 57.8 ms | 0.26 s | 0.998 | 0 s |
| **C = 192（本 ADR）** | **0.780**（0.7845／0.7804／0.7763） | **23.80 rps**（23.86／23.87／23.67） | 21.8 %（21.4／21.7／22.2）／50.7 %（49.8／50.7／51.6） | **0.491**（0.501／0.489／0.483） | **45.2 ms**（44.3／44.9／46.4） | 0.23 s（0.20／0.28／0.21） | 0.998（0.9985／0.9974／0.9977） | 0 s |

同 seed（同一條 trace）的配對差，C = 192 減 C = 256，三個 seed 全部同號：

| 指標 | 各 seed | 平均 |
|---|---|---|
| 整段 attainment | +0.211／+0.206／+0.205 | +0.208 |
| goodput | +6.42／+6.31／+6.26 rps | +6.33 rps（+36 %） |
| 拒絕率 | +0.5／+0.8／+1.3 pt | +0.9 pt |
| 突發段 attainment | +0.491／+0.482／+0.477 | +0.484 |
| 突發段 TPOT p95 | −13.5／−12.9／−11.3 ms | −12.6 ms |

對照 W3 的其他組合（ADR 0013）：FP8 原生排隊 0.19、有界佇列（Q = 256）0.59；BF16 hard cap（C = 40）突發段 0.69。

## 發現

1. **C = 192 在突發下守得住 TPOT。** 突發段 TPOT p95 44–46 ms（門檻 50），放行的請求幾乎全部達標：突發段 attainment 0.48–0.50，等於 1 減突發段拒絕率（0.50–0.52）再減不到 1 個百分點。整段 attainment 0.78 是 W3 所有 FP8 組合中最高的（有界佇列 0.59），goodput 23.8 rps 比 C = 256 高 36 %。
2. **代價只有拒絕率多約 1 個百分點。** C = 256 多放行的 64 個並行沒有換來更多達標的請求，只是把所有在途請求都拖過 50 ms。因此 hard cap 的 C 要以「突發下仍 TPOT 安全的並行」取，不能直接用 closed-loop 的平台；預註冊規則對 FP8 過於樂觀這一點（ADR 0013 發現 3）在此量化：差 0.21 的整段 attainment。
3. **邊際只有 5 ms。** 45 ms 對 50 ms 的門檻。更低的 C（例如 160）邊際更大但拒絕更多，192 與 256 之間的最佳值也沒有量；這是單一 C 值的探索，不宣稱「安全 C = 192」是通則。
4. 突發前後與 W3 相同：pre／recovery 段 attainment 0.997–0.999，突發結束後 0 s 恢復；突發段 TTFT p95 0.20–0.28 s（shim 立即拒絕，不排隊）。

## 量測品質

- 3 段全部乾淨、零隔離重跑：probe TPOT 19.9–20.1 ms（W3 的 hard cap 段 19.8–20.0）；Windows committed 最高 23.4 GB（manifest；顯存 log 199 筆最高 23,340 MB，全部低於實體 24,564 MiB）；零 timeout；文字全空的串流 45–53 筆／段（0.1 %，計為未達，W3 為 47–59）。
- 引擎：三個 session 的 KV cache 都是 76,112 tokens（同 W3 FP8）；權重載入 84／54／36 s（page cache 逐次變熱，不是證據）。
- 時程：seed 1 21:34–22:10，seed 2 22:12–22:44，seed 3 22:46–23:18；promote 與分析 23:18；估 1.5 小時，實際 1 小時 44 分。

## Claim ceiling

同 ADR 0013，再加：只量了 C = 192 一個值、只有 FP8、只有 hard cap，不外推到其他 C、有界佇列、BF16 或其他突發形狀；非預註冊，是看了 W3 結果之後加的補點，不能當作事前假設的驗證。

## 對其他文件的影響

README 的 W3 段加一句與一列；`docs/model-card.md` 的 W3 列；`analysis/claims_audit.md` 第 22 列（第 16 列的「安全 C 未量」改為指向第 22 列）；ADR 0013 與 0016 的「尚未量測」補記指向本 ADR；`analysis/preregistration.md` 加非預註冊列；`evidence/README.md` 加 `raw/w3/fp8-c192/` 段。
