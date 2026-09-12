# ADR 0016 — W4 結果：speculative decoding 在低負載加速 TPOT，在高批次下的代價（量測中，2026-09-12）

- 日期：2026-09-12（GPU 夜間量測 04:05 起；本文件隨量測進度更新，結果表在 `W4 NIGHT DONE` 後填入）
- 狀態：草稿
- 協定：ADR 0015；計畫與逐時紀錄：`docs/superpowers/plans/2026-09-11-w4-specdec.md`
- 證據：`evidence/raw/w4/<cell>/{closed-loop/seed-1,open-loop/seed-{1,2,3}}/`，smoke 在 `evidence/raw/w4/smoke/`；表 `analysis/tables/w4-{fp8,q4b}-specdec/`（`make reproduce` 從證據重建，含 `specdec.json` 的配對表）；圖 `evidence/plots/w4-*.svg`

## 結果表

### fp8 family：Qwen3-8B-FP8，n-gram 對 none（量測 04:20–09:09；表 `analysis/tables/w4-fp8-specdec/`）

| cell | r_sat（closed-loop） | C | 粗網格 r_SLO | 接受率 | 平均接受長度 |
|---|---|---|---|---|---|
| fp8-none | 40.17 rps（c = 256，attainment 0.98，TPOT p95 46 ms） | 256 | 20.09 rps | — | — |
| fp8-ngram | 26.67 rps（c = 128；c = 256 三個 session 都分頁，排除，見第 7 條） | 128 | 20.09 rps | 0.52 | 2.56 |

closed-loop 配對（同 concurrency，seed 1）：

| c | rps 比（ngram ÷ none） | TPOT p50 差 | TPOT p95 差 |
|---|---|---|---|
| 1 | **1.72** | −8.8 ms（20.3 → 11.5） | −2.8 ms |
| 8 | 1.28 | −3.3 ms | +1.5 ms |
| 32 | 1.17 | −1.6 ms | +4.1 ms |
| 128 | **0.85**（31.4 → 26.7 rps） | +8.2 ms | +20.3 ms（29 → 49 ms） |
| 256 | 排除（分頁；原始讀值 28.4 rps、attainment 0.22、TPOT p95 92 ms） | | |

open-loop 配對（同 offered rate、同 seed，3 seeds 平均；括號內為三個 seed 是否同號）：

| offered rps | attainment 差 | TPOT p50 差 | TPOT p95 差 | tok/Wh 比 |
|---|---|---|---|---|
| 4.02 | 0（兩者都 1.0） | −3.6 ms（同號） | **+1.7 ms**（同號） | 1.07 |
| 10.04 | −0.001 | −3.5 ms（同號） | **+1.9 ms**（同號） | 1.04 |
| 20.09 | −0.001 | −2.5 ms（同號） | **+3.8 ms**（同號） | 0.95 |
| 30.13、36.15 | 排除（分頁）；原始讀值 attainment 0–0.08（none 0.77–0.90、0） | | | |

讀法：n-gram 在單流快 1.7 倍、到 c = 32 仍有 17 % 的吞吐增益，但 c = 128 就反轉為 −15 %（每步多驗證 3 個草稿 token，批次一大就吃掉解碼步的餘裕）；穩態 Poisson 下同 rate 的 attainment 不變、TPOT 中位數降 2.5–3.6 ms、p95 卻升 1.7–3.8 ms，三個 seed 一致，也就是**n-gram 讓典型請求變快、尾端變慢**，在這組 SLO（p95 門檻）下容量沒有增加；30 rps 以上因記憶體超額而量不到。能耗每 token 在低負載略好（+7 %）、20 rps 略差（−5 %）。

### q4b family：Qwen3-4B，EAGLE-3 與 n-gram 對 none

（量測中，09:09 起；完成後填入。）

## 量測前與 smoke 就確定的事實

1. **指標名稱**：開加速的伺服器多出 `vllm:spec_decode_num_{drafts,draft_tokens,accepted_tokens}_total` 與 `vllm:spec_decode_num_accepted_tokens_per_pos_total{position}`（各有 `_created` 對應），與 ADR 0015 依原始碼推得的名稱一致；已補進 `evidence/metrics-names.txt`（96 → 104）。
2. **probe TPOT 在加速 cell 讀到的是每步時間，不是每 token 時間。** speculative decoding 讓 vLLM 每個 SSE chunk 帶多個 token（4B + EAGLE-3 的單流請求 132 個 token 只有 75 個 chunk，37–102），warm-up 探針用 chunk 數除，所以讀到 12.6 ms；逐筆紀錄用伺服器的 `completion_tokens` 算，同一段的單流 TPOT 是 7.5 ms（p95 9.5 ms）。結論：各段的 TPOT／attainment 不受影響；`probe_tpot_median_s` 在加速 cell 只當同 cell 內的租戶訊號，不能拿來當單流 TPOT 報告，單流數字以 c = 1 段的紀錄為準。
3. **n-gram 讓同預算下的 KV cache 少 25 %。** 同一組 FP8 旗標，沒開加速 KV 76,112 tokens（W3、`fp8-none`），開 n-gram 56,832 tokens：vLLM 0.28 為每個請求保留 k = 3 個草稿 slot 的緩衝，占掉 0.82 預算的一部分。c = 256 滿載需要 61,440 tokens，`fp8-ngram` 在 c = 256 可能 preemption；旗標不改（ADR 0015 第 5 條），manifest 的 `preemptions` 記錄實際發生數，這是「同記憶體預算下加速的代價」的一部分。4B + EAGLE-3 的 KV 為 79,424 tokens，高於滿載需求。
4. **取樣設定**：inference-perf 的請求只帶 `max_tokens`、`ignore_eos`、`stream`，不帶 temperature；vLLM 0.28 預設套用 checkpoint 的 `generation_config.json`，Qwen3-8B-FP8 與 Qwen3-4B 都是 temperature 0.6、top_p 0.95、top_k 20。五個 cell 相同；接受率是 rejection sampling 在這組取樣參數下的值（warm-up 探針另用 temperature 0）。
5. **smoke 數字（不進結果表）**：4B + EAGLE-3 接受率 0.26、平均接受長度 1.77–1.80（每個位置 0.47／0.22／0.09；模型卡在對話 benchmark 上報 2.08，本 lab 是 Shakespeare 續寫），20 rps 時 TPOT p50 11.4 ms／p95 15.4 ms，c = 256 只有 18.2 rps、TPOT p95 70 ms；8B FP8 + n-gram 接受率 0.52、平均接受長度 2.57（0.64／0.51／0.43），單流 TPOT p50 12.4 ms（W2 直連沒加速 18.9 ms），20 rps 時 p95 27.3 ms。五段都零 preemption、shim 零 502。
6. **時程**：smoke 04:05–04:20（含兩次伺服器啟動），閘門 `SMOKE_OK`；`fp8-none` 04:20–06:26（20 段全乾淨）；`fp8-ngram` 06:26 起。

## 量測中的偏離（08:34 停鏈、08:36 續跑）

7. **n-gram 在 256 並行時超出記憶體預算，桌面卡分頁。** `fp8-ngram` 的 c = 256 段一開始，WSL VM 的獨占顯存從 20.9 GB 跳到 24.1 GB（vLLM 剖析時的預算是 0.82 × 24.5 = 20.1 GB），Windows committed 25.4–26.6 GB 超過實體 24.5 GB，桌面被換出、引擎步長變慢（28 rps、attainment 0.22、TPOT p95 92 ms）。三個獨立 session（原始加兩輪重跑）數字相同，證明是 cell 的性質而非租戶。依可疑規則排除；c = 128（26.7 rps，TPOT p95 49 ms）成為 `fp8-ngram` 的 r_sat（下界）。open-loop 的 30 與 36 rps 段（過載、256 個在跑）同樣分頁，同樣排除；**同預算下 n-gram 的高負載點在這張與桌面共用的卡上量不到**，這本身是結果。
8. **分析漏洞與修正。** 共用 session 的 `quiet_gpu.json` 只在 seed-1 目錄，分析器對 seed-2、seed-3 的段查不到實體 VRAM，committed 規則因此對它們失效：`fp8-ngram` seed 2、3 的過載段沒被標，且它們的低負載段（seed 為主序下排在第一個過載段之後）也在超額狀態下量測。修正：分析器改為同時查同一 cell 的其他 `seed-*` 目錄（加測試；重新分析後三個 seed 的六個過載段全部標為可疑）；seed 2、3 的 4／10／20 rps 段作廢重跑。低負載段在超額狀態下的數字與 seed 1 乾淨段相同（分頁換出的是閒置桌面），但規則一視同仁。
9. **open-loop 改為 rate 為主序**（所有 seed 的 4 rps 先跑，再 10、20、30、36），讓過載段落在 session 最後，不再汙染同 session 的低負載段。ADR 0015 寫的是 seed 為主序；各段獨立（各自 re-warm、丟棄 60 s），順序不影響乾淨段的數字。
10. **重跑輪數**：`fp8-ngram` 續跑不再重跑已確認的分頁段（`RERUN_MAX=0`，三個 session 已足夠），三個 q4b cell 用一輪（`RERUN_MAX=1`，協定上限是兩輪）。原因：每一輪過載段的重跑約 55 分鐘且結果可預期。

## 量測品質

（量測完成後填入：可疑段與隔離重跑、`CLOSED_LOOP_SHORT` 觸發、preemption 數、Windows committed 最大值、probe 差距、錯誤率、空文字串流。）

## Claim ceiling

所有數字只對：這組引擎旗標與 `--gpu-memory-utilization 0.82`、WSL2、與 Windows 桌面共用的單張 RTX 4090、Qwen3-8B-FP8 與 Qwen3-4B（BF16）、inference-perf `synthetic` 的 Shakespeare 語料切片 108 → 132 tokens、`ignore_eos`、Qwen3 預設取樣（temperature 0.6）、ADR 0015 的 n-gram（k 3、lookup 2–4）與 EAGLE-3（k 3）參數、所有段經 passthrough shim。n-gram 的接受率隨文本重複程度變化，不外推到其他語料；EAGLE-3 只屬於 Qwen3-4B（claim ceiling 4）；TMMLU+ 不重測（greedy 下 speculative decoding 的輸出與目標模型相同）。

## 尚未量測

- FP8 突發安全上限 C = 192（ADR 0013）。
- 其他 k 或 lookup 範圍、其他語料、spec decode × admission、× BF16／AWQ。
