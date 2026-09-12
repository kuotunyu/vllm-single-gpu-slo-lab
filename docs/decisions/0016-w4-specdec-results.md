# ADR 0016 — W4 結果：speculative decoding 讓典型請求變快、尾端變慢，在 p95 門檻的 SLO 下沒有增加容量，高批次下吞吐反轉為 0.6 到 0.85 倍

- 日期：2026-09-12（GPU 量測 04:05–16:51，五個 cell）
- 狀態：已採納；W4 量測結案
- 協定：ADR 0015；計畫與逐時紀錄：`docs/superpowers/plans/2026-09-11-w4-specdec.md`
- 證據：`evidence/raw/w4/<cell>/{closed-loop/seed-1,open-loop/seed-{1,2,3}}/`，smoke 在 `evidence/raw/w4/smoke/`，Windows 端顯存取樣 `evidence/raw/w4/win-vram-2026-09-12.log`；表 `analysis/tables/w4-{fp8,q4b}-specdec/`（`make reproduce` 從證據重建，含 `specdec.json` 的配對表）；圖 `evidence/plots/w4-{tpot,attainment}-vs-rate-{fp8,q4b}.svg`

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

### q4b family：Qwen3-4B，EAGLE-3 與 n-gram 對 none（量測 09:09 起；表 `analysis/tables/w4-q4b-specdec/`）

| cell | r_sat（closed-loop） | C | 粗網格 r_SLO | 接受率 | 平均接受長度 |
|---|---|---|---|---|---|
| q4b-none | 47.62 rps（c = 256，attainment 0.985，TPOT p95 38.7 ms） | 256 | 23.81 rps | — | — |
| q4b-eagle3 | 28.77 rps（c = 256，attainment 0.11；下界旗標，128 → 256 只增 5 %） | 128 | 23.81 rps | 0.27 | 1.80 |
| q4b-ngram | 32.12 rps（c = 256，attainment 0.30；下界旗標） | 128 | 23.81 rps | 0.53 | 2.60 |

EAGLE-3 對 none 的 closed-loop 配對（同 concurrency，seed 1）：

| c | rps 比 | TPOT p50 差 | TPOT p95 差 | tok/Wh 比 |
|---|---|---|---|---|
| 1 | **1.46**（0.70 → 1.02 rps） | −3.4 ms（10.7 → 7.4） | −1.5 ms | 1.59 |
| 8 | 1.45 | −3.5 ms | −1.6 ms | 1.24 |
| 32 | 1.20 | −1.5 ms | +0.4 ms | 0.94 |
| 128 | **0.66**（41.7 → 27.4 rps） | +13.2 ms | +20.5 ms（22.4 → 42.9） | 0.64 |
| 256 | **0.60**（47.6 → 28.8 rps） | +27.7 ms | +43.2 ms（38.7 → 82.0） | 0.62 |

EAGLE-3 對 none 的 open-loop 配對（同 offered rate、同 seed，3 seeds 平均）：

| offered rps | attainment 差 | TPOT p50 差 | TPOT p95 差 | TTFT p95 差 | tok/Wh 比 |
|---|---|---|---|---|---|
| 4.76 | 0（都 1.0） | −4.0 ms | **−2.2 ms**（同號） | +6 ms | 0.94 |
| 11.90 | 0 | −4.0 ms | **−1.9 ms**（同號） | +11 ms | 0.89 |
| 23.81 | 0 | +1.9 ms | **+10.2 ms**（同號，16.5 → 26.7 ms） | +72 ms | 0.86 |
| 35.71 | **−0.91**（none 0.87–0.98，EAGLE-3 三個 seed 都 0） | +35.5 ms | +32.4 ms | +67 s（排隊） | 0.77 |
| 42.86 | 0（都 0） | +15.2 ms | +32.0 ms | +96 s | 0.74 |

讀法：EAGLE-3 在單流與小批次快 1.45 倍、c = 32 剩 1.2 倍，c = 128 起反轉為 0.66 倍、c = 256 為 0.60 倍，比 8B 的 n-gram 反轉得更早也更深（接受率只有 0.27，每步為 3 個草稿 token 付出的驗證成本大多白費）。穩態 Poisson 下：12 rps 以下 TPOT p95 低 2 ms（attainment 都是 1.0，SLO 容量沒有變），24 rps 時 p95 已高 10 ms，36 rps 時 none 還守得住 SLO（0.87–0.98）而 EAGLE-3 已完全過載（它的 r_sat 28.8 rps 低於 offered 35.7）。在這組 SLO 下，**EAGLE-3 沒有增加容量，反而把 4B 的服務上限從約 36 rps 砍到約 29 rps**；它的收益只在低負載的延遲，且每 token 能耗在所有 rate 都較差（0.74–0.94）。與 8B n-gram 的對照：n-gram 接受率高（0.52）所以 c = 32 仍有增益、c = 128 才反轉；EAGLE-3 接受率低（0.27）且多一個 draft head 的前向，c = 128 就掉到 0.66。

n-gram 對 none 的 closed-loop 配對（4B；seed 1）：

| c | rps 比 | TPOT p50 差 | TPOT p95 差 | tok/Wh 比 |
|---|---|---|---|---|
| 1 | **1.12**（0.70 → 0.78 rps） | −1.1 ms（10.7 → 9.7） | +2.6 ms | 1.39 |
| 8 | 1.09 | −0.7 ms | +3.4 ms | 1.20 |
| 32 | 1.04 | +0.3 ms | +4.7 ms | 1.04 |
| 128 | **0.73**（41.7 → 30.3 rps） | +10.0 ms | +21.3 ms | 0.78 |
| 256 | **0.67**（47.6 → 32.1 rps） | +21.2 ms | +43.7 ms | 0.75 |

n-gram 對 none 的 open-loop 配對（4B；3 seeds 平均）：

| offered rps | attainment 差 | TPOT p50 差 | TPOT p95 差 | tok/Wh 比 |
|---|---|---|---|---|
| 4.76 | 0 | −2.2 ms | **+1.6 ms**（同號） | 1.07 |
| 11.90 | 0 | −2.2 ms | **+1.7 ms**（同號） | 1.04 |
| 23.81 | −0.001 | −0.6 ms | **+4.1 ms**（同號） | 0.98 |
| 35.71、42.86 | 排除（分頁，第 7 條）；原始讀值 attainment 0（none 0.87–0.98、0） | | | |

讀法：同樣接受率 0.53 的 n-gram，在 4B 上單流只快 1.12 倍（8B 上 1.72 倍）：4B 的解碼步本來就短（10.7 ms），每步多驗證 3 個 token 的固定成本比例更高；c = 32 已無增益，c = 128 起 0.73 倍。open-loop 與 8B 同型：中位數略降、p95 升 1.6 到 4.1 ms、attainment 不變。

### 三個加速 cell 的共同結論

1. 單流與小批次變快（n-gram 8B 1.72×、EAGLE-3 4B 1.45×、n-gram 4B 1.12×），增益隨 concurrency 遞減，c = 128 起吞吐反轉為 0.66 到 0.85 倍，c = 256 為 0.60 到 0.67 倍（8B n-gram 在 c = 256 因分頁量不到）。
2. 穩態 Poisson 下，同 offered rate 的 attainment 與 none 相同，粗網格 r_SLO 也相同（fp8 20.09、q4b 23.81）；TPOT 中位數降 0.6 到 4 ms，p95 卻升 1.6 到 10 ms（唯一例外是 EAGLE-3 在 12 rps 以下 p95 低 2 ms），三個 seed 同號。**在「TPOT p95 ≤ 50 ms」這種以尾端定義的 SLO 下，speculative decoding 沒有增加容量**；它縮短典型請求、拉長尾端（草稿被拒的那些步）。
3. 每 token 能耗只在低負載略好（n-gram 1.04 到 1.07），EAGLE-3 全程較差（0.74 到 0.94）。
4. 加速 cell 的服務上限比 none 低：r_sat 26.7 對 40.2（8B）、28.8 與 32.1 對 47.6（4B），也就是在需要吞吐的高負載段，關掉 speculative decoding 才是正確的。

## 量測品質

- 五個 cell 共 100 個正式段（每 cell closed-loop 5 段、open-loop 15 段）全部提交；其中 13 段被可疑規則標記並由分析器排除，全部是同一個原因：Windows committed VRAM 超過實體（`fp8-ngram` 的 c = 256 與 30／36 rps × 3 seeds，`q4b-ngram` 的 35.71／42.86 rps × 3 seeds），都是 n-gram cell 在 256 個請求並行時超出 0.82 預算所致（第 7 條）；被排除的段各以一個新 session 重跑過一次（`fp8-ngram` 的 c = 256 兩次），結果相同。`q4b-eagle3` 的 42.86 rps seed 3 因 committed 超過實體 55 MB 重跑一次，重跑乾淨。probe TPOT 漂移、W／util 兩條規則全程沒有標到任何段；`CLOSED_LOOP_SHORT` 沒有觸發。
- 零 preemption（每段 `preemptions` 都是 0）；shim 對 vLLM 的失敗數每段都是 0；每段 `errors` 都是 0；inference-perf 退出碼全部為 0。
- probe TPOT（同 cell 內）：fp8-none 19.8–20.0 ms、fp8-ngram 17.4–18.3、q4b-none 10.5、q4b-eagle3 12.4–13.0、q4b-ngram 12.3–13.8，都在 15 % 內；加速 cell 的 probe 是每步時間（第 2 條）。
- Windows committed（`evidence/raw/w4/win-vram-2026-09-12.log`，1,438 筆、每 30 s）：乾淨段最高 24.4 GB（`q4b-eagle3` 過載段），全程最高 27,187 MB（16:37，`q4b-ngram` 被排除的過載段重跑），307 筆超過實體，全部落在被排除的段。
- 時程：04:05–16:51 共 12 小時 46 分 GPU，含 smoke 15 分鐘、三個 session 的 c = 256 重跑約 30 分鐘、兩輪過載段重跑約 1 小時 45 分鐘、08:34 的停鏈與補跑 6 段約 35 分鐘；五個 cell 的正式段本身約 9 小時 40 分。
- 分析器修正（第 8 條）只影響共用 session 的 W4；W2、W3 的表在 `make reproduce` 下零 diff。

## 量測前與 smoke 就確定的事實

1. **指標名稱**：開加速的伺服器多出 `vllm:spec_decode_num_{drafts,draft_tokens,accepted_tokens}_total` 與 `vllm:spec_decode_num_accepted_tokens_per_pos_total{position}`（各有 `_created` 對應），與 ADR 0015 依原始碼推得的名稱一致；已補進 `evidence/metrics-names.txt`（96 → 104）。
2. **probe TPOT 在加速 cell 讀到的是每步時間，不是每 token 時間。** speculative decoding 讓 vLLM 每個 SSE chunk 帶多個 token（4B + EAGLE-3 的單流請求 132 個 token 只有 75 個 chunk，37–102），warm-up 探針用 chunk 數除，所以讀到 12.6 ms；逐筆紀錄用伺服器的 `completion_tokens` 算，同一段的單流 TPOT 是 7.5 ms（p95 9.5 ms）。結論：各段的 TPOT／attainment 不受影響；`probe_tpot_median_s` 在加速 cell 只當同 cell 內的租戶訊號，不能拿來當單流 TPOT 報告，單流數字以 c = 1 段的紀錄為準。
3. **n-gram 讓同預算下的 KV cache 少 25 %。** 同一組 FP8 旗標，沒開加速 KV 76,112 tokens（W3、`fp8-none`），開 n-gram 56,832 tokens：vLLM 0.28 為每個請求保留 k = 3 個草稿 slot 的緩衝，占掉 0.82 預算的一部分。c = 256 滿載需要 61,440 tokens，`fp8-ngram` 在 c = 256 可能 preemption；旗標不改（ADR 0015 第 5 條），manifest 的 `preemptions` 記錄實際發生數，這是「同記憶體預算下加速的代價」的一部分。4B + EAGLE-3 的 KV 為 79,424 tokens，高於滿載需求。
4. **取樣設定**：inference-perf 的請求只帶 `max_tokens`、`ignore_eos`、`stream`，不帶 temperature；vLLM 0.28 預設套用 checkpoint 的 `generation_config.json`，Qwen3-8B-FP8 與 Qwen3-4B 都是 temperature 0.6、top_p 0.95、top_k 20。五個 cell 相同；接受率是 rejection sampling 在這組取樣參數下的值（warm-up 探針另用 temperature 0）。
5. **smoke 數字（不進結果表）**：4B + EAGLE-3 接受率 0.26、平均接受長度 1.77–1.80（每個位置 0.47／0.22／0.09；模型卡在對話 benchmark 上報 2.08，本 lab 是 Shakespeare 續寫），20 rps 時 TPOT p50 11.4 ms／p95 15.4 ms，c = 256 只有 18.2 rps、TPOT p95 70 ms；8B FP8 + n-gram 接受率 0.52、平均接受長度 2.57（0.64／0.51／0.43），單流 TPOT p50 12.4 ms（W2 直連沒加速 18.9 ms），20 rps 時 p95 27.3 ms。五段都零 preemption、shim 零 502。
6. **時程**：smoke 04:05–04:20（含兩次伺服器啟動），閘門 `SMOKE_OK`；`fp8-none` 04:20–06:26；`fp8-ngram` 06:26–09:09（含 08:34 停鏈與補跑）；`q4b-none` 09:09–11:19；`q4b-eagle3` 11:19–13:57；`q4b-ngram` 13:57–16:51。

## 量測中的偏離（08:34 停鏈、08:36 續跑）

7. **n-gram 在 256 並行時超出記憶體預算，桌面卡分頁。** `fp8-ngram` 的 c = 256 段一開始，WSL VM 的獨占顯存從 20.9 GB 跳到 24.1 GB（vLLM 剖析時的預算是 0.82 × 24.5 = 20.1 GB），Windows committed 25.4–26.6 GB 超過實體 24.5 GB，桌面被換出、引擎步長變慢（28 rps、attainment 0.22、TPOT p95 92 ms）。三個獨立 session（原始加兩輪重跑）數字相同，證明是 cell 的性質而非租戶。依可疑規則排除；c = 128（26.7 rps，TPOT p95 49 ms）成為 `fp8-ngram` 的 r_sat（下界）。open-loop 的 30 與 36 rps 段（過載、256 個在跑）同樣分頁，同樣排除；**同預算下 n-gram 的高負載點在這張與桌面共用的卡上量不到**，這本身是結果。
8. **分析漏洞與修正。** 共用 session 的 `quiet_gpu.json` 只在 seed-1 目錄，分析器對 seed-2、seed-3 的段查不到實體 VRAM，committed 規則因此對它們失效：`fp8-ngram` seed 2、3 的過載段沒被標，且它們的低負載段（seed 為主序下排在第一個過載段之後）也在超額狀態下量測。修正：分析器改為同時查同一 cell 的其他 `seed-*` 目錄（加測試；重新分析後三個 seed 的六個過載段全部標為可疑）；seed 2、3 的 4／10／20 rps 段作廢重跑。低負載段在超額狀態下的數字與 seed 1 乾淨段相同（分頁換出的是閒置桌面），但規則一視同仁。
9. **open-loop 改為 rate 為主序**（所有 seed 的 4 rps 先跑，再 10、20、30、36），讓過載段落在 session 最後，不再汙染同 session 的低負載段。ADR 0015 寫的是 seed 為主序；各段獨立（各自 re-warm、丟棄 60 s），順序不影響乾淨段的數字。
10. **重跑輪數**：`fp8-ngram` 續跑不再重跑已確認的分頁段（`RERUN_MAX=0`，三個 session 已足夠），三個 q4b cell 用一輪（`RERUN_MAX=1`，協定上限是兩輪）。原因：每一輪過載段的重跑約 55 分鐘且結果可預期。`q4b-ngram` 的六個過載段果然在重跑中再次分頁（committed 26.0–27.2 GB），排除。
11. **`q4b-ngram` 曾差點被刪**：使用者中午一度要求盡快收工，`q4b-ngram` 依 ADR 0015 第 11 條的刪減選項準備略過；隨後使用者改為「該做的都做完」，於是照原計畫量測。

## Claim ceiling

所有數字只對：這組引擎旗標與 `--gpu-memory-utilization 0.82`、WSL2、與 Windows 桌面共用的單張 RTX 4090、Qwen3-8B-FP8 與 Qwen3-4B（BF16）、inference-perf `synthetic` 的 Shakespeare 語料切片 108 → 132 tokens、`ignore_eos`、Qwen3 預設取樣（temperature 0.6）、ADR 0015 的 n-gram（k 3、lookup 2–4）與 EAGLE-3（k 3）參數、所有段經 passthrough shim。n-gram 的接受率隨文本重複程度變化，不外推到其他語料；EAGLE-3 只屬於 Qwen3-4B（claim ceiling 4）；TMMLU+ 不重測（greedy 下 speculative decoding 的輸出與目標模型相同）。

## 尚未量測

- FP8 突發安全上限 C = 192（ADR 0013）。（2026-09-12 晚補記：已量，ADR 0018。）
- 其他 k 或 lookup 範圍、其他語料、spec decode × admission、× BF16／AWQ。
