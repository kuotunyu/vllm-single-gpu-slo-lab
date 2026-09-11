# ADR 0016 — W4 結果：speculative decoding 在低負載加速 TPOT，在高批次下的代價（量測中，2026-09-12）

- 日期：2026-09-12（GPU 夜間量測 04:05 起；本文件隨量測進度更新，結果表在 `W4 NIGHT DONE` 後填入）
- 狀態：草稿
- 協定：ADR 0015；計畫與逐時紀錄：`docs/superpowers/plans/2026-09-11-w4-specdec.md`
- 證據：`evidence/raw/w4/<cell>/{closed-loop/seed-1,open-loop/seed-{1,2,3}}/`，smoke 在 `evidence/raw/w4/smoke/`；表 `analysis/tables/w4-{fp8,q4b}-specdec/`（`make reproduce` 從證據重建，含 `specdec.json` 的配對表）；圖 `evidence/plots/w4-*.svg`

## 結果表

（量測完成後填入：每個 family 的 r_sat、r_sat 比、C、粗網格 r_SLO、接受率；closed-loop 逐 concurrency 的配對差；open-loop 逐 rate 的 attainment 與 TPOT p95 配對差與同號檢查。）

## 量測前與 smoke 就確定的事實

1. **指標名稱**：開加速的伺服器多出 `vllm:spec_decode_num_{drafts,draft_tokens,accepted_tokens}_total` 與 `vllm:spec_decode_num_accepted_tokens_per_pos_total{position}`（各有 `_created` 對應），與 ADR 0015 依原始碼推得的名稱一致；已補進 `evidence/metrics-names.txt`（96 → 104）。
2. **probe TPOT 在加速 cell 讀到的是每步時間，不是每 token 時間。** speculative decoding 讓 vLLM 每個 SSE chunk 帶多個 token（4B + EAGLE-3 的單流請求 132 個 token 只有 75 個 chunk，37–102），warm-up 探針用 chunk 數除，所以讀到 12.6 ms；逐筆紀錄用伺服器的 `completion_tokens` 算，同一段的單流 TPOT 是 7.5 ms（p95 9.5 ms）。結論：各段的 TPOT／attainment 不受影響；`probe_tpot_median_s` 在加速 cell 只當同 cell 內的租戶訊號，不能拿來當單流 TPOT 報告，單流數字以 c = 1 段的紀錄為準。
3. **n-gram 讓同預算下的 KV cache 少 25 %。** 同一組 FP8 旗標，沒開加速 KV 76,112 tokens（W3、`fp8-none`），開 n-gram 56,832 tokens：vLLM 0.28 為每個請求保留 k = 3 個草稿 slot 的緩衝，占掉 0.82 預算的一部分。c = 256 滿載需要 61,440 tokens，`fp8-ngram` 在 c = 256 可能 preemption；旗標不改（ADR 0015 第 5 條），manifest 的 `preemptions` 記錄實際發生數，這是「同記憶體預算下加速的代價」的一部分。4B + EAGLE-3 的 KV 為 79,424 tokens，高於滿載需求。
4. **取樣設定**：inference-perf 的請求只帶 `max_tokens`、`ignore_eos`、`stream`，不帶 temperature；vLLM 0.28 預設套用 checkpoint 的 `generation_config.json`，Qwen3-8B-FP8 與 Qwen3-4B 都是 temperature 0.6、top_p 0.95、top_k 20。五個 cell 相同；接受率是 rejection sampling 在這組取樣參數下的值（warm-up 探針另用 temperature 0）。
5. **smoke 數字（不進結果表）**：4B + EAGLE-3 接受率 0.26、平均接受長度 1.77–1.80（每個位置 0.47／0.22／0.09；模型卡在對話 benchmark 上報 2.08，本 lab 是 Shakespeare 續寫），20 rps 時 TPOT p50 11.4 ms／p95 15.4 ms，c = 256 只有 18.2 rps、TPOT p95 70 ms；8B FP8 + n-gram 接受率 0.52、平均接受長度 2.57（0.64／0.51／0.43），單流 TPOT p50 12.4 ms（W2 直連沒加速 18.9 ms），20 rps 時 p95 27.3 ms。五段都零 preemption、shim 零 502。
6. **時程**：smoke 04:05–04:20（含兩次伺服器啟動），閘門 `SMOKE_OK`；`fp8-none` 04:20 開始。

## 量測品質

（量測完成後填入：可疑段與隔離重跑、`CLOSED_LOOP_SHORT` 觸發、preemption 數、Windows committed 最大值、probe 差距、錯誤率、空文字串流。）

## Claim ceiling

所有數字只對：這組引擎旗標與 `--gpu-memory-utilization 0.82`、WSL2、與 Windows 桌面共用的單張 RTX 4090、Qwen3-8B-FP8 與 Qwen3-4B（BF16）、inference-perf `synthetic` 的 Shakespeare 語料切片 108 → 132 tokens、`ignore_eos`、Qwen3 預設取樣（temperature 0.6）、ADR 0015 的 n-gram（k 3、lookup 2–4）與 EAGLE-3（k 3）參數、所有段經 passthrough shim。n-gram 的接受率隨文本重複程度變化，不外推到其他語料；EAGLE-3 只屬於 Qwen3-4B（claim ceiling 4）；TMMLU+ 不重測（greedy 下 speculative decoding 的輸出與目標模型相同）。

## 尚未量測

- FP8 突發安全上限 C = 192（ADR 0013）。
- 其他 k 或 lookup 範圍、其他語料、spec decode × admission、× BF16／AWQ。
