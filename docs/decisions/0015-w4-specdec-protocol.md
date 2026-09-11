# ADR 0015 — W4 speculative decoding 協定：自然文字 prompt、穩態 Poisson 掃描、以 none cell 的 r_sat 錨定速率、n-gram 與 EAGLE-3 參數、接受率指標

- 日期：2026-09-11（任何 W4 資料產生之前定案；GPU 時段由使用者另行安排）
- 狀態：已採納（協定）；量測結果另寫 ADR 0016
- 計畫：`docs/superpowers/plans/2026-09-11-w4-specdec.md`
- 依據：`docs/HANDOFF.md`「W4 開工前必須定案的事」1–8 項；規格 §3.1 decoding 軸、§2.2 claim ceiling 4

## 要回答的問題

加速解碼（n-gram、EAGLE-3）在低負載與高負載下，對 TPOT 與容量各有什麼影響（規格 §3.1）。五個 cell，同一晚、同一套協定：

| cell | 模型 | 加速 | family | 用途 |
|---|---|---|---|---|
| `fp8-none` | `Qwen/Qwen3-8B-FP8` | 無 | fp8 | 8B 基準（重量，不沿用 W2，理由見第 7 條） |
| `fp8-ngram` | 同上 | n-gram | fp8 | 8B 唯一可用的加速（沒有 8B 的 EAGLE-3 head，claim ceiling 4） |
| `q4b-none` | `Qwen/Qwen3-4B` | 無 | q4b | 4B 基準 |
| `q4b-eagle3` | 同上 + `AngelSlim/Qwen3-4B_eagle3` | EAGLE-3 | q4b | W1 已驗證可載入（ADR 0004） |
| `q4b-ngram` | 同上 | n-gram | q4b | 同一目標模型上兩種 drafter 的對照 |

量測順序依價值排：`fp8-none` → `fp8-ngram` → `q4b-none` → `q4b-eagle3` → `q4b-ngram`。時段被中斷時先損失最不重要的 cell。

## 決定

1. **Prompt 是自然文字，不是隨機 token。** 用 inference-perf 的 `synthetic` 產生器（W2 的設定）：從內附的 Shakespeare 語料以每筆隨機起點切出恰好 108 個 token 的文字，輸出以 `ignore_eos` 固定 132 個 token。W3 的 `random`（隨機 token）不能用：n-gram 靠序列裡重複的片段猜下一個 token，EAGLE-3 的 draft head 是用自然文字訓練的，隨機 token 下量到的必然是「沒有加速」。`shareGPT` 需要從 HF Hub 下載，量測主機是 `HF_HUB_OFFLINE=1`，不採用。
   - Claim ceiling：n-gram 的接受率取決於文本的重複程度。Shakespeare 切片加上模型的自由續寫是**低重複的自然文字**，量到的是 n-gram 的下界情境；不外推到 RAG、摘要、程式碼補全這類 prompt 與輸出大量重疊的工作負載。
   - W2 的 prefix cache 命中率為 0（每筆隨機起點），W4 同樣記錄 `prefix_cache_hits_total` 證明。
2. **負載是穩態 open-loop Poisson 掃描加 closed-loop 網格，不用突發 trace。** W4 不量 admission，`trace_replay` 只能配隨機 token prompt，所以不用。inference-perf 的 Poisson timer 沒有種子（W2 也是如此，讀原始碼確認 `np.random.default_rng()` 未帶種子），三個 seed 只固定 prompt 序列，不固定到達序列；這與 W2 相同，可以接受。
3. **網格。**
   - closed-loop：c ∈ {1, 8, 32, 128, 256}，seed 1，`num_requests` 沿用 W2 的表（90、780、2,800、6,700、9,000）乘以 cell 的速度係數（`fp8-*` 1、`q4b-none`／`q4b-ngram` 2、`q4b-eagle3` 3；`w4-night.sh` 的 `NREQ_SCALE_*`），丟棄前 60 s。係數的理由：CPU 演練發現加速 2.2 倍的假引擎在 W2 的筆數下 c = 1、8 兩段都在 60 s 丟棄期內跑完，量測窗為空、r_sat 無法計算，cell 就停了；4B 的吞吐約為 8B FP8 的兩倍、EAGLE-3 的單流最多再快兩到三倍，GPU 上會碰到同樣的事。chain 另會把仍然沒有量測窗的段隔離並以 3 倍筆數重跑一次（log 標 `CLOSED_LOOP_SHORT`）。給每個 cell 自己的 r_sat、C、單流與小批次的 TPOT，以及接受率對批次大小的曲線。
   - open-loop：{0.1, 0.25, 0.5, 0.75, 0.9} × r_ref，seeds 1、2、3，每點 300 s，丟棄前 60 s。低 rate 看 TPOT 是否改善，高 rate 看容量是否下降。
   - **r_ref 是同一 family 的 `none` cell 的 closed-loop r_sat**（fp8 family 用 `fp8-none` 的，q4b family 用 `q4b-none` 的），不是各 cell 自己的 r_sat。這樣同一 family 的三個 cell 在**相同的 offered rate、相同的 seed** 下量測，可以逐 (rate, seed) 配對比較；若各 cell 用自己的 r_sat，速率不同就無法配對。加速 cell 的容量若比基準低，0.9 × r_ref 會過載，這本身就是「容量下降」的證據。W2 各 cell 用自己的 r_sat 是因為 W2 問的是每個精度各自的 r_SLO；W4 問的是相對基準的差。
   - 這個 5 點網格上算出的 r_SLO 解析度很粗（0.5 → 0.75 之間差 50 %），只作附帶報告並標明「粗網格」；W4 的 headline 是每個 rate 的 attainment 與 TPOT 配對差，不是 r_SLO。
   - 每個 cell 兩個 vLLM session：closed-loop 一個，open-loop 三個 seed 共用一個（15 段，seed 為主序）。W2 每個 seed 一個 session 是因為 BF16 的 KV 大小跨 session 不穩；FP8 在 W3 的三個 session KV 完全相同（76,112 tokens），4B 的 KV 有餘裕（第 5 條），且 W4 的配對是跨 cell 而不是跨 session。省下每個 cell 兩次伺服器啟動與 100 筆 warm-up，約 14 分鐘。session 層的 `vllm.log`、`quiet_gpu.json` 放在 `seed-1` 目錄，`seed-2`、`seed-3` 目錄只有各段的檔案。
4. **spec-decode 參數（量測前凍結）。**
   - n-gram：`num_speculative_tokens: 3`、`prompt_lookup_max: 4`、`prompt_lookup_min: 2`（`config/specdec/ngram.yaml`）。k = 3 與 EAGLE-3 相同，兩種 drafter 每步提出同樣長度的草稿，差別只在 drafter；`prompt_lookup_max` 4 取 vLLM 文件範例的值；`prompt_lookup_min` 2 讓 2、3-gram 的匹配也能出草稿（vLLM 0.28 原始碼在兩者都沒給時預設 5／5，並註明 arbitrarily chosen、待調；只給 max 時 min 會等於 max）。在低重複的文本上，4-gram 才出草稿會幾乎永遠不出草稿，量不到 n-gram 的代價與收益。
   - EAGLE-3：`num_speculative_tokens: 3`（W1 載入時的值，ADR 0004；`config/specdec/eagle3_4b.yaml`）。
   - 兩個 YAML 由 `slo-lab specdec-flags <yaml>` 轉成 `--speculative-config <緊湊 JSON>`，driver 不手寫 JSON；manifest 的 `engine_flags.extra` 記下完整字串。
5. **引擎旗標與 W2／W3 相同**：`--max-model-len 4096 --gpu-memory-utilization 0.82 --max-num-seqs 256 --max-num-batched-tokens 2048`，環境 `VLLM_WSL2_ENABLE_PIN_MEMORY=1`、`VLLM_USE_FLASHINFER_SAMPLER=0`、`HF_HUB_OFFLINE=1`。0.82 絕不調高。
   - 讀 vLLM 0.28 原始碼（`config/vllm.py` `_set_max_num_scheduled_tokens`）確認：開 spec decode 時 `max_num_scheduled_tokens` 仍等於 `max_num_batched_tokens`（2048），只是因為小於 8192 會印一則 suboptimal 警告（W1 的 EAGLE-3 載入 log 就有）。草稿 token 的額外 slot（EAGLE-3 為 0，n-gram 為 k）只要求 `max_num_batched_tokens` 大於 slot 數。因此 prefill chunk 與 token budget 在五個 cell 完全相同，警告是預期的、無害的，寫進 runbook 不當錯誤處理。
   - 4B 的 `--max-num-seqs` 用 256：W1 在 0.90 預算下 4B 的 KV 為 90,480 tokens（EAGLE-3 為 83,952），0.82 估約 75,000／69,000，高於 256 × 240 = 61,440 的滿載需求。smoke 在 c = 256 下確認 `num_preemptions_total` 的差分為 0；若有 preemption，4B 三個 cell 的 `--max-num-seqs` 一律降到 192，寫進 ADR 0016。
6. **接受率指標。** `evidence/metrics-names.txt` 是在沒開加速時凍結的，沒有 spec-decode 指標。依 vLLM 0.28 原始碼（`v1/spec_decode/metrics.py`），開加速的伺服器多出三個 counter 與一個帶 `position` 標籤的 counter：`vllm:spec_decode_num_drafts_total`、`vllm:spec_decode_num_draft_tokens_total`、`vllm:spec_decode_num_accepted_tokens_total`、`vllm:spec_decode_num_accepted_tokens_per_pos_total{position}`（`_total` 後綴是 prometheus_client 對 Counter 的慣例，與 `request_success_total` 同）。
   - 前三個加進 `metrics_scraper.TRACKED`（連同 `vllm:num_preemptions_total`），所以 `metrics.csv` 每 5 s 一列、manifest 的 `metrics_before/after` 有段前後快照。
   - 每段 manifest 新增 `spec_decode`：段內差分的 drafts、draft_tokens、accepted_tokens；接受率 = accepted ÷ draft tokens；平均接受長度 = 1 + accepted ÷ drafts；每個位置的接受率 = 該位置 accepted ÷ drafts。沒開加速的伺服器沒有這些 counter，欄位為 null。另新增 `preemptions`（`num_preemptions_total` 的段內差分）。
   - smoke 第一次在開了加速的伺服器上 `curl /metrics`，把實際出現的名稱補進 `metrics-names.txt`（標明 W4 補記）；名稱若與原始碼推得的不同，以實際為準並修正 `TRACKED`，在任何正式段之前 commit。
7. **所有 cell 都經過 `passthrough` shim**（port 8021，vLLM 在 8013）。直連時 inference-perf 重用 keep-alive 連線會撞上 uvicorn 的 5 s 閒置關閉，約 0.3 % 的請求變成連線錯誤（ADR 0012 附錄）；經 shim 為零，且與 W3 一致。shim 的 `/_shim/stats` 每段記錄，`upstream_errors` 必須為 0。W3 smoke 量到 shim 對 TPOT p50 加 3 %、p95 加 2 %，W4 的比較全在 shim 之後，不受影響。
   - **`fp8-none` 重量，不沿用 W2 的 FP8**，三個理由：W2 直連（無 shim）；W2 的 `records.jsonl` 用 inference-perf 重新 tokenize 的 token 數（1.45 % 請求算錯，adapter 已改用伺服器的 `completion_tokens`）；W2 的到達序列不可重現。配對差裡不能混進這三個差異。W2 的 FP8 數字仍可作為量級對照（c = 1 TPOT 18.9 ms、r_sat 41.35、r_SLO 26.2）。
8. **比較方式。** 同一 family 內，每個加速 cell 對 `none` cell 以 (seed, concurrency) 與 (seed, offered rate) 配對：報 TPOT p50／p95 差、TTFT p95 差、attainment 差、達成 rps 比、output tok/s 比、tok/Wh 比、r_sat 比、C、粗網格 r_SLO，以及該段的接受率與平均接受長度。open-loop 的配對差每個 rate 取三個 seed 的平均並標示是否同號；n = 3，不做 bootstrap CI，attainment 另附 Wilson 95 %。任一邊被標為可疑的段不配對。實作在 `slo_lab.specdec_analysis`，由 `batch_analysis.run` 在 manifest 帶 `specdec` 標籤時自動產生 `specdec.json` 與 `tables.md` 的「Speculative decoding」節，`make reproduce` 重建。
9. **可疑規則沿用 W2**：probe TPOT 超過同 cell 最佳值 15 %、W／util 點低於 2.0、Windows committed 超過實體，即隔離該段並重跑，最多兩輪，證據只在重跑結束後 promote。probe TPOT 在加速 cell 是「加速後的單流 TPOT」，規則本來就只在同 cell 內比較，不受影響。
10. **GPU smoke 與閘門**（約 20 分鐘，`scripts/wsl/w4-night.sh` 第一步，`check_w4_smoke.py` 判定，失敗即停在 `SMOKE_FAILED`）：
    - `q4b-eagle3` 一個 session：closed c = 1（60 筆）、c = 256（2,000 筆）、open-loop 20 rps 90 s；`fp8-ngram` 一個 session：closed c = 1（60 筆）、open-loop 20 rps 90 s。warm-up 30 筆、re-warm 10 筆（smoke 不進結果表）。
    - 通過條件：每段 manifest 存在、inference-perf 退出碼 0、紀錄數 > 0、錯誤 ≤ 0.5 %、`spec_decode.drafts` > 0 且接受率在 (0, 1]、`preemptions` = 0（含 4B 的 c = 256）、shim `upstream_errors` = 0、`metrics.csv` 有列、manifest 的 `specdec`／`family` 標籤正確。接受率為 0 或沒有草稿表示設定沒生效（例如 n-gram 從未匹配），停下來檢查而不是量一整晚的「沒有加速」。
    - 證據 promote 到 `evidence/raw/w4/smoke/`，不進結果表。
11. **GPU 預算**（每段估時依 W2／W3 實測：伺服器啟動 2.5 min、100 筆 warm-up 4.3 min、20 筆 re-warm 0.9 min、inference-perf 收尾每千筆 10 s）：

    | 段 | 內容 | 估時 |
    |---|---|---|
    | smoke | 第 10 條 | 0.35 h |
    | 每個 cell 的 closed-loop | 啟動 + warm-up + 5 點（負載約 18 min）+ re-warm + 收尾 | 0.55 h |
    | 每個 cell 的 open-loop | 啟動 + warm-up + 15 段 × 約 7.4 min | 1.95 h |
    | 5 個 cell | 5 × 2.5 h（4B 的 warm-up 較快，略少） | 12.6 h |
    | **合計** | smoke + 5 cell | **約 13 h** |
    | 緩衝 | 一輪隔離重跑（不一定發生） | +0.5 h |

    交接文件估的 10 小時偏低：它沒有算進每段的 re-warm、inference-perf 收尾與每個 session 的啟動與 warm-up。可刪減的選項（不影響協定其餘部分）：

    | 選項 | 省下 | 代價 |
    |---|---|---|
    | 拿掉 `q4b-ngram` | 2.4 h | 4B 上兩種 drafter 的對照消失；n-gram 只剩 8B 一組 |
    | open-loop 去掉 0.1 × | 每 cell 0.3 h（5 cell 1.5 h） | 最低負載點只剩 0.25 ×；單流與小批次的 TPOT 由 closed-loop c = 1、8 提供 |
    | closed-loop 去掉 c = 32 | 每 cell 0.08 h（0.4 h） | 接受率對批次大小的曲線少一點 |
    | closed-loop 只留 {1, 256} | 每 cell 0.25 h（1.2 h） | 只剩單流 TPOT 與 r_sat，沒有批次曲線 |
    | open-loop seeds 3 → 2 | 每 cell 0.6 h（3.0 h） | 違反規格 n ≥ 3；同號檢查只剩兩個 seed，要寫進 claim ceiling。不建議 |
    | 用 W2 的 FP8 當 8B 基準 | 2.5 h | 第 7 條的三個差異會混進配對差。不建議 |

    建議的最小組合：拿掉 `q4b-ngram` 並去掉 0.1 ×，約 9.4 h（含 smoke），一晚可完成，四個 cell 仍能回答規格的問題（8B 的 n-gram、4B 的 EAGLE-3，各對自己的基準）。FP8 突發安全上限補點（ADR 0013）另計 1.5 h，與本 ADR 無關。
12. **不在範圍**：8B 的 EAGLE-3（沒有 head，claim ceiling 4）；其他 k 或 lookup 範圍的掃描；其他 prompt 語料；spec decode × admission、× BF16／AWQ；TMMLU+（greedy 下 speculative decoding 的輸出與目標模型相同，品質不重測；只在 smoke 用一筆請求目視確認回覆正常）。

## 量測前的驗證

- 單元測試（`tests/test_specdec.py`）：新指標進 `TRACKED` 且解析正確、per-position counter 解析、段內差分與接受率（含無草稿與無指標的情形）、YAML 轉 `--speculative-config` 為單一 shell word、CLI、配對分析（含缺基準與可疑段）、表格決定性、W2 的舊 manifest 不受影響。
- CPU dry run（`scripts/wsl/w4-dryrun.sh`，2026-09-11 22:13–22:56，兩次）：假引擎 `fake_vllm.py --specdec` 模擬每步 3 個草稿 token 的接受與加速（接受率約 0.39、平均接受長度 2.16），跑 `dry-none` 與 `dry-ngram` 兩個 cell 各 closed {1, 8} 與 open 2 rates × 2 seeds（同一 session）、全部經 shim、promote 到 `evidence/raw/w4-dryrun/`、從證據重建表、跑 `check_w4_smoke.py`，最後刪掉演練證據。第一次演練發現第 3 條的空窗問題並修正；第二次 12 段全部乾淨：`dry-ngram` 兩段空窗被隔離並以 3 倍筆數重跑（窗 86／808 筆），r_sat 9.82 為基準的 2.30 倍，open-loop 速率與基準相同（2.13／3.84 rps），每段 manifest 都有 `specdec`／`family`／`spec_decode`／`preemptions` 與 shim 計數（upstream_errors 0），配對表正確產生，閘門 `SMOKE_OK`。逐段數字在計畫的 run log。

## 對規格與預註冊的關係

規格 §3.1 的 decoding 軸與 §3.3 的 open-loop 網格（{0.25 … 2.0} × r_sat，每點 5 min，丟棄 60 s）是本 ADR 的上位；W4 的網格是規格網格的子集加 0.1 ×，rate 錨定在 family 的 none cell，5 min 與 60 s 不變。`analysis/preregistration.md` 新增 W4 三列（prompt 語料、網格與錨定、spec-decode 參數），引用本 ADR。
