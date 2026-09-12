# ADR 0012 — W3 admission trace 協定：trace replay、三策略都經過 shim、每個 seed 一個 session、輪換順序

- 日期：2026-09-11（任何 W3 資料產生之前定案）
- 狀態：已採納；是 `analysis/preregistration.md`「Admission trace」與「C、Q、T」兩列的實作細節，不改動任何凍結值
- 計畫：`docs/superpowers/plans/2026-09-11-w3-admission-trace.md`

## 決定

1. **負載以 inference-perf `trace_replay` 產生，不用多段 Poisson。** inference-perf 0.6.1 的 `run_stage` 要等一段的所有請求完成才開始下一段（`while finished_requests_counter.value < num_requests`），再睡 `load.interval`。三段 Poisson 設定會讓突發段的積壓在沒有新請求的情況下排完，恢復段也延後開始，W3 要量的東西會被抹掉。因此預先產生整段 25 分鐘的到達時間檔（Azure 格式 `TIMESTAMP,ContextTokens,GeneratedTokens`），以單一 stage 重播。
   - 到達時間是分段常數率的 Poisson 過程，由 `random.Random(seed)` 產生：t = 0 固定一筆，之後依 0.5·r、1.5·r、0.5·r 抽指數間隔。reader 只保留兩位小數，所以時間在產生時就落在 10 ms 格點上，提交的檔案就是實際重播的內容（`slo_lab.harness.trace`，`slo-lab make-trace`）。
   - 同一 seed 的三個策略重播同一個檔案，是配對設計。inference-perf 內建的 Poisson timer 沒有種子，W2 的到達序列其實不可重現；W3 的可以，檔案與 sha256 隨證據提交。
   - Prompt 由 `random` 產生器依 trace 每列的 108／132 tokens 產生隨機 token（W2 用 `synthetic` 文本）。延遲與 prompt 內容無關，這個差異不影響比較。
   - 用戶端的 `worker_max_concurrency` 與 `worker_max_tcp_connections` 都設 4096（後者預設每個 worker 2,500），用戶端不可以成為佇列。
2. **速率**：FP8 的 r = 43.67，即預註冊的 21.8／65.5／21.8 rps。這沿用 0.90 預算時的 r_sat，以免改動凍結值；相對 0.82 預算下的 r_sat 41.35，突發是 1.58 倍。BF16 的 r = 11.40，即 5.70／17.10／5.70 rps。
3. **C、Q、T**：FP8 C = Q = 256，BF16 C = Q = 40，都等於各自 W2 closed-loop 的 C，也等於 `--max-num-seqs`。T = 1 s，Retry-After 1 s。
4. **三個策略都經過 shim**，native 用 `passthrough`，所以 shim 對每個策略都相同。量測前修正：shim 上游 `aiohttp` 連線池原本是預設的 100 條，會把引擎限制在 100 個並行請求，已改為不限。shim 行程的 CPU 時間逐段記錄。GPU smoke 另外把 passthrough 的前段 TPOT、TTFT 與 W2 直連同速率（21.84 rps）比較，確認 shim 沒有加入可見的延遲。
5. **Session 與順序**：每個 (precision, seed) 一個 vLLM session。三個策略依 Latin square 輪換，每個策略在每個位置各一次：seed 1 為 passthrough、hard_cap、bounded_queue；seed 2 為 hard_cap、bounded_queue、passthrough；seed 3 為 bounded_queue、passthrough、hard_cap。session 的第一段 warm-up 100 筆，之後各段 20 筆；每段開始前等引擎的 running 與 waiting 都回到 0。
6. **量測窗與指標**：headline 是整段 trace [0, 1500) 的 offered attainment，429、timeout、5xx 都算未達。分段是 pre [60, 300)、burst [300, 600)、recovery [600, 1500)，各段報 goodput、拒絕率、served 請求的 TTFT 與 TPOT p95。time-to-recover 依規格 §3.3：突發結束後第一個 5 s 格點，當時的總佇列（vLLM waiting + shim waiting）為 0，且其後 60 s 內送出的請求 TTFT p95 ≤ 1 s。另報 attainment 版本，即其後 60 s 的 offered attainment ≥ 95%，因為直接拒絕的策略可以立刻滿足 TTFT 條件。
7. **時鐘**：佇列取樣（`metrics.csv`、`shim.csv`）以 monotonic clock 的 `t_mono` 欄位對齊逐筆紀錄。CPU dry run 發現 WSL2 的 wall clock 在一段 2.5 分鐘的測試中相對 monotonic 漂移約 7 s，manifest 另記 `wall_clock_drift_s`。
8. **可疑規則**：probe TPOT 超過同 cell 最佳值 15 %，或 Windows committed VRAM 超過實體，就隔離該段並在新 session 重跑，最多兩輪；證據只在隔離與重跑都結束後才 promote。W/util 只記錄、不作判準，因為 0.5·r 段本來就輕。
9. **比較方式**：每個 (cell, policy) 報各 seed 的值與平均；相對 passthrough 的差以 seed 配對呈現，並標出三個 seed 是否同號。n = 3，不做 bootstrap CI，各段 attainment 另附 Wilson 95 %。

## 量測前的驗證

- 單元測試：trace 產生（決定性、10 ms 格點、各段筆數在 4σ 內、依 inference-perf 的時間規則重播得到相同 offset）、shim 同時放行 150 個請求、trace_replay 設定、adapter origin、shim scraper、`t_mono` 對齊、兩種 time-to-recover、配對差。
- CPU dry run（`scripts/wsl/w3-dryrun.sh`，假引擎 8 slots）：三個策略的紀錄數都等於 trace 列數；hard_cap 的 530 筆與 bounded_queue 的 376 筆拒絕都記為 `rejected_429`；`shim.csv`、`metrics.csv` 有資料；首個請求在啟動後 2.8–19 s 送出；promote 產生 gzip 且沒有 home 路徑；admission 表可從證據重建。

## 不在範圍

- AWQ 的 admission 三策略（ADR 0001：W5 有餘裕再補）。
- `config/traffic/cloud_2p5x.yaml` 的 2.5 倍形狀。

## 附錄：GPU smoke 發現的三件事（2026-09-11 03:49–04:33，正式量測開始前修正）

GPU smoke 在任何正式資料產生前抓到以下問題，修正都在第一段正式量測（04:31 開始）之前 commit，所以 W3 的每一筆正式紀錄都使用修正後的程式。

1. **shim 與 vLLM 之間的 keep-alive 競態**：第一次 smoke 的 passthrough 有 9 筆 502（`Server disconnected`、`Connection reset by peer`），都發生在積壓期間。原因是 shim 重用閒置連線，而 uvicorn 在 5 s 後關閉閒置連線。shim 改為每個請求一條上游連線（`force_close`），修正後 shim 對 vLLM 的失敗數為 0。直連的 `direct` 對照組仍有同類錯誤（inference-perf 的連線池會重用 15 s，比 uvicorn 的 5 s 長），經過 shim 時不會發生，因為 shim 的 aiohttp server 閒置 75 s 才關。`run` 也開啟 handler cancellation；實測 aiohttp 3.14 在 shim 的情境下本來就會把用戶端斷線傳到上游，這個設定是防護。
2. **inference-perf 的 output token 數是把回傳文字重新 tokenize 算出來的**，不是伺服器的 `completion_tokens`。隨機 token prompt 下有 25 % 的請求算成 128–131（W2 的文本 prompt 為 1.45 %），模型先輸出 EOS 時可見文字更短，使 (e2e − TTFT) / (n − 1) 高估 TPOT。adapter 改為優先採用伺服器回報的 `completion_tokens`（`server_usage`，每筆都是 132）。重新解析 smoke 的原始檔，direct 組 [30, 60) s 的 TPOT p95 由 39.2 ms 降為 33.7 ms。W2 的 `records.jsonl` 仍是當時的計數；1.45 % 的差異對 W2 的 TPOT p95 影響待 W5 以原始檔重新解析確認。
3. **文字全空的串流**：模型先輸出 EOS 時，`ignore_eos` 讓它繼續生成特殊 token，每個 chunk 的文字都是空字串，inference-perf 不會為它們記時間戳。這類請求確實完成了，但沒有 TTFT，adapter 保守地算為未達（錯誤）。每段約 0.1 %，三個策略都會遇到。

另外兩項觀察：smoke 的前段只有 60 s，最後幾秒送出的請求會碰到突發，p95 被拉高，所以 smoke 與 W2 的延遲比較只作參考，shim 開銷以同 session 的 direct 對照為準（TPOT p50 +3 %、p95 +2 %）。WSL2 的 wall clock 在開機後持續漂移，每 25 分鐘約 130 s，對齊一律使用 monotonic clock，不受影響。今晚引擎的單流 probe TPOT 為 19.8–20.0 ms，比 W2 高約 5 %，前段 TPOT p95 26 ms（W2 24.6 ms）。

## 補記（2026-09-13）

三策略的請求流程畫成時序圖放在 README「量測系統怎麼接」一節；機制與本 ADR 的決定相同，圖上不放數字。
