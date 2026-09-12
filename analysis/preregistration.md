# Preregistration（2026-09-09 凍結；W2 起不得更動，例外列於下）

狀態：**已凍結**（commit 於 ADR 0006 同批；ADR 0007 修訂引擎記憶體預算並凍結 warm-up；ADR 0009 補齊各精度 `--max-num-seqs`，無例外項）。之後任何變更都要在 `docs/decisions/` 留 ADR。

| 項目 | 規格值 | 定值（來源） |
|---|---|---|
| SLO | TTFT p95 ≤ 1 s 且 TPOT p95 ≤ 50 ms（re-plan §4） | 同規格；`slo_lab.slo.DEFAULT_SLO` |
| Attainment 分母 | offered（429／timeout／5xx 皆計未達；提案） | 同規格 |
| r_SLO 規則 | 所有 seed attainment ≥ 95% 的最高 offered rate，自最低 rate 連續向上（提案） | 同規格；`slo_lab.slo.r_slo` |
| SLO 敏感度網格 | TTFT ∈ {0.5, 1, 2} s × TPOT ∈ {30, 50, 100} ms（提案） | 同規格 |
| Open-loop 網格 | {0.25 … 2.0} × r_sat，每點 5 min，前 60 s 不計 | r_sat（FP8）= 43.7 rps → offered ∈ {10.9, 21.8, 32.7, 43.7, 54.6, 65.5, 87.3} rps；其他精度以各自 r_sat 換算；5 min、丟棄 60 s（ADR 0006）。**增補（ADR 0008）**：{0.55, 0.60, 0.65, 0.70} × r_sat = {24.0, 26.2, 28.4, 30.6} rps，因膝點落在 0.5–0.75 之間 |
| Closed-loop 網格 | concurrency {1 … 128}，每點 3 min（提案） | {1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256}；每點 `num_requests` 依上一輪 rps 取 ≥ 180 s；**丟棄前 60 s**；256 = `--max-num-seqs`（ADR 0006） |
| r_sat 定義 | closed-loop 吞吐平台 | 網格內最大平均 rps；最後一格增幅 ≥ 5% 時標「下界」；不得為追平台把 `--max-num-seqs` 推進 preemption 區（ADR 0006） |
| Admission trace | `config/traffic/burst25.yaml`（提案） | 0.5·r_sat 5 min → 1.5·r_sat 5 min → 0.5·r_sat 15 min；FP8：21.8 / 65.5 / 21.8 rps；BF16：5.70 / 17.10 / 5.70 rps（r_sat 11.40）。以 seeded trace replay 實作，三策略重播同一檔案（ADR 0012，量測前） |
| C、Q、T | C = closed-loop 仍守 SLO 的最大 concurrency；Q = C；T = 1 s（提案） | FP8：C = 256（乾淨點 attainment ≥ 0.9998 直到引擎上限；TPOT p95 41.5 ms）；Q = 256；T = 1 s。BF16：C = Q = 40（W2 closed-loop，ADR 0009）；T = 1 s（ADR 0012） |
| Warm-up | 100 sequential；101–200 vs 201–300 TTFT 中位數差 ≤ 5% 否則 200（提案） | **100 sequential（凍結）**：300 筆檢查 23.6 vs 22.9 ms、差 3.0%（ADR 0007）；各 stage 間 20 筆 re-warm；warm-up 同時記單流 TPOT 作租戶 probe |
| 租戶污染排除（新增） | — | 每 stage `w_per_util_point` < 2.0 或 probe TPOT 偏離同批最佳值 > 15% 即標可疑，排除於 r_sat／C／r_SLO，且必須重跑；門檻為 4090 校準值（ADR 0006）。根因是桌面程式把 WDDM total committed 推過實體 VRAM（ADR 0007）：manifest 記 Windows 端 `committed_mb`，超過實體即污染 |
| Seeds | 3 個，paired | 1、2、3 |
| Bootstrap | B = 1000，percentile bootstrap，95%（提案） | 同規格；attainment 另附 Wilson 95% |
| Client timeout | 300 s（提案） | 同規格（inference-perf `request_timeout: 300`） |
| Quiet-GPU 門檻 | 1024 MiB（提案） | 記憶體 3,072 MiB（WSL2 閒置基線 2.6 GiB）＋ utilization 10%（5 × 1 s 平均）；WSL2 上 process 準則無效（ADR 0002） |
| `--max-num-seqs` | 預設起，preemption 即下調 | FP8：256；AWQ：256；GPTQ-Int4：256（三者在 c = 256 皆無 preemption）；BF16：**40**（0.82 預算下 KV 僅 11,168–29,696 tokens，網格封頂於 40；峰值 KV 86.1%、零 preemption）。ADR 0009 |
| 其他引擎旗標 | — | `--max-model-len 4096 --gpu-memory-utilization 0.82 --max-num-batched-tokens 2048`；`VLLM_WSL2_ENABLE_PIN_MEMORY=1`、`VLLM_USE_FLASHINFER_SAMPLER=0`、`HF_HUB_OFFLINE=1`（ADR 0002）。0.90 → 0.82 於 2026-09-09 中午（ADR 0007）：4090 與 Windows 桌面共用，0.90 只留 0.5 GiB 給桌面，桌面一活動就觸發 VidMm 分頁；夜間 0.90 的乾淨點保留為對照 |
| Prompt 形狀 | 108 / 132 tokens、nonce、`ignore_eos`、thinking 關閉 | 同規格；inference-perf synthetic，completion API（chat 不支援，ADR 0005） |
| Spec-decode 參數 | 依安裝版本文件 | EAGLE-3：`num_speculative_tokens` 3（W1，ADR 0004）；n-gram：`num_speculative_tokens` 3、`prompt_lookup_max` 4、`prompt_lookup_min` 2（`config/specdec/ngram.yaml`；W4，ADR 0015。vLLM 0.28 兩者都沒給時預設 5／5，原始碼註明 arbitrarily chosen；只給 max 時 min = max） |
| W4 prompt 語料（ADR 0015） | 自然文字（規格 §3.1：n-gram 與 EAGLE-3 都依賴文本） | inference-perf `synthetic`（Shakespeare 語料切片、每筆隨機起點）108 → 132 tokens、`ignore_eos`；不用 W3 的隨機 token。claim ceiling：低重複自然文字的下界，不外推到 RAG／摘要／程式碼 |
| W4 網格（ADR 0015） | 規格 §3.3 open-loop 網格的子集 | closed-loop c ∈ {1, 8, 32, 128, 256}（seed 1，`num_requests` 同 W2 的表）；open-loop {0.1, 0.25, 0.5, 0.75, 0.9} × r_ref，seeds 1–3，每點 5 min、丟棄 60 s；**r_ref = 同 family 的 none cell 的 closed-loop r_sat**（`fp8-none`、`q4b-none`），同 family 的 cell 在相同 rate 與 seed 下配對；每 cell 兩個 session（closed-loop 一個、open-loop 三個 seed 共用一個）；所有段經 passthrough shim |
| W4 cell 與順序（ADR 0015） | 規格 §3.1 部分因子 | `fp8-none`、`fp8-ngram`、`q4b-none`、`q4b-eagle3`、`q4b-ngram`（依價值排序，時段中斷先損失最不重要的）；4B `--max-num-seqs` 256，smoke 在 c = 256 零 preemption 才生效，否則 192 並寫 ADR |
| W3 補點：FP8 hard cap 的 C（**非預註冊**，ADR 0018） | — | 預註冊規則取的 C = 256 在突發下對 FP8 過於樂觀（ADR 0013 發現 3），2026-09-12 晚以 C = 192 重跑 hard cap 一個策略、同一套 seeded trace、3 seeds（`scripts/wsl/w3-c192.sh`，表 `w3-fp8-c192-admission`）。這是看了 W3 結果之後才加的探索性補點，不改 W3 的預註冊表與 ADR 0013 的數字；比較只在同 seed 的兩個 hard cap 之間（同一條 trace） |
