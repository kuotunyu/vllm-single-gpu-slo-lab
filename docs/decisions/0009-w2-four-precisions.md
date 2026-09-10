# ADR 0009 — W2 四精度結果：FP8 在品質、SLO 容量、能耗三軸同時勝出

- 日期：2026-09-10（整夜無人值守量測，04:35–01:15）
- 狀態：已採納；W2 量測結案。`analysis/preregistration.md` 的 `--max-num-seqs` 列由本 ADR 補齊
- 執行計畫與逐時紀錄：`docs/superpowers/plans/2026-09-10-w2-overnight-run.md`（Run log 與 Results 兩節是本 ADR 的原始筆記）
- 證據：`evidence/raw/w2/{bf16,fp8,awq,gptq,fp8-mbt8192}/`；表 `analysis/tables/w2-*`（`make reproduce` 從證據重建並 diff）；配對品質 `analysis/tables/w2-quality-paired/`
- 共同條件：RTX 4090（與 Windows 桌面共用）、WSL2、vLLM 0.28.0、Qwen3-8B、`--gpu-memory-utilization 0.82`、`--max-model-len 4096`、`--max-num-batched-tokens 2048`（對照組 8192）、prompt 108 → output 132 tokens、`ignore_eos`；SLO = TTFT p95 ≤ 1 s 且 TPOT p95 ≤ 50 ms

## 四精度總表

| cell | 權重 | c = 1 rps | 單流 TPOT | r_sat | C | **r_SLO** | 能耗 @ r_SLO | **TMMLU+ 全集** |
|---|---|---|---|---|---|---|---|---|
| BF16 | 15.3 GiB | 0.385 | 19.3 ms | 11.40（被 max-num-seqs 40 封頂） | 40 | **8.55** | 89.8 Wh／百萬 token | **0.5911** |
| **FP8** | 8.8 GiB | 0.384 | 18.9 ms | 41.35（下界，c = 256 = 上限） | 256 | **26.2** | **31.2** | **0.5909** |
| AWQ | 5.7 GiB | 0.893 | 7.8 ms | 30.15（平台） | 192 | **22.61** | 36.2 | 0.5797 |
| GPTQ-Int4 | 5.7 GiB | 0.945 | 7.6 ms | 30.26（平台） | 192 | **22.70** | 36.3 | 0.5703 |

r_SLO：open-loop Poisson，11 個 rate × 3 seeds × 每點 5 分鐘（丟棄前 60 s），凍結規則「所有 seed attainment ≥ 95%、自最低 rate 連續向上」。能耗是 GPU 板卡在 r_SLO 點三個 seed 的平均功耗換算，不含主機。TMMLU+ 全集 19,680 題、greedy、thinking 關閉。

## 發現

1. **FP8 在這張卡上三軸全勝。** 品質與 BF16 無法區分（見下節，p = 0.945），SLO 容量是 BF16 的 3.1 倍、4-bit 的 1.16 倍，每百萬 token 能耗是 BF16 的 35%、4-bit 的 86%。4-bit 唯一贏的是單流延遲。對 24 GB 桌面卡的預設精度選擇：FP8。
2. **精度反轉：4-bit 單流快、飽和慢。** AWQ／GPTQ 單流 TPOT 7.6–7.8 ms，是 FP8 的 2.5 倍快；但飽和吞吐 30.2 rps 比 FP8 的 41.4 低 27%，且在 c = 192 就守不住 SLO（FP8 到 256 仍守得住）。weight-only 4-bit 在小批次 decode（記憶體頻寬受限）時省下權重讀取；大批次時反量化變成算力瓶頸，而 FP8 走 Ada 的 W8A8 tensor core。4-bit 的 prefill 也較慢：c = 64 時 TTFT p95 0.541 s，FP8 0.334 s。
3. **位元寬度決定服務包絡，量化方法不決定。** AWQ 與 GPTQ 的 r_sat（30.15 vs 30.26）、C（192 vs 192）、r_SLO（22.61 vs 22.70）、敏感度網格形狀都一致；兩者只在品質上有顯著差異（AWQ 高 0.95 pts）。
4. **FP8 單流沒有比 BF16 快。** c = 1 時 0.384 vs 0.385 rps、TPOT 18.9 vs 19.3 ms。權重位元組減半並沒有移動 batch-1 decode；降到 4-bit 才有 2.5 倍。FP8 的優勢全在批次與記憶體。
5. **瓶頸在不同精度間換手。** FP8、AWQ、GPTQ 的 SLO 都先被 TPOT 打破（敏感度網格：TPOT 門檻 30 ms 時 r_SLO 降一格，TTFT 0.5–2 s 不影響）。BF16 則是 TTFT 先破：11.4 rps 時 TPOT p95 只有 25–27 ms，TTFT p95 卻 1.1–2.8 s——max-num-seqs 40 與薄 KV 讓請求在佇列裡等。BF16 的敏感度網格在每個門檻都是 8.55 rps，是佇列懸崖而非 decode 速度。
6. **Poisson 過載的服務率低於閉環飽和。** 以完成時間計的 `served_rps`（`slo_lab.batch_analysis.served_rps`）：FP8 在 43.7–87.3 rps 過載時實際完成 33.9–39.4 rps，低於 closed-loop r_sat 41.4；AWQ 在 30.15 rps 時只完成 24.4–25.3 rps。突發到達比等速閉環少了約 15% 吞吐——這是 W3 admission control 要處理的現象。

## 配對品質（同題、以 BF16 為基準）

同一批 19,680 題由四個 cell 各答一次，是配對樣本而非獨立樣本。獨立的 Wilson 區間（各約 ±0.007）會掩蓋真實差異：AWQ 與 GPTQ 的區間重疊，配對檢定卻顯示 AWQ 顯著較高（p = 0.00077）。

| cell | 正確率 | 對 BF16 | 95% 配對 bootstrap 區間 | 只有該 cell 答對 | 只有 BF16 答對 | exact McNemar p |
|---|---|---|---|---|---|---|
| FP8 | 0.5909 | −0.02 pts | [−0.29, +0.27] | 413 | 416 | 0.945 |
| AWQ | 0.5797 | −1.13 pts | [−1.64, −0.63] | 1,110 | 1,333 | 7.0 × 10⁻⁶ |
| GPTQ-Int4 | 0.5703 | −2.08 pts | [−2.55, −1.56] | 1,020 | 1,429 | 1.4 × 10⁻¹⁶ |

實作：`slo_lab.quality`（`mcnemar_exact` 在對數空間加總，因為上千題不一致時 `2**n` 會溢位；以常態近似交叉驗證）；`reproduce-lite` 自動重建此表。GPTQ 為第三方 checkpoint（`JunHowie/Qwen3-8B-GPTQ-Int4`，ADR 0003），品質數字只代表該 checkpoint。

## 對照組：`--max-num-batched-tokens` 8192（FP8，seed 1）

| 設定 | KV cache | r_sat | C | r_SLO | 26 rps 附近 TPOT p95 |
|---|---|---|---|---|---|
| 2048（預註冊） | 76,112 tokens | 41.35 | 256 | 26.2（3 seeds） | 29–31 ms |
| 8192 | 67,888 tokens | 39.44 | **128** | 25.64（1 seed） | 42.1 ms |

更大的 prefill chunk 讓 decode 被打斷更久，FP8 的 SLO 又是 TPOT 受限，所以 8192 沒有任何一項更好，且把 SLO 安全並行數腰斬。2048 維持。

## 交叉驗證：`vllm bench serve`

第二套 load generator 在吞吐上與 inference-perf 相差 5–7%（c = 256：39.42 vs 41.35 rps；c = 64：20.61 vs 22.05 rps），21.8 rps open-loop 的延遲落在同一區間（TTFT p95 107 ms、TPOT p95 29.9 ms）。依規格只記錄、不作結論依據。以摘要 JSON 提交，理由見下節第 4 點。

## 整夜發現並修正的 harness 缺陷

1. **就緒探測競態（遺失整個 AWQ cell 一次）**：`curl … | grep -q 200` 在 `set -o pipefail` 下，`grep -q` 一比對到就結束、curl 收到 SIGPIPE（141），pipefail 讓成功的探測變成失敗。AWQ 伺服器回了 200 後一秒被自己的 `SERVER_NOT_READY` 分支關掉。改用命令替換、重用迴圈結果、curl 5 s 上限、輪詢 180 次（commit `dc3a755`）。以最小重現確認機制。
2. **鏈日誌遺失**：Windows 端 `| tr | grep | tee` 區塊緩衝吃掉所有輸出，AWQ 的失敗只能從 manifest 與 server log 重建。改在 WSL 內寫 ext4 日誌（`run-logged.sh`）並以 `watch-night.sh` 串流事件。
3. **quiet-GPU 單次讀數跳過整個 cell**：對照組的閘門在 BF16 伺服器關閉 20 s 後讀到 utilization 11%（門檻 10%，取樣 15 → 4 遞減），2.5 小時的 cell 被跳過。改為重試 5 次、間隔 30 s，門檻不變（commit `4bb528f`）。
4. **生成文字不可提交**：`vllm bench serve --save-detailed` 的 `generated_texts` 存了模型對隨機 prompt 的自由續寫，其中重現了訓練資料片段（他人的 build 路徑與網站路徑）。secrets 審核抓不到（它查 IP、金鑰、email），是 `grep -r "/home/"` 抓到的。交叉驗證改提交摘要 JSON，逐請求陣列留在 repo 外並記錄原檔 sha256——與 inference-perf 逐請求 JSON 的既有政策一致。
5. **TMMLU+ 評分一錯全失**：`asyncio.gather` 未設 `return_exceptions`，19,680 題中任何一題的連線錯誤會取消全部。改為單題錯誤轉成記錄並計數（全夜 0 錯誤，但保護了四次各約 4–5 分鐘的全集評分）。

## 協定偏離（誠實記錄）

1. **closed-loop 視窗長度不一**：`num_requests` 以 FP8 速率校準，較快的 cell 較早跑完，AWQ／GPTQ 低並行點的量測窗只有 38–124 s（凍結規格為 ≥ 180 s）。刻意維持各 cell 相同的請求數——每個並行度提供完全相同的工作量，才使四精度可比；代價是精度而非偏差。每列都附 `window_s`、`window_records`。
2. **KV cache 大小不可重現**：vLLM 依啟動時可用顯存決定 KV，桌面共用卡上會隨合成器波動。AWQ／GPTQ 四個 session 完全一致（98,400／98,464），BF16 的 closed-loop 與 open-loop session 差 2.7 倍（11,168 vs 29,696 tokens）。BF16 closed-loop 峰值 KV 使用 86.1%、零 preemption、等待佇列始終 0，所以 c = 40 仍有效；但這本身就是發現：BF16 在 24 GB 桌面卡上的服務容量跨重啟不可重現。
3. **每個 cell 橫跨四個伺服器 session**（1 closed-loop + 3 open-loop seed），這是設計而非缺陷。GPTQ 的 closed-loop 另多一個 session 邊界（c = 8 → 16，就緒 bug 中斷所致），KV 差異在 c ≤ 8 不會卡到（8 × 240 = 1,920 tokens），不重跑。
4. **膝點補點**：AWQ、GPTQ、BF16 的膝點都落在凍結網格 0.75 → 1.0 × r_sat 的空洞（FP8 的膝點已由 ADR 0008 的 0.55–0.70 解析到約 8%），故三者各補 0.80／0.85／0.90 × r_sat、三個 seed。見下節。

## 膝點補點結果

（補點完成後填入：AWQ、GPTQ、BF16 在 0.80／0.85／0.90 × r_sat 的 attainment 與更新後的 r_SLO。）

## Claim ceiling

所有數字只對：這組旗標、`--gpu-memory-utilization 0.82`、WSL2、與 Windows 桌面共用的單張 RTX 4090、Qwen3-8B、108 → 132 tokens 的合成負載成立。不外推到原生 Linux、其他 GPU、其他模型大小、其他 prompt 長度分布。GPTQ 數字只代表 `JunHowie/Qwen3-8B-GPTQ-Int4`。能耗只算 GPU 板卡。r_sat 對 FP8 與 BF16 是下界（被 `--max-num-seqs` 封頂），對 AWQ／GPTQ 是實測平台。

## 尚未量測

- W3：admission 三策略（原生佇列 vs 硬上限 429 vs 有界佇列）在 burst trace 下的 attainment／goodput／拒絕率。
- W4：speculative decoding 兩個 cell（n-gram、EAGLE-3）。
- 成本表：`config/cost.yaml` 仍待使用者填 4090 攤提與電價，$／百萬 token 尚未計算（Wh 已有）。
