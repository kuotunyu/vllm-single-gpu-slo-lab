# ADR 0009 — W2 四精度結果：FP8 在品質、SLO 容量、能耗三軸同時勝出

- 日期：2026-09-11（量測連續進行於 2026-09-10 04:35 至 09-11 01:00）
- 狀態：已採納；W2 量測結案。`analysis/preregistration.md` 的 `--max-num-seqs` 列由本 ADR 補齊
- 執行計畫與逐時紀錄：`docs/superpowers/plans/2026-09-10-w2-overnight-run.md`（Run log 與 Results 兩節是本 ADR 的原始筆記）
- 證據：`evidence/raw/w2/{bf16,fp8,awq,gptq,fp8-mbt8192}/`；表 `analysis/tables/w2-*`（`make reproduce` 從證據重建並 diff）；配對品質 `analysis/tables/w2-quality-paired/`
- 共同條件：RTX 4090（與 Windows 桌面共用）、WSL2、vLLM 0.28.0、Qwen3-8B、`--gpu-memory-utilization 0.82`、`--max-model-len 4096`、`--max-num-batched-tokens 2048`（對照組 8192）、prompt 108 → output 132 tokens、`ignore_eos`；SLO = TTFT p95 ≤ 1 s 且 TPOT p95 ≤ 50 ms

## 四精度總表

| cell | 權重 | c = 1 rps | 單流 TPOT | r_sat | C | **r_SLO** | 下一格（失敗） | 能耗 @ r_SLO | **TMMLU+ 全集** |
|---|---|---|---|---|---|---|---|---|---|
| BF16 | 15.3 GiB | 0.385 | 19.3 ms | 11.40（被 max-num-seqs 40 封頂） | 40 | **10.26** | 11.40（+11%） | 74.6 Wh／百萬 token | **0.5911** |
| **FP8** | 8.8 GiB | 0.384 | 18.9 ms | 41.35（下界，c = 256 = 上限） | 256 | **26.2** | 28.39（+8%） | **31.2** | **0.5909** |
| AWQ | 5.7 GiB | 0.893 | 7.8 ms | 30.15（平台） | 192 | **22.61** | 24.12（+7%） | 36.2 | 0.5797 |
| GPTQ-Int4 | 5.7 GiB | 0.945 | 7.6 ms | 30.26（平台） | 192 | **22.70** | 24.21（+7%） | 36.3 | 0.5703 |

r_SLO：open-loop Poisson，每個 cell 11 個預註冊 rate（AWQ／GPTQ／BF16 另加 3 個膝點補點）× 3 seeds × 每點 5 分鐘（丟棄前 60 s），凍結規則「所有 seed attainment ≥ 95%、自最低 rate 連續向上」。「下一格」是第一個有 seed 失敗的 rate，四個 cell 的膝點都被夾到 7–11% 以內，可以互相比較。能耗是 GPU 板卡在 r_SLO 點三個 seed 的平均功耗換算，不含主機。TMMLU+ 全集 19,680 題、greedy、thinking 關閉。

## 發現

1. **FP8 在這張卡上三軸全勝。** 品質與 BF16 無法區分（見下節，p = 0.945），SLO 容量是 BF16 的 2.55 倍、4-bit 的 1.16 倍，每百萬 token 能耗是 BF16 的 42%、4-bit 的 86%。4-bit 唯一贏的是單流延遲。對 24 GB 桌面卡的預設精度選擇：FP8。（補點前 BF16 的 r_SLO 是 8.55，會把 FP8 的優勢誇大成 3.1 倍；見「膝點補點結果」。）
2. **精度反轉：4-bit 單流快、飽和慢。** AWQ／GPTQ 單流 TPOT 7.6–7.8 ms，是 FP8 的 2.5 倍快；但飽和吞吐 30.2 rps 比 FP8 的 41.4 低 27%，且在 c = 192 就守不住 SLO（FP8 到 256 仍守得住）。weight-only 4-bit 在小批次 decode（記憶體頻寬受限）時省下權重讀取；大批次時反量化變成算力瓶頸，而 FP8 走 Ada 的 W8A8 tensor core。4-bit 的 prefill 也較慢：c = 64 時 TTFT p95 0.541 s，FP8 0.334 s。
3. **位元寬度決定服務包絡，量化方法不決定。** AWQ 與 GPTQ 的 r_sat（30.15 vs 30.26）、C（192 vs 192）、r_SLO（22.61 vs 22.70）、敏感度網格形狀都一致；兩者只在品質上有顯著差異（AWQ 高 0.95 pts）。
4. **FP8 單流沒有比 BF16 快。** c = 1 時 0.384 vs 0.385 rps、TPOT 18.9 vs 19.3 ms。權重位元組減半並沒有移動 batch-1 decode；降到 4-bit 才有 2.5 倍。FP8 的優勢全在批次與記憶體。
5. **瓶頸在不同精度間換手，敏感度網格恰好轉置。** 以同一份 raw 重算 TTFT {0.5, 1, 2} s × TPOT {30, 50, 100} ms 的 r_SLO：

   | cell | TPOT 門檻 30／50／100 ms（TTFT 固定 1 s） | TTFT 門檻 0.5／1／2 s（TPOT 固定 50 ms） | 受限於 |
   |---|---|---|---|
   | FP8 | 24.02／26.2／26.2 | 26.2／26.2／26.2 | TPOT（唯一例外：TTFT 2 s 且 TPOT 100 ms 的角落為 30.57） |
   | AWQ | 21.1／22.61／24.12 | 22.61／22.61／22.61 | TPOT |
   | GPTQ | 21.18／22.70／24.21 | 22.70／22.70／22.70 | TPOT |
   | **BF16** | 10.26／10.26／10.26 | **9.69**／10.26／10.26 | **TTFT** |

   完整 3 × 3 網格在各 cell 的 `analysis/tables/w2-<cell>-open-loop/tables.md`。

   量化的三個 cell 對 TTFT 門檻不敏感、對 TPOT 門檻敏感：它們的極限是 decode 速度。BF16 恰好相反：放寬 TPOT 毫無幫助，收緊 TTFT 才降一格——11.4 rps 時 TPOT p95 只有 25–27 ms，TTFT p95 卻 1.15–2.80 s。max-num-seqs 40 與薄 KV 讓 BF16 收不下足夠的並行請求，請求在佇列裡等，第一個 token 才是誤過 SLO 的那個。一張 24 GB 卡上，量化把瓶頸從「收得下多少」移到「解得多快」。
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

## 量測期間發現並修正的 harness 缺陷

1. **就緒探測競態（遺失整個 AWQ cell 一次）**：`curl … | grep -q 200` 在 `set -o pipefail` 下，`grep -q` 一比對到就結束、curl 收到 SIGPIPE（141），pipefail 讓成功的探測變成失敗。AWQ 伺服器回了 200 後一秒被自己的 `SERVER_NOT_READY` 分支關掉。改用命令替換、重用迴圈結果、curl 5 s 上限、輪詢 180 次（commit `dc3a755`）。以最小重現確認機制。
2. **鏈日誌遺失**：Windows 端 `| tr | grep | tee` 區塊緩衝吃掉所有輸出，AWQ 的失敗只能從 manifest 與 server log 重建。改在 WSL 內寫 ext4 日誌（`run-logged.sh`）並以 `watch-night.sh` 串流事件。
3. **quiet-GPU 單次讀數跳過整個 cell**：對照組的閘門在 BF16 伺服器關閉 20 s 後讀到 utilization 11%（門檻 10%，取樣 15 → 4 遞減），2.5 小時的 cell 被跳過。改為重試 5 次、間隔 30 s，門檻不變（commit `4bb528f`）。
4. **生成文字不可提交**：`vllm bench serve --save-detailed` 的 `generated_texts` 存了模型對隨機 prompt 的自由續寫，其中重現了訓練資料片段（他人的 build 路徑與網站路徑）。secrets 審核抓不到（它查 IP、金鑰、email），是 `grep -r "/home/"` 抓到的。交叉驗證改提交摘要 JSON，逐請求陣列留在 repo 外並記錄原檔 sha256——與 inference-perf 逐請求 JSON 的既有政策一致。
5. **TMMLU+ 評分一錯全失**：`asyncio.gather` 未設 `return_exceptions`，19,680 題中任何一題的連線錯誤會取消全部。改為單題錯誤轉成記錄並計數（全夜 0 錯誤，但保護了四次各約 4–5 分鐘的全集評分）。
6. **同一批次目錄的第二個伺服器 session 覆蓋 server log**：`batch.sh` 每次啟動伺服器都重寫該批次目錄的 `serve.log` 與 `quiet_gpu.json`，`promote-w2.sh` 再把它搬成證據裡的 `vllm.log`，所以同一個 seed 目錄裡後來的 session 會蓋掉前一個 session 的 log（`io-pressure.log` 是附加寫入，不受影響）。W2 收尾時發現補點把 AWQ／GPTQ／BF16 九個 seed 的主量測 log 蓋掉了；這九份從 git 還原，補點 session 另存為 `vllm-refine-0.80.log`／`quiet_gpu-refine-0.80.json`（ADR 0011 起 log 以 `.log.gz` 提交）。`promote-w2.sh` 新增 session tag 參數，`refine-cell.sh` 一律帶 tag。**已無法還原的三份**：FP8 open-loop seed 1 的 7 個基準 rate（2026-09-09 13:44–14:39）、GPTQ closed-loop c = 1–8（09-10 04:20–04:30）、AWQ 第一次啟動（04:11–04:18，沒有留下任何 stage）的 server log，都在提交前就被同目錄的下一個 session 蓋掉。量測本身不受影響：每個 stage 的 manifest 自帶伺服器端直方圖差分、主機狀態與 raw sha256；缺的是那幾個 session 的啟動資訊（KV cache 大小等），AWQ 的 91,088 tokens 只記在執行計畫的 Run log。隔離重跑（`w2-cell-chain.sh`）走同一條路徑，有 suspect 時也會蓋掉 seed 的 log，W3 前要改成每個 session 分檔。

## 協定偏離（誠實記錄）

1. **closed-loop 視窗長度不一**：`num_requests` 以 FP8 速率校準，較快的 cell 較早跑完，AWQ／GPTQ 低並行點的量測窗只有 38–124 s（凍結規格為 ≥ 180 s）。刻意維持各 cell 相同的請求數——每個並行度提供完全相同的工作量，才使四精度可比；代價是精度而非偏差。每列都附 `window_s`、`window_records`。
2. **KV cache 大小不可重現**：vLLM 依啟動時可用顯存決定 KV，桌面共用卡上會隨合成器波動。AWQ／GPTQ 四個 session 完全一致（98,400／98,464），BF16 的 closed-loop 與 open-loop session 差 2.7 倍（11,168 vs 29,696 tokens）。BF16 closed-loop 峰值 KV 使用 86.1%、零 preemption、等待佇列始終 0，所以 c = 40 仍有效；但這本身就是發現：BF16 在 24 GB 桌面卡上的服務容量跨重啟不可重現。
3. **每個 cell 橫跨四個伺服器 session**（1 closed-loop + 3 open-loop seed），這是設計而非缺陷。GPTQ 的 closed-loop 另多一個 session 邊界（c = 8 → 16，就緒 bug 中斷所致），KV 差異在 c ≤ 8 不會卡到（8 × 240 = 1,920 tokens），不重跑。
4. **膝點補點**：AWQ、GPTQ、BF16 的膝點都落在凍結網格 0.75 → 1.0 × r_sat 的空洞（FP8 的膝點已由 ADR 0008 的 0.55–0.70 解析到約 8%），故三者各補 0.80／0.85／0.90 × r_sat、三個 seed。見下節。

## 膝點補點結果（2026-09-10 21:40 – 09-11 01:00）

凍結網格 {0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0} × r_sat 在 0.75 → 1.0 之間是 33% 的空洞，AWQ、GPTQ、BF16 的膝點都落在裡面（FP8 的膝點已由 ADR 0008 的 0.55–0.70 解析到 8%）。三者各補 0.80／0.85／0.90 × r_sat、三個 seed（`scripts/wsl/refine-cell.sh`），規則不變。

| cell | 0.75 × | 0.80 × | 0.85 × | 0.90 × | 1.0 × | r_SLO 補點前 → 後 |
|---|---|---|---|---|---|---|
| AWQ | 22.61：0.974 | 24.12：**0.842**（s1 1.000，s2 0.873，s3 0.842） | 25.63：0.000 | 27.13：0.000 | 30.15：0.000 | 22.61 → **22.61** |
| GPTQ | 22.70：0.998 | 24.21：**0.813**（s1 0.813，s2 1.000，s3 1.000） | 25.72：0.000 | 27.23：0.000 | 30.26：0.000 | 22.70 → **22.70** |
| BF16 | 8.55：1.000 | 9.12：0.996 | 9.69：0.998 | **10.26：0.972** | 11.40：0.746 | 8.55 → **10.26** |

（每格為三個 seed 中的最小 attainment。）

- **AWQ 與 GPTQ：補點確認了原值。** 0.80 × 那一格已經有 seed 失敗，r_SLO 不變，但夾擊區間從 33% 收斂到 7%。4-bit 的膝點是懸崖：再往上 6%（AWQ 25.63、GPTQ 25.72 rps），六個 seed-run 的 attainment 全部跌到 0.26 以下，其中四個是 0。GPTQ 在 24.21 rps 其實有兩個 seed 乾淨通過（1.000），只有 seed 1 的 0.813 拉低了最小值；若規則改用中位數 seed，GPTQ 會是 24.21。依預註冊的保守規則維持 22.70，並把這個敏感性記在這裡。
- **BF16：補點把 r_SLO 抬高了 20%。** 三個補點全數通過，10.26 rps 最差的 seed 仍有 0.972；最差 seed 的 TTFT p95 隨佇列平滑上升（0.10 → 0.16 → 0.39 → 0.82 s），直到 11.4 rps 越過 1 s。粗網格下的 8.55 比補點後低 17%，會把 FP8 對 BF16 的優勢從誠實的 2.55 倍誇大成 3.1 倍。這就是補點存在的理由：四個 cell 的 r_SLO 要以相近的解析度量出來才能放在同一張表裡比較。
- 補點新增的 27 個 stage 全部乾淨（無 suspect），證據併入各 cell 的 `open-loop/seed-{1,2,3}/`，表由 `make reproduce` 重建。

## 桌面共用的顯存監控

Windows 端取樣器全程每 32 s 記一次 WDDM 顯存計數器（`evidence/raw/w2/win-vram-2026-09-10.log`，04:12–01:00，2,315 筆）。total committed 最高 24,187 MB（21:31，對照組 78.9 rps 過載點），始終低於實體 24,564 MiB，所以 ADR 0007 的 VidMm 分頁機制全程沒有觸發；`w_per_util_point`、probe TPOT、committed 三條可疑規則也沒有標出任何 stage，隔離與重跑機制全程未被動用。

## Claim ceiling

所有數字只對：這組旗標、`--gpu-memory-utilization 0.82`、WSL2、與 Windows 桌面共用的單張 RTX 4090、Qwen3-8B、108 → 132 tokens 的合成負載成立。不外推到原生 Linux、其他 GPU、其他模型大小、其他 prompt 長度分布。GPTQ 數字只代表 `JunHowie/Qwen3-8B-GPTQ-Int4`。能耗只算 GPU 板卡。r_sat 對 FP8 與 BF16 是下界（被 `--max-num-seqs` 封頂），對 AWQ／GPTQ 是實測平台。FP8 全程使用 vLLM 預設的 W8A8 block FP8 kernel config（4090 沒有 tuned config，server log 每次啟動都警告）；ADR 0002 留給 W2 的「是否為 4090 產生 config」決定為不產生，FP8 數字就是未調校 kernel 的數字。

## 尚未量測

- W3：admission 三策略（原生佇列 vs 硬上限 429 vs 有界佇列）在 burst trace 下的 attainment／goodput／拒絕率。
- W4：speculative decoding 兩個 cell（n-gram、EAGLE-3）。
- 成本表：`config/cost.yaml` 仍待使用者填 4090 攤提與電價，$／百萬 token 尚未計算（Wh 已有）。
