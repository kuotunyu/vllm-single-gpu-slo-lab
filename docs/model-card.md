# Model card（量測卡，2026-09-12；W2、W3、W4 三軸的結果都已填入）

## 這份 card 描述什麼

不是新模型：是 Qwen3-8B（BF16／FP8／AWQ／GPTQ-Int4）與 Qwen3-4B（none／n-gram／EAGLE-3）在**一張與 Windows 桌面共用的 RTX 4090** 上、SLO 約束下的容量、能耗、品質與流量控制量測。所有數字都能由 repo 內的證據用 `make reproduce` 重建；每個數字的出處與 n 在 `analysis/claims_audit.md`。

## 模型、權重與授權

| cell | 權重 | 授權 | 備註 |
|---|---|---|---|
| BF16 | `Qwen/Qwen3-8B` | Apache-2.0 | 15.3 GiB；0.82 預算下 KV 只剩 11k–30k tokens，`--max-num-seqs` 40 |
| FP8 | `Qwen/Qwen3-8B-FP8` | Apache-2.0 | 8.8 GiB；Ada（compute 8.9）原生 W8A8；vLLM 預設 block FP8 kernel config（4090 沒有 tuned config，未調校） |
| AWQ | `Qwen/Qwen3-8B-AWQ` | Apache-2.0 | 5.7 GiB |
| GPTQ-Int4 | `JunHowie/Qwen3-8B-GPTQ-Int4` | Apache-2.0（第三方量化，gptqmodel 4.0.0） | 品質數字只屬於這個 checkpoint |
| 4B | `Qwen/Qwen3-4B` | Apache-2.0 | 只作 speculative decoding 軸的基準 |
| 4B + EAGLE-3 | `AngelSlim/Qwen3-4B_eagle3`（revision `fd331e59`） | repo 內 `License_AngelSlim_model_and_dataset.txt`：Apache-2.0（Tencent） | 0.4 GiB draft head；`num_speculative_tokens` 3 |

各次啟動的 vLLM 版本、旗標全文、環境變數與原始檔 sha256 都在每段的 `manifest.json`。授權盤點在 `docs/licences.md`。

## 量測環境

- RTX 4090 24 GB，Windows 11 宿主，量測在 WSL2 Ubuntu（kernel 6.6）；vLLM 0.28.0、torch 2.13.0+cu130；inference-perf 0.6.1 裸機模式（4 workers）。
- 引擎旗標（所有 cell 相同，只有 `--speculative-config` 在 W4 不同）：`--max-model-len 4096 --gpu-memory-utilization 0.82 --max-num-batched-tokens 2048`，`--max-num-seqs` 256（BF16 40）；`VLLM_WSL2_ENABLE_PIN_MEMORY=1`、`VLLM_USE_FLASHINFER_SAMPLER=0`（torch 原生 sampler）、`HF_HUB_OFFLINE=1`。
- 0.82 而非 0.90：桌面程式與 vLLM 合計超過實體 VRAM 時 WDDM 會分頁，吞吐減半（ADR 0007）；每段 manifest 記 Windows 端 committed VRAM，超過實體即作廢重跑。
- 功耗：NVML 1 s 取樣，只含 GPU 板卡。

## Workload

- Prompt 108 tokens、輸出 132 tokens（`ignore_eos`），Qwen3 thinking 關閉，completion API，串流。
- W2、W4：inference-perf `synthetic`（Shakespeare 語料切片，每筆隨機起點，prefix cache 命中率 0）；W3：seeded trace replay 的隨機 token prompt。
- 負載：closed-loop concurrency 網格（r_sat、C）；open-loop Poisson 掃描（每點 5 min、丟棄前 60 s、3 seeds）；W3 為 25 分鐘單峰突發（0.5 → 1.5 → 0.5 × r_sat）。
- warm-up：每個伺服器 session 先 100 筆 sequential，段間 20 筆；同時量單流 TPOT 作租戶探針。

## SLO 與指標

- SLO：TTFT p95 ≤ 1 s 且 TPOT p95 ≤ 50 ms（`analysis/preregistration.md` 凍結）。TPOT = (e2e − TTFT) ÷ (output tokens − 1)，token 數取伺服器回報的 `completion_tokens`。
- Attainment 分母為 offered：429、timeout、5xx 都算未達；r_SLO = 所有 seed attainment ≥ 95 % 的最高 offered rate（自最低 rate 連續向上）。
- 能耗：r_SLO 點的 GPU 板卡平均功耗換算成 Wh／百萬 output token；4090 為自有硬體，不算 $／百萬 token（ADR 0010）。

## 結果摘要（細節與 CI 在各 ADR 與 `analysis/tables/`）

| 軸 | 結果 | 出處 |
|---|---|---|
| 精度（W2） | r_SLO：BF16 10.26、FP8 26.2、AWQ 22.61、GPTQ-Int4 22.70 req/s；能耗 @ r_SLO：74.6／31.2／36.2／36.3 Wh／百萬 token；TMMLU+ 全集 0.5911／0.5909／0.5797／0.5703，FP8 對 BF16 配對差 −0.02 pts（p = 0.945） | ADR 0009 |
| Admission（W3） | 1.5 倍、5 分鐘突發下整段 attainment：FP8 原生排隊 0.19、hard cap 0.57、有界佇列 0.59；BF16 0.47、0.87、0.86；限流 10 s 內恢復，原生排隊 3.5–14 分鐘；補點：FP8 hard cap 改 C = 192 後突發段 TPOT p95 45 ms、突發段 attainment 0.49（C = 256 為 0.01）、整段 0.78（非預註冊） | ADR 0013、0018 |
| Speculative decoding（W4） | 接受率：8B n-gram 0.52、4B EAGLE-3 0.27、4B n-gram 0.53；單流 TPOT 1.72×／1.45×／1.12×，c = 128 起吞吐 0.85×／0.66×／0.73×；同 rate 的 attainment 與粗網格 r_SLO 和 none 相同，TPOT p95 升 1.6 到 10 ms（EAGLE-3 在 12 rps 以下降 2 ms 為唯一例外）；8B n-gram 在 256 並行時超出 0.82 預算而分頁 | ADR 0015、0016 |

## What this does not show（規格 §2.2，發佈時逐條保留）

1. 不宣稱多 replica、擴縮、生產可靠度。
2. 不宣稱跨 GPU class 推論：只講實際量過的 RTX 4090（A1 雲端對照已取消，ADR 0014）。
3. 不宣稱任何 $／百萬 token，也不做「比 API 便宜」的比較（ADR 0010）。
4. 不宣稱 EAGLE-3 在 Qwen3-8B 上的效果：EAGLE-3 數字只屬於 Qwen3-4B；8B 只有 n-gram。
5. 不宣稱 TMMLU+ 分數可與他人 leaderboard 比較：只報自跑數字與精度間配對差。
6. 不宣稱能耗為整機功耗：只量 GPU 板卡，Wh 數字是下限。
7. 不宣稱 WSL2 數字等於裸機 Linux：pinned memory 開關與 torch 原生 sampler 下量測，WSL2 開銷未分離。
8. 不宣稱 thinking 模式下的延遲。
9. 不外推到其他 prompt 長度分布或語料：n-gram 的接受率取決於文本重複程度，本 lab 的 Shakespeare 續寫是低重複自然文字（ADR 0015）。
