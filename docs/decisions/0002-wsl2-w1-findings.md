# ADR 0002 — W1 驗證結果：WSL2 上的 vLLM 0.28 可用，但有三個必須記錄的條件

- 日期：2026-09-08
- 狀態：已採納
- 關聯：規格 §4（W1 驗證清單）、§10（風險：WSL2 vLLM 支援）；`docs/runbook-wsl2.md`

## 結論

Qwen3-8B-FP8 可在 WSL2 + RTX 4090 上以 vLLM 0.28.0 正常服務，但只有在下列條件下：

1. `VLLM_WSL2_ENABLE_PIN_MEMORY=1`。vLLM 在 WSL2 預設關閉 pinned memory；0.28 的 GPU runner 需要 UVA host-mapped buffer，否則 engine 初始化直接失敗（`UVA is not available`）。這是 vLLM 官方環境開關；同一問題在 0.27.1 也重現，降版無效。
2. `VLLM_USE_FLASHINFER_SAMPLER=0`。FlashInfer sampler 需要 JIT（`nvcc` + `ninja`），WSL2 主機沒有；關閉後用 torch 原生 sampler。
3. `nvidia-smi --query-gpu` 在此 driver 不可用，功耗與記憶體一律由 NVML 讀取；NVML 在 WSL2 列不出 compute process。

## 對證據與 claim 的影響

- 所有 4090 數字都是「WSL2 + pinned memory 開關」下的數字。vLLM 自述此設定有小幅效能退化；本 lab 不主張 4090 原生 Linux 的效能，claim ceiling 加一條。A1（RunPod L4，原生 Linux）因此不只是第二個 GPU class，也是「非 WSL2」的對照。
- Sampler 差異（torch 原生 vs FlashInfer）會影響 decode 吞吐；所有 cell 一致使用 torch 原生 sampler，並在 model card 註明。
- `quiet-gpu` 在此主機只靠記憶體門檻（3,072 MiB，依實測閒置 2,609 MiB 校準）與 utilization 門檻（5%）；快照的 `process_list_trustworthy=false` 讓讀者知道 process 準則沒有生效。

## 其他 W1 觀察

- FP8 block kernel 沒有 4090 的 tuned config（vLLM 警告）。W2 決定是否為 4090 產生 config；證據需標明用的是哪一種。
- 權重載入 64.32 s、模型載入合計 74.7 s / 8.8 GiB；KV cache 83,024 tokens（4,096 tokens/request 時 20.27x 並發）；CUDA graph 3 s。
- Server 閒置待命 143 W / 23,436 MiB（`--gpu-memory-utilization 0.90`），主機閒置 19.7 W / 2,609 MiB。
