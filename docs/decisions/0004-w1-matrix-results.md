# ADR 0004 — W1 載入矩陣結果：五個 cell 皆可在 4090／WSL2 上服務

- 日期：2026-09-09 凌晨
- 狀態：已採納（W1 清單「四種精度載入」「EAGLE-3 4B head 載入」完成）
- 原始紀錄：`evidence/raw/w1/matrix-load-check.jsonl`（每 cell 一列：載入時間、KV cache 容量、一次 completion、NVML 功耗與記憶體）
- 共同旗標：`--max-model-len 4096 --gpu-memory-utilization 0.90 --max-num-seqs 64`，attention backend 皆為 `FLASH_ATTN`，ADR 0002 的兩個 WSL2 環境開關

| cell | 模型 | 權重載入 | 載入記憶體 | KV cache 容量（tokens） | 一次請求 | 備註 |
|---|---|---|---|---|---|---|
| fp8 | Qwen/Qwen3-8B-FP8 | 64 s（暖）／— | 8.8 GiB | 83,024 | 正常 | 9/8 深夜先跑；FP8 block kernel 無 4090 tuned config |
| awq | Qwen/Qwen3-8B-AWQ | 137.5 s（冷） | 5.71 GiB | 106,000 | 正常 | |
| gptq | JunHowie/Qwen3-8B-GPTQ-Int4 | 98.2 s（冷） | 5.68 GiB | 106,064 | 正常 | 第三方量化（ADR 0003） |
| bf16 | Qwen/Qwen3-8B | 155.0 s（冷） | 15.27 GiB | **25,056** | 正常 | 24 GB 卡上 KV 只剩 4 個 4,096-token 請求的量；W2 的 BF16 cell 必須用較小 `max-num-seqs` 或接受較早的 capacity 拒絕，並在 claim 中說明 |
| q4b | Qwen/Qwen3-4B | 106.8 s（冷） | 7.56 GiB | 90,480 | 正常（但把 SLO 解成 Student Learning Outcome） | 只作 EAGLE-3 軸的基準 |
| q4b_eagle3 | Qwen/Qwen3-4B + AngelSlim/Qwen3-4B_eagle3（`num_speculative_tokens: 3`） | 4.7 s（權重已在 cache） | 7.97 GiB | 83,952 | 正常 | log 確認 `Resolved architecture: Eagle3LlamaForCausalLM`、draft max len 由 40,960 覆寫為 4,096 |

## 讀法與限制

- 「冷／暖」指 page cache 狀態；冷載入受當時磁碟競爭影響（runbook 記錄 io pressure 50–60%），**不是**冷啟動記帳的證據，只證明可載入。
- KV cache 容量是 vLLM 在 0.90 記憶體占比下的自報值；BF16 的 25,056 tokens 是 W2 設計要正面處理的約束，不是缺陷。
- 一次請求後的 NVML 功耗（AWQ 88 W、GPTQ 86 W、BF16 190 W、4B 128 W、4B+E3 136 W）是「剛回覆完」的瞬時值，不是閒置或穩態功耗；W2 以 1 s 取樣整段 run 才算數。
- Qwen3-4B 對「SLO」的誤解是品質訊號，屬 TMMLU+／品質軌的範圍，與載入檢查無關。

## 對 W2 的決定

1. BF16 cell 的 `max-num-seqs` 另訂（提案 16），其餘 cell 沿用 64；差異寫進 manifest。
2. EAGLE-3 軸只在 Qwen3-4B 上跑（claim ceiling 4 不變）。
3. 所有 cell 先做一次「暖載入」再量測，載入時間以第二次為準。
