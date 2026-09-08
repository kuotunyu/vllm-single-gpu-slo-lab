# ADR 0003 — W1 模型與資料來源定案

- 日期：2026-09-08
- 狀態：已採納
- 關聯：規格 §4（W1 驗證清單「GPTQ-Int4 repo id／授權」「EAGLE-3 4B head repo id」「TMMLU+ 授權」）

## GPTQ-Int4：`JunHowie/Qwen3-8B-GPTQ-Int4`

Qwen 官方只釋出 BF16、FP8、AWQ 三種 Qwen3-8B；GPTQ-Int4 必須用第三方。候選比較（HF API，2026-09-08）：

| 候選 | 授權欄位 | base_model 欄位 | 量化中繼資料 | 下載數 |
|---|---|---|---|---|
| `JunHowie/Qwen3-8B-GPTQ-Int4` | apache-2.0 | Qwen/Qwen3-8B | gptqmodel 4.0.0、bits 4、group 128、desc_act false、sym、damp 0.05、true_sequential | 1,646 |
| `AngelSlim/Qwen3-8B_int4_gptq` | 未填 | 未填 | bits 4、group 128、desc_act true、sym；無 quantizer 版本 | 31 |

選 JunHowie：授權明示、base model 明示、量化器版本與參數完整可寫進 model card。代價：desc_act=false 與 AngelSlim 不同，屬第三方量化決策，**不代表 Qwen 官方 GPTQ 品質**——claim ceiling 加一條「GPTQ-Int4 為第三方量化（gptqmodel 4.0.0），品質數字只屬於這個 checkpoint」。AngelSlim 仍是 EAGLE-3 head 的來源（`AngelSlim/Qwen3-4B_eagle3`，0.4 GiB，卡片未填授權欄位；W1 載入測試後決定是否需另行確認授權）。

## TMMLU+：`ikala/tmmluplus`，MIT

HF dataset card `license: mit`，未 gated。MIT 允許重製 200 題切片並隨 repo 發布（附版權聲明）；規格 §5 的「切片可否重製」問題解除。leaderboard 仍不引用（memo §4：12 個月未更新）。

## 官方權重（皆 Apache-2.0，未 gated）

`Qwen/Qwen3-8B` 15.3 GiB、`Qwen/Qwen3-8B-AWQ` 5.7 GiB、`Qwen/Qwen3-8B-FP8` 8.8 GiB、`Qwen/Qwen3-4B` 7.5 GiB。全部快取於 WSL2 `~/.cache/huggingface/hub`，量測時 `HF_HUB_OFFLINE=1`。
