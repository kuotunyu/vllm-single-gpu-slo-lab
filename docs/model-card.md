# Model card（W5 填寫；W0 只有骨架）

## 這份 card 描述什麼

不是新模型：是 Qwen3-8B（BF16／FP8／AWQ／GPTQ-Int4）與 Qwen3-4B 在單張 GPU 上、SLO 約束下的容量、成本與品質量測。

## 待填欄位

- 模型與授權：Qwen3-8B／Qwen3-4B，Apache-2.0；各精度權重來源與 revision（W1 manifest）
- 實測 GPU class 清單：**尚無**（W1 後：RTX 4090／WSL2；A1 後：L4 或 L40S）
- Workload 形狀：108 in／132 out、nonce 前綴、`ignore_eos`、thinking 關閉
- SLO 與 attainment 定義：`analysis/preregistration.md`
- 結果表：`evidence/tables/quality_cost.md`（由 `make reproduce` 產生）

## What this does not show（規格 §2.2 全文，發佈時逐條保留）

1. 不宣稱多 replica、擴縮、生產可靠度。
2. 不宣稱跨 GPU class 推論——只講實際量過的 class。
3. 不宣稱絕對「比 API 便宜」——只報在實際驅動的 utilisation 下的 $/M token，並列 1/U 警語。
4. 不宣稱 EAGLE-3 在 Qwen3-8B 上的效果——EAGLE-3 數字只屬於 Qwen3-4B。
5. 不宣稱 TMMLU+ 分數可與他人 leaderboard 比較——只報自跑數字與精度間配對差。
6. 不宣稱電費為整機功耗——只量 GPU 板卡，電費項為下限。
7. 不宣稱 WSL2 數字等於裸機 Linux。
8. 不宣稱 thinking 模式下的延遲。
