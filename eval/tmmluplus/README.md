# TMMLU+ 品質軌（W1 切片、W2 評分、W4 全集）

規格 §5：

- 切片（4090，W2）：3 個不重疊的 200 題切片（`slice-{1,2,3}.jsonl`，固定 seed 20260908、依科目比例分層，SHA-256 在 `slices.json`），greedy、thinking 關閉；Wilson 95% CI；與 BF16 的差用同題配對 bootstrap。FP8 結果：366/600 = 0.610（`evidence/raw/w2/fp8/tmmluplus/`，ADR 0007）。
- 全集（W4）：`full.jsonl` 為同一 snapshot（`45e7c9b0…`）的完整 test split，19,680 題、67 科，依 (subject, index) 排序，SHA-256 在 `full.json`；由 `scripts/tmmluplus_full.py` 產生，`scripts/wsl/w2-cell-chain.sh` 在每個 cell 的切片之後直接跑全集（約 11 分鐘／精度）。
- 授權：ikala/tmmluplus 為 MIT（ADR 0003），題目原文隨切片與全集提交。
- 評分：`scripts/tmmluplus_eval.py`（chat endpoint、`/no_think`、temperature 0、max_tokens 8、逐題輸出與 `items_sha256_scored`）。
