# TMMLU+ 品質軌（W1／W2；目前為空）

規格 §5：

- 切片（4090，W2）：3 個不重疊的 200 題切片（固定 seed、依科目比例分層），greedy、thinking 關閉；Wilson 95% CI；與 BF16 的差用同題配對 bootstrap。
- 全量（W4）：FP8（與 AWQ）在 4090 跑；BF16 與 GPTQ-Int4 視 ADR 0001 #3 決定是否用 Colab。
- 授權（W1 清單第 4 項）：讀 dataset card 後決定題目原文是否提交；否則只提交 ID、逐題輸出、分數。

尚未存在：分層抽樣器、vLLM 離線 runner、Colab notebook。
