# 授權盤點

| 項目 | 授權 | 狀態 |
|---|---|---|
| 本 repo 程式碼與文件 | Apache-2.0（kuotunyu, 2026） | 定案 |
| Qwen3-8B、Qwen3-8B-FP8、Qwen3-8B-AWQ、Qwen3-4B 權重 | Apache-2.0（上游） | W1 記 revision |
| GPTQ-Int4 權重（`JunHowie/Qwen3-8B-GPTQ-Int4`） | Apache-2.0（HF 卡片 `license: apache-2.0`，base model `Qwen/Qwen3-8B`；第三方量化，品質數字只屬於這個 checkpoint） | 已核對（ADR 0003） |
| AngelSlim EAGLE-3 head（`AngelSlim/Qwen3-4B_eagle3`，revision `fd331e59`） | HF 卡片的 metadata 沒有 `license` 欄位，但 repo 內附 `License_AngelSlim_model_and_dataset.txt`（2026-09-12 核對）：AngelSlim 本體與 Qwen 系列衍生模型為 Apache-2.0（Tencent），允許商用與再散布，需保留版權與授權聲明；訓練資料 sharegpt_gpt4 為 CC BY 4.0（不隨本 repo 散布）；沒有對輸出或 benchmark 的限制 | 已核對；本 repo 只提交量測數字與 head 的 revision，不提交權重 |
| TMMLU+（`ikala/tmmluplus`） | MIT（ADR 0003）；題目原文隨切片與全集提交於 `eval/tmmluplus/` | 已核對 |
| inference-perf、vLLM | Apache-2.0（上游工具，不 vendor） | — |

權重、資料集、工具皆不被本 repo 的 LICENSE 重新授權。
