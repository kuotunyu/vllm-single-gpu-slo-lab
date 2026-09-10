# 授權盤點

| 項目 | 授權 | 狀態 |
|---|---|---|
| 本 repo 程式碼與文件 | Apache-2.0（kuotunyu, 2026） | 定案 |
| Qwen3-8B、Qwen3-8B-FP8、Qwen3-8B-AWQ、Qwen3-4B 權重 | Apache-2.0（上游） | W1 記 revision |
| GPTQ-Int4 權重 | 未定：W1 核對 repo id 與授權；無 Apache-2.0 官方權重則降為 optional | 待核對 |
| AngelSlim EAGLE-3 head（Qwen3-4B） | 未定：W1 核對 | 待核對 |
| TMMLU+（`ikala/tmmluplus`） | MIT（ADR 0003）；題目原文隨切片與全集提交於 `eval/tmmluplus/` | 已核對 |
| inference-perf、vLLM | Apache-2.0（上游工具，不 vendor） | — |

權重、資料集、工具皆不被本 repo 的 LICENSE 重新授權。
