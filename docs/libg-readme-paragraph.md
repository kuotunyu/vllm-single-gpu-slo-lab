# 給 `local-inference-bench-gateway`（LIBG）README 的分工段落草稿

規格 §8 W6 的交付物之一：兩邊 README 各放一段互指分工。本 repo 的那一段在 `README.md` 的「與 LIBG 的分工」；下面是給 LIBG 那一側的草稿，依規格「不推送，待使用者」，由 LIBG 的作者決定是否放進去（建議放在「系統邊界」或「設計決策與誠實範圍」一節的結尾）。

---

## 與 `vllm-single-gpu-slo-lab` 的分工

本專案回答「哪個本機 engine、gateway 該怎麼擋」：以共同 workload 比較 llama.cpp／Ollama／LM Studio 三種桌面級 engine，並提供 OpenAI-compatible gateway（alias routing、ordered failover、容量滿時回 HTTP 429、不建無上限 in-memory queue、SQLite telemetry）。它不做 vLLM、不做 open-loop 流量、不做量化對照、不畫 SLO 曲線。

那些問題由姊妹專案 [`vllm-single-gpu-slo-lab`](https://github.com/kuotunyu/vllm-single-gpu-slo-lab) 回答：只用一個 engine（vLLM），在 SLO 約束下（TTFT p95 ≤ 1 s 且 TPOT p95 ≤ 50 ms）量一張 RTX 4090 能接多少 open-loop Poisson 流量、每百萬 output token 的實測能耗、四種精度與三種 admission 策略（原生排隊、硬上限 429、有界佇列）在突發流量下的取捨，以及 speculative decoding 對容量的影響；每個數字都由提交的證據以 `make reproduce` 重建。那邊的 429 語意向本專案借用，但獨立實作、不 import 本專案。

---

字數約 330 字；與本 repo README 的分工段落互相對應，沒有名次或比較性的宣稱。
