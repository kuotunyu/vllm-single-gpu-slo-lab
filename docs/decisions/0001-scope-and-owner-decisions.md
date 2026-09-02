# ADR 0001 — 範圍與 owner 決策

- 日期：2026-09-03
- 狀態：已定案
- 依據：設計規格 `2026-09-03-vllm-single-gpu-slo-lab-design.md` §11 與文末「決策紀錄」；使用者授權「都照你的規劃進行」，§11 四項依建議定案

## 四項決策

| # | 決策 | 定案 |
|---|---|---|
| 1 | Admission 與 decoding 軸全因子（≈ 30 h trace）或 §3.1 部分因子（≈ 15 h） | **部分因子**：精度軸全做（policy (i)、無加速）；admission 軸在預設精度（預期 FP8）與 BF16 上做；decoding 軸在預設精度 8B 做 none vs n-gram，另在 Qwen3-4B 做 none／n-gram／EAGLE-3。W5 有餘裕再補 AWQ 的 admission 三策略 |
| 2 | A1 的 GPU class | **RunPod L4 Secure**；上限 USD 10 或 10 GPU 小時先到者停；L4 缺貨才退 L40S；不用 Community tier。開帳號、儲值為使用者動作；開 pod 前再問一次 |
| 3 | A3 是否仍用 Colab 跑全量 TMMLU+ | **W2 後決定**：以 4090 切片 throughput 推算四精度全量所需時數；若一個下午跑得完，A3 取消、Colab notebook 只留作第三方可重跑的證據。FP8 全量無論如何在 4090 跑（Colab A100 compute 8.0 會退回 weight-only） |
| 4 | Repo 名 | **保留 `vllm-single-gpu-slo-lab`**：「single-gpu」把 claim boundary 寫進名字，A1 後（單 replica）仍成立 |

任何付費動作仍逐筆先問；本專案總上限 USD 15。

## W0 骨架時的實作選擇（非 owner 決策；記錄以供追溯）

1. **LICENSE**：規格 §9 樹狀圖寫 MIT，骨架採 **Apache-2.0**（與 owner 其他公開專案一致，holder `kuotunyu`）；模型權重 Apache-2.0 與 TMMLU+ 授權另記 `docs/licences.md`。
2. **佈局**：規格 §9 的 `harness/` 與 `analysis/*.py` 合併為單一 package `src/slo_lab/`，以便 uv 打包與測試：`harness/admission_shim/` → `slo_lab/admission/`；`harness/power_sampler.py` → `slo_lab/power/sampler.py`；`harness/quiet_gpu.py` → `slo_lab/quiet_gpu.py`；`analysis/ci.py` → `slo_lab/stats.py`；`analysis/cost.py` → `slo_lab/cost.py`；`analysis/aggregate.py` 的 attainment／r_SLO 部分 → `slo_lab/slo.py`。`harness/run.py`、`metrics_scraper.py`、`crosscheck/`、`datasets/`、`analysis/plots.py`、`eval/tmmluplus/` 尚未存在。
3. **HTTP stack**：admission shim 用 aiohttp（理由見 `pyproject.toml` 註解）。
4. **成本 headline 用 gross board power**；idle-subtracted（net）值作 sensitivity。兩者都只涵蓋 GPU 板卡（claim ceiling 6）。
5. **r_SLO 搜尋**自最低 rate 連續向上，遇到任一 seed 未達 95% 即停；非單調的「高 rate 回升」不計（保守）。缺 seed 的 rate 視為未達。
6. **Per-request 記錄格式**：本 package 的 canonical `RequestRecord` JSONL；inference-perf raw JSON → records 的 adapter 是 W1 工作。
7. **`config/engine/common.yaml`**：規格樹狀圖沒有；加入以免五個 cell 各抄一份共同旗標。
8. **功耗來源**：規格 §6 寫 `nvidia-smi --query-gpu`，此 driver 不支援，改 NVML 直讀（同一組計數器）。
