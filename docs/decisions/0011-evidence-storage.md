# ADR 0011 — 證據存放：不用 Git LFS，逐筆紀錄與 server log 以 gzip 提交；audit 掃描壓縮檔與大檔

- 日期：2026-09-11
- 狀態：已採納（使用者交由判斷：「交給你來判斷，照你建議的方式來進行」）
- 實作：`scripts/compress_evidence.py`、`slo_lab.slo.open_evidence_text`／`evidence_path`、`slo_lab.redact`；測試 `tests/test_compress_evidence.py`、`tests/test_redact.py`

## 觸發

W2 結束時 `evidence/` 在磁碟上 430 MB：逐筆紀錄 `records.jsonl` 244 MB（243 個 stage）、vLLM server log 133 MB（36 份），其餘全部加起來約 50 MB。W3、W4 還會再加約同等份量，clone 後的工作目錄會接近 1 GB。GitHub 單檔上限 100 MB，本 repo 最大的檔案 11 MB，沒有單檔問題。

## 決定

1. **不用 Git LFS。** LFS 檔在一般 clone 只拿到指標檔，CI 要另外拉取並消耗配額，fork 與下載 zip 也不帶內容；最大的檔案才 11 MB，LFS 解決的是不存在的問題。
2. **`records.jsonl` 與 server log（`vllm*.log`）以 gzip 提交**，存成 `records.jsonl.gz`、`vllm.log.gz`。manifest、功耗、metrics、表格、TMMLU+ 輸出等其餘檔案維持純文字，在 GitHub 上可直接閱讀。
   - 讀取端透明：`open_evidence_text`／`evidence_path` 以邏輯檔名（`records.jsonl`）開啟，只有 `.gz` 時自動解壓；`reproduce-lite`、`batch_analysis`、`served_rps` 都走這條路徑。
   - 壓縮是決定性的（level 9、mtime 0、不存檔名），同樣的輸入永遠得到同樣的位元組，重跑不產生差異。`compress_evidence.py` 逐檔確認解壓後與原檔逐位元相同才刪原檔。
   - 每個 promote 腳本在搬完證據後呼叫它，W3 之後的證據一進 repo 就是壓縮的；harness 在 WSL 端仍寫純文字，量測中途的分析不受影響。
3. **驗證**：轉換前後 `reproduce-lite` 的輸出逐行相同，`analysis/tables/` 每張表的 sha256 與轉換前相同。279 個檔案 378.9 MiB 壓成 48.1 MiB，`evidence/` 由 430 MB 降到 99 MB。
4. **歷史**：未壓縮的舊版本仍在本機 git 歷史裡。發佈前（W6）與作者、mailmap 檢查一起決定是否用 `git filter-repo` 改寫歷史，讓公開 repo 只含壓縮版本。

## 同批修正的兩個缺口

- **secrets audit 靜默略過大於 2 MB 的檔案。** `scan_tree` 原本的 `max_bytes=2_000_000` 讓幾乎所有 server log 與較大的 records 從未被掃描，`audit-secrets: clean` 對它們並不成立。它們在 promote 時經過 `redact.py redact`，實際上有去敏，但 audit 沒有驗證到。上限改為 64 MB，並依 magic bytes 解壓掃描 gzip 檔（解壓上限 512 MB）；名為 `.gz` 的純文字檔仍當文字掃描，不會被略過。第一次完整掃描找到 13 筆，全部是 `eval/tmmluplus/full.jsonl` 網路科目考題裡的範例 IP（子網路遮罩、192.168.x.x）。對 `eval/tmmluplus/` 只放行 ipv4 規則，金鑰、token、email 規則照常適用。所有 server log 與 records 掃描乾淨，完整掃描約 1 分鐘。
- **W1 的 inference-perf smoke 提交了 prompt 與模型回應原文。** `evidence/raw/w1/inference-perf-smoke/per_request_lifecycle_metrics.json` 的 20 筆 `request`／`response` 欄位含 prompt（Shakespeare 文本）與模型續寫。檢查過沒有路徑、email 或 URL，但違反「生成文字不提交」的原則（ADR 0009 缺陷 4）。兩個欄位改為字數與 sha256；adapter 不讀這兩個欄位，測試照過。
