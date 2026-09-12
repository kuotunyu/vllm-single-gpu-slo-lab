# ADR 0017 — W2 以伺服器的 token 數重解析：243 段全部重算，r_SLO 與表都不變

- 日期：2026-09-12（W4 量測結束後，16:57–17:40）
- 狀態：已採納；W5 第 1 項結案
- 起因：ADR 0012 附錄第 2 點。W2 的 `records.jsonl` 是用 inference-perf 重新 tokenize 回傳文字算出的 output token 數解析的；W3 smoke 發現這個計數在隨機 token prompt 下有 25 % 的請求算短、文本 prompt 下 1.45 %，會高估這些請求的 TPOT。adapter 自 commit `18789bd` 起改用伺服器回報的 `completion_tokens`，W3、W4 的紀錄都是新計數，W2 的不是。
- 證據：`evidence/raw/w2/reparse-compare-2026-09-12.json`（逐段舊／新 attainment、TPOT p95、計數不同的請求數，以及每個 cell 的 r_SLO 舊／新）；工具 `slo_lab.reparse`、`slo-lab reparse-records`／`reparse-compare`、`scripts/wsl/reparse-w2.sh`；重解析後的紀錄留在量測主機 `~/vllm-slo-lab/reparse-w2/`（247 MB），不提交

## 做法

`reparse-w2.sh` 對 `~/vllm-slo-lab/runs-w2` 與 `runs` 下的 245 個 `per_request_lifecycle_metrics.json` 算 sha256，用每段 manifest 的 `raw_sha256` 對上原始檔（243 段全部對上，沒有缺檔），以現行 adapter 重新解析成平行目錄，再逐段與提交的 `records.jsonl.gz` 比較；open-loop 段另以凍結規則重算各 cell 的 r_SLO。

## 結果

| cell | 計數不同的請求 | TPOT p95 變化（各段最小／最大） | attainment 變化（最小／最大） | r_SLO 舊 → 新 |
|---|---|---|---|---|
| FP8 | 1.69 %（open-loop 6,468／383,211；closed-loop 1.75 %） | −0.21／0.00 ms | 0／+0.0008 | 26.2 → 26.2 |
| AWQ | 2.69 %（closed-loop 2.82 %） | −0.07／+0.01 ms | −0.0002／+0.0003 | 22.61 → 22.61 |
| GPTQ-Int4 | 2.72 %（closed-loop 2.78 %） | −0.07／0.00 ms | 0／+0.0007 | 22.70 → 22.70 |
| BF16 | 1.80 %（closed-loop 1.78 %） | −0.10／0.00 ms | 0／0 | 10.26 → 10.26 |
| FP8 mbt8192 | 1.68 % | −0.19／0.00 ms | 0／+0.0001 | 25.64 → 25.64 |

沒有任何一段的 TPOT p95 變化超過 0.5 ms，也沒有任何一段的 attainment 變化超過 0.005；五個 cell 的 r_SLO 完全不變。ADR 0012 附錄估的 1.45 % 是 smoke 一段的值，全 W2 是 1.7 到 2.8 %（4-bit cell 較高），方向一致：舊計數只會把少數請求的 TPOT 算高，重算後 p95 只降 0 到 0.2 ms。

## 決定

- **不更新 W2 的紀錄與表。** 依交接文件的規則（r_SLO 有變才改），W2 提交的 `records.jsonl.gz` 維持原樣；表、README、claims audit 的 W2 數字不動。本 ADR 與比較檔是「已檢查、無影響」的證據。
- W2 的 claim ceiling 加一句：TPOT 以 inference-perf 重新 tokenize 的計數解析，與伺服器計數的差異經重解析確認對 p95 的影響在 0.2 ms 內、對 r_SLO 無影響（`analysis/claims_audit.md` 第 1 列）。
- 量測主機的 `runs-w2`（117 GB）自此可以在使用者同意後清掉（交接文件 W6 第 4 項）；重解析的平行目錄可一併刪除。
