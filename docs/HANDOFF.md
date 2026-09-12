# 交接文件：vllm-single-gpu-slo-lab（2026-09-11）

之後回來繼續時，先讀這份。它說明現在做到哪裡、還剩什麼、每一項要多少時間、怎麼開工，以及這個專案踩過的坑。

## 一句話現況

W0–W3 已完成：四種精度的比較（W2）與三種流量控制策略的比較（W3）都有結果、證據與 ADR。剩下 W4（加速解碼）、一項 FP8 補充量測、W5（補分析與文件）、W6（寫作與發佈前檢查）。A1 雲端對照已取消。

## 位置

| 項目 | 位置 |
|---|---|
| 專案 repo（本機 git，68+ 個 commit，無 remote） | `D:\AI-Portfolio\CC_github部隊\vllm-single-gpu-slo-lab` |
| GitHub | **沒有**。從未推送，`kuotunyu` 底下沒有這個 repo；是否公開在 W6 由使用者決定 |
| 量測環境 | WSL2 發行版 `Ubuntu-bench`，`~/vllm-slo-lab`：`.venv`（vLLM 0.28.0）、`.venv-slolab`（本 repo 的 editable install）、`.venv-loadgen`（inference-perf 0.6.1） |
| 原始量測輸出 | `~/vllm-slo-lab/runs-w2`（117 GB，235 個原始逐筆檔）、`runs-w3`（36 GB，18 個） |
| 控制塔（帳本、專案登記表） | `D:\AI-Portfolio\CC_github部隊\_portfolio_control`（本機） |
| 儀表板 | https://claude.ai/code/artifact/ca431bf0-b18d-400b-a553-c9d3e4d33ed0 （私人） |

## 已完成

| 階段 | 結果 | 決策紀錄 |
|---|---|---|
| W0 骨架 | 套件、CLI、測試（目前 156 個全過）、`make reproduce` | ADR 0001 |
| W1 驗證 | 五個 cell 都能在 4090 服務，含 Qwen3-4B + EAGLE-3 head | ADR 0002–0005 |
| W2 四精度 | FP8 三軸全勝：TMMLU+ 與 BF16 無差（p = 0.945），SLO 容量 2.55 倍，能耗 42 % | ADR 0006–0009 |
| 範圍決定 | 4090 不計成本與電價，只報能耗；證據以 gzip 提交、不用 LFS | ADR 0010、0011 |
| W3 admission | 1.5 倍 5 分鐘突發下整段 attainment：FP8 原生排隊 0.19、hard cap 0.57、有界佇列 0.59；BF16 0.47、0.87、0.86；FP8 由 closed-loop 推得的 C = 256 在突發下 TPOT 超標 | ADR 0012、0013 |
| A1 取消 | 不做雲端第二種 GPU 的對照 | ADR 0014 |

逐時紀錄在 `docs/superpowers/plans/`（W2：`2026-09-10-w2-overnight-run.md`，W3：`2026-09-11-w3-admission-trace.md`）。

## 使用者的決定（2026-09-11）

- A1 取消，其餘全部要做，但「之後有時間再回來做」。
- W4 做**完整版**。GPU 時段由使用者安排，在使用者主動提出之前**不要求 GPU**。
- 4090 不計成本與電價（ADR 0010）；證據體積交由判斷，採 gzip（ADR 0011）。
- 做完後要不要公開，W6 時由使用者決定。
- 使用者對 W2（約 22 小時）與 W3（約 12 小時）逐段追加 GPU 時間非常不滿。**任何 GPU 工作開始前，先一次列出全部剩餘 GPU 時數與可刪減的選項。**

## 剩餘工作與時間

先看 GPU 總預算（2026-09-11 晚間依 W2／W3 的每段實測重估；原估的 10 小時沒算進每段的 re-warm、inference-perf 收尾與每個 session 的啟動與 warm-up）：**全做約 14.5 小時（W4 五個 cell 約 13 小時 + FP8 補點 1.5 小時）；建議的刪減版約 11 小時（W4 約 9.4 小時 + 補點 1.5 小時）**。可以一晚做完，也可以拆成兩次。

| 項目 | 需要 GPU | 預計時間 | 需要使用者做的事 |
|---|---|---|---|
| W4 開工前準備：計畫、程式、ADR 0015、CPU 演練 | 不用 | **已完成（2026-09-11）** | 無 |
| W4 GPU smoke | 需要 | 約 20 分鐘 | 同一晚 |
| W4 完整版：5 個 cell，每個約 2.5 小時 | 需要 | 約 12.6 小時 | 排一晚（或兩晚） |
| W4 刪減版：拿掉 `q4b-ngram` 與 open-loop 的 0.1 × 點 | 需要 | 約 9 小時 | 一晚 |
| FP8 突發安全上限補點 | 需要 | 約 1.5 小時 | 可併入同一晚，或另排 |
| W5 補分析與文件 | 不用 | 約半天（我做） | 無 |
| W6 寫作與發佈前檢查 | 不用 | 約半天（我做） | 決定是否公開 |

每個 cell 的 2.5 小時怎麼來：closed-loop 五點約 0.55 小時（伺服器啟動 2.5 min、100 筆 warm-up 4.3 min、負載約 18 min、四次 re-warm、收尾）；open-loop 15 段約 1.95 小時（啟動、warm-up，每段 5 min 負載加約 2.4 min 的 re-warm、取樣與收尾）。全部可刪減的選項與代價在 ADR 0015 第 11 條。拆成兩次的做法：第一晚 W4（全做 13 小時或刪減版 9.4 小時，含 smoke），第二次 FP8 補點約 1.5 小時。

### W4：speculative decoding 完整版

**狀態（2026-09-12）：GPU 夜間量測（04:05 起）跑完五個 cell（`fp8-none`、`fp8-ngram`、`q4b-none`、`q4b-eagle3`、`q4b-ngram`）；結果、偏離與量測品質在 ADR 0016，逐時紀錄在計畫的 run log；表 `analysis/tables/w4-{fp8,q4b}-specdec/`、圖 `evidence/plots/w4-*.svg`。** 協定全部定案在 ADR 0015（`docs/decisions/0015-w4-specdec-protocol.md`），計畫與 GPU 段落在 `docs/superpowers/plans/2026-09-11-w4-specdec.md`，runbook 有 W4 節。程式：`scripts/wsl/w4-night.sh`（smoke 閘門 → 五個 cell）、`w4-cell-chain.sh`、`check_w4_smoke.py`、`w4-dryrun.sh`（CPU 演練，已跑通）；`batch.sh` 支援 shim、每段 seed、假引擎；harness 記 spec-decode 接受率與 preemption；`slo_lab.specdec_analysis` 做同 family 對 none cell 的配對表。下面 1–8 項的定案版本以 ADR 0015 為準；重點：prompt 用 inference-perf `synthetic`（Shakespeare 文本）；open-loop 的 rate 以同 family 的 none cell 的 r_sat 錨定，三個 cell 在相同 rate 與 seed 下配對；n-gram k = 3、lookup 2–4；所有 cell 經 passthrough shim；cell 順序 `fp8-none` → `fp8-ngram` → `q4b-none` → `q4b-eagle3` → `q4b-ngram`。

開跑：`MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-bench -- bash /mnt/d/.../scripts/wsl/run-logged.sh w4-night.sh`；刪減版寫成腳本檔，例如 `CELLS="fp8-none fp8-ngram q4b-none q4b-eagle3" MULTIPLIERS="0.25 0.5 0.75 0.9"`。收尾步驟在計畫的 Task B4（index.json、reproduce、ADR 0016、README、claims audit）。

**要回答的問題**（規格 §3.1）：加速解碼在低負載與高負載下，對 TPOT 與容量各有什麼影響。

**5 個 cell**，同一晚、同一套協定量測：

| cell | 模型 | 加速方式 | 為什麼要量 |
|---|---|---|---|
| 8B none | `Qwen/Qwen3-8B-FP8` | 無 | 8B 的基準。不直接沿用 W2，因為 W2 的 prompt、到達序列與 token 計數方式和 W4 不同 |
| 8B n-gram | 同上 | n-gram | 8B 唯一可用的加速方式（沒有 8B 的 EAGLE-3 head，claim ceiling 4） |
| 4B none | `Qwen/Qwen3-4B` | 無 | 4B 的基準 |
| 4B n-gram | 同上 | n-gram | |
| 4B EAGLE-3 | 同上 + `AngelSlim/Qwen3-4B_eagle3` | EAGLE-3 | W1 已驗證可載入（ADR 0004） |

**開工前必須定案、寫進 ADR 0015 的事**：
1. **prompt 必須是自然文字**，不能用 W3 的隨機 token。n-gram 靠 prompt 裡重複的片段猜下一個 token，EAGLE-3 的 draft head 用自然文字訓練；用隨機 token 量出來的一定是「沒有加速」，結論無效。建議用 inference-perf 的 `synthetic`（W2 用的文本）或 `shareGPT`，輸出仍用 `ignore_eos` 固定 132 tokens。
2. **負載方式**：W4 不需要突發，用 W2 的穩態 open-loop Poisson 掃描（`open_loop_config`）即可。inference-perf 的 Poisson 沒有種子，W2 也是如此，可以接受。`trace_replay` 只能配隨機 token prompt，所以 W4 不用它。
3. **網格（提案，每個 cell 約 2 小時）**：closed-loop c ∈ {1, 8, 32, 128, 256} 取 r_sat 與低並行的 TPOT；open-loop {0.1, 0.25, 0.5, 0.75, 0.9} × r_sat，3 seeds × 每點 5 分鐘，丟棄前 60 s。低 rate 看 TPOT 改善，高 rate 看容量是否下降。
4. **n-gram 參數**：`config/specdec/ngram.yaml` 的 `num_speculative_tokens` 與 `prompt_lookup_max` 仍是 null，要依 vLLM 0.28 的文件定值（提案 3 與 4）。EAGLE-3 用 `num_speculative_tokens: 3`（W1 載入時的值）。
5. **接受率指標**：`evidence/metrics-names.txt` 是在沒開加速時凍結的，沒有 spec-decode 的指標。smoke 時在開了加速的伺服器上 `curl /metrics | grep spec_decode`，把 accepted／draft token 計數加進 `slo_lab.harness.metrics_scraper.TRACKED`，每段報接受率。
6. **直連或經過 shim**：W4 不量 admission。直連時用戶端與 uvicorn 的 keep-alive 競態會造成約 0.3 % 的連線錯誤（ADR 0012 附錄）；建議所有 cell 都經過 `passthrough` shim，錯誤歸零且與 W3 一致。
7. **4B 的 `--max-num-seqs`**：W1 時 4B 的 KV cache 約 9 萬 tokens，smoke 時確認 256 不會 preemption。
8. 其餘沿用：`--gpu-memory-utilization 0.82`（絕不調高）、`--max-model-len 4096`、`--max-num-batched-tokens 2048`、warm-up 100、可疑規則（probe TPOT 超過最佳值 15 %、Windows committed 超過實體）。

**做法**：參考 W3 的模式寫計畫（`docs/superpowers/plans/`）、driver（可沿用 `scripts/wsl/batch.sh` 的開伺服器與就緒檢查、`w2-cell-chain.sh` 的掃描與隔離重跑），先用 `scripts/wsl/fake_vllm.py` 做 CPU 演練，GPU 時段一開始先跑 20 分鐘 smoke，通過才進正式量測。

### FP8 突發安全上限補點（約 1.5 小時）

W3 發現 FP8 的 C = 256 在突發下 TPOT p95 升到 57 ms，限流在突發段無效。補點用 C = 192 重跑 hard cap，3 seeds，看突發段 attainment 能不能像 BF16 一樣回到 0.7 左右。這是看了 W3 結果才加的探索性補點，要在 ADR 裡標明「非預註冊」。

現成的 driver 可以直接用：

```bash
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-bench -- bash <寫一個腳本檔，內容如下>
# SEEDS="1 2 3" POLICIES="hard_cap" RUN_ROOT="$HOME/vllm-slo-lab/runs-w3-c192" \
#   bash /mnt/d/AI-Portfolio/CC_github部隊/vllm-single-gpu-slo-lab/scripts/wsl/w3-trace-chain.sh fp8-c192 Qwen/Qwen3-8B-FP8 256 192 43.67
```

結果 promote 到 `evidence/raw/w3/fp8-c192/trace/`，在 `analysis/tables/index.json` 加 `w3-fp8-c192-admission`。

### W5：補分析與文件（不用 GPU）

1. **用伺服器的 token 數重算 W2**。W2 的 `records.jsonl` 用的是 inference-perf 重新 tokenize 的計數（1.45 % 的請求算錯，ADR 0012 附錄）。原始逐筆檔只在 `~/vllm-slo-lab/runs-w2`，**W5 做完前不可刪**。重新解析後比較 TPOT p95 與 r_SLO；若有任何 r_SLO 改變，寫 ADR 說明並更新表。
2. **圖**（規格 §8 的交付物）：W2 的 attainment 對 rate 曲線、W3 的佇列時間線（`metrics.csv` + `shim.csv`，以 `t_mono` 對齊）、W4 的 TPOT 對 rate。
3. claims audit 更新、`docs/model-card.md`、README 的敘事。
4. 可選：每段紀錄數都比 trace 列數少 1 筆的原因（ADR 0013）。

**W5 進度（2026-09-12 凌晨，W4 量測進行中順手做的）**：`docs/model-card.md` 草稿（缺 W4 列）；`docs/licences.md` 補齊 GPTQ 與 EAGLE-3 head 的授權（head 的 HF repo 內附 `License_AngelSlim_model_and_dataset.txt`，Apache-2.0）；`slo_lab.plots` 以純 Python 產生 SVG（W2 attainment 對 rate、W3 佇列時間線；W4 的圖在表存在時自動加），`reproduce-lite` 重建、`make reproduce` diff；W5 第 4 項查過 inference-perf 原始碼：trace 列數與請求數相同（`get_request_count` = 列數），少的那一筆發生在派發之後，未再追，影響 0.002 %。W5 第 1 項（重解析 W2）等 GPU 跑完再做，因為要讀 117 GB 原始檔，量測中會擾動磁碟。

### W6：寫作與發佈前檢查（不用 GPU）

**預檢結果（2026-09-12）**：歷史 73 個 commit 作者全部是 `kuotunyu <61350295+kuotunyu@users.noreply.github.com>`，沒有 Co-Authored-By；全樹（含 gzip 證據）掃 `/home/` 只出現在腳本與文件的說明文字、email 只有 noreply 與 example.com、`http` 除 uv.lock 與授權連結外都是 localhost，**唯一要在公開前拿掉的是 `docs/HANDOFF.md` 與 `docs/superpowers/plans/2026-09-10-w2-overnight-run.md` 裡的私人儀表板網址**。

1. 使用者決定是否公開。
2. 若公開：先用 `git filter-repo` 改寫歷史，讓公開 repo 只含 gzip 版證據（目前 `.git` 約 125 MB，因為舊的未壓縮版本還在歷史裡）；確認所有 commit 作者是 `kuotunyu <61350295+kuotunyu@users.noreply.github.com>`、沒有 Co-Authored-By；跑 secrets audit，並另外 grep `/home/`、`@`、`http`；確認 EAGLE-3 head 的授權（model card 沒寫授權，ADR 0003）。
3. 推送後看乾淨 checkout 上的 CI（ruff、pytest、audit、`make reproduce`）是否全綠。GitHub About 精簡、不提名次。
4. 做完後，經使用者同意再清掉 WSL 的 run 目錄，約可釋放 156 GB。

## GPU 時段開工檢查清單

1. 先把本文件「剩餘工作與時間」的 GPU 總預算給使用者看，說清楚哪些可以刪減。
2. 使用者重開機，不開 Docker，不跑其他會用 GPU 的專案；一般桌面使用可以。
3. `MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-bench -- bash /mnt/d/AI-Portfolio/CC_github部隊/vllm-single-gpu-slo-lab/scripts/wsl/preflight.sh`：GPU 閒置、quiet-gpu 通過、權重在快取、磁碟足夠、沒有殘留行程。
4. 啟動 Windows 端 VRAM 取樣：`powershell.exe -NoProfile -ExecutionPolicy Bypass -File <repo>\scripts\win\vram-sampler.ps1 -Out <scratch>\win-vram-<date>.log -Samples 1800 -IntervalSeconds 30`（背景執行）。
5. 用 `scripts/wsl/run-logged.sh <driver>` 啟動，日誌寫在 ext4；用 `scripts/wsl/watch-night.sh` 掛監控。
6. 先跑 smoke，通過才進正式量測。每段結束看 manifest：紀錄數、錯誤數、probe TPOT、Windows committed。
7. 結束後：停取樣器、把 VRAM log 複製到 `evidence/raw/`，`slo-lab reproduce-lite` 跑兩次比對 hash，ruff、pytest、audit，寫 ADR、更新 README 與 claims audit，commit。

## 踩過的坑（都已修正，新程式不要再犯）

| 坑 | 症狀 | 處理 |
|---|---|---|
| 顯存超額 | 桌面程式加上 vLLM 0.90 預算，WDDM 分頁，吞吐減半 | 預算一律 0.82，manifest 記 Windows committed（ADR 0007） |
| inference-perf 多段 Poisson | 每段等請求全部做完才開始下一段，突發的積壓被排空 | 突發用 seeded `trace_replay`（ADR 0012） |
| inference-perf 的 token 計數 | 重新 tokenize 輸出文字，隨機 prompt 下 25 % 算錯，TPOT 被高估 | adapter 優先用伺服器的 `completion_tokens` |
| 文字全空的串流 | 模型先出 EOS，`ignore_eos` 下之後都是空字串，沒有時間戳 | 保守計為未達，每段約 0.1 % |
| keep-alive 競態 | 重用閒置連線撞上 uvicorn 5 s 關閉，502 或連線重設 | shim 對 vLLM 每請求一條連線 |
| shim 連線池 | aiohttp 預設 100 條，悄悄把引擎限制在 100 並行 | `TCPConnector(limit=0)` |
| WSL2 時鐘 | wall clock 每 25 分鐘漂移約 130 s | 一律以 monotonic 的 `t_mono` 對齊 |
| 同一批次目錄的第二個 session | 覆蓋前一個 session 的 server log | `promote-w2.sh` 帶 session tag（ADR 0009 缺陷 6） |
| 就緒檢查 | `curl \| grep -q` 在 pipefail 下因 SIGPIPE 誤判失敗 | 用 command substitution |
| Windows 端 `tee` | 區塊緩衝吃掉日誌 | 日誌寫在 WSL 的 ext4（`run-logged.sh`） |
| quiet-gpu | 前一個伺服器剛關時使用率殘留，誤擋整個 cell | 重試 5 次、每次 30 s |
| benchmark 生成文字 | `vllm bench serve` 的輸出重現訓練資料片段 | 只提交純量摘要與 sha256 |
| 大檔沒被 audit | secrets audit 原本略過 2 MB 以上檔案 | 現在掃描大檔與 gzip 內容 |
| Bash heredoc | 反斜線被吃掉，`\n`、`\v` 變成控制字元，無聲損壞 | 含反斜線的內容用 Write/Edit 工具，寫完掃控制字元 |
| WSL 呼叫 | `wsl.exe bash -c '…$var…'` 會吃掉變數 | 一律寫成腳本檔再執行 |

## 重要檔案

- 規格：`_portfolio_control/docs/superpowers/specs/2026-09-03-vllm-single-gpu-slo-lab-design.md`
- 預註冊：`analysis/preregistration.md`；數字稽核：`analysis/claims_audit.md`
- 決策紀錄：`docs/decisions/0001`–`0014`
- 操作手冊：`docs/runbook-wsl2.md`（W2、W3 段落）
- 證據規格：`evidence/README.md`；表的索引：`analysis/tables/index.json`
- driver：`scripts/wsl/`（W3：`w3-night.sh`、`w3-trace-chain.sh`、`check_w3_smoke.py`、`w3-dryrun.sh`、`fake_vllm.py`）
