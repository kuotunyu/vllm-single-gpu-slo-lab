# 交接文件：vllm-single-gpu-slo-lab（2026-09-13，結案並公開版）

之後回來時先讀這份。它說明專案做到哪裡、還剩什麼（只剩使用者的兩個決定）、怎麼公開、以及這個專案踩過的坑。前一版（2026-09-11，W4 開工前）的內容已由 ADR 0015–0018 與各週計畫取代。

## 一句話現況

W0–W6 全部完成：四種精度（W2）、三種流量控制策略（W3，含 FP8 C = 192 補點）、兩種加速解碼（W4）都有結果、證據與 ADR；W5 的重解析、圖、model card、claims audit、run ledger 都做完；W6 的發佈前檢查已做完並在乾淨 clone 上模擬過 CI。**2026-09-12 深夜依使用者決定公開到 GitHub（<https://github.com/kuotunyu/vllm-single-gpu-slo-lab>），Actions 的 CI 第一次就全綠；contributors 只有 kuotunyu。** 剩下的只有清掉量測主機的 run 目錄（使用者已同意，但永久刪除檔案由使用者自己執行，指令在下方）。不再需要任何 GPU 時間。

## 位置

| 項目 | 位置 |
|---|---|
| 專案 repo（本機 git，remote `origin` 指向 GitHub） | `D:\AI-Portfolio\CC_github部隊\vllm-single-gpu-slo-lab` |
| GitHub | <https://github.com/kuotunyu/vllm-single-gpu-slo-lab>（public，2026-09-12 23:53 首次推送，CI 全綠；About 與 topics 已設；contributors 只有 kuotunyu） |
| 量測環境 | WSL2 發行版 `Ubuntu-bench`，`~/vllm-slo-lab`：`.venv`（vLLM 0.28.0）、`.venv-slolab`（本 repo 的 editable install）、`.venv-loadgen`（inference-perf 0.6.1） |
| 原始量測輸出（只在量測主機，repo 內有 gzip 副本與 sha256） | `~/vllm-slo-lab/runs-w2`（117 GB）、`runs-w3`（36 GB）、`runs-w4`（48 GB）、`runs-w3-c192`（8.5 GB）、`runs-w4-smoke`（0.3 GB）、`reparse-w2`（0.25 GB）；磁碟剩約 154 GB |
| 控制塔（帳本、專案登記表） | `D:\AI-Portfolio\CC_github部隊\_portfolio_control`（本機；2026-09-12 晚已更新 W4–W6 的紀錄） |
| 儀表板 | 舊的私人儀表板連結 2026-09-12 已查無此 artifact，連結已自本文件與 W2 計畫移除 |

## 已完成

| 階段 | 結果 | 決策紀錄 |
|---|---|---|
| W0 骨架 | 套件、CLI、測試（目前 179 個全過）、`make reproduce` | ADR 0001 |
| W1 驗證 | 五個 cell 都能在 4090 服務，含 Qwen3-4B + EAGLE-3 head | ADR 0002–0005 |
| W2 四精度 | FP8 三軸全勝：TMMLU+ 與 BF16 無差（p = 0.945），SLO 容量 2.55 倍，能耗 42 % | ADR 0006–0009 |
| 範圍決定 | 4090 不計成本與電價，只報能耗；證據以 gzip 提交、不用 LFS | ADR 0010、0011 |
| W3 admission | 1.5 倍 5 分鐘突發下整段 attainment：FP8 原生排隊 0.19、hard cap 0.57、有界佇列 0.59；BF16 0.47、0.87、0.86；FP8 由 closed-loop 推得的 C = 256 在突發下 TPOT 超標 | ADR 0012、0013 |
| A1 取消 | 不做雲端第二種 GPU 的對照 | ADR 0014 |
| W4 speculative decoding | 五個 cell：加速只在單流與小批次（1.12 到 1.72 倍），c = 128 起吞吐反轉為 0.6 到 0.85 倍；同 rate 的 attainment 與 r_SLO 不變、TPOT p95 升 1.6 到 10 ms；8B n-gram 在 256 並行時超出記憶體預算而分頁 | ADR 0015、0016 |
| W5 重解析 | W2 的 243 段以伺服器 token 數重算：r_SLO 全部不變、TPOT p95 差在 0.2 ms 內，表不改 | ADR 0017 |
| W3 補點（非預註冊） | FP8 hard cap 改 C = 192 重播同一套突發 trace：突發段 TPOT p95 45 ms（C = 256 為 58）、突發段 attainment 0.49（0.01）、整段 0.78（0.57）、goodput +36 %、拒絕率只多 1 個百分點，三 seed 同號、零重跑 | ADR 0018 |
| W5 其餘 | 7 張 SVG 圖、model card、授權盤點、claims audit 1–22 列、run ledger（`analysis/ledger/runs.csv`，每段一列）都由 `reproduce-lite` 重建或已填 | — |
| W6 發佈前檢查 | 見下節；乾淨 clone 的 CI 模擬（2026-09-12 21:44–21:55，commit `de7a685`）全過：sync、ruff、179 個測試、audit 乾淨、`make reproduce` 零 diff | — |

逐時紀錄在 `docs/superpowers/plans/`（W2：`2026-09-10-w2-overnight-run.md`，W3：`2026-09-11-w3-admission-trace.md`，W4 與 2026-09-12 的收尾：`2026-09-11-w4-specdec.md` 的 run log）。

## 使用者的決定

- 2026-09-11：A1 取消；W4 做完整版；4090 不計成本與電價（ADR 0010）；證據採 gzip（ADR 0011）；做完後要不要公開，W6 時由使用者決定；**任何 GPU 工作開始前，先一次列出全部剩餘 GPU 時數與可刪減的選項**（使用者對 W2、W3 逐段追加 GPU 時間非常不滿）。
- 2026-09-12 凌晨：「電腦開著給你盡情地跑，遇到問題自己判斷，一路做到結束」→ W4 五個 cell 全做（12 小時 46 分 GPU）。
- 2026-09-12 晚：「你來判斷這個專案還有哪些需要做的事，請都完成」→ 我的判斷：做 FP8 C = 192 補點（唯一剩下的 GPU 項目，1.5 小時，開跑前已列出）、補 run ledger、W6 檢查與 CI 模擬、更新控制塔；**不推送、不刪 run 目錄**（這兩件仍是使用者的決定）。

## 還剩什麼（只有使用者能做）

1. ~~是否公開到 GitHub~~ 已公開（2026-09-12）。
2. **清掉量測主機的 run 目錄**（使用者 2026-09-13 已同意；永久刪除檔案由使用者自己執行），約可釋放 210 GB。W2 重解析已完成，`runs-w2` 沒有其他用途；W3、W4、C = 192 的原始逐筆檔在 repo 內都有 gzip 副本與 sha256。指令（在 WSL）：`rm -rf ~/vllm-slo-lab/runs-w2 ~/vllm-slo-lab/runs-w3 ~/vllm-slo-lab/runs-w4 ~/vllm-slo-lab/runs-w3-c192 ~/vllm-slo-lab/runs-w4-smoke ~/vllm-slo-lab/reparse-w2`。

## W6 發佈前檢查結果（2026-09-12）

- **作者**：88 個 commit 的 author 與 committer 全部是 `kuotunyu <61350295+kuotunyu@users.noreply.github.com>`，沒有任何 Co-Authored-By trailer（之後的 commit 也維持同樣做法）。
- **機密**：`make audit-secrets` 全樹乾淨（含 gzip 與 2 MB 以上的檔）；另外 grep `/home/` 只出現在腳本與文件的說明文字，email 只有 noreply 與 example.com，`http` 除 uv.lock 與授權連結外都是 localhost。
- **歷史體積**：`.git` 約 172 MB；歷史中最大的 blob 是 11.7 MB（`evidence/raw/w2/fp8/open-loop/seed-2/vllm.log` 的未壓縮舊版），遠低於 GitHub 的 100 MB 上限與 50 MB 警告。**因此不需要用 `git filter-repo` 改寫歷史**；改寫會讓 ADR 與計畫裡引用的 commit hash 全部失效，建議不改寫、直接推送。
- **乾淨 clone 的 CI 模擬**（`git clone` 到暫存目錄，`uv sync --all-extras --frozen`、`ruff check`、`ruff format --check`、`pytest`、`make audit-secrets`、`make reproduce`，與 `.github/workflows/ci.yml` 相同的步驟）：2026-09-12 21:44–21:55 在 commit `de7a685` 的乾淨 clone 上**全部通過**（`uv sync --frozen` 正常、ruff 乾淨、111 檔格式正確、179 個測試 4 分 24 秒、audit 乾淨、`make reproduce` 重建 17 張表、配對品質表、run ledger 與 7 張圖後零 diff）。C = 192 補點與文件收尾之後，在 commit `b459c92` 的乾淨 clone 上再跑一次同樣的模擬（23:40–23:52）：同樣**全部通過**（179 個測試、audit 乾淨、18 張表、run ledger 364 段、7 張圖，零 diff）。中間全樹 audit 抓到一個新問題並修正：交接文件裡的 SSH 位址 `git@…` 被當成 email 樣式，改用 HTTPS 形式。
- **授權**：`docs/licences.md`（Qwen3 權重 Apache-2.0；GPTQ-Int4 checkpoint Apache-2.0；EAGLE-3 head 的 HF repo 內附 `License_AngelSlim_model_and_dataset.txt`，Apache-2.0；TMMLU+ MIT）。model card 在 `docs/model-card.md`。
- **私人資訊**：私人儀表板網址已移除；`docs/superpowers/plans/` 裡的本機路徑（`D:\...`、`/mnt/d/...`）與 WSL 使用者名稱是說明文字，不含機密。
- **claims audit**：`analysis/claims_audit.md` 22 列，每列對回證據路徑、n、CI 與 claim ceiling；README 的每個數字都在表內。

### 公開步驟（已於 2026-09-12 執行，留作紀錄）

```bash
gh repo create kuotunyu/vllm-single-gpu-slo-lab --public --source . --remote origin --push
```

或手動：在 GitHub 建空的 public repo（不要勾 README／LICENSE），然後 `git remote add origin https://github.com/kuotunyu/vllm-single-gpu-slo-lab.git`（或 SSH 形式；文件裡用 HTTPS 是因為 `make audit-secrets` 會把 `git@` 開頭的 SSH 位址當成 email 樣式） 與 `git push -u origin main`。推送後看 Actions 的 `CI` 是否全綠（ruff、pytest、audit、`make reproduce`），全綠才算發佈。GitHub About 草稿（精簡、不提名次）：

> SLO-bounded capacity, energy and quality of Qwen3-8B on one desktop-shared RTX 4090 (vLLM 0.28, WSL2): four precisions, three admission policies, speculative decoding; every number rebuilt from committed evidence.

不需要 Git LFS，不需要改寫歷史。

## GPU 時段開工檢查清單（若日後再量）

1. 先把剩餘 GPU 總預算與可刪減的選項給使用者看。
2. 使用者重開機，不開 Docker，不跑其他會用 GPU 的專案；一般桌面使用可以。
3. `MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-bench -- bash /mnt/d/AI-Portfolio/CC_github部隊/vllm-single-gpu-slo-lab/scripts/wsl/preflight.sh`：GPU 閒置、quiet-gpu 通過、權重在快取、磁碟足夠、沒有殘留行程。
4. 啟動 Windows 端 VRAM 取樣：`powershell.exe -NoProfile -ExecutionPolicy Bypass -File <repo>\scripts\win\vram-sampler.ps1 -Out <scratch>\win-vram-<date>.log -Samples 1800 -IntervalSeconds 30`（背景執行）。
5. 用 `scripts/wsl/run-logged.sh <driver>` 啟動，日誌寫在 ext4；用 `scripts/wsl/watch-night.sh` 掛監控。
6. 先跑 smoke，通過才進正式量測。每段結束看 manifest：紀錄數、錯誤數、probe TPOT、Windows committed。
7. 結束後：停取樣器、把 VRAM log 複製到 `evidence/raw/`，`slo-lab reproduce-lite` 跑兩次比對 hash，ruff、pytest、audit，寫 ADR、更新 README 與 claims audit，commit。

現成的 driver：W2 `w2-night.sh`／`w2-cell-chain.sh`／`refine-cell.sh`，W3 `w3-night.sh`／`w3-trace-chain.sh`／`w3-c192.sh`，W4 `w4-night.sh`／`w4-cell-chain.sh`，各有 `*-dryrun.sh` 可用 `fake_vllm.py` 在 CPU 上演練。

## 踩過的坑（都已修正，新程式不要再犯）

| 坑 | 症狀 | 處理 |
|---|---|---|
| 顯存超額 | 桌面程式加上 vLLM 0.90 預算，WDDM 分頁，吞吐減半 | 預算一律 0.82，manifest 記 Windows committed，分析器把超過實體的段標可疑（ADR 0007） |
| n-gram 的記憶體成長 | 8B n-gram 在 256 並行時比 0.82 的 profile 多用約 3 GB，桌面分頁 | 依可疑規則排除、三個 session 確認；4B 有餘裕不受影響（ADR 0016） |
| 共用 session 的實體 VRAM 查詢 | seed-2、seed-3 目錄沒有自己的 `quiet_gpu.json`，分析器查不到實體值而漏標分頁段 | 向同批的 `seed-*/quiet_gpu.json` 查（ADR 0016 偏離 8） |
| inference-perf 多段 Poisson | 每段等請求全部做完才開始下一段，突發的積壓被排空 | 突發用 seeded `trace_replay`（ADR 0012） |
| inference-perf 的 token 計數 | 重新 tokenize 輸出文字，隨機 prompt 下 25 % 算錯，TPOT 被高估 | adapter 優先用伺服器的 `completion_tokens`；W2 重解析確認無影響（ADR 0017） |
| spec decode 的 probe | 加速時一個 SSE chunk 有多個 token，probe 讀到的是每步時間 | 單流數字取自紀錄，probe 只作同 cell 內的漂移偵測（ADR 0016） |
| 文字全空的串流 | 模型先出 EOS，`ignore_eos` 下之後都是空字串，沒有時間戳 | 保守計為未達，每段約 0.1 % |
| keep-alive 競態 | 重用閒置連線撞上 uvicorn 5 s 關閉，502 或連線重設 | shim 對 vLLM 每請求一條連線；所有 cell 經 passthrough shim |
| shim 連線池 | aiohttp 預設 100 條，悄悄把引擎限制在 100 並行 | `TCPConnector(limit=0)` |
| WSL2 時鐘 | wall clock 每 25 分鐘漂移約 130 s | 一律以 monotonic 的 `t_mono` 對齊 |
| 同一批次目錄的第二個 session | 覆蓋前一個 session 的 server log | `promote-w2.sh` 帶 session tag（ADR 0009 缺陷 6） |
| 就緒檢查 | `curl \| grep -q` 在 pipefail 下因 SIGPIPE 誤判失敗 | 用 command substitution |
| Windows 端 `tee` | 區塊緩衝吃掉日誌 | 日誌寫在 WSL 的 ext4（`run-logged.sh`） |
| quiet-gpu | 前一個伺服器剛關時使用率殘留，誤擋整個 cell | 重試 5 次、每次 30 s |
| benchmark 生成文字 | `vllm bench serve` 的輸出重現訓練資料片段 | 只提交純量摘要與 sha256 |
| 大檔沒被 audit | secrets audit 原本略過 2 MB 以上檔案 | 現在掃描大檔與 gzip 內容 |
| Bash heredoc | 反斜線被吃掉，`\n`、`\v` 變成控制字元，無聲損壞 | 含反斜線的內容用 Write/Edit 工具，寫完掃控制字元；Python 寫檔用 `newline="\n"`，否則 Windows 上會變 CRLF |
| WSL 呼叫 | `wsl.exe bash -c '…$var…'` 會吃掉變數 | 一律寫成腳本檔再執行 |
| `pkill -f` | 樣式會匹配到自己的 `bash -c` 命令列，把檢查殺掉 | 檢查也寫成腳本檔 |
| 管線尾端 `\| tail -1` | 會吃掉前一個指令的失敗退出碼 | 看退出碼，不接 `tail` |

## 重要檔案

- 規格：`_portfolio_control/docs/superpowers/specs/2026-09-03-vllm-single-gpu-slo-lab-design.md`
- 預註冊與偏離：`analysis/preregistration.md`；數字稽核：`analysis/claims_audit.md`；run ledger：`analysis/ledger/runs.csv`
- 決策紀錄：`docs/decisions/0001`–`0018`
- 操作手冊：`docs/runbook-wsl2.md`；model card：`docs/model-card.md`；授權：`docs/licences.md`
- 證據規格：`evidence/README.md`；表的索引：`analysis/tables/index.json`；圖：`evidence/plots/`
- driver：`scripts/wsl/`；Windows 端顯存取樣：`scripts/win/vram-sampler.ps1`
