# W3 admission 解說動畫（Manim Community）設計

- 日期：2026-09-13
- 狀態：使用者核准設計（「OK 開始」）；實作計畫另立
- 目的：把 W3 的結果（ADR 0013）與 C = 192 補點（ADR 0018）做成一支約 75 秒的解說動畫，放在 README 的 W3 段下面。動畫是**示意**，不是證據：每個數字都讀自已提交的表與紀錄，但影片不進 `make reproduce`、不進 CI、不列入 claims audit 的證據路徑。

## 範圍

做：一個場景 `W3AdmissionBurst`（W3 三策略重播 + C = 256 對 C = 192 的尾聲）。
不做：3b1b 的 ManimGL、LaTeX、把現有 SVG 靜態圖重畫、在 CI 渲染、W2 膝點與 W4 spec-decode 的場景（等這支看過再決定）。

## 資料來源與對應

| 動畫元素 | 來源 | 現有工具 |
|---|---|---|
| 三段到達率 21.8 / 65.5 / 21.8 rps、相位邊界 300 s 與 600 s | `evidence/raw/w3/fp8/trace/seed-1/trace-hard_cap/manifest.json` 的 `trace.phases`；rate_ref 43.67 × burst25.yaml 的倍率 | — |
| 每條泳道的佇列長度（vLLM waiting + shim waiting） | `evidence/raw/w3/fp8/trace/seed-1/trace-<policy>/metrics.csv`、`shim.csv`，以 `t_mono` 對齊紀錄原點 | `slo_lab.admission_analysis.waiting_series` |
| 每條泳道滾動的 attainment、TTFT p95、429 與逾時計數（每 10 s 一桶） | 同目錄的 `records.jsonl.gz`（`offered_at_s`、`outcome`、`ttft_s`、`e2e_s`、`output_tokens`） | `slo_lab.slo.read_records_jsonl`、`meets_slo`；**新增** `slo_lab.timeline` 做分桶 |
| 記分板：整段 attainment 0.19 / 0.57 / 0.59、goodput 5.8 / 17.5 / 18.0 rps、拒絕率 0 / 20.9 / 19.9 %、time-to-recover 745–820 s（1 seed 未恢復）/ 0 / 5–10 s | `analysis/tables/w3-fp8-admission/admission.json` 的 `per_cell_policy` | — |
| 尾聲：C = 256 對 C = 192 的突發段 TPOT p95 57.8 / 45.2 ms、突發段 attainment 0.007 / 0.491、整段 0.573 / 0.780 | `analysis/tables/w3-fp8-admission/` 與 `w3-fp8-c192-admission/` 的 `admission.json` | — |

重播只用 seed 1（三個策略重播同一條 trace），記分板用 3 seeds 平均（與 ADR 0013 表相同）。

## 分鏡（總長約 75 s，1920 × 1080，30 fps）

1. **標題卡（3 s）**：「一張 RTX 4090、1.5 倍 5 分鐘突發、三種 admission」；副標 SLO：TTFT p95 ≤ 1 s 且 TPOT p95 ≤ 50 ms。
2. **佈景（8 s）**：畫面上方是 0–1500 s 的時間軸，三段底色（pre 0.5×、burst 1.5×、recovery 0.5×）；到達率階梯 21.8 → 65.5 → 21.8 rps 以動畫畫出。下方三條橫向泳道，由上而下：原生排隊（passthrough）、hard cap + 429（C = 256）、有界佇列 + 1 s 逾時（Q = 256）。每條泳道左側是標籤，中間是佇列長條（寬度 ∝ 佇列長度，上限以 passthrough 的最大值定標），右側三個讀數：attainment（滾動 10 s 桶）、TTFT p95、拒絕／逾時計數。
3. **重播（30 s）**：游標從 0 掃到 1500 s（每秒動畫 = 50 s 真實時間）。佇列長條依 `waiting_series` 逐幀更新；讀數依 `slo_lab.timeline` 的 10 s 桶更新；hard cap 泳道在突發段每桶依 429 數量彈出紅色小點（數量取對數縮放，不逐筆畫）；有界佇列泳道以橘色小點表示逾時。原生排隊泳道的佇列在 300 s 起持續堆高，TTFT p95 讀數飆到百秒級並變紅。
4. **恢復（10 s）**：游標到 600 s 後放慢，標出原生排隊的 time-to-recover（745／820 s，seed 2 未恢復）與兩種限流的 0／5–10 s；原生排隊的佇列長條緩慢縮短。
5. **記分板（8 s）**：三列 × 四欄的表（整段 attainment、goodput、拒絕率、time-to-recover），數字逐格出現，attainment 欄用色階。
6. **尾聲（12 s）**：兩條泳道「hard cap C = 256」與「hard cap C = 192」，同一條到達序列；突發段 TPOT p95 57.8 ms（紅，超過 50）對 45.2 ms（綠）；突發段 attainment 0.01 → 0.49；整段 0.57 → 0.78；拒絕率 20.9 → 21.8 %。一句話：「上限要用突發下仍 TPOT 安全的並行取，不是 closed-loop 的平台」。
7. **結尾卡（4 s）**：「數字出自 analysis/tables/w3-fp8-admission、w3-fp8-c192-admission（make reproduce）；ADR 0013、0018；示意動畫」與 repo 網址。

文字：繁體中文加英文術語，字型 Microsoft JhengHei（Windows 內建；找不到時退回 Noto Sans TC 或系統預設）。配色：深底、三策略各一色（與 `evidence/plots/w3-queue-timeline-fp8.svg` 相同順序），SLO 超標用紅、達標用綠。

## 架構

```
src/slo_lab/timeline.py          # 純邏輯：分桶（attainment、TTFT p95、429／逾時數）、佇列序列降採樣、相位讀取
tests/test_timeline.py           # 合成紀錄的單元測試（進 CI）
scripts/manim/w3_admission.py    # Manim 場景：只畫，不算；讀 timeline 模組與兩張表
scripts/manim/README.md          # 安裝與渲染步驟
scripts/manim/requirements.txt   # manim==0.21.0（另加本 repo 的 editable install）
docs/media/w3-admission-burst.mp4, .gif   # 產出（提交）
```

- `slo_lab.timeline`
  - `bucket_records(records, *, bucket_s=10.0, end_s=1500.0, slo=DEFAULT_SLO) -> list[Bucket]`：每桶 `start_s`、`offered`、`met`、`rejected`、`timeouts`、`attainment`（met/offered，offered 為 0 時 None）、`ttft_p95_s`（該桶 ok 紀錄的 p95，無則 None）。以 `offered_at_s` 分桶。
  - `rolling(buckets, window)`：每桶回看 `window` 個桶的合併 attainment 與 TTFT p95（動畫讀數用 3 桶 = 30 s，避免逐桶跳動）。
  - `downsample(series, step_s)`：把 `waiting_series` 的點降到固定步長（每 5 s 取最後一筆），給動畫逐幀查值。
  - `phases_of(manifest) -> list[Phase]`：讀 `trace.phases`。
  - 不依賴 manim；只用 `slo_lab.slo` 與標準庫。
- `scripts/manim/w3_admission.py`
  - 啟動時載入三個策略目錄與兩張表，先算好所有序列（`timeline`），再建構 Mobject；每幀以 `ValueTracker`（時間游標）驅動 `always_redraw` 的長條與讀數。
  - 常數：`REPLAY_S = 30`、`BUCKET_S = 10`、`ROLL = 3`；泳道與顏色的順序固定為 passthrough、hard_cap、bounded_queue。
  - 只讀 repo 內的檔案，路徑以 repo 根為基準（`Path(__file__).resolve().parents[2]`）。

## 渲染與產出

```bash
uv venv .venv-manim --python 3.12
uv pip install --python .venv-manim/Scripts/python.exe -r scripts/manim/requirements.txt -e .
.venv-manim/Scripts/manim -qh --disable_caching scripts/manim/w3_admission.py W3AdmissionBurst   # media/videos/w3_admission/1080p30/W3AdmissionBurst.mp4
ffmpeg -y -i media/videos/w3_admission/1080p30/W3AdmissionBurst.mp4 -vf "fps=12,scale=640:-1:flags=lanczos,palettegen" media/palette.png
ffmpeg -y -i media/videos/w3_admission/1080p30/W3AdmissionBurst.mp4 -i media/palette.png -filter_complex "fps=12,scale=640:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5" docs/media/w3-admission-burst.gif
```

- `media/`（Manim 的工作目錄）加進 `.gitignore`；`.venv-manim` 已被現有的 `.venv*` 規則忽略（確認，否則加）。
- GIF 目標 8 MB 以內；超過就把 GIF 裁成重播段（分鏡 3–4，約 40 s）並在 README 註明完整版在 mp4。mp4 直接提交在 `docs/media/`（估 5–8 MB）。
- README 的 W3 段下加：`![W3 admission 示意動畫](docs/media/w3-admission-burst.gif)` 與一行說明「示意動畫（Manim），數字出自 `analysis/tables/w3-*-admission/`，完整版 `docs/media/w3-admission-burst.mp4`；不屬於證據」。
- `evidence/README.md` 加一句：`docs/media/` 是示意動畫，不在 `make reproduce` 的 diff 範圍。
- claims audit 不加列（沒有新數字）；`analysis/claims_audit.md` 開頭的共同條件段加一句「`docs/media/` 的動畫只重述表內數字」。

## 測試與驗收

- `tests/test_timeline.py`：合成紀錄（已知的 offered／met／429／timeout 分布）→ 桶數、attainment、TTFT p95、rolling 的合併值、downsample 的取點；空桶的 None；`phases_of` 讀 manifest。
- 渲染後人工核對：影片裡出現的每個數字與 ADR 0013／0018 表相同（列在 `scripts/manim/README.md` 的核對表）；GIF 在 GitHub README 可播放、≤ 8 MB；`make reproduce` 仍零 diff（動畫不在 diff 範圍）；CI 全綠（timeline 測試在內，manim 不在 CI）。
- 錯誤處理：缺檔（例如沒有 shim.csv）時 `waiting_series` 已回空，動畫以 0 佇列畫並在 log 警告；表缺欄位直接 fail，不畫錯的數字。

## 風險

- Manim 的 CJK 字型：Pango 找不到 Microsoft JhengHei 時會用預設字型，文字仍可讀；渲染前先用 `manim` 的 `Text` 試一幀。
- 檔案大小：1080p 30 fps 75 s 的 mp4 以 Manim 預設 libx264 約 5–10 MB；GIF 依內容變動大，先量再決定裁不裁。
- 時間：資料模組與測試約 1 小時，場景撰寫與調整約 3–4 小時，渲染每次 3–6 分鐘（CPU）。
