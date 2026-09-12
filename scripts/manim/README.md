# 解說動畫（Manim Community）

這裡的動畫是**示意**，不是證據：每個數字都讀自 `analysis/tables/` 與 `evidence/raw/`，但影片不進 `make reproduce`、不進 CI、不列入 claims audit 的證據路徑（設計：`docs/superpowers/specs/2026-09-13-w3-admission-animation-design.md`，計畫：`docs/superpowers/plans/2026-09-13-w3-admission-animation.md`）。

## 安裝（一次）

```bash
uv venv .venv-manim --python 3.12
uv pip install --python .venv-manim/Scripts/python.exe -r scripts/manim/requirements.txt -e .
```

不需要 LaTeX（只用 Pango 文字）；GIF 需要 ffmpeg 在 PATH。字型用 Windows 內建的 Microsoft JhengHei；其他系統把 `scripts/manim/w3_admission.py` 的 `FONT` 改成有的 CJK 字型（例如 Noto Sans TC）。`.venv-manim/` 與 Manim 的工作目錄 `media/` 都在 `.gitignore`。

## 渲染（以 W3 為例；W2 換成 `w2_knee.py W2Knee` 與 `docs/media/w2-knee`，W4 換成 `w4_specdec.py W4Specdec` 與 `docs/media/w4-specdec`）

```bash
.venv-manim/Scripts/manim -qh --disable_caching scripts/manim/w3_admission.py W3AdmissionBurst
# -> media/videos/w3_admission/1080p60/W3AdmissionBurst.mp4（Manim 0.21 的 -qh 是 1080p60）
ffmpeg -y -i media/videos/w3_admission/1080p60/W3AdmissionBurst.mp4 -r 30 -c:v libx264 -crf 20 -preset slow -pix_fmt yuv420p -movflags +faststart -an docs/media/w3-admission-burst.mp4
ffmpeg -y -i docs/media/w3-admission-burst.mp4 -vf "fps=12,scale=640:-1:flags=lanczos,palettegen=max_colors=128" media/palette.png
ffmpeg -y -i docs/media/w3-admission-burst.mp4 -i media/palette.png -filter_complex "fps=12,scale=640:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5" docs/media/w3-admission-burst.gif
```

迭代時用 `-ql`（480p15）看時間軸與版面，最後才用 `-qh`。

## 場景與資料

- `style.py`：三個場景共用的字型、配色、文字輔助、底部註記與出處卡。
- `w2_data.py` + `w2_knee.py`（場景 `W2Knee`，約 42 s）：四精度的 attainment 對 offered rate（`slo_lab.plots.attainment_series`，與 SVG 圖同一套函式）、FP8 的 TPOT p95 對 rate 與 50 ms 線、SLO 敏感度網格、四精度總表（r_SLO、r_sat、Wh／百萬 token、TMMLU+ 與配對差）；`tests/test_w2_anim_data.py` 釘值。
- `w4_data.py` + `w4_specdec.py`（場景 `W4Specdec`，約 42 s）：接受率與平均接受長度、closed-loop 的 rps 比對並行數（`fp8-ngram` 的 c = 256 標「分頁」）、open-loop 同 rate 配對的 TPOT p50／p95 差（只到 0.5 × none cell 的 r_sat）、結論與 r_sat；`tests/test_w4_anim_data.py` 釘值。
- `w3_data.py`：讀 seed 1 的三個策略目錄（`evidence/raw/w3/fp8/trace/seed-1/trace-<policy>/`）、`analysis/tables/w3-fp8-admission/admission.json`、`analysis/tables/w3-fp8-c192-admission/admission.json`，用 `slo_lab.timeline` 分桶（10 s 桶、30 s 滾動讀數、5 s 的佇列格點）；不 import manim，`tests/test_w3_anim_data.py` 在 CI 驗證它讀出的值與表相同。
- `w3_admission.py`：場景 `W3AdmissionBurst`，只畫，不算：時間軸與三條泳道（SLO 條件在畫面底部）→ 30 s 重播（25 分鐘壓縮）→ 恢復時間 → 記分板 → C = 256 對 C = 192 → 出處卡。沒有標題卡、沒有裝飾性特效（使用者要求直接進正題）。佇列長條是對數尺度（原生排隊最高 9,479 筆，有界佇列最高 79 筆，線性尺度看不到後者）。

## 沒有 LaTeX 時的注意事項

- `DecimalNumber`／`Integer` 預設用 MathTex 畫數字，要傳 `mob_class=Text`；它們的 `unit=` 一律走 LaTeX，不能用，單位放到旁邊的標籤。
- `NumberLine(include_numbers=True)` 的刻度要傳 `label_constructor=Text`（`decimal_number_config` 裡不能放 `mob_class`，會和它自己傳的衝突）。
- 出現 `FileNotFoundError: [WinError 2]` 幾乎都是 LaTeX 被呼叫；`media/Tex/` 有 `.tex` 檔就是證據。

## 核對表（渲染後人工核對）

| 影片裡的數字 | 出處 |
|---|---|
| 到達率 21.8 / 65.5 / 21.8 rps、相位邊界 300 / 600 s | `evidence/raw/w3/fp8/trace/seed-1/trace-seed-1.json` |
| 整段 attainment 0.19 / 0.57 / 0.59；goodput 5.8 / 17.5 / 18.0 rps；拒絕率 0 / 20.9 / 19.9 %；time-to-recover 745–820 s（1 個 seed 未恢復）/ 0 / 5–10 s | `analysis/tables/w3-fp8-admission/admission.json` 的 `per_cell_policy.fp8` |
| 尾聲：突發段 TPOT p95 57.8 / 45.2 ms、突發段 attainment 0.007 / 0.491、整段 0.573 / 0.780、拒絕率 20.9 / 21.8 % | 同上與 `analysis/tables/w3-fp8-c192-admission/admission.json` |
| W2：r_SLO 10.26 / 26.2 / 22.61 / 22.70；r_sat 11.40 / 41.35 / 30.15 / 30.26；Wh／百萬 token 74.6 / 31.2 / 36.2 / 36.3；TMMLU+ 0.5911 / 0.5909 / 0.5797 / 0.5703；FP8 敏感度 24.02 / 26.2 / 30.57 | `analysis/tables/w2-*-open-loop/open_loop.json`（`per_cell`、r_SLO 那個 rate 的 `tok_per_wh`）、`w2-*-closed-loop*/closed_loop.json`、`w2-quality-paired/paired.json` |
| W4：接受率 0.52 / 0.27 / 0.53；rps 比 c = 1 1.72 / 1.46 / 1.12，c = 128 0.85 / 0.66 / 0.73，c = 256 分頁 / 0.60 / 0.67；TPOT p95 差 +1.7 到 +3.8、−2.2 到 +10.2、+1.6 到 +4.1 ms；r_sat 26.7 / 28.8 / 32.1 對 40.2 / 47.6 | `analysis/tables/w4-{fp8,q4b}-specdec/specdec.json` |
