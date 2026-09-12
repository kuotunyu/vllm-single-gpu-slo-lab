# 解說動畫（Manim Community）

這裡的動畫是**示意**，不是證據：每個數字都讀自 `analysis/tables/` 與 `evidence/raw/`，但影片不進 `make reproduce`、不進 CI、不列入 claims audit 的證據路徑（設計：`docs/superpowers/specs/2026-09-13-w3-admission-animation-design.md`，計畫：`docs/superpowers/plans/2026-09-13-w3-admission-animation.md`）。

## 安裝（一次）

```bash
uv venv .venv-manim --python 3.12
uv pip install --python .venv-manim/Scripts/python.exe -r scripts/manim/requirements.txt -e .
```

不需要 LaTeX（只用 Pango 文字）；GIF 需要 ffmpeg 在 PATH。字型用 Windows 內建的 Microsoft JhengHei；其他系統把 `scripts/manim/w3_admission.py` 的 `FONT` 改成有的 CJK 字型（例如 Noto Sans TC）。`.venv-manim/` 與 Manim 的工作目錄 `media/` 都在 `.gitignore`。

## 渲染

```bash
.venv-manim/Scripts/manim -qh --disable_caching scripts/manim/w3_admission.py W3AdmissionBurst
# -> media/videos/w3_admission/1080p30/W3AdmissionBurst.mp4
cp media/videos/w3_admission/1080p30/W3AdmissionBurst.mp4 docs/media/w3-admission-burst.mp4
ffmpeg -y -i docs/media/w3-admission-burst.mp4 -vf "fps=12,scale=640:-1:flags=lanczos,palettegen=max_colors=128" media/palette.png
ffmpeg -y -i docs/media/w3-admission-burst.mp4 -i media/palette.png -filter_complex "fps=12,scale=640:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5" docs/media/w3-admission-burst.gif
```

迭代時用 `-ql`（480p15）看時間軸與版面，最後才用 `-qh`。

## 場景與資料

- `w3_data.py`：讀 seed 1 的三個策略目錄（`evidence/raw/w3/fp8/trace/seed-1/trace-<policy>/`）、`analysis/tables/w3-fp8-admission/admission.json`、`analysis/tables/w3-fp8-c192-admission/admission.json`，用 `slo_lab.timeline` 分桶（10 s 桶、30 s 滾動讀數、5 s 的佇列格點）；不 import manim，`tests/test_w3_anim_data.py` 在 CI 驗證它讀出的值與表相同。
- `w3_admission.py`：場景 `W3AdmissionBurst`，只畫，不算：標題卡 → 時間軸與三條泳道 → 30 s 重播（25 分鐘壓縮）→ 恢復時間 → 記分板 → C = 256 對 C = 192 → 結尾卡。佇列長條是對數尺度（原生排隊最高 9,479 筆，有界佇列最高 79 筆，線性尺度看不到後者）。

## 核對表（渲染後人工核對）

| 影片裡的數字 | 出處 |
|---|---|
| 到達率 21.8 / 65.5 / 21.8 rps、相位邊界 300 / 600 s | `evidence/raw/w3/fp8/trace/seed-1/trace-seed-1.json` |
| 整段 attainment 0.19 / 0.57 / 0.59；goodput 5.8 / 17.5 / 18.0 rps；拒絕率 0 / 20.9 / 19.9 %；time-to-recover 745–820 s（1 個 seed 未恢復）/ 0 / 5–10 s | `analysis/tables/w3-fp8-admission/admission.json` 的 `per_cell_policy.fp8` |
| 尾聲：突發段 TPOT p95 57.8 / 45.2 ms、突發段 attainment 0.007 / 0.491、整段 0.573 / 0.780、拒絕率 20.9 / 21.8 % | 同上與 `analysis/tables/w3-fp8-c192-admission/admission.json` |
