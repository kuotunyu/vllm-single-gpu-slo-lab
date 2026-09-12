# W2 膝點與 W4 speculative decoding 解說動畫設計

- 日期：2026-09-13
- 狀態：使用者委託（「都照你建議的方式進行」）；口味沿用第一支修改後的版本：不做標題卡、不做裝飾性特效，直接進資料，只留淡入與長條成長
- 目的：把 W2（四精度在 SLO 下的容量與膝點，ADR 0007–0009）與 W4（speculative decoding，ADR 0015–0016）各做成一支約 55 秒的示意動畫，放在 README 對應段落下面。與第一支相同：每個數字讀自已提交的表，影片不是證據、不進 `make reproduce`、不進 CI。

## 共同規則

- 樣式抽成 `scripts/manim/style.py`（字型、配色、文字輔助、底部註記、出處卡），三個場景共用；`w3_admission.py` 改用它。
- 每支動畫的資料由一個不 import manim 的載入器提供（`w2_data.py`、`w4_data.py`），各有 CI 測試把讀到的值釘在表上。
- 產出 `docs/media/w2-knee.{mp4,gif}`、`docs/media/w4-specdec.{mp4,gif}`（1080p30 與 640 px 12 fps GIF，各不超過 8 MB）。
- Manim 0.21、無 LaTeX：數字物件 `mob_class=Text`，座標軸 `label_constructor=Text`，單位放標籤。

## 第 2 支：W2 膝點（`W2Knee`，約 55 s）

資料（`w2_data.load_knee`）：

| 元素 | 來源 |
|---|---|
| 四精度的 attainment 對 offered rate（每個 rate 取 3 seeds 的最小值） | `analysis/tables/w2-<cell>-open-loop/open_loop.json` 的非可疑列；與 `evidence/plots/w2-attainment-vs-rate.svg` 同一套函式 `slo_lab.plots.attainment_series` |
| r_SLO | 同表 `per_cell.<cell>.r_slo` |
| FP8 的 TPOT p95、TTFT p95 對 rate（3 seeds 平均） | 同表列；`slo_lab.plots.tpot_series` 與載入器自己算的 TTFT 平均 |
| FP8 的 SLO 敏感度網格 | `per_cell.fp8.sensitivity.r_slo_by_threshold` |
| r_sat | `analysis/tables/w2-<cell>-closed-loop*/closed_loop.json` 的 `per_cell.<cell>.r_sat_rps`（FP8 用 `w2-fp8-closed-loop-v3`） |
| 能耗 Wh／百萬 output token | r_SLO 那個 rate 的三個 seed 的 `tok_per_wh` 平均取倒數（與 README 的 74.6／31.2／36.2／36.3 相同） |
| TMMLU+ 與對 BF16 的配對差 | `analysis/tables/w2-quality-paired/paired.json` |

分鏡：

1. **膝點（20 s）**：attainment 對 offered rate 的座標軸（0–90 rps、0–1，95 % 水平線），四條曲線依 BF16、FP8、AWQ、GPTQ 順序畫出；每條畫完在 r_SLO 立一條虛線並標值（10.26 / 26.2 / 22.61 / 22.70 rps）。
2. **膝點是 TPOT（12 s）**：換成 FP8 的 TPOT p95 對 rate（50 ms 線），曲線在 26–31 rps 之間穿過 50 ms；一行字：TTFT p95 到 30.6 rps 都在 0.2 s 以下，膝點由 decode 步長決定，不是排隊。
3. **SLO 敏感度（10 s）**：FP8 的 3 × 3 表（TTFT 0.5／1／2 s × TPOT 30／50／100 ms）逐格填入 r_SLO；一行字：r_SLO 對 TPOT 門檻敏感、對 TTFT 門檻不敏感。
4. **記分板（10 s）**：四精度 × r_SLO、r_sat、Wh／百萬 token、TMMLU+、對 BF16 的配對差（p 值）。
5. **出處卡（3 s）**。

底部註記：`--gpu-memory-utilization 0.82`、WSL2、與桌面共用的 RTX 4090、Qwen3-8B、108 → 132 tokens。

## 第 3 支：W4 speculative decoding（`W4Specdec`，約 55 s）

資料（`w4_data.load_specdec`）：

| 元素 | 來源（`analysis/tables/w4-<family>-specdec/specdec.json`） |
|---|---|
| 接受率、平均接受長度 | `cells.<cell>.acceptance_rate_mean`、`mean_acceptance_length_mean` |
| closed-loop 的 rps 比（c = 1, 8, 32, 128, 256） | `cells.<cell>.closed_paired[].rps_ratio`；缺的點（`fp8-ngram` c = 256，分頁排除）畫成「分頁」標記 |
| open-loop 同 rate 的 TPOT p50／p95 差與 attainment 差 | `cells.<cell>.open_paired_by_rate`，只取 rate ≤ 0.5 × 同 family none cell 的 r_sat 的點（過載段不比） |
| r_sat 與對 none 的比 | `cells.<cell>.r_sat_rps`、`r_sat_vs_base`、`base_cell` 的 `r_sat_rps` |

分鏡：

1. **機制與接受率（10 s）**：一行字說明每步先猜 3 個 token、目標模型一次驗證；三條長條：8B n-gram 0.52、4B EAGLE-3 0.27、4B n-gram 0.53，旁邊標平均接受長度 2.56 / 1.80 / 2.60。
2. **closed-loop（15 s）**：分組長條圖，x 為並行數 1 / 8 / 32 / 128 / 256，每組三根（三個加速 cell 對 none 的 rps 比），1.0× 水平線；c = 128 起低於 1；`fp8-ngram` 的 c = 256 標「分頁」。
3. **open-loop（15 s）**：每個 cell 一組：在 0.1×、0.25×、0.5× r_ref 三個 rate 的 TPOT p50 差（多為負，綠）與 p95 差（多為正，紅）長條，單位 ms；attainment 差以文字標「0」。EAGLE-3 在 12 rps 以下 p95 也降的例外照畫。
4. **結論（8 s）**：三行字：單流與小批次變快；c ≥ 128 吞吐反轉；同 rate 的 p95 變慢，以 p95 定義的 SLO 下容量沒有增加；r_sat 26.7 / 28.8 / 32.1 對 none 40.2 / 47.6。
5. **出處卡（3 s）**。

底部註記：Shakespeare 自然文字 prompt、Qwen3 預設取樣、全部經 passthrough shim、0.82 預算。

## 架構

```
scripts/manim/style.py        # 共用樣式與文字輔助（w3_admission.py 改用）
scripts/manim/w2_data.py      # W2 載入器（不 import manim）
scripts/manim/w2_knee.py      # 場景 W2Knee
scripts/manim/w4_data.py      # W4 載入器（不 import manim）
scripts/manim/w4_specdec.py   # 場景 W4Specdec
tests/test_w2_anim_data.py, tests/test_w4_anim_data.py
docs/media/w2-knee.{mp4,gif}, docs/media/w4-specdec.{mp4,gif}
```

## 驗收

- 兩個載入器測試在 CI 通過，值與表相同（r_SLO、能耗、TMMLU+、敏感度、接受率、rps 比、TPOT 差）。
- 影片裡的每個數字能在 README 的 W2／W4 表或 claims audit 第 1–9、17–19 列找到；GIF 各 ≤ 8 MB；`make reproduce` 零 diff；CI 全綠。
- 不做的：標題卡、裝飾性特效、W2 的 TMMLU+ 逐題動畫、W4 的 draft token 逐字動畫。
