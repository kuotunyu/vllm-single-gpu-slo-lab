# W2 Overnight Multi-Precision Run — Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to run this plan task-by-task in this session (background jobs + monitoring; a fresh subagent per task does not fit an unattended GPU run). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the three remaining precision cells (AWQ, GPTQ-Int4, BF16) plus the FP8 `--max-num-batched-tokens 8192` contrast and the `vllm bench serve` cross-check in one unattended overnight window, then land the evidence, tables, ADR, README numbers and ledger so that the four-precision comparison table exists by tomorrow evening.

**Architecture:** One master shell chain (`scripts/wsl/w2-night.sh`) runs the cells serially on the 4090 inside WSL2; each cell (`scripts/wsl/w2-cell-chain.sh`) is resumable (stages with a `manifest.json` are skipped), quarantines and re-runs stages flagged by the tenancy detector, and promotes its own evidence. This session launches it, samples Windows-side VRAM in parallel, checks in every ~10 minutes, reports per cell, and does the write-up after `NIGHT DONE`.

**Tech Stack:** vLLM 0.28.0 (WSL2 Ubuntu-bench), inference-perf 0.6.1, `slo_lab` harness/analysis (Python 3.12, uv), bash drivers under `scripts/wsl/`, PowerShell GPU counters via interop.

## Global Constraints

- GPU memory budget `--gpu-memory-utilization 0.82` for every cell (ADR 0007); never raise it to fit BF16.
- Engine flags otherwise frozen (`analysis/preregistration.md`): `--max-model-len 4096 --max-num-batched-tokens 2048` (contrast cell 8192), `VLLM_WSL2_ENABLE_PIN_MEMORY=1`, `VLLM_USE_FLASHINFER_SAMPLER=0`, `HF_HUB_OFFLINE=1`.
- Per-cell `--max-num-seqs`: AWQ 256, GPTQ 256, FP8 contrast 256, BF16 **40** (KV ≈ 1.7 GiB at 0.82; ADR 0004/0007).
- Closed-loop grid: `1 2 4 8 16 32 64 96 128 192 256` (BF16: `1 2 4 8 16 24 32 40`); ≥ 180 s per point, 60 s discard.
- Open-loop grid: `{0.25, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 1.0, 1.25, 1.5, 2.0} × r_sat(cell)`, 300 s per point, 60 s discard, seeds 1 2 3 (contrast cell: seed 1 only) — ADR 0008.
- Warm-up 100 sequential + 20 re-warm between stages (ADR 0007).
- Suspect rule (analysis): probe TPOT > best +15 %, W/util < 2.0, or Windows committed VRAM > physical → quarantine to `runs-w2/*/quarantine/<cell>/` and re-run, max two rounds; never average a suspect stage in.
- Commits as `-c user.name=kuotunyu -c user.email=61350295+kuotunyu@users.noreply.github.com`, no Co-Authored-By, `rm -f .git/index.lock` + `GIT_ASK_YESNO=false` + `</dev/null` (non-ASCII path).
- Nothing is pushed to GitHub; nothing is paid for; no engine flag or grid changes beyond this plan without an ADR.
- All WSL invocations go through script files: `MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-bench -- bash <script>`; never `bash -c '…$var…'` (wsl.exe eats `$`).
- Windows paths inside file contents are written with the Write/Edit tools, never Bash heredocs.

## Reference values (FP8, 2026-09-09, for sanity checks)

| quantity | FP8 | what a clean cell looks like |
|---|---|---|
| single-stream probe TPOT | 18.6–19.6 ms | AWQ/GPTQ 4-bit: expect 12–17 ms; BF16: expect 22–30 ms |
| W per util point, c=1 → c=max | 2.27 → 4.26 | rises with concurrency; < 2.0 anywhere = suspect |
| Windows committed VRAM | 22.6–23.7 GB | must stay < 24,564 MB |
| r_sat | 41.4 rps (c=256) | 4-bit cells likely higher; BF16 far lower (grid ends at 40) |
| r_SLO | 26.2 rps | knee where TPOT p95 crosses 50 ms |
| closed-loop cell wall time | 57 min | BF16 shorter |
| open-loop 11 rates × 1 seed | 100 min | overload points 8–12 min each (loadgen wrap-up on CPU) |
| TMMLU+ 3 slices | 366/600 = 0.610 | full set 19,680 items ≈ 11 min at ~30 items/s |

Paths: repo `D:\AI-Portfolio\CC_github部隊\vllm-single-gpu-slo-lab` = `/mnt/d/AI-Portfolio/CC_github部隊/vllm-single-gpu-slo-lab` (`$REPO`); runs `~/vllm-slo-lab/runs-w2/{closed-loop-cells,open-loop-cells,tmmluplus}/<cell>/`; scratch `C:\Users\3Hml\AppData\Local\Temp\claude\D--AI-Portfolio\2ca314bc-35ae-404a-8e70-998f59664031\scratchpad` (`$SP`; may vanish — everything reusable is already in `scripts/wsl/`).

---

### Task 0: Pre-flight after the reboot (user says「開始」)

**Files:**
- Read only: `scripts/wsl/w2-night.sh`, `scripts/wsl/w2-cell-chain.sh`, `scripts/wsl/batch.sh`
- Run: `scripts/wsl/preflight.sh` (committed copy of the block below)

**Interfaces:**
- Produces: a go/no-go decision and the baseline numbers for the report (idle VRAM, disk free, weights present).

- [x] **Step 1: Write the pre-flight script (WSL side)**

```bash
#!/usr/bin/env bash
# Pre-flight for the overnight chain: WSL up, GPU idle, weights cached, venvs importable, disk free.
set -uo pipefail
export HF_HUB_OFFLINE=1
echo "== time =="; date
echo "== gpu =="; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader
echo "== quiet-gpu =="; "$HOME/vllm-slo-lab/.venv-slolab/bin/slo-lab" quiet-gpu --out /tmp/preflight-quiet.json | head -4
echo "== weights in HF cache =="
for m in models--Qwen--Qwen3-8B-AWQ models--JunHowie--Qwen3-8B-GPTQ-Int4 models--Qwen--Qwen3-8B models--Qwen--Qwen3-8B-FP8; do
  d="$HOME/.cache/huggingface/hub/$m/snapshots"; [ -d "$d" ] && echo "ok  $m ($(du -sh "$d" | cut -f1))" || echo "MISSING $m"
done
echo "== venvs =="
"$HOME/vllm-slo-lab/.venv/bin/python" -c "import vllm; print('vllm', vllm.__version__)"
"$HOME/vllm-slo-lab/.venv-slolab/bin/python" -c "import slo_lab.batch_analysis, slo_lab.harness.stage; print('slo_lab ok')"
"$HOME/vllm-slo-lab/.venv-loadgen/bin/inference-perf" --help >/dev/null 2>&1 && echo "inference-perf ok"
echo "== disk =="; df -h /home | tail -1; du -sh "$HOME/vllm-slo-lab/runs-w2" 2>/dev/null
echo "== eval set =="; wc -l "/mnt/d/AI-Portfolio/CC_github部隊/vllm-single-gpu-slo-lab/eval/tmmluplus/full.jsonl"
echo "== leftovers =="; pgrep -af "vllm serve|inference-perf|w2-|wsl/batch" | grep -v pgrep || echo "none"
echo "== io pressure =="; head -1 /proc/pressure/io
```

- [x] **Step 2: Run it**

Run: `MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-bench -- bash "/mnt/d/AI-Portfolio/CC_github部隊/vllm-single-gpu-slo-lab/scripts/wsl/preflight.sh" 2>&1 | tr -d '\0' | grep -v "systemd user session"`
Expected: GPU ≤ 1,700 MiB used, `quiet-gpu` `"ok": true`, four `ok` weight lines, `vllm 0.28.0`, `slo_lab ok`, `inference-perf ok`, ≥ 150 GB free on `/home`, `19680 …full.jsonl`, leftovers `none`.

- [x] **Step 3: Windows-side baseline**

Run (PowerShell): `& "C:\Windows\System32\nvidia-smi.exe" --query-gpu=memory.used,memory.total --format=csv,noheader; (Get-Counter '\GPU Adapter Memory(*)\Total Committed').CounterSamples | ? { $_.CookedValue -gt 0 } | % { '{0:N0} MB committed' -f ($_.CookedValue/1MB) }`
Expected: committed ≤ 2,500 MB (desktop only). If > 6,000 MB, something else holds VRAM: list it with the `GPU Process Memory` counter and tell the user before launching.

- [x] **Step 4: No-go rules**

Do not launch if any of: a weight directory is MISSING (report which); `quiet-gpu` refuses (report reasons); free disk < 60 GB (each cell writes ~15 GB of raw loadgen JSON under `runs-w2`; delete `runs-w2/open-loop/fp8/seed-*/ol-rate-*/ipf/per_request_lifecycle_metrics.json` from the finished FP8 runs first — they are ignored files already summarised into `records.jsonl` with sha256 in the manifests); leftovers not `none` (kill with `scripts/wsl/stop-chain.sh`).

---

### Task 1: Launch the chain and the Windows VRAM sampler

**Files:**
- Run: `scripts/wsl/w2-night.sh`
- Run: `$SP/win-vram-sampler.ps1` (append mode; re-create from the block below if the scratchpad vanished)

**Interfaces:**
- Produces: background task ids `NIGHT_TASK` (chain) and `VRAM_TASK` (sampler); log at `$SP/w2-night.log`; sampler log at `$SP/win-vram.log`.

- [x] **Step 1: Dry-run the plan one more time**

Run: `MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-bench -- env DRY=1 bash "/mnt/d/AI-Portfolio/CC_github部隊/vllm-single-gpu-slo-lab/scripts/wsl/w2-night.sh" 2>&1 | tr -d '\0' | grep -E "START|DRY: WARMUP" | head -8`
Expected: `START awq` then a `DRY: WARMUP=100 MAX_NUM_SEQS=256 RUN_ROOT=…/closed-loop-cells batch.sh awq Qwen/Qwen3-8B-AWQ 1 '' cl:1:90 … cl:256:9000` line.

- [x] **Step 2: Launch the chain in the background**

Run (Bash, `run_in_background: true`, timeout 600000):
```bash
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-bench -- bash "/mnt/d/AI-Portfolio/CC_github部隊/vllm-single-gpu-slo-lab/scripts/wsl/w2-night.sh" 2>&1 | tr -d '\0' | grep --line-buffered -v "systemd user session" | tee "$SP/w2-night.log"
```
Record the task id as `NIGHT_TASK`. (Background tasks are not killed at the nominal timeout; the 2026-09-09 chains ran 4–7 h this way.)

- [x] **Step 3: Launch the Windows VRAM sampler**

If `$SP/win-vram-sampler.ps1` is missing, re-create it with this content (loop 1,200 × 30 s = 10 h; appends):
```powershell
$out = "C:\Users\3Hml\AppData\Local\Temp\claude\D--AI-Portfolio\2ca314bc-35ae-404a-8e70-998f59664031\scratchpad\win-vram.log"
if (-not (Test-Path $out)) { "t_iso dedicated_mb shared_mb committed_mb top_nonvm_dedicated" | Out-File -FilePath $out -Encoding utf8 }
for ($i = 0; $i -lt 1200; $i++) {
  try {
    $c = (Get-Counter -Counter '\GPU Adapter Memory(*)\Dedicated Usage','\GPU Adapter Memory(*)\Shared Usage','\GPU Adapter Memory(*)\Total Committed' -ErrorAction Stop).CounterSamples
    $d = ($c | ? { $_.Path -like '*dedicated usage' } | measure CookedValue -Sum).Sum / 1MB
    $s = ($c | ? { $_.Path -like '*shared usage' } | measure CookedValue -Sum).Sum / 1MB
    $t = ($c | ? { $_.Path -like '*total committed' } | measure CookedValue -Sum).Sum / 1MB
    $p = (Get-Counter -Counter '\GPU Process Memory(*)\Dedicated Usage' -ErrorAction Stop).CounterSamples | Sort-Object CookedValue -Descending
    $top = ""
    foreach ($x in $p) { $id = [regex]::Match($x.InstanceName, 'pid_(\d+)').Groups[1].Value; $n = (Get-Process -Id $id -ErrorAction SilentlyContinue).ProcessName; if ($n -and $n -ne 'vmwp' -and $n -ne 'vmmem' -and $x.CookedValue -gt 50MB) { $top += ('{0}:{1:N0} ' -f $n, ($x.CookedValue/1MB)) }; if ($top.Length -gt 120) { break } }
    ('{0} {1:F0} {2:F0} {3:F0} {4}' -f (Get-Date -Format 'HH:mm:ss'), $d, $s, $t, $top.Trim()) | Out-File -FilePath $out -Encoding utf8 -Append
  } catch { ('{0} error {1}' -f (Get-Date -Format 'HH:mm:ss'), $_.Exception.Message) | Out-File -FilePath $out -Encoding utf8 -Append }
  Start-Sleep -Seconds 30
}
```
Run (PowerShell, `run_in_background: true`): `& powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\3Hml\AppData\Local\Temp\claude\D--AI-Portfolio\2ca314bc-35ae-404a-8e70-998f59664031\scratchpad\win-vram-sampler.ps1"`
Record the task id as `VRAM_TASK`. Re-launch it once after ~10 h (it stops itself).

- [x] **Step 4: Confirm the first server came up (≈ 4 min after launch)**

Run: `MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-bench -- bash "/mnt/d/AI-Portfolio/CC_github部隊/vllm-single-gpu-slo-lab/scripts/wsl/wait-ready.sh" /home/tun2404/vllm-slo-lab/runs-w2/closed-loop-cells/awq/seed-1/serve.log 2>&1 | tr -d '\0' | grep -v "systemd user session"`
Expected: `READY after poll N`, `Loading weights took … seconds`, `GPU KV cache size: …` (AWQ at 0.82: expect ≈ 90,000 tokens or more). `SERVER_GONE` → go to Task 4 (failure playbook, case B).

- [x] **Step 5: Tell the user it started (one line)**

Message: cell order, ETA table (AWQ ≈ 6.5 h, GPTQ ≈ 6.5 h, BF16 ≈ 5.5 h, contrast ≈ 2.7 h, crosscheck ≈ 0.5 h), that they can go to sleep.

---

### Task 2: First-stage validation (≈ 15 min after launch, then every cell start)

**Files:**
- Run: `$SP/wsl-warmup-read.sh <batch dir>` (prints warm-up medians, probe TPOT, rps, W/util per stage). If the scratchpad vanished, use `scripts/wsl/w2-extras.sh <batch dir>` (same numbers, different layout) — note its stage sort expects `cl-conc-*` names, so use it for closed-loop dirs only.

**Interfaces:**
- Consumes: `runs-w2/closed-loop-cells/<cell>/seed-1/cl-conc-1/manifest.json`.
- Produces: a clean/contaminated verdict for the cell's first point.

- [x] **Step 1: Read the first completed stage**

Run: `MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-bench -- bash "$SP/wsl-warmup-read.sh" /home/tun2404/vllm-slo-lab/runs-w2/closed-loop-cells/awq/seed-1 2>&1 | tr -d '\0' | grep -v "systemd user session"`
Expected for a clean 4-bit cell: `probe_tpot` 0.012–0.017, `W/util` ≥ 2.0, `win` ≈ 150–200 s, rps at c=1 ≈ 0.45–0.6.

- [x] **Step 2: Cross-check the Windows side**

Run: `tail -3 "$SP/win-vram.log" | tr -d '\r'`
Expected: committed < 24,564 and no process other than `vmwp`/desktop apps holding > 1 GB. A `python`/`pythonw`/other compute process with ≥ 1 GB means another project is on the GPU: do not stop the chain (the detector will quarantine and re-run), but note the time window for the report.

- [x] **Step 3: Verdict**

If the probe TPOT is > 1.3 × the cell's own best (there is only one point yet: compare with the 4-bit expectation 12–17 ms; for BF16 22–30 ms) **and** committed > 24,564 → paging is back: stop the chain (`scripts/wsl/stop-chain.sh`), run the Windows `GPU Process Memory` counter to name the tenant, tell the user, and relaunch only after it is gone. Otherwise continue.

---

### Task 3: Monitoring cadence and per-cell checkpoint

**Files:**
- Run: `$SP/wsl-ol-progress.sh /home/tun2404/vllm-slo-lab/runs-w2/open-loop-cells/<cell>` (open-loop progress; the `root` argument is the cell dir holding `seed-*`)
- Run: `$SP/wsl-batch-progress.sh /home/tun2404/vllm-slo-lab/runs-w2/closed-loop-cells/<cell>/seed-1` (closed-loop progress)
- Modify: `analysis/tables/index.json` (one entry per finished cell)

**Interfaces:**
- Consumes: `[<cell>] CELL DONE` lines in `$SP/w2-night.log`; promoted evidence under `evidence/raw/w2/<cell>/{closed-loop/seed-1,open-loop/seed-{1,2,3},tmmluplus}`.
- Produces: `analysis/tables/w2-<cell>-closed-loop/`, `analysis/tables/w2-<cell>-open-loop/`, one commit per cell, one interim message per cell.

- [x] **Step 1: Wait in ≤ 9.5-minute blocks**

Use `TaskOutput` on `NIGHT_TASK` with `block: true, timeout: 570000`. Between blocks, once per ~30 min, run the progress script for the cell currently running and `tail -1 "$SP/win-vram.log"`. Do not poll more often; nothing changes faster than a stage (5–12 min).

- [x] **Step 2: On `[<cell>] CELL DONE`, add the cell's tables to the index**

Edit `analysis/tables/index.json` — add, keeping the existing entries:
```json
  "w2-awq-closed-loop": ["evidence/raw/w2/awq/closed-loop"],
  "w2-awq-open-loop": ["evidence/raw/w2/awq/open-loop"]
```
(same pattern with `gptq`, `bf16`, `fp8-mbt8192`; the contrast cell has open-loop seed 1 only.)

- [x] **Step 3: Rebuild tables, audit, test**

Run (repo root):
```bash
cd "/d/AI-Portfolio/CC_github部隊/vllm-single-gpu-slo-lab" && uv run --frozen slo-lab reproduce-lite --root . 2>&1 | grep -E "tables rebuilt|problem|warning"; uv run --frozen python scripts/redact.py audit . 2>&1 | tail -1; uv run --frozen pytest -q 2>&1 | tail -1
```
Expected: `tables rebuilt from evidence: … w2-<cell>-closed-loop, w2-<cell>-open-loop`, `audit-secrets: clean`, `109 passed` (or more). A `problem:` line means a records file with no window — inspect that stage before committing.

- [x] **Step 4: Extract the cell's headline numbers**

Run:
```bash
cd "/d/AI-Portfolio/CC_github部隊/vllm-single-gpu-slo-lab" && uv run --frozen python - <<'PY'
import json
cell = "awq"  # change per cell
cl = json.load(open(f"analysis/tables/w2-{cell}-closed-loop/closed_loop.json", encoding="utf-8"))["per_cell"][cell]
ol = json.load(open(f"analysis/tables/w2-{cell}-open-loop/open_loop.json", encoding="utf-8"))["per_cell"][cell]
print("r_sat", round(cl["r_sat_rps"], 2), "lower_bound", cl["r_sat_is_lower_bound"], "C", cl["capacity_C"], "cl_suspects", cl["suspect_concurrencies_excluded"])
print("r_slo", ol.get("r_slo"), "ol_suspects", ol.get("suspect_rates_excluded"))
print("min attainment by rate:", {k: round(v, 3) for k, v in ol["min_attainment_by_rate"].items()})
print(ol["sensitivity"]["markdown"])
for s in (1, 2, 3):
    try:
        m = json.load(open(f"evidence/raw/w2/{cell}/tmmluplus/slice-{s}.json", encoding="utf-8"))
        print(f"tmmlu slice-{s}: {m['correct']}/{m['n']} = {m['accuracy']}")
    except FileNotFoundError:
        pass
try:
    m = json.load(open(f"evidence/raw/w2/{cell}/tmmluplus/full.json", encoding="utf-8"))
    print(f"tmmlu full: {m['correct']}/{m['n']} = {m['accuracy']} wilson {m['wilson95']}")
except FileNotFoundError:
    print("tmmlu full: not present")
PY
```
Expected: numbers, no exceptions. `cl_suspects`/`ol_suspects` non-empty after the chain's two re-run rounds → keep them excluded, mention in the report.

- [x] **Step 5: Commit the cell**

```bash
cd "/d/AI-Portfolio/CC_github部隊/vllm-single-gpu-slo-lab" && rm -f .git/index.lock && git add -A evidence/raw/w2 analysis/tables && GIT_ASK_YESNO=false git -c user.name=kuotunyu -c user.email=61350295+kuotunyu@users.noreply.github.com commit -q -m "evidence: <cell> closed-loop + open-loop x3 + TMMLU+ (r_sat <x> rps, r_SLO <y> rps, TMMLU+ <n>/600, full <m>/19680)" </dev/null 2>&1 | tail -1; git log --oneline -1
```
Expected: a new commit hash.

- [x] **Step 6: Interim message to the user (one short paragraph per cell)**

Contents: r_sat, C, r_SLO, knee rates, TMMLU+ slices and full, any suspects/quarantine, wall time, whether the next cell started. Numbers in a 4-column table, nothing else.

---

### Task 4: Failure playbook (apply the matching case, then return to Task 3)

**Files:**
- Run: `scripts/wsl/stop-chain.sh` (kills chain, driver, loadgen, server, samplers); re-create if missing:
```bash
#!/usr/bin/env bash
pkill -f "w2-night.sh" 2>/dev/null; pkill -f "w2-cell-chain.sh" 2>/dev/null
pkill -f "scripts/wsl/batch.sh" 2>/dev/null; pkill -f "slo-lab run-stage" 2>/dev/null
pkill -f "inference-perf" 2>/dev/null; pkill -f ".venv/bin/vllm serve" 2>/dev/null
pkill -f "EngineCore" 2>/dev/null; pkill -f "io-sampler.sh" 2>/dev/null
sleep 4; echo "remaining:"; pgrep -af "w2-|wsl/batch|run-stage|inference-perf|vllm serve|EngineCore|io-sampler" | grep -v pgrep || true
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
```

- [x] （contingency 條款）**Case A — `NIGHT_TASK` completed early (exit code ≠ 0 or log ends before `NIGHT DONE`)**

Read the last 40 lines of `$SP/w2-night.log`. Whatever the cause, the fix is the same: run `scripts/wsl/stop-chain.sh`, confirm `remaining:` is empty, then relaunch Task 1 Step 2. The chain resumes (finished stages have manifests and are skipped; `w2-cell-chain.sh` recomputes r_sat from the finished closed-loop). Do not delete run directories.

- [x] （contingency 條款）**Case B — `SERVER_NOT_READY` / `SERVER_EXITED_EARLY` for a cell**

Run: `MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-bench -- bash "$SP/wsl-serve-details.sh" /home/tun2404/vllm-slo-lab/runs-w2/closed-loop-cells/<cell>/seed-1/serve.log` and grep the log for `CUDA out of memory|ValueError|Error`.
- BF16 OOM or `KV cache` too small for `--max-num-seqs 40`: the cell's own chain line in `scripts/wsl/w2-night.sh` is `bash "$WSL/w2-cell-chain.sh" bf16 Qwen/Qwen3-8B 40 "$BF16"` — change **only** the max-num-seqs argument and the grid (`BF16="1 2 4 8 16 24"` with 24) and relaunch; record the change in ADR 0009 (Task 5). Never raise 0.82.
- Any other cell failing to load: skip it by commenting out its line in `w2-night.sh`, relaunch, and report the error text verbatim.

- [x] （contingency 條款）**Case C — `QUIET_GPU_REFUSED` at a batch start**

Read `reasons` in the printed JSON. Memory > 3,072 MiB or utilization > 10 % means something else is on the card. Name it (Windows `GPU Process Memory` counter or `nvidia-smi.exe`), wait 10 minutes, retry once by relaunching Task 1 Step 2. If it persists, stop and report; do not lower the gate.

- [x] （contingency 條款）**Case D — suspects persist after the chain's two re-run rounds**

Leave them quarantined (`runs-w2/*/quarantine/<cell>/`); the analyzer already excludes them. Report the affected rates/concurrencies and the tenant seen in `win-vram.log` at those times. Do not hand-edit results.

- [x] （contingency 條款）**Case E — disk below 30 GB (`df -h /home`)**

Delete `runs-w2/*/<cell>/seed-*/*/ipf/per_request_lifecycle_metrics.json` for cells whose evidence is already promoted and committed (their `records.jsonl` and sha256 are in the repo). Nothing else.

- [x] （contingency 條款）**Case F — Windows sampler stopped (`VRAM_TASK` completed)**

Relaunch Task 1 Step 3 (append mode). Losing sampler coverage is acceptable; the manifests carry `host_before/after.windows_gpu_memory` regardless.

---

### Task 5: Wrap-up after `NIGHT DONE` — ADR 0009, README, claims audit, index, cross-check evidence

**Files:**
- Create: `docs/decisions/0009-w2-four-precisions.md`
- Modify: `README.md` (status table), `analysis/claims_audit.md` (rows 6+), `analysis/tables/index.json` (contrast cell), `docs/runbook-wsl2.md` (W2 status paragraph), `analysis/preregistration.md` (per-cell `--max-num-seqs`)
- Promote: `~/vllm-slo-lab/runs-w2/crosscheck/fp8/*.json` → `evidence/raw/w2/fp8/crosscheck/`

- [x] **Step 1: Promote the cross-check results**

Run (script file):
```bash
#!/usr/bin/env bash
set -uo pipefail
REPO="/mnt/d/AI-Portfolio/CC_github部隊/vllm-single-gpu-slo-lab"
SRC="$HOME/vllm-slo-lab/runs-w2/crosscheck/fp8"; DST="$REPO/evidence/raw/w2/fp8/crosscheck"; mkdir -p "$DST"
for f in "$SRC"/*.json; do sed -e "s#$HOME#~#g" "$f" > "$DST/$(basename "$f")"; done
sed -e "s#$HOME#~#g" "$SRC/serve.log" | "$HOME/vllm-slo-lab/.venv-slolab/bin/python" "$REPO/scripts/redact.py" redact - -o "$DST/vllm.log"
ls -la "$DST"
"$HOME/vllm-slo-lab/.venv-slolab/bin/python" - "$DST" <<'PY'
import json, sys, glob
for p in sorted(glob.glob(f"{sys.argv[1]}/*.json")):
    d = json.load(open(p)); print(p.split("/")[-1], "completed", d.get("completed"), "req/s", round(d.get("request_throughput", 0), 2), "ttft p95 ms", round(d.get("p95_ttft_ms", 0), 1), "tpot p95 ms", round(d.get("p95_tpot_ms", 0), 1), "tok/s", round(d.get("output_throughput", 0)))
PY
```
Expected: `closed-c64.json`, `closed-c256.json`, `open-r21.83.json` plus `vllm.log`; c256 request throughput within ±10 % of the inference-perf r_sat (41.4 rps at 0.82); open 21.83 TTFT/TPOT p95 in the same buckets as the inference-perf 21.84 point (0.09 s / 24–25 ms).

- [x] **Step 2: Build the four-precision table**

Run:
```bash
cd "/d/AI-Portfolio/CC_github部隊/vllm-single-gpu-slo-lab" && uv run --frozen python - <<'PY'
import json
rows = []
for cell, cl_tab, ol_tab in (("fp8", "w2-fp8-closed-loop-v3", "w2-fp8-open-loop"), ("awq", "w2-awq-closed-loop", "w2-awq-open-loop"), ("gptq", "w2-gptq-closed-loop", "w2-gptq-open-loop"), ("bf16", "w2-bf16-closed-loop", "w2-bf16-open-loop"), ("fp8-mbt8192", "w2-fp8-mbt8192-closed-loop", "w2-fp8-mbt8192-open-loop")):
    try:
        cl = json.load(open(f"analysis/tables/{cl_tab}/closed_loop.json", encoding="utf-8"))
        ol = json.load(open(f"analysis/tables/{ol_tab}/open_loop.json", encoding="utf-8"))
    except FileNotFoundError as exc:
        print(cell, "missing", exc); continue
    c = cl["per_cell"][cell]; o = ol["per_cell"][cell]
    knee = [r for r in ol["rows"] if r["offered_rps"] == o.get("r_slo")]
    w = knee[0]["power_mean_w"] if knee else None; e = knee[0]["tok_per_wh"] if knee else None
    tm = {}
    for name in ("slice-1", "slice-2", "slice-3", "full"):
        try: m = json.load(open(f"evidence/raw/w2/{cell}/tmmluplus/{name}.json", encoding="utf-8")); tm[name] = (m["correct"], m["n"])
        except FileNotFoundError: pass
    sl = sum(v[0] for k, v in tm.items() if k.startswith("slice")); sn = sum(v[1] for k, v in tm.items() if k.startswith("slice"))
    rows.append((cell, round(c["r_sat_rps"], 1), c["capacity_C"], o.get("r_slo"), w, e, round(1e6 / e, 1) if e else None, f"{sl}/{sn}" if sn else "-", f"{tm['full'][0]}/{tm['full'][1]}" if "full" in tm else "-"))
print("| cell | r_sat rps | C | r_SLO rps | W @ r_SLO | tok/Wh | Wh per M tok | TMMLU+ slices | TMMLU+ full |")
print("|---|---|---|---|---|---|---|---|---|")
for r in rows: print("| " + " | ".join(str(x) for x in r) + " |")
PY
```
Expected: five rows (contrast row has open-loop seed 1 only — say so in the ADR).

- [x] **Step 3: Write ADR 0009**

`docs/decisions/0009-w2-four-precisions.md` with: date, status, evidence paths, the Step 2 table, per-cell notes (BF16 grid cap and KV size from its `serve.log`, GPTQ third-party checkpoint claim ceiling, contrast cell finding: does 8192 move the TPOT knee?), the sensitivity grids (paste `sensitivity.markdown` per cell), suspects/quarantine log, tenant windows from `win-vram.log`, cross-check agreement, and what is still not measured (admission policies W3, spec-decode W4, cost table pending `config/cost.yaml`). Every number must be traceable to a table or manifest path.

- [x] **Step 4: README status table**

Replace the FP8-only status block at the top of `README.md` with the four-precision table from Step 2 (same column set, Wh per M token instead of raw tok/Wh) and keep the claim-ceiling sentence: numbers hold for the 0.82 budget, WSL2, desktop-shared 4090; no other precision/host extrapolation. Update `analysis/claims_audit.md` with one row per new README number (evidence path, n, CI, ceiling, `make reproduce` check).

- [x] **Step 5: Freeze the remaining preregistration row**

In `analysis/preregistration.md`, row `--max-num-seqs`: fill AWQ 256, GPTQ 256, BF16 40 (or the value actually used after Case B) and cite ADR 0009; drop the "例外一項" clause from the status line.

- [x] **Step 6: Reproduce, audit, test, commit**

```bash
cd "/d/AI-Portfolio/CC_github部隊/vllm-single-gpu-slo-lab" && uv run --frozen slo-lab reproduce-lite --root . 2>&1 | grep -E "tables rebuilt|problem"; sha256sum analysis/tables/*/*.json analysis/tables/*/tables.md > "$TMP/h1.txt"; uv run --frozen slo-lab reproduce-lite --root . >/dev/null 2>&1; sha256sum analysis/tables/*/*.json analysis/tables/*/tables.md > "$TMP/h2.txt"; diff -q "$TMP/h1.txt" "$TMP/h2.txt" && echo deterministic; uv run --frozen ruff check src tests scripts | tail -1; uv run --frozen pytest -q | tail -1; uv run --frozen python scripts/redact.py audit . | tail -1
```
Expected: `deterministic`, `All checks passed!`, all tests pass, `audit-secrets: clean`. Then:
```bash
rm -f .git/index.lock && git add -A && GIT_ASK_YESNO=false git -c user.name=kuotunyu -c user.email=61350295+kuotunyu@users.noreply.github.com commit -q -m "W2 complete: AWQ, GPTQ-Int4, BF16 cells, FP8 batched-tokens contrast, bench-serve cross-check; ADR 0009 four-precision table; README numbers; claims audit" </dev/null 2>&1 | tail -1; git log --oneline -1
```

---

### Task 6: Control tower, memory, dashboard, final report

**Files:**
- Modify: `D:\AI-Portfolio\CC_github部隊\_portfolio_control\docs\inventory\2026-09-02-arsenal-v3-inventory.md` (§19 補記), `…\project-registry.md` (lab row)
- Modify: `C:\Users\3Hml\.claude\projects\D--AI-Portfolio\memory\portfolio-inventory-snapshot.md`
- Modify + publish: `C:\Users\3Hml\.claude\projects\D--AI-Portfolio\2ca314bc-35ae-404a-8e70-998f59664031\tool-results\artifact-ca431bf0-1788377722-93d2.html` (url `<private dashboard artifact; link removed 2026-09-12, the artifact no longer resolves>`)

- [x] **Step 1: Ledger 補記** — one bullet: window used, cells done, headline table, suspects, tenant windows, lab commit hash, next steps (W3 admission trace, W4 spec-decode, cost table owner input, publish prep incl. the 117 MB+ evidence size decision). Registry row: status text → "W2 complete (four precisions)". Commit the control tower with the same author flags.

- [x] **Step 2: Memory** — update `portfolio-inventory-snapshot.md` W2 line with the four-precision headline and the next-step list; no new memory file unless a new non-derivable fact appeared (e.g., a BF16 load limit).

- [x] （私人儀表板 artifact 已不存在，略）**Step 3: Dashboard** — add one `<li>` under the Phase 4 panel (after the "W2 FP8 全套" item) with the four-precision table's r_SLO / TMMLU+ per cell; `Artifact` publish with the url (read the live version first if the publish is refused).

- [x] **Step 4: Final report to the user** — lead with the four-precision table; then what happened overnight (interruptions, suspects, tenants); then the two decisions they own next (W3 window; `config/cost.yaml` inputs). No em-dashes, numbers in the table only.

---

## Time plan (start = user's「開始」, assumed ≈ 23:30)

| block | starts | ends | GPU |
|---|---|---|---|
| Task 0–1 pre-flight + launch | 23:30 | 23:40 | idle |
| AWQ | 23:40 | ≈ 06:10 | busy |
| GPTQ-Int4 | ≈ 06:10 | ≈ 12:40 | busy |
| BF16 | ≈ 12:40 | ≈ 18:10 | busy |
| FP8 mbt8192 contrast | ≈ 18:10 | ≈ 20:50 | busy |
| cross-check | ≈ 20:50 | ≈ 21:20 | busy |
| Task 5–6 wrap-up | ≈ 21:20 | ≈ 22:30 | idle |

Slips: each quarantined stage costs 6–12 min; a paging incident that forces a stop costs the current stage plus a 4-minute server restart.

## Run log (appended during execution)

| time | event |
|---|---|
| 04:09 | Task 0 pre-flight passed: machine rebooted 00:32, GPU 1,049 MiB idle, quiet-gpu ok, four weight dirs cached, 332 GB free, `full.jsonl` 19,680 items, no leftovers. Windows committed 1,276 MB (desktop only). Host disk pressure elevated (io avg10 69 %) from post-reboot activity, not a no-go. |
| 04:11 | Chain launched. AWQ server started. |
| 04:18 | AWQ server ready: weights 136.2 s (cold), KV cache 91,088 tokens (FP8 at the same budget: 76,112). |
| 04:17:52 | **AWQ cell lost.** `batch.sh` answered `GET /v1/models` with 200 and then killed its own server via the `SERVER_NOT_READY` branch. Root cause: the readiness probe `curl … \| grep -q 200` under `set -o pipefail` — `grep -q` exits on the first match, the producer takes SIGPIPE (141), and pipefail turns a successful probe into a failed pipeline. Non-deterministic: GPTQ passed the same code minutes later (two `GET /v1/models` lines in its log; AWQ has one). Reproduced the mechanism in `scratchpad/sigpipe-repro.sh` (long producer: rc=141 every time; 3-byte producer: passes, i.e. a race). The cell chain then found no manifests, could not read r_sat, exited 1, and the night chain moved on to GPTQ. |
| 04:28 | Chain stopped at a stage boundary (`stop-chain.sh`). GPTQ kept its four finished closed-loop stages (c = 1, 2, 4, 16); its in-flight stage has no manifest and will re-run. |
| 04:33 | Fix committed (`dc3a755`): readiness uses command substitution (no pipe, no SIGPIPE), reuses the loop's own result instead of probing twice, caps curl at 5 s, and raises the budget from 120 to 180 polls (15 min) because a cold post-reboot AWQ load took 6.3 min and BF16 weights are twice as big. No other driver had the same pattern. |
| 04:33 | Second harness defect fixed: the chain's log was piped through `tr`/`grep`/`tee` on the Windows side, which block-buffered and lost every line, so the AWQ failure had to be reconstructed from manifests and server logs. `w2-night-logged.sh` now captures the log on ext4; `watch-night.sh` streams progress and failure signatures to an external monitor. |
| 04:35 | Chain relaunched with the fixed driver. AWQ restarts from scratch, GPTQ resumes from four stages. |
| 04:40 | AWQ ready under the fixed probe: weights 84.0 s (warm), **KV cache 98,400 tokens** — the same cell reported 91,088 tokens at 04:18. vLLM sizes the KV cache from free VRAM at profiling time, so on a desktop-shared card it is not deterministic: the earlier attempt profiled while the compositor held more VRAM. Record the per-run value from each `serve.log`; never quote one cell's KV size as a constant. |
| 04:47 | **Protocol deviation to record in ADR 0009.** The closed-loop `num_requests` table is calibrated on FP8 rates, so faster cells finish a point sooner and get a shorter measurement window: AWQ c = 1 offered 90 requests at 0.893 rps (FP8: 0.384) and the window after the 60 s discard held 34 records over ~38 s, against the frozen "≥ 180 s per point". Keeping the counts uniform is the deliberate choice: every cell is offered exactly the same work at every concurrency, which is what makes the four-precision table comparable; scaling counts per cell would make the cells differ in offered load instead. The cost is precision, not bias — report `window_s` and `window_records` in every row and state the minimum window per cell. |
| 09:23 | AWQ cell complete (commit `de65963`), zero suspects. GPTQ server started 09:24. |
| 13:58 | GPTQ-Int4 cell complete (commit `545b2ca`), zero suspects. BF16 started. |
| 18:48 | BF16 cell complete (commit `b032bb4`), zero suspects. The FP8 8192 contrast cell was skipped by a single quiet-GPU reading (utilization 11 % vs 10 %, 20 s after BF16's server exited). |
| 19:00 | `NIGHT DONE`. Cross-check finished; promoted summary-only (commit `1dbd9ea`) because `generated_texts` contained regurgitated third-party paths. quiet-GPU retry (5 × 30 s, threshold unchanged) committed `4bb528f`. |
| 19:01 | Follow-up chain launched via `run-logged.sh w2-followup-night.sh`: contrast cell, FP8 TMMLU+ full set, knee refinement for AWQ, GPTQ, BF16. |
| 21:34 | Contrast cell done: KV 67,888 tokens, r_sat 39.44 (lower bound), C = 128, r_SLO 25.64 (seed 1), zero suspects. `--max-num-batched-tokens` stays 2048. |
| 21:40 | FP8 TMMLU+ full set done: 11,629 / 19,680 = 0.5909, zero errors. |
| 22:45 | AWQ refinement done (24.12 / 25.63 / 27.13 rps × 3 seeds): r_SLO stays 22.61. |
| 23:48 | GPTQ refinement done (24.21 / 25.72 / 27.23 rps × 3 seeds): r_SLO stays 22.70. |
| 01:00 | BF16 refinement done (9.12 / 9.69 / 10.26 rps × 3 seeds): r_SLO 8.55 → 10.26. `FOLLOWUP DONE`. Windows VRAM sampler stopped, log copied to `evidence/raw/w2/win-vram-2026-09-10.log` (2,315 samples, committed max 24,187 MB). |
| 01:05 | `slo-lab reproduce-lite`: all 13 index tables and the paired quality table rebuilt, secrets audit clean. GPU work finished; Tasks 5–6 (write-up) follow. |
| 09-11 wrap-up | Tasks 5–6 done. ADR 0009 final (BF16 row, brackets, sensitivity transpose, refinement results, VRAM monitoring, FP8 untuned-kernel decision, defect 6); README, claims audit (13 rows), runbook, evidence README updated; reproduce deterministic, ruff clean, 126 tests, audit clean. Found at wrap-up: the refinement had overwritten nine seeds' main-run `vllm.log`/`quiet_gpu.json`; restored from git, refinement copies kept as `*-refine-0.80.*`, `promote-w2.sh` gained a session tag (`a31aeef`). Lab `254c171`; control tower `55de225`; memory and dashboard updated. |

## Results as they land (raw notes for ADR 0009)

**AWQ closed-loop, seed 1, finished 05:21** (`runs-w2/closed-loop-cells/awq/seed-1`, all 11 points clean: probe TPOT 7.74–7.84 ms, W/util 2.65 → 3.83, Windows committed 22.9 GB):

| c | window | rps | tok/s | TTFT p95 | TPOT p95 | attainment | W | tok/Wh |
|---|---|---|---|---|---|---|---|---|
| 1 | 38 s | 0.89 | 118 | 0.099 s | 8.2 ms | 1.000 | 251 | 804 |
| 8 | 54 s | 6.73 | 888 | 0.084 | 8.7 | 1.000 | 262 | 9,818 |
| 32 | 84 s | 19.33 | 2,551 | 0.347 | 12.1 | 1.000 | 298 | 23,844 |
| 64 | 124 s | 25.35 | 3,345 | 0.541 | 18.3 | 1.000 | 311 | 30,062 |
| 96 | 156 s | 26.77 | 3,533 | 0.638 | 26.0 | 0.996 | 319 | 31,510 |
| 128 | 172 s | 28.52 | 3,763 | 0.663 | 32.8 | 0.997 | 316 | 33,697 |
| 192 | 207 s | 29.38 | 3,877 | 0.808 | 46.3 | 0.980 | 312 | 35,495 |
| 256 | 235 s | **30.15** | 3,980 | 0.724 | **60.5** | **0.017** | 317 | 36,118 |

- **r_sat(AWQ) = 30.15 rps, a real plateau** (192 → 256 gains 2.6 % < 5 %), unlike FP8 whose 41.4 rps is still a lower bound at the engine's `--max-num-seqs` ceiling.
- **C(AWQ) = 192**: at c = 256 TPOT p95 crosses 50 ms and attainment collapses to 0.017 while throughput barely moves. AWQ runs out of SLO headroom before it runs out of throughput.
- **The precision reversal is the story.** AWQ is 2.3× faster single-stream (0.89 vs 0.38 rps; probe TPOT 7.8 vs 18.6–19.6 ms) but saturates 27 % *lower* than FP8 (30.2 vs 41.4 rps) and is 17 % less energy-efficient at its own maximum (36.1k vs 43.7k tok/Wh). Weight-only 4-bit wins when decode is memory-bound at small batches and loses when large batches make dequantization compute-bound, while FP8 uses the Ada W8A8 tensor cores. AWQ also pays on prefill: at c = 64 its TTFT p95 is 0.541 s against FP8's 0.334 s.
- Open-loop grid for AWQ therefore runs 7.54 … 60.30 rps (0.25 … 2.0 × 30.15), started 05:22.

**AWQ cell complete 09:23** (`evidence/raw/w2/awq/`, commit `de65963`; 33 open-loop + 11 closed-loop stages, zero suspects):

- **r_sat 30.15 rps** (true plateau), **C = 192**, **r_SLO 22.61 rps**, sensitivity grid 21.1 rps at TPOT 30 ms and 22.61 at 50/100 ms, flat across TTFT 0.5–2 s — TPOT-bound like FP8.
- Open-loop knee is a cliff, not a slope: 22.61 rps holds attainment 1.000 with TPOT p95 32.8–40.5 ms, and the next preregistered point (30.15 rps) collapses to 0.000 with TTFT p95 54–64 s. `served_rps` there is only 24.4–25.3 rps, so the sustainable rate sits near 24–25 and the frozen 0.75 → 1.0 gap straddles it.
- **TMMLU+ full set, first ever run: 11,409 / 19,680 = 0.5797**, Wilson [0.573, 0.587] — ten times tighter than the slices' ±0.065; slices were 119/117/124 = 0.595/0.585/0.620. Zero unparsed, zero non-200, zero errors, 77 items/s at concurrency 32.

**Queued for after the main chain** (each needs the GPU, so none of it runs until the four cells are done):

1. **FP8 TMMLU+ full set** (~7 min). FP8 only has slices, so the quality column would otherwise compare a ±0.065 estimate against AWQ's ±0.007.
2. **GPTQ closed-loop c = 1, 2, 4, 8 re-run** (~12 min). Those four stages come from the 04:18 server session that the readiness bug interrupted; the rest of the cell runs in the 09:24 session, which profiled a different KV cache size (98,464 vs 91,152 tokens). The difference does not bind at any grid point, but one server session per cell is the protocol every other cell follows.
3. **Open-loop knee refinement** per cell via `scripts/wsl/refine-cell.sh` — AWQ at 0.80/0.85/0.90 × r_sat = 24.12/25.63/27.14 rps (~63 min); the other cells once their knees are known.

**GPTQ-Int4 cell complete 13:58** (`evidence/raw/w2/gptq/`, commit `545b2ca`): r_sat 30.26 rps (plateau), C = 192, r_SLO 22.70 rps, sensitivity 21.18 rps at TPOT 30 ms and 22.70 at 50/100 ms, TMMLU+ full 11,223/19,680 = 0.5703. Against AWQ's 30.15 / 192 / 22.61 / 0.5797 the two 4-bit cells are indistinguishable in serving behaviour, so **bit width sets the serving envelope and the quantisation method does not**.

**Paired quality (commit `9915dba`)**: on the shared 19,680 items GPTQ scores **0.95 points below AWQ, 95% paired CI [-1.50, -0.37] points, exact McNemar p = 0.00077** (1,606 items only AWQ answered, 1,420 only GPTQ). The independent Wilson intervals ([0.573, 0.587] and [0.563, 0.577]) overlap and cannot support that conclusion; the preregistered paired test can. Two implementation notes worth keeping: McNemar has to be summed in log space because `2 ** n` overflows a float once thousands of items are discordant, and it was the real data — not the unit tests — that exposed it.

**BF16, r_sat 11.40 rps, and the KV cache instability (14:47)**:

| cell | model weights | KV cache per session | sessions agree? |
|---|---|---|---|
| AWQ | 5.7 GiB | 98,400 tokens × 4 | yes, exactly |
| GPTQ | 5.7 GiB | 98,464 tokens × 4 | yes, exactly |
| FP8 | 8.8 GiB | 76,112 tokens (0.82) | — |
| **BF16** | **15.3 GiB** | **11,168 (1.53 GiB) closed-loop vs 29,696 (4.08 GiB) open-loop** | **no, 2.7×** |

vLLM sizes the KV cache from whatever VRAM is free when it profiles, so on a desktop-shared card the residual after a 15.3 GiB model swings with the compositor. The closed-loop grid is still valid — peak KV usage 86.1 %, **zero preemptions**, waiting queue never left 0, so c = 40 fit — but BF16 is the only cell whose own two sessions disagree, and that is itself the finding: on a 24 GB desktop GPU, BF16 leaves so little headroom that its serving capacity is not reproducible across restarts. r_sat 11.40 rps is 3.6× below FP8 and 2.6× below the 4-bit cells.

**BF16 fails on TTFT, not TPOT (17:30, seeds 1–2).** At 11.40 rps (1.0 × r_sat) attainment is 0.746 / 0.938 with TTFT p95 2.80 / 1.15 s while TPOT p95 stays at 25–27 ms, far inside 50 ms; at 8.55 rps everything holds (TTFT p95 0.08 s). FP8, AWQ and GPTQ all broke on TPOT first. BF16 cannot decode faster than the others per token — it cannot *admit* enough: `--max-num-seqs 40` and the thin KV cache cap concurrency, so under Poisson bursts requests queue and the first token is what misses the SLO. The binding constraint switches from decode speed (quantised cells) to admission capacity (BF16).

**Queue revision.** Dropped the "GPTQ closed-loop c = 1, 2, 4, 8 re-run": every cell already spans four server sessions by design (one closed-loop, one per open-loop seed), so "one session per cell" was never the protocol, and GPTQ's closed-loop merely has one extra session boundary at c = 8 → 16 with a KV difference that cannot bind at c ≤ 8 (8 × 240 = 1,920 tokens). Documented instead. The remaining follow-up is `scripts/wsl/w2-followup-night.sh`: FP8 TMMLU+ full set, then knee refinement at 0.80 / 0.85 / 0.90 × r_sat for AWQ, GPTQ and BF16 — all three knees fell in the frozen 0.75 → 1.0 gap, while FP8's was resolved to ~8 % by ADR 0008's 0.55–0.70 points. About 3.3 h of GPU after the main chain.

**BF16 cell complete 18:48** (commit `b032bb4`): r_sat 11.40 rps — not a plateau, still climbing at c = 40 where `--max-num-seqs` caps it, so it is an admission-limited lower bound; C = 40; r_SLO 8.55 rps; TMMLU+ full 11,632 / 19,680 = 0.5911. Its sensitivity grid is 8.55 rps at **every** threshold: the next rate fails on a TTFT queueing cliff that no bound in the grid rescues.

**Paired quality against the unquantised baseline** (same 19,680 items, exact McNemar):

| cell | accuracy | vs BF16 | 95 % paired CI | p |
|---|---|---|---|---|
| BF16 | 0.5911 | — | — | — |
| AWQ | 0.5797 | −1.13 pts | [−1.64, −0.63] | 7.0e−6 |
| GPTQ-Int4 | 0.5703 | −2.08 pts | [−2.55, −1.56] | 1.4e−16 |

**19:00 NIGHT DONE, and one more harness defect.** The FP8 `--max-num-batched-tokens 8192` contrast cell was skipped at 18:48: its quiet-GPU gate ran 20 s after BF16's server exited and read utilization 11 % against the 10 % threshold (samples 15 → 4, memory already back to 2.1 GB). One borderline reading dropped a 2.5-hour cell because the gate had no retry. Fixed in `batch.sh` and `tmmlu-only.sh` (commit `4bb528f`): five attempts 30 s apart, threshold unchanged. The contrast cell is the first item of the follow-up.

**Cross-check (commit `1dbd9ea`).** `vllm bench serve` agrees with inference-perf within 5–7 % on throughput (c = 256: 39.42 vs 41.35 rps; c = 64: 20.61 vs 22.05 rps) and lands in the same latency buckets at 21.8 rps open-loop (TTFT p95 107 ms, TPOT p95 29.9 ms). Committed as **summary-only JSON**: `--save-detailed` stores `generated_texts`, the model's free continuations of random-token prompts, and those reproduced training-data fragments — other people's build paths and website paths. They do not belong in a public repo. Per-request arrays stay under `runs-w2` with the original's sha256 recorded, the same policy already applied to inference-perf's per-request JSON. A repo-wide check found no other committed model text and no `/home/` string.

**19:01 follow-up launched** (`scripts/wsl/w2-followup-night.sh` via `run-logged.sh`): FP8 contrast cell (KV 67,888 tokens at 8192 batched tokens vs 76,112 at 2048), FP8 TMMLU+ full set, then knee refinement for AWQ, GPTQ and BF16. About 6 h of GPU.

Also worth the ADR: **FP8 buys nothing single-stream.** At c = 1 BF16 gives 0.38 rps with probe TPOT 19.3 ms and FP8 gives 0.384 rps at 18.6–19.6 ms, while AWQ/GPTQ give 0.89 rps at 7.8 ms. Halving the weight bytes from BF16 to FP8 does not move batch-1 decode on this card; going to 4-bit does, 2.5×.

**Follow-up results (01:00, raw notes for ADR 0009).**

| cell | 0.80 × r_sat | 0.85 × | 0.90 × | r_SLO before → after | next failing rate |
|---|---|---|---|---|---|
| AWQ | 24.12: 1.000 / 0.873 / 0.842 | 25.63: 0.258 / 0.243 / 0.000 | 27.13: 0 / 0 / 0 | 22.61 → 22.61 | 24.12 (+7 %) |
| GPTQ | 24.21: 0.813 / 1.000 / 1.000 | 25.72: 0 / 0 / 0 | 27.23: 0 / 0 / 0 | 22.70 → 22.70 | 24.21 (+7 %) |
| BF16 | 9.12: 0.999 / 1.000 / 0.996 | 9.69: 0.999 / 0.998 / 1.000 | 10.26: 0.996 / 0.972 / 0.991 | 8.55 → **10.26** | 11.40 (+11 %) |

Attainment per seed 1 / 2 / 3. The quantised knees are cliffs; BF16's is a slope driven by queueing (worst-seed TTFT p95 0.10 → 0.16 → 0.39 → 0.82 s, TPOT p95 flat at 24–26 ms). The coarse grid had under-stated BF16 by 17 %, which would have inflated FP8's SLO advantage from 2.55× to 3.1×. BF16's sensitivity grid is now 9.69 rps at TTFT 0.5 s and 10.26 at 1–2 s, identical across TPOT thresholds, the transpose of the quantised cells. Energy at the new BF16 r_SLO: 74.6 Wh per million output tokens (was 89.8 at 8.55), so FP8 uses 42 % of BF16's energy, not 35 %.

## Self-review

- Spec coverage: closed-loop, open-loop × 3 seeds, TMMLU+ slices + full, contrast cell, cross-check, suspects handling, evidence promotion, tables/reproduce, ADR, README, claims audit, preregistration row, ledger, memory, dashboard, report — each has a task. Cost table and W3/W4 are explicitly out of scope (owner input / later windows).
- Placeholders: none; every command is verbatim. `<cell>`, `<x>`, `<y>` in Task 3 are the per-cell substitutions named in the same step.
- Consistency: cell names `awq`, `gptq`, `bf16`, `fp8-mbt8192` match `w2-night.sh`; table names `w2-<cell>-closed-loop` / `w2-<cell>-open-loop` match Task 3 Step 2 and Task 5 Step 2; evidence paths match `w2-cell-chain.sh` promote step (`evidence/raw/w2/<cell>/{closed-loop/seed-1,open-loop/seed-N,tmmluplus}`).
