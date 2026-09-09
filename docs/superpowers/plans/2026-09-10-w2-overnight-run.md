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

- [ ] **Step 1: Write the pre-flight script (WSL side)**

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

- [ ] **Step 2: Run it**

Run: `MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-bench -- bash "/mnt/d/AI-Portfolio/CC_github部隊/vllm-single-gpu-slo-lab/scripts/wsl/preflight.sh" 2>&1 | tr -d '\0' | grep -v "systemd user session"`
Expected: GPU ≤ 1,700 MiB used, `quiet-gpu` `"ok": true`, four `ok` weight lines, `vllm 0.28.0`, `slo_lab ok`, `inference-perf ok`, ≥ 150 GB free on `/home`, `19680 …full.jsonl`, leftovers `none`.

- [ ] **Step 3: Windows-side baseline**

Run (PowerShell): `& "C:\Windows\System32\nvidia-smi.exe" --query-gpu=memory.used,memory.total --format=csv,noheader; (Get-Counter '\GPU Adapter Memory(*)\Total Committed').CounterSamples | ? { $_.CookedValue -gt 0 } | % { '{0:N0} MB committed' -f ($_.CookedValue/1MB) }`
Expected: committed ≤ 2,500 MB (desktop only). If > 6,000 MB, something else holds VRAM: list it with the `GPU Process Memory` counter and tell the user before launching.

- [ ] **Step 4: No-go rules**

Do not launch if any of: a weight directory is MISSING (report which); `quiet-gpu` refuses (report reasons); free disk < 60 GB (each cell writes ~15 GB of raw loadgen JSON under `runs-w2`; delete `runs-w2/open-loop/fp8/seed-*/ol-rate-*/ipf/per_request_lifecycle_metrics.json` from the finished FP8 runs first — they are ignored files already summarised into `records.jsonl` with sha256 in the manifests); leftovers not `none` (kill with `scripts/wsl/stop-chain.sh`).

---

### Task 1: Launch the chain and the Windows VRAM sampler

**Files:**
- Run: `scripts/wsl/w2-night.sh`
- Run: `$SP/win-vram-sampler.ps1` (append mode; re-create from the block below if the scratchpad vanished)

**Interfaces:**
- Produces: background task ids `NIGHT_TASK` (chain) and `VRAM_TASK` (sampler); log at `$SP/w2-night.log`; sampler log at `$SP/win-vram.log`.

- [ ] **Step 1: Dry-run the plan one more time**

Run: `MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-bench -- env DRY=1 bash "/mnt/d/AI-Portfolio/CC_github部隊/vllm-single-gpu-slo-lab/scripts/wsl/w2-night.sh" 2>&1 | tr -d '\0' | grep -E "START|DRY: WARMUP" | head -8`
Expected: `START awq` then a `DRY: WARMUP=100 MAX_NUM_SEQS=256 RUN_ROOT=…/closed-loop-cells batch.sh awq Qwen/Qwen3-8B-AWQ 1 '' cl:1:90 … cl:256:9000` line.

- [ ] **Step 2: Launch the chain in the background**

Run (Bash, `run_in_background: true`, timeout 600000):
```bash
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-bench -- bash "/mnt/d/AI-Portfolio/CC_github部隊/vllm-single-gpu-slo-lab/scripts/wsl/w2-night.sh" 2>&1 | tr -d '\0' | grep --line-buffered -v "systemd user session" | tee "$SP/w2-night.log"
```
Record the task id as `NIGHT_TASK`. (Background tasks are not killed at the nominal timeout; the 2026-09-09 chains ran 4–7 h this way.)

- [ ] **Step 3: Launch the Windows VRAM sampler**

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

- [ ] **Step 4: Confirm the first server came up (≈ 4 min after launch)**

Run: `MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-bench -- bash "/mnt/d/AI-Portfolio/CC_github部隊/vllm-single-gpu-slo-lab/scripts/wsl/wait-ready.sh" /home/tun2404/vllm-slo-lab/runs-w2/closed-loop-cells/awq/seed-1/serve.log 2>&1 | tr -d '\0' | grep -v "systemd user session"`
Expected: `READY after poll N`, `Loading weights took … seconds`, `GPU KV cache size: …` (AWQ at 0.82: expect ≈ 90,000 tokens or more). `SERVER_GONE` → go to Task 4 (failure playbook, case B).

- [ ] **Step 5: Tell the user it started (one line)**

Message: cell order, ETA table (AWQ ≈ 6.5 h, GPTQ ≈ 6.5 h, BF16 ≈ 5.5 h, contrast ≈ 2.7 h, crosscheck ≈ 0.5 h), that they can go to sleep.

---

### Task 2: First-stage validation (≈ 15 min after launch, then every cell start)

**Files:**
- Run: `$SP/wsl-warmup-read.sh <batch dir>` (prints warm-up medians, probe TPOT, rps, W/util per stage). If the scratchpad vanished, use `scripts/wsl/w2-extras.sh <batch dir>` (same numbers, different layout) — note its stage sort expects `cl-conc-*` names, so use it for closed-loop dirs only.

**Interfaces:**
- Consumes: `runs-w2/closed-loop-cells/<cell>/seed-1/cl-conc-1/manifest.json`.
- Produces: a clean/contaminated verdict for the cell's first point.

- [ ] **Step 1: Read the first completed stage**

Run: `MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-bench -- bash "$SP/wsl-warmup-read.sh" /home/tun2404/vllm-slo-lab/runs-w2/closed-loop-cells/awq/seed-1 2>&1 | tr -d '\0' | grep -v "systemd user session"`
Expected for a clean 4-bit cell: `probe_tpot` 0.012–0.017, `W/util` ≥ 2.0, `win` ≈ 150–200 s, rps at c=1 ≈ 0.45–0.6.

- [ ] **Step 2: Cross-check the Windows side**

Run: `tail -3 "$SP/win-vram.log" | tr -d '\r'`
Expected: committed < 24,564 and no process other than `vmwp`/desktop apps holding > 1 GB. A `python`/`pythonw`/other compute process with ≥ 1 GB means another project is on the GPU: do not stop the chain (the detector will quarantine and re-run), but note the time window for the report.

- [ ] **Step 3: Verdict**

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

- [ ] **Step 1: Wait in ≤ 9.5-minute blocks**

Use `TaskOutput` on `NIGHT_TASK` with `block: true, timeout: 570000`. Between blocks, once per ~30 min, run the progress script for the cell currently running and `tail -1 "$SP/win-vram.log"`. Do not poll more often; nothing changes faster than a stage (5–12 min).

- [ ] **Step 2: On `[<cell>] CELL DONE`, add the cell's tables to the index**

Edit `analysis/tables/index.json` — add, keeping the existing entries:
```json
  "w2-awq-closed-loop": ["evidence/raw/w2/awq/closed-loop"],
  "w2-awq-open-loop": ["evidence/raw/w2/awq/open-loop"]
```
(same pattern with `gptq`, `bf16`, `fp8-mbt8192`; the contrast cell has open-loop seed 1 only.)

- [ ] **Step 3: Rebuild tables, audit, test**

Run (repo root):
```bash
cd "/d/AI-Portfolio/CC_github部隊/vllm-single-gpu-slo-lab" && uv run --frozen slo-lab reproduce-lite --root . 2>&1 | grep -E "tables rebuilt|problem|warning"; uv run --frozen python scripts/redact.py audit . 2>&1 | tail -1; uv run --frozen pytest -q 2>&1 | tail -1
```
Expected: `tables rebuilt from evidence: … w2-<cell>-closed-loop, w2-<cell>-open-loop`, `audit-secrets: clean`, `109 passed` (or more). A `problem:` line means a records file with no window — inspect that stage before committing.

- [ ] **Step 4: Extract the cell's headline numbers**

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

- [ ] **Step 5: Commit the cell**

```bash
cd "/d/AI-Portfolio/CC_github部隊/vllm-single-gpu-slo-lab" && rm -f .git/index.lock && git add -A evidence/raw/w2 analysis/tables && GIT_ASK_YESNO=false git -c user.name=kuotunyu -c user.email=61350295+kuotunyu@users.noreply.github.com commit -q -m "evidence: <cell> closed-loop + open-loop x3 + TMMLU+ (r_sat <x> rps, r_SLO <y> rps, TMMLU+ <n>/600, full <m>/19680)" </dev/null 2>&1 | tail -1; git log --oneline -1
```
Expected: a new commit hash.

- [ ] **Step 6: Interim message to the user (one short paragraph per cell)**

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

- [ ] **Case A — `NIGHT_TASK` completed early (exit code ≠ 0 or log ends before `NIGHT DONE`)**

Read the last 40 lines of `$SP/w2-night.log`. Whatever the cause, the fix is the same: run `scripts/wsl/stop-chain.sh`, confirm `remaining:` is empty, then relaunch Task 1 Step 2. The chain resumes (finished stages have manifests and are skipped; `w2-cell-chain.sh` recomputes r_sat from the finished closed-loop). Do not delete run directories.

- [ ] **Case B — `SERVER_NOT_READY` / `SERVER_EXITED_EARLY` for a cell**

Run: `MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-bench -- bash "$SP/wsl-serve-details.sh" /home/tun2404/vllm-slo-lab/runs-w2/closed-loop-cells/<cell>/seed-1/serve.log` and grep the log for `CUDA out of memory|ValueError|Error`.
- BF16 OOM or `KV cache` too small for `--max-num-seqs 40`: the cell's own chain line in `scripts/wsl/w2-night.sh` is `bash "$WSL/w2-cell-chain.sh" bf16 Qwen/Qwen3-8B 40 "$BF16"` — change **only** the max-num-seqs argument and the grid (`BF16="1 2 4 8 16 24"` with 24) and relaunch; record the change in ADR 0009 (Task 5). Never raise 0.82.
- Any other cell failing to load: skip it by commenting out its line in `w2-night.sh`, relaunch, and report the error text verbatim.

- [ ] **Case C — `QUIET_GPU_REFUSED` at a batch start**

Read `reasons` in the printed JSON. Memory > 3,072 MiB or utilization > 10 % means something else is on the card. Name it (Windows `GPU Process Memory` counter or `nvidia-smi.exe`), wait 10 minutes, retry once by relaunching Task 1 Step 2. If it persists, stop and report; do not lower the gate.

- [ ] **Case D — suspects persist after the chain's two re-run rounds**

Leave them quarantined (`runs-w2/*/quarantine/<cell>/`); the analyzer already excludes them. Report the affected rates/concurrencies and the tenant seen in `win-vram.log` at those times. Do not hand-edit results.

- [ ] **Case E — disk below 30 GB (`df -h /home`)**

Delete `runs-w2/*/<cell>/seed-*/*/ipf/per_request_lifecycle_metrics.json` for cells whose evidence is already promoted and committed (their `records.jsonl` and sha256 are in the repo). Nothing else.

- [ ] **Case F — Windows sampler stopped (`VRAM_TASK` completed)**

Relaunch Task 1 Step 3 (append mode). Losing sampler coverage is acceptable; the manifests carry `host_before/after.windows_gpu_memory` regardless.

---

### Task 5: Wrap-up after `NIGHT DONE` — ADR 0009, README, claims audit, index, cross-check evidence

**Files:**
- Create: `docs/decisions/0009-w2-four-precisions.md`
- Modify: `README.md` (status table), `analysis/claims_audit.md` (rows 6+), `analysis/tables/index.json` (contrast cell), `docs/runbook-wsl2.md` (W2 status paragraph), `analysis/preregistration.md` (per-cell `--max-num-seqs`)
- Promote: `~/vllm-slo-lab/runs-w2/crosscheck/fp8/*.json` → `evidence/raw/w2/fp8/crosscheck/`

- [ ] **Step 1: Promote the cross-check results**

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

- [ ] **Step 2: Build the four-precision table**

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

- [ ] **Step 3: Write ADR 0009**

`docs/decisions/0009-w2-four-precisions.md` with: date, status, evidence paths, the Step 2 table, per-cell notes (BF16 grid cap and KV size from its `serve.log`, GPTQ third-party checkpoint claim ceiling, contrast cell finding: does 8192 move the TPOT knee?), the sensitivity grids (paste `sensitivity.markdown` per cell), suspects/quarantine log, tenant windows from `win-vram.log`, cross-check agreement, and what is still not measured (admission policies W3, spec-decode W4, cost table pending `config/cost.yaml`). Every number must be traceable to a table or manifest path.

- [ ] **Step 4: README status table**

Replace the FP8-only status block at the top of `README.md` with the four-precision table from Step 2 (same column set, Wh per M token instead of raw tok/Wh) and keep the claim-ceiling sentence: numbers hold for the 0.82 budget, WSL2, desktop-shared 4090; no other precision/host extrapolation. Update `analysis/claims_audit.md` with one row per new README number (evidence path, n, CI, ceiling, `make reproduce` check).

- [ ] **Step 5: Freeze the remaining preregistration row**

In `analysis/preregistration.md`, row `--max-num-seqs`: fill AWQ 256, GPTQ 256, BF16 40 (or the value actually used after Case B) and cite ADR 0009; drop the "例外一項" clause from the status line.

- [ ] **Step 6: Reproduce, audit, test, commit**

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
- Modify + publish: `C:\Users\3Hml\.claude\projects\D--AI-Portfolio\2ca314bc-35ae-404a-8e70-998f59664031\tool-results\artifact-ca431bf0-1788377722-93d2.html` (url `https://claude.ai/code/artifact/ca431bf0-b18d-400b-a553-c9d3e4d33ed0`)

- [ ] **Step 1: Ledger 補記** — one bullet: window used, cells done, headline table, suspects, tenant windows, lab commit hash, next steps (W3 admission trace, W4 spec-decode, cost table owner input, publish prep incl. the 117 MB+ evidence size decision). Registry row: status text → "W2 complete (four precisions)". Commit the control tower with the same author flags.

- [ ] **Step 2: Memory** — update `portfolio-inventory-snapshot.md` W2 line with the four-precision headline and the next-step list; no new memory file unless a new non-derivable fact appeared (e.g., a BF16 load limit).

- [ ] **Step 3: Dashboard** — add one `<li>` under the Phase 4 panel (after the "W2 FP8 全套" item) with the four-precision table's r_SLO / TMMLU+ per cell; `Artifact` publish with the url (read the live version first if the publish is refused).

- [ ] **Step 4: Final report to the user** — lead with the four-precision table; then what happened overnight (interruptions, suspects, tenants); then the two decisions they own next (W3 window; `config/cost.yaml` inputs). No em-dashes, numbers in the table only.

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

## Self-review

- Spec coverage: closed-loop, open-loop × 3 seeds, TMMLU+ slices + full, contrast cell, cross-check, suspects handling, evidence promotion, tables/reproduce, ADR, README, claims audit, preregistration row, ledger, memory, dashboard, report — each has a task. Cost table and W3/W4 are explicitly out of scope (owner input / later windows).
- Placeholders: none; every command is verbatim. `<cell>`, `<x>`, `<y>` in Task 3 are the per-cell substitutions named in the same step.
- Consistency: cell names `awq`, `gptq`, `bf16`, `fp8-mbt8192` match `w2-night.sh`; table names `w2-<cell>-closed-loop` / `w2-<cell>-open-loop` match Task 3 Step 2 and Task 5 Step 2; evidence paths match `w2-cell-chain.sh` promote step (`evidence/raw/w2/<cell>/{closed-loop/seed-1,open-loop/seed-N,tmmluplus}`).
