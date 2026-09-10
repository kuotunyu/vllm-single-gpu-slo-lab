# W3 Admission Trace Implementation and Overnight Run Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the three admission policies (vLLM native queue, hard cap + 429, bounded queue + timeout) under the preregistered 25-minute burst trace on FP8 and BF16, three seeds each, in one overnight window on the desktop-shared RTX 4090, and turn the result into ADR-backed tables.

**Architecture:** Part A (CPU only, before the window) adds a seeded burst-trace generator, an inference-perf trace-replay stage that runs through the admission shim, a shim-stats scraper, a connector-limit fix in the shim, a per-seed driver, and an admission analysis module wired into `make reproduce`. Part B (the GPU window) runs one vLLM session per (precision, seed) and replays the same trace through each policy in rotated order, then promotes gzip evidence and rebuilds tables.

**Tech Stack:** Python 3.12 (uv), vLLM 0.28.0 in WSL2 `Ubuntu-bench`, inference-perf 0.6.1 (`~/vllm-slo-lab/.venv-loadgen`), aiohttp shim (`slo-lab shim`), bash drivers under `scripts/wsl/`, pytest, ruff.

## Global Constraints

- Engine flags frozen from W2: `--max-model-len 4096 --gpu-memory-utilization 0.82 --max-num-batched-tokens 2048`; `--max-num-seqs` FP8 256, BF16 40; env `VLLM_WSL2_ENABLE_PIN_MEMORY=1`, `VLLM_USE_FLASHINFER_SAMPLER=0`, `HF_HUB_OFFLINE=1`. Never raise 0.82.
- Trace `config/traffic/burst25.yaml`: 0.5·r × 300 s → 1.5·r × 300 s → 0.5·r × 900 s, Poisson, 108 in / 132 out tokens, `ignore_eos`, thinking off, client timeout 300 s.
- Rate reference r: FP8 43.67 rps (preregistered 21.8 / 65.5 / 21.8), BF16 11.40 rps (5.70 / 17.10 / 5.70).
- C, Q, T (preregistration): FP8 C = Q = 256, BF16 C = Q = 40, T = 1 s, Retry-After 1 s.
- Seeds 1, 2, 3; paired: the three policies of a seed replay the identical trace file.
- SLO: TTFT p95 ≤ 1 s and TPOT p95 ≤ 50 ms; attainment denominator = offered (429, timeout, 5xx all count as misses).
- All three policies go through the shim (`passthrough` for native), so the shim is common to every arm.
- Suspect rule for trace stages: probe TPOT > best + 15 % in the batch, or Windows committed VRAM > physical → quarantine and re-run (max 2 rounds, never average). W/util is recorded but not used as a trace-stage flag (a 0.5·r phase is legitimately light).
- Nothing pushed to GitHub, nothing paid, no Co-Authored-By; commits use `-c user.name=kuotunyu -c user.email=61350295+kuotunyu@users.noreply.github.com`, `GIT_ASK_YESNO=false`, stdin closed.
- WSL invocations only through script files: `MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-bench -- bash <script>`; never `bash -c '…$var…'`.
- Files that contain backslashes are written with the Write/Edit tools, never through a Bash heredoc.

## Finding that shaped the design (2026-09-11, read from the installed inference-perf source)

`LoadGenerator.run_stage` waits until every request of a stage has finished before the next stage starts (`while finished_requests_counter.value < num_requests`), then sleeps `load.interval`. A three-stage Poisson config would therefore drain the burst backlog with no arrivals and start the recovery stage late, which erases exactly what W3 measures. `load.type: trace_replay` with a pre-generated Azure-format trace (`TIMESTAMP,ContextTokens,GeneratedTokens`) is one continuous stage. The reader keeps two fractional digits, so timestamps are generated on a 10 ms grid and the committed file is exactly what is replayed. inference-perf's own Poisson timer uses an unseeded generator, so seeded trace files are also what makes the three policies see identical arrivals.

A second finding: the shim's upstream `aiohttp.ClientSession` used the default connector limit of 100 connections, which would silently cap vLLM at 100 concurrent requests behind every policy. Fixed in Task A2.

## File structure

| File | Responsibility |
|---|---|
| `src/slo_lab/harness/trace.py` (new) | Seeded piecewise-Poisson arrivals on a 10 ms grid, Azure-format writer, phase table from `burst25.yaml` |
| `src/slo_lab/harness/ipf_config.py` | `trace_replay_config` beside the existing open/closed-loop configs |
| `src/slo_lab/harness/shim_scraper.py` (new) | Background poll of `/_shim/stats` into `shim.csv` |
| `src/slo_lab/harness/stage.py` | `kind="trace"`: warm-up through the shim, both scrapers, shim CPU, clock alignment, per-phase summaries in the manifest |
| `src/slo_lab/harness/ipf_adapter.py` | `adapt_file_with_origin` so the stage can align records with scraped series |
| `src/slo_lab/admission/shim.py` | Unlimited upstream connector |
| `src/slo_lab/admission_analysis.py` (new) | Phase summaries, time-to-recover, per-policy tables with paired differences vs native |
| `src/slo_lab/batch_analysis.py` | `run()` writes admission tables when the manifests are trace stages |
| `src/slo_lab/cli.py` | `make-trace` command; `run-stage` trace options |
| `scripts/wsl/w3-trace-chain.sh` (new) | One cell: per seed, one vLLM session, three policies in rotated order, promote |
| `scripts/wsl/w3-night.sh` (new) | GPU smoke, then FP8 then BF16 |
| `scripts/wsl/promote-w2.sh` | `EVIDENCE_WEEK` (default `w2`) and `shim.csv`, `trace-seed-*.csv` |
| `scripts/compress_evidence.py` | Also gzip `trace-seed-*.csv` |
| `docs/decisions/0012-w3-protocol.md` (new) | Every W3 choice fixed before data |

## Part A: implementation (CPU only, 2026-09-11 afternoon)

### Task A1: Burst trace generator and `make-trace`

**Files:** Create `src/slo_lab/harness/trace.py`, `tests/test_trace.py`; modify `src/slo_lab/cli.py`.

**Interfaces (produces):**
- `load_profile(path: Path) -> list[tuple[float, int]]` → `[(0.5, 300), (1.5, 300), (0.5, 900)]` from `burst25.yaml`.
- `phase_bounds(profile) -> list[tuple[str, float, float]]` → `[("pre", 0, 300), ("burst", 300, 600), ("recovery", 600, 1500)]` (names by position: first, the highest multiplier, last).
- `burst_arrivals(rate_ref: float, profile, seed: int, tick_s: float = 0.01) -> list[float]` — first arrival at 0.0, then per phase exponential gaps at `multiplier × rate_ref`; each time floored to the tick; `random.Random(seed)`.
- `write_azure_trace(path, arrivals, *, input_tokens=108, output_tokens=132) -> str` — header + rows `2026-01-01 00:MM:SS.ff,108,132`; returns sha256 of the file.
- CLI `slo-lab make-trace --rate-ref R --seed S --out PATH [--profile config/traffic/burst25.yaml]` prints `{"arrivals": n, "sha256": ..., "phases": [...]}`.

- [x] Tests: deterministic for a seed and different across seeds; first arrival 0.0; all times on the 10 ms grid and sorted; counts per phase within 4 σ of `rate × duration`; written file round-trips through inference-perf's own timestamp rule (keep two fractional digits) to the same offsets; header present.
- [x] Implement, run `uv run --frozen pytest -q tests/test_trace.py`, commit.

### Task A2: Shim connector limit

**Files:** Modify `src/slo_lab/admission/shim.py`; test `tests/test_shim.py`.

- [x] Test: 150 concurrent requests through `Passthrough` reach a fake upstream that holds each one until 150 are in flight (5 s timeout); fails with the default connector (100).
- [x] Implement `aiohttp.TCPConnector(limit=0)` in `_open_session`; tests pass; commit.

### Task A3: Trace-replay config and adapter origin

**Files:** Modify `src/slo_lab/harness/ipf_config.py`, `src/slo_lab/harness/ipf_adapter.py`; tests `tests/test_harness.py`.

**Interfaces:** `trace_replay_config(*, model, base_url, report_dir, trace_file, duration_s, seed, workers=4, timeout_s=300.0, worker_max_concurrency=4096, worker_max_tcp_connections=4096) -> dict` with `data: {type: random, trace: {file, format: AzurePublicDataset}}` and `load: {type: trace_replay, trace: {...}, stages: [{rate, duration}], interval: 0, ...}`; `adapt_file_with_origin(path) -> tuple[list[RequestRecord], float]`.

- [x] Tests: config shape; adapter origin equals min `start_time`; `adapt_file` unchanged.
- [x] Implement; commit.

### Task A4: Shim scraper and the trace stage

**Files:** Create `src/slo_lab/harness/shim_scraper.py`; modify `src/slo_lab/harness/stage.py`, `src/slo_lab/cli.py`; tests `tests/test_harness.py`.

**Interfaces:** `ShimScraper(url, out_path, interval_s=5.0, fetch=...)` writes `t_unix,in_flight,waiting,admitted,completed,upstream_errors,rejected_cap_full,rejected_queue_full,rejected_queue_timeout`; `run_stage(..., kind="trace", trace_file=Path, policy=str, shim_stats_url=str|None, shim_pid=int|None)`; manifest adds `policy`, `trace` (`file`, `sha256`, `arrivals`, `phases`), `clock_offset_s` (unix − monotonic, before and after), `records_origin_monotonic_s`, `shim_final`, `shim_cpu_s`, `phase_summaries`, `time_to_recover_s`, `time_to_recover_attainment_s`.

- [x] Tests: scraper rows from a fake fetch; trace-kind window covers the whole trace with no discard; manifest phase summaries computed from a synthetic record set (uses Task A5 functions).
- [x] Implement; commit.

### Task A5: Admission analysis and reproduce wiring

**Files:** Create `src/slo_lab/admission_analysis.py`, `tests/test_admission_analysis.py`; modify `src/slo_lab/batch_analysis.py`.

**Interfaces:**
- `phase_summary(records, start_s, end_s) -> dict` (offered, met, attainment, attainment_ci95, goodput_rps, rejection_rate, ttft_p95_s, tpot_p95_s over OK records offered in the phase).
- `waiting_series(stage_dir, origin_unix) -> list[tuple[float, float]]` (vLLM waiting + shim waiting, seconds from trace start).
- `time_to_recover(records, waiting, *, burst_end_s, trace_end_s, window_s=60, step_s=5, ttft_slo_s=1.0) -> float | None` — first grid time ≥ burst end with total waiting 0 and TTFT p95 of OK requests offered in `[t, t+60)` ≤ 1 s (spec §3.3); `time_to_recover_attainment(...)` — first t whose offered attainment over `[t, t+60)` ≥ 0.95.
- `analyze_trace(manifests) -> dict` rows per (cell, policy, seed) and `per_cell_policy` means with per-seed paired differences vs `passthrough`; `write_trace_tables(out, result)` → `admission.json`, `tables.md`.
- `batch_analysis.run()` writes admission tables instead of empty closed/open tables when every manifest is a trace stage.

- [x] Tests: hand-built records (a backlog that clears at a known time) give the expected time-to-recover; never-recovering input gives None; cap-style rejections lower attainment but not TTFT; paired differences computed per seed.
- [x] Implement; commit.

### Task A6: Drivers, promotion, compression

**Files:** Create `scripts/wsl/w3-trace-chain.sh`, `scripts/wsl/w3-night.sh`; modify `scripts/wsl/promote-w2.sh`, `scripts/compress_evidence.py`, `tests/test_compress_evidence.py`.

- `w3-trace-chain.sh <cell> <model> <max_num_seqs> <capacity> <rate_ref>`; env `SEEDS` (default `1 2 3`), `PROFILE` (default `config/traffic/burst25.yaml`), `RUN_ROOT` (default `~/vllm-slo-lab/runs-w3`), `PROMOTE` (default 1). Per seed: skip if all three manifests exist; quiet-GPU gate (5 × 30 s); start vLLM; readiness loop identical to `batch.sh`; `make-trace` once; for each policy in the seed's rotated order (seed 1: passthrough, hard_cap, bounded_queue; seed 2: hard_cap, bounded_queue, passthrough; seed 3: bounded_queue, passthrough, hard_cap): wait for vLLM idle (running + waiting = 0, ≤ 600 s), start the shim on 8021, wait for `/_shim/stats`, `run-stage --kind trace` (warm-up 100 for the session's first stage, else 20), stop the shim; stop vLLM; promote to `evidence/raw/w3/<cell>/trace/seed-N` with tag-free session files.
- `w3-night.sh`: smoke (FP8, mini profile 60/60/120 s, passthrough then bounded_queue, `RUN_ROOT=runs-w3-smoke`, promoted to `evidence/raw/w3/smoke/`), then FP8 chain, then BF16 chain, then `analyze-batch` into `analysis/tables/w3-*`.
- [x] `bash -n` both; DRY mode prints the plan; commit.

### Task A7: CPU dry run in WSL against a fake vLLM

**Files:** Create `scripts/wsl/fake_vllm.py` (aiohttp: `/v1/models`, streaming `/v1/completions` with 132 chunks at 2 ms, at most N in service with FIFO waiting, `/metrics` exposing `vllm:num_requests_running` / `vllm:num_requests_waiting`), `scripts/wsl/w3-dryrun.sh`.

- [x] Run a 30 s mini trace at 40 rps through `passthrough` and `hard_cap` (C = 8) against the fake; check: request count = trace rows, 429s become `rejected_429` records, `shim.csv` and `metrics.csv` have rows, phases and time-to-recover present in the manifest, promotion writes gz and `home paths clean`.
- [x] Fix whatever breaks; commit.

### Task A8: ADR 0012, preregistration row, docs

- [x] `docs/decisions/0012-w3-protocol.md` with every choice above; preregistration row "W3 協定（ADR 0012）"; runbook W3 section; commit. Full suite, ruff, audit, reproduce determinism.

## Part B: overnight run (GPU window, 2026-09-11 night)

### Task B0: Pre-flight (when the user says 開始)

- [ ] `MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-bench -- bash /mnt/d/AI-Portfolio/CC_github部隊/vllm-single-gpu-slo-lab/scripts/wsl/preflight.sh` → GPU idle, quiet-gpu ok, weights cached for `Qwen/Qwen3-8B-FP8` and `Qwen/Qwen3-8B`, disk free > 50 GB, no leftover vLLM or shim.
- [ ] Start the Windows VRAM sampler: `powershell -File scripts\win\vram-sampler.ps1 -Out <scratch>\win-vram-2026-09-11.log -IntervalSeconds 30 -Samples 2000` in the background.

### Task B1: GPU smoke (≈ 15 min, inside `w3-night.sh`)

- [ ] Checks: both smoke manifests exist; passthrough smoke has zero 429; bounded smoke has 429s with reason `queue_full` or `queue_timeout` in `shim_final`; `shim_cpu_s / load_wall_s` < 0.8; the passthrough pre-phase TPOT p95 is within 15 % of W2's direct 21.84 rps stage (29–31 ms). A failure here stops the chain (`SMOKE_FAILED`) and goes to Task B3.

### Task B2: Main chain

- [ ] Launch: `run-logged.sh w3-night.sh` (ext4 log, `runs-w3/w3-night-latest.log`), watcher on `watch-night.sh`.
- [ ] Expected timing (from W2's measured wrap-up of about 10 s per 1,000 requests):

| block | stages | ≈ time |
|---|---|---|
| smoke | 2 × 4 min + start | 0.3 h |
| FP8 | 3 sessions × 3 traces × (25 min + 8 min) | 5.3 h |
| BF16 | 3 sessions × 3 traces × (25 min + 3 min) | 4.7 h |
| buffer | one re-run | 1.0 h |

### Task B3: Monitoring and failure playbook

- [ ] After each stage: manifest exists, `inference_perf_returncode` 0, records ≈ trace rows, `phase_summaries` present, probe TPOT within 15 % of the session's first stage, Windows committed < physical.
- [ ] Failures: server not ready → re-run the chain (stages with manifests are skipped); shim died → the stage has 502s: quarantine the stage dir and re-run; quiet-gpu refused 5 times → wait 10 min and relaunch; suspect stage → quarantine to `runs-w3/quarantine/`, re-run at the end of the cell (max 2 rounds).

### Task B4: Wrap-up

- [ ] `reproduce-lite` twice (hash-identical), ruff, pytest, audit; ADR 0013 W3 results; README status block; claims audit rows; plan run log; commit.
- [ ] Control tower ledger and registry, memory snapshot, dashboard `<li>`; final report leads with the admission table.

## Run log (appended during execution)

| time | event |
|---|---|
| 09-11 afternoon | Part A done (commits `0ca22ae`..`4c29da1`, ADR 0012). Findings before any data: inference-perf multistage stages drain between stages, so W3 uses trace replay; the shim's upstream pool was capped at 100 connections (fixed); the WSL2 wall clock drifted ~7 s per 2.5-min dry-run stage, so queues are aligned on `t_mono`. |
| 09-11 afternoon | CPU dry run against `fake_vllm.py`: records = trace rows for all three policies; 530 (hard_cap) and 376 (bounded_queue) rejections recorded as `rejected_429`; first request 2.8–19 s after launch; promotion gzip, home paths clean; admission table rebuilt from evidence. |
| 03:47 | Task B0: machine rebooted 03:43, GPU 573 MiB / 7 %, quiet-gpu ok, weights cached, 250 GB free, no leftovers; IO pressure elevated after the reboot (avg10 67 %), not a no-go. Windows VRAM sampler started (30 s, 1,800 samples). |
| 03:49 | Chain launched (`run-logged.sh w3-night.sh`). FP8 smoke server ready 03:52 (weights 70 s, KV 76,112 tokens, identical to W2). |
| 04:01 | Smoke attempt 1, passthrough done: every trace row became a record, but 23 errors, 9 of them 502 `Server disconnected` / `Connection reset by peer` from the shim during the backlog (keep-alive reuse racing uvicorn's 5 s idle close). Pre-burst TPOT p95 39 ms through the shim vs 24.6 ms in W2 direct at the same rate. Prefix-cache hits 0 in both W2 and W3, so that is not the cause. The 60 s smoke burst left 1,865 queued and the 120 s smoke recovery could not drain it (about 12 rps spare), which the 900 s burst25 recovery can. |
| 04:08 | Chain stopped before any main-cell stage (no data lost). Shim fixed: one upstream connection per request (`force_close`), handler cancellation on (aiohttp 3.14 already propagated a client disconnect to the queued upstream request in tests; kept as a guard). The smoke now runs a `direct` no-shim control arm first, and the gate fails if the shim adds more than 10 % TPOT p95 or 50 ms TTFT p95 in the pre-burst phase, or if an unrejected arm has any error. Commit `ddc0bd9`. |
| 04:11 | Chain relaunched; first smoke kept at `runs-w3-smoke-attempt1-0410`. |
| 04:20 | Smoke 2, `direct` (no shim) arm: the same pre-burst TPOT p95 (39 ms) and 23 errors as attempt 1, so neither comes from the shim. Diagnosis: (1) inference-perf counts output tokens by re-tokenizing the returned text; with random-token prompts 25 % of requests count 128-131 of 132 (W2 text prompts: 1.45 %), which inflates (e2e - TTFT) / (n - 1); (2) the smoke's pre window [30, 60) ends at the burst, so requests still decoding when it starts carry burst-slowed TPOT; over [0, 60) p95 is 26.7 ms, p50 23.2 ms (W2 24.6 / 22.2, +5-8 %, the probe is +5 %); (3) direct's errors are inference-perf's keep-alive reuse racing uvicorn's 5 s idle close (6 disconnects, 6 resets) plus 11 streams whose every chunk has empty text (EOS first, then special tokens under ignore_eos). |
| 04:26 | Smoke 2, passthrough with the fixed shim: 0 shim-to-vLLM failures, 8 empty-text streams (0.1 %), pre-burst TPOT p50 23.9 vs 23.2 ms direct (+3 %), p95 27.3 vs 26.7 ms. |
| 04:33 | Adapter fixed to take the server's `completion_tokens` (commit `18789bd`); re-adapting the smoke drops direct's [30, 60) TPOT p95 from 39.2 to 33.7 ms. Gate now judges the shim on its own 502 count and allows up to 0.5 % empty-text streams (counted as misses). Main-cell stages start after this commit, so every W3 record uses the server count. |

## Results as they land

## Self-review

- Spec coverage: §3.3 trace shape and time-to-recover (A1, A5); §3.5 three policies through one shim with C/Q/T (A2, A4, A6); §3.6 three paired seeds (A1, A6); W3 deliverables in spec §8 (three-way table, time-to-recover, queue timeline data in `metrics.csv` + `shim.csv`, shim overhead check in B1).
- Consistency: policy names are the shim's (`passthrough`, `hard_cap`, `bounded_queue`) everywhere; config file names (`native`, `cap429`, `bounded`) appear only in `config/admission/`.
