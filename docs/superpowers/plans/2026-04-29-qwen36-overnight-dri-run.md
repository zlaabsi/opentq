# Qwen3.6 Overnight DRI Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the remaining safe Qwen3.6 validation work overnight with Codex as Direct Responsible Individual, producing local evidence and a morning-ready summary without publishing unready artifacts.

**Architecture:** The overnight runner is a detached, caffeinated shell pipeline that writes all evidence into a timestamped artifact directory. It uses the existing benchmark runner for LiveCodeBench and SWE-bench, keeps external-harness rows separate from score claims, and treats Packed/Metal as local staging checks only.

**Tech Stack:** Bash 3 compatible scripts, uv, pytest, ruff, llama.cpp through `scripts/run_qwen36_benchmark_subsets.py`, Hugging Face Dataset Viewer/raw dataset fetches, local artifact logs.

---

## Ownership

- DRI: Codex owns launch, monitoring commands, evidence collection, and morning summary.
- Repository owner: `zlaabsi/opentq`.
- Runtime owner: local M1 Max 32 GB machine.
- Publication owner: no publication is allowed from this overnight run unless a later explicit instruction overrides this plan.

## File Structure

- Create `scripts/run_qwen36_overnight_remaining.sh`: unattended pipeline with preflight, LiveCodeBench full subset, SWE patch generation, Packed/Metal restage sanity, and summary generation.
- Create `scripts/launch_qwen36_overnight_remaining.sh`: detached launcher with `nohup` and `caffeinate`.
- Create `docs/qwen36-overnight-todo.md`: concise operator checklist and morning resume commands.
- Create `docs/superpowers/plans/2026-04-29-qwen36-overnight-dri-run.md`: this implementation plan and ownership record.
- Modify `tests/test_shell_scripts.py`: add bash syntax checks for the two new scripts.

## Task 1: Add Overnight Runner

**Files:**
- Create: `scripts/run_qwen36_overnight_remaining.sh`
- Create: `scripts/launch_qwen36_overnight_remaining.sh`

- [ ] **Step 1: Add `scripts/run_qwen36_overnight_remaining.sh`**

The script must:

- create `artifacts/qwen3.6-27b-overnight/<timestamp>/`;
- write per-step logs in `logs/`;
- write `RUN_SUMMARY.md`;
- stop before heavy work when disk is below `25 GiB`;
- run no upload and no deletion command;
- run LiveCodeBench v6 Q3/Q4 with 12 tasks;
- run SWE-bench patch generation with `--allow-external-harness` but no synthetic pass/fail claim.

- [ ] **Step 2: Add `scripts/launch_qwen36_overnight_remaining.sh`**

The launcher must start the runner with `nohup`, write `overnight.pid`, write `caffeinate.pid`, and print the log and summary paths.

- [ ] **Step 3: Verify shell syntax**

Run:

```bash
bash -n scripts/run_qwen36_overnight_remaining.sh
bash -n scripts/launch_qwen36_overnight_remaining.sh
```

Expected:

```text
No output and exit code 0 for both commands.
```

## Task 2: Add Overnight TODO And Test Coverage

**Files:**
- Create: `docs/qwen36-overnight-todo.md`
- Modify: `tests/test_shell_scripts.py`

- [ ] **Step 1: Add operator TODO**

Create `docs/qwen36-overnight-todo.md` with:

- DRI set to Codex;
- launch command;
- morning resume command;
- hard stop conditions;
- explicit no-upload/no-delete/no-BF16/no-Packed-Metal-publication policy.

- [ ] **Step 2: Extend shell script tests**

Add tests that run:

```bash
bash -n scripts/run_qwen36_overnight_remaining.sh
bash -n scripts/launch_qwen36_overnight_remaining.sh
```

Expected:

```text
Both tests pass.
```

## Task 3: Validate Before Launch

**Files:**
- Read: `scripts/run_qwen36_benchmark_subsets.py`
- Read: `tests/test_qwen36_benchmark_matrix.py`

- [ ] **Step 1: Run focused tests**

Run:

```bash
uv run pytest tests/test_qwen36_benchmark_matrix.py tests/test_qwen36_runtime_repos.py tests/test_shell_scripts.py -q
```

Expected:

```text
All selected tests pass.
```

- [ ] **Step 2: Run lint**

Run:

```bash
uv run ruff check scripts/run_qwen36_benchmark_subsets.py tests/test_qwen36_benchmark_matrix.py
```

Expected:

```text
All checks passed!
```

- [ ] **Step 3: Run benchmark dry-run**

Run:

```bash
uv run python scripts/run_qwen36_benchmark_subsets.py --dry-run --models q3,q4 --sample-mode quick
```

Expected:

```text
LiveCodeBench is planned, SWE-bench is requires_external_harness, judge rows are skipped, and multimodal rows are blocked.
```

## Task 4: Launch Overnight Run

**Files:**
- Execute: `scripts/launch_qwen36_overnight_remaining.sh`
- Output: `artifacts/qwen3.6-27b-overnight/<timestamp>/RUN_SUMMARY.md`

- [ ] **Step 1: Launch detached run**

Run:

```bash
./scripts/launch_qwen36_overnight_remaining.sh
```

Expected:

```text
run_root=artifacts/qwen3.6-27b-overnight/<timestamp>
pid=<pid>
caffeinate_pid=<pid>
log=artifacts/qwen3.6-27b-overnight/<timestamp>/overnight.log
summary=artifacts/qwen3.6-27b-overnight/<timestamp>/RUN_SUMMARY.md
```

- [ ] **Step 2: Confirm process is alive**

Run:

```bash
ps -p "$(cat artifacts/qwen3.6-27b-overnight/*/overnight.pid | tail -n 1)" -o pid,stat,etime,command
```

Expected:

```text
The overnight bash process is listed, or the process has already exited after writing RUN_SUMMARY.md.
```

## Task 5: Morning Review

**Files:**
- Read: `artifacts/qwen3.6-27b-overnight/<timestamp>/RUN_SUMMARY.md`
- Read: `artifacts/qwen3.6-27b-overnight/<timestamp>/logs/*.log`

- [ ] **Step 1: Read run summary**

Run:

```bash
cat artifacts/qwen3.6-27b-overnight/*/RUN_SUMMARY.md | tail -n 160
```

Expected:

```text
The summary lists pass/fail for preflight, LiveCodeBench subset results, SWE-bench patch evidence paths, final git status, and final disk status.
```

- [ ] **Step 2: Apply publication rules**

Use these decisions:

- LiveCodeBench can be added to local reports only if Q3 and Q4 JSONs exist and every result has a real `score`.
- SWE-bench patch JSONs are evidence only; do not publish pass/fail without the official harness.
- Packed and Metal remain blocked if staged manifests say `public_release_ready=false`.
- No BF16 degradation claim is allowed without a matching BF16 mini sidecar or officially compatible scoring.

## Self-Review

- Spec coverage: the plan covers remaining benchmark work, SWE-bench limits, LiveCodeBench full subset, Packed/Metal gating, disk safety, and morning handoff.
- Placeholder scan: no task uses a vague placeholder; each command and expected result is explicit.
- Type consistency: script names, artifact paths, and benchmark ids match the current repo.
