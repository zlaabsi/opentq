# OpenTQ Research/UI Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the KV-cache, quantization-aware pruning, and allocation UI roadmaps into executable OpenTQ artifacts.

**Architecture:** Add three independent CLI surfaces that consume existing OpenTQ dynamic GGUF plans. `kv-cache-plan` emits a per-layer runtime policy, `pruning-candidates` emits reversible offline structured-unit candidates, and `allocation-ui` emits an inspectable tensor map plus a React/Vite dashboard source. A chained overnight launcher waits for the active native run to finish before executing the research/UI run.

**Tech Stack:** Python 3.11, argparse CLI, pytest, JSON/TSV/YAML/HTML artifacts, Vite/React dashboard source.

---

### Task 1: KV Cache Per-Layer Policy

**Files:**
- Create: `src/opentq/kv_cache.py`
- Modify: `src/opentq/cli.py`
- Test: `tests/test_kv_cache.py`

- [x] Add `KVCachePlanOptions`, dtype validation, weight-plan coupling, and writer outputs.
- [x] Add CLI command:

```bash
uv run opentq kv-cache-plan \
  --weight-plan artifacts/qwen36-plan/plan.json \
  --output artifacts/qwen36-kv-policy \
  --default-dtype fp8_e4m3 \
  --promote-dtype bf16
```

- [x] Verify JSON, TSV, and rationale outputs.

### Task 2: Quantization-Aware Pruning Candidates

**Files:**
- Create: `src/opentq/pruning.py`
- Modify: `src/opentq/cli.py`
- Test: `tests/test_pruning.py`

- [x] Rank structured units by tensor family, layer position, assigned precision, and policy rationale.
- [x] Emit `pruning-candidates.json`, `pruning-candidates.jsonl`, `pruning-policy.yaml`, and `paired-pruning-report.md`.
- [x] Keep this reversible and explicitly offline until paired validation exists.

### Task 3: Allocation UI Artifact

**Files:**
- Create: `src/opentq/allocation_ui.py`
- Create: `ui/allocation-dashboard/package.json`
- Create: `ui/allocation-dashboard/index.html`
- Create: `ui/allocation-dashboard/src/main.jsx`
- Create: `ui/allocation-dashboard/src/styles.css`
- Modify: `src/opentq/cli.py`
- Test: `tests/test_allocation_ui.py`

- [x] Generate `allocation-ui-data.json` from `plan.json`.
- [x] Generate a standalone `index.html` for instant local inspection.
- [x] Add React/Vite dashboard source for a richer first-class UI.

### Task 4: Overnight Chain Runner

**Files:**
- Create: `scripts/run_qwen36_overnight_research_ui.sh`
- Create: `scripts/launch_qwen36_research_ui_after_overnight.sh`

- [x] Build a second run that creates a fresh Q4 dynamic plan, KV policy, pruning candidates, and allocation UI.
- [x] Add a launcher that waits for `opentq_overnight_native_20260505T002533Z` to disappear before starting `opentq_overnight_research_ui_<stamp>`.

### Task 5: Verification

- [x] Run targeted pytest for new modules.
- [x] Run CLI smoke commands against an existing dynamic GGUF plan.
- [ ] Let the watcher launch the second overnight run after the current benchmark/native run completes.
