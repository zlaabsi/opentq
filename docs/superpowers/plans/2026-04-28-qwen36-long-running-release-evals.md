# Qwen3.6 Long-Running Release And Evaluation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the Qwen3.6-27B OpenTQ release work without chat-dependent context: publish only validated artifacts, run practical quantized-model quality subsets, report degradation only when defensible, and reclaim disk space only after release evidence exists.

**Architecture:** Treat release, runtime checks, benchmark subsets, degradation reporting, and cleanup as separate gated phases. The canonical stock-compatible GGUF release remains `zlaabsi/Qwen3.6-27B-OTQ-GGUF`; OpenTQ-native Packed and Metal tracks stay gated until their runtimes are public and measured. The benchmark work uses a fixed matrix in `benchmarks/qwen36_long_running_benchmark_matrix.json` so workers cannot silently substitute vague benchmark names or unsupported deltas.

**Tech Stack:** Python/uv, llama.cpp `llama-cli` and `llama-bench`, Hugging Face Hub CLI, OpenTQ CLI, JSONL benchmark suites, local Apple Silicon Metal runtime checks, optional external benchmark harnesses only behind explicit flags.

## Current Status 2026-04-28 23:38 CEST

- GitHub `main` is pushed through `30f11f2`.
- Phase 4 benchmark adapters are implemented in `scripts/run_qwen36_benchmark_subsets.py` for MMLU, MMLU-Pro, ARC, HellaSwag, GSM8K, MATH, AIME, HumanEval, MBPP, BBH, GPQA, IFEval, TruthfulQA, WinoGrande, DROP, PIQA, and CommonsenseQA.
- The practical Q3/Q4 local GGUF subset run is complete at `artifacts/qwen3.6-27b-benchmark-subsets-practical`, with 68 samples per model.
- Practical mini-subset totals: Q3 `39/68` (`57.4%`), Q4 `39/68` (`57.4%`).
- The practical report is generated at `artifacts/qwen3.6-27b-degradation-report-practical/degradation-report.md`. It labels official-baseline comparisons as candidates requiring review and labels non-official rows as OTQ-only unless a BF16 mini sidecar exists.
- No Hugging Face upload was performed in this phase. Keep upload gated on explicit `HF_UPLOAD=1` or a direct upload instruction.
- No deletion was performed. Free disk is about `10 GiB`; `hf cache scan --dir ~/.cache/huggingface/hub` reports `Qwen/Qwen3.6-27B` at `55.6G` and the Xet cache is about `4.7G`. Cleanup still requires manifest review plus explicit deletion enablement.
- Remaining quality gates: SWE-bench and LiveCodeBench need real harness adapters; judge-based MT-Bench, Chatbot Arena, and AlpacaEval need pinned judge setup; MMMU and MathVista remain blocked for the current text-only GGUF release.

---

## Operating Rules

- Continue autonomously through the next phase when the previous command exits successfully.
- Do not ask the user before ordinary read, test, staging, or non-destructive build steps.
- Stop and summarize only when a command needs credentials, paid remote hardware, destructive deletion, a full BF16 benchmark, or a release upload that has not been explicitly enabled by environment variables.
- Do not delete artifacts unless the cleanup manifest marks the exact path as `release_verified` and `ALLOW_DELETE=1` is set.
- Do not claim a quality delta versus BF16 unless the benchmark row is `official_comparable` and the task/split/scoring matches, or a matching mini-BF16 sidecar run exists.
- Do not rename public artifacts to vague labels. Keep `OTQ` for the public brand/namespace, keep `DYN` for dynamic stock-GGUF allocation, and keep technical quant tokens such as `Q3_K_M`, `Q4_K_M`, `TQ3_SB4`, and `TQ4R2`.
- Do not publish `XL` in public artifact names.
- Do not publish `IQ4_NL` without an imatrix/calibration record.

## Official Baseline Snapshot

Source checked on 2026-04-28: `https://huggingface.co/Qwen/Qwen3.6-27B`.

The current local baseline file is `benchmarks/qwen36_official_language_baseline.json`. It records Qwen's official language scores for:

- SWE-bench Verified: `77.2`
- SWE-bench Pro: `53.5`
- SWE-bench Multilingual: `71.3`
- Terminal-Bench 2.0: `59.3`
- SkillsBench Avg5: `48.2`
- QwenWebBench: `1487` Elo
- NL2Repo: `36.2`
- Claw-Eval Avg: `72.4`
- Claw-Eval Pass^3: `60.6`
- QwenClawBench: `53.4`
- MMLU-Pro: `86.2`
- MMLU-Redux: `93.5`
- SuperGPQA: `66.0`
- C-Eval: `91.4`
- GPQA Diamond: `87.8`
- HLE: `24.0`
- LiveCodeBench v6: `83.9`
- HMMT Feb 25: `93.8`
- HMMT Nov 25: `90.7`
- HMMT Feb 26: `84.3`
- IMOAnswerBench: `80.8`
- AIME26: `94.1`

The model card also publishes vision-language scores including MMMU `82.9` and MathVista mini `87.4`. These are tracked in `benchmarks/qwen36_long_running_benchmark_matrix.json` as blocked for the current text-only GGUF release path.

## Benchmark Scope

All benchmark families requested by the user are explicitly classified in `benchmarks/qwen36_long_running_benchmark_matrix.json`:

| Benchmark | Long-running treatment |
| --- | --- |
| MMLU | OTQ subset now; BF16 mini-run required for degradation claim |
| MMLU-Pro | OTQ subset now; official delta allowed only with compatible scoring against `86.2` |
| ARC | OTQ subset now; BF16 mini-run required for degradation claim |
| HellaSwag | OTQ subset now; BF16 mini-run required for degradation claim |
| GSM8K | OTQ subset now; BF16 mini-run required for degradation claim |
| MATH | OTQ subset now; BF16 mini-run required for degradation claim |
| AIME | AIME26-compatible subset only for official delta against `94.1`; otherwise AIME-style OTQ subset |
| HumanEval | OTQ code subset now; BF16 mini-run required for degradation claim |
| MBPP | OTQ code subset now; BF16 mini-run required for degradation claim |
| SWE-bench | 3-task verified subset max locally; official delta only with real SWE-bench harness/split |
| LiveCodeBench | v6-compatible subset only for official delta against `83.9` |
| BIG-Bench Hard (BBH) | OTQ subset now; BF16 mini-run required for degradation claim |
| GPQA | GPQA Diamond-compatible subset only for official delta against `87.8` |
| MT-Bench | Judge-based OTQ sentinel; no degradation claim without pinned judge and BF16 sidecar |
| Chatbot Arena | Local arena-style sentinel only; official Arena cannot be reproduced locally |
| AlpacaEval | Judge-based OTQ sentinel; no degradation claim without pinned judge and BF16 sidecar |
| IFEval | OTQ subset now; BF16 mini-run required for degradation claim |
| MMMU | Blocked until vision-capable artifact/runtime exists |
| MathVista | Blocked until vision-capable artifact/runtime exists |
| TruthfulQA | OTQ subset now; BF16 mini-run required for degradation claim |
| WinoGrande | OTQ subset now; BF16 mini-run required for degradation claim |
| DROP | OTQ subset now; BF16 mini-run required for degradation claim |
| PIQA | OTQ subset now; BF16 mini-run required for degradation claim |
| CommonsenseQA | OTQ subset now; BF16 mini-run required for degradation claim |

## File Structure

- `benchmarks/qwen36_long_running_benchmark_matrix.json`: authoritative benchmark classification, subset policy, official baseline mapping, and claim rule.
- `docs/qwen36-long-running-todo.md`: operator checklist and resume commands.
- `docs/llm-benchmark-protocol.md`: public reporting rules for release gates versus benchmark claims.
- `docs/release-todo.md`: short release page that links to this plan.
- `scripts/run_qwen36_runtime_checks.py`: existing local runtime gate for bounded generation and `llama-bench`.
- `scripts/run_qwen36_otq_eval.sh`: existing deterministic OTQ-only quality micro-suite runner.
- `scripts/stage_qwen36_otq_gguf_repo.py`: existing canonical GGUF staging script.
- `scripts/stage_qwen36_otq_runtime_repos.py`: existing Packed and Metal staging script.
- `scripts/build_qwen36_cleanup_manifest.py`: existing deletion candidate manifest builder.
- `scripts/run_qwen36_benchmark_subsets.py`: create in Phase 3; plans benchmark matrix subsets in `--dry-run` mode and refuses score JSON output until Phase 4 adapters exist.
- `scripts/build_qwen36_degradation_report.py`: create in Phase 5; merges official baselines, OTQ subset runs, optional BF16 mini-runs, and reporting disclaimers.
- `artifacts/qwen3.6-27b-benchmark-subsets/`: generated benchmark subset evidence.
- `artifacts/qwen3.6-27b-degradation-report/`: generated degradation summary for local review and Hugging Face card input.

## Phase 0: Preflight And Resume State

- [ ] **Step 0.1: Confirm repository state**

Run:

```bash
cd /Users/zlaabsi/Documents/GitHub/opentq
git status --short --branch
```

Expected:

```text
## main...origin/main [ahead N]
```

Continue when only intentional local edits are present. If unrelated user edits appear, leave them in place and avoid touching those files.

- [ ] **Step 0.2: Confirm disk headroom before long runs**

Run:

```bash
df -h /Users/zlaabsi
du -sh artifacts/qwen3.6-27b artifacts/hf-runtime artifacts/qwen3.6-27b-source artifacts/hf-gguf-canonical 2>/dev/null
```

Expected:

```text
Filesystem usage is printed, and large artifact directories are visible.
```

Stop if free disk is below `15 GiB`; benchmark writes and staging can fail or corrupt partial outputs.

- [ ] **Step 0.3: Confirm llama.cpp binaries**

Run:

```bash
test -x /Users/zlaabsi/Documents/GitHub/llama.cpp/build/bin/llama-cli
test -x /Users/zlaabsi/Documents/GitHub/llama.cpp/build/bin/llama-bench
/Users/zlaabsi/Documents/GitHub/llama.cpp/build/bin/llama-cli --version
```

Expected:

```text
llama.cpp version information is printed.
```

- [ ] **Step 0.4: Run focused regression tests**

Run:

```bash
uv run pytest tests/test_dynamic_gguf.py tests/test_hf_gguf_release.py tests/test_qwen36_release_audit.py tests/test_qwen36_runtime_checks.py tests/test_qwen36_cleanup_manifest.py tests/test_qwen36_runtime_repos.py tests/test_quality_eval.py -q
```

Expected:

```text
All selected tests pass.
```

## Phase 1: Canonical GGUF Evidence Refresh

- [ ] **Step 1.1: Audit local release artifacts**

Run:

```bash
uv run python scripts/audit_qwen36_release_state.py \
  --root artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF \
  --output artifacts/release-audit/qwen36-release-audit.json
```

Expected:

```text
Audit JSON is written under artifacts/release-audit.
```

Check:

```bash
uv run python - <<'PY'
import json
from pathlib import Path
p = Path("artifacts/release-audit/qwen36-release-audit.json")
d = json.loads(p.read_text())
print("naming_findings=", d.get("naming_findings"))
print("gguf_files=", [row["name"] for row in d.get("gguf_files", [])])
PY
```

Expected:

```text
naming_findings= []
gguf_files= ['Qwen3.6-27B-OTQ-DYN-Q3_K_M.gguf', 'Qwen3.6-27B-OTQ-DYN-Q4_K_M.gguf']
```

- [ ] **Step 1.2: Re-run M1 Max runtime gates for stock GGUFs**

Run:

```bash
for quant in Q3_K_M Q4_K_M; do
  uv run python scripts/run_qwen36_runtime_checks.py \
    --model "artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF/Qwen3.6-27B-OTQ-DYN-${quant}.gguf" \
    --machine "M1 Max 32GB" \
    --llama-cpp /Users/zlaabsi/Documents/GitHub/llama.cpp \
    --output "artifacts/release-audit/runtime-${quant}-M1_Max_32GB.json"
done
```

Expected:

```text
Each runtime JSON reports bounded_generation_passed=true and bench_passed=true.
```

- [ ] **Step 1.3: Re-stage canonical HF GGUF repo**

Run:

```bash
uv run python scripts/stage_qwen36_otq_gguf_repo.py \
  --output artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF-next \
  --link-mode hardlink
```

Expected:

```text
README.md, BENCHMARKS.md, GGUF files, release JSON, evidence, assets, and benchmark CSVs are written.
```

- [ ] **Step 1.4: Check public naming**

Run:

```bash
rg -n "Q[345]_XL|OTQ3|OTQ4|_XL|XL" artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF-next \
  -g '!*.svg' -g '!*.png' -g '!*.pdf' -g '!*.json'
```

Expected:

```text
No matches.
```

## Phase 2: Existing OTQ Micro-Eval Refresh

- [ ] **Step 2.1: Run deterministic OTQ release suite**

Run:

```bash
LLAMA_CPP_DIR=/Users/zlaabsi/Documents/GitHub/llama.cpp \
OUTPUT_ROOT=artifacts/qwen3.6-27b-otq-eval \
SUITE=benchmarks/qwen36_release_extended_samples.jsonl \
CTX_SIZE=8192 \
NGL=99 \
FLASH_ATTN=on \
PROMPT_FORMAT=qwen3-no-think \
TIMEOUT=1800 \
./scripts/run_qwen36_otq_eval.sh
```

Expected:

```text
artifacts/qwen3.6-27b-otq-eval/OTQ-DYN-Q3_K_M.json
artifacts/qwen3.6-27b-otq-eval/OTQ-DYN-Q4_K_M.json
```

- [ ] **Step 2.2: Rebuild report plots and CSVs**

Run:

```bash
uv run python scripts/build_qwen36_release_report.py \
  --repo artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF-next \
  --quant-eval-root artifacts/qwen3.6-27b-otq-eval
```

Expected:

```text
wrote publication-grade benchmark report assets under artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF-next
```

## Phase 3: Benchmark Subset Planner

- [ ] **Step 3.1: Create benchmark subset dry-run planner**

Create `scripts/run_qwen36_benchmark_subsets.py` with these responsibilities:

- Load `benchmarks/qwen36_long_running_benchmark_matrix.json`.
- Accept `--models q3,q4`, `--matrix`, `--output-root`, `--llama-cpp`, `--sample-mode quick`, `--dry-run`, and `--allow-judge`.
- Resolve stock GGUF paths:
  - `q3`: `artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF/Qwen3.6-27B-OTQ-DYN-Q3_K_M.gguf`
  - `q4`: `artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF/Qwen3.6-27B-OTQ-DYN-Q4_K_M.gguf`
- Skip `blocked_modality` rows and record the blocked reason in the output JSON.
- Skip `judge_based` rows unless `--allow-judge` is present.
- For `mini_bf16_required` and `official_comparable` rows, print the fixed-limit sample counts and claim rules.
- Refuse non-`--dry-run` execution with a clear message until Phase 4 adapters exist. This prevents fake score JSONs.

Initial command:

```bash
uv run python scripts/run_qwen36_benchmark_subsets.py \
  --matrix benchmarks/qwen36_long_running_benchmark_matrix.json \
  --models q3,q4 \
  --output-root artifacts/qwen3.6-27b-benchmark-subsets \
  --llama-cpp /Users/zlaabsi/Documents/GitHub/llama.cpp \
  --sample-mode quick \
  --dry-run
```

Expected:

```text
The runner prints the selected benchmark families, planned sample counts, skipped judge rows, and blocked multimodal rows.
```

- [ ] **Step 3.2: Add tests for benchmark matrix loading**

Create `tests/test_qwen36_benchmark_matrix.py` with checks that:

- all user-requested benchmark ids are present;
- every row has `comparison_mode`, `subset_policy`, and `claim_rule`;
- `mmmu` and `mathvista` are `blocked_modality`;
- `mmlu_pro`, `gpqa`, `aime`, `swe_bench`, and `livecodebench` contain official baselines;
- no row permits a BF16 delta without `official_comparable` or a BF16 sidecar.

Run:

```bash
uv run pytest tests/test_qwen36_benchmark_matrix.py -q
```

Expected:

```text
All benchmark matrix tests pass.
```

## Phase 4: Benchmark Subset Runs

- [ ] **Step 4.1: Implement benchmark execution adapters**

Extend `scripts/run_qwen36_benchmark_subsets.py` before running this phase:

- Add task adapters for multiple-choice, exact numeric answer, code execution, and rule-based instruction-following samples.
- Record `task_ids`, `dataset_revision`, `prompt_format`, `scoring_rule`, `temperature`, `ctx_size`, `max_tokens`, `runtime_seconds`, and `passed`.
- Keep judge-based and multimodal rows skipped unless their explicit gates are enabled.
- Write one score JSON per model under `artifacts/qwen3.6-27b-benchmark-subsets/<model>.json`.

Run:

```bash
uv run pytest tests/test_qwen36_benchmark_matrix.py -q
```

Expected:

```text
All benchmark matrix and runner tests pass.
```

- [ ] **Step 4.2: Run quick OTQ benchmark subsets**

Run:

```bash
uv run python scripts/run_qwen36_benchmark_subsets.py \
  --matrix benchmarks/qwen36_long_running_benchmark_matrix.json \
  --models q3,q4 \
  --output-root artifacts/qwen3.6-27b-benchmark-subsets \
  --llama-cpp /Users/zlaabsi/Documents/GitHub/llama.cpp \
  --sample-mode quick
```

Expected:

```text
artifacts/qwen3.6-27b-benchmark-subsets/q3.json
artifacts/qwen3.6-27b-benchmark-subsets/q4.json
```

- [ ] **Step 4.3: Run optional judge-based sentinels only when configured**

Run only if a local judge endpoint and judge prompt are pinned:

```bash
ALLOW_JUDGE=1 \
JUDGE_BASE_URL=http://localhost:8000/v1 \
JUDGE_MODEL=local-judge \
uv run python scripts/run_qwen36_benchmark_subsets.py \
  --matrix benchmarks/qwen36_long_running_benchmark_matrix.json \
  --models q3,q4 \
  --output-root artifacts/qwen3.6-27b-benchmark-subsets \
  --llama-cpp /Users/zlaabsi/Documents/GitHub/llama.cpp \
  --sample-mode quick \
  --allow-judge
```

Expected:

```text
Judge-based rows are marked as local sentinels, not official benchmark deltas.
```

- [ ] **Step 4.4: Run optional BF16 mini sidecar only with explicit opt-in**

Run only when remote or local BF16 capacity is explicitly available:

```bash
RUN_REMOTE_BF16=1 \
BF16_MODEL=Qwen/Qwen3.6-27B \
uv run python scripts/run_qwen36_benchmark_subsets.py \
  --matrix benchmarks/qwen36_long_running_benchmark_matrix.json \
  --models bf16 \
  --output-root artifacts/qwen3.6-27b-benchmark-subsets \
  --sample-mode quick
```

Expected:

```text
BF16 subset evidence is written, with hardware/backend details and matching task ids.
```

Stop instead of running this command if it would require paid remote hardware without explicit approval.

## Phase 5: Degradation Report

- [ ] **Step 5.1: Create report builder**

Create `scripts/build_qwen36_degradation_report.py` with these responsibilities:

- Load the matrix from `benchmarks/qwen36_long_running_benchmark_matrix.json`.
- Load official language baseline from `benchmarks/qwen36_official_language_baseline.json`.
- Load OTQ subset JSONs from `artifacts/qwen3.6-27b-benchmark-subsets/`.
- Load optional BF16 sidecar JSON from the same output root when present.
- For `official_comparable` rows with compatible subset metadata, compute `otq_score - official_score`.
- For rows with a BF16 mini sidecar, compute `otq_score - bf16_mini_score`.
- For all other rows, render `no_delta_claim`.
- Render Markdown and JSON to `artifacts/qwen3.6-27b-degradation-report/`.

Run:

```bash
uv run python scripts/build_qwen36_degradation_report.py \
  --matrix benchmarks/qwen36_long_running_benchmark_matrix.json \
  --official-baseline benchmarks/qwen36_official_language_baseline.json \
  --subset-root artifacts/qwen3.6-27b-benchmark-subsets \
  --output-root artifacts/qwen3.6-27b-degradation-report
```

Expected:

```text
degradation-report.json and degradation-report.md are written.
```

- [ ] **Step 5.2: Verify no unsupported delta claims**

Run:

```bash
rg -n "delta|degradation|BF16|official" artifacts/qwen3.6-27b-degradation-report/degradation-report.md
```

Expected:

```text
Rows without compatible official or BF16 mini baselines say no_delta_claim.
```

## Phase 6: Q5_K_M Decision

- [ ] **Step 6.1: Check disk before generating Q5**

Run:

```bash
df -h /Users/zlaabsi
```

Expected:

```text
At least 80 GiB free before starting another full GGUF quantization.
```

Skip Q5 generation when free disk is below `80 GiB`.

- [ ] **Step 6.2: Generate Q5 only when storage is sufficient**

Run only if Step 6.1 passes:

```bash
QWEN36_DYNAMIC_PROFILES=Q5_K_M \
./scripts/launch_qwen36_dynamic_ggufs.sh
```

Expected:

```text
Q5_K_M conversion job starts and writes logs under artifacts.
```

- [ ] **Step 6.3: Gate Q5 before staging**

Run:

```bash
QWEN36_DYNAMIC_PROFILES=Q5_K_M ./scripts/status_qwen36_dynamic_ggufs.sh
```

Expected:

```text
Q5_K_M shows conversion, validation, quality eval, release eval, and benchmark evidence.
```

Do not add Q5 to public HF staging until validation, quality, release, and runtime benchmark gates all pass.

## Phase 7: OTQ-Packed Gate And Upload Staging

- [ ] **Step 7.1: Re-stage runtime repos locally**

Run:

```bash
uv run python scripts/stage_qwen36_otq_runtime_repos.py \
  --output-root artifacts/hf-runtime
```

Expected:

```text
Qwen3.6-27B-OTQ-Packed and Qwen3.6-27B-OTQ-Metal-GGUF staging directories are refreshed.
```

- [ ] **Step 7.2: Inspect Packed release status**

Run:

```bash
sed -n '1,220p' artifacts/hf-runtime/Qwen3.6-27B-OTQ-Packed/README.md
```

Expected:

```text
README states the runtime requirements and does not imply stock llama.cpp compatibility.
```

- [ ] **Step 7.3: Upload Packed only when public runtime evidence exists**

Run only if the README, manifest, and runtime gate explicitly mark the Packed runtime as public and passing:

```bash
HF_UPLOAD=1 hf upload zlaabsi/Qwen3.6-27B-OTQ-Packed artifacts/hf-runtime/Qwen3.6-27B-OTQ-Packed .
```

Expected:

```text
Hugging Face upload completes.
```

Stop if the repo does not yet exist, authentication is missing, or runtime evidence is incomplete.

## Phase 8: OTQ-Metal-GGUF Gate

- [ ] **Step 8.1: Inspect Metal staged repo**

Run:

```bash
sed -n '1,240p' artifacts/hf-runtime/Qwen3.6-27B-OTQ-Metal-GGUF/README.md
```

Expected:

```text
README clearly states that stock llama.cpp is not the target runtime.
```

- [ ] **Step 8.2: Run Metal runtime only when loader/kernels are release-grade**

Run from the OpenTQ Metal repo when its loader and kernels are complete:

```bash
cd /Users/zlaabsi/Documents/GitHub/opentq-metal
git status --short --branch
```

Expected:

```text
Metal repo status is visible and release runtime commands can be selected from that repo.
```

Stop if the Metal repo still documents incomplete loader/kernels.

- [ ] **Step 8.3: Upload Metal only when runtime gate passes**

Run only after Step 8.2 produces passing generation, long-context, and benchmark evidence:

```bash
HF_UPLOAD=1 hf upload zlaabsi/Qwen3.6-27B-OTQ-Metal-GGUF artifacts/hf-runtime/Qwen3.6-27B-OTQ-Metal-GGUF .
```

Expected:

```text
Hugging Face upload completes.
```

Stop if authentication is missing or runtime evidence is incomplete.

## Phase 9: Cleanup Manifest And Disk Reclaim

- [ ] **Step 9.1: Build cleanup manifest**

Run:

```bash
uv run python scripts/build_qwen36_cleanup_manifest.py \
  --output artifacts/release-audit/qwen36-cleanup-manifest.json
```

Expected:

```text
Cleanup manifest is written and defaults risky paths to investigate.
```

- [ ] **Step 9.2: Inspect cleanup decisions**

Run:

```bash
uv run python - <<'PY'
import json
from pathlib import Path
p = Path("artifacts/release-audit/qwen36-cleanup-manifest.json")
d = json.loads(p.read_text())
for item in d:
    print(item.get("classification"), round(item.get("bytes", 0) / 1024**3, 2), item.get("path"))
PY
```

Expected:

```text
Every large path is printed with a decision.
```

- [ ] **Step 9.3: Delete only release-verified paths with explicit opt-in**

Run only when the manifest has been manually reviewed, the HF release has been checked, and `ALLOW_DELETE=1` is set:

```bash
ALLOW_DELETE=1 rm -ri <reviewed-release-verified-path>
```

Expected:

```text
Only reviewed release-verified paths are removed.
```

Stop before running if any selected path contains the only local copy of an unpublished artifact.

## Phase 10: Final Release Report

- [ ] **Step 10.1: Capture final repo state**

Run:

```bash
git status --short --branch
git log --oneline -5
```

Expected:

```text
Local changes and recent commits are visible.
```

- [ ] **Step 10.2: Capture HF artifact inventory**

Run:

```bash
hf repo files zlaabsi/Qwen3.6-27B-OTQ-GGUF | sort
```

Expected:

```text
Remote file list includes README, BENCHMARKS, GGUF files, evidence, assets, and manifests.
```

- [ ] **Step 10.3: Write final local report**

Create `artifacts/release-audit/qwen36-final-release-report.md` with:

- Git commit hash.
- HF repos touched.
- Local runtime checks and hardware.
- GGUF variants released.
- Packed and Metal status.
- Benchmark subset summary.
- Degradation claims allowed and blocked.
- Cleanup manifest decisions.
- Remaining manual blockers.

Run:

```bash
sed -n '1,260p' artifacts/release-audit/qwen36-final-release-report.md
```

Expected:

```text
The report can be read without chat context.
```

## Resume Commands

Use these when the session resumes after compaction or a machine restart:

```bash
cd /Users/zlaabsi/Documents/GitHub/opentq
git status --short --branch
sed -n '1,260p' docs/qwen36-long-running-todo.md
sed -n '1,260p' docs/superpowers/plans/2026-04-28-qwen36-long-running-release-evals.md
```

Then continue from the first unchecked phase in `docs/qwen36-long-running-todo.md`.
