# Qwen3.6 Overnight DRI TODO

DRI: Codex.

This checklist is for the unattended overnight run started from `scripts/launch_qwen36_overnight_remaining.sh`. It is intentionally conservative: no upload, no deletion, no BF16 sidecar, no Packed/Metal publication, and no synthetic SWE-bench score.

## Run Contract

- [ ] Preflight Git state, lint, focused tests, and benchmark dry-run.
- [ ] Verify pinned LiveCodeBench v6 row fetch from `livecodebench/code_generation_lite` `test6.jsonl`.
- [ ] Verify pinned SWE-bench Verified row fetch from `princeton-nlp/SWE-bench_Verified`.
- [ ] Stop before heavy work if free disk is below `25 GiB`.
- [ ] Run LiveCodeBench lite v6 subset on Q3 and Q4 with 12 pinned tasks each.
- [ ] Generate SWE-bench Verified patch candidates for Q3 and Q4 only; do not report them as pass/fail.
- [ ] Re-stage Packed/Metal runtime repos locally for manifest sanity; do not upload them.
- [ ] Write `RUN_SUMMARY.md` under the run root with evidence paths and remaining blockers.

## Overnight Launch

```bash
cd /Users/zlaabsi/Documents/GitHub/opentq
./scripts/launch_qwen36_overnight_remaining.sh
```

## Morning Resume

```bash
cd /Users/zlaabsi/Documents/GitHub/opentq
cat artifacts/qwen3.6-27b-overnight/*/RUN_SUMMARY.md | tail -n 120
git status --short --branch
df -h .
```

## Hard Stop Conditions

- Free disk below `25 GiB` before LiveCodeBench or SWE generation.
- Focused tests fail.
- Dataset pins cannot be fetched.
- Any command tries to upload, delete, run BF16 sidecar, or publish Packed/Metal.
