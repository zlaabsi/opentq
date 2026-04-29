# Qwen3.6 Disk Cleanup Arbitrage

This note records the cleanup decision for the Qwen3.6 release workspace.

## Current Constraint

As of the post-Q5/post-HF-refresh audit on 2026-04-29, the Mac has about
`99 GiB` free. That is enough for normal repo work, report refreshes, and
small benchmark runs, but it is still not enough to treat all local artifacts
as disposable. The cleanup manifest is therefore non-destructive by default.

The relevant thresholds are:

- below `25 GiB`: stop before heavy benchmark work;
- below `80 GiB`: do not start another full GGUF quantization;
- above `80 GiB`: acceptable for small benchmark/report work, but still review
  hardlinks before deleting large release directories.

## Options

| Option | Immediate disk gain | Future option value | Regret if wrong | Decision |
| --- | ---: | --- | --- | --- |
| Keep all Hugging Face caches | 0 GiB | Keeps local safetensors cache for possible BF16 work | Can block large local work when disk is low | Reject only under disk pressure |
| Delete local BF16 GGUF source | ~50 GiB | Removes the source needed for re-quantization | High: rework becomes harder and more expensive | Reject |
| Delete regenerable HF caches only | Small after previous cleanup | Requires re-download if needed later | Low to moderate: network time only | Accept only under disk pressure |
| Delete staged GGUF release artifacts | 30-50 GiB apparent per staging tree | Public HF copy exists, but local hardlinks make actual reclaim non-obvious | Moderate until HF inventory is repeated | Candidate after HF re-verification |
| Delete Packed/Metal/native artifacts | 16-168 GiB apparent | Removes unreleased runtime evidence and payloads | High until public runtime gates pass | Reject for now |

## Chosen Strategy

Current strategy:

- generate `artifacts/release-audit/qwen36-cleanup-manifest.json`;
- do not delete anything automatically;
- delete only `regenerable-cache` paths if disk pressure returns;
- require a fresh HF inventory check before deleting `uploaded-verified`
  staging directories;
- preserve Packed, Metal, and native artifacts until their runtime/publication
  gates are explicitly resolved.

Preserved anchors:

- `artifacts/qwen3.6-27b-source/Qwen3.6-27B-BF16.gguf`;
- `artifacts/hf-gguf-canonical/Qwen3.6-27B-OTQ-GGUF`;
- `artifacts/qwen3.6-27b-dynamic-gguf`;
- `artifacts/hf-runtime` until Packed/Metal gates are reviewed.

## Latest Manifest

Regenerate with:

```bash
uv run python scripts/build_qwen36_cleanup_manifest.py
```

The latest local audit classified the large paths as:

| Classification | Action | Approx local disk bytes | Meaning |
| --- | --- | ---: | --- |
| `blocked` | preserve | 50 GiB | BF16 source anchor; do not delete. |
| `gated-local-artifact` | preserve | 354 GiB | Packed, Metal, native/raw artifacts; not public-release-ready. |
| `uploaded-verified` | candidate after HF re-verification | 100 GiB | Public/reproducibility staging exists, but verify remote inventory first. |
| `regenerable` | candidate after local rebuild check | 30 GiB | Local staging that can be rebuilt, but not automatically deleted. |
| `regenerable-cache` | delete only under disk pressure | <1 GiB | HF cache leftovers; low option value. |

The manifest records `apparent_bytes`, `disk_bytes`, inode data, and maximum
file hardlink count. Treat `disk_bytes` as an upper bound when hardlinks exist
outside the inspected path.

## Rationale

This is a minimax/regret decision. Cache deletion has low regret when disk is
low because caches can be re-downloaded. Deleting the BF16 source, native
payloads, Packed artifacts, or Metal artifacts has high regret because those
paths preserve release evidence and future runtime work. Public GGUF staging is
a cleanup candidate only after a fresh HF inventory check confirms the remote
contains the same files, checksums, and evidence.
