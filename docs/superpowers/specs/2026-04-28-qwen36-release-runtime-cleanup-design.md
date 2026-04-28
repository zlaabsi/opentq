# Qwen3.6 OpenTQ Release, Runtime, And Cleanup Design

## Goal

Finish the Qwen3.6-27B OpenTQ release work without ambiguous names, unverifiable runtime claims, or unsafe disk cleanup. The release must separate stock-compatible GGUFs, OpenTQ packed payloads, and OpenTQ Metal/custom-runtime GGUFs.

## Scope

This design covers:

- local runtime validation for currently staged Qwen3.6-27B release artifacts;
- Hugging Face model-card updates for compatibility, hardware guidance, and evidence;
- publication gating for `OTQ-Packed` and `OTQ-Metal-GGUF`;
- a deletion manifest for reclaiming Mac storage only after upload and verification;
- optional Claude Code delegation for bounded, reviewable tasks.

This design does not cover:

- running BF16 baseline benchmarks locally just to reproduce official Qwen scores;
- publishing custom OpenTQ GGUFs as stock `llama.cpp` GGUFs;
- deleting unique artifacts before the corresponding release has been uploaded and verified.

## Naming Contract

Public names must keep the brand namespace separate from the technical quantization format.

- `OTQ` means OpenTQ brand, release namespace, and toolchain.
- `TQ*` means native OpenTQ/TurboQuant-style weight format or profile.
- `DYN` means dynamic tensor-level allocation using stock GGUF tensor types.
- `Q3_K_M`, `Q4_K_M`, `Q5_K_M`, and `IQ4_NL` are stock `llama.cpp`/GGUF quant names, not OpenTQ-native names.
- `XL` is not allowed in public filenames, repo names, README sections, generated plots, or release tables.

Canonical naming:

- Stock GGUF repo: `zlaabsi/Qwen3.6-27B-OTQ-GGUF`.
- Stock GGUF files: `Qwen3.6-27B-OTQ-DYN-Q3_K_M.gguf`, `Qwen3.6-27B-OTQ-DYN-Q4_K_M.gguf`, and future stock files using the same pattern.
- Packed repo: `zlaabsi/Qwen3.6-27B-OTQ-Packed`.
- Packed public folders: `Qwen3.6-27B-OTQ-TQ3_SB4`, `Qwen3.6-27B-OTQ-TQ4_SB4`, `Qwen3.6-27B-OTQ-TQ4R2`, `Qwen3.6-27B-OTQ-TQ4R4`, `Qwen3.6-27B-OTQ-TQ4_BAL_V2`.
- Metal/custom-runtime repo: `zlaabsi/Qwen3.6-27B-OTQ-Metal-GGUF`.
- Custom GGUF tensor types: `OPENTQ_TQ3_SB4`, `OPENTQ_TQ4_SB4`, `OPENTQ_TQ4R2`, `OPENTQ_TQ4R4`, `OPENTQ_TQ4_BAL_V2` if represented as a profile marker.

`IQ4_NL` may remain in docs only as a stock `llama.cpp` nonlinear 4-bit experiment. It is not a primary release target unless an imatrix calibration artifact and runtime evidence exist.

## Release Tracks

### Track 1: Stock-Compatible GGUF

The current public release track is `Qwen3.6-27B-OTQ-GGUF`. It may contain only standard GGUF tensor types that load in stock `llama.cpp`.

Required work:

- verify local runtime for `Q3_K_M` and `Q4_K_M` from the canonical HF/staging paths;
- confirm the HF repo contains no `XL` names and no legacy single-variant public filenames;
- add a hardware compatibility table that separates measured results from estimates;
- decide whether `Q5_K_M` is worth generating after disk cleanup;
- defer `IQ4_NL` unless an imatrix path is available and validated.

Release gate:

- `llama-cli` bounded generation passes locally;
- `llama-bench` or equivalent prefill/decode evidence exists;
- quality/eval JSONs are present and referenced;
- README gives exact `llama.cpp` commands and settings;
- model card labels measured hardware separately from inferred hardware guidance.

### Track 2: OTQ-Packed

The packed release is a format release for OpenTQ `.otq` payloads plus `opentq-pack.json`. It is not a stock inference release.

Required work:

- verify every public folder has `opentq-pack.json`, complete `.otq` payloads, and a SHA256/size manifest;
- update the model card to explain the format boundary, loader status, and unsupported runtimes;
- stage and upload only after the manifest matches local files;
- mark release status per variant: `runtime-prepared`, `experimental`, or `reference`.

Release gate:

- each uploaded folder count matches local staging;
- manifest payload bytes match local artifact sizes;
- README makes clear that users need OpenTQ tooling or a future runtime loader.

### Track 3: OTQ-Metal-GGUF

The Metal/custom-runtime release is blocked unless runtime evidence exists for a given artifact. It must not imply stock GGUF compatibility.

Required work:

- verify which custom GGUFs are complete and which are partial or inconsistent;
- keep `TQ3_SB4` as the first candidate if its smoke and Metal validation evidence still passes;
- block `TQ4_SB4` until the inconsistent GGUF export size is audited;
- block `TQ4_BAL_V2`, `TQ4R2`, and `TQ4R4` until runtime validation exists;
- document required fork/runtime commands explicitly.

Release gate:

- metadata read passes;
- bounded generation passes;
- long-context wall-clock benchmark passes;
- required runtime and unsupported runtimes are documented;
- HF card labels the artifacts as custom OpenTQ/Metal runtime files.

## Hardware Compatibility

The HF model cards should include a compatibility section with two categories:

- `Measured`: hardware actually tested locally. Current target is M1 Max 32 GB.
- `Expected`: other Apple Silicon machines inferred from memory capacity, bandwidth class, and artifact size. These rows must not claim benchmark numbers.

The first matrix should include:

- 32 GB Apple Silicon: `Q3_K_M` primary, `Q4_K_M` limited-context or tighter settings;
- 48 GB Apple Silicon: `Q4_K_M` primary, `Q5_K_M` possible after generation;
- 64 GB+ Apple Silicon: quality-first GGUFs and larger native/custom candidates;
- 16 GB Apple Silicon: no 27B daily-driver claim for current artifacts.

Exact chip names and memory-bandwidth notes must be sourced from Apple documentation or measured locally before final HF upload.

## Cleanup Policy

Disk cleanup must be driven by a manifest, not manual deletion.

The cleanup manifest must classify each large path as:

- `keep`: source of truth or needed for a pending release;
- `uploaded-verified`: safe cleanup candidate after HF SHA/file-count verification;
- `regenerable`: safe cleanup candidate if scripts and inputs remain available;
- `blocked`: do not delete because a release gate still depends on it;
- `investigate`: size, hardlink, or process state is inconsistent.

The manifest must record path, apparent size, inode/link-count notes for large files, HF target, verification command, and estimated reclaimable bytes. Deletion should remain a separate explicit action after the manifest is reviewed.

## Claude Code Delegation Policy

Claude Code is allowed only for bounded tasks where Codex can review deterministic outputs.

Good Claude Code tasks:

- produce an inventory report from local files;
- draft README/model-card sections without uploading;
- inspect docs for naming violations;
- propose tests for a specific helper script.

Tasks kept in Codex:

- Hugging Face upload/delete operations;
- local artifact deletion;
- release-gate decisions;
- final code review and commits;
- changing shared release scripts without a Codex diff review.

Claude Code outputs must be treated like junior-contributor patches: reviewed with `git diff`, tested locally, and either accepted, edited, or discarded.

## Execution Order

1. Create a release audit script/report that checks naming, HF state, artifact presence, runtime evidence, and disk cleanup candidates.
2. Add or update tests for naming rules and release-track boundaries.
3. Update docs/model-card generation for hardware compatibility and `IQ4_NL` wording.
4. Run local runtime validation for stock GGUF `Q3_K_M` and `Q4_K_M`.
5. Generate cleanup manifest without deleting anything.
6. Stage HF updates for `OTQ-GGUF` and review generated README.
7. Decide whether to generate `Q5_K_M` after enough space is reclaimed.
8. Gate `OTQ-Packed` upload from verified manifests.
9. Gate `OTQ-Metal-GGUF` only for artifacts with current runtime evidence.

## Review Criteria

The work is complete when:

- no public generated artifact contains `XL`;
- all public names match the naming contract;
- `OTQ-GGUF` hardware compatibility is explicit and evidence-backed;
- local runtime checks are recorded for current public GGUFs;
- `OTQ-Packed` and `OTQ-Metal-GGUF` are either released with gates or explicitly blocked with reasons;
- cleanup candidates are listed with verification status and no deletion has happened without explicit approval.
