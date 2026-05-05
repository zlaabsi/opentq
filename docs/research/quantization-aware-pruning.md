# Quantization-aware Pruning Roadmap

OpenTQ's core idea is that model components do not all deserve the same precision. The same reasoning can drive pruning: some components should be kept at higher precision, some can be quantized harder, and some may be removable.

## Decision Space

For each structured unit, OpenTQ should eventually choose one action:

| Action | Meaning |
| --- | --- |
| `keep_high_precision` | Sensitive unit; preserve with F16/Q8/Q6 depending on runtime. |
| `quantize_standard` | Stable unit; use the profile default. |
| `quantize_aggressive` | Low-sensitivity unit; lower precision before considering pruning. |
| `prune_candidate` | Unit is low-sensitivity and low-utility under paired validation. |

Structured units should start conservative:

- attention heads
- MLP channels or blocks
- linear-attention state/projection groups
- repeated low-impact tensor slices found by allocation diagnostics

## Why Coupling Matters

Pruning and quantization are usually optimized separately. That can waste capacity: a low-sensitivity head might be better removed than quantized, while a high-sensitivity head should not be forced into a low-bit format just because the rest of its tensor family is cheap.

OpenTQ can expose this as a single budget problem:

```text
minimize(memory + latency)
subject to quality_delta <= threshold
actions in {keep, quantize, prune}
```

## Current Interface

Do not prune public models yet. OpenTQ now exposes an offline experiment harness:

```bash
uv run opentq pruning-candidates \
  --plan artifacts/qwen36-otq-dyn-q4/plan.json \
  --output artifacts/qwen36-pruning-candidates
```

| Artifact | Purpose |
| --- | --- |
| `pruning-candidates.jsonl` | Ranked structured units with saliency and quantization error. |
| `pruning-policy.yaml` | Human-editable keep/quantize/prune decisions. |
| `paired-pruning-report.md` | Quality and runtime deltas against no-prune baseline. |

The first implementation ranks candidates from allocation evidence and tensor role heuristics. It does not rewrite weights. Any public pruned artifact needs paired quality and runtime validation against the non-pruned baseline.

## Required Evidence Before Release

- Same-runner paired quality subset.
- Runtime latency improvement.
- Memory improvement.
- Per-layer breakdown of removed capacity.
- Reversibility: users can inspect and disable pruning decisions.

Until then, pruning stays a research track and must not be mixed into stock GGUF release claims.
