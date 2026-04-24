# Architecture

`OpenTQ` is deliberately split into three logical layers:

1. `quantizer core`
   - block partitioning
   - WHT / sign rotation
   - Lloyd-Max codebooks
   - residual passes
   - calibration and tensor-role policies
2. `artifact emitters`
   - GGUF custom tensor payloads for `llama.cpp`
   - a compact runtime-native format for `opentq-metal`
3. `runtime adapters`
   - `llama.cpp` patchsets and integration notes
   - Apple Silicon runtime work that is free to diverge when it materially improves wall-clock performance

## Why the repo is split this way

The public TQ forks prove that runtime support alone is not enough. If the quantizer is private, the format family cannot evolve in the open. `OpenTQ` treats the quantizer as the product and the runtimes as targets.

## Design principles

- Weight formats should be explicit and reproducible.
- Runtime-specific tricks should not leak into the core bitstream format unless they are necessary.
- `llama.cpp` compatibility matters, but it is not the ceiling.
- Apple Silicon needs a separate runtime track because compressed-domain attention and speculative decoding can justify different architectural decisions.

## Format strategy

### Scalar codebooks over Gaussianized weights

The baseline assumption is that random-sign Walsh-Hadamard rotation is the simplest public path to make block statistics closer to the Gaussian priors for which Lloyd-Max scalar codebooks are natural.

### Residual variants

Residual quantization is treated as a first-class family member, not a special case. `TQ4R2` and `TQ4R4` exist because an open family needs a clear route above the base 3/4-bit formats.

### Mixed MoE profile

`TQ_MIX_MOE` is a release policy, not a single tensor encoding. The plan is:

- experts: aggressive low bit
- attention projections: conservative
- embeddings and output head: anchored high precision

## Runtime split

### `llama.cpp`

The `llama.cpp` track optimizes for:

- broad ecosystem compatibility
- robust prefill on long contexts
- easy OpenAI-compatible serving

### `opentq-metal`

The Apple Silicon runtime optimizes for:

- compressed-domain decode kernels
- KV compression that is native to unified memory constraints
- optional DFlash-style speculative decoding
- prefix cache reuse and branch-friendly agentic workloads

