# llama.cpp plan

`OpenTQ` does not start from a giant fork. It starts from a patch strategy.

## Why

- a huge fork slows iteration before the tensor formats are stable
- the first value is in the quantizer, manifests, and reproducible evals
- `llama.cpp` integration needs to follow the formats, not define them prematurely

## Planned milestones

### Milestone 1: offline artifact shape

- converter reads safetensors
- per-tensor variant selection
- GGUF metadata manifest emission
- tensor payload layout frozen for `TQ3_SB4`, `TQ4_SB4`, `TQ4R2`

### Milestone 2: CPU reference path

- dequantize + matmul reference kernels
- correctness tests against reconstructed weights
- load-time validation in a narrow `llama.cpp` branch

### Milestone 3: Metal path

- native dequant/unpack kernels
- fused matmul path for `TQ3_SB4` and `TQ4_SB4`
- residual merge path for `TQ4R2`

### Milestone 4: serving validation

- long-context prefill benchmarks
- tool-calling / cache-reuse agentic traces
- M1 Max 32 GB as the floor target

