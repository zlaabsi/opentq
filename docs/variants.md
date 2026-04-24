# Variants

## Naming

The public `TQ3_4S` label is useful as a reference point, but not as a long-term open naming scheme. `OpenTQ` uses names that describe the structure:

- `TQx_0`: low-bit scalar baseline
- `TQx_SBy`: `x` weight bits, `y` sub-block scales
- `TQ4R2`: `4+2` residual
- `TQ4R4`: `4+4` residual

## Initial family

### `TQ1_0`

- role: minimum footprint
- target: tiny RAM budgets and stress tests
- runtime expectation: quality floor, not daily driver

### `TQ2_0`

- role: aggressive 32 GB unified-memory target
- target: MoE or lighter dense models where long context matters more than top-end quality

### `TQ3_SB4`

- role: compact dense-model release
- target: 27B-class models on machines that cannot carry a safer 4-bit daily driver
- relationship to public TQ releases: same high-level open ingredients, different naming and spec ownership

### `TQ4_SB4`

- role: default release
- target: strongest first candidate for Apple Silicon and `llama.cpp`

### `TQ4_SB2`

- role: redesigned 16 GiB-class uniform 4-bit path
- target: Qwen3.6-27B-scale models where `TQ4_SB4` is too large but a 4-bit primary quantizer is still preferred

### `TQ4R2`

- role: quality-biased 6-bit total
- target: bridge format when `TQ4_SB4` is slightly too lossy but `TQ4R4` is too large

### `TQ4R4`

- role: regression target and near-lossless release
- target: reference checkpoints, eval baselines, and top-end Macs

### `TQ_MIX_MOE`

- role: release recipe for MoE
- target: architectures where a single global bit policy leaves quality on the table
