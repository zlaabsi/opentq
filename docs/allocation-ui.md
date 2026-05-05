# Allocation UI Roadmap

The OpenTQ allocation policy is the differentiator. YAML policies and TSV files are precise, but they are not enough for users who want to understand why a tensor family receives `Q3_K`, `Q6_K`, `Q8_0`, or `F16`.

The UI should turn OpenTQ into a decision tool.

## Core Experience

The first screen should show the full tensor map:

| View | Purpose |
| --- | --- |
| Tensor treemap | 851 tensors sized by bytes or parameter count, colored by assigned quant type. |
| Layer strip | Layer-by-layer compression and promoted anchors. |
| Family filter | `mlp_proj`, `self_attn_proj`, `linear_attn_proj`, `linear_attn_state`, `embeddings`, `lm_head`, `layernorm`. |
| Error overlay | MSE / max error / proxy quality impact when available. |
| Policy diff | Compare `OTQ-DYN-Q3_K_M`, `OTQ-DYN-Q4_K_M`, `OTQ-DYN-Q5_K_M`, and custom YAML policies. |

## Inputs

The UI should read existing OpenTQ artifacts:

| File | Source |
| --- | --- |
| `plan.json` | `uv run opentq dynamic-gguf-plan ...` |
| `tensor-types.annotated.tsv` | generated dynamic GGUF plan |
| `manifest.json` | native quantization release workdir |
| `opentq-pack.json` | packed release |
| `RUN_SUMMARY.md` / validation JSON | runtime gates and benchmark evidence |

## User Actions

- Open a built-in policy.
- Load a custom `policy.yaml`.
- Select a layer or tensor family.
- See why it was assigned a precision.
- Edit a policy and export YAML.
- Generate `tensor-types.txt` for stock-compatible GGUF quantization.

## Sensitivity Layer

The UI should not only display allocation. It should explain the decision:

```text
tensor family: self_attn_proj
default: Q4_K
promoted: Q6_K
reason: edge layer + attention anchor
measured proxy: lower error tolerance than bulk MLP
```

When empirical metrics exist, show them. When they do not, label the reason as policy heuristic rather than measured sensitivity.

## Implementation Shape

Start with a static artifact that can be opened locally from a generated run directory. Then promote it to a React/Vite app once the data contract is stable:

```bash
uv run opentq dynamic-gguf-plan \
  --policy-file policies/qwen36-custom-dyn-q4.yaml \
  --output artifacts/my-policy

# Future:
uv run opentq allocation-ui \
  --plan artifacts/my-policy/plan.json \
  --open
```

The release rule is the same as the model card rule: the UI can show policy evidence immediately, but claims about quality impact require paired validation.
