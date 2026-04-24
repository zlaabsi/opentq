# llama.cpp patch strategy

This directory will hold the narrow patchsets that teach `llama.cpp` how to:

- recognize `OpenTQ` GGUF tensor types
- allocate the right scale / index payload views
- dispatch Metal kernels for `TQ3_SB4`, `TQ4_SB4`, and `TQ4R2`

The deliberate choice is to keep the patches small and reviewable before creating a long-lived public fork.
