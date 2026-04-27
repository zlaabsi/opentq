#!/usr/bin/env bash

qwen36_dynamic_quant_suffix() {
  local profile="$1"
  case "$profile" in
    OTQ-DYN-Q3_K_M) echo "Q3_K_M" ;;
    OTQ-DYN-Q4_K_M) echo "Q4_K_M" ;;
    OTQ-DYN-Q5_K_M) echo "Q5_K_M" ;;
    OTQ-DYN-IQ4_NL) echo "IQ4_NL" ;;
    *) echo "$profile" ;;
  esac
}

qwen36_dynamic_public_filename() {
  local profile="$1"
  local quant
  quant="$(qwen36_dynamic_quant_suffix "$profile")"
  echo "Qwen3.6-27B-OTQ-DYN-${quant}.gguf"
}

qwen36_dynamic_legacy_filename() {
  local profile="$1"
  echo "Qwen3.6-27B-${profile}.gguf"
}

qwen36_dynamic_ensure_public_alias() {
  local out_dir="$1"
  local profile="$2"
  local public_path="$out_dir/$(qwen36_dynamic_public_filename "$profile")"
  local legacy_path="$out_dir/$(qwen36_dynamic_legacy_filename "$profile")"

  if [[ -s "$public_path" ]]; then
    return 0
  fi
  if [[ -s "$legacy_path" ]]; then
    ln "$legacy_path" "$public_path" 2>/dev/null || cp -p "$legacy_path" "$public_path"
  fi
}
