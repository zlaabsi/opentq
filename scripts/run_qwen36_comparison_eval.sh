#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "run_qwen36_comparison_eval.sh is deprecated; use scripts/run_qwen36_otq_eval.sh."
exec ./scripts/run_qwen36_otq_eval.sh "$@"
