from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from opentq.status import build_status_payload, print_status


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "artifacts/qwen3.6-27b"
    print_status(build_status_payload(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
