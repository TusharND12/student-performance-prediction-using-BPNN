#!/usr/bin/env python3
"""Print calibration JSON (same as GET /meta/calibration) for CI or reports."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.meta import calibration_bins  # noqa: E402


def main() -> None:
    out = calibration_bins()
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
