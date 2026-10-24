"""MkDocs hook: generate the example figures and tables when they are missing.

`make docs-figures` forces a regeneration. Outputs go to docs/assets/generated (gitignored).
Generation only runs when the outputs are absent, so `mkdocs serve` does not loop on its own writes.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "generated"
SENTINELS = ("znpc-fit-light.png", "alq3-delta-light.png", "znpc_angles.md", "alq3_summary.md")


def on_pre_build(config) -> None:
    if all((OUT / s).exists() for s in SENTINELS):
        return
    sys.path.insert(0, str(ROOT / "examples"))
    import alq3_optical_constants
    import znpc_nexafs

    znpc_nexafs.run(OUT)
    alq3_optical_constants.run(OUT)
