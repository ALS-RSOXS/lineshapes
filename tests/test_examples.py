"""The docs examples run headless and keep their fit quality (they generate the docs figures)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("scipy")
pytest.importorskip("matplotlib")

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture(scope="module")
def examples_on_path():
    sys.path.insert(0, str(EXAMPLES))
    yield
    sys.path.remove(str(EXAMPLES))


def test_znpc_example(examples_on_path, tmp_path):
    import znpc_nexafs

    r = znpc_nexafs.run(tmp_path)
    assert r["rel_rms"] < 0.01  # observed 0.34% of mean beta
    assert all(v < 0.01 for v in r["angle_rel_rms"].values())
    for name in ("znpc-fit", "znpc-decomposition", "znpc-angles"):
        for mode in ("light", "dark"):
            assert (tmp_path / f"{name}-{mode}.png").stat().st_size > 10_000
    assert "Fitted density scale" in (tmp_path / "znpc_peaks.md").read_text()


def test_alq3_example(examples_on_path, tmp_path):
    import alq3_optical_constants

    r = alq3_optical_constants.run(tmp_path)
    assert r["rel_rms"] < 0.01  # observed 0.55%
    # analytic delta reproduces the independent delta table to a few percent of its range
    assert r["delta_rms"] < 0.05 * r["delta_span"]  # observed 2.0%
    assert (tmp_path / "alq3-delta-light.png").stat().st_size > 10_000
