"""Example datasets shipped with the package: ZnPc and Alq3 carbon K-edge spectra.

All optical-constant tables are per unit density (1 g/cm^3). The continuum of each material is
fixed ahead of time (bare-atom beta, with its delta from kkcalc2); the peaks are fit on top.
Provenance and the generating script: ``tools/make_example_data.py``.

Tables (see [`available`][lineshapes.datasets.available]):

| name | columns |
|---|---|
| `znpc_nexafs` | `energy_ev`, `theta_deg`, `mu_cm2_per_g`, `intensity_raw` (five angles) |
| `znpc_dft_clusters`, `alq3_dft_clusters` | `energy_ev`, `amplitude`, `sigma_ev` (DFT transition clusters) |
| `alq3_oc_tey` | `energy_ev`, `delta`, `beta` (optical constants from total-electron-yield XAS) |
| `alq3_oc_model` | `energy_ev`, `delta`, `beta` (building-block model optical constants) |
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import numpy as np
import numpy.typing as npt

from lineshapes._core import Model

FloatArray = npt.NDArray[np.float64]

_DATA = resources.files("lineshapes") / "data"
MATERIALS = ("znpc", "alq3")


def available() -> list[str]:
    """Names of the tables accepted by [`load_table`][lineshapes.datasets.load_table]."""
    return sorted(p.name.removesuffix(".csv") for p in _DATA.iterdir() if p.name.endswith(".csv") and "continuum" not in p.name)


def path(name: str) -> Path:
    """Filesystem path of a shipped file (``name`` with or without ``.csv``)."""
    fname = name if "." in name else f"{name}.csv"
    p = Path(str(_DATA / fname))
    if not p.exists():
        raise FileNotFoundError(f"unknown dataset {name!r}; available: {available()}")
    return p


def _read(p: Path) -> dict[str, FloatArray]:
    with open(p) as fh:
        header = fh.readline().strip().split(",")
    data = np.loadtxt(p, delimiter=",", skiprows=1, ndmin=2)
    return {h: np.ascontiguousarray(data[:, i]) for i, h in enumerate(header)}


def load_table(name: str) -> dict[str, FloatArray]:
    """Load a shipped table as ``{column: float64 array}``."""
    return _read(path(name))


def load_continuum(material: str) -> dict:
    """Fixed continuum of a material (``"znpc"`` or ``"alq3"``).

    Returns the parameters (``bkg``, ``slope``, ``step`` = ``e0, width, height, decay``), provenance,
    and the precomputed continuum delta table as ``energy`` and ``delta`` arrays.
    """
    if material not in MATERIALS:
        raise ValueError(f"material must be one of {MATERIALS}, got {material!r}")
    meta = json.loads(path(f"{material}_continuum.json").read_text())
    table = _read(path(f"{material}_continuum_delta.csv"))
    meta["energy"], meta["delta"] = table["energy_ev"], table["delta"]
    return meta


def continuum_model(material: str) -> Model:
    """A [`Model`][lineshapes.Model] holding only the material's fixed continuum (beta + delta table).

    Add peaks to it with the usual chained methods.
    """
    c = load_continuum(material)
    s = c["step"]
    return Model().continuum(c["bkg"], c["slope"]).step(s["e0"], s["width"], s["height"], s["decay"]).delta_table(c["energy"], c["delta"])
