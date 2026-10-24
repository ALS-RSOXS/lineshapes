"""Shared fixtures.

Backends
--------
Every precision test runs against each available *backend*: a module exposing
``gaussian, lorentzian, voigt, pseudo_voigt, continuum_beta, optical_model``
with the signatures of ``tests/reference/lineshapes_ref.py``.

* ``reference``: the pure-numpy reference (always available).
* ``rust``: ``lineshapes_rs`` once it exposes those names (skipped until then).

Speed tests compare non-reference backends against the reference timing
measured in the same session, so they are machine-independent.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from tests.reference import lineshapes_ref

DATA = Path(__file__).parent / "data"
API = ("gaussian", "lorentzian", "voigt", "pseudo_voigt", "continuum_beta", "optical_model")


def _rust_backend():
    import importlib.util

    if importlib.util.find_spec("lineshapes_rs") is None:
        return None
    import lineshapes_rs  # installed but broken -> fail loudly, never skip silently

    return lineshapes_rs if all(hasattr(lineshapes_rs, n) for n in API) else None


BACKENDS = {"reference": lineshapes_ref, "rust": _rust_backend()}


@pytest.fixture(params=list(BACKENDS))
def backend(request):
    mod = BACKENDS[request.param]
    if mod is None:
        pytest.skip(f"backend {request.param!r} not built yet")
    return mod


@pytest.fixture(params=[n for n in BACKENDS if n != "reference"])
def fast_backend(request):
    mod = BACKENDS[request.param]
    if mod is None:
        pytest.skip(f"backend {request.param!r} not built yet")
    return mod


@pytest.fixture(scope="session")
def ref():
    return lineshapes_ref


@pytest.fixture(scope="session")
def cu_model():
    """Cu L-edge model from lr-optical-models (see tests/data/cu/PROVENANCE.md)."""
    with open(DATA / "cu" / "peaks.csv") as fh:
        rows = list(csv.DictReader(fh))
    peaks = [
        ("gaussian", dict(e0=float(r["location"]), width=float(r["width"]), amp=float(r["height"])))
        for r in rows
    ]
    ion = json.loads((DATA / "cu" / "ionization_edge.json").read_text())
    steps = [
        dict(e0=v["edge_energy"], width=v["width"], height=v["height"], decay=v.get("decay", 0.0))
        for v in ion["edges"].values()
    ]
    return peaks, dict(bkg=ion["bkg"], slope=ion["slope"], steps=steps)


def resolve_peaks(backend, peaks):
    """Map ("name", kwargs) peak specs onto a backend's functions."""
    return [(getattr(backend, name), kw) for name, kw in peaks]


CU_GRID = np.linspace(900.0, 1000.0, 4001)
