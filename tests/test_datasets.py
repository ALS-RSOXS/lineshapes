"""Shipped example datasets: loading, units, and the fixed continuum."""

from __future__ import annotations

import numpy as np
import pytest

from lineshapes import beta_to_mu, datasets, mu_to_beta


def test_available_tables():
    assert datasets.available() == [
        "alq3_dft_clusters", "alq3_oc_model", "alq3_oc_tey", "znpc_dft_clusters", "znpc_nexafs",
    ]


@pytest.mark.parametrize("name", datasets.available())
def test_tables_are_finite_float_columns(name):
    t = datasets.load_table(name)
    assert t and all(v.dtype == np.float64 and v.ndim == 1 and np.all(np.isfinite(v)) for v in t.values())
    assert len({len(v) for v in t.values()}) == 1


def test_unknown_dataset_raises():
    with pytest.raises(FileNotFoundError, match="available"):
        datasets.path("nope")


def test_znpc_nexafs_angles_and_range():
    t = datasets.load_table("znpc_nexafs")
    assert sorted(set(t["theta_deg"])) == [20.0, 40.0, 54.7, 70.0, 90.0]
    assert t["energy_ev"].min() < 271 and t["energy_ev"].max() > 329


def test_alq3_tables_carry_delta_and_beta():
    for name in ("alq3_oc_tey", "alq3_oc_model"):
        t = datasets.load_table(name)
        assert set(t) == {"energy_ev", "delta", "beta"} and np.all(t["beta"] > 0)


@pytest.mark.parametrize("material", datasets.MATERIALS)
def test_continuum_matches_its_parameters(material):
    c = datasets.load_continuum(material)
    m = datasets.continuum_model(material)
    E = c["energy"][::50]
    s = c["step"]
    from scipy.special import erf

    beta = c["bkg"] + c["slope"] * E + 0.5 * s["height"] * (1 + erf((E - s["e0"]) / s["width"])) + s["decay"] * np.maximum(E - s["e0"], 0)
    z = m(E)
    np.testing.assert_allclose(z.imag, beta, rtol=1e-12)
    np.testing.assert_allclose(z.real, c["delta"][::50], rtol=1e-12)


def test_continuum_unknown_material():
    with pytest.raises(ValueError, match="material"):
        datasets.load_continuum("cu")


def test_mu_beta_roundtrip_and_value():
    E = np.array([285.0, 300.0])
    mu = np.array([3.0e4, 2.0e4])
    np.testing.assert_allclose(beta_to_mu(E, mu_to_beta(E, mu, 1.3), 1.3), mu, rtol=1e-14)
    # 3e4 cm^2/g at 285 eV, unit density: beta = mu*lambda/(4 pi), lambda = 43.5 A
    assert mu_to_beta(285.0, 3.0e4) == pytest.approx(3.0e4 * 43.5e-8 / (4 * np.pi), rel=2e-3)
