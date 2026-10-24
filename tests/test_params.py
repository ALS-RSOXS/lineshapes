"""Flat parameter vector + bound Evaluator (fit without rebuilding the model)."""

from __future__ import annotations

import numpy as np
import pytest

lineshapes = pytest.importorskip("lineshapes")
from lineshapes import Model  # noqa: E402

E = np.linspace(900.0, 1000.0, 4001)


def full_model():
    return (Model()
            .gaussian(933.8, 0.49, 4.1e-5)
            .lorentzian(936.0, 0.3, 2e-5)
            .voigt(938.3, 0.7, 0.2, 5e-5)
            .pseudo_voigt(955.0, 1.5, 0.4, 2e-5)
            .step(939.4, 6.1, 7.2e-5)
            .step(960.7, 2.2, 5.1e-5, 1e-7)
            .continuum(3.6e-4, -3.7e-7)
            .delta_table(E, 1e-4 * np.cos(E / 30.0))
            .scale(1.1)
            .shift(0.05))


def rebuild(theta):
    """Rebuild full_model() from scratch at theta (the slow path we avoid)."""
    t = list(theta)
    return (Model()
            .gaussian(*t[0:3]).lorentzian(*t[3:6]).voigt(*t[6:10]).pseudo_voigt(*t[10:14])
            .step(*t[14:18]).step(*t[18:22]).continuum(t[22], t[23])
            .delta_table(E, 1e-4 * np.cos(E / 30.0)).scale(t[24]).shift(t[25]))


def test_param_names_and_roundtrip():
    m = full_model()
    assert m.n_params == len(m.params) == len(m.param_names) == 26
    assert m.param_names[:4] == ["gaussian[0].e0", "gaussian[0].width", "gaussian[0].amp", "lorentzian[0].e0"]
    assert m.param_names[14:16] == ["step[0].e0", "step[0].width"]
    assert m.param_names[18] == "step[1].e0"
    assert m.param_names[-4:] == ["continuum.bkg", "continuum.slope", "scale", "shift"]
    theta = m.params
    np.testing.assert_array_equal(m.set_params(theta).params, theta)


def test_set_params_equals_rebuild():
    m = full_model()
    theta = m.params * (1 + 0.01 * np.random.default_rng(0).standard_normal(m.n_params))
    theta[12] = 0.37  # eta
    np.testing.assert_allclose(m.set_params(theta)(E), rebuild(theta)(E), rtol=0, atol=1e-18)


def test_call_with_params_does_not_mutate():
    m = full_model()
    theta0 = m.params.copy()
    theta = theta0.copy()
    theta[0] += 0.5
    a = m(E, params=theta)
    np.testing.assert_array_equal(m.params, theta0)
    np.testing.assert_allclose(a, rebuild(theta)(E), atol=1e-18)


def test_invalid_params_raise_and_leave_model_unchanged():
    m = full_model()
    theta0 = m.params.copy()
    bad = theta0.copy()
    bad[1] = -0.1  # negative width
    with pytest.raises(ValueError, match="width"):
        m.set_params(bad)
    with pytest.raises(ValueError, match="expected 26"):
        m.set_params(theta0[:-1])
    np.testing.assert_array_equal(m.params, theta0)


@pytest.mark.parametrize("cache", [True, False])
def test_evaluator_matches_model_through_fit_like_updates(cache):
    """Random single-parameter (FD-column-like), multi-parameter and global updates."""
    m = full_model()
    ev = m.bind(E, cache=cache)
    rng = np.random.default_rng(1)
    theta = m.params.copy()
    out = np.empty(E.size, complex)
    for it in range(200):
        k = rng.integers(theta.size) if it % 25 else slice(None)
        step = 1e-3 * rng.standard_normal(theta[k].shape)
        theta[k] = theta[k] * (1 + step)
        theta[12] = np.clip(theta[12], 0, 1)
        got = ev(theta, out=out) if it % 2 else ev(theta)
        want = rebuild(theta)(E)
        assert np.max(np.abs(got - want)) / np.max(np.abs(want)) < 1e-13
    assert got is out or it % 2 == 0


def test_evaluator_out_validation():
    ev = full_model().bind(E)
    with pytest.raises(ValueError, match="length"):
        ev(out=np.empty(10, complex))
