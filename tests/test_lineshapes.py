"""Precision of each peak lineshape (delta + i beta) and of the continuum.

Thresholds are set from the prototype figures in prototypes/figures/ (2026-10-14):
the analytic delta matches an FFT Hilbert transform of beta to ~2e-6 (amp = 1),
the Voigt limits reach the Gaussian/Lorentzian to 5e-7 / 3e-6.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import dawsn, erf

E0 = 930.0
# Wide uniform grid so the FFT Hilbert transform sees the tails.
E_WIDE = np.linspace(E0 - 2000.0, E0 + 2000.0, 2**18)
WIN = (E_WIDE > E0 - 15.0) & (E_WIDE < E0 + 15.0)
E_LOCAL = E_WIDE[WIN]

PEAKS = {
    "gaussian": dict(e0=E0, width=1.0, amp=1.0),
    "lorentzian": dict(e0=E0, gamma=0.5, amp=1.0),
    "voigt": dict(e0=E0, sigma=0.5, gamma=0.3, amp=1.0),
    "pseudo_voigt": dict(e0=E0, fwhm=1.2, eta=0.4, amp=1.0),
}


@pytest.mark.parametrize("name", list(PEAKS))
def test_delta_is_hilbert_of_beta(backend, ref, name):
    """Analytic delta == FFT Hilbert[beta] within the window, to 5e-6 (amp = 1)."""
    z = np.asarray(getattr(backend, name)(E_WIDE, **PEAKS[name]))
    numeric = ref.hilbert_fft(E_WIDE, z.imag)
    assert np.max(np.abs(z.real - numeric)[WIN]) < 5e-6


@pytest.mark.parametrize("name", list(PEAKS))
def test_matches_reference(backend, ref, name):
    """Backend agrees with the numpy reference to near machine precision."""
    z = np.asarray(getattr(backend, name)(E_LOCAL, **PEAKS[name]))
    z_ref = getattr(ref, name)(E_LOCAL, **PEAKS[name])
    np.testing.assert_allclose(z, z_ref, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("name", list(PEAKS))
def test_peak_height_and_sign(backend, name):
    """beta(E0) = amp; delta < 0 below E0, > 0 above (delta = +H[beta])."""
    kw = dict(PEAKS[name], amp=2.5)
    z = np.asarray(getattr(backend, name)(np.array([E0 - 1.0, E0, E0 + 1.0]), **kw))
    assert z[1].imag == pytest.approx(2.5, rel=1e-12)
    assert abs(z[1].real) < 1e-12
    assert z[0].real < 0 < z[2].real


def test_gaussian_matches_lr_optical_models_formula(backend):
    """Same formula as lr-optical-models peak_set.Peak.beta/.delta (1/e half-width)."""
    e0, width, amp = 933.8, 0.49, 4.1e-5
    sigma = width / np.sqrt(2.0)
    d = E_LOCAL - e0
    beta = amp * np.exp(-0.5 * (d / sigma) ** 2)
    delta = amp * (2.0 / np.sqrt(np.pi)) * dawsn(d / (np.sqrt(2.0) * sigma))
    z = np.asarray(backend.gaussian(E_LOCAL, e0=e0, width=width, amp=amp))
    np.testing.assert_allclose(z.imag, beta, rtol=1e-12, atol=1e-20)
    np.testing.assert_allclose(z.real, delta, rtol=1e-12, atol=1e-20)


def test_voigt_gaussian_limit(backend):
    v = np.asarray(backend.voigt(E_LOCAL, e0=E0, sigma=1.0 / np.sqrt(2.0), gamma=1e-6, amp=1.0))
    g = np.asarray(backend.gaussian(E_LOCAL, e0=E0, width=1.0, amp=1.0))
    assert np.max(np.abs(v - g)) < 1e-6


def test_voigt_lorentzian_limit(backend):
    v = np.asarray(backend.voigt(E_LOCAL, e0=E0, sigma=1e-3, gamma=0.5, amp=1.0))
    lo = np.asarray(backend.lorentzian(E_LOCAL, e0=E0, gamma=0.5, amp=1.0))
    assert np.max(np.abs(v - lo)) < 1e-5


def test_tch_pseudo_voigt_approximates_voigt(backend, ref):
    """Thompson-Cox-Hastings pseudo-Voigt is within 2.5e-2 of the exact Voigt."""
    fwhm, eta = ref.tch_pseudo_voigt_params(0.5, 0.3)
    v = np.asarray(backend.voigt(E_LOCAL, e0=E0, sigma=0.5, gamma=0.3, amp=1.0))
    p = np.asarray(backend.pseudo_voigt(E_LOCAL, e0=E0, fwhm=fwhm, eta=eta, amp=1.0))
    assert np.max(np.abs(v - p)) < 2.5e-2


@pytest.mark.parametrize("eta", [0.0, 1.0])
def test_pseudo_voigt_endpoints(backend, eta):
    """eta = 0 is the Gaussian, eta = 1 the Lorentzian, at the same FWHM."""
    fwhm = 1.2
    p = np.asarray(backend.pseudo_voigt(E_LOCAL, e0=E0, fwhm=fwhm, eta=eta, amp=1.0))
    if eta == 0.0:
        width = fwhm / (2.0 * np.sqrt(np.log(2.0)))  # 1/e half-width at this FWHM
        other = backend.gaussian(E_LOCAL, e0=E0, width=width, amp=1.0)
    else:
        other = backend.lorentzian(E_LOCAL, e0=E0, gamma=fwhm / 2.0, amp=1.0)
    np.testing.assert_allclose(p, np.asarray(other), rtol=1e-12, atol=1e-14)


def test_continuum_beta_formula(backend, cu_model):
    """bkg + slope*E + sum 0.5*h*(1 + erf((E-e0)/w)) + decay*max(E-e0, 0)."""
    _, cont = cu_model
    E = np.linspace(900.0, 1000.0, 2001)
    expected = cont["bkg"] + cont["slope"] * E
    for s in cont["steps"]:
        expected = expected + 0.5 * s["height"] * (1.0 + erf((E - s["e0"]) / s["width"]))
        expected = expected + s["decay"] * np.maximum(E - s["e0"], 0.0)
    np.testing.assert_allclose(np.asarray(backend.continuum_beta(E, **cont)), expected, rtol=1e-12)


def test_continuum_decay_term(backend):
    E = np.array([940.0, 950.0, 960.0])
    step = dict(e0=950.0, width=1e-9, height=0.0, decay=2.0)
    got = np.asarray(backend.continuum_beta(E, steps=[step]))
    np.testing.assert_allclose(got, [0.0, 0.0, 20.0], atol=1e-12)
