"""Pure-numpy reference implementations of lineshape peaks and continuum.

Conventions (match lr-optical-models ``src/optical_model/peak_set.py``):

* Energy ``E`` in eV. Every peak returns a complex array ``delta + 1j * beta``
  (the refloxide "decrement" form, n = 1 - delta + i beta).
* ``delta = H[beta]`` with H f(x) = (1/pi) PV int f(t) / (x - t) dt
  (scipy.signal.hilbert convention). Result: delta < 0 below a resonance,
  delta > 0 above it. No mirror term at -E0 (resonances are far from 0 eV).
* Amplitude ``A`` is the peak HEIGHT of beta at E0 for every lineshape.

Width parameters
----------------
gaussian      ``width``: 1/e half-width (beta(E0 ± width) = A/e), sigma = width / sqrt(2)
lorentzian    ``gamma``: half width at half maximum (HWHM)
voigt         ``sigma`` (Gaussian std) and ``gamma`` (Lorentzian HWHM)
pseudo_voigt  ``fwhm`` shared by both parts, mixing ``eta`` in [0, 1]

All peak functions share one primitive, the Faddeeva function
w(z) = exp(-z^2) erfc(-iz): for real x, Re w = exp(-x^2) and
Im w = (2/sqrt(pi)) dawsn(x), so beta ~ Re w and delta ~ Im w.
"""

from __future__ import annotations

import numpy as np
from scipy.special import dawsn, erf, wofz

SQRT2 = np.sqrt(2.0)
SQRTPI = np.sqrt(np.pi)
LN2 = np.log(2.0)


# --------------------------------------------------------------------- peaks
def gaussian(E, e0, width, amp):
    """Gaussian beta with its Dawson delta partner (1/e half-width)."""
    sigma = np.asarray(width, dtype=float) / SQRT2
    x = (np.asarray(E, dtype=float) - e0) / (SQRT2 * sigma)
    beta = amp * np.exp(-(x**2))
    delta = amp * (2.0 / SQRTPI) * dawsn(x)
    return delta + 1j * beta


def lorentzian(E, e0, gamma, amp):
    """Lorentzian of HWHM ``gamma``: delta + i beta = A*gamma / (x - i*gamma)."""
    x = np.asarray(E, dtype=float) - e0
    return amp * gamma / (x - 1j * gamma)


def voigt(E, e0, sigma, gamma, amp):
    """Exact Voigt via Faddeeva, normalized so beta(E0) = amp."""
    z = (np.asarray(E, dtype=float) - e0 + 1j * gamma) / (SQRT2 * sigma)
    w = wofz(z)
    norm = wofz(1j * gamma / (SQRT2 * sigma)).real
    return amp * (w.imag + 1j * w.real) / norm


def pseudo_voigt(E, e0, fwhm, eta, amp):
    """eta * Lorentzian + (1 - eta) * Gaussian with a shared FWHM.

    Exact KK pair by linearity: each part carries its own analytic delta.
    """
    gamma = fwhm / 2.0  # Lorentzian HWHM
    sigma = fwhm / (2.0 * np.sqrt(2.0 * LN2))
    width = sigma * SQRT2  # 1/e half-width for gaussian()
    return eta * lorentzian(E, e0, gamma, amp) + (1 - eta) * gaussian(
        E, e0, width, amp
    )


def tch_pseudo_voigt_params(sigma, gamma):
    """Thompson-Cox-Hastings (fwhm, eta) approximating a Voigt(sigma, gamma)."""
    fg = 2.0 * sigma * np.sqrt(2.0 * LN2)
    fl = 2.0 * gamma
    f = (
        fg**5
        + 2.69269 * fg**4 * fl
        + 2.42843 * fg**3 * fl**2
        + 4.47163 * fg**2 * fl**3
        + 0.07842 * fg * fl**4
        + fl**5
    ) ** 0.2
    r = fl / f
    eta = 1.36603 * r - 0.47719 * r**2 + 0.11116 * r**3
    return f, eta


# ----------------------------------------------------------------- continuum
def continuum_beta(E, bkg=0.0, slope=0.0, steps=()):
    """Continuum beta: bkg + slope*E + sum of erf ionization steps.

    ``steps`` is an iterable of dicts with keys e0, width, height and an
    optional decay (linear post-edge term, decay * max(E - e0, 0)); see
    lr-optical-models ``edge.IonizationStep.beta``.
    """
    E = np.asarray(E, dtype=float)
    beta = bkg + slope * E
    for s in steps:
        beta = beta + 0.5 * s["height"] * (1.0 + erf((E - s["e0"]) / s["width"]))
        beta = beta + s.get("decay", 0.0) * np.maximum(E - s["e0"], 0.0)
    return beta


def continuum_delta_from_table(E, e_table, delta_table):
    """Fixed continuum delta, precomputed by kkcalc, linearly interpolated."""
    return np.interp(E, e_table, delta_table)


# ------------------------------------------------------------------ assembly
def optical_model(E, peaks, continuum=None, delta_table=None, scale=1.0, shift=0.0):
    """Total delta + i beta = scale * (sum peaks + continuum)(E + shift).

    ``peaks``: list of (func, kwargs). ``delta_table``: (e, delta) arrays.
    """
    Es = np.asarray(E, dtype=float) + shift
    out = np.zeros(Es.shape, dtype=complex)
    for func, kw in peaks:
        out += func(Es, **kw)
    if continuum is not None:
        out += 1j * continuum_beta(Es, **continuum)
    if delta_table is not None:
        out += continuum_delta_from_table(Es, *delta_table)
    return scale * out


# ------------------------------------------------------- numerical KK checks
def hilbert_fft(E, beta):
    """delta = H[beta] by FFT (scipy.signal.hilbert); needs a uniform, wide grid."""
    from scipy.signal import hilbert

    return np.imag(hilbert(beta))


def hilbert_pv_direct(E_eval, E, beta):
    """Brute-force PV: (1/pi) sum beta(t)/(x - t) dt, skipping the singular point
    (midpoint-style; second-order accurate on a uniform grid)."""
    dE = E[1] - E[0]
    out = np.empty(len(E_eval))
    for i, x in enumerate(E_eval):
        d = x - E
        m = np.abs(d) > 0.5 * dE
        out[i] = np.sum(beta[m] / d[m]) * dE / np.pi
    return out
