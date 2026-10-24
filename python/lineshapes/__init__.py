"""Fast composable optical-constant models (Rust core).

Build a model by chaining, then call it for ``delta + 1j*beta``::

    from lineshapes import Model
    m = (Model()
         .gaussian(e0=933.8, width=0.49, amp=4.1e-5)
         .voigt(e0=935.7, sigma=0.8, gamma=0.3, amp=7.4e-5)
         .continuum(bkg=3.6e-4, slope=-3.7e-7)
         .step(e0=939.4, width=6.1, height=7.2e-5)
         .delta_table(energy, delta_bg))      # fixed continuum delta (e.g. kkcalc)
    n = m(energy)                  # single core
    n = m(energy, parallel=True)   # all cores

Conventions: energies in eV; ``delta = H[beta]`` (delta < 0 below a resonance);
``amp`` is the beta peak height. Widths: gaussian ``width`` = 1/e half-width
(sigma = width/sqrt(2)); lorentzian ``gamma`` = HWHM; voigt ``sigma`` (Gaussian std)
and ``gamma`` (HWHM); pseudo_voigt shared ``fwhm`` and mixing ``eta``.

Fitting without rebuilding: the model exposes a flat parameter vector
(``m.params``, ``m.param_names``, ``m.set_params(theta)``, ``m(E, params=theta)``), and
``ev = m.bind(E)`` fixes the energy grid and caches each component's contribution so
``ev(theta)`` recomputes only the components whose parameters changed.

The module-level functions below are one-shot conveniences (they build a
throwaway model); for repeated evaluation build a ``Model`` once.
"""

from __future__ import annotations

import numpy as np

from lineshapes import datasets, units
from lineshapes._core import Evaluator, Model
from lineshapes.units import beta_to_mu, mu_to_beta

__all__ = [
    "Evaluator",
    "Model",
    "beta_to_mu",
    "datasets",
    "mu_to_beta",
    "units",
    "gaussian",
    "lorentzian",
    "voigt",
    "pseudo_voigt",
    "continuum_beta",
    "optical_model",
]


def _e(E) -> np.ndarray:
    return np.ascontiguousarray(E, dtype=np.float64)


def gaussian(E, e0, width, amp):
    return Model().gaussian(e0, width, amp)(_e(E))


def lorentzian(E, e0, gamma, amp):
    return Model().lorentzian(e0, gamma, amp)(_e(E))


def voigt(E, e0, sigma, gamma, amp):
    return Model().voigt(e0, sigma, gamma, amp)(_e(E))


def pseudo_voigt(E, e0, fwhm, eta, amp):
    return Model().pseudo_voigt(e0, fwhm, eta, amp)(_e(E))


def _add_continuum(m: Model, bkg=0.0, slope=0.0, steps=()) -> Model:
    m.continuum(bkg, slope)
    for s in steps:
        m.step(s["e0"], s["width"], s["height"], s.get("decay", 0.0))
    return m


def continuum_beta(E, bkg=0.0, slope=0.0, steps=()):
    return _add_continuum(Model(), bkg, slope, steps)(_e(E)).imag


def optical_model(E, peaks, continuum=None, delta_table=None, scale=1.0, shift=0.0):
    """Function-style API matching tests/reference: ``peaks`` = [(func, kwargs)]."""
    m = Model()
    for func, kw in peaks:
        getattr(m, func.__name__)(**kw)
    if continuum is not None:
        _add_continuum(m, **continuum)
    if delta_table is not None:
        m.delta_table(_e(delta_table[0]), _e(delta_table[1]))
    return m.scale(scale).shift(shift)(_e(E))
