"""Simulated fitting: randomly perturbed peak positions, heights, widths and density scale.

Each iteration mimics one optimizer step (perturb, rebuild, evaluate). Checks that
(1) every backend stays within 1e-12 of the numpy reference across random models, and
(2) the fit loop on a fast backend is not slower than the reference.
Scaling sweep + figure: `uv run python -m benchmarks.fit_sim`.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from tests.conftest import resolve_peaks

CONT = dict(
    bkg=3.6e-4, slope=-3.7e-7,
    steps=[dict(e0=939.4, width=6.1, height=7.2e-5, decay=0.0),
           dict(e0=960.7, width=2.2, height=5.1e-5, decay=1e-7)],
)
E = np.linspace(900.0, 1000.0, 4001)
TABLE = (E, 1e-4 * np.cos(E / 30.0))  # stand-in continuum delta
KINDS = ["gaussian", "lorentzian", "voigt", "pseudo_voigt"]


def random_model(rng, n_peaks, kinds=KINDS):
    """Random peak list over all lineshapes plus a random density scale and shift."""
    peaks = []
    for _ in range(n_peaks):
        kind = kinds[rng.integers(len(kinds))]
        e0, amp = rng.uniform(925, 975), rng.uniform(1e-6, 1e-4)
        kw = {
            "gaussian": dict(width=rng.uniform(0.2, 4.0)),
            "lorentzian": dict(gamma=rng.uniform(0.1, 2.0)),
            "voigt": dict(sigma=rng.uniform(0.1, 2.0), gamma=rng.uniform(0.0, 1.0)),
            "pseudo_voigt": dict(fwhm=rng.uniform(0.3, 4.0), eta=rng.uniform(0, 1)),
        }[kind]
        peaks.append((kind, dict(e0=e0, amp=amp, **kw)))
    return peaks, float(rng.uniform(0.5, 2.0)), float(rng.normal(0, 0.3))


@pytest.mark.parametrize("n_peaks", [1, 10, 50])
def test_random_models_match_reference(backend, ref, n_peaks):
    """30 random 'fit iterations' per size: backend == reference to 1e-12 (relative to max)."""
    rng = np.random.default_rng(n_peaks)
    for _ in range(30):
        peaks, scale, shift = random_model(rng, n_peaks)
        kw = dict(continuum=CONT, delta_table=TABLE, scale=scale, shift=shift)
        got = np.asarray(backend.optical_model(E, resolve_peaks(backend, peaks), **kw))
        want = ref.optical_model(E, resolve_peaks(ref, peaks), **kw)
        err = np.max(np.abs(got - want)) / np.max(np.abs(want))
        assert err < 1e-12, f"rel err {err:.2e} for {peaks}"


def fit_loop_time(backend, n_peaks, iters, seed=0):
    """Mean seconds per iteration: perturb (e0, width, amp, scale), rebuild, evaluate."""
    rng = np.random.default_rng(seed)
    e0 = np.sort(rng.uniform(925, 975, n_peaks))
    width = rng.uniform(0.3, 3.0, n_peaks)
    amp = rng.uniform(1e-5, 8e-5, n_peaks)
    t0 = time.perf_counter()
    for _ in range(iters):
        de, dw, da = (rng.normal(0, s, n_peaks) for s in (0.05, 0.02, 0.02))
        peaks = [(backend.gaussian, dict(e0=a, width=b, amp=c))
                 for a, b, c in zip(e0 + de, width * np.exp(dw), amp * np.exp(da))]
        backend.optical_model(E, peaks, continuum=CONT, delta_table=TABLE,
                              scale=float(np.exp(rng.normal(0, 0.01))))
    return (time.perf_counter() - t0) / iters


@pytest.mark.speed
@pytest.mark.parametrize("n_peaks", [10, 50])
def test_fit_loop_not_slower_than_reference(fast_backend, ref, n_peaks):
    fit_loop_time(fast_backend, n_peaks, 3)  # warm-up
    t_ref = min(fit_loop_time(ref, n_peaks, 20) for _ in range(3))
    t_new = min(fit_loop_time(fast_backend, n_peaks, 20) for _ in range(3))
    assert t_new <= t_ref, f"{t_new * 1e6:.0f} us > reference {t_ref * 1e6:.0f} us"
