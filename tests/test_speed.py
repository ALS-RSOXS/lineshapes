"""Speed: every non-reference backend must be at least as fast as the numpy reference.

The reference is timed in the same session (machine-independent comparison).
Reference baseline on the dev machine, 4001 points (2026-10-14):
gaussian 26 us, lorentzian 7 us, voigt 59 us, pseudo_voigt 37 us,
Cu optical_model (11 Gaussians + continuum + delta table) 400 us.
"""

from __future__ import annotations

import timeit

import numpy as np
import pytest

from tests.conftest import CU_GRID, resolve_peaks
from tests.test_lineshapes import PEAKS

pytestmark = pytest.mark.speed


def best_time(fn, number=200, repeat=7):
    fn()  # warm-up
    return min(timeit.repeat(fn, number=number, repeat=repeat)) / number


@pytest.mark.parametrize("name", list(PEAKS))
def test_peak_not_slower_than_reference(fast_backend, ref, name):
    kw = PEAKS[name]
    t_ref = best_time(lambda: getattr(ref, name)(CU_GRID, **kw))
    t_new = best_time(lambda: getattr(fast_backend, name)(CU_GRID, **kw))
    assert t_new <= t_ref, f"{name}: {t_new * 1e6:.1f} us > reference {t_ref * 1e6:.1f} us"


def test_cu_model_not_slower_than_reference(fast_backend, ref, cu_model):
    peaks, cont = cu_model
    tab = (CU_GRID, np.zeros_like(CU_GRID))
    p_ref, p_new = resolve_peaks(ref, peaks), resolve_peaks(fast_backend, peaks)
    t_ref = best_time(lambda: ref.optical_model(CU_GRID, p_ref, continuum=cont, delta_table=tab), 50)
    t_new = best_time(
        lambda: fast_backend.optical_model(CU_GRID, p_new, continuum=cont, delta_table=tab), 50
    )
    assert t_new <= t_ref, f"model: {t_new * 1e6:.1f} us > reference {t_ref * 1e6:.1f} us"
