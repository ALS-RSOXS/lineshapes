"""Full Cu L-edge model: analytic peaks + kkcalc continuum table vs kkcalc on total beta.

Prototype result (prototypes/figures/03_cu_model_kkcalc.png, 2026-10-14):
max|model - kkcalc(total)| / max|kkcalc(total)| = 1.99e-2, residual ~1e-6
(mostly a near-constant offset). Threshold below: 2.5e-2.

NOTE: kkcalc2@e750040 output scales linearly with ``density`` (the stoichiometry
number density overrides it on one side of the conversion), so density=1.0 is
used, matching lr-optical-models bookended_*.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import CU_GRID, resolve_peaks

kkcalc2 = pytest.importorskip("kkcalc2")


def kkcalc_delta(E, beta, formula="Cu", density=1.0):
    from kkcalc2 import stoichiometry
    from kkcalc2.models import asf_im, asp_db_im, asp_db_im_extended

    st = stoichiometry(formula)
    asf = asf_im.from_betas(energies=E, absorption=beta, stoichiometry=st, density=density)
    kk = asp_db_im_extended(data_asf=asf, database=asp_db_im(st)).kk_transform(
        relativistic_correction=0.0, tolerance=0.01, max_iter=1
    )
    return np.interp(E, np.asarray(kk.energies), np.asarray(kk.deltas))


@pytest.fixture(scope="module")
def kk_reference(cu_model):
    """kkcalc delta of the continuum alone (the fixed table) and of the total beta."""
    from tests.reference import lineshapes_ref as ref

    peaks, cont = cu_model
    beta_bg = ref.continuum_beta(CU_GRID, **cont)
    beta_pk = sum(getattr(ref, n)(CU_GRID, **kw).imag for n, kw in peaks)
    return kkcalc_delta(CU_GRID, beta_bg), kkcalc_delta(CU_GRID, beta_bg + beta_pk)


@pytest.mark.kkcalc
def test_full_model_delta_matches_kkcalc(backend, cu_model, kk_reference):
    peaks, cont = cu_model
    d_bg, d_tot = kk_reference
    model = np.asarray(
        backend.optical_model(
            CU_GRID, resolve_peaks(backend, peaks), continuum=cont, delta_table=(CU_GRID, d_bg)
        )
    )
    rel = np.max(np.abs(model.real - d_tot)) / np.max(np.abs(d_tot))
    assert rel < 2.5e-2
    assert np.max(np.abs(model.real - d_tot)) < 2.5e-6


def test_full_model_beta_is_sum(backend, ref, cu_model):
    """Model beta is exactly continuum + peaks; scale/shift applied as documented."""
    peaks, cont = cu_model
    tab = (CU_GRID, np.zeros_like(CU_GRID))
    got = np.asarray(
        backend.optical_model(
            CU_GRID, resolve_peaks(backend, peaks), continuum=cont, delta_table=tab,
            scale=1.3, shift=0.2,
        )
    )
    Es = CU_GRID + 0.2
    beta = ref.continuum_beta(Es, **cont) + sum(getattr(ref, n)(Es, **kw).imag for n, kw in peaks)
    np.testing.assert_allclose(got.imag, 1.3 * beta, rtol=1e-12, atol=1e-18)
