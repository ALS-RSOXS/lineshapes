"""Unit conversions between absorption measurements and the optical constant beta."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

#: h*c in eV*Angstrom.
HC_EV_ANGSTROM = 12398.419843320026

FloatArray = npt.NDArray[np.float64]


def wavelength_angstrom(energy_ev: npt.ArrayLike) -> FloatArray:
    """Photon wavelength in Angstrom for energies in eV."""
    return HC_EV_ANGSTROM / np.asarray(energy_ev, dtype=float)


def mu_to_beta(energy_ev: npt.ArrayLike, mu_cm2_per_g: npt.ArrayLike, density_g_cm3: float = 1.0) -> FloatArray:
    """Convert a mass absorption coefficient to beta: ``beta = mu * rho * lambda / (4 pi)``.

    Args:
        energy_ev: Photon energy in eV.
        mu_cm2_per_g: Mass absorption coefficient in cm^2/g.
        density_g_cm3: Mass density in g/cm^3. The example tables are per unit density (1.0); the
            film density is applied later as the model ``scale``.
    """
    lam_cm = wavelength_angstrom(energy_ev) * 1e-8
    return np.asarray(mu_cm2_per_g, dtype=float) * density_g_cm3 * lam_cm / (4.0 * np.pi)


def beta_to_mu(energy_ev: npt.ArrayLike, beta: npt.ArrayLike, density_g_cm3: float = 1.0) -> FloatArray:
    """Inverse of [`mu_to_beta`][lineshapes.units.mu_to_beta]."""
    lam_cm = wavelength_angstrom(energy_ev) * 1e-8
    return 4.0 * np.pi * np.asarray(beta, dtype=float) / (density_g_cm3 * lam_cm)
