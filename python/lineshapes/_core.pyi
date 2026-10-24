"""Compiled core: the composable optical-constant `Model` and the grid-bound `Evaluator`."""

from typing import Self

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]

class Model:
    """Composable optical-constant model; calling it returns ``delta + 1j*beta``.

    Build by chaining; every builder method mutates the model and returns it. Energies are in eV.
    ``delta`` is the exact Hilbert partner of ``beta`` for each peak (delta < 0 below a resonance),
    and ``amp`` is the height of ``beta`` at the peak center for every lineshape.

    Example:
        ```python
        m = (Model()
             .gaussian(e0=933.8, width=0.49, amp=4.1e-5)
             .continuum(bkg=3.6e-4, slope=-3.7e-7)
             .step(e0=939.4, width=6.1, height=7.2e-5))
        n = m(energy)                 # complex128, single core
        n = m(energy, parallel=True)  # all cores
        ```

    The model also exposes a flat parameter vector (`params`, `param_names`, `set_params`) for
    fitting without rebuilding, and `bind` to fix an energy grid for repeated evaluation. A new
    model is empty and evaluates to zero.
    """

    def __init__(self) -> None: ...
    def gaussian(self, e0: float, width: float, amp: float) -> Self:
        """Add a Gaussian peak with its Dawson-function delta.

        Args:
            e0: Center in eV.
            width: 1/e half-width in eV: beta falls to ``amp/e`` at ``e0 +- width`` (``sigma = width/sqrt(2)``).
            amp: Peak height of beta.
        """
    def lorentzian(self, e0: float, gamma: float, amp: float) -> Self:
        """Add a Lorentzian peak, ``delta + i beta = amp*gamma / (E - e0 - i*gamma)``.

        Args:
            e0: Center in eV.
            gamma: Half width at half maximum in eV.
            amp: Peak height of beta.
        """
    def voigt(self, e0: float, sigma: float, gamma: float, amp: float) -> Self:
        """Add an exact Voigt peak (Gaussian convolved with Lorentzian), via the Faddeeva function.

        Args:
            e0: Center in eV.
            sigma: Gaussian standard deviation in eV.
            gamma: Lorentzian half width at half maximum in eV (``>= 0``).
            amp: Peak height of beta.
        """
    def pseudo_voigt(self, e0: float, fwhm: float, eta: float, amp: float) -> Self:
        """Add a pseudo-Voigt ``eta*Lorentzian + (1-eta)*Gaussian`` sharing one FWHM.

        Its delta is exact by linearity. It is a single component with its own ``eta`` parameter.

        Args:
            e0: Center in eV.
            fwhm: Full width at half maximum of both parts, in eV.
            eta: Lorentzian fraction in ``[0, 1]``.
            amp: Peak height of beta.
        """
    def continuum(self, bkg: float = 0.0, slope: float = 0.0) -> Self:
        """Set the linear continuum in beta, ``bkg + slope*E`` (replaces any previous one)."""
    def step(self, e0: float, width: float, height: float, decay: float = 0.0) -> Self:
        """Add an erf ionization step to beta: ``0.5*height*(1 + erf((E-e0)/width)) + decay*max(E-e0, 0)``.

        Only beta is modeled analytically; the matching delta belongs in the fixed delta table.

        Args:
            e0: Edge energy in eV.
            width: Erf width in eV.
            height: Step height of beta.
            decay: Linear post-edge slope added above ``e0``.
        """
    def delta_table(self, energy: FloatArray, delta: FloatArray) -> Self:
        """Set the fixed continuum delta (e.g. computed once with kkcalc), added to ``delta``.

        Interpolated linearly like ``numpy.interp`` and clamped to the end values outside the table.
        It is not a fit parameter.

        Args:
            energy: Strictly increasing energies in eV.
            delta: Delta values at ``energy``.
        """
    def scale(self, scale: float) -> Self:
        """Set the overall multiplicative scale (e.g. the density relative to a unit-density table)."""
    def shift(self, shift: float) -> Self:
        """Set the energy offset: the model is evaluated at ``E + shift``."""
    def copy(self) -> Model:
        """Return an independent copy, so a base model can be extended in several ways."""
    @property
    def n_components(self) -> int:
        """Number of peaks and steps (a pseudo-Voigt counts as one)."""
    @property
    def n_params(self) -> int:
        """Length of the flat parameter vector."""
    @property
    def params(self) -> FloatArray:
        """Flat parameter vector: each component's parameters in insertion order, then
        ``continuum.bkg, continuum.slope, scale, shift``."""
    @property
    def param_names(self) -> list[str]:
        """Names matching `params`, e.g. ``'gaussian[0].e0'``, ``'step[1].height'``, ``'scale'``."""
    def set_params(self, params: FloatArray) -> Self:
        """Set all parameters from a flat vector.

        Raises:
            ValueError: On a wrong length or an invalid value (non-positive width, ``eta`` outside
                ``[0, 1]``, non-finite entries). The model is unchanged on error.
        """
    def bind(self, energy: FloatArray, cache: bool = True) -> Evaluator:
        """Bind the model to a fixed energy grid for fitting loops.

        Args:
            energy: Energy grid in eV.
            cache: Keep each component's contribution (``n_components * len(energy) * 16`` bytes) so a
                call recomputes only components whose parameters changed.
        """
    def __call__(
        self, energy: FloatArray, params: FloatArray | None = None, parallel: bool = False
    ) -> ComplexArray:
        """Evaluate ``delta + 1j*beta`` on ``energy``.

        Args:
            energy: 1-D float64 energies in eV (sorted input is fastest).
            params: Evaluate at this parameter vector without modifying the model.
            parallel: Use all cores; results are identical to the single-core path.
        """

class Evaluator:
    """A `Model` bound to a fixed energy grid; recomputes only the components that changed.

    Created by `Model.bind`. Changing ``scale``, ``continuum.bkg`` or ``continuum.slope`` costs no
    peak work; changing ``shift`` recomputes every component.

    Example:
        ```python
        ev = model.bind(energy)
        theta = ev.params.copy()
        n = ev(theta)
        theta[0] += 1e-3          # one peak moves
        n = ev(theta, out=buf)    # only that peak is recomputed; buf is a preallocated complex128 array
        ```
    """

    @property
    def n_params(self) -> int:
        """Length of the flat parameter vector."""
    @property
    def params(self) -> FloatArray:
        """Current parameter vector."""
    @property
    def param_names(self) -> list[str]:
        """Names matching `params`."""
    @property
    def energy(self) -> FloatArray:
        """The bound energy grid."""
    @property
    def model(self) -> Model:
        """A copy of the bound model at the current parameters."""
    def set_params(self, params: FloatArray) -> None:
        """Set the parameters without evaluating."""
    def __call__(
        self, params: FloatArray | None = None, out: ComplexArray | None = None
    ) -> ComplexArray:
        """Evaluate at ``params`` (or the current parameters).

        Args:
            params: Flat parameter vector to set first.
            out: Optional contiguous complex128 array of the grid length to write into.
        """
