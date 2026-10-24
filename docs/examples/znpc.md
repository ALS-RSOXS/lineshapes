# ZnPc carbon K-edge NEXAFS

Zinc phthalocyanine (ZnPc) thin-film NEXAFS at the carbon K-edge, measured at five incidence angles
(20°–90°). The goal: describe the spectrum as a sum of Gaussian peaks on a fixed continuum, then read
off the matching $\delta$ without any numerical KK transform.

Everything here runs from the data shipped in the package:

```python
from lineshapes import datasets, mu_to_beta

t = datasets.load_table("znpc_nexafs")       # energy_ev, theta_deg, mu_cm2_per_g, intensity_raw
m = t["theta_deg"] == 54.7                   # magic angle: orientation-independent
E = t["energy_ev"][m]
beta = mu_to_beta(E, t["mu_cm2_per_g"][m])   # beta = mu * rho * lambda / (4 pi), per unit density
```

The script is `examples/znpc_nexafs.py` in the repository; `make docs-figures` regenerates the figures.

## The recipe

1. Start from the material's **fixed continuum**: bare-atom $\beta$ (a line plus an erf ionization step
   at the C 1s ionization energy), with its $\delta$ from kkcalc2.
2. Add one Gaussian per **seed**. The seeds are DFT transition clusters: their energies and widths
   become the starting positions and widths.
3. Solve the peak heights linearly (non-negative least squares) for the starting guess.
4. Refine every peak's position, width and height, and the global density `scale`, with
   `scipy.optimize.least_squares`, evaluating through a bound `Evaluator`.

```python
--8<-- "examples/_common.py:fit"
```

## Fit

The starting guess already captures the structure; refinement removes the remaining mismatch. The
residual is at the level of the measurement noise, apart from a slow drift below the edge where the
molecular spectrum departs from the bare-atom continuum.

![ZnPc beta fit with starting guess, continuum and residual](../assets/generated/znpc-fit-light.png#only-light)
![ZnPc beta fit with starting guess, continuum and residual](../assets/generated/znpc-fit-dark.png#only-dark)

--8<-- "docs/assets/generated/znpc_peaks.md"

## Absorption and its Kramers-Kronig partner

Each Gaussian contributes a bell in $\beta$ and a dispersive Dawson-function shape in $\delta$ (thin gray
lines). The total $\delta$ is the fixed continuum table plus the sum of the analytic peak partners:
no transform was run to get it.

![Peak decomposition of beta and delta](../assets/generated/znpc-decomposition-light.png#only-light)
![Peak decomposition of beta and delta](../assets/generated/znpc-decomposition-dark.png#only-dark)

## Five angles

The same recipe, applied independently to each incidence angle, with the same seeds. The strong angular
dependence of the $\pi^*$ resonances near 285 eV is the dichroism of the oriented molecular film.
The library has no tensor machinery: each angle gives its own scalar optical constant, which is the building
block for orientation-resolved models.

![Beta fits for five incidence angles](../assets/generated/znpc-angles-light.png#only-light)
![Beta fits for five incidence angles](../assets/generated/znpc-angles-dark.png#only-dark)

--8<-- "docs/assets/generated/znpc_angles.md"

Fit times are for the machine that built these docs. Each fit refines 19 peaks (58 parameters including
the scale) on 599 energies.
