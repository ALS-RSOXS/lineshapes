# Concepts

## The model

For photon energy $E$ in eV the model returns the complex quantity

$$
\delta(E) + i\,\beta(E) \;=\; s\Big[\;\sum_k f_k(E+\Delta) \;+\; i\,\beta_\text{cont}(E+\Delta) \;+\; \delta_\text{cont}(E+\Delta)\Big]
$$

with the refractive index $n = 1 - \delta + i\beta$. Here $f_k$ are the peaks, $\beta_\text{cont}$ is a
parametric continuum in $\beta$, $\delta_\text{cont}$ is a fixed continuum $\delta$ table, $s$ is an overall
`scale` (typically a density) and $\Delta$ is an energy `shift`.

## Peaks come in Kramers-Kronig pairs

The real and imaginary parts of each peak are a Hilbert-transform pair,
$\delta = \mathcal{H}[\beta]$ with $\mathcal{H}f(x) = \frac{1}{\pi}\,\mathrm{PV}\!\!\int \frac{f(t)}{x-t}\,dt$.
So $\delta < 0$ below a resonance and $\delta > 0$ above it. Resonances sit far from $0\,\text{eV}$, so
no mirror term at $-E_0$ is used. The amplitude `amp` is the height of $\beta$ at the peak center for
every lineshape.

| Lineshape | Parameters | $\beta$ and $\delta$ |
|---|---|---|
| `gaussian` | `e0`, `width`, `amp` | $x=\frac{E-E_0}{w}$; $\beta=A e^{-x^2}$, $\delta=\frac{2A}{\sqrt{\pi}}\,D(x)$ with $D$ Dawson's function |
| `lorentzian` | `e0`, `gamma`, `amp` | $\delta+i\beta=\dfrac{A\gamma}{E-E_0-i\gamma}$ |
| `voigt` | `e0`, `sigma`, `gamma`, `amp` | $z=\frac{E-E_0+i\gamma}{\sqrt2\sigma}$; $\beta=\frac{A}{N}\mathrm{Re}\,w(z)$, $\delta=\frac{A}{N}\mathrm{Im}\,w(z)$, $N=\mathrm{Re}\,w\!\left(\tfrac{i\gamma}{\sqrt2\sigma}\right)$, $w$ the Faddeeva function |
| `pseudo_voigt` | `e0`, `fwhm`, `eta`, `amp` | $\eta\,L+(1-\eta)\,G$ with one shared FWHM; exact by linearity |

Width conventions, since they differ between programs:

- **Gaussian `width`** is the $1/e$ half-width: $\beta$ falls to $A/e$ at $E_0\pm w$, so $\sigma=w/\sqrt2$ and
  $\mathrm{FWHM}=2\sqrt{\ln 2}\,w$.
- **Lorentzian `gamma`** is the half width at half maximum (HWHM).
- **Voigt** takes the Gaussian standard deviation `sigma` and the Lorentzian HWHM `gamma`.
- **Pseudo-Voigt** uses a single `fwhm` for both parts and the Lorentzian fraction `eta`.

## The continuum

Real spectra sit on a smooth background with an ionization step. The library models it in $\beta$:
`continuum(bkg, slope)` gives $\beta_\text{bkg}+\text{slope}\cdot E$, and each `step(e0, width, height, decay)` adds
$\tfrac{h}{2}\left[1+\mathrm{erf}\!\left(\tfrac{E-E_0}{w}\right)\right]+\text{decay}\cdot\max(E-E_0,0)$.

The $\delta$ that goes with a continuum and step has no simple closed form (a step does not decay, so its
Hilbert transform needs a reference energy). Instead, the continuum $\delta$ is **fixed ahead of time**
with a full numerical KK transform ([kkcalc2](https://github.com/xraysoftmat/kkcalc), which also folds in
the bare-atom behavior far from the edge) and supplied as a table with `delta_table(energy, delta)`. Only
the peaks are fit; the continuum, and so its $\delta$, stays fixed.

!!! note "kkcalc2 and density"
    With the pinned kkcalc2 version the computed $\delta$ scales linearly with its `density` argument.
    The shipped tables are per unit density (`density=1.0`) and the film density enters as the model `scale`.

## Units

The example tables are per unit density (1 g/cm³). Absorption data given as a mass absorption
coefficient $\mu$ (cm²/g) convert to $\beta$ with

$$
\beta = \frac{\mu\,\rho\,\lambda}{4\pi}, \qquad \lambda = \frac{hc}{E},
$$

available as `lineshapes.mu_to_beta(energy_ev, mu, density)` (and `beta_to_mu`).

## What is not in the model

No fitting, no reflectivity, no tensor or orientation algebra. The library builds the optical-constant
callable and gets out of the way: pair it with `scipy.optimize` for fitting (see
[Fitting without rebuilding](guide/fitting.md)) and with a reflectivity code that takes $\delta$ and $\beta$
per energy.
