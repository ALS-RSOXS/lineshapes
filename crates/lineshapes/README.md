# lineshapes

Fast, composable optical-constant models for soft X-ray work. A [`Model`] is a sum of
analytic Kramers-Kronig-paired peaks (Gaussian, Lorentzian, Voigt, pseudo-Voigt), a continuum
in beta (linear background + erf ionization steps) and a fixed continuum delta table.
Evaluating it returns `delta + i*beta` on an energy grid in eV.

```rust
use lineshapes::Model;

let mut m = Model::new()
    .gaussian(933.8, 0.49, 4.1e-5)            // e0, 1/e half-width, beta height
    .voigt(938.3, 0.7, 0.2, 5e-5)             // e0, sigma, gamma (HWHM), height
    .continuum(3.6e-4, -3.7e-7)
    .step(939.4, 6.1, 7.2e-5, 0.0);
let n = m.eval(&energies);                    // Vec<Complex64>, single core
let n = m.eval_par(&energies);                // feature "parallel" (default), rayon

// Fitting: update through the flat parameter vector instead of rebuilding.
let mut theta = m.params();
theta[0] += 0.01;
m.set_params(&theta)?;

// Or bind a fixed grid; only components whose parameters changed are recomputed.
let mut ev = lineshapes::Evaluator::new(m, &energies, true);
ev.eval_params_into(&theta, &mut out)?;
```

No Python dependency. The Python bindings live in the sibling crate `lineshapes-py`
(package `lineshapes`). Conventions: `delta = H[beta]` (delta < 0 below a resonance, no mirror
term at -E0); `amp` is the beta peak height.
