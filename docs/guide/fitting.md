# Fitting without rebuilding

Build the model once with the chained methods. During a fit, never rebuild it: use the flat parameter
vector, and bind the energy grid so unchanged peaks are not recomputed.

## The parameter vector

```python
m = (Model()
     .gaussian(933.8, 0.49, 4.1e-5)
     .voigt(938.3, 0.7, 0.2, 5e-5)
     .step(939.4, 6.1, 7.2e-5)
     .continuum(3.6e-4, -3.7e-7))

m.param_names   # ['gaussian[0].e0', 'gaussian[0].width', 'gaussian[0].amp', 'voigt[0].e0', ..., 'scale', 'shift']
theta = m.params                # flat float64 array
```

Layout: each component's parameters in the order you added it, then `continuum.bkg`, `continuum.slope`,
`scale`, `shift`. A pseudo-Voigt is one component, so `eta` is a fittable parameter.

| operation | effect |
|---|---|
| `m(E, params=theta)` | evaluate at `theta`; the model is unchanged |
| `m.set_params(theta)` | update the model in place (validated: bad values raise `ValueError`, model unchanged) |

## Bind a grid

```python
ev = m.bind(E)              # fix the energy grid once
n = ev(theta)               # set parameters and evaluate
n = ev(theta, out=buf)      # same, into a preallocated complex128 array
```

The evaluator caches each component's contribution. A call recomputes only the components whose
parameters changed since the previous call:

- `scale`, `continuum.bkg` and `continuum.slope` cost no peak work at all.
- Changing one peak costs one peak, not all of them. This is exactly the access pattern of a
  finite-difference Jacobian, which perturbs one parameter at a time.
- `shift` moves every component, so it recomputes all of them.

## Use it with scipy

```python
import numpy as np
from scipy.optimize import least_squares

ev = model.bind(E)
buf = np.empty(E.size, complex)
theta = model.params.copy()
free = np.array([n.startswith("gaussian") or n == "scale" for n in model.param_names])

def residual(p):
    theta[free] = p
    return ev(theta, out=buf).imag - beta      # fit beta; delta comes along for free

sol = least_squares(residual, theta[free], x_scale="jac")
```

The complete version, with bounds and DFT-seeded starts, is in the [ZnPc example](../examples/znpc.md#the-recipe).
Freezing a parameter is just leaving it out of `free`.

## How much does it save?

Forward finite-difference Jacobian, one parameter changed per column, Gaussians plus two steps, a delta table
and a scale, on 4001 energies (Apple Silicon, single core):

| peaks | rebuild the model per column | bound evaluator (cached) | speedup |
|--:|--:|--:|--:|
| 20 | 22.7 ms | 2.6 ms | 9× |
| 50 | 106 ms | 5.8 ms | 18× |
| 100 | 498 ms | 12.0 ms | 41× |

A full optimizer step where every parameter changes costs the same either way: the time is the peaks
themselves (about 0.6 ms for 50 peaks on 4001 energies). Reproduce with
`python -m benchmarks.no_rebuild`.

## Evaluation speed

- Energies sorted ascending take a faster path: far from each peak the tails use asymptotic series,
  and the delta table is walked with a moving pointer.
- `m(E, parallel=True)` uses all cores and returns identical results. It already helps at a few thousand
  energies (the Cu model on 4001 points: about 1.6× faster) and approaches linear scaling on large grids
  (400,000 points: 12.8 ms down to 3.2 ms on this machine).
- The Rust crate exposes the same model, parameter vector and evaluator; see the [Rust API](../api/rust.md).

!!! info "Analytic Jacobians"
    Each peak's derivatives with respect to position, width and height have closed forms, which would remove
    the per-column cost entirely. Not implemented yet; the evaluator is the supported path today.
