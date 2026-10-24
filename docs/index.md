# lineshapes

Fast, composable optical constants for soft X-ray work. A model is a **sum of analytic peaks**
(Gaussian, Lorentzian, Voigt, pseudo-Voigt) on top of a **fixed continuum**, and evaluating it returns
the complex optical constant $\delta + i\beta$ on any energy grid.

The point of the design: every peak carries its **exact Kramers-Kronig partner**. When a fit changes
a peak's position, width or height, $\delta$ updates in closed form. No numerical KK transform runs
inside the fitting loop.

- **Composable.** Build a model by chaining `gaussian`, `voigt`, `step`, ... calls.
- **Fast.** A Rust core: about 0.2 ms for a dozen peaks plus continuum on 4000 energies, single core,
  with an optional multi-core path.
- **Made for fitting.** A flat parameter vector and a grid-bound `Evaluator` that recomputes only the
  peaks that changed, so finite-difference Jacobians are cheap.
- **Validated.** Checked against a numpy reference, numerical Hilbert transforms and
  [kkcalc2](https://github.com/xraysoftmat/kkcalc). See [Validation](validation.md).
- **Two packages, one version.** A pure-Rust crate (`lineshapes`) and a Python package
  (`lineshapes`) built on it.

## Install

Python (3.12+):

```bash
pip install lineshapes
```

The example scripts also need `scipy` and `matplotlib`: `pip install "lineshapes[examples]"`.

Rust:

```bash
cargo add lineshapes
```

## A first model

The package ships ZnPc and Alq3 carbon K-edge spectra, with each material's fixed continuum
([Concepts](concepts.md#the-continuum)). Add peaks on top of it and evaluate:

```python
import numpy as np
from lineshapes import datasets

E = np.linspace(280, 300, 2001)

model = (
    datasets.continuum_model("znpc")        # fixed continuum: beta and its delta table
    .gaussian(e0=285.2, width=0.45, amp=1.4e-3)   # a pi* resonance
    .gaussian(e0=293.0, width=3.0, amp=0.8e-3)    # a broad sigma* resonance
)

n = model(E)                    # complex128: delta + 1j*beta
delta, beta = n.real, n.imag
```

Then see the worked examples: [ZnPc NEXAFS](examples/znpc.md) and
[Alq3 optical constants](examples/alq3.md), which fit real spectra and reproduce an independent
$\delta$ table from a $\beta$ fit alone.
