# lineshapes

Fast, composable optical-constant models for soft X-ray work. A model is a sum of
**analytic Kramers-Kronig-paired peaks** (Gaussian, Lorentzian, Voigt, pseudo-Voigt) plus a
**continuum** parameterized in beta (linear background + erf ionization steps) and a
**fixed continuum delta table** (e.g. precomputed once with kkcalc). Rust core, Python bindings.

Evaluating a model returns `delta + 1j*beta` (n = 1 - delta + i beta) on an energy grid in eV.

## Python

```python
import numpy as np
from lineshapes import Model

E = np.linspace(900, 1000, 4001)
m = (Model()
     .gaussian(e0=933.8, width=0.49, amp=4.1e-5)        # width = 1/e half-width
     .lorentzian(e0=936.0, gamma=0.3, amp=2e-5)         # gamma = HWHM
     .voigt(e0=938.3, sigma=0.7, gamma=0.2, amp=5e-5)   # sigma = Gaussian std, gamma = HWHM
     .pseudo_voigt(e0=955.0, fwhm=1.5, eta=0.4, amp=2e-5)
     .continuum(bkg=3.6e-4, slope=-3.7e-7)
     .step(e0=939.4, width=6.1, height=7.2e-5, decay=0.0)
     .delta_table(E, delta_continuum)                   # fixed continuum delta (kkcalc)
     .scale(1.0).shift(0.0))
n = m(E)                   # complex128, single core
n = m(E, parallel=True)    # all cores
```

Builder methods mutate the model and return it; `m.copy()` gives an independent copy to extend.
Build once, call many times.

Conventions:
- `delta = H[beta]`, so delta < 0 below a resonance and > 0 above it. No mirror term at -E0.
- `amp` is the height of beta at `e0` for every lineshape.
- The pseudo-Voigt is stored as one Lorentzian plus one Gaussian, so its delta is exact.

## Documentation

`make docs` builds the site (mkdocs with the `mkdocs_rsoxs` theme) into `site/`; `make docs-serve` previews it.
The docs run the ZnPc and Alq3 examples (`examples/`, data shipped in `lineshapes.datasets`) to render their
figures; `make docs-figures` regenerates them. The theme needs Python >= 3.13, so docs use their own
environment (`.venv-docs`); the library itself supports Python >= 3.12.

## Layout

```
crates/lineshapes/      pure-Rust crate `lineshapes` (no Python/pyo3): Model, Evaluator, special functions
crates/lineshapes-py/   thin PyO3 bindings crate -> lineshapes._core (not published to crates.io)
python/lineshapes/        Python package (re-exports, function API, type stubs)
docs/ examples/ tools/   mkdocs site, example scripts, data/docs generators
tests/ benchmarks/ prototypes/   pytest suite vs numpy reference, timings, figures
```

Each half stands alone: `cargo add lineshapes` for Rust only (feature `parallel`, on by default,
adds rayon), or `pip install lineshapes` for Python (bundles the compiled core). See
`crates/lineshapes/README.md` for Rust usage.

## Development

```sh
cargo test --workspace --release                          # Rust unit, integration and doc tests
uv sync                                                   # builds the extension (release)
uv run --with maturin maturin develop --release --uv      # rebuild after Rust changes
uv run pytest                                             # precision + speed vs numpy reference
uv run python -m benchmarks.bench                         # timings
```

- `strip = false` (workspace `[profile.release]`) is required: stripping corrupts the extension
  on this macOS toolchain ("mis-aligned LINKEDIT string pool").

Validation (`tests/`) compares against a pure-numpy reference (`tests/reference/lineshapes_ref.py`),
numerical Hilbert transforms, and kkcalc2 (pinned commit `e750040`) on a real Cu L-edge model.
Note that kkcalc2's output scales with its `density` argument. Use `density=1.0`.
