# lineshapes (WebAssembly / TypeScript)

WebAssembly and TypeScript/TSX bindings for the [`lineshapes`](https://github.com/ALS-RSOXS/lineshapes) optical-constant modeling library.

Provides composable optical-constant models (`Model`) and grid-bound evaluators (`Evaluator`) for soft X-ray NEXAFS / RSoXS work, returning `delta` and `beta` (`n = 1 - delta + i*beta`) on an energy grid in eV.

## Installation

```sh
bun add lineshapes
# or: npm install lineshapes
```

## Usage in TypeScript / TSX

The WASM module is embedded and initializes automatically on first use in Bun, Node.js, and Web Workers (or asynchronously via `await init()` when preloading on the browser main thread):

```ts
import { Model, muToBeta } from "lineshapes";

const energy = new Float64Array([930.0, 933.8, 936.0, 938.3, 940.0]);

// Build by chaining (supports both positional arguments and named parameter objects)
const model = new Model()
  .gaussian({ e0: 933.8, width: 0.49, amp: 4.1e-5 })
  .lorentzian({ e0: 936.0, gamma: 0.3, amp: 2e-5 })
  .voigt({ e0: 938.3, sigma: 0.7, gamma: 0.2, amp: 5e-5 })
  .pseudoVoigt({ e0: 955.0, fwhm: 1.5, eta: 0.4, amp: 2e-5 })
  .continuum({ bkg: 3.6e-4, slope: -3.7e-7 })
  .step({ e0: 939.4, width: 6.1, height: 7.2e-5, decay: 0.0 })
  .scale(1.0)
  .shift(0.0);

const { delta, beta } = model.eval(energy);
```

### Fitting with Flat Parameter Vectors and `Evaluator`

During a fit, update parameters via the flat parameter vector or bind the model to a fixed energy grid so only modified components are recomputed:

```ts
const ev = model.bind(energy); // caches each component's contribution on `energy`
const theta = ev.params;       // Float64Array of length ev.nParams
console.log(ev.paramNames);    // ['gaussian[0].e0', ..., 'continuum.bkg', 'continuum.slope', 'scale', 'shift']

// Preallocate output buffers for zero-allocation evaluation inside fitting loops:
const out = {
  delta: new Float64Array(energy.length),
  beta: new Float64Array(energy.length),
};

theta[0] += 0.05; // perturb gaussian[0].e0 -> only gaussian[0] is recomputed
ev.eval(theta, out);
```

### Declarative Specification (React / TSX state)

```ts
import { opticalModel, type ComponentSpec } from "lineshapes";

const components: ComponentSpec[] = [
  { kind: "gaussian", e0: 933.8, width: 0.49, amp: 4.1e-5 },
  { kind: "voigt", e0: 938.3, sigma: 0.7, gamma: 0.2, amp: 5e-5 },
  { kind: "step", e0: 939.4, width: 6.1, height: 7.2e-5 },
];

const { delta, beta } = opticalModel(energy, {
  components,
  continuum: { bkg: 3.6e-4, slope: -3.7e-7 },
  scale: 1.0,
  shift: 0.0,
});
```

## Building & Testing

```sh
bun crates/lineshapes-wasm/scripts/build.ts   # compile wasm32-unknown-unknown + wasm-bindgen -> pkg/
bun test crates/lineshapes-wasm               # run the Bun test suite
cargo test -p lineshapes-wasm --release       # run the Rust unit tests
```
