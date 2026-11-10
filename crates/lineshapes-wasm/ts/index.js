import __wbg_init, {
  Evaluator as RawEvaluator,
  Model as RawModel,
  betaToMu as rawBetaToMu,
  fastDawson as rawFastDawson,
  fastErf as rawFastErf,
  fastW as rawFastW,
  initSync as rawInitSync,
  muToBeta as rawMuToBeta,
  wavelengthAngstrom as rawWavelengthAngstrom,
} from "./lineshapes_wasm.js";
import { getWasmBytes } from "./wasm_bytes.js";

const RAW = Symbol("lineshapes.raw");
const FROM_RAW = Symbol("lineshapes.fromRaw");

let initialized = false;

/**
 * Synchronously initialize the WebAssembly module.
 * Called automatically on first use if `init()` or `initSync()` has not been called yet.
 *
 * @param {BufferSource | WebAssembly.Module | { module: BufferSource | WebAssembly.Module }} [module]
 */
export function initSync(module) {
  const input =
    module !== undefined &&
    module !== null &&
    typeof module === "object" &&
    Object.getPrototypeOf(module) === Object.prototype &&
    "module" in module
      ? module.module
      : (module ?? getWasmBytes());
  const out = rawInitSync({ module: input });
  initialized = true;
  return out;
}

/**
 * Asynchronously initialize the WebAssembly module.
 * Uses the embedded WASM binary by default so no external `.wasm` network fetch or bundler loader is required.
 *
 * @param {RequestInfo | URL | Response | BufferSource | WebAssembly.Module | Promise<RequestInfo | URL | Response | BufferSource | WebAssembly.Module> | { module_or_path: any }} [moduleOrPath]
 */
export async function init(moduleOrPath) {
  const input =
    moduleOrPath !== undefined &&
    moduleOrPath !== null &&
    typeof moduleOrPath === "object" &&
    Object.getPrototypeOf(moduleOrPath) === Object.prototype &&
    "module_or_path" in moduleOrPath
      ? moduleOrPath.module_or_path
      : (moduleOrPath ?? getWasmBytes());
  const out = await __wbg_init({ module_or_path: input });
  initialized = true;
  return out;
}

export default init;

function ensureInit() {
  if (!initialized) {
    initSync();
  }
}

function isPlainObject(v) {
  return (
    v !== null &&
    typeof v === "object" &&
    !ArrayBuffer.isView(v) &&
    !Array.isArray(v)
  );
}

/** Planck's constant times the speed of light (`h * c`) in `eV * Angstrom`. */
export const HC_EV_ANGSTROM = 12398.419843320026;

/**
 * Composable optical-constant model returning `delta + i*beta` (`n = 1 - delta + i*beta`)
 * on an energy grid in eV.
 *
 * Build by chaining (every builder method mutates the model in-place and returns `this`),
 * then evaluate with `.eval(energy)`.
 */
export class Model {
  constructor(token, raw) {
    ensureInit();
    if (token === FROM_RAW) {
      this[RAW] = raw;
    } else {
      this[RAW] = new RawModel();
    }
  }

  /**
   * Build a `Model` from a declarative options object (components, continuum, deltaTable, scale, shift).
   *
   * @param {import("./index.js").OpticalModelOptions} options
   * @returns {Model}
   */
  static fromOptions(options = {}) {
    const m = new Model();
    if (options.components) {
      for (const comp of options.components) {
        m.addComponent(comp);
      }
    }
    if (options.continuum) {
      const { bkg = 0.0, slope = 0.0, steps } = options.continuum;
      m.continuum(bkg, slope);
      if (steps) {
        for (const s of steps) {
          m.step(s.e0, s.width, s.height, s.decay ?? 0.0);
        }
      }
    }
    if (options.deltaTable) {
      m.deltaTable(options.deltaTable.energy, options.deltaTable.delta);
    }
    if (options.scale !== undefined) {
      m.scale(options.scale);
    }
    if (options.shift !== undefined) {
      m.shift(options.shift);
    }
    return m;
  }

  /**
   * Add a component from a tagged component specification object.
   *
   * @param {import("./index.js").ComponentSpec} spec
   * @returns {this}
   */
  addComponent(spec) {
    switch (spec.kind) {
      case "gaussian":
        return this.gaussian(spec.e0, spec.width, spec.amp);
      case "lorentzian":
        return this.lorentzian(spec.e0, spec.gamma, spec.amp);
      case "voigt":
        return this.voigt(spec.e0, spec.sigma, spec.gamma, spec.amp);
      case "pseudoVoigt":
      case "pseudo_voigt":
        return this.pseudoVoigt(spec.e0, spec.fwhm, spec.eta, spec.amp);
      case "step":
        return this.step(spec.e0, spec.width, spec.height, spec.decay ?? 0.0);
      default:
        throw new Error(`Unknown component kind: ${String(spec?.kind)}`);
    }
  }

  /**
   * Add a Gaussian peak; `width` is the 1/e half-width (`sigma = width / sqrt(2)`).
   */
  gaussian(e0OrParams, width, amp) {
    if (isPlainObject(e0OrParams)) {
      this[RAW].gaussian(e0OrParams.e0, e0OrParams.width, e0OrParams.amp);
    } else {
      this[RAW].gaussian(e0OrParams, width, amp);
    }
    return this;
  }

  /**
   * Add a Lorentzian peak; `gamma` is the half width at half maximum (HWHM).
   */
  lorentzian(e0OrParams, gamma, amp) {
    if (isPlainObject(e0OrParams)) {
      this[RAW].lorentzian(e0OrParams.e0, e0OrParams.gamma, e0OrParams.amp);
    } else {
      this[RAW].lorentzian(e0OrParams, gamma, amp);
    }
    return this;
  }

  /**
   * Add an exact Voigt peak; `sigma` Gaussian std, `gamma` Lorentzian HWHM.
   */
  voigt(e0OrParams, sigma, gamma, amp) {
    if (isPlainObject(e0OrParams)) {
      this[RAW].voigt(
        e0OrParams.e0,
        e0OrParams.sigma,
        e0OrParams.gamma,
        e0OrParams.amp,
      );
    } else {
      this[RAW].voigt(e0OrParams, sigma, gamma, amp);
    }
    return this;
  }

  /**
   * Add a pseudo-Voigt peak `eta*L + (1-eta)*G` with shared `fwhm`.
   */
  pseudoVoigt(e0OrParams, fwhm, eta, amp) {
    if (isPlainObject(e0OrParams)) {
      this[RAW].pseudoVoigt(
        e0OrParams.e0,
        e0OrParams.fwhm,
        e0OrParams.eta,
        e0OrParams.amp,
      );
    } else {
      this[RAW].pseudoVoigt(e0OrParams, fwhm, eta, amp);
    }
    return this;
  }

  /** Alias for {@link Model.pseudoVoigt} matching the Python API. */
  pseudo_voigt(e0OrParams, fwhm, eta, amp) {
    return this.pseudoVoigt(e0OrParams, fwhm, eta, amp);
  }

  /**
   * Add an erf ionization step in `beta`, with optional linear post-edge `decay` (default `0.0`).
   */
  step(e0OrParams, width, height, decay = 0.0) {
    if (isPlainObject(e0OrParams)) {
      this[RAW].step(
        e0OrParams.e0,
        e0OrParams.width,
        e0OrParams.height,
        e0OrParams.decay ?? 0.0,
      );
    } else {
      this[RAW].step(e0OrParams, width, height, decay);
    }
    return this;
  }

  /**
   * Set the linear continuum in `beta`: `bkg + slope*E`.
   */
  continuum(bkgOrParams = 0.0, slope = 0.0) {
    if (isPlainObject(bkgOrParams)) {
      this[RAW].continuum(bkgOrParams.bkg ?? 0.0, bkgOrParams.slope ?? 0.0);
    } else {
      this[RAW].continuum(bkgOrParams, slope);
    }
    return this;
  }

  /**
   * Set the fixed continuum `delta` table (e.g. from kkcalc), interpolated linearly and clamped at the ends.
   */
  deltaTable(energy, delta) {
    this[RAW].deltaTable(energy, delta);
    return this;
  }

  /** Alias for {@link Model.deltaTable} matching the Python API. */
  delta_table(energy, delta) {
    return this.deltaTable(energy, delta);
  }

  /**
   * Set the overall multiplicative scale.
   */
  scale(scale) {
    this[RAW].scale(scale);
    return this;
  }

  /**
   * Set the energy offset: the model is evaluated at `E + shift`.
   */
  shift(shift) {
    this[RAW].shift(shift);
    return this;
  }

  /**
   * Return an independent copy of this model.
   */
  copy() {
    return new Model(FROM_RAW, this[RAW].copy());
  }

  /** Alias for {@link Model.copy}. */
  clone() {
    return this.copy();
  }

  /** Number of peaks and steps (a pseudo-Voigt counts as one). */
  get nComponents() {
    return this[RAW].nComponents;
  }

  /** Alias for {@link Model.nComponents} matching the Python API. */
  get n_components() {
    return this[RAW].nComponents;
  }

  /** Length of the flat parameter vector. */
  get nParams() {
    return this[RAW].nParams;
  }

  /** Alias for {@link Model.nParams} matching the Python API. */
  get n_params() {
    return this[RAW].nParams;
  }

  /**
   * Current flat parameter vector (components in insertion order, then
   * `continuum.bkg, continuum.slope, scale, shift`).
   */
  get params() {
    return this[RAW].params;
  }

  /** Names matching `params`, e.g. `'gaussian[0].e0'`, `'step[1].height'`, `'scale'`. */
  get paramNames() {
    return this[RAW].paramNames;
  }

  /** Alias for {@link Model.paramNames} matching the Python API. */
  get param_names() {
    return this[RAW].paramNames;
  }

  /**
   * Set all parameters from a flat vector (validated; model is unchanged on error).
   */
  setParams(params) {
    this[RAW].setParams(params);
    return this;
  }

  /** Alias for {@link Model.setParams} matching the Python API. */
  set_params(params) {
    return this.setParams(params);
  }

  /**
   * Bind to a fixed energy grid for repeated evaluation (see {@link Evaluator}).
   */
  bind(energy, cache = true) {
    return new Evaluator(FROM_RAW, this[RAW].bind(energy, cache));
  }

  /**
   * Evaluate `{ delta, beta }` on `energy` (1-D float64 energies in eV).
   * Optionally pass a flat `params` vector (or `{ params, out }`) to evaluate without modifying the model.
   */
  eval(energy, paramsOrOptions) {
    let params = undefined;
    let out = undefined;
    if (paramsOrOptions !== undefined && paramsOrOptions !== null) {
      if (isPlainObject(paramsOrOptions)) {
        params = paramsOrOptions.params ?? undefined;
        out = paramsOrOptions.out ?? undefined;
      } else {
        params = paramsOrOptions;
      }
    }
    const n = energy.length;
    const delta = out ? out.delta : new Float64Array(n);
    const beta = out ? out.beta : new Float64Array(n);
    if (params !== undefined && params !== null) {
      this[RAW].evalParamsInto(energy, params, delta, beta);
    } else {
      this[RAW].evalInto(energy, delta, beta);
    }
    return out ?? { delta, beta };
  }

  /**
   * Evaluate `delta` and `beta` on `energy` directly into caller-provided `Float64Array` buffers.
   */
  evalInto(energy, deltaOut, betaOut, params) {
    if (params !== undefined && params !== null) {
      this[RAW].evalParamsInto(energy, params, deltaOut, betaOut);
    } else {
      this[RAW].evalInto(energy, deltaOut, betaOut);
    }
  }

  /**
   * Evaluate only the real refractive decrement `delta` on `energy`.
   */
  evalDelta(energy, params) {
    if (params !== undefined && params !== null) {
      return this[RAW].evalParamsDelta(energy, params);
    }
    return this[RAW].evalDelta(energy);
  }

  /**
   * Evaluate only the absorption index `beta` on `energy`.
   */
  evalBeta(energy, params) {
    if (params !== undefined && params !== null) {
      return this[RAW].evalParamsBeta(energy, params);
    }
    return this[RAW].evalBeta(energy);
  }

  /**
   * Evaluate on `energy`, returning interleaved `[delta_0, beta_0, delta_1, beta_1, ...]`.
   */
  evalInterleaved(energy, params) {
    if (params !== undefined && params !== null) {
      return this[RAW].evalParamsInterleaved(energy, params);
    }
    return this[RAW].evalInterleaved(energy);
  }

  /** Release the underlying WebAssembly memory immediately. */
  free() {
    this[RAW].free();
  }

  toString() {
    return this[RAW].toString();
  }
}

if (typeof Symbol.dispose === "symbol") {
  Model.prototype[Symbol.dispose] = Model.prototype.free;
}

/**
 * A {@link Model} bound to a fixed energy grid (from {@link Model.bind}), for fitting loops.
 *
 * `scale` and `continuum.*` never trigger component work; changing `shift` recomputes all.
 */
export class Evaluator {
  constructor(modelOrToken, energyOrRaw, cache = true) {
    ensureInit();
    if (modelOrToken === FROM_RAW) {
      this[RAW] = energyOrRaw;
    } else {
      if (!(modelOrToken instanceof Model)) {
        throw new TypeError("expected instance of Model");
      }
      this[RAW] = new RawEvaluator(modelOrToken[RAW], energyOrRaw, cache);
    }
  }

  /** Length of the flat parameter vector. */
  get nParams() {
    return this[RAW].nParams;
  }

  /** Alias for {@link Evaluator.nParams} matching the Python API. */
  get n_params() {
    return this[RAW].nParams;
  }

  /** Number of points in the bound energy grid. */
  get nEnergy() {
    return this[RAW].nEnergy;
  }

  /** Alias for {@link Evaluator.nEnergy}. */
  get n_energy() {
    return this[RAW].nEnergy;
  }

  /** Current parameter vector. */
  get params() {
    return this[RAW].params;
  }

  /** Names matching `params`. */
  get paramNames() {
    return this[RAW].paramNames;
  }

  /** Alias for {@link Evaluator.paramNames} matching the Python API. */
  get param_names() {
    return this[RAW].paramNames;
  }

  /** The bound energy grid in eV. */
  get energy() {
    return this[RAW].energy;
  }

  /** A copy of the bound model at the current parameters. */
  get model() {
    return new Model(FROM_RAW, this[RAW].model);
  }

  /**
   * Set the flat parameter vector without evaluating.
   */
  setParams(params) {
    this[RAW].setParams(params);
    return this;
  }

  /** Alias for {@link Evaluator.setParams} matching the Python API. */
  set_params(params) {
    return this.setParams(params);
  }

  /**
   * Evaluate `{ delta, beta }` at `params` (or the current parameters).
   * When `out` (`{ delta, beta }`) is supplied, writes into those preallocated `Float64Array`s
   * without allocating and returns `out`.
   */
  eval(params, out) {
    const n = this[RAW].nEnergy;
    const delta = out ? out.delta : new Float64Array(n);
    const beta = out ? out.beta : new Float64Array(n);
    if (params !== undefined && params !== null) {
      this[RAW].evalParamsInto(params, delta, beta);
    } else {
      this[RAW].evalInto(delta, beta);
    }
    return out ?? { delta, beta };
  }

  /**
   * Evaluate into caller-provided `deltaOut` and `betaOut` `Float64Array`s.
   */
  evalInto(deltaOut, betaOut, params) {
    if (params !== undefined && params !== null) {
      this[RAW].evalParamsInto(params, deltaOut, betaOut);
    } else {
      this[RAW].evalInto(deltaOut, betaOut);
    }
  }

  /**
   * Evaluate into caller-provided interleaved `out` `Float64Array` (`2 * nEnergy`).
   */
  evalInterleavedInto(out, params) {
    if (params !== undefined && params !== null) {
      this[RAW].evalParamsInterleavedInto(params, out);
    } else {
      this[RAW].evalInterleavedInto(out);
    }
  }

  /**
   * Evaluate only `delta` at `params` (or the current parameters).
   */
  evalDelta(params) {
    if (params !== undefined && params !== null) {
      this[RAW].setParams(params);
    }
    return this[RAW].evalDelta();
  }

  /**
   * Evaluate only `beta` at `params` (or the current parameters).
   */
  evalBeta(params) {
    if (params !== undefined && params !== null) {
      this[RAW].setParams(params);
    }
    return this[RAW].evalBeta();
  }

  /**
   * Evaluate at `params` (or current parameters), returning interleaved `[delta_0, beta_0, ...]`.
   */
  evalInterleaved(params) {
    if (params !== undefined && params !== null) {
      this[RAW].setParams(params);
    }
    return this[RAW].evalInterleaved();
  }

  /** Release the underlying WebAssembly memory immediately. */
  free() {
    this[RAW].free();
  }

  toString() {
    return this[RAW].toString();
  }
}

if (typeof Symbol.dispose === "symbol") {
  Evaluator.prototype[Symbol.dispose] = Evaluator.prototype.free;
}

/**
 * One-shot Gaussian peak evaluation returning `{ delta, beta }`.
 * For repeated evaluations, build a {@link Model} once.
 */
export function gaussian(energy, e0OrParams, width, amp) {
  const m = new Model().gaussian(e0OrParams, width, amp);
  try {
    return m.eval(energy);
  } finally {
    m.free();
  }
}

/**
 * One-shot Lorentzian peak evaluation returning `{ delta, beta }`.
 * For repeated evaluations, build a {@link Model} once.
 */
export function lorentzian(energy, e0OrParams, gamma, amp) {
  const m = new Model().lorentzian(e0OrParams, gamma, amp);
  try {
    return m.eval(energy);
  } finally {
    m.free();
  }
}

/**
 * One-shot exact Voigt peak evaluation returning `{ delta, beta }`.
 * For repeated evaluations, build a {@link Model} once.
 */
export function voigt(energy, e0OrParams, sigma, gamma, amp) {
  const m = new Model().voigt(e0OrParams, sigma, gamma, amp);
  try {
    return m.eval(energy);
  } finally {
    m.free();
  }
}

/**
 * One-shot pseudo-Voigt peak evaluation returning `{ delta, beta }`.
 * For repeated evaluations, build a {@link Model} once.
 */
export function pseudoVoigt(energy, e0OrParams, fwhm, eta, amp) {
  const m = new Model().pseudoVoigt(e0OrParams, fwhm, eta, amp);
  try {
    return m.eval(energy);
  } finally {
    m.free();
  }
}

/** Alias for {@link pseudoVoigt} matching the Python API. */
export const pseudo_voigt = pseudoVoigt;

/**
 * Evaluate a linear + erf-step continuum in `beta`.
 */
export function continuumBeta(energy, options = {}) {
  const { bkg = 0.0, slope = 0.0, steps = [] } = options;
  const m = new Model().continuum(bkg, slope);
  try {
    for (const s of steps) {
      m.step(s.e0, s.width, s.height, s.decay ?? 0.0);
    }
    return m.evalBeta(energy);
  } finally {
    m.free();
  }
}

/** Alias for {@link continuumBeta} matching the Python API. */
export const continuum_beta = continuumBeta;

/**
 * Evaluate a declarative optical model specification on `energy`.
 */
export function opticalModel(energy, options = {}) {
  const m = Model.fromOptions(options);
  try {
    return m.eval(energy);
  } finally {
    m.free();
  }
}

/** Alias for {@link opticalModel} matching the Python API. */
export const optical_model = opticalModel;

/**
 * Photon wavelength in Angstrom for energies in eV (`h*c / E`).
 */
export function wavelengthAngstrom(energyEv) {
  ensureInit();
  return rawWavelengthAngstrom(energyEv);
}

/** Alias for {@link wavelengthAngstrom} matching the Python API. */
export const wavelength_angstrom = wavelengthAngstrom;

/**
 * Convert a mass absorption coefficient (`cm^2/g`) to `beta`: `beta = mu * rho * lambda / (4 pi)`.
 */
export function muToBeta(energyEv, muCm2PerG, densityGCm3 = 1.0) {
  ensureInit();
  return rawMuToBeta(energyEv, muCm2PerG, densityGCm3);
}

/** Alias for {@link muToBeta} matching the Python API. */
export const mu_to_beta = muToBeta;

/**
 * Convert optical `beta` to a mass absorption coefficient (`cm^2/g`): `mu = 4 pi * beta / (rho * lambda)`.
 */
export function betaToMu(energyEv, beta, densityGCm3 = 1.0) {
  ensureInit();
  return rawBetaToMu(energyEv, beta, densityGCm3);
}

/** Alias for {@link betaToMu} matching the Python API. */
export const beta_to_mu = betaToMu;

/** Unit conversion namespace matching `lineshapes.units` in Python. */
export const units = Object.freeze({
  HC_EV_ANGSTROM,
  wavelengthAngstrom,
  wavelength_angstrom,
  muToBeta,
  mu_to_beta,
  betaToMu,
  beta_to_mu,
});

/** Dawson's function `D(x)` with asymptotic fast path for `|x| >= 10`. */
export function fastDawson(x) {
  ensureInit();
  return rawFastDawson(x);
}

/** Alias for {@link fastDawson}. */
export const fast_dawson = fastDawson;

/** Error function `erf(x)` short-circuited to `+-1` for `|x| > 6`. */
export function fastErf(x) {
  ensureInit();
  return rawFastErf(x);
}

/** Alias for {@link fastErf}. */
export const fast_erf = fastErf;

/** Faddeeva function `w(z)` for `Im z >= 0`, returning `[re, im]`. */
export function fastW(re, im) {
  ensureInit();
  const out = rawFastW(re, im);
  return [out[0], out[1]];
}

/** Alias for {@link fastW}. */
export const fast_w = fastW;
