export type InitInput =
  | RequestInfo
  | URL
  | Response
  | BufferSource
  | WebAssembly.Module;

export type SyncInitInput = BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
}

/** A 1-D sequence of finite numbers (`Float64Array` or `readonly number[]`). */
export type NumericArray = Float64Array | readonly number[];

/** Complex optical constants `delta + i*beta` (`n = 1 - delta + i*beta`) on an energy grid in eV. */
export interface OpticalConstants {
  /** Real refractive index decrement `delta` (`delta = H[beta]`). */
  delta: Float64Array;
  /** Imaginary absorption index `beta`. */
  beta: Float64Array;
}

/** Parameters for a Gaussian peak (`width` is the 1/e half-width, `sigma = width / sqrt(2)`). */
export interface GaussianParams {
  /** Center energy in eV. */
  readonly e0: number;
  /** 1/e half-width in eV (`sigma = width / sqrt(2)`). */
  readonly width: number;
  /** Peak height of `beta` at `e0`. */
  readonly amp: number;
}

/** Parameters for a Lorentzian peak (`gamma` is the HWHM). */
export interface LorentzianParams {
  /** Center energy in eV. */
  readonly e0: number;
  /** Half width at half maximum (HWHM) in eV. */
  readonly gamma: number;
  /** Peak height of `beta` at `e0`. */
  readonly amp: number;
}

/** Parameters for an exact Voigt peak (`sigma` Gaussian std, `gamma` Lorentzian HWHM). */
export interface VoigtParams {
  /** Center energy in eV. */
  readonly e0: number;
  /** Gaussian standard deviation in eV (`> 0`). */
  readonly sigma: number;
  /** Lorentzian half width at half maximum in eV (`>= 0`). */
  readonly gamma: number;
  /** Peak height of `beta` at `e0`. */
  readonly amp: number;
}

/** Parameters for a pseudo-Voigt peak (`eta*L + (1-eta)*G` with shared `fwhm`). */
export interface PseudoVoigtParams {
  /** Center energy in eV. */
  readonly e0: number;
  /** Full width at half maximum of both parts in eV (`> 0`). */
  readonly fwhm: number;
  /** Lorentzian fraction in `[0, 1]`. */
  readonly eta: number;
  /** Peak height of `beta` at `e0`. */
  readonly amp: number;
}

/** Parameters for an erf ionization step in `beta`. */
export interface StepParams {
  /** Edge energy in eV. */
  readonly e0: number;
  /** Erf width in eV (`> 0`). */
  readonly width: number;
  /** Step height of `beta`. */
  readonly height: number;
  /** Linear post-edge slope added above `e0` (default `0.0`). */
  readonly decay?: number;
}

/** Parameters for a linear continuum in `beta`: `bkg + slope*E`. */
export interface ContinuumParams {
  /** Constant background offset in `beta` (default `0.0`). */
  readonly bkg?: number;
  /** Linear slope in `beta` per eV (default `0.0`). */
  readonly slope?: number;
}

/** Discriminated union of model components for declarative model construction. */
export type ComponentSpec =
  | ({ readonly kind: "gaussian" } & GaussianParams)
  | ({ readonly kind: "lorentzian" } & LorentzianParams)
  | ({ readonly kind: "voigt" } & VoigtParams)
  | ({ readonly kind: "pseudoVoigt" | "pseudo_voigt" } & PseudoVoigtParams)
  | ({ readonly kind: "step" } & StepParams);

/** Options for {@link continuumBeta}. */
export interface ContinuumBetaOptions extends ContinuumParams {
  readonly steps?: readonly StepParams[];
}

/** Options for declarative model construction via {@link Model.fromOptions} and {@link opticalModel}. */
export interface OpticalModelOptions {
  readonly components?: readonly ComponentSpec[];
  readonly continuum?: ContinuumBetaOptions;
  readonly deltaTable?: {
    readonly energy: NumericArray;
    readonly delta: NumericArray;
  };
  readonly scale?: number;
  readonly shift?: number;
}

/** Options for {@link Model.eval}. */
export interface ModelEvalOptions {
  /** Evaluate at this flat parameter vector without modifying the model. */
  readonly params?: NumericArray;
  /** Preallocated output buffers `{ delta, beta }` of length `energy.length`. */
  readonly out?: OpticalConstants;
}

/**
 * Synchronously initialize the WebAssembly module.
 * Called automatically on first use if `init()` or `initSync()` has not been called yet.
 */
export function initSync(
  module?: { module: SyncInitInput } | SyncInitInput,
): InitOutput;

/**
 * Asynchronously initialize the WebAssembly module.
 * Uses the embedded WASM binary by default so no external `.wasm` network fetch or bundler loader is required.
 */
export function init(
  moduleOrPath?:
    | { module_or_path: InitInput | Promise<InitInput> }
    | InitInput
    | Promise<InitInput>,
): Promise<InitOutput>;

export default init;

/** Planck's constant times the speed of light (`h * c`) in `eV * Angstrom`. */
export const HC_EV_ANGSTROM: number;

/**
 * Composable optical-constant model returning `delta + i*beta` (`n = 1 - delta + i*beta`)
 * on an energy grid in eV.
 *
 * Build by chaining (every builder method mutates the model in-place and returns `this`),
 * then evaluate with `.eval(energy)`:
 *
 * ```ts
 * const m = new Model()
 *   .gaussian(933.8, 0.49, 4.1e-5)
 *   .voigt(935.7, 0.8, 0.3, 7.4e-5)
 *   .continuum(3.6e-4, -3.7e-7)
 *   .step(939.4, 6.1, 7.2e-5)
 *   .deltaTable(energy, deltaBg);
 * const { delta, beta } = m.eval(energy);
 * ```
 */
export class Model {
  constructor();

  /** Build a `Model` from a declarative options object. */
  static fromOptions(options?: OpticalModelOptions): Model;

  /** Add a component from a tagged component specification object. */
  addComponent(spec: ComponentSpec): this;

  /** Add a Gaussian peak; `width` is the 1/e half-width (`sigma = width / sqrt(2)`). */
  gaussian(e0: number, width: number, amp: number): this;
  gaussian(params: GaussianParams): this;

  /** Add a Lorentzian peak; `gamma` is the half width at half maximum (HWHM). */
  lorentzian(e0: number, gamma: number, amp: number): this;
  lorentzian(params: LorentzianParams): this;

  /** Add an exact Voigt peak; `sigma` Gaussian std, `gamma` Lorentzian HWHM. */
  voigt(e0: number, sigma: number, gamma: number, amp: number): this;
  voigt(params: VoigtParams): this;

  /** Add a pseudo-Voigt peak `eta*L + (1-eta)*G` with shared `fwhm`. */
  pseudoVoigt(e0: number, fwhm: number, eta: number, amp: number): this;
  pseudoVoigt(params: PseudoVoigtParams): this;

  /** Alias for {@link Model.pseudoVoigt} matching the Python API. */
  pseudo_voigt(e0: number, fwhm: number, eta: number, amp: number): this;
  pseudo_voigt(params: PseudoVoigtParams): this;

  /** Add an erf ionization step in `beta`, with optional linear post-edge `decay` (default `0.0`). */
  step(e0: number, width: number, height: number, decay?: number): this;
  step(params: StepParams): this;

  /** Set the linear continuum in `beta`: `bkg + slope*E` (replaces any previous continuum). */
  continuum(bkg?: number, slope?: number): this;
  continuum(params: ContinuumParams): this;

  /** Set the fixed continuum `delta` table (e.g. from kkcalc), interpolated linearly and clamped at the ends. */
  deltaTable(energy: NumericArray, delta: NumericArray): this;

  /** Alias for {@link Model.deltaTable} matching the Python API. */
  delta_table(energy: NumericArray, delta: NumericArray): this;

  /** Set the overall multiplicative scale. */
  scale(scale: number): this;

  /** Set the energy offset: the model is evaluated at `E + shift`. */
  shift(shift: number): this;

  /** Return an independent copy of this model. */
  copy(): Model;

  /** Alias for {@link Model.copy}. */
  clone(): Model;

  /** Number of peaks and steps (a pseudo-Voigt counts as one). */
  readonly nComponents: number;

  /** Alias for {@link Model.nComponents} matching the Python API. */
  readonly n_components: number;

  /** Length of the flat parameter vector. */
  readonly nParams: number;

  /** Alias for {@link Model.nParams} matching the Python API. */
  readonly n_params: number;

  /**
   * Current flat parameter vector (components in insertion order, then
   * `continuum.bkg, continuum.slope, scale, shift`).
   */
  readonly params: Float64Array;

  /** Names matching `params`, e.g. `'gaussian[0].e0'`, `'step[1].height'`, `'scale'`. */
  readonly paramNames: string[];

  /** Alias for {@link Model.paramNames} matching the Python API. */
  readonly param_names: string[];

  /** Set all parameters from a flat vector (validated; model is unchanged on error). */
  setParams(params: NumericArray): this;

  /** Alias for {@link Model.setParams} matching the Python API. */
  set_params(params: NumericArray): this;

  /**
   * Bind the model to a fixed energy grid for fitting loops.
   * `cache = true` keeps each component's contribution (`nComponents * energy.length * 16` bytes)
   * so calls recompute only the components whose parameters changed.
   */
  bind(energy: NumericArray, cache?: boolean): Evaluator;

  /**
   * Evaluate `{ delta, beta }` on `energy` (1-D float64 energies in eV).
   * Optionally pass a flat `params` vector (or `{ params, out }`) to evaluate without modifying the model.
   */
  eval(
    energy: NumericArray,
    paramsOrOptions?: NumericArray | ModelEvalOptions | null,
  ): OpticalConstants;

  /** Evaluate `delta` and `beta` on `energy` directly into caller-provided `Float64Array` buffers. */
  evalInto(
    energy: NumericArray,
    deltaOut: Float64Array,
    betaOut: Float64Array,
    params?: NumericArray | null,
  ): void;

  /** Evaluate only the real refractive decrement `delta` on `energy`. */
  evalDelta(energy: NumericArray, params?: NumericArray | null): Float64Array;

  /** Evaluate only the absorption index `beta` on `energy`. */
  evalBeta(energy: NumericArray, params?: NumericArray | null): Float64Array;

  /** Evaluate on `energy`, returning interleaved `[delta_0, beta_0, delta_1, beta_1, ...]`. */
  evalInterleaved(
    energy: NumericArray,
    params?: NumericArray | null,
  ): Float64Array;

  /** Release the underlying WebAssembly memory immediately. */
  free(): void;
  toString(): string;
}

/**
 * A {@link Model} bound to a fixed energy grid (from {@link Model.bind}), for fitting loops.
 *
 * `scale` and `continuum.*` never trigger component work; changing `shift` recomputes all.
 */
export class Evaluator {
  constructor(model: Model, energy: NumericArray, cache?: boolean);

  /** Length of the flat parameter vector. */
  readonly nParams: number;

  /** Alias for {@link Evaluator.nParams} matching the Python API. */
  readonly n_params: number;

  /** Number of points in the bound energy grid. */
  readonly nEnergy: number;

  /** Alias for {@link Evaluator.nEnergy}. */
  readonly n_energy: number;

  /** Current parameter vector. */
  readonly params: Float64Array;

  /** Names matching `params`. */
  readonly paramNames: string[];

  /** Alias for {@link Evaluator.paramNames} matching the Python API. */
  readonly param_names: string[];

  /** The bound energy grid in eV. */
  readonly energy: Float64Array;

  /** A copy of the bound model at the current parameters. */
  readonly model: Model;

  /** Set the flat parameter vector without evaluating. */
  setParams(params: NumericArray): this;

  /** Alias for {@link Evaluator.setParams} matching the Python API. */
  set_params(params: NumericArray): this;

  /**
   * Evaluate `{ delta, beta }` at `params` (or the current parameters).
   * When `out` (`{ delta, beta }`) is supplied, writes into those preallocated `Float64Array`s
   * without allocating and returns `out`.
   */
  eval(
    params?: NumericArray | null,
    out?: OpticalConstants | null,
  ): OpticalConstants;

  /** Evaluate into caller-provided `deltaOut` and `betaOut` `Float64Array`s. */
  evalInto(
    deltaOut: Float64Array,
    betaOut: Float64Array,
    params?: NumericArray | null,
  ): void;

  /** Evaluate into caller-provided interleaved `out` `Float64Array` (`2 * nEnergy`). */
  evalInterleavedInto(
    out: Float64Array,
    params?: NumericArray | null,
  ): void;

  /** Evaluate only `delta` at `params` (or the current parameters). */
  evalDelta(params?: NumericArray | null): Float64Array;

  /** Evaluate only `beta` at `params` (or the current parameters). */
  evalBeta(params?: NumericArray | null): Float64Array;

  /** Evaluate at `params` (or current parameters), returning interleaved `[delta_0, beta_0, ...]`. */
  evalInterleaved(params?: NumericArray | null): Float64Array;

  /** Release the underlying WebAssembly memory immediately. */
  free(): void;
  toString(): string;
}

/** One-shot Gaussian peak evaluation returning `{ delta, beta }`. */
export function gaussian(
  energy: NumericArray,
  e0: number,
  width: number,
  amp: number,
): OpticalConstants;
export function gaussian(
  energy: NumericArray,
  params: GaussianParams,
): OpticalConstants;

/** One-shot Lorentzian peak evaluation returning `{ delta, beta }`. */
export function lorentzian(
  energy: NumericArray,
  e0: number,
  gamma: number,
  amp: number,
): OpticalConstants;
export function lorentzian(
  energy: NumericArray,
  params: LorentzianParams,
): OpticalConstants;

/** One-shot exact Voigt peak evaluation returning `{ delta, beta }`. */
export function voigt(
  energy: NumericArray,
  e0: number,
  sigma: number,
  gamma: number,
  amp: number,
): OpticalConstants;
export function voigt(
  energy: NumericArray,
  params: VoigtParams,
): OpticalConstants;

/** One-shot pseudo-Voigt peak evaluation returning `{ delta, beta }`. */
export function pseudoVoigt(
  energy: NumericArray,
  e0: number,
  fwhm: number,
  eta: number,
  amp: number,
): OpticalConstants;
export function pseudoVoigt(
  energy: NumericArray,
  params: PseudoVoigtParams,
): OpticalConstants;

/** Alias for {@link pseudoVoigt} matching the Python API. */
export const pseudo_voigt: typeof pseudoVoigt;

/** Evaluate a linear + erf-step continuum in `beta`. */
export function continuumBeta(
  energy: NumericArray,
  options?: ContinuumBetaOptions,
): Float64Array;

/** Alias for {@link continuumBeta} matching the Python API. */
export const continuum_beta: typeof continuumBeta;

/** Evaluate a declarative optical model specification on `energy`. */
export function opticalModel(
  energy: NumericArray,
  options?: OpticalModelOptions,
): OpticalConstants;

/** Alias for {@link opticalModel} matching the Python API. */
export const optical_model: typeof opticalModel;

/** Photon wavelength in Angstrom for energies in eV (`h*c / E`). */
export function wavelengthAngstrom(energyEv: NumericArray): Float64Array;

/** Alias for {@link wavelengthAngstrom} matching the Python API. */
export const wavelength_angstrom: typeof wavelengthAngstrom;

/** Convert a mass absorption coefficient (`cm^2/g`) to `beta`: `beta = mu * rho * lambda / (4 pi)`. */
export function muToBeta(
  energyEv: NumericArray,
  muCm2PerG: NumericArray,
  densityGCm3?: number,
): Float64Array;

/** Alias for {@link muToBeta} matching the Python API. */
export const mu_to_beta: typeof muToBeta;

/** Convert optical `beta` to a mass absorption coefficient (`cm^2/g`): `mu = 4 pi * beta / (rho * lambda)`. */
export function betaToMu(
  energyEv: NumericArray,
  beta: NumericArray,
  densityGCm3?: number,
): Float64Array;

/** Alias for {@link betaToMu} matching the Python API. */
export const beta_to_mu: typeof betaToMu;

/** Unit conversion namespace matching `lineshapes.units` in Python. */
export const units: Readonly<{
  HC_EV_ANGSTROM: number;
  wavelengthAngstrom: typeof wavelengthAngstrom;
  wavelength_angstrom: typeof wavelengthAngstrom;
  muToBeta: typeof muToBeta;
  mu_to_beta: typeof muToBeta;
  betaToMu: typeof betaToMu;
  beta_to_mu: typeof betaToMu;
}>;

/** Dawson's function `D(x)` with asymptotic fast path for `|x| >= 10`. */
export function fastDawson(x: number): number;

/** Alias for {@link fastDawson}. */
export const fast_dawson: typeof fastDawson;

/** Error function `erf(x)` short-circuited to `+-1` for `|x| > 6`. */
export function fastErf(x: number): number;

/** Alias for {@link fastErf}. */
export const fast_erf: typeof fastErf;

/** Faddeeva function `w(z)` for `Im z >= 0`, returning `[re, im]`. */
export function fastW(re: number, im: number): [number, number];

/** Alias for {@link fastW}. */
export const fast_w: typeof fastW;
