import { describe, expect, test } from "bun:test";
import {
  Evaluator,
  HC_EV_ANGSTROM,
  Model,
  betaToMu,
  beta_to_mu,
  continuumBeta,
  continuum_beta,
  fastDawson,
  fastErf,
  fastW,
  gaussian,
  init,
  initSync,
  lorentzian,
  muToBeta,
  mu_to_beta,
  opticalModel,
  pseudoVoigt,
  pseudo_voigt,
  units,
  voigt,
  wavelengthAngstrom,
} from "../pkg/index.js";

function linspace(start: number, stop: number, num: number): Float64Array {
  const out = new Float64Array(num);
  if (num === 1) {
    out[0] = start;
    return out;
  }
  const step = (stop - start) / (num - 1);
  for (let i = 0; i < num; i++) {
    out[i] = start + i * step;
  }
  return out;
}

const E0 = 930.0;
const E_LOCAL = linspace(E0 - 15.0, E0 + 15.0, 601);
const E_GRID = linspace(900.0, 1000.0, 4001);

function fullModel(): Model {
  const cosTable = new Float64Array(E_GRID.length);
  for (let i = 0; i < E_GRID.length; i++) {
    cosTable[i] = 1e-4 * Math.cos(E_GRID[i]! / 30.0);
  }
  return new Model()
    .gaussian(933.8, 0.49, 4.1e-5)
    .lorentzian(936.0, 0.3, 2e-5)
    .voigt(938.3, 0.7, 0.2, 5e-5)
    .pseudoVoigt(955.0, 1.5, 0.4, 2e-5)
    .step(939.4, 6.1, 7.2e-5)
    .step(960.7, 2.2, 5.1e-5, 1e-7)
    .continuum(3.6e-4, -3.7e-7)
    .deltaTable(E_GRID, cosTable)
    .scale(1.1)
    .shift(0.05);
}

function rebuild(theta: Float64Array | readonly number[]): Model {
  const t = Array.from(theta);
  const cosTable = new Float64Array(E_GRID.length);
  for (let i = 0; i < E_GRID.length; i++) {
    cosTable[i] = 1e-4 * Math.cos(E_GRID[i]! / 30.0);
  }
  return new Model()
    .gaussian(t[0]!, t[1]!, t[2]!)
    .lorentzian(t[3]!, t[4]!, t[5]!)
    .voigt(t[6]!, t[7]!, t[8]!, t[9]!)
    .pseudoVoigt(t[10]!, t[11]!, t[12]!, t[13]!)
    .step(t[14]!, t[15]!, t[16]!, t[17]!)
    .step(t[18]!, t[19]!, t[20]!, t[21]!)
    .continuum(t[22]!, t[23]!)
    .deltaTable(E_GRID, cosTable)
    .scale(t[24]!)
    .shift(t[25]!);
}

describe("lineshapes-wasm initialization", () => {
  test("works synchronously without explicit init and also resolves async init()", async () => {
    const m = new Model().gaussian(930.0, 1.0, 2.0);
    const { delta, beta } = m.eval([929.0, 930.0, 931.0]);
    expect(beta[1]).toBeCloseTo(2.0, 12);
    expect(delta[0]!).toBeLessThan(0);
    expect(delta[2]!).toBeGreaterThan(0);
    m.free();

    const wasmOut = await init();
    expect(wasmOut.memory).toBeInstanceOf(WebAssembly.Memory);
    expect(initSync().memory).toBe(wasmOut.memory);
  });
});

describe("peak lineshapes", () => {
  test("peak height and sign (beta(E0) = amp, delta < 0 below E0, > 0 above)", () => {
    const e = [E0 - 1.0, E0, E0 + 1.0] as const;
    const cases = [
      gaussian(e, E0, 1.0, 2.5),
      lorentzian(e, E0, 0.5, 2.5),
      voigt(e, E0, 0.5, 0.3, 2.5),
      pseudoVoigt(e, E0, 1.2, 0.4, 2.5),
    ];
    for (const { delta, beta } of cases) {
      expect(Math.abs(beta[1]! - 2.5)).toBeLessThan(1e-12);
      expect(Math.abs(delta[1]!)).toBeLessThan(1e-12);
      expect(delta[0]!).toBeLessThan(0);
      expect(delta[2]!).toBeGreaterThan(0);
    }
  });

  test("named parameter objects match positional arguments", () => {
    const mPos = new Model()
      .gaussian(933.8, 0.49, 4.1e-5)
      .lorentzian(936.0, 0.3, 2e-5)
      .voigt(938.3, 0.7, 0.2, 5e-5)
      .pseudoVoigt(955.0, 1.5, 0.4, 2e-5)
      .step(939.4, 6.1, 7.2e-5, 1e-7)
      .continuum(3.6e-4, -3.7e-7);

    const mObj = new Model()
      .gaussian({ e0: 933.8, width: 0.49, amp: 4.1e-5 })
      .lorentzian({ e0: 936.0, gamma: 0.3, amp: 2e-5 })
      .voigt({ e0: 938.3, sigma: 0.7, gamma: 0.2, amp: 5e-5 })
      .pseudoVoigt({ e0: 955.0, fwhm: 1.5, eta: 0.4, amp: 2e-5 })
      .step({ e0: 939.4, width: 6.1, height: 7.2e-5, decay: 1e-7 })
      .continuum({ bkg: 3.6e-4, slope: -3.7e-7 });

    const a = mPos.eval(E_GRID);
    const b = mObj.eval(E_GRID);
    expect(Array.from(a.delta)).toEqual(Array.from(b.delta));
    expect(Array.from(a.beta)).toEqual(Array.from(b.beta));
    mPos.free();
    mObj.free();
  });

  test("voigt Gaussian and Lorentzian limits", () => {
    const vG = voigt(E_LOCAL, {
      e0: E0,
      sigma: 1.0 / Math.SQRT2,
      gamma: 1e-6,
      amp: 1.0,
    });
    const g = gaussian(E_LOCAL, { e0: E0, width: 1.0, amp: 1.0 });
    for (let i = 0; i < E_LOCAL.length; i++) {
      expect(Math.hypot(vG.delta[i]! - g.delta[i]!, vG.beta[i]! - g.beta[i]!)).toBeLessThan(1e-6);
    }

    const vL = voigt(E_LOCAL, {
      e0: E0,
      sigma: 1e-3,
      gamma: 0.5,
      amp: 1.0,
    });
    const lo = lorentzian(E_LOCAL, { e0: E0, gamma: 0.5, amp: 1.0 });
    for (let i = 0; i < E_LOCAL.length; i++) {
      expect(Math.hypot(vL.delta[i]! - lo.delta[i]!, vL.beta[i]! - lo.beta[i]!)).toBeLessThan(1e-5);
    }
  });

  test("pseudoVoigt endpoints (eta = 0 is Gaussian, eta = 1 is Lorentzian)", () => {
    const fwhm = 1.2;
    const p0 = pseudoVoigt(E_LOCAL, E0, fwhm, 0.0, 1.0);
    const gWidth = fwhm / (2.0 * Math.sqrt(Math.LN2));
    const g = gaussian(E_LOCAL, E0, gWidth, 1.0);
    for (let i = 0; i < E_LOCAL.length; i++) {
      expect(Math.abs(p0.delta[i]! - g.delta[i]!)).toBeLessThan(1e-13);
      expect(Math.abs(p0.beta[i]! - g.beta[i]!)).toBeLessThan(1e-13);
    }

    const p1 = pseudo_voigt(E_LOCAL, E0, fwhm, 1.0, 1.0);
    const lo = lorentzian(E_LOCAL, E0, fwhm / 2.0, 1.0);
    for (let i = 0; i < E_LOCAL.length; i++) {
      expect(Math.abs(p1.delta[i]! - lo.delta[i]!)).toBeLessThan(1e-13);
      expect(Math.abs(p1.beta[i]! - lo.beta[i]!)).toBeLessThan(1e-13);
    }
  });

  test("continuumBeta and post-edge decay", () => {
    const e = [940.0, 950.0, 960.0];
    const got = continuumBeta(e, {
      steps: [{ e0: 950.0, width: 1e-9, height: 0.0, decay: 2.0 }],
    });
    expect(Math.abs(got[0]! - 0.0)).toBeLessThan(1e-12);
    expect(Math.abs(got[1]! - 0.0)).toBeLessThan(1e-12);
    expect(Math.abs(got[2]! - 20.0)).toBeLessThan(1e-12);
    expect(Array.from(continuum_beta(e, { bkg: 1.5, slope: 0.0 }))).toEqual([
      1.5, 1.5, 1.5,
    ]);
  });
});

describe("flat parameter vector and bound Evaluator", () => {
  test("paramNames and roundtrip", () => {
    const m = fullModel();
    expect(m.nComponents).toBe(6);
    expect(m.n_components).toBe(6);
    expect(m.nParams).toBe(26);
    expect(m.n_params).toBe(26);
    expect(m.params.length).toBe(26);
    expect(m.paramNames.length).toBe(26);
    expect(m.paramNames.slice(0, 4)).toEqual([
      "gaussian[0].e0",
      "gaussian[0].width",
      "gaussian[0].amp",
      "lorentzian[0].e0",
    ]);
    expect(m.paramNames.slice(14, 16)).toEqual(["step[0].e0", "step[0].width"]);
    expect(m.paramNames[18]).toBe("step[1].e0");
    expect(m.paramNames.slice(-4)).toEqual([
      "continuum.bkg",
      "continuum.slope",
      "scale",
      "shift",
    ]);
    const theta = m.params;
    expect(Array.from(m.setParams(theta).params)).toEqual(Array.from(theta));
    m.free();
  });

  test("eval with params does not mutate the model and matches rebuild", () => {
    const m = fullModel();
    const theta0 = new Float64Array(m.params);
    const theta = new Float64Array(theta0);
    theta[0]! += 0.5;

    const a = m.eval(E_GRID, theta);
    expect(Array.from(m.params)).toEqual(Array.from(theta0));

    const reb = rebuild(theta);
    const b = reb.eval(E_GRID);
    for (let i = 0; i < E_GRID.length; i++) {
      expect(Math.abs(a.delta[i]! - b.delta[i]!)).toBeLessThan(1e-18);
      expect(Math.abs(a.beta[i]! - b.beta[i]!)).toBeLessThan(1e-18);
    }
    m.free();
    reb.free();
  });

  test("invalid parameters throw and leave the model unchanged", () => {
    const m = fullModel();
    const theta0 = new Float64Array(m.params);
    const bad = new Float64Array(theta0);
    bad[1] = -0.1; // negative width
    expect(() => m.setParams(bad)).toThrow(/width/);
    expect(() => m.setParams(theta0.subarray(0, 25))).toThrow(/expected 26/);
    expect(Array.from(m.params)).toEqual(Array.from(theta0));
    m.free();
  });

  test.each([true, false])(
    "Evaluator(cache=%p) matches full rebuild across fit-like updates",
    (cache) => {
      const m = fullModel();
      const ev = m.bind(E_GRID, cache);
      expect(ev.nEnergy).toBe(E_GRID.length);
      expect(ev.nParams).toBe(26);

      const theta = new Float64Array(m.params);
      const out = {
        delta: new Float64Array(E_GRID.length),
        beta: new Float64Array(E_GRID.length),
      };

      let seed = 42;
      const nextRand = () => {
        seed = (seed * 1664525 + 1013904223) >>> 0;
        return seed / 0xffffffff - 0.5;
      };

      for (let it = 0; it < 60; it++) {
        if (it % 15 === 0) {
          for (let j = 0; j < theta.length; j++) {
            theta[j]! *= 1 + 1e-3 * nextRand();
          }
        } else {
          const k = it % theta.length;
          theta[k]! *= 1 + 1e-3 * nextRand();
        }
        theta[12] = Math.max(0.0, Math.min(1.0, theta[12]!));

        const got = it % 2 === 0 ? ev.eval(theta, out) : ev.eval(theta);
        if (it % 2 === 0) {
          expect(got).toBe(out);
        }

        const reb = rebuild(theta);
        const want = reb.eval(E_GRID);
        reb.free();

        let maxErr = 0;
        let maxMag = 0;
        for (let i = 0; i < E_GRID.length; i++) {
          const err = Math.hypot(
            got.delta[i]! - want.delta[i]!,
            got.beta[i]! - want.beta[i]!,
          );
          const mag = Math.hypot(want.delta[i]!, want.beta[i]!);
          if (err > maxErr) maxErr = err;
          if (mag > maxMag) maxMag = mag;
        }
        expect(maxErr / maxMag).toBeLessThan(1e-13);
      }

      ev.free();
      m.free();
    },
  );

  test("Evaluator output buffer length validation", () => {
    const m = fullModel();
    const ev = m.bind(E_GRID);
    expect(() =>
      ev.eval(null, {
        delta: new Float64Array(10),
        beta: new Float64Array(10),
      }),
    ).toThrow(/length/);
    ev.free();
    m.free();
  });
});

describe("declarative opticalModel, units, and special functions", () => {
  test("opticalModel matches chained Model", () => {
    const res = opticalModel(E_LOCAL, {
      components: [
        { kind: "gaussian", e0: 930.0, width: 0.8, amp: 1e-4 },
        { kind: "voigt", e0: 932.0, sigma: 0.4, gamma: 0.2, amp: 2e-4 },
        { kind: "step", e0: 931.0, width: 1.5, height: 5e-5 },
      ],
      continuum: { bkg: 1e-5, slope: -1e-8 },
      scale: 1.2,
      shift: -0.1,
    });

    const m = new Model()
      .gaussian(930.0, 0.8, 1e-4)
      .voigt(932.0, 0.4, 0.2, 2e-4)
      .step(931.0, 1.5, 5e-5)
      .continuum(1e-5, -1e-8)
      .scale(1.2)
      .shift(-0.1);
    const want = m.eval(E_LOCAL);
    m.free();

    expect(Array.from(res.delta)).toEqual(Array.from(want.delta));
    expect(Array.from(res.beta)).toEqual(Array.from(want.beta));
  });

  test("unit conversions roundtrip", () => {
    const energy = [285.0, 532.0, 933.8];
    const mu = [1.5e4, 9.2e3, 2.4e4];
    const rho = 1.35;

    const wl = wavelengthAngstrom(energy);
    expect(wl[0]).toBeCloseTo(HC_EV_ANGSTROM / 285.0, 12);

    const beta = muToBeta(energy, mu, rho);
    const muBack = betaToMu(energy, beta, rho);
    for (let i = 0; i < mu.length; i++) {
      expect(Math.abs(muBack[i]! - mu[i]!) / mu[i]!).toBeLessThan(1e-15);
    }

    expect(units.mu_to_beta).toBe(mu_to_beta);
    expect(units.beta_to_mu).toBe(beta_to_mu);
  });

  test("special functions", () => {
    expect(fastErf(0.0)).toBe(0.0);
    expect(fastErf(7.0)).toBe(1.0);
    expect(fastErf(-7.0)).toBe(-1.0);
    expect(fastDawson(0.0)).toBe(0.0);
    expect(fastDawson(1.0)).toBeCloseTo(0.5380795069127684, 15);
    const [wre, wim] = fastW(0.0, 0.0);
    expect(wre).toBeCloseTo(1.0, 15);
    expect(wim).toBeCloseTo(0.0, 15);
  });
});
