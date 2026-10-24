//! Compiled (precomputed-constant) components and the hot evaluation loops.
//!
//! [`Kernels`] is derived from the user-facing [`crate::Component`] list; recompiling it is
//! O(number of components), so parameter updates never touch the energy grid.

use errorfunctions::ComplexErrorFunctions;
use num_complex::Complex64;

use crate::model::Component;
use crate::special::{dawson_asymp, fast_dawson, fast_erf, fast_w, w_asymp, ASYMP_X, ERF_SAT};

const FRAC_2_SQRT_PI: f64 = std::f64::consts::FRAC_2_SQRT_PI;
const SQRT_2: f64 = std::f64::consts::SQRT_2;
/// Gaussian beta is skipped for `x^2 >= 64` (`exp(-64) = 1.6e-28` relative to the height;
/// avoids slow subnormal `exp` results).
const EXP_CUTOFF: f64 = 64.0;

/// Gaussian, `x = (E - e0) / width`.
#[derive(Clone, Debug)]
struct Gaussian {
    e0: f64,
    inv_width: f64,
    amp: f64,
    amp_delta: f64,
}

/// Lorentzian of HWHM `gamma`: `delta + i beta = A*gamma / (x - i*gamma)`.
#[derive(Clone, Debug)]
struct Lorentzian {
    e0: f64,
    gamma: f64,
    amp_gamma: f64,
}

/// Exact Voigt via Faddeeva, normalized to height `amp` at `e0`.
#[derive(Clone, Debug)]
struct Voigt {
    e0: f64,
    inv_s: f64,    // 1 / (sqrt(2) sigma)
    y: f64,        // gamma / (sqrt(2) sigma)
    amp_norm: f64, // amp / Re w(i y)
}

/// Erf ionization step: `0.5*h*(1 + erf((E - e0)/w)) + decay*max(E - e0, 0)`.
#[derive(Clone, Debug)]
struct Step {
    e0: f64,
    inv_width: f64,
    half_height: f64,
    decay: f64,
}

/// Struct-of-kinds compiled form of a component list.
#[derive(Clone, Debug, Default)]
pub(crate) struct Kernels {
    gaussians: Vec<Gaussian>,
    lorentzians: Vec<Lorentzian>,
    voigts: Vec<Voigt>,
    steps: Vec<Step>,
}

impl Kernels {
    pub(crate) fn clear(&mut self) {
        self.gaussians.clear();
        self.lorentzians.clear();
        self.voigts.clear();
        self.steps.clear();
    }

    /// Compile one (already validated) component.
    pub(crate) fn push(&mut self, c: &Component) {
        match *c {
            Component::Gaussian { e0, width, amp } => self.push_gaussian(e0, width, amp),
            Component::Lorentzian { e0, gamma, amp } => self.push_lorentzian(e0, gamma, amp),
            Component::Voigt { e0, sigma, gamma, amp } => {
                let inv_s = 1.0 / (SQRT_2 * sigma);
                let y = gamma * inv_s;
                let norm = Complex64::new(0.0, y).w().re;
                self.voigts.push(Voigt { e0, inv_s, y, amp_norm: amp / norm });
            }
            Component::PseudoVoigt { e0, fwhm, eta, amp } => {
                // eta*L + (1-eta)*G, shared FWHM; Gaussian 1/e half-width = fwhm / (2 sqrt(ln 2)).
                if eta > 0.0 {
                    self.push_lorentzian(e0, 0.5 * fwhm, eta * amp);
                }
                if eta < 1.0 {
                    let width = fwhm / (2.0 * std::f64::consts::LN_2.sqrt());
                    self.push_gaussian(e0, width, (1.0 - eta) * amp);
                }
            }
            Component::Step { e0, width, height, decay } => {
                self.steps.push(Step { e0, inv_width: 1.0 / width, half_height: 0.5 * height, decay });
            }
        }
    }

    fn push_gaussian(&mut self, e0: f64, width: f64, amp: f64) {
        self.gaussians.push(Gaussian { e0, inv_width: 1.0 / width, amp, amp_delta: amp * FRAC_2_SQRT_PI });
    }

    fn push_lorentzian(&mut self, e0: f64, gamma: f64, amp: f64) {
        self.lorentzians.push(Lorentzian { e0, gamma, amp_gamma: amp * gamma });
    }

    /// Add all compiled components evaluated at `energy + shift` into `out`.
    /// `sorted` enables the split far-tail / near-range fast path.
    pub(crate) fn accumulate(&self, energy: &[f64], shift: f64, sorted: bool, out: &mut [Complex64]) {
        let n = energy.len();
        // Index range [lo, hi) where |x| < t, x = (e + shift - e0) * inv (x increasing in e).
        let split = |e0: f64, inv: f64, t: f64| -> (usize, usize) {
            if !sorted {
                return (0, n);
            }
            let x = |e: f64| (e + shift - e0) * inv;
            (energy.partition_point(|&e| x(e) <= -t), energy.partition_point(|&e| x(e) < t))
        };

        for s in &self.steps {
            let (lo, hi) = split(s.e0, s.inv_width, ERF_SAT);
            // below: erf = -1 and d < 0 -> no contribution
            for (o, &e) in out[hi..].iter_mut().zip(&energy[hi..]) {
                o.im += 2.0 * s.half_height + s.decay * (e + shift - s.e0);
            }
            for (o, &e) in out[lo..hi].iter_mut().zip(&energy[lo..hi]) {
                let d = e + shift - s.e0;
                o.im += s.half_height * (1.0 + fast_erf(d * s.inv_width)) + s.decay * d.max(0.0);
            }
        }
        for g in &self.gaussians {
            let (lo, hi) = split(g.e0, g.inv_width, ASYMP_X);
            let far = |o: &mut Complex64, e: f64| {
                o.re += g.amp_delta * dawson_asymp((e + shift - g.e0) * g.inv_width);
            };
            out[..lo].iter_mut().zip(&energy[..lo]).for_each(|(o, &e)| far(o, e));
            out[hi..].iter_mut().zip(&energy[hi..]).for_each(|(o, &e)| far(o, e));
            for (o, &e) in out[lo..hi].iter_mut().zip(&energy[lo..hi]) {
                let x = (e + shift - g.e0) * g.inv_width;
                o.re += g.amp_delta * fast_dawson(x);
                let x2 = x * x;
                if x2 < EXP_CUTOFF {
                    o.im += g.amp * (-x2).exp();
                }
            }
        }
        for l in &self.lorentzians {
            let g2 = l.gamma * l.gamma;
            for (o, &e) in out.iter_mut().zip(energy) {
                let x = e + shift - l.e0;
                let r = l.amp_gamma / (x * x + g2);
                o.re += r * x;
                o.im += r * l.gamma;
            }
        }
        for v in &self.voigts {
            let (lo, hi) = split(v.e0, v.inv_s, ASYMP_X);
            let add = |o: &mut Complex64, w: Complex64| {
                o.re += v.amp_norm * w.im;
                o.im += v.amp_norm * w.re;
            };
            let far = |o: &mut Complex64, e: f64| add(o, w_asymp(Complex64::new((e + shift - v.e0) * v.inv_s, v.y)));
            out[..lo].iter_mut().zip(&energy[..lo]).for_each(|(o, &e)| far(o, e));
            out[hi..].iter_mut().zip(&energy[hi..]).for_each(|(o, &e)| far(o, e));
            for (o, &e) in out[lo..hi].iter_mut().zip(&energy[lo..hi]) {
                add(o, fast_w(Complex64::new((e + shift - v.e0) * v.inv_s, v.y)));
            }
        }
    }
}
