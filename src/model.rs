//! Composable optical-constant model: analytic KK-paired peaks + continuum.
//!
//! The model returns `delta + i*beta` (refloxide decrement form, `n = 1 - delta + i beta`)
//! at photon energy `E` (eV):
//!
//! ```text
//! f(E) = scale * [ sum_k peak_k(E + shift) + i*continuum_beta(E + shift) + delta_table(E + shift) ]
//! ```
//!
//! Every peak carries its exact Hilbert partner, `delta = H[beta]` with
//! `H f(x) = (1/pi) PV int f(t) / (x - t) dt`, so `delta < 0` below a resonance and
//! `delta > 0` above it. Amplitudes are the height of `beta` at `E0`.
//!
//! Build once with the chaining methods; during a fit, update parameters through the flat
//! parameter vector ([`Model::set_params`]) instead of rebuilding, or bind a fixed energy grid
//! with [`crate::Evaluator`] to cache per-component work.
//!
//! ```
//! use _core::Model;
//! let mut model = Model::new()
//!     .gaussian(933.8, 0.49, 4.1e-5)
//!     .voigt(935.7, 0.8, 0.3, 7.4e-5)
//!     .continuum(3.6e-4, -3.7e-7)
//!     .step(939.4, 6.1, 7.2e-5, 0.0);
//! let n = model.eval(&[930.0, 940.0]);
//! assert_eq!(n.len(), 2);
//! let mut theta = model.params();
//! theta[0] += 0.1; // gaussian[0].e0
//! model.set_params(&theta).unwrap();
//! ```

use num_complex::Complex64;

use crate::kernels::Kernels;
use crate::special::{interp, interp_at, is_sorted};

/// One user-facing model component, in its natural parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Component {
    /// `width` is the 1/e half-width (`beta(e0 +- width) = amp/e`, `sigma = width/sqrt(2)`).
    Gaussian { e0: f64, width: f64, amp: f64 },
    /// `gamma` is the half width at half maximum.
    Lorentzian { e0: f64, gamma: f64, amp: f64 },
    /// Exact Voigt: Gaussian std `sigma`, Lorentzian HWHM `gamma`.
    Voigt { e0: f64, sigma: f64, gamma: f64, amp: f64 },
    /// `eta*L + (1-eta)*G` with shared `fwhm`.
    PseudoVoigt { e0: f64, fwhm: f64, eta: f64, amp: f64 },
    /// Erf ionization step in beta with linear post-edge `decay`.
    Step { e0: f64, width: f64, height: f64, decay: f64 },
}

impl Component {
    /// Kind name used in parameter names.
    pub fn kind(&self) -> &'static str {
        match self {
            Component::Gaussian { .. } => "gaussian",
            Component::Lorentzian { .. } => "lorentzian",
            Component::Voigt { .. } => "voigt",
            Component::PseudoVoigt { .. } => "pseudo_voigt",
            Component::Step { .. } => "step",
        }
    }

    /// Parameter names, in parameter-vector order.
    pub fn param_names(&self) -> &'static [&'static str] {
        match self {
            Component::Gaussian { .. } => &["e0", "width", "amp"],
            Component::Lorentzian { .. } => &["e0", "gamma", "amp"],
            Component::Voigt { .. } => &["e0", "sigma", "gamma", "amp"],
            Component::PseudoVoigt { .. } => &["e0", "fwhm", "eta", "amp"],
            Component::Step { .. } => &["e0", "width", "height", "decay"],
        }
    }

    pub fn n_params(&self) -> usize {
        self.param_names().len()
    }

    fn write_params(&self, out: &mut Vec<f64>) {
        match *self {
            Component::Gaussian { e0, width, amp } => out.extend([e0, width, amp]),
            Component::Lorentzian { e0, gamma, amp } => out.extend([e0, gamma, amp]),
            Component::Voigt { e0, sigma, gamma, amp } => out.extend([e0, sigma, gamma, amp]),
            Component::PseudoVoigt { e0, fwhm, eta, amp } => out.extend([e0, fwhm, eta, amp]),
            Component::Step { e0, width, height, decay } => out.extend([e0, width, height, decay]),
        }
    }

    /// Same kind with parameters taken from `p` (length [`Component::n_params`]).
    fn with_params(&self, p: &[f64]) -> Component {
        match self {
            Component::Gaussian { .. } => Component::Gaussian { e0: p[0], width: p[1], amp: p[2] },
            Component::Lorentzian { .. } => Component::Lorentzian { e0: p[0], gamma: p[1], amp: p[2] },
            Component::Voigt { .. } => Component::Voigt { e0: p[0], sigma: p[1], gamma: p[2], amp: p[3] },
            Component::PseudoVoigt { .. } => Component::PseudoVoigt { e0: p[0], fwhm: p[1], eta: p[2], amp: p[3] },
            Component::Step { .. } => Component::Step { e0: p[0], width: p[1], height: p[2], decay: p[3] },
        }
    }

    /// Check parameter domains (finite; widths > 0; Voigt gamma >= 0; eta in [0, 1]).
    pub fn validate(&self) -> Result<(), String> {
        let mut p = Vec::with_capacity(4);
        self.write_params(&mut p);
        for (name, v) in self.param_names().iter().zip(&p) {
            if !v.is_finite() {
                return Err(format!("{}.{name} must be finite, got {v}", self.kind()));
            }
        }
        let positive = |name: &str, v: f64| {
            if v > 0.0 { Ok(()) } else { Err(format!("{}.{name} must be > 0, got {v}", self.kind())) }
        };
        match *self {
            Component::Gaussian { width, .. } => positive("width", width),
            Component::Lorentzian { gamma, .. } => positive("gamma", gamma),
            Component::Voigt { sigma, gamma, .. } => {
                positive("sigma", sigma)?;
                if gamma >= 0.0 { Ok(()) } else { Err(format!("voigt.gamma must be >= 0, got {gamma}")) }
            }
            Component::PseudoVoigt { fwhm, eta, .. } => {
                positive("fwhm", fwhm)?;
                if (0.0..=1.0).contains(&eta) { Ok(()) } else { Err(format!("pseudo_voigt.eta must be in [0, 1], got {eta}")) }
            }
            Component::Step { width, .. } => positive("width", width),
        }
    }
}

/// Number of global parameters appended after the components: bkg, slope, scale, shift.
pub const N_GLOBAL: usize = 4;
const GLOBAL_NAMES: [&str; N_GLOBAL] = ["continuum.bkg", "continuum.slope", "scale", "shift"];

/// A composable optical-constant model. Build with the chaining methods, then [`Model::eval`].
///
/// Parameter vector layout: each component's parameters in insertion order
/// (see [`Component::param_names`]), then `continuum.bkg, continuum.slope, scale, shift`.
#[derive(Clone, Debug)]
pub struct Model {
    components: Vec<Component>,
    pub(crate) kernels: Kernels,
    bkg: f64,
    slope: f64,
    table_e: Vec<f64>,
    table_delta: Vec<f64>,
    scale: f64,
    shift: f64,
}

impl Default for Model {
    fn default() -> Self {
        Self::new()
    }
}

impl Model {
    /// Empty model (evaluates to zero).
    pub fn new() -> Self {
        Self {
            components: Vec::new(),
            kernels: Kernels::default(),
            bkg: 0.0,
            slope: 0.0,
            table_e: Vec::new(),
            table_delta: Vec::new(),
            scale: 1.0,
            shift: 0.0,
        }
    }

    /// Add a component after validating it.
    pub fn try_add(&mut self, c: Component) -> Result<(), String> {
        c.validate()?;
        self.kernels.push(&c);
        self.components.push(c);
        Ok(())
    }

    fn add(&mut self, c: Component) {
        if let Err(e) = self.try_add(c) {
            panic!("{e}");
        }
    }

    /// Gaussian peak; `width` is the 1/e half-width. Panics on invalid parameters
    /// (use [`Model::try_add`] for a `Result`).
    pub fn gaussian(mut self, e0: f64, width: f64, amp: f64) -> Self {
        self.add(Component::Gaussian { e0, width, amp });
        self
    }

    /// Lorentzian peak with half width at half maximum `gamma`.
    pub fn lorentzian(mut self, e0: f64, gamma: f64, amp: f64) -> Self {
        self.add(Component::Lorentzian { e0, gamma, amp });
        self
    }

    /// Exact Voigt: Gaussian std `sigma` convolved with Lorentzian HWHM `gamma`.
    pub fn voigt(mut self, e0: f64, sigma: f64, gamma: f64, amp: f64) -> Self {
        self.add(Component::Voigt { e0, sigma, gamma, amp });
        self
    }

    /// Pseudo-Voigt `eta*L + (1-eta)*G` with shared `fwhm` (exact delta by linearity).
    pub fn pseudo_voigt(mut self, e0: f64, fwhm: f64, eta: f64, amp: f64) -> Self {
        self.add(Component::PseudoVoigt { e0, fwhm, eta, amp });
        self
    }

    /// Erf ionization step in beta with optional linear post-edge `decay`.
    pub fn step(mut self, e0: f64, width: f64, height: f64, decay: f64) -> Self {
        self.add(Component::Step { e0, width, height, decay });
        self
    }

    /// Linear continuum in beta: `bkg + slope*E` (replaces any previous continuum).
    pub fn continuum(mut self, bkg: f64, slope: f64) -> Self {
        self.set_continuum(bkg, slope);
        self
    }

    pub fn set_continuum(&mut self, bkg: f64, slope: f64) {
        self.bkg = bkg;
        self.slope = slope;
    }

    /// Fixed continuum delta (e.g. precomputed by kkcalc), linearly interpolated and
    /// clamped to the end values outside the table (like `numpy.interp`). Not a parameter.
    pub fn delta_table(mut self, energy: &[f64], delta: &[f64]) -> Self {
        self.set_delta_table(energy, delta);
        self
    }

    pub fn set_delta_table(&mut self, energy: &[f64], delta: &[f64]) {
        assert_eq!(energy.len(), delta.len(), "delta table energy/delta length mismatch");
        assert!(!energy.is_empty(), "delta table must be non-empty");
        assert!(energy.windows(2).all(|w| w[0] < w[1]), "delta table energies must be strictly increasing");
        self.table_e = energy.to_vec();
        self.table_delta = delta.to_vec();
    }

    /// Overall multiplicative scale (e.g. density).
    pub fn scale(mut self, scale: f64) -> Self {
        self.scale = scale;
        self
    }

    /// Energy offset: the model is evaluated at `E + shift`.
    pub fn shift(mut self, shift: f64) -> Self {
        self.shift = shift;
        self
    }

    pub fn set_scale(&mut self, scale: f64) {
        self.scale = scale;
    }

    pub fn set_shift(&mut self, shift: f64) {
        self.shift = shift;
    }

    // ------------------------------------------------------------- parameters

    /// The components in insertion order.
    pub fn components(&self) -> &[Component] {
        &self.components
    }

    /// Number of components (a pseudo-Voigt counts as one).
    pub fn n_components(&self) -> usize {
        self.components.len()
    }

    /// Length of the flat parameter vector.
    pub fn n_params(&self) -> usize {
        self.components.iter().map(Component::n_params).sum::<usize>() + N_GLOBAL
    }

    /// Current flat parameter vector.
    pub fn params(&self) -> Vec<f64> {
        let mut p = Vec::with_capacity(self.n_params());
        for c in &self.components {
            c.write_params(&mut p);
        }
        p.extend([self.bkg, self.slope, self.scale, self.shift]);
        p
    }

    /// Names matching [`Model::params`], e.g. `"gaussian[0].e0"`, `"step[1].height"`, `"scale"`.
    /// Indices count components of the same kind.
    pub fn param_names(&self) -> Vec<String> {
        let mut counts = std::collections::HashMap::<&str, usize>::new();
        let mut names = Vec::with_capacity(self.n_params());
        for c in &self.components {
            let k = counts.entry(c.kind()).or_insert(0);
            for p in c.param_names() {
                names.push(format!("{}[{}].{}", c.kind(), k, p));
            }
            *k += 1;
        }
        names.extend(GLOBAL_NAMES.iter().map(|s| s.to_string()));
        names
    }

    /// Set every parameter from a flat vector (validated; on error the model is unchanged).
    /// Cost is O(number of components): no energy-grid work.
    pub fn set_params(&mut self, theta: &[f64]) -> Result<(), String> {
        if theta.len() != self.n_params() {
            return Err(format!("expected {} parameters, got {}", self.n_params(), theta.len()));
        }
        let mut new = Vec::with_capacity(self.components.len());
        let mut i = 0;
        for c in &self.components {
            let n = c.n_params();
            let c2 = c.with_params(&theta[i..i + n]);
            c2.validate()?;
            new.push(c2);
            i += n;
        }
        let g = &theta[i..];
        if let Some(v) = g.iter().find(|v| !v.is_finite()) {
            return Err(format!("global parameters must be finite, got {v}"));
        }
        self.components = new;
        self.kernels.clear();
        for c in &self.components {
            self.kernels.push(c);
        }
        (self.bkg, self.slope, self.scale, self.shift) = (g[0], g[1], g[2], g[3]);
        Ok(())
    }

    pub(crate) fn global(&self) -> (f64, f64, f64, f64) {
        (self.bkg, self.slope, self.scale, self.shift)
    }

    // ------------------------------------------------------------- evaluation

    /// Evaluate `delta + i*beta` on `energy` (single core).
    pub fn eval(&self, energy: &[f64]) -> Vec<Complex64> {
        let mut out = vec![Complex64::new(0.0, 0.0); energy.len()];
        self.eval_into(energy, &mut out);
        out
    }

    /// Evaluate into a caller-provided buffer (single core, no allocation).
    pub fn eval_into(&self, energy: &[f64], out: &mut [Complex64]) {
        assert_eq!(energy.len(), out.len(), "energy/out length mismatch");
        let sorted = is_sorted(energy);
        self.init_continuum(energy, out);
        self.kernels.accumulate(energy, self.shift, sorted, out);
        self.add_table(energy, self.shift, sorted, out);
        self.apply_scale(out);
    }

    /// `out = i*(bkg + slope*(E + shift))`.
    pub(crate) fn init_continuum(&self, energy: &[f64], out: &mut [Complex64]) {
        let shift = self.shift;
        for (o, &e) in out.iter_mut().zip(energy) {
            *o = Complex64::new(0.0, self.bkg + self.slope * (e + shift));
        }
    }

    pub(crate) fn apply_scale(&self, out: &mut [Complex64]) {
        if self.scale != 1.0 {
            for o in out.iter_mut() {
                *o *= self.scale;
            }
        }
    }

    pub(crate) fn has_table(&self) -> bool {
        !self.table_e.is_empty()
    }

    /// Add the delta table evaluated at `E + shift` to `out.re`.
    pub(crate) fn add_table(&self, energy: &[f64], shift: f64, sorted: bool, out: &mut [Complex64]) {
        if self.table_e.is_empty() {
            return;
        }
        let (xs, ys) = (&self.table_e[..], &self.table_delta[..]);
        if sorted {
            // Sorted input: walk the table with a cursor (O(n + m)).
            let mut i = 1usize;
            for (o, &e) in out.iter_mut().zip(energy) {
                let x = e + shift;
                while i < xs.len() - 1 && xs[i] <= x {
                    i += 1;
                }
                o.re += interp_at(xs, ys, x, i);
            }
        } else {
            for (o, &e) in out.iter_mut().zip(energy) {
                o.re += interp(xs, ys, e + shift);
            }
        }
    }

    /// Add one component's contribution at `E + shift` into `out` (used by the evaluator cache).
    pub(crate) fn accumulate_component(c: &Component, energy: &[f64], shift: f64, sorted: bool, out: &mut [Complex64]) {
        let mut k = Kernels::default();
        k.push(c);
        k.accumulate(energy, shift, sorted, out);
    }

    /// Evaluate on all cores (rayon), chunked; results identical to [`Model::eval_into`].
    #[cfg(feature = "parallel")]
    pub fn eval_par_into(&self, energy: &[f64], out: &mut [Complex64]) {
        use rayon::prelude::*;
        assert_eq!(energy.len(), out.len(), "energy/out length mismatch");
        const CHUNK: usize = 2048;
        out.par_chunks_mut(CHUNK)
            .zip(energy.par_chunks(CHUNK))
            .for_each(|(o, e)| self.eval_into(e, o));
    }

    /// Allocating form of [`Model::eval_par_into`].
    #[cfg(feature = "parallel")]
    pub fn eval_par(&self, energy: &[f64]) -> Vec<Complex64> {
        let mut out = vec![Complex64::new(0.0, 0.0); energy.len()];
        self.eval_par_into(energy, &mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::special::{fast_dawson, fast_w};
    use errorfunctions::{ComplexErrorFunctions, RealErrorFunctions};

    #[test]
    fn peak_height_and_sign() {
        for m in [
            Model::new().gaussian(930.0, 1.0, 2.5),
            Model::new().lorentzian(930.0, 0.5, 2.5),
            Model::new().voigt(930.0, 0.5, 0.3, 2.5),
            Model::new().pseudo_voigt(930.0, 1.2, 0.4, 2.5),
        ] {
            let z = m.eval(&[929.0, 930.0, 931.0]);
            assert!((z[1].im - 2.5).abs() < 1e-12);
            assert!(z[1].re.abs() < 1e-12);
            assert!(z[0].re < 0.0 && z[2].re > 0.0);
        }
    }

    #[test]
    fn fast_dawson_matches_reference() {
        let mut worst: f64 = 0.0;
        for i in 0..200_000 {
            let x = 9.0 + i as f64 * 0.005; // spans the 10.0 switch point up to ~1009
            for x in [x, -x] {
                let r = (fast_dawson(x) - x.dawson()).abs() / x.dawson().abs();
                worst = worst.max(r);
            }
        }
        assert!(worst < 5e-15, "worst rel err {worst}");
    }

    #[test]
    fn fast_w_matches_reference() {
        let mut worst: f64 = 0.0;
        for i in 0..2000 {
            for j in 0..200 {
                let z = Complex64::new(-300.0 + i as f64 * 0.3, j as f64 * 0.1);
                if z.norm_sqr() < 81.0 {
                    continue;
                }
                let r = (fast_w(z) - z.w()).norm() / z.w().norm();
                worst = worst.max(r);
            }
        }
        assert!(worst < 5e-15, "worst rel err {worst}");
    }

    #[test]
    fn fast_erf_is_exact_at_cutoff() {
        assert_eq!(RealErrorFunctions::erf(6.0), 1.0);
        assert_eq!(RealErrorFunctions::erf(-6.0), -1.0);
    }

    #[test]
    fn sorted_and_unsorted_interp_agree() {
        let xs: Vec<f64> = (0..50).map(|i| i as f64 * 0.37).collect();
        let ys: Vec<f64> = xs.iter().map(|x| x.sin()).collect();
        let m = Model::new().delta_table(&xs, &ys);
        let sorted: Vec<f64> = (0..400).map(|i| -1.0 + i as f64 * 0.05).collect();
        let mut rev = sorted.clone();
        rev.reverse();
        let mut a = m.eval(&sorted);
        a.reverse();
        assert_eq!(a, m.eval(&rev));
    }

    #[test]
    fn sorted_fast_path_matches_unsorted() {
        let m = Model::new()
            .gaussian(930.0, 1.0, 1.0)
            .voigt(935.0, 0.5, 0.3, 1.0)
            .lorentzian(940.0, 0.4, 1.0)
            .step(938.0, 3.0, 1.0, 0.01)
            .continuum(0.1, -1e-4)
            .shift(0.3)
            .scale(1.7);
        let e: Vec<f64> = (0..8001).map(|i| 880.0 + i as f64 * 0.0125).collect();
        // deterministic shuffle
        let mut idx: Vec<usize> = (0..e.len()).collect();
        let mut st = 12345u64;
        for i in (1..idx.len()).rev() {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            idx.swap(i, (st >> 33) as usize % (i + 1));
        }
        let shuffled: Vec<f64> = idx.iter().map(|&i| e[i]).collect();
        let a = m.eval(&e);
        let b = m.eval(&shuffled);
        for (k, &i) in idx.iter().enumerate() {
            assert_eq!(a[i], b[k], "mismatch at E = {}", e[i]);
        }
    }

    #[test]
    fn interp_clamps() {
        let m = Model::new().delta_table(&[1.0, 2.0], &[10.0, 20.0]);
        let z = m.eval(&[0.0, 1.5, 3.0]);
        assert_eq!([z[0].re, z[1].re, z[2].re], [10.0, 15.0, 20.0]);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_matches_serial() {
        let m = Model::new().gaussian(930.0, 1.0, 1.0).voigt(940.0, 0.5, 0.3, 1.0).step(935.0, 2.0, 1.0, 0.1);
        let e: Vec<f64> = (0..10_000).map(|i| 900.0 + i as f64 * 0.01).collect();
        assert_eq!(m.eval(&e), m.eval_par(&e));
    }
}
