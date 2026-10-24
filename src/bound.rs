//! [`Evaluator`]: a model bound to a fixed energy grid, for fitting loops.
//!
//! Binding fixes the grid once (sortedness, delta-table interpolation) and optionally caches
//! each component's contribution. A call then recomputes only the components whose parameters
//! changed since the previous call, so finite-difference Jacobian columns or fits with many
//! frozen peaks cost about one component instead of the whole model. `scale`, `continuum.bkg`
//! and `continuum.slope` never trigger component work; changing `shift` recomputes everything.
//!
//! ```
//! use _core::{Evaluator, Model};
//! let model = Model::new().gaussian(930.0, 1.0, 1.0).gaussian(940.0, 2.0, 0.5);
//! let energy: Vec<f64> = (0..1000).map(|i| 900.0 + 0.1 * i as f64).collect();
//! let mut ev = Evaluator::new(model, &energy, true);
//! let mut out = vec![Default::default(); energy.len()];
//! let mut theta = ev.params();
//! ev.eval_params_into(&theta, &mut out).unwrap();
//! theta[3] += 1e-6; // gaussian[1].e0: only that peak is recomputed
//! ev.eval_params_into(&theta, &mut out).unwrap();
//! ```

use num_complex::Complex64;

use crate::model::{Component, Model};
use crate::special::is_sorted;

/// Full re-summation of the cached contributions after this many incremental updates
/// (bounds floating-point drift from `sum += new - old`).
const RESUM_EVERY: usize = 64;

#[derive(Clone, Debug)]
struct Cache {
    /// Parameters each `contrib[k]` was computed with (`None` = stale).
    comps: Vec<Option<Component>>,
    shift: f64,
    contrib: Vec<Vec<Complex64>>,
    sum: Vec<Complex64>,
    scratch: Vec<Complex64>,
    updates: usize,
}

/// A [`Model`] bound to a fixed energy grid; see the module docs.
#[derive(Clone, Debug)]
pub struct Evaluator {
    model: Model,
    energy: Vec<f64>,
    sorted: bool,
    /// Delta table interpolated at `E + shift`, with the shift it was computed for.
    table: Option<(f64, Vec<f64>)>,
    cache: Option<Cache>,
}

impl Evaluator {
    /// Bind `model` to `energy`. With `cache = true`, per-component contributions are kept
    /// (memory: `n_components * len(energy) * 16` bytes).
    pub fn new(model: Model, energy: &[f64], cache: bool) -> Self {
        let n = energy.len();
        let nc = model.n_components();
        let cache = cache.then(|| Cache {
            comps: vec![None; nc],
            shift: f64::NAN,
            contrib: vec![vec![Complex64::new(0.0, 0.0); n]; nc],
            sum: vec![Complex64::new(0.0, 0.0); n],
            scratch: vec![Complex64::new(0.0, 0.0); n],
            updates: 0,
        });
        Self { sorted: is_sorted(energy), energy: energy.to_vec(), model, table: None, cache }
    }

    pub fn model(&self) -> &Model {
        &self.model
    }

    pub fn energy(&self) -> &[f64] {
        &self.energy
    }

    pub fn params(&self) -> Vec<f64> {
        self.model.params()
    }

    pub fn param_names(&self) -> Vec<String> {
        self.model.param_names()
    }

    pub fn n_params(&self) -> usize {
        self.model.n_params()
    }

    pub fn set_params(&mut self, theta: &[f64]) -> Result<(), String> {
        self.model.set_params(theta)
    }

    /// Set parameters, then evaluate into `out`.
    pub fn eval_params_into(&mut self, theta: &[f64], out: &mut [Complex64]) -> Result<(), String> {
        self.model.set_params(theta)?;
        self.eval_into(out);
        Ok(())
    }

    /// Evaluate at the current parameters into `out` (length of the bound grid).
    pub fn eval_into(&mut self, out: &mut [Complex64]) {
        assert_eq!(out.len(), self.energy.len(), "out length must match the bound energy grid");
        let (bkg, slope, scale, shift) = self.model.global();
        let (energy, sorted) = (&self.energy[..], self.sorted);

        // Peaks + steps: cached sum, or a direct pass.
        match &mut self.cache {
            Some(c) => {
                update_cache(c, &self.model, energy, shift, sorted);
                out.copy_from_slice(&c.sum);
            }
            None => {
                out.fill(Complex64::new(0.0, 0.0));
                self.model.kernels.accumulate(energy, shift, sorted, out);
            }
        }
        // Continuum beta (cheap; recomputed every call).
        for (o, &e) in out.iter_mut().zip(energy) {
            o.im += bkg + slope * (e + shift);
        }
        // Fixed delta table, interpolated once per distinct shift.
        if self.model.has_table() {
            if self.table.as_ref().is_none_or(|(s, _)| *s != shift) {
                let mut tmp = vec![Complex64::new(0.0, 0.0); energy.len()];
                self.model.add_table(energy, shift, sorted, &mut tmp);
                self.table = Some((shift, tmp.iter().map(|z| z.re).collect()));
            }
            let (_, t) = self.table.as_ref().unwrap();
            for (o, &d) in out.iter_mut().zip(t) {
                o.re += d;
            }
        }
        if scale != 1.0 {
            for o in out.iter_mut() {
                *o *= scale;
            }
        }
    }

    /// Allocating form of [`Evaluator::eval_into`].
    pub fn eval(&mut self) -> Vec<Complex64> {
        let mut out = vec![Complex64::new(0.0, 0.0); self.energy.len()];
        self.eval_into(&mut out);
        out
    }
}

fn update_cache(c: &mut Cache, model: &Model, energy: &[f64], shift: f64, sorted: bool) {
    if c.shift != shift {
        c.comps.iter_mut().for_each(|p| *p = None);
        c.shift = shift;
    }
    let comps = model.components();
    let changed: Vec<usize> = (0..comps.len()).filter(|&k| c.comps[k] != Some(comps[k])).collect();
    if changed.is_empty() {
        return;
    }
    let incremental = 2 * changed.len() <= comps.len() && c.updates < RESUM_EVERY;
    for &k in &changed {
        c.scratch.fill(Complex64::new(0.0, 0.0));
        Model::accumulate_component(&comps[k], energy, shift, sorted, &mut c.scratch);
        if incremental {
            for ((s, &new), &old) in c.sum.iter_mut().zip(&c.scratch).zip(&c.contrib[k]) {
                *s += new - old;
            }
        }
        std::mem::swap(&mut c.scratch, &mut c.contrib[k]);
        c.comps[k] = Some(comps[k]);
    }
    if incremental {
        c.updates += 1;
    } else {
        c.sum.fill(Complex64::new(0.0, 0.0));
        for col in &c.contrib {
            for (s, &v) in c.sum.iter_mut().zip(col) {
                *s += v;
            }
        }
        c.updates = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> Model {
        Model::new()
            .gaussian(930.0, 1.0, 1.0)
            .voigt(935.0, 0.5, 0.3, 0.8)
            .lorentzian(940.0, 0.4, 0.6)
            .pseudo_voigt(945.0, 1.2, 0.4, 0.5)
            .step(938.0, 3.0, 1.0, 0.01)
            .continuum(0.1, -1e-4)
            .delta_table(&[900.0, 950.0, 1000.0], &[0.1, -0.2, 0.3])
            .scale(1.3)
            .shift(0.2)
    }

    fn rel_err(a: &[Complex64], b: &[Complex64]) -> f64 {
        let m = b.iter().map(|z| z.norm()).fold(0.0, f64::max);
        a.iter().zip(b).map(|(x, y)| (x - y).norm()).fold(0.0, f64::max) / m
    }

    #[test]
    fn cached_matches_direct_through_random_updates() {
        let energy: Vec<f64> = (0..3000).map(|i| 890.0 + 0.04 * i as f64).collect();
        let mut m = model();
        let mut ev = Evaluator::new(m.clone(), &energy, true);
        let mut ev_nc = Evaluator::new(m.clone(), &energy, false);
        let mut theta = m.params();
        let mut out = vec![Complex64::new(0.0, 0.0); energy.len()];
        let mut out2 = out.clone();
        let mut st = 7u64;
        let mut rnd = || {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (st >> 11) as f64 / (1u64 << 53) as f64
        };
        for it in 0..300 {
            // perturb one parameter (FD-like), occasionally all of them, or a global
            let k = (rnd() * theta.len() as f64) as usize;
            let names = m.param_names();
            let bump = |v: f64, name: &str, r: f64| {
                if name.ends_with("eta") { (v + 0.01 * (r - 0.5)).clamp(0.0, 1.0) } else { v * (1.0 + 0.01 * (r - 0.5)) + 1e-3 * (r - 0.5) * (name.ends_with("e0") as u8 as f64) }
            };
            if it % 50 == 0 {
                for j in 0..theta.len() {
                    theta[j] = bump(theta[j], &names[j], rnd());
                }
            } else {
                theta[k] = bump(theta[k], &names[k], rnd());
            }
            ev.eval_params_into(&theta, &mut out).unwrap();
            ev_nc.eval_params_into(&theta, &mut out2).unwrap();
            m.set_params(&theta).unwrap();
            let direct = m.eval(&energy);
            assert!(rel_err(&out, &direct) < 1e-13, "cached it {it}: {}", rel_err(&out, &direct));
            assert!(rel_err(&out2, &direct) < 1e-15, "uncached it {it}");
        }
    }

    #[test]
    fn params_roundtrip_and_names() {
        let mut m = model();
        let p = m.params();
        assert_eq!(p.len(), m.n_params());
        assert_eq!(m.param_names().len(), p.len());
        assert_eq!(m.param_names()[0], "gaussian[0].e0");
        assert_eq!(m.param_names()[11], "pseudo_voigt[0].fwhm");
        assert_eq!(&m.param_names()[p.len() - 4..], &["continuum.bkg", "continuum.slope", "scale", "shift"]);
        m.set_params(&p).unwrap();
        assert_eq!(m.params(), p);
        let mut bad = p.clone();
        bad[1] = -1.0;
        assert!(m.set_params(&bad).is_err());
        assert_eq!(m.params(), p, "model unchanged after a failed set_params");
    }
}
