//! WebAssembly bindings (`wasm-bindgen`) for `lineshapes`:
//! [`WasmModel`] (composable optical-constant model) and [`WasmEvaluator`] (model bound to an energy grid).

use std::f64::consts::PI;

use lineshapes::{Complex64, Component, Evaluator, Model, fast_dawson, fast_erf, fast_w};
use wasm_bindgen::prelude::*;

/// Planck's constant times the speed of light (`h * c`) in `eV * Angstrom`.
pub const HC_EV_ANGSTROM: f64 = 12398.419843320026;

fn js_err(e: String) -> JsError {
    JsError::new(&e)
}

fn validate_finite(name: &str, v: f64) -> Result<(), String> {
    if v.is_finite() {
        Ok(())
    } else {
        Err(format!("{name} must be finite, got {v}"))
    }
}

fn validate_delta_table(energy: &[f64], delta: &[f64]) -> Result<(), String> {
    if energy.len() != delta.len() || energy.is_empty() {
        return Err("delta table energy/delta must be non-empty and equal length".into());
    }
    if energy.iter().any(|v| !v.is_finite()) || delta.iter().any(|v| !v.is_finite()) {
        return Err("delta table energy/delta values must be finite".into());
    }
    if !energy.windows(2).all(|w| w[0] < w[1]) {
        return Err("delta table energies must be strictly increasing".into());
    }
    Ok(())
}

fn split_complex(
    src: &[Complex64],
    delta_out: &mut [f64],
    beta_out: &mut [f64],
) -> Result<(), String> {
    let n = src.len();
    if delta_out.len() != n {
        return Err(format!(
            "delta_out has length {}, expected {n}",
            delta_out.len()
        ));
    }
    if beta_out.len() != n {
        return Err(format!(
            "beta_out has length {}, expected {n}",
            beta_out.len()
        ));
    }
    for ((d, b), z) in delta_out.iter_mut().zip(beta_out.iter_mut()).zip(src) {
        *d = z.re;
        *b = z.im;
    }
    Ok(())
}

fn write_interleaved(src: &[Complex64], out: &mut [f64]) -> Result<(), String> {
    let expected = 2 * src.len();
    if out.len() != expected {
        return Err(format!(
            "interleaved out has length {}, expected {expected}",
            out.len()
        ));
    }
    for ([re, im], z) in out.as_chunks_mut::<2>().0.iter_mut().zip(src) {
        *re = z.re;
        *im = z.im;
    }
    Ok(())
}

fn complex_to_interleaved(src: &[Complex64]) -> Vec<f64> {
    let mut out = vec![0.0; 2 * src.len()];
    for ([re, im], z) in out.as_chunks_mut::<2>().0.iter_mut().zip(src) {
        *re = z.re;
        *im = z.im;
    }
    out
}

/// Composable optical-constant model returning `delta + i*beta` on an energy grid in eV.
#[wasm_bindgen(js_name = Model)]
#[derive(Clone, Debug, Default)]
pub struct WasmModel {
    inner: Model,
    scratch: Vec<Complex64>,
}

impl WasmModel {
    fn try_add(&mut self, c: Component) -> Result<(), String> {
        self.inner.try_add(c)
    }

    fn try_continuum(&mut self, bkg: f64, slope: f64) -> Result<(), String> {
        validate_finite("continuum.bkg", bkg)?;
        validate_finite("continuum.slope", slope)?;
        self.inner.set_continuum(bkg, slope);
        Ok(())
    }

    fn try_delta_table(&mut self, energy: &[f64], delta: &[f64]) -> Result<(), String> {
        validate_delta_table(energy, delta)?;
        self.inner.set_delta_table(energy, delta);
        Ok(())
    }

    fn try_scale(&mut self, scale: f64) -> Result<(), String> {
        validate_finite("scale", scale)?;
        self.inner.set_scale(scale);
        Ok(())
    }

    fn try_shift(&mut self, shift: f64) -> Result<(), String> {
        validate_finite("shift", shift)?;
        self.inner.set_shift(shift);
        Ok(())
    }

    fn eval_scratch(&mut self, energy: &[f64]) -> &[Complex64] {
        self.scratch.resize(energy.len(), Complex64::new(0.0, 0.0));
        self.inner.eval_into(energy, &mut self.scratch);
        &self.scratch
    }

    fn try_eval_into(
        &mut self,
        energy: &[f64],
        delta_out: &mut [f64],
        beta_out: &mut [f64],
    ) -> Result<(), String> {
        let src = self.eval_scratch(energy);
        split_complex(src, delta_out, beta_out)
    }

    fn try_eval_params_into(
        &self,
        energy: &[f64],
        params: &[f64],
        delta_out: &mut [f64],
        beta_out: &mut [f64],
    ) -> Result<(), String> {
        let mut tmp = self.inner.clone();
        tmp.set_params(params)?;
        let out = tmp.eval(energy);
        split_complex(&out, delta_out, beta_out)
    }
}

#[wasm_bindgen(js_class = Model)]
impl WasmModel {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: Model::new(),
            scratch: Vec::new(),
        }
    }

    /// Gaussian peak; `width` is the 1/e half-width (`sigma = width / sqrt(2)`).
    pub fn gaussian(&mut self, e0: f64, width: f64, amp: f64) -> Result<(), JsError> {
        self.try_add(Component::Gaussian { e0, width, amp })
            .map_err(js_err)
    }

    /// Lorentzian peak; `gamma` is the half width at half maximum.
    pub fn lorentzian(&mut self, e0: f64, gamma: f64, amp: f64) -> Result<(), JsError> {
        self.try_add(Component::Lorentzian { e0, gamma, amp })
            .map_err(js_err)
    }

    /// Exact Voigt; `sigma` Gaussian std, `gamma` Lorentzian HWHM.
    pub fn voigt(&mut self, e0: f64, sigma: f64, gamma: f64, amp: f64) -> Result<(), JsError> {
        self.try_add(Component::Voigt {
            e0,
            sigma,
            gamma,
            amp,
        })
        .map_err(js_err)
    }

    /// Pseudo-Voigt `eta*L + (1-eta)*G` with shared `fwhm`.
    #[wasm_bindgen(js_name = pseudoVoigt)]
    pub fn pseudo_voigt(&mut self, e0: f64, fwhm: f64, eta: f64, amp: f64) -> Result<(), JsError> {
        self.try_add(Component::PseudoVoigt { e0, fwhm, eta, amp })
            .map_err(js_err)
    }

    /// Erf ionization step in beta, with optional linear post-edge `decay`.
    pub fn step(&mut self, e0: f64, width: f64, height: f64, decay: f64) -> Result<(), JsError> {
        self.try_add(Component::Step {
            e0,
            width,
            height,
            decay,
        })
        .map_err(js_err)
    }

    /// Linear continuum in beta: `bkg + slope*E`.
    pub fn continuum(&mut self, bkg: f64, slope: f64) -> Result<(), JsError> {
        self.try_continuum(bkg, slope).map_err(js_err)
    }

    /// Fixed continuum delta table (e.g. from kkcalc), interpolated like `numpy.interp`.
    #[wasm_bindgen(js_name = deltaTable)]
    pub fn delta_table(&mut self, energy: &[f64], delta: &[f64]) -> Result<(), JsError> {
        self.try_delta_table(energy, delta).map_err(js_err)
    }

    /// Overall multiplicative scale.
    pub fn scale(&mut self, scale: f64) -> Result<(), JsError> {
        self.try_scale(scale).map_err(js_err)
    }

    /// Energy offset: the model is evaluated at `E + shift`.
    pub fn shift(&mut self, shift: f64) -> Result<(), JsError> {
        self.try_shift(shift).map_err(js_err)
    }

    /// Independent copy (so a base model can be extended in several ways).
    pub fn copy(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            scratch: Vec::new(),
        }
    }

    /// Number of peaks and steps (a pseudo-Voigt counts as one).
    #[wasm_bindgen(getter, js_name = nComponents)]
    pub fn n_components(&self) -> usize {
        self.inner.n_components()
    }

    /// Length of the flat parameter vector.
    #[wasm_bindgen(getter, js_name = nParams)]
    pub fn n_params(&self) -> usize {
        self.inner.n_params()
    }

    /// Current flat parameter vector (components in insertion order, then
    /// `continuum.bkg, continuum.slope, scale, shift`).
    #[wasm_bindgen(getter)]
    pub fn params(&self) -> Vec<f64> {
        self.inner.params()
    }

    /// Names matching `params`, e.g. `'gaussian[0].e0'`, `'step[1].height'`, `'scale'`.
    #[wasm_bindgen(getter, js_name = paramNames)]
    pub fn param_names(&self) -> Vec<String> {
        self.inner.param_names()
    }

    /// Set all parameters from a flat vector (validated; model unchanged on error).
    #[wasm_bindgen(js_name = setParams)]
    pub fn set_params(&mut self, params: &[f64]) -> Result<(), JsError> {
        self.inner.set_params(params).map_err(js_err)
    }

    /// Bind to a fixed energy grid for repeated evaluation (see `Evaluator`).
    pub fn bind(&self, energy: &[f64], cache: bool) -> WasmEvaluator {
        WasmEvaluator::new(self, energy, cache)
    }

    /// Evaluate `delta` and `beta` on `energy` into caller-supplied slices.
    #[wasm_bindgen(js_name = evalInto)]
    pub fn eval_into(
        &mut self,
        energy: &[f64],
        delta_out: &mut [f64],
        beta_out: &mut [f64],
    ) -> Result<(), JsError> {
        self.try_eval_into(energy, delta_out, beta_out)
            .map_err(js_err)
    }

    /// Evaluate `delta` and `beta` on `energy` at `params` without mutating the model.
    #[wasm_bindgen(js_name = evalParamsInto)]
    pub fn eval_params_into(
        &self,
        energy: &[f64],
        params: &[f64],
        delta_out: &mut [f64],
        beta_out: &mut [f64],
    ) -> Result<(), JsError> {
        self.try_eval_params_into(energy, params, delta_out, beta_out)
            .map_err(js_err)
    }

    /// Evaluate on `energy`, returning interleaved `[delta_0, beta_0, delta_1, beta_1, ...]`.
    #[wasm_bindgen(js_name = evalInterleaved)]
    pub fn eval_interleaved(&mut self, energy: &[f64]) -> Vec<f64> {
        complex_to_interleaved(self.eval_scratch(energy))
    }

    /// Evaluate on `energy` at `params` without mutating the model, returning interleaved
    /// `[delta_0, beta_0, delta_1, beta_1, ...]`.
    #[wasm_bindgen(js_name = evalParamsInterleaved)]
    pub fn eval_params_interleaved(
        &self,
        energy: &[f64],
        params: &[f64],
    ) -> Result<Vec<f64>, JsError> {
        let mut tmp = self.inner.clone();
        tmp.set_params(params).map_err(js_err)?;
        Ok(complex_to_interleaved(&tmp.eval(energy)))
    }

    /// Evaluate only the real refractive decrement `delta` on `energy`.
    #[wasm_bindgen(js_name = evalDelta)]
    pub fn eval_delta(&mut self, energy: &[f64]) -> Vec<f64> {
        self.eval_scratch(energy).iter().map(|z| z.re).collect()
    }

    /// Evaluate only `delta` on `energy` at `params` without mutating the model.
    #[wasm_bindgen(js_name = evalParamsDelta)]
    pub fn eval_params_delta(&self, energy: &[f64], params: &[f64]) -> Result<Vec<f64>, JsError> {
        let mut tmp = self.inner.clone();
        tmp.set_params(params).map_err(js_err)?;
        Ok(tmp.eval(energy).iter().map(|z| z.re).collect())
    }

    /// Evaluate only the absorption index `beta` on `energy`.
    #[wasm_bindgen(js_name = evalBeta)]
    pub fn eval_beta(&mut self, energy: &[f64]) -> Vec<f64> {
        self.eval_scratch(energy).iter().map(|z| z.im).collect()
    }

    /// Evaluate only `beta` on `energy` at `params` without mutating the model.
    #[wasm_bindgen(js_name = evalParamsBeta)]
    pub fn eval_params_beta(&self, energy: &[f64], params: &[f64]) -> Result<Vec<f64>, JsError> {
        let mut tmp = self.inner.clone();
        tmp.set_params(params).map_err(js_err)?;
        Ok(tmp.eval(energy).iter().map(|z| z.im).collect())
    }

    #[wasm_bindgen(js_name = toString)]
    pub fn to_string_repr(&self) -> String {
        format!(
            "Model(n_components={}, n_params={})",
            self.inner.n_components(),
            self.inner.n_params()
        )
    }
}

/// A `Model` bound to a fixed energy grid (from `Model.bind`), for fitting loops.
///
/// `scale` and `continuum.*` never trigger component work; changing `shift` recomputes all.
#[wasm_bindgen(js_name = Evaluator)]
#[derive(Clone, Debug)]
pub struct WasmEvaluator {
    inner: Evaluator,
    scratch: Vec<Complex64>,
}

impl WasmEvaluator {
    fn try_eval_into(&mut self, delta_out: &mut [f64], beta_out: &mut [f64]) -> Result<(), String> {
        let n = self.inner.energy().len();
        if delta_out.len() != n {
            return Err(format!(
                "delta_out has length {}, expected {n}",
                delta_out.len()
            ));
        }
        if beta_out.len() != n {
            return Err(format!(
                "beta_out has length {}, expected {n}",
                beta_out.len()
            ));
        }
        self.inner.eval_into(&mut self.scratch);
        split_complex(&self.scratch, delta_out, beta_out)
    }

    fn try_eval_params_into(
        &mut self,
        params: &[f64],
        delta_out: &mut [f64],
        beta_out: &mut [f64],
    ) -> Result<(), String> {
        let n = self.inner.energy().len();
        if delta_out.len() != n {
            return Err(format!(
                "delta_out has length {}, expected {n}",
                delta_out.len()
            ));
        }
        if beta_out.len() != n {
            return Err(format!(
                "beta_out has length {}, expected {n}",
                beta_out.len()
            ));
        }
        self.inner.eval_params_into(params, &mut self.scratch)?;
        split_complex(&self.scratch, delta_out, beta_out)
    }

    fn try_eval_interleaved_into(&mut self, out: &mut [f64]) -> Result<(), String> {
        let expected = 2 * self.inner.energy().len();
        if out.len() != expected {
            return Err(format!(
                "interleaved out has length {}, expected {expected}",
                out.len()
            ));
        }
        self.inner.eval_into(&mut self.scratch);
        write_interleaved(&self.scratch, out)
    }

    fn try_eval_params_interleaved_into(
        &mut self,
        params: &[f64],
        out: &mut [f64],
    ) -> Result<(), String> {
        let expected = 2 * self.inner.energy().len();
        if out.len() != expected {
            return Err(format!(
                "interleaved out has length {}, expected {expected}",
                out.len()
            ));
        }
        self.inner.eval_params_into(params, &mut self.scratch)?;
        write_interleaved(&self.scratch, out)
    }
}

#[wasm_bindgen(js_class = Evaluator)]
impl WasmEvaluator {
    #[wasm_bindgen(constructor)]
    pub fn new(model: &WasmModel, energy: &[f64], cache: bool) -> Self {
        let n = energy.len();
        Self {
            inner: Evaluator::new(model.inner.clone(), energy, cache),
            scratch: vec![Complex64::new(0.0, 0.0); n],
        }
    }

    /// Length of the flat parameter vector.
    #[wasm_bindgen(getter, js_name = nParams)]
    pub fn n_params(&self) -> usize {
        self.inner.n_params()
    }

    /// Number of points in the bound energy grid.
    #[wasm_bindgen(getter, js_name = nEnergy)]
    pub fn n_energy(&self) -> usize {
        self.inner.energy().len()
    }

    /// Current parameter vector.
    #[wasm_bindgen(getter)]
    pub fn params(&self) -> Vec<f64> {
        self.inner.params()
    }

    /// Names matching `params`.
    #[wasm_bindgen(getter, js_name = paramNames)]
    pub fn param_names(&self) -> Vec<String> {
        self.inner.param_names()
    }

    /// Bound energy grid in eV.
    #[wasm_bindgen(getter)]
    pub fn energy(&self) -> Vec<f64> {
        self.inner.energy().to_vec()
    }

    /// Copy of the bound model at the current parameters.
    #[wasm_bindgen(getter)]
    pub fn model(&self) -> WasmModel {
        WasmModel {
            inner: self.inner.model().clone(),
            scratch: Vec::new(),
        }
    }

    /// Set the parameters without evaluating.
    #[wasm_bindgen(js_name = setParams)]
    pub fn set_params(&mut self, params: &[f64]) -> Result<(), JsError> {
        self.inner.set_params(params).map_err(js_err)
    }

    /// Evaluate at the current parameters into `delta_out` and `beta_out`.
    #[wasm_bindgen(js_name = evalInto)]
    pub fn eval_into(
        &mut self,
        delta_out: &mut [f64],
        beta_out: &mut [f64],
    ) -> Result<(), JsError> {
        self.try_eval_into(delta_out, beta_out).map_err(js_err)
    }

    /// Set `params` and evaluate into `delta_out` and `beta_out`.
    #[wasm_bindgen(js_name = evalParamsInto)]
    pub fn eval_params_into(
        &mut self,
        params: &[f64],
        delta_out: &mut [f64],
        beta_out: &mut [f64],
    ) -> Result<(), JsError> {
        self.try_eval_params_into(params, delta_out, beta_out)
            .map_err(js_err)
    }

    /// Evaluate at the current parameters into interleaved `out` (`2 * n_energy`).
    #[wasm_bindgen(js_name = evalInterleavedInto)]
    pub fn eval_interleaved_into(&mut self, out: &mut [f64]) -> Result<(), JsError> {
        self.try_eval_interleaved_into(out).map_err(js_err)
    }

    /// Set `params` and evaluate into interleaved `out` (`2 * n_energy`).
    #[wasm_bindgen(js_name = evalParamsInterleavedInto)]
    pub fn eval_params_interleaved_into(
        &mut self,
        params: &[f64],
        out: &mut [f64],
    ) -> Result<(), JsError> {
        self.try_eval_params_interleaved_into(params, out)
            .map_err(js_err)
    }

    /// Evaluate at the current parameters, returning interleaved `[delta_0, beta_0, ...]`.
    #[wasm_bindgen(js_name = evalInterleaved)]
    pub fn eval_interleaved(&mut self) -> Vec<f64> {
        self.inner.eval_into(&mut self.scratch);
        complex_to_interleaved(&self.scratch)
    }

    /// Evaluate only `delta` at the current parameters.
    #[wasm_bindgen(js_name = evalDelta)]
    pub fn eval_delta(&mut self) -> Vec<f64> {
        self.inner.eval_into(&mut self.scratch);
        self.scratch.iter().map(|z| z.re).collect()
    }

    /// Evaluate only `beta` at the current parameters.
    #[wasm_bindgen(js_name = evalBeta)]
    pub fn eval_beta(&mut self) -> Vec<f64> {
        self.inner.eval_into(&mut self.scratch);
        self.scratch.iter().map(|z| z.im).collect()
    }

    #[wasm_bindgen(js_name = toString)]
    pub fn to_string_repr(&self) -> String {
        format!(
            "Evaluator(n_energy={}, n_params={})",
            self.inner.energy().len(),
            self.inner.n_params()
        )
    }
}

/// Photon wavelength in Angstrom for energies in eV (`h*c / E`).
#[wasm_bindgen(js_name = wavelengthAngstrom)]
pub fn wavelength_angstrom(energy_ev: &[f64]) -> Vec<f64> {
    energy_ev.iter().map(|&e| HC_EV_ANGSTROM / e).collect()
}

fn try_mu_to_beta(
    energy_ev: &[f64],
    mu_cm2_per_g: &[f64],
    density_g_cm3: f64,
) -> Result<Vec<f64>, String> {
    if energy_ev.len() != mu_cm2_per_g.len() {
        return Err(format!(
            "energy_ev and mu_cm2_per_g length mismatch ({} vs {})",
            energy_ev.len(),
            mu_cm2_per_g.len()
        ));
    }
    let scale = density_g_cm3 * 1e-8 / (4.0 * PI);
    Ok(energy_ev
        .iter()
        .zip(mu_cm2_per_g)
        .map(|(&e, &mu)| mu * (HC_EV_ANGSTROM / e) * scale)
        .collect())
}

fn try_beta_to_mu(energy_ev: &[f64], beta: &[f64], density_g_cm3: f64) -> Result<Vec<f64>, String> {
    if energy_ev.len() != beta.len() {
        return Err(format!(
            "energy_ev and beta length mismatch ({} vs {})",
            energy_ev.len(),
            beta.len()
        ));
    }
    let scale = 4.0 * PI / (density_g_cm3 * 1e-8);
    Ok(energy_ev
        .iter()
        .zip(beta)
        .map(|(&e, &b)| b * scale / (HC_EV_ANGSTROM / e))
        .collect())
}

/// Convert a mass absorption coefficient (`cm^2/g`) to `beta`: `beta = mu * rho * lambda / (4 pi)`.
#[wasm_bindgen(js_name = muToBeta)]
pub fn mu_to_beta(
    energy_ev: &[f64],
    mu_cm2_per_g: &[f64],
    density_g_cm3: f64,
) -> Result<Vec<f64>, JsError> {
    try_mu_to_beta(energy_ev, mu_cm2_per_g, density_g_cm3).map_err(js_err)
}

/// Convert optical `beta` to a mass absorption coefficient (`cm^2/g`): `mu = 4 pi * beta / (rho * lambda)`.
#[wasm_bindgen(js_name = betaToMu)]
pub fn beta_to_mu(
    energy_ev: &[f64],
    beta: &[f64],
    density_g_cm3: f64,
) -> Result<Vec<f64>, JsError> {
    try_beta_to_mu(energy_ev, beta, density_g_cm3).map_err(js_err)
}

/// Dawson's function `D(x)` with asymptotic fast path for `|x| >= 10`.
#[wasm_bindgen(js_name = fastDawson)]
pub fn wasm_fast_dawson(x: f64) -> f64 {
    fast_dawson(x)
}

/// Error function `erf(x)` short-circuited to `+-1` for `|x| > 6`.
#[wasm_bindgen(js_name = fastErf)]
pub fn wasm_fast_erf(x: f64) -> f64 {
    fast_erf(x)
}

/// Faddeeva function `w(z)` for `Im z >= 0`, returning `[re, im]`.
#[wasm_bindgen(js_name = fastW)]
pub fn wasm_fast_w(re: f64, im: f64) -> Vec<f64> {
    let w = fast_w(Complex64::new(re, im));
    vec![w.re, w.im]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wasm_model_and_evaluator_roundtrip() {
        let energy: Vec<f64> = (0..401).map(|i| 920.0 + 0.1 * i as f64).collect();
        let mut m = WasmModel::new();
        m.try_add(Component::Gaussian {
            e0: 933.8,
            width: 0.49,
            amp: 4.1e-5,
        })
        .unwrap();
        m.try_add(Component::Voigt {
            e0: 938.3,
            sigma: 0.7,
            gamma: 0.2,
            amp: 5e-5,
        })
        .unwrap();
        m.try_add(Component::Step {
            e0: 939.4,
            width: 6.1,
            height: 7.2e-5,
            decay: 0.0,
        })
        .unwrap();
        m.try_continuum(3.6e-4, -3.7e-7).unwrap();
        m.try_delta_table(&[900.0, 950.0, 1000.0], &[-1e-4, 0.0, 1e-4])
            .unwrap();
        m.try_scale(1.1).unwrap();
        m.try_shift(0.05).unwrap();

        assert_eq!(m.n_components(), 3);
        assert_eq!(m.n_params(), 3 + 4 + 4 + 4);

        let mut delta = vec![0.0; energy.len()];
        let mut beta = vec![0.0; energy.len()];
        m.try_eval_into(&energy, &mut delta, &mut beta).unwrap();

        let interleaved = m.eval_interleaved(&energy);
        assert_eq!(interleaved.len(), 2 * energy.len());
        for i in 0..energy.len() {
            assert_eq!(interleaved[2 * i], delta[i]);
            assert_eq!(interleaved[2 * i + 1], beta[i]);
        }

        let mut ev = m.bind(&energy, true);
        let mut ev_delta = vec![0.0; energy.len()];
        let mut ev_beta = vec![0.0; energy.len()];
        ev.try_eval_into(&mut ev_delta, &mut ev_beta).unwrap();
        for i in 0..energy.len() {
            assert!((ev_delta[i] - delta[i]).abs() < 1e-18);
            assert!((ev_beta[i] - beta[i]).abs() < 1e-18);
        }

        let mut theta = ev.params();
        theta[0] += 0.2;
        ev.try_eval_params_into(&theta, &mut ev_delta, &mut ev_beta)
            .unwrap();
        let mut direct_delta = vec![0.0; energy.len()];
        let mut direct_beta = vec![0.0; energy.len()];
        m.try_eval_params_into(&energy, &theta, &mut direct_delta, &mut direct_beta)
            .unwrap();
        for i in 0..energy.len() {
            assert!((ev_delta[i] - direct_delta[i]).abs() < 1e-18);
            assert!((ev_beta[i] - direct_beta[i]).abs() < 1e-18);
        }
    }

    #[test]
    fn units_roundtrip() {
        let energy = [280.0, 530.0, 930.0];
        let mu = [1.2e4, 8.5e3, 2.1e4];
        let rho = 1.45;
        let beta = try_mu_to_beta(&energy, &mu, rho).unwrap();
        let mu_back = try_beta_to_mu(&energy, &beta, rho).unwrap();
        for (a, b) in mu.iter().zip(&mu_back) {
            assert!((a - b).abs() / a < 1e-15);
        }
    }

    #[test]
    fn delta_table_validation() {
        let mut m = WasmModel::new();
        assert!(m.try_delta_table(&[], &[]).is_err());
        assert!(m.try_delta_table(&[1.0, 2.0], &[1.0]).is_err());
        assert!(m.try_delta_table(&[2.0, 1.0], &[1.0, 2.0]).is_err());
        assert!(m.try_delta_table(&[1.0, f64::NAN], &[1.0, 2.0]).is_err());
    }
}
