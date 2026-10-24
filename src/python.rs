//! Python bindings: `Model` (chainable builder, callable) and `Evaluator` (model bound to a grid).

use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadwriteArray1};
use num_complex::Complex64;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::bound::Evaluator;
use crate::model::{Component, Model};

fn err(e: String) -> PyErr {
    PyValueError::new_err(e)
}

/// Borrow a 1-D float64 array as a slice, copying only if it is not contiguous.
fn with_slice<R>(a: &PyReadonlyArray1<'_, f64>, f: impl FnOnce(&[f64]) -> R) -> R {
    match a.as_slice() {
        Ok(s) => f(s),
        Err(_) => f(&a.as_array().to_vec()),
    }
}

/// Composable optical-constant model returning ``delta + 1j*beta``.
///
/// Build by chaining (each method mutates and returns the model), then call it::
///
///     m = (Model()
///          .gaussian(e0=933.8, width=0.49, amp=4.1e-5)
///          .continuum(bkg=3.6e-4, slope=-3.7e-7)
///          .step(e0=939.4, width=6.1, height=7.2e-5)
///          .delta_table(energy, delta_bg))
///     n = m(energy)                 # complex128, single core
///     n = m(energy, parallel=True)  # all cores
///
/// For fitting, use the flat parameter vector instead of rebuilding::
///
///     m.param_names                  # ['gaussian[0].e0', ..., 'scale', 'shift']
///     n = m(energy, params=theta)    # evaluate at theta (model unchanged)
///     ev = m.bind(energy)            # fixed grid + per-component cache
///     n = ev(theta)                  # recomputes only components whose params changed
#[pyclass(name = "Model", module = "lineshapes_rs._core", skip_from_py_object)]
#[derive(Clone)]
pub struct PyModel {
    inner: Model,
}

impl PyModel {
    fn add(mut slf: PyRefMut<'_, Self>, c: Component) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.try_add(c).map_err(err)?;
        Ok(slf)
    }
}

#[pymethods]
impl PyModel {
    #[new]
    fn new() -> Self {
        Self { inner: Model::new() }
    }

    /// Gaussian peak; ``width`` is the 1/e half-width (sigma = width / sqrt(2)).
    fn gaussian(slf: PyRefMut<'_, Self>, e0: f64, width: f64, amp: f64) -> PyResult<PyRefMut<'_, Self>> {
        Self::add(slf, Component::Gaussian { e0, width, amp })
    }

    /// Lorentzian peak; ``gamma`` is the half width at half maximum.
    fn lorentzian(slf: PyRefMut<'_, Self>, e0: f64, gamma: f64, amp: f64) -> PyResult<PyRefMut<'_, Self>> {
        Self::add(slf, Component::Lorentzian { e0, gamma, amp })
    }

    /// Exact Voigt; ``sigma`` Gaussian std, ``gamma`` Lorentzian HWHM.
    fn voigt(slf: PyRefMut<'_, Self>, e0: f64, sigma: f64, gamma: f64, amp: f64) -> PyResult<PyRefMut<'_, Self>> {
        Self::add(slf, Component::Voigt { e0, sigma, gamma, amp })
    }

    /// Pseudo-Voigt ``eta*L + (1-eta)*G`` with shared ``fwhm``.
    fn pseudo_voigt(slf: PyRefMut<'_, Self>, e0: f64, fwhm: f64, eta: f64, amp: f64) -> PyResult<PyRefMut<'_, Self>> {
        Self::add(slf, Component::PseudoVoigt { e0, fwhm, eta, amp })
    }

    /// Erf ionization step in beta, with optional linear post-edge ``decay``.
    #[pyo3(signature = (e0, width, height, decay=0.0))]
    fn step(slf: PyRefMut<'_, Self>, e0: f64, width: f64, height: f64, decay: f64) -> PyResult<PyRefMut<'_, Self>> {
        Self::add(slf, Component::Step { e0, width, height, decay })
    }

    /// Linear continuum in beta: ``bkg + slope*E``.
    #[pyo3(signature = (bkg=0.0, slope=0.0))]
    fn continuum(mut slf: PyRefMut<'_, Self>, bkg: f64, slope: f64) -> PyRefMut<'_, Self> {
        slf.inner.set_continuum(bkg, slope);
        slf
    }

    /// Fixed continuum delta table (e.g. from kkcalc), interpolated like ``numpy.interp``.
    fn delta_table<'py>(
        mut slf: PyRefMut<'py, Self>,
        energy: PyReadonlyArray1<'py, f64>,
        delta: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let e = with_slice(&energy, <[f64]>::to_vec);
        let d = with_slice(&delta, <[f64]>::to_vec);
        if e.len() != d.len() || e.is_empty() {
            return Err(err("delta table energy/delta must be non-empty and equal length".into()));
        }
        if !e.windows(2).all(|w| w[0] < w[1]) {
            return Err(err("delta table energies must be strictly increasing".into()));
        }
        slf.inner.set_delta_table(&e, &d);
        Ok(slf)
    }

    /// Overall multiplicative scale.
    fn scale(mut slf: PyRefMut<'_, Self>, scale: f64) -> PyRefMut<'_, Self> {
        slf.inner.set_scale(scale);
        slf
    }

    /// Energy offset: the model is evaluated at ``E + shift``.
    fn shift(mut slf: PyRefMut<'_, Self>, shift: f64) -> PyRefMut<'_, Self> {
        slf.inner.set_shift(shift);
        slf
    }

    /// Independent copy (so a base model can be extended in several ways).
    fn copy(&self) -> Self {
        self.clone()
    }

    #[getter]
    fn n_components(&self) -> usize {
        self.inner.n_components()
    }

    #[getter]
    fn n_params(&self) -> usize {
        self.inner.n_params()
    }

    /// Current flat parameter vector (components in insertion order, then
    /// ``continuum.bkg, continuum.slope, scale, shift``).
    #[getter]
    fn params<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.params().into_pyarray(py)
    }

    /// Names matching ``params``, e.g. ``'gaussian[0].e0'``, ``'step[1].height'``, ``'scale'``.
    #[getter]
    fn param_names(&self) -> Vec<String> {
        self.inner.param_names()
    }

    /// Set all parameters from a flat vector (validated; model unchanged on error).
    fn set_params<'py>(mut slf: PyRefMut<'py, Self>, params: PyReadonlyArray1<'py, f64>) -> PyResult<PyRefMut<'py, Self>> {
        with_slice(&params, |p| slf.inner.set_params(p)).map_err(err)?;
        Ok(slf)
    }

    /// Bind to a fixed energy grid for repeated evaluation (see ``Evaluator``).
    /// ``cache=True`` keeps each component's contribution (n_components * len(energy) * 16 bytes)
    /// so only components whose parameters change are recomputed.
    #[pyo3(signature = (energy, cache=true))]
    fn bind(&self, energy: PyReadonlyArray1<'_, f64>, cache: bool) -> PyEvaluator {
        let e = with_slice(&energy, <[f64]>::to_vec);
        PyEvaluator { inner: Evaluator::new(self.inner.clone(), &e, cache) }
    }

    /// Evaluate ``delta + 1j*beta`` on ``energy`` (1-D float64). GIL released.
    /// ``params``: evaluate at this parameter vector without modifying the model.
    #[pyo3(signature = (energy, params=None, parallel=false))]
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        energy: PyReadonlyArray1<'py, f64>,
        params: Option<PyReadonlyArray1<'py, f64>>,
        parallel: bool,
    ) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
        let tmp;
        let model = match params {
            None => &self.inner,
            Some(p) => {
                let mut m = self.inner.clone();
                with_slice(&p, |p| m.set_params(p)).map_err(err)?;
                tmp = m;
                &tmp
            }
        };
        let out = with_slice(&energy, |e| {
            py.detach(|| {
                if parallel {
                    #[cfg(feature = "parallel")]
                    {
                        return model.eval_par(e);
                    }
                }
                model.eval(e)
            })
        });
        Ok(out.into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        format!("Model(n_components={}, n_params={})", self.inner.n_components(), self.inner.n_params())
    }
}

/// A ``Model`` bound to a fixed energy grid (from ``Model.bind``), for fitting loops.
///
///     ev = m.bind(energy)
///     n = ev(theta)            # set params + evaluate; only changed components recomputed
///     ev(theta, out=buf)       # write into a preallocated complex128 array (no allocation)
///
/// ``scale``, ``continuum.*`` never trigger component work; changing ``shift`` recomputes all.
#[pyclass(name = "Evaluator", module = "lineshapes_rs._core")]
pub struct PyEvaluator {
    inner: Evaluator,
}

#[pymethods]
impl PyEvaluator {
    #[getter]
    fn n_params(&self) -> usize {
        self.inner.n_params()
    }

    #[getter]
    fn params<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.params().into_pyarray(py)
    }

    #[getter]
    fn param_names(&self) -> Vec<String> {
        self.inner.param_names()
    }

    #[getter]
    fn energy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.energy().to_vec().into_pyarray(py)
    }

    /// Copy of the bound model at the current parameters.
    #[getter]
    fn model(&self) -> PyModel {
        PyModel { inner: self.inner.model().clone() }
    }

    fn set_params(&mut self, params: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        with_slice(&params, |p| self.inner.set_params(p)).map_err(err)
    }

    /// Evaluate at ``params`` (or the current parameters). With ``out``, write into that
    /// contiguous complex128 array of the grid length and return it.
    #[pyo3(signature = (params=None, out=None))]
    fn __call__<'py>(
        &mut self,
        py: Python<'py>,
        params: Option<PyReadonlyArray1<'py, f64>>,
        out: Option<Bound<'py, PyArray1<Complex64>>>,
    ) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
        if let Some(p) = params {
            with_slice(&p, |p| self.inner.set_params(p)).map_err(err)?;
        }
        let n = self.inner.energy().len();
        let arr = match out {
            Some(a) => a,
            None => unsafe { PyArray1::<Complex64>::new(py, n, false) },
        };
        {
            let mut rw: PyReadwriteArray1<'_, Complex64> = arr.try_readwrite().map_err(|e| err(e.to_string()))?;
            let buf = rw.as_slice_mut().map_err(|_| err("out must be a contiguous complex128 array".into()))?;
            if buf.len() != n {
                return Err(err(format!("out has length {}, expected {n}", buf.len())));
            }
            let ev = &mut self.inner;
            py.detach(|| ev.eval_into(buf));
        }
        Ok(arr)
    }

    fn __repr__(&self) -> String {
        format!("Evaluator(n_energy={}, n_params={})", self.inner.energy().len(), self.inner.n_params())
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyModel>()?;
    m.add_class::<PyEvaluator>()?;
    Ok(())
}
