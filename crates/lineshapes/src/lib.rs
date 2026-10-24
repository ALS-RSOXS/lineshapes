//! Fast composable optical-constant models: analytic KK-paired lineshape peaks
//! (Gaussian, Lorentzian, Voigt, pseudo-Voigt) plus a continuum in beta and a fixed
//! continuum delta table. See [`Model`].

mod bound;
mod kernels;
pub mod model;
mod special;

pub use bound::Evaluator;
pub use model::{Component, Model};
pub use num_complex::Complex64;
pub use special::{fast_dawson, fast_erf, fast_w};
