//! Special functions with fast paths, and linear interpolation helpers.

use errorfunctions::{ComplexErrorFunctions, RealErrorFunctions};
use num_complex::Complex64;

const FRAC_1_SQRT_PI: f64 = 0.5 * std::f64::consts::FRAC_2_SQRT_PI;
/// |x| at and beyond which the asymptotic Dawson / Faddeeva series are used.
pub(crate) const ASYMP_X: f64 = 10.0;
/// |x| beyond which erf(x) is exactly +-1 in double precision.
pub(crate) const ERF_SAT: f64 = 6.0;

/// Asymptotic coefficients (2n-1)!!/2^n of Dawson's function:
/// `D(x) = 1/(2x) * sum_n c_n / x^(2n)`; 13 terms give < 1e-16 relative error for |x| >= 10.
const DAWSON_ASYMP: [f64; 13] = [
    1.0, 0.5, 0.75, 1.875, 6.5625, 29.53125, 162.421875, 1055.7421875, 7918.06640625,
    67303.564453125, 639383.8623046875, 6713530.554199219, 77205601.37329102,
];

/// `sum_{n=0}^{12} DAWSON_ASYMP[n] * u^n` by Estrin's scheme (shallow dependency tree
/// instead of Horner's 12-deep chain; measured faster for complex `u`, slower for real). `lin(a, b, t) = a + b*t` with real coefficients.
#[inline(always)]
fn estrin12<T: Copy>(u: T, lin: impl Fn(f64, f64, T) -> T, mul: impl Fn(T, T) -> T) -> T
where
    T: std::ops::Add<Output = T>,
{
    let c = &DAWSON_ASYMP;
    let u2 = mul(u, u);
    let u4 = mul(u2, u2);
    let u8 = mul(u4, u4);
    // pairs c[k] + c[k+1] u
    let p01 = lin(c[0], c[1], u);
    let p23 = lin(c[2], c[3], u);
    let p45 = lin(c[4], c[5], u);
    let p67 = lin(c[6], c[7], u);
    let p89 = lin(c[8], c[9], u);
    let p1011 = lin(c[10], c[11], u);
    let q0 = p01 + mul(p23, u2);
    let q1 = p45 + mul(p67, u2);
    let q2 = p89 + mul(p1011, u2);
    let lo = q0 + mul(q1, u4);
    // degree 8..12 part: (c8 + c9 u + c10 u^2 + c11 u^3) + c12 u^4, times u^8
    lo + mul(q2 + lin(0.0, c[12], u4), u8)
}

/// Dawson's function: asymptotic series for |x| >= 10, Faddeeva-package table otherwise.
#[inline]
pub fn fast_dawson(x: f64) -> f64 {
    if x.abs() >= ASYMP_X { dawson_asymp(x) } else { x.dawson() }
}

/// Asymptotic Dawson series; valid (< 1e-16 relative) for |x| >= 10.
#[inline(always)]
pub(crate) fn dawson_asymp(x: f64) -> f64 {
    {
        let r = 1.0 / x;
        let u = r * r;
        let mut p = DAWSON_ASYMP[12];
        for &c in DAWSON_ASYMP[..12].iter().rev() {
            p = p.mul_add(u, c);
        }
        0.5 * p * r
    }
}

/// Faddeeva `w(z)` for `Im z >= 0`: asymptotic series `i/(sqrt(pi) z) sum_n c_n / z^(2n)`
/// for |z| >= 10 (truncation < 1e-16 relative; the dropped `exp(-z^2)` term is < 1e-43
/// there), Faddeeva-package algorithm otherwise.
#[inline]
pub fn fast_w(z: Complex64) -> Complex64 {
    if z.norm_sqr() >= ASYMP_X * ASYMP_X { w_asymp(z) } else { z.w() }
}

/// Asymptotic Faddeeva series; valid for |z| >= 10, Im z >= 0.
#[inline(always)]
pub(crate) fn w_asymp(z: Complex64) -> Complex64 {
    {
        let r = z.inv();
        let u = r * r;
        let p = estrin12(u, |a, b, t| a + t * b, |a, b| a * b) * r;
        Complex64::new(-FRAC_1_SQRT_PI * p.im, FRAC_1_SQRT_PI * p.re) // i/sqrt(pi) * p
    }
}

/// erf, short-circuited where it is exactly +-1 in double precision (|x| > 6).
#[inline]
pub fn fast_erf(x: f64) -> f64 {
    if x > ERF_SAT {
        1.0
    } else if x < -ERF_SAT {
        -1.0
    } else {
        RealErrorFunctions::erf(x)
    }
}

/// Non-short-circuiting monotonicity check (vectorizes; ~0.1 ns/point).
#[inline]
pub(crate) fn is_sorted(x: &[f64]) -> bool {
    x.windows(2).fold(true, |acc, w| acc & (w[0] <= w[1]))
}

/// Linear interpolation with `xs[i-1] <= x < xs[i]` already located (clamped at the ends).
#[inline]
pub(crate) fn interp_at(xs: &[f64], ys: &[f64], x: f64, i: usize) -> f64 {
    let n = xs.len();
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[n - 1] {
        return ys[n - 1];
    }
    let (x0, x1, y0, y1) = (xs[i - 1], xs[i], ys[i - 1], ys[i]);
    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
}

/// `numpy.interp` semantics: linear, clamped to the end values.
#[inline]
pub(crate) fn interp(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    let n = xs.len();
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[n - 1] {
        return ys[n - 1];
    }
    let i = xs.partition_point(|&v| v <= x); // xs[i-1] <= x < xs[i]
    let (x0, x1, y0, y1) = (xs[i - 1], xs[i], ys[i - 1], ys[i]);
    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
}

