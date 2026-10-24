//! The crate used standalone from Rust: public API only, no Python.

use lineshapes::{Complex64, Evaluator, Model};

fn energies() -> Vec<f64> {
    (0..2001).map(|i| 900.0 + 0.05 * i as f64).collect()
}

fn model() -> Model {
    Model::new()
        .gaussian(933.8, 0.49, 4.1e-5)
        .lorentzian(936.0, 0.3, 2e-5)
        .voigt(938.3, 0.7, 0.2, 5e-5)
        .pseudo_voigt(955.0, 1.5, 0.4, 2e-5)
        .step(939.4, 6.1, 7.2e-5, 0.0)
        .continuum(3.6e-4, -3.7e-7)
}

#[test]
fn eval_peak_height_and_sign() {
    let e = [930.0, 933.8, 937.0];
    let z = Model::new().gaussian(933.8, 0.49, 2.0).eval(&e);
    assert!((z[1].im - 2.0).abs() < 1e-12);
    assert!(z[0].re < 0.0 && z[2].re > 0.0);
}

#[test]
fn params_roundtrip_and_validation() {
    let mut m = model();
    let theta = m.params();
    assert_eq!(theta.len(), m.n_params());
    assert_eq!(m.param_names()[0], "gaussian[0].e0");
    m.set_params(&theta).unwrap();
    let mut bad = theta.clone();
    bad[1] = -1.0;
    assert!(m.set_params(&bad).is_err());
    assert!(
        m.try_add(lineshapes::Component::Gaussian {
            e0: 1.0,
            width: 0.0,
            amp: 1.0
        })
        .is_err()
    );
}

#[test]
fn evaluator_matches_model() {
    let e = energies();
    let mut m = model();
    let mut ev = Evaluator::new(m.clone(), &e, true);
    let mut theta = m.params();
    let mut out = vec![Complex64::new(0.0, 0.0); e.len()];
    for k in [0usize, 5, 9, 14] {
        theta[k] *= 1.001;
        ev.eval_params_into(&theta, &mut out).unwrap();
        m.set_params(&theta).unwrap();
        let direct = m.eval(&e);
        let err = out
            .iter()
            .zip(&direct)
            .map(|(a, b)| (a - b).norm())
            .fold(0.0, f64::max);
        assert!(err < 1e-13 * direct.iter().map(|z| z.norm()).fold(0.0, f64::max));
    }
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_equals_serial() {
    let e = energies();
    assert_eq!(model().eval(&e), model().eval_par(&e));
}
