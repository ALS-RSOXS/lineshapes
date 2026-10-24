use _core::Model;
use std::time::Instant;
fn bench(name: &str, m: &Model, e: &[f64]) {
    let mut out = m.eval(e);
    let reps = 3000;
    let t = Instant::now();
    for _ in 0..reps { m.eval_into(e, &mut out); }
    println!("{name:10} {:.2} ns/pt  ({:.1} us)", t.elapsed().as_nanos() as f64 / (e.len()*reps) as f64,
             t.elapsed().as_nanos() as f64 / reps as f64 / 1e3);
    std::hint::black_box(&out);
}
fn main() {
    let e: Vec<f64> = (0..4001).map(|i| 900.0 + i as f64 * 0.025).collect();
    bench("empty", &Model::new(), &e);
    bench("gaussian", &Model::new().gaussian(930.0, 1.0, 1.0), &e);
    bench("lorentz", &Model::new().lorentzian(930.0, 0.5, 1.0), &e);
    bench("voigt", &Model::new().voigt(930.0, 0.5, 0.3, 1.0), &e);
    bench("step", &Model::new().step(930.0, 6.0, 1.0, 0.0), &e);
    bench("table", &Model::new().delta_table(&e, &e), &e);
}
