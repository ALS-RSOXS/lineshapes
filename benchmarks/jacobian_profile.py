"""Profile Jacobian strategies for a Gaussian-peak model (+ continuum + table + scale).

Parameters theta = [e0_k, width_k, amp_k for each peak] + [scale]; output J = d(delta + i beta)/d theta,
shape (n_energy, 3P + 1), complex. Strategies:

  A  naive forward FD: rebuild + evaluate the whole model per column        O(P^2 N)
  B  incremental FD: only the perturbed peak is re-evaluated per column       O(P N)
  C  analytic (numpy prototype): closed-form derivatives, one pass per peak   O(P N)

    uv run python -m benchmarks.jacobian_profile
"""

from __future__ import annotations

import time

import numpy as np
from scipy.special import dawsn

from lineshapes_rs import Model

RNG = np.random.default_rng(1)
STEPS = [(939.4, 6.1, 7.2e-5, 0.0), (960.7, 2.2, 5.1e-5, 0.0)]
TWO_SQRTPI = 2.0 / np.sqrt(np.pi)


def build(e0, w, a, scale, table):
    m = Model()
    for x, y, z in zip(e0.tolist(), w.tolist(), a.tolist()):
        m.gaussian(x, y, z)
    m.continuum(3.6e-4, -3.7e-7)
    for s in STEPS:
        m.step(*s)
    return m.delta_table(*table).scale(scale)


def jac_naive_fd(E, table, e0, w, a, s, rel=1e-7):
    f0 = build(e0, w, a, s, table)(E)
    P = e0.size
    J = np.empty((E.size, 3 * P + 1), complex)
    for k in range(P):
        for j, arr in enumerate((e0, w, a)):
            p = arr.copy()
            h = rel * max(abs(arr[k]), 1e-3)
            p[k] += h
            args = [e0, w, a]
            args[j] = p
            J[:, 3 * k + j] = (build(*args, s, table)(E) - f0) / h
    J[:, -1] = f0 / s
    return J


def jac_incremental_fd(E, table, e0, w, a, s, rel=1e-7):
    f0 = build(e0, w, a, s, table)(E)
    P = e0.size
    J = np.empty((E.size, 3 * P + 1), complex)
    for k in range(P):
        base = Model().gaussian(e0[k], w[k], a[k])(E)
        for j, (x, y, z) in enumerate(
            ((e0[k] + (h0 := rel * abs(e0[k])), w[k], a[k]),
             (e0[k], w[k] + (h1 := rel * w[k]), a[k]),
             (e0[k], w[k], a[k] + (h2 := rel * a[k]))),
        ):
            J[:, 3 * k + j] = s * (Model().gaussian(x, y, z)(E) - base) / (h0, h1, h2)[j]
    J[:, -1] = f0 / s
    return J


def jac_analytic(E, table, e0, w, a, s):
    """f_k = a [ (2/sqrt(pi)) D(x) + i exp(-x^2) ],  x = (E - e0)/w.
    df/dx = a [ (2/sqrt(pi)) (1 - 2x D(x)) - 2 i x exp(-x^2) ];  dx/de0 = -1/w;  dx/dw = -x/w."""
    f0 = build(e0, w, a, s, table)(E)
    P = e0.size
    J = np.empty((E.size, 3 * P + 1), complex)
    x = (E[:, None] - e0[None, :]) / w[None, :]
    D = dawsn(x)
    g = np.exp(-x * x)
    shape = TWO_SQRTPI * D + 1j * g
    dfdx = a * (TWO_SQRTPI * (1.0 - 2.0 * x * D) - 2j * x * g)
    J[:, 0:3 * P:3] = s * dfdx * (-1.0 / w)
    J[:, 1:3 * P:3] = s * dfdx * (-x / w)
    J[:, 2:3 * P:3] = s * shape
    J[:, -1] = f0 / s
    return J


def best(f, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def main():
    print(f"{'P':>4s} {'N':>6s} {'eval':>9s} {'A naive FD':>12s} {'B incr FD':>11s} {'C analytic':>11s}"
          f" {'A/eval':>7s} {'C/eval':>7s}   max|C-B|/max|C|")
    for N in (4001, 20000):
        E = np.linspace(900, 1000, N)
        table = (E, 1e-4 * np.cos(E / 30))
        for P in (5, 20, 50, 100):
            e0 = np.sort(RNG.uniform(925, 975, P))
            w = RNG.uniform(0.3, 3.0, P)
            a = RNG.uniform(1e-5, 8e-5, P)
            s = 1.1
            m = build(e0, w, a, s, table)
            t_eval = best(lambda: m(E), 20)
            tA = best(lambda: jac_naive_fd(E, table, e0, w, a, s), 1 if P >= 50 else 3)
            tB = best(lambda: jac_incremental_fd(E, table, e0, w, a, s))
            tC = best(lambda: jac_analytic(E, table, e0, w, a, s))
            JB = jac_incremental_fd(E, table, e0, w, a, s)
            JC = jac_analytic(E, table, e0, w, a, s)
            err = np.max(np.abs(JC - JB)) / np.max(np.abs(JC))
            print(f"{P:4d} {N:6d} {t_eval * 1e3:7.3f}ms {tA * 1e3:10.1f}ms {tB * 1e3:9.2f}ms {tC * 1e3:9.2f}ms"
                  f" {tA / t_eval:7.0f} {tC / t_eval:7.1f}   {err:.1e}")


if __name__ == "__main__":
    main()
