"""Fitting without rebuilding: rebuild vs params vector vs bound Evaluator (cached).

Two workloads on a Gaussian-peak model (+ 2 steps + continuum + delta table + scale):
  * fit iteration: every parameter changes (what a full optimizer step does)
  * forward-FD Jacobian: one parameter changes per column (3P + 4 columns)

    uv run python -m benchmarks.no_rebuild
"""

from __future__ import annotations

import time

import numpy as np

from lineshapes_rs import Model

RNG = np.random.default_rng(2)


def build(E, e0, w, a):
    m = Model()
    for x, y, z in zip(e0.tolist(), w.tolist(), a.tolist()):
        m.gaussian(x, y, z)
    m.step(939.4, 6.1, 7.2e-5).step(960.7, 2.2, 5.1e-5).continuum(3.6e-4, -3.7e-7)
    return m.delta_table(E, 1e-4 * np.cos(E / 30)).scale(1.1)


def best(f, reps=5):
    f()
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        out.append(time.perf_counter() - t0)
    return min(out)


def main():
    print(f"{'P':>4s} {'N':>6s} | {'fit iteration (all params change)':^38s} | {'FD Jacobian (one param per column)':^44s}")
    print(f"{'':>11s} | {'rebuild':>9s} {'m(E,params)':>12s} {'ev(theta)':>10s}   | "
          f"{'rebuild':>9s} {'m(E,params)':>12s} {'ev cached':>10s} {'speedup':>8s}")
    for N in (4001, 20000):
        E = np.linspace(900, 1000, N)
        for P in (5, 20, 50, 100):
            e0 = np.sort(RNG.uniform(925, 975, P))
            w = RNG.uniform(0.3, 3.0, P)
            a = RNG.uniform(1e-5, 8e-5, P)
            m = build(E, e0, w, a)
            theta0 = m.params
            ev, ev_nc = m.bind(E), m.bind(E, cache=False)
            out = np.empty(N, complex)
            pert = lambda: theta0 * (1 + 1e-4 * RNG.standard_normal(theta0.size))  # noqa: E731

            def it_rebuild():
                t = pert()
                build(E, t[0:3 * P:3], t[1:3 * P:3], t[2:3 * P:3])(E)

            t_it = (best(it_rebuild), best(lambda: m(E, params=pert())), best(lambda: ev_nc(pert(), out=out)))

            h = 1e-7 * np.maximum(np.abs(theta0), 1e-3)
            cols = range(theta0.size)

            def fd_rebuild():
                f0 = build(E, e0, w, a)(E)
                for k in cols:
                    t = theta0.copy()
                    t[k] += h[k]
                    (build(E, t[0:3 * P:3], t[1:3 * P:3], t[2:3 * P:3])(E) - f0) / h[k]

            def fd_params():
                f0 = m(E, params=theta0)
                for k in cols:
                    t = theta0.copy()
                    t[k] += h[k]
                    (m(E, params=t) - f0) / h[k]

            def fd_cached():
                f0 = ev(theta0).copy()
                t = theta0.copy()
                for k in cols:
                    t[k] += h[k]
                    (ev(t, out=out) - f0) / h[k]
                    t[k] = theta0[k]

            reps = 1 if P >= 50 else 3
            t_fd = (best(fd_rebuild, reps), best(fd_params, reps), best(fd_cached, reps))
            print(f"{P:4d} {N:6d} | " + " ".join(f"{x * 1e3:9.3f}ms"[:11].rjust(11) for x in t_it)
                  + "   | " + " ".join(f"{x * 1e3:9.1f}ms".rjust(11) for x in t_fd)
                  + f" {t_fd[0] / t_fd[2]:7.0f}x")


if __name__ == "__main__":
    main()
