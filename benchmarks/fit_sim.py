"""Simulated fit loop: how many peaks / energies before evaluation gets slow?

Each "iteration" mimics one optimizer step: perturb every Gaussian's (e0, width, amp)
and the global density scale, rebuild the model, evaluate on the grid.
Continuum (2 erf steps + linear) and a fixed delta table are always present (Cu-like).

    uv run python -m benchmarks.fit_sim            # table + figure
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from lineshapes import Model
from tests.reference import lineshapes_ref as ref

FIG = Path(__file__).parent / "figures"
RNG = np.random.default_rng(0)
CONT = dict(bkg=3.6e-4, slope=-3.7e-7,
            steps=[dict(e0=939.4, width=6.1, height=7.2e-5, decay=0.0),
                   dict(e0=960.7, width=2.2, height=5.1e-5, decay=0.0)])


def base_params(n_peaks):
    e0 = np.sort(RNG.uniform(925, 975, n_peaks))
    width = RNG.uniform(0.3, 3.0, n_peaks)
    amp = RNG.uniform(1e-5, 8e-5, n_peaks)
    return e0, width, amp


def perturb(e0, width, amp):
    return (e0 + RNG.normal(0, 0.05, e0.size),
            width * np.exp(RNG.normal(0, 0.02, e0.size)),
            amp * np.exp(RNG.normal(0, 0.02, e0.size)),
            float(np.exp(RNG.normal(0, 0.01))))


def rust_step(E, table, e0, width, amp, scale):
    m = Model()
    for a, b, c in zip(e0.tolist(), width.tolist(), amp.tolist()):
        m.gaussian(a, b, c)
    m.continuum(CONT["bkg"], CONT["slope"])
    for s in CONT["steps"]:
        m.step(s["e0"], s["width"], s["height"], s["decay"])
    return m.delta_table(*table).scale(scale)(E)


def ref_step(E, table, e0, width, amp, scale):
    peaks = [(ref.gaussian, dict(e0=a, width=b, amp=c)) for a, b, c in zip(e0, width, amp)]
    return ref.optical_model(E, peaks, continuum=CONT, delta_table=table, scale=scale)


def time_loop(step, E, table, params, budget_s=0.3, max_iter=2000):
    """Mean seconds per fit iteration (build + eval), fresh perturbation each time."""
    step(E, table, *perturb(*params))  # warm-up
    n, t0 = 0, time.perf_counter()
    while n < max_iter and time.perf_counter() - t0 < budget_s:
        step(E, table, *perturb(*params))
        n += 1
    return (time.perf_counter() - t0) / n


def main():
    peaks = [1, 5, 10, 20, 50, 100, 200, 500]
    grids = [500, 2000, 4001, 20000, 100000]
    rust = np.zeros((len(peaks), len(grids)))
    numpy_ = np.zeros_like(rust)
    for j, ne in enumerate(grids):
        E = np.linspace(900, 1000, ne)
        table = (E, np.zeros_like(E))
        for i, npk in enumerate(peaks):
            params = base_params(npk)
            rust[i, j] = time_loop(rust_step, E, table, params)
            numpy_[i, j] = time_loop(ref_step, E, table, params, budget_s=0.2, max_iter=200)
    print("seconds per fit iteration (build + eval); rust | speedup vs numpy")
    print(f"{'peaks':>6s}" + "".join(f"{g:>18d}" for g in grids))
    for i, npk in enumerate(peaks):
        row = "".join(f"{rust[i, j] * 1e6:10.0f}us {numpy_[i, j] / rust[i, j]:5.1f}x" for j in range(len(grids)))
        print(f"{npk:6d}{row}")
    ns = rust / (np.array(peaks)[:, None] * np.array(grids)[None, :])
    print(f"\nrust cost per (peak x energy): {ns.min() * 1e9:.2f} - {ns.max() * 1e9:.2f} ns")
    plot(peaks, grids, rust, numpy_)


def plot(peaks, grids, rust, numpy_):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for j, ne in enumerate(grids):
        (l,) = axes[0].loglog(peaks, rust[:, j] * 1e3, "o-", label=f"{ne} energies")
        axes[0].loglog(peaks, numpy_[:, j] * 1e3, "--", color=l.get_color(), alpha=0.5)
        axes[1].semilogx(peaks, numpy_[:, j] / rust[:, j], "o-", label=f"{ne} energies")
    for y, txt in [(1.0, "1 ms (1k evals/s)"), (10.0, "10 ms")]:
        axes[0].axhline(y, color="k", lw=0.6, ls=":")
        axes[0].text(peaks[0], y * 1.1, txt, fontsize=8)
    axes[0].set(xlabel="number of Gaussian peaks", ylabel="ms per fit iteration",
                title="Fit iteration: rust (solid) vs numpy (dashed)")
    axes[1].set(xlabel="number of Gaussian peaks", ylabel="speedup (numpy / rust)", title="Speedup")
    axes[1].axhline(1, color="k", lw=0.6)
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fit_sim.png", dpi=120)
    print("wrote", FIG / "fit_sim.png")


if __name__ == "__main__":
    main()
