"""Shared helpers for the example scripts: plot style, the peak-fitting recipe, figure output.

The fitting recipe is deliberately small: it is the pattern the docs teach.

1. Start from the material's fixed continuum (`datasets.continuum_model`).
2. Add one Gaussian per seed (DFT cluster).
3. Solve the peak heights linearly (non-negative least squares) for a good starting point.
4. Refine every peak's position, width and height plus the global density ``scale`` with
   ``scipy.optimize.least_squares``, evaluating through a bound `Evaluator` so the finite-difference
   Jacobian recomputes only the peak whose parameter changed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from scipy.optimize import least_squares, nnls

from lineshapes import Model, datasets

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical slots 1-3 (blue, orange, aqua) and text/surface tokens, light and dark.
STYLE = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e6e5e1", series=("#2a78d6", "#eb6834", "#1baf7a")),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", grid="#34342f", series=("#3987e5", "#d95926", "#199e70")),
}
#: One series color per entity, fixed across every figure: model, continuum, starting guess.
MODEL, CONTINUUM, SEED = 0, 1, 2


def apply_style(mode: str) -> dict:
    s = STYLE[mode]
    plt.rcParams.update(
        {
            "figure.facecolor": s["surface"], "axes.facecolor": s["surface"], "savefig.facecolor": s["surface"],
            "text.color": s["ink"], "axes.labelcolor": s["ink2"], "xtick.color": s["ink2"], "ytick.color": s["ink2"],
            "axes.edgecolor": s["grid"], "axes.grid": True, "grid.color": s["grid"], "grid.linewidth": 0.6,
            "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
            "legend.frameon": False, "legend.fontsize": 9, "font.size": 10, "axes.titlesize": 11, "axes.titlelocation": "left",
            "axes.formatter.use_mathtext": True,
        }
    )
    return s


def sci(ax) -> None:
    """Scientific-notation y axis."""
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)


def fit_figure(nrows_ratio=(4, 1), figsize=(9, 5.2)):
    """Main panel above a short residual panel sharing x."""
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw=dict(height_ratios=nrows_ratio, hspace=0.06))
    return fig, axes


def save(fig, out: Path, name: str, mode: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{name}-{mode}.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return p


# ------------------------------------------------------------------------------------ fitting
@dataclass
class PeakFit:
    model: Model  # fitted model (continuum + peaks)
    seed_model: Model  # starting guess before refinement
    names: list[str]
    theta: np.ndarray
    n_peaks: int
    n_evals: int
    seconds: float
    rms: float  # rms beta residual
    rel_rms: float  # rms residual / mean beta


# --8<-- [start:fit]
def fit_peaks(energy, beta, material: str, seeds: dict, *, max_nfev=150) -> PeakFit:
    """Fit Gaussian peaks (plus the global scale) to ``beta`` on top of the fixed continuum."""
    e0s = np.asarray(seeds["energy_ev"])
    sigmas = np.asarray(seeds["sigma_ev"])
    widths = np.sqrt(2.0) * sigmas  # 1/e half-width of a Gaussian with std sigma

    cont = datasets.continuum_model(material)
    y = beta - cont(energy).imag  # what the peaks must explain
    basis = np.column_stack([Model().gaussian(e, w, 1.0)(energy).imag for e, w in zip(e0s, widths)])
    amp0, _ = nnls(basis, y)
    keep = amp0 > 1e-6 * amp0.max()  # a seed the data does not need stays out of the fit
    e0s, sigmas, widths, amp0 = e0s[keep], sigmas[keep], widths[keep], amp0[keep]

    model = datasets.continuum_model(material)
    for e, w, a in zip(e0s, widths, amp0):
        model.gaussian(float(e), float(w), float(a))
    seed_model = model.copy()

    names = model.param_names
    theta0 = model.params.copy()
    free = np.array([n.startswith("gaussian") or n == "scale" for n in names])
    lo, hi = np.full(theta0.size, -np.inf), np.full(theta0.size, np.inf)
    for i, n in enumerate(names):
        if n.endswith(".e0") and n.startswith("gaussian"):
            k = int(n.split("[")[1].split("]")[0])
            span = max(1.0, 1.5 * sigmas[k])
            lo[i], hi[i] = theta0[i] - span, theta0[i] + span
        elif n.endswith(".width"):
            lo[i], hi[i] = 0.1, 15.0
        elif n.endswith(".amp"):
            lo[i], hi[i] = 0.0, np.inf
        elif n == "scale":
            lo[i], hi[i] = 0.5, 1.5

    ev = model.bind(energy)
    buf = np.empty(energy.size, dtype=complex)
    norm = float(np.max(beta))
    theta = theta0.copy()

    def residual(p):
        theta[free] = p
        return (ev(theta, out=buf).imag - beta) / norm

    t0 = time.perf_counter()
    sol = least_squares(residual, theta0[free], bounds=(lo[free], hi[free]), x_scale="jac", max_nfev=max_nfev, ftol=1e-10, xtol=1e-10)
    seconds = time.perf_counter() - t0
    theta[free] = sol.x
    model.set_params(theta)
    r = model(energy).imag - beta
    return PeakFit(
        model=model, seed_model=seed_model, names=names, theta=theta.copy(), n_peaks=len(e0s), n_evals=sol.nfev,
        seconds=seconds, rms=float(np.sqrt(np.mean(r**2))), rel_rms=float(np.sqrt(np.mean(r**2)) / np.mean(beta)),
    )
# --8<-- [end:fit]


def peak_table_markdown(fit: PeakFit) -> str:
    """Markdown table of the fitted Gaussian peaks, sorted by energy."""
    rows = []
    for k in range(fit.n_peaks):
        e0, w, a = (fit.theta[fit.names.index(f"gaussian[{k}].{p}")] for p in ("e0", "width", "amp"))
        rows.append((e0, w, a))
    rows.sort()
    lines = ["| # | E0 (eV) | 1/e half-width (eV) | height (beta) |", "|--:|--:|--:|--:|"]
    lines += [f"| {i + 1} | {e:.2f} | {w:.2f} | {a:.2e} |" for i, (e, w, a) in enumerate(rows)]
    scale = fit.theta[fit.names.index("scale")]
    lines.append(f"\nFitted density scale: **{scale:.3f}** (the tables are per unit density).")
    return "\n".join(lines) + "\n"


def peak_contributions(fit: PeakFit, energy):
    """Per-peak complex contributions (delta + i beta) on ``energy``."""
    out = []
    scale = fit.theta[fit.names.index("scale")]
    for k in range(fit.n_peaks):
        e0, w, a = (fit.theta[fit.names.index(f"gaussian[{k}].{p}")] for p in ("e0", "width", "amp"))
        out.append(scale * Model().gaussian(e0, w, a)(energy))
    return out
