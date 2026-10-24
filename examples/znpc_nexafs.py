"""ZnPc carbon K-edge NEXAFS: fit Gaussian peaks on a fixed continuum, get delta analytically.

    uv run python examples/znpc_nexafs.py [output_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import _common as c  # noqa: E402

from lineshapes import datasets, mu_to_beta  # noqa: E402

MAGIC_ANGLE = 54.7  # orientation-independent spectrum
MATERIAL = "znpc"


def spectrum(theta_deg: float):
    """Energy and beta (unit density) of the NEXAFS taken at ``theta_deg``."""
    t = datasets.load_table("znpc_nexafs")
    m = np.isclose(t["theta_deg"], theta_deg)
    return t["energy_ev"][m], mu_to_beta(t["energy_ev"][m], t["mu_cm2_per_g"][m])


def figures(fit, E, beta, out: Path, mode: str) -> None:
    s = c.apply_style(mode)
    blue, orange, aqua = s["series"]
    total = fit.model(E)
    seed = fit.seed_model(E)
    cont = datasets.continuum_model(MATERIAL)(E)

    # 1. fit and residual
    fig, (ax, res) = c.fit_figure()
    ax.plot(E, beta, ".", ms=2.5, color=s["ink2"], label="data (magic angle)", zorder=1)
    ax.plot(E, seed.imag, "--", color=aqua, lw=1.2, label="starting guess (DFT seeds)")
    ax.plot(E, cont.imag, color=orange, label="fixed continuum")
    ax.plot(E, total.imag, color=blue, label="fit")
    ax.set(ylabel=r"$\beta$", title="ZnPc C K-edge: beta fit")
    ax.legend(loc="upper right")
    c.sci(ax)
    res.axhline(0, color=s["ink2"], lw=0.6)
    res.plot(E, total.imag - beta, color=blue, lw=1.0)
    res.set(xlabel="photon energy (eV)", ylabel="fit - data")
    c.sci(res)
    c.save(fig, out, "znpc-fit", mode)

    # 2. decomposition: beta and the analytic delta partners
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True, gridspec_kw=dict(hspace=0.08))
    parts = c.peak_contributions(fit, E)
    for z in parts:
        a1.plot(E, z.imag, color=s["ink2"], lw=0.7, alpha=0.7)
        a2.plot(E, z.real, color=s["ink2"], lw=0.7, alpha=0.7)
    a1.plot([], [], color=s["ink2"], lw=0.7, label="individual peaks")
    a1.plot(E, cont.imag, color=orange, label="continuum")
    a1.plot(E, total.imag, color=blue, label="total")
    a2.plot(E, cont.real, color=orange, label="continuum delta (fixed kkcalc table)")
    a2.plot(E, total.real, color=blue, label="total delta = table + analytic peak deltas")
    a2.axhline(0, color=s["ink2"], lw=0.6)
    a1.set(ylabel=r"$\beta$", title="Building blocks: absorption and its Kramers-Kronig partner")
    a2.set(ylabel=r"$\delta$", xlabel="photon energy (eV)")
    a1.legend(loc="upper right")
    a2.legend(loc="lower right")
    for a in (a1, a2):
        c.sci(a)
    c.save(fig, out, "znpc-decomposition", mode)

    # 3. all five angles, each fit independently (warm start = same seeds)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for theta, col in zip(ANGLE_FITS, ANGLE_COLORS[mode]):
        Ea, ba = spectrum(theta)
        f = ANGLE_FITS[theta]
        ax.plot(Ea, f.model(Ea).imag, color=col, label=f"{theta:g}°")
        ax.plot(Ea, ba, ".", ms=1.5, color=col, alpha=0.35)
    ax.set(xlabel="photon energy (eV)", ylabel=r"$\beta$", title="Angle series: five independent fits (points = data, lines = fit)")
    ax.legend(title="incidence angle", ncol=5, loc="upper right")
    c.sci(ax)
    c.save(fig, out, "znpc-angles", mode)


ANGLE_FITS: dict = {}
ANGLE_COLORS = {
    "light": ["#2a78d6", "#1baf7a", "#eda100", "#eb6834", "#6250d6"],
    "dark": ["#3987e5", "#199e70", "#c98500", "#d95926", "#9085e9"],
}


def run(out: Path | str = "docs/assets/generated") -> dict:
    out = Path(out)
    seeds = datasets.load_table("znpc_dft_clusters")
    E, beta = spectrum(MAGIC_ANGLE)
    fit = c.fit_peaks(E, beta, MATERIAL, seeds)

    ANGLE_FITS.clear()
    angle_rows = []
    t = datasets.load_table("znpc_nexafs")
    for theta in sorted(set(t["theta_deg"])):
        Ea, ba = spectrum(theta)
        f = fit if np.isclose(theta, MAGIC_ANGLE) else c.fit_peaks(Ea, ba, MATERIAL, seeds)
        ANGLE_FITS[float(theta)] = f
        angle_rows.append((theta, f.seconds, f.n_evals, f.rel_rms))

    for mode in ("light", "dark"):
        figures(fit, E, beta, out, mode)

    out.mkdir(parents=True, exist_ok=True)
    (out / "znpc_peaks.md").write_text(c.peak_table_markdown(fit))
    lines = ["| angle | fit time (s) | model evaluations | rms residual / mean beta |", "|--:|--:|--:|--:|"]
    lines += [f"| {a:g}° | {s:.2f} | {n} | {r:.2%} |" for a, s, n, r in angle_rows]
    (out / "znpc_angles.md").write_text("\n".join(lines) + "\n")
    return dict(rel_rms=fit.rel_rms, seconds=fit.seconds, n_evals=fit.n_evals, angle_rel_rms={a: r for a, _, _, r in angle_rows})


if __name__ == "__main__":
    res = run(sys.argv[1] if len(sys.argv) > 1 else "docs/assets/generated")
    print(f"magic-angle fit: rms residual {res['rel_rms']:.2%} of mean beta in {res['seconds']:.2f} s ({res['n_evals']} evaluations)")
