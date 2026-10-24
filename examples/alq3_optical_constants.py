"""Alq3 carbon K-edge: fit beta, then compare the analytic delta with an independent delta table.

The shipped table (`alq3_oc_tey`) carries both beta and delta. We fit only beta, on top of the fixed
continuum, and ask how well the model's delta (analytic peak partners + the fixed continuum table)
reproduces the table's delta.

    uv run python examples/alq3_optical_constants.py [output_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import _common as c  # noqa: E402

from lineshapes import datasets  # noqa: E402

MATERIAL = "alq3"


def figures(fit, E, beta, delta, out: Path, mode: str) -> None:
    s = c.apply_style(mode)
    blue, orange, aqua = s["series"]
    total = fit.model(E)
    cont = datasets.continuum_model(MATERIAL)(E)

    fig, (ax, res) = c.fit_figure()
    ax.plot(E, beta, ".", ms=2.5, color=s["ink2"], label="data (TEY-derived)", zorder=1)
    ax.plot(E, fit.seed_model(E).imag, "--", color=aqua, lw=1.2, label="starting guess (DFT seeds)")
    ax.plot(E, cont.imag, color=orange, label="fixed continuum")
    ax.plot(E, total.imag, color=blue, label="fit")
    ax.set(ylabel=r"$\beta$", title="Alq3 C K-edge: beta fit")
    ax.legend(loc="upper left")
    c.sci(ax)
    res.axhline(0, color=s["ink2"], lw=0.6)
    res.plot(E, total.imag - beta, color=blue, lw=1.0)
    res.set(xlabel="photon energy (eV)", ylabel="fit - data")
    c.sci(res)
    c.save(fig, out, "alq3-fit", mode)

    fig, (ax, res) = c.fit_figure()
    ax.plot(E, delta, ".", ms=2.5, color=s["ink2"], label="delta table (independent)", zorder=1)
    ax.plot(E, cont.real, color=orange, label="continuum delta (fixed kkcalc table)")
    ax.plot(E, total.real, color=blue, label="model delta = table + analytic peak deltas")
    ax.axhline(0, color=s["ink2"], lw=0.6)
    ax.set(ylabel=r"$\delta$", title="Alq3: delta predicted from the beta fit")
    ax.legend(loc="lower right")
    c.sci(ax)
    res.axhline(0, color=s["ink2"], lw=0.6)
    res.plot(E, total.real - delta, color=blue, lw=1.0)
    res.set(xlabel="photon energy (eV)", ylabel="model - table")
    c.sci(res)
    c.save(fig, out, "alq3-delta", mode)


def run(out: Path | str = "docs/assets/generated") -> dict:
    out = Path(out)
    t = datasets.load_table("alq3_oc_tey")
    E, beta, delta = t["energy_ev"], t["beta"], t["delta"]
    fit = c.fit_peaks(E, beta, MATERIAL, datasets.load_table("alq3_dft_clusters"))
    model_delta = fit.model(E).real
    delta_rms = float(np.sqrt(np.mean((model_delta - delta) ** 2)))
    delta_max = float(np.max(np.abs(model_delta - delta)))
    delta_span = float(np.ptp(delta))
    offset = float(np.median(model_delta - delta))

    for mode in ("light", "dark"):
        figures(fit, E, beta, delta, out, mode)
    out.mkdir(parents=True, exist_ok=True)
    (out / "alq3_peaks.md").write_text(c.peak_table_markdown(fit))
    (out / "alq3_summary.md").write_text(
        "| quantity | value |\n|--|--:|\n"
        f"| beta rms residual / mean beta | {fit.rel_rms:.2%} |\n"
        f"| fit time, {fit.n_evals} model evaluations | {fit.seconds:.2f} s |\n"
        f"| delta rms difference vs table | {delta_rms:.1e} ({delta_rms / delta_span:.1%} of the delta range) |\n"
        f"| delta max difference vs table | {delta_max:.1e} |\n"
        f"| median delta offset (model - table) | {offset:.1e} |\n"
    )
    return dict(rel_rms=fit.rel_rms, seconds=fit.seconds, delta_rms=delta_rms, delta_max=delta_max, delta_span=delta_span, offset=offset)


if __name__ == "__main__":
    r = run(sys.argv[1] if len(sys.argv) > 1 else "docs/assets/generated")
    print(f"beta fit rms {r['rel_rms']:.2%} of mean; delta vs table: rms {r['delta_rms']:.2e}, max {r['delta_max']:.2e}, offset {r['offset']:.2e}")
