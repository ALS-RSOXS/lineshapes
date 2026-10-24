"""Visual sanity checks for lineshapes_ref.py. Writes PNGs to prototypes/figures/.

Run:
  uv run --no-project --python 3.12 --with numpy --with scipy --with matplotlib \
    --with "kkcalc2 @ git+https://github.com/xraysoftmat/kkcalc.git@e750040f96fded5cf8044282d6db9a84e96d353c" \
    python prototypes/visualize.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import lineshapes_ref as ls

HERE = Path(__file__).parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
LRO = Path("/Users/hduva/projects/lr-optical-models/@models/optical/cu/bb")

E0 = 930.0
# Very wide uniform grid so the FFT Hilbert sees the full tails.
E_WIDE = np.linspace(E0 - 2000, E0 + 2000, 2**18)
WIN = (E_WIDE > E0 - 15) & (E_WIDE < E0 + 15)


def sci(ax):
    """Scientific-notation y axis."""
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)


RES = dict(height_ratios=[4, 1])


def fig_peaks():
    """Each lineshape: beta, analytic delta, FFT-Hilbert delta, and residual."""
    cases = [
        ("Gaussian (1/e half-width=1.0)", ls.gaussian, dict(e0=E0, width=1.0, amp=1.0)),
        ("Lorentzian (gamma=0.5 HWHM)", ls.lorentzian, dict(e0=E0, gamma=0.5, amp=1.0)),
        ("Voigt (sigma=0.5, gamma=0.3)", ls.voigt, dict(e0=E0, sigma=0.5, gamma=0.3, amp=1.0)),
        ("pseudo-Voigt (fwhm=1.2, eta=0.4)", ls.pseudo_voigt, dict(e0=E0, fwhm=1.2, eta=0.4, amp=1.0)),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 6), sharex=True,
                             gridspec_kw=RES)
    for j, (title, f, kw) in enumerate(cases):
        z = f(E_WIDE, **kw)
        num = ls.hilbert_fft(E_WIDE, z.imag)
        x = E_WIDE[WIN]
        ax = axes[0, j]
        ax.plot(x, z.imag[WIN], label="beta")
        ax.plot(x, z.real[WIN], label="delta analytic")
        ax.plot(x, num[WIN], "--", label="delta = FFT Hilbert[beta]")
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(E0, color="k", lw=0.5, ls=":")
        ax.set_title(title)
        axes[1, j].plot(x, (z.real - num)[WIN], color="C3")
        axes[1, j].set_ylabel("analytic - FFT")
        sci(axes[1, j])
        axes[1, j].set_xlabel("E (eV)")
    axes[0, 0].legend()
    fig.suptitle("Peak lineshapes: delta + i beta (delta = +H[beta]; delta<0 below E0)")
    fig.tight_layout()
    fig.savefig(FIG / "01_peaks_vs_numeric_hilbert.png", dpi=120)


def fig_voigt_limits():
    """Voigt reduces to Gaussian / Lorentzian; TCH pseudo-Voigt vs exact Voigt."""
    x = E_WIDE[WIN]
    sig = 1.0 / np.sqrt(2)  # 1/e half-width 1.0
    f, eta = ls.tch_pseudo_voigt_params(0.5, 0.3)
    cases = [
        ("Voigt gamma->0 vs Gaussian", ls.voigt(x, E0, sig, 1e-6, 1.0), ls.gaussian(x, E0, 1.0, 1.0), "G"),
        ("Voigt sigma->0 vs Lorentzian", ls.voigt(x, E0, 1e-3, 0.5, 1.0), ls.lorentzian(x, E0, 0.5, 1.0), "L"),
        (f"Voigt vs TCH pV (fwhm={f:.3f}, eta={eta:.3f})", ls.voigt(x, E0, 0.5, 0.3, 1.0),
         ls.pseudo_voigt(x, E0, f, eta, 1.0), "pV"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 5.5), sharex=True, gridspec_kw=RES)
    for j, (title, v, ref, nm) in enumerate(cases):
        ax = axes[0, j]
        ax.plot(x, v.imag, label="Voigt beta"); ax.plot(x, v.real, label="Voigt delta")
        ax.plot(x, ref.imag, "--", label=f"{nm} beta"); ax.plot(x, ref.real, "--", label=f"{nm} delta")
        ax.set_title(title); ax.legend(fontsize=8)
        r = axes[1, j]
        r.plot(x, v.imag - ref.imag, label="beta"); r.plot(x, v.real - ref.real, label="delta")
        r.set_ylabel(f"Voigt - {nm}"); r.set_xlabel("E (eV)"); sci(r); r.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "02_voigt_limits.png", dpi=120)


def load_cu():
    with open(LRO / "peaks.csv") as fh:
        rows = list(csv.DictReader(fh))
    peaks = [(ls.gaussian, dict(e0=float(r["location"]), width=float(r["width"]),
                                amp=float(r["height"]))) for r in rows]
    ion = json.loads((LRO / "ionization_edge.json").read_text())
    steps = [dict(e0=v["edge_energy"], width=v["width"], height=v["height"],
                  decay=v.get("decay", 0.0)) for v in ion["edges"].values()]
    cont = dict(bkg=ion["bkg"], slope=ion["slope"], steps=steps)
    return peaks, cont


def kkcalc_delta(E, beta, formula="Cu", density=1.0):
    from kkcalc2 import stoichiometry
    from kkcalc2.models import asf_im, asp_db_im, asp_db_im_extended

    st = stoichiometry(formula)
    asf = asf_im.from_betas(energies=E, absorption=beta, stoichiometry=st, density=density)
    kk = asp_db_im_extended(data_asf=asf, database=asp_db_im(st)).kk_transform(
        relativistic_correction=0.0, tolerance=0.01, max_iter=1)
    return np.interp(E, np.asarray(kk.energies), np.asarray(kk.deltas))


def fig_cu_model():
    """NOTE: kkcalc2@e750040 output scales linearly with `density` (stoichiometry number
    density overrides it for beta->f2 but not f1->delta), so density=1.0 is used,
    matching lr-optical-models bookended_*.py.

    Real Cu L-edge model: peaks (analytic delta) + continuum (kkcalc delta table).

    Also isolates each peak's kkcalc delta as kk(bg + peaks) - kk(bg) and
    compares it with the analytic Dawson sum.
    """
    peaks, cont = load_cu()
    E = np.linspace(900, 1000, 4001)
    beta_bg = ls.continuum_beta(E, **cont)
    beta_pk = sum(f(E, **kw).imag for f, kw in peaks)
    delta_pk = sum(f(E, **kw).real for f, kw in peaks)

    d_bg = kkcalc_delta(E, beta_bg)
    d_tot = kkcalc_delta(E, beta_bg + beta_pk)
    d_pk_kk = d_tot - d_bg

    model = ls.optical_model(E, peaks, continuum=cont, delta_table=(E, d_bg))

    fig, axes = plt.subplots(
        3, 1, figsize=(10, 10.5), sharex=True,
        gridspec_kw=dict(height_ratios=[4, 4, 1]))
    a = axes[0]
    a.plot(E, beta_bg, label="continuum beta (erf steps + linear)")
    a.plot(E, model.imag, label="total beta (model)")
    for f, kw in peaks:
        a.plot(E, f(E, **kw).imag, lw=0.6, color="gray")
    a.set_ylabel("beta"); a.legend(fontsize=8)
    a.set_title("Cu L-edge model (lr-optical-models @models/optical/cu/bb)")
    a = axes[1]
    a.plot(E, d_bg, label="continuum delta (kkcalc, fixed table)")
    a.plot(E, model.real, label="total delta = table + analytic peaks")
    a.plot(E, d_tot, "--", label="kkcalc(total beta)")
    a.set_ylabel("delta (full model)"); a.legend(fontsize=8)
    axes[2].plot(E, model.real - d_tot, color="C3")
    axes[2].set_ylabel("model -\nkkcalc", fontsize=8)
    axes[2].set_xlabel("E (eV)")
    for ax in axes:
        sci(ax)
    fig.tight_layout()
    fig.savefig(FIG / "03_cu_model_kkcalc.png", dpi=120)
    full = np.abs(model.real - d_tot).max() / np.abs(d_tot).max()
    print(f"Cu full model: max|model - kkcalc(total)| / max|kkcalc| = {full:.3e}")
    rel = np.abs(delta_pk - d_pk_kk).max() / np.abs(delta_pk).max()
    print(f"Cu peaks: max|analytic - kkcalc| / max|analytic| = {rel:.3e}")


if __name__ == "__main__":
    fig_peaks()
    fig_voigt_limits()
    fig_cu_model()
    print("wrote", sorted(p.name for p in FIG.glob("*.png")))
