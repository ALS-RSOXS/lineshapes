"""Build the example datasets shipped in python/lineshapes/data/.

Sources are the ZnPc and Alq3 files in lr-optical-models (``@models/optical``). Run with the dev
environment plus periodictable (bare-atom data); kkcalc2 is a dev dependency:

    LR_OPTICAL_MODELS=/path/to/lr-optical-models \
      uv run --with periodictable python tools/make_example_data.py

Outputs (all committed, so users and CI never need the sources):

  znpc_nexafs.csv           ZnPc C K-edge NEXAFS at five angles: mu in cm^2/g, normalized to the bare atom
  znpc_dft_clusters.csv     DFT transition clusters (energy_ev, amplitude, sigma_ev) used as peak seeds
  alq3_oc_tey.csv           Alq3 optical constants (delta, beta) from total-electron-yield XAS
  alq3_oc_model.csv         Alq3 optical constants of the lr-optical-models building-block model
  alq3_dft_clusters.csv     DFT transition clusters for Alq3
  <name>_continuum.json     Continuum beta parameters (fixed ahead of time, from the bare atom) + provenance
  <name>_continuum_delta.csv  Continuum delta table = kkcalc2 of that continuum beta (density 1.0)

Optical-constant tables are per unit density (1 g/cm^3); the film density is applied as the model
``scale``.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

SRC = Path(os.environ.get("LR_OPTICAL_MODELS", "/Users/hduva/projects/lr-optical-models")) / "@models/optical"
OUT = Path(__file__).resolve().parents[1] / "python" / "lineshapes" / "data"
HC_EV_ANGSTROM = 12398.419843320026


def rows(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def write_csv(name, header, table, fmt="{:.8g}"):
    with open(OUT / name, "w") as fh:
        fh.write(",".join(header) + "\n")
        for r in table:
            fh.write(",".join(fmt.format(v) if isinstance(v, float) else str(v) for v in r) + "\n")
    print(f"  {name}: {len(table)} rows")


# ----------------------------------------------------------------------------- spectra
def znpc_nexafs():
    r = rows(SRC / "znpc/exp/nexafs.csv")
    table = [(float(x["energy"]), float(x["theta_deg"]), float(x["intensity_norm"]), float(x["intensity_raw"])) for x in r]
    write_csv("znpc_nexafs.csv", ["energy_ev", "theta_deg", "mu_cm2_per_g", "intensity_raw"], table, "{:.7g}")
    return min(t[0] for t in table), max(t[0] for t in table)


def clusters(material, extra_cols=()):
    r = rows(SRC / f"{material}/exp/dft_clusters.csv")
    cols = ["energy_ev", "amplitude", "sigma_ev", *extra_cols]
    write_csv(f"{material}_dft_clusters.csv", cols, [tuple(float(x[c]) for c in cols) for x in r])


def alq3_oc():
    for src, name in (("alq3_TEY_Dec2025_short.csv", "alq3_oc_tey.csv"), ("Alq3_model_oc.csv", "alq3_oc_model.csv")):
        r = rows(SRC / "alq3/exp" / src)
        for x in r:  # isotropic tables: in-plane and out-of-plane agree
            assert abs(float(x["n_xx"]) - float(x["n_zz"])) < 1e-12 and abs(float(x["n_ixx"]) - float(x["n_izz"])) < 1e-12
        write_csv(name, ["energy_ev", "delta", "beta"], [(float(x["energy"]), float(x["n_xx"]), float(x["n_ixx"])) for x in r])
    r = rows(SRC / "alq3/exp/alq3_TEY_Dec2025_short.csv")
    return float(r[0]["energy"]), float(r[-1]["energy"])


# ----------------------------------------------------------------------------- continuum
def bare_atom(formula, energy_ev):
    """Unit-density bare-atom (delta, beta) from periodictable (n = 1 - delta - i beta)."""
    from periodictable import xsf

    n = xsf.index_of_refraction(formula, density=1.0, energy=np.asarray(energy_ev) / 1000.0)
    return 1.0 - n.real, -n.imag


def kkcalc_delta(energy, beta, formula):
    from kkcalc2 import stoichiometry
    from kkcalc2.models import asf_im, asp_db_im, asp_db_im_extended

    st = stoichiometry(formula)
    asf = asf_im.from_betas(energies=energy, absorption=beta, stoichiometry=st, density=1.0)
    kk = asp_db_im_extended(data_asf=asf, database=asp_db_im(st)).kk_transform(
        relativistic_correction=0.0, tolerance=0.01, max_iter=1
    )
    return np.interp(energy, np.asarray(kk.energies), np.asarray(kk.deltas))


def continuum(name, formula, window, e_ion, width, pre_end, post_start):
    """Fixed continuum: linear pre-edge + erf ionization step (+ post-edge slope change) at ``e_ion``.

    bkg, slope, height, decay are fit to the bare-atom beta outside [pre_end, post_start] (the
    near-edge region, where the molecular spectrum differs from the atom). The continuum delta
    is kkcalc2 of the resulting beta, computed once.
    """
    from scipy.optimize import least_squares

    from lineshapes import Model

    lo, hi = window
    E = np.arange(lo, hi + 1e-9, 0.05)
    _, beta_bare = bare_atom(formula, E)
    mask = (E < pre_end) | (E > post_start)

    def beta_of(p, e):
        bkg, slope, height, decay = p
        return Model().continuum(bkg, slope).step(e_ion, width, height, decay)(e).imag

    p0 = [beta_bare[0], 0.0, beta_bare[-1] - beta_bare[0], 0.0]
    sol = least_squares(lambda p: (beta_of(p, E[mask]) - beta_bare[mask]) / beta_bare[mask], p0, x_scale=[1e-4, 1e-6, 1e-4, 1e-6])
    bkg, slope, height, decay = (float(v) for v in sol.x)
    beta_c = beta_of(sol.x, E)
    delta_c = kkcalc_delta(E, beta_c, formula)
    meta = dict(
        formula=formula, density_g_cm3=1.0, window_ev=[lo, hi], bkg=bkg, slope=slope,
        step=dict(e0=e_ion, width=width, height=height, decay=decay),
        fit_regions_ev=[[lo, pre_end], [post_start, hi]],
        rms_rel_beta_error_in_fit_regions=float(np.sqrt(np.mean(((beta_c - beta_bare)[mask] / beta_bare[mask]) ** 2))),
        note="Continuum beta = bkg + slope*E + erf step at the ionization energy; delta table = kkcalc2 of this "
             "beta on the example window (density=1.0, per unit density). Fixed ahead of time; peaks are fit on top.",
    )
    (OUT / f"{name}_continuum.json").write_text(json.dumps(meta, indent=2) + "\n")
    write_csv(f"{name}_continuum_delta.csv", ["energy_ev", "delta"], list(zip(E.tolist(), delta_c.tolist())), "{:.9g}")
    print(f"  {name}_continuum.json: rel rms in fit regions {meta['rms_rel_beta_error_in_fit_regions']:.2e}")


def mean_ip(path, col):
    return float(np.mean([float(x[col]) for x in rows(path)]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("ZnPc")
    z_win = znpc_nexafs()
    clusters("znpc")
    ip = mean_ip(SRC / "znpc/exp/ionization_energies.csv", "ionization_potential_shifted_ev")
    continuum("znpc", "ZnC32H16N8", z_win, ip, 1.5, pre_end=280.0, post_start=305.0)
    print("Alq3")
    a_win = alq3_oc()
    clusters("alq3", ("theta_deg",))
    ip = mean_ip(SRC / "alq3/exp/ionization_energies.csv", "ionization_potential_ev")
    continuum("alq3", "AlC27H18N3O3", a_win, ip, 1.5, pre_end=280.0, post_start=305.0)


if __name__ == "__main__":
    main()
