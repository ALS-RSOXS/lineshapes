"""Timing: numpy reference vs Rust (one-shot function and prebuilt Model). `uv run python benchmarks/bench.py`"""

import csv
import json
import timeit
from pathlib import Path

import numpy as np

import lineshapes_rs as rs
from lineshapes_rs import Model
from tests.reference import lineshapes_ref as ref

ROOT = Path(__file__).resolve().parents[1]
E = np.linspace(900, 1000, 4001)


def t(f, n=300):
    f()
    return min(timeit.repeat(f, number=n, repeat=7)) / n * 1e6


PEAKS = [
    ("gaussian", dict(e0=930, width=1, amp=1)),
    ("lorentzian", dict(e0=930, gamma=0.5, amp=1)),
    ("voigt", dict(e0=930, sigma=0.5, gamma=0.3, amp=1)),
    ("pseudo_voigt", dict(e0=930, fwhm=1.2, eta=0.4, amp=1)),
]
print(f"{'4001 points':14s}{'reference':>12s}{'rust fn':>10s}{'prebuilt':>10s}{'speedup':>9s}")
for name, kw in PEAKS:
    m = getattr(Model(), name)(**kw)
    tr, tf, tp = t(lambda: getattr(ref, name)(E, **kw)), t(lambda: getattr(rs, name)(E, **kw)), t(lambda: m(E))
    print(f"{name:14s}{tr:10.1f}us{tf:8.1f}us{tp:8.1f}us{tr / tp:8.1f}x")

rows = list(csv.DictReader(open(ROOT / "tests/data/cu/peaks.csv")))
ion = json.loads((ROOT / "tests/data/cu/ionization_edge.json").read_text())
tab = (E, np.zeros_like(E))
m = Model()
for r in rows:
    m.gaussian(float(r["location"]), float(r["width"]), float(r["height"]))
m.continuum(ion["bkg"], ion["slope"])
for v in ion["edges"].values():
    m.step(v["edge_energy"], v["width"], v["height"], v["decay"])
m.delta_table(*tab)
peaks = [(ref.gaussian, dict(e0=float(r["location"]), width=float(r["width"]), amp=float(r["height"]))) for r in rows]
cont = dict(bkg=ion["bkg"], slope=ion["slope"],
            steps=[dict(e0=v["edge_energy"], width=v["width"], height=v["height"], decay=v["decay"]) for v in ion["edges"].values()])
tr = t(lambda: ref.optical_model(E, peaks, continuum=cont, delta_table=tab), 50)
tp, tpar = t(lambda: m(E)), t(lambda: m(E, parallel=True))
print(f"{'Cu model':14s}{tr:10.1f}us{'':10s}{tp:8.1f}us{tr / tp:8.1f}x   parallel {tpar:.1f}us")
Eb = np.linspace(900, 1000, 400_001)
print(f"Cu model 400k points: serial {t(lambda: m(Eb), 5) / 1e3:.2f} ms, parallel {t(lambda: m(Eb, parallel=True), 5) / 1e3:.2f} ms")
