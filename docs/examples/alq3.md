# Alq3 optical constants

Tris(8-hydroxyquinoline)aluminum (Alq3) at the carbon K-edge. The shipped table `alq3_oc_tey` holds both
$\beta$ and $\delta$ derived from a total-electron-yield measurement, so it lets us ask a sharper question
than a $\beta$ fit alone:

> If we fit only $\beta$, how well does the model's analytic $\delta$ reproduce an independent
> $\delta$ table?

```python
from lineshapes import datasets

t = datasets.load_table("alq3_oc_tey")        # energy_ev, delta, beta  (per unit density)
E, beta, delta = t["energy_ev"], t["beta"], t["delta"]
seeds = datasets.load_table("alq3_dft_clusters")
```

The fit is the same recipe as the [ZnPc example](znpc.md#the-recipe): fixed continuum, DFT-seeded
Gaussians, linear start, nonlinear refinement. Only $\beta$ enters the fit.

![Alq3 beta fit](../assets/generated/alq3-fit-light.png#only-light)
![Alq3 beta fit](../assets/generated/alq3-fit-dark.png#only-dark)

## Predicted $\delta$ against the table

The model's $\delta$ (continuum table plus analytic peak partners) follows every feature of the table's
$\delta$: the dispersive dips at the $\pi^*$ resonances, the shoulder above the edge, and the long
low-energy tail.

![Model delta against the delta table](../assets/generated/alq3-delta-light.png#only-light)
![Model delta against the delta table](../assets/generated/alq3-delta-dark.png#only-dark)

--8<-- "docs/assets/generated/alq3_summary.md"

The difference is dominated by a nearly constant offset. That is expected: the two $\delta$ curves rest on
different continuum assumptions. The continuum $\delta$ fixes the baseline (here the bare atom, through
kkcalc2), and a constant in $\delta$ is not constrained by $\beta$ in a finite window. The *structure*
in $\delta$ (everything the peaks contribute) is reproduced by the analytic partners.

--8<-- "docs/assets/generated/alq3_peaks.md"
