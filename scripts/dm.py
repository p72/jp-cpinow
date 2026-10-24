"""Diebold–Mariano test with the Harvey–Leybourne–Newbold small-sample correction.

One-sided p-value for "model A has smaller loss than model B" (absolute-error loss),
long-run variance with h−1 autocovariances, t(n−1) reference.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def dm(e_a, e_b, h: int = 1, power: int = 1) -> tuple[float, float]:
    ea, eb = np.asarray(e_a, float), np.asarray(e_b, float)
    ok = ~(np.isnan(ea) | np.isnan(eb))
    d = np.abs(ea[ok]) ** power - np.abs(eb[ok]) ** power
    n = len(d)
    if n < 10:
        return np.nan, np.nan
    dbar = d.mean()
    gam = [np.mean((d[k:] - dbar) * (d[:n - k] - dbar)) for k in range(h)]
    var = (gam[0] + 2 * sum(gam[1:])) / n
    if var <= 0:
        return np.nan, np.nan
    stat = dbar / np.sqrt(var)
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(stat), float(stats.t.cdf(stat, df=n - 1))
