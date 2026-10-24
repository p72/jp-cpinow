"""生鮮野菜: blend the weekly MAFF retail survey (食品価格動向調査) into the forecast.

  CPI fresh_veg log m/m ≈ a + b · Δlog(monthly mean of the weekly veg index)
For a target month with k weekly surveys already out, the bridge variance is the
full-month residual variance × (4 / k); bridge and model are combined by inverse
variance. Needs the previous month in the survey too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _monthly(v: pd.Series):
    g = v.groupby(v.index.to_period("M"))
    return g.mean().to_timestamp(), g.size().to_timestamp() if hasattr(g.size(), "to_timestamp") else None


def blend(L: pd.Series, veg: pd.DataFrame | None, g_model: pd.Series, rmse_model: float | None):
    """g_model: log m/m forecasts for months. Returns (updated g, info)."""
    if veg is None or "veg_index" not in veg or veg["veg_index"].dropna().empty:
        return g_model, {}
    v = veg["veg_index"].dropna()
    per = v.index.to_period("M")
    mean = v.groupby(per).mean()
    cnt = v.groupby(per).size()
    mean.index, cnt.index = mean.index.to_timestamp(), cnt.index.to_timestamp()
    x = np.log(mean).diff()
    y = np.log(L).diff()
    T = L.dropna().index[-1]
    full = cnt[cnt >= 4].index
    d = pd.concat([y.rename("y"), x.rename("x")], axis=1).loc[:T].dropna()
    d = d[d.index.isin(full)]
    if len(d) < 18:
        return g_model, {}
    b, a = np.polyfit(d["x"], d["y"], 1)
    var_full = float(np.var(d["y"] - (a + b * d["x"])))
    var_m = (rmse_model or float(y.loc[:T].std())) ** 2
    g = g_model.copy()
    info = {}
    for m in g.index:
        prev = m - pd.offsets.MonthBegin(1)
        if m not in cnt.index or prev not in mean.index:
            continue
        k = int(cnt[m])
        # previous month mean is from the survey too (CPI level of prev is from the model
        # when prev > T, which is consistent because we chain m/m changes)
        xb = np.log(mean[m] / mean[prev])
        gb = a + b * xb
        var_b = var_full * 4 / max(k, 1)
        w = var_m / (var_m + var_b)
        g[m] = w * gb + (1 - w) * g[m]
        info[m] = {"weeks": k, "w": round(w, 2)}
    return g, info
