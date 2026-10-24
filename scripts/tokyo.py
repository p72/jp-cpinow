"""東京都区部ブリッジ: when Tokyo's figure for a target month is out (中旬速報, ~3 weeks
before the national release), blend it into the national leaf forecast.

  national dev_t = β · Tokyo dev_t + ε     (dev = log m/m minus the 5-year seasonal median)
The bridge and the model forecast are combined by inverse error variance.
Tokyo-only policies (policy.csv region=東京) are removed from the Tokyo change first.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, e1, policy

SKIP = {"electricity", "city_gas", "gasoline", "kerosene"}   # structural models are better


def tokyo_only_impact(leaf: str, months) -> pd.Series:
    df = policy._raw()
    df = df[(df["kind"] == "event") & (df["leaf"] == leaf) & (df["region"] == "東京")]
    s = pd.Series(0.0, index=pd.DatetimeIndex(months))
    for _, r in df.iterrows():
        if np.isnan(r["value"]):
            continue
        for m in policy._months(r):
            if m in s.index:
                s[m] += np.log1p(r["value"] / 100)
    return s


def bridge(leaf: config.Leaf, Ln: pd.Series, Lt: pd.Series, month: pd.Timestamp,
           g_model: float, rmse_model: float | None):
    """Return (blended log m/m for `month`, weight on Tokyo, β) or None if unavailable."""
    if leaf.key in SKIP or leaf.policy_only or Lt is None or month not in Lt.dropna().index:
        return None
    idx = pd.date_range(Ln.index[0], month, freq="MS")
    yn = np.log(Ln).diff().reindex(idx)
    yt = np.log(Lt).diff().reindex(idx)
    yt = yt - tokyo_only_impact(leaf.key, idx).reindex(idx).fillna(0)
    mn = policy.masked(leaf.key, idx)
    Sn, St = e1.seasonal(yn, mn), e1.seasonal(yt, mn)
    dn, dt = (yn - Sn).where(~mn), (yt - St).where(~mn)
    d = pd.concat([dn.rename("n"), dt.rename("t")], axis=1).loc[e1.TRAIN_START:month - pd.offsets.MonthBegin(1)].dropna()
    if len(d) < 36 or np.isnan(dt.get(month, np.nan)):
        return None
    beta = float((d["n"] * d["t"]).sum() / (d["t"] ** 2).sum())
    resid = d["n"] - beta * d["t"]
    var_b = float(resid.var())
    var_m = (rmse_model or float(d["n"].std())) ** 2
    w = var_m / (var_m + var_b)
    S_m = Sn.get(month, 0.0)
    S_m = 0.0 if np.isnan(S_m) else S_m
    g_bridge = S_m + beta * dt[month]
    return w * g_bridge + (1 - w) * g_model, w, beta
