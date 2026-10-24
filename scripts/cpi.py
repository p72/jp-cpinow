"""CPI data (e-Stat, 2025-base connected indices) and the leaf ⇄ aggregate algebra.

Residual leaves: R = (Σ w_c C_c over `plus` − Σ over `minus`) / W_R, which is the
Laspeyres index of the remaining items within one base (exact for 2025-base data,
a close approximation on the connected history, which only matters for training).
Aggregating leaves back to a target uses the same weights:
  A_t / A_{t-1} = Σ_l w_l L_{l,t} / Σ_l w_l L_{l,t-1}
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .sources import estat


def fetch_raw(area: str = config.AREA_NAT) -> pd.DataFrame:
    """Wide DataFrame (index: month start, columns: codes) of published index levels."""
    codes = config.all_codes()
    out = []
    for i in range(0, len(codes), 20):   # keep the URL short
        d = estat().get_stats_df(config.ESTAT_CPI, cdCat01=",".join(codes[i:i + 20]),
                                 cdArea=area, cdTab="1", monthly=False)
        out.append(d)
    d = pd.concat(out)
    d = d[d["time_code"].str[4:6] == "00"]
    d = d[d["time_code"].str[6:8] != "00"]   # months only (YYYY00MMMM)
    d["date"] = pd.to_datetime(d["time_code"].str[:4] + "-" + d["time_code"].str[6:8] + "-01")
    wide = d.pivot_table(index="date", columns="cat01_code", values="value", aggfunc="first")
    return wide.sort_index()


def fetch_yoy(area: str = config.AREA_NAT) -> pd.DataFrame:
    """Published y/y (%) of the five targets (表章項目 3 = 前年同月比). Columns = target keys.
    The official figure is computed from unrounded indices, so it can differ by ±0.1
    from a y/y recomputed from the published (0.1-rounded) indices."""
    codes = {c: k for k, c, _ in config.TARGETS}
    d = estat().get_stats_df(config.ESTAT_CPI, cdCat01=",".join(codes), cdArea=area, cdTab="3",
                             monthly=False)
    d = d[(d["time_code"].str[4:6] == "00") & (d["time_code"].str[6:8] != "00")]
    d["date"] = pd.to_datetime(d["time_code"].str[:4] + "-" + d["time_code"].str[6:8] + "-01")
    wide = d.pivot_table(index="date", columns="cat01_code", values="value", aggfunc="first")
    return wide.rename(columns=codes)[[k for k, _, _ in config.TARGETS]].sort_index()


def leaf_index(raw: pd.DataFrame, area: str = "nat") -> pd.DataFrame:
    """Leaf index levels on the 2025 = 100 basis. Residual leaves are the weighted
    difference of official indices divided by the residual weight (exact within the
    2025 base; a close approximation on the connected history)."""
    w = config.weights(area)
    out = {}
    for l in config.LEAVES:
        if l.code:
            if l.code in raw:
                out[l.key] = raw[l.code]
            continue
        if any(c not in raw for c in l.plus + l.minus):
            continue
        sign = {c: 1.0 for c in l.plus}
        sign.update({c: -1.0 for c in l.minus})
        num = sum(sign[c] * w[c] * raw[c] for c in sign)
        out[l.key] = num / config.leaf_weight(l, area)
    return pd.DataFrame(out)


def aggregate(leaves: pd.DataFrame, target: str, area: str = "nat",
              anchor: pd.Series | None = None) -> pd.Series:
    """Target index from leaf levels. With `anchor` (official series), the official
    level is kept up to its last observation and only the change after it is taken
    from the leaves (removes the residual/rounding drift)."""
    ls = config.leaves_of(target)
    wl = pd.Series({l.key: config.leaf_weight(l, area) for l in ls})
    L = leaves[wl.index]
    num = (L * wl).sum(axis=1, min_count=len(wl))
    agg = num / wl.sum()
    if anchor is None:
        return agg
    a = anchor.dropna()
    t0 = a.index[-1]
    fut = agg[agg.index > t0]
    if fut.empty:
        return a
    return pd.concat([a, a.iloc[-1] * fut / agg.loc[t0]])


def yoy(level: pd.Series, rounded: bool = True) -> pd.Series:
    """Published-style y/y: from indices rounded to 0.1, result rounded to 0.1."""
    x = level.round(1) if rounded else level
    r = (x / x.shift(12) - 1) * 100
    return r.round(1) if rounded else r


def mom(level: pd.DataFrame | pd.Series):
    return level / level.shift(1) - 1
