"""Prediction intervals from pseudo-real-time backtest errors (scripts/error_bands.csv).

For each target × h × nowcast point (a/c/b) the 5% / 25% / 75% / 95% quantiles of the
forecast error (forecast − published) are stored, pooled over 2015–19 and 2022–26.
interval = [fc − q95, fc − q05] (90%) and [fc − q75, fc − q25] (50%): no normality
assumption, and a bias in the errors shifts the interval. Policy uncertainty (subsidies
not yet decided) is NOT included. Rebuild with `python -m scripts.cli bands`.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import release

PATH = os.path.join(os.path.dirname(__file__), "error_bands.csv")


def build(bt: pd.DataFrame, engine: str = "model") -> pd.DataFrame:
    b = bt[(bt["engine"] == engine) & ~bt["T"].between("2020-01-01", "2021-12-01")]
    rows = []
    for (k, p, h), g in b.groupby(["target", "point", "h"]):
        e = g["err"].dropna()
        rows.append(dict(target=k, point=p, h=h, n=len(e), mae=e.abs().mean(),
                         rmse=float(np.sqrt((e ** 2).mean())), bias=e.mean(),
                         q05=e.quantile(0.05), q25=e.quantile(0.25),
                         q75=e.quantile(0.75), q95=e.quantile(0.95)))
    df = pd.DataFrame(rows)
    df.to_csv(PATH, index=False)
    return df


def point_of(asof, T) -> str:
    """Which nowcast point (a/c/b) an as-of date corresponds to."""
    o = release.origins(T)
    a = pd.Timestamp(asof)
    if a >= o["b"]:
        return "b"
    if a >= o["c"]:
        return "c"
    return "a"


def interval(target: str, h: int, point: str, fc: float, level: float = 0.9):
    if not os.path.exists(PATH):
        return np.nan, np.nan
    df = pd.read_csv(PATH)
    r = df[(df["target"] == target) & (df["h"] == h) & (df["point"] == point)]
    if r.empty:
        return np.nan, np.nan
    r = r.iloc[0]
    lo, hi = ("q95", "q05") if level == 0.9 else ("q75", "q25")
    return fc - r[lo], fc - r[hi]
