"""Pseudo-real-time backtest on the release calendar.

For every latest national month T, three nowcast points (release.origins):
  a = just after national CPI(T)  c = after Tokyo(T+1)  b = after BOJ(T+1)
the data set is cut to what was public, the engine forecasts T+1..T+3, and the
y/y is compared with the published value. Benchmarks: rw (y/y of T carried forward)
and seas (5-year seasonal-median m/m for every leaf, no other information).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, cpi, e1, forecast, release
from .data import Dataset

PERIODS = {"2015-19": ("2015-01-01", "2019-12-01"), "2022-26": ("2022-01-01", "2026-12-01")}


def _actual(ds: Dataset) -> pd.DataFrame:
    return pd.DataFrame({k: cpi.yoy(ds.nat[c]) for k, c, _ in config.TARGETS})


def _seas_modules():
    """Benchmark: every leaf = seasonal median only."""
    def f(leaf, s, ds, months, asof):
        idx = pd.date_range(s.index[0], months[-1], freq="MS")
        y = np.log(s).diff().reindex(idx)
        S = e1.seasonal(y, pd.Series(False, index=idx)).reindex(months).fillna(0)
        return s.iloc[-1] * np.exp(S.cumsum()), {}
    return {l.key: f for l in config.LEAVES}


def run(ds: Dataset, start="2015-01-01", end=None, points=("a", "c", "b"),
        engines=None, verbose=False) -> pd.DataFrame:
    """engines: {name: kwargs for forecast.run}; default = model + seas benchmark."""
    engines = engines or {"model": {}, "seas": {"modules": _seas_modules(), "use_tokyo": False}}
    act = _actual(ds)
    last = ds.T
    end = pd.Timestamp(end) if end else last - pd.offsets.MonthBegin(1)
    rows = []
    for T in pd.date_range(start, end, freq="MS"):
        for p, d in release.origins(T).items():
            if p not in points:
                continue
            cut = ds.cut(d)
            if cut.T != T:
                continue
            for name, kw in engines.items():
                try:
                    fc = forecast.run(cut, **kw)
                except Exception as ex:
                    print(f"[backtest] {T:%Y-%m} {p} {name}: {ex}")
                    continue
                for h, m in enumerate(fc.targets.index, start=1):
                    if m > last:
                        continue
                    for k in fc.targets.columns:
                        rows.append(dict(T=T, point=p, engine=name, h=h, month=m, target=k,
                                         fc=fc.targets.loc[m, k], actual=act.loc[m, k],
                                         rw=act.loc[T, k]))
        if verbose:
            print(f"[backtest] {T:%Y-%m}")
    df = pd.DataFrame(rows)
    df["err"] = df["fc"] - df["actual"]
    return df


def scores(bt: pd.DataFrame) -> pd.DataFrame:
    """MAE / RMSE by period × target × point × h, with the random walk alongside."""
    out = []
    for per, (s, e) in PERIODS.items():
        b = bt[(bt["T"] >= s) & (bt["T"] <= e)]
        for (eng, k, p, h), g in b.groupby(["engine", "target", "point", "h"]):
            out.append(dict(period=per, engine=eng, target=k, point=p, h=h, n=len(g),
                            mae=g["err"].abs().mean(), rmse=np.sqrt((g["err"] ** 2).mean()),
                            mae_rw=(g["rw"] - g["actual"]).abs().mean()))
    return pd.DataFrame(out)


def table(sc: pd.DataFrame, engine="model", point="a") -> pd.DataFrame:
    x = sc[(sc["engine"] == engine) & (sc["point"] == point)]
    return x.pivot_table(index=["period", "target"], columns="h", values="mae").round(3)
