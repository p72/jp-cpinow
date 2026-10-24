"""E1: per-leaf structural-statistical model for month-on-month log changes.

  y_t = S_t + dev_t,   S_t = median of the same calendar month over the previous 5
  clean years (policy events / masked periods excluded), and the deviation is
  predicted directly for each horizon h from information at the origin o = t − h:
    dev_t ~ dev_o, mean(dev_{o-2..o}), mean(dev_{o-11..o}), upstream changes
  (ridge regression on clean history). Policy-only leaves skip the regression.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, policy

UPSTREAM = {
    "food": ["imp_food", "cgpi_food", "usdjpy"],
    "dining": ["cgpi_food", "imp_food"],
    "goods": ["usdjpy", "imp_all", "cgpi_all"],
}
RIDGE = 2.0
TRAIN_START = "2005-01-01"
# per-leaf options chosen by the leaf-level backtest (SKILL.md): dev = use the deviation
# regression; covid = drop the COVID window (policy.csv mask rows with empty leaf) from S
OPTS: dict = {  # leaf-level backtest, h=1..3, both 2015–19 and 2022–26 (2026-10-24)
    "alcohol": dict(dev=False, covid=False),
    "dining": dict(covid=False),
    "food_other": dict(covid=False),
    "fresh_fish": dict(dev=False, covid=False),
    "fresh_fruit": dict(dev=False),
    "goods_other": dict(covid=False),
    "handset": dict(dev=False, covid=False),
    "mobile_fee": dict(dev=False, covid=False),
    "package_tour": dict(dev=False, covid=False),
    "public_svc": dict(covid=False),
    "rent_imputed": dict(dev=False, covid=False),
    "svc_other": dict(covid=False),
    "water": dict(covid=False),
}
DEFAULT = dict(dev=True, covid=True, years=5)


def opts(key: str) -> dict:
    return {**DEFAULT, **OPTS.get(key, {})}


def seasonal(y: pd.Series, mask: pd.Series, years: int = 5) -> pd.Series:
    """S_t from the same month of the previous `years` years (clean months only)."""
    yc = y.where(~mask.reindex(y.index, fill_value=False))
    lags = pd.concat([yc.shift(12 * k) for k in range(1, years + 1)], axis=1)
    s = lags.median(axis=1, skipna=True)
    return s.where(lags.notna().sum(axis=1) >= 2)


def _features(dev: pd.Series, up: pd.DataFrame | None, o: pd.DatetimeIndex) -> pd.DataFrame:
    f = pd.DataFrame(index=dev.index)
    f["dev1"] = dev
    f["dev3"] = dev.rolling(3, min_periods=2).mean()
    f["dev12"] = dev.rolling(12, min_periods=8).mean()
    if up is not None:
        for c in up.columns:
            lu = np.log(up[c]).reindex(dev.index)
            f[f"{c}_3"] = lu - lu.shift(3)
            f[f"{c}_6"] = lu.shift(3) - lu.shift(6)
    return f.reindex(o)


def _ridge(X: np.ndarray, y: np.ndarray, lam: float):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    Z = (X - mu) / sd
    A = Z.T @ Z + lam * np.eye(Z.shape[1])
    beta = np.linalg.solve(A, Z.T @ (y - y.mean()))
    return lambda x: y.mean() + ((x - mu) / sd) @ beta


def forecast_leaf(leaf: config.Leaf, L: pd.Series, boj: pd.DataFrame, months, asof,
                  horizons=(1, 2, 3)):
    """Returns (levels for months, info) — months = T+1..T+H."""
    months = pd.DatetimeIndex(months)
    T = L.dropna().index[-1]
    idx = pd.date_range(L.index[0], months[-1], freq="MS")
    y = np.log(L).diff().reindex(idx)
    o = opts(leaf.key)
    if o["covid"] == "both":
        m1, m0 = policy.masked(leaf.key, idx, covid=True), policy.masked(leaf.key, idx, covid=False)
        S = 0.5 * (seasonal(y, m1, o["years"]) + seasonal(y, m0, o["years"]))
        mask = m1
    else:
        mask = policy.masked(leaf.key, idx, covid=o["covid"])
        S = seasonal(y, mask, o["years"])
    dev = (y - S).where(~mask)
    out, info = {}, {"S": {}, "dev": {}}
    ups = UPSTREAM.get(leaf.kind)
    up = None
    if ups and boj is not None and len(boj):
        up = boj[[c for c in ups if c in boj.columns]].copy()
        # (ds is already cut to the release calendar; later months hold the last value)
        up = up.reindex(pd.date_range(up.index[0], months[-1], freq="MS")).ffill()
    lvl = L.loc[T]
    for h, m in zip(horizons, months):
        S_m = S.get(m, np.nan)
        if np.isnan(S_m):
            S_m = 0.0
        d_hat = 0.0
        if not leaf.policy_only and o["dev"]:
            o_all = dev.index[(dev.index >= TRAIN_START) & (dev.index <= T - pd.offsets.MonthBegin(h))]
            F = _features(dev.ffill(limit=2), up, o_all)
            tgt = dev.shift(-h).reindex(o_all)
            ok = F.notna().all(axis=1) & tgt.notna()
            if ok.sum() > 36:
                pred = _ridge(F[ok].values, tgt[ok].values, RIDGE * ok.sum() / 100)
                fo = _features(dev.ffill(limit=2), up, pd.DatetimeIndex([T]))
                if fo.notna().all(axis=1).iloc[0]:
                    d_hat = float(pred(fo.values)[0])
                res = tgt[ok].values - pred(F[ok].values)
                info.setdefault("rmse", {})[h] = float(np.sqrt(np.mean(res ** 2)))
        g = S_m + d_hat
        info["S"][m], info["dev"][m] = S_m, d_hat
        out[m] = g
    g = pd.Series(out)
    g = g + policy.future_events(leaf.key, months, asof)
    lv = L.loc[T] * np.exp(g.cumsum())
    return lv, info
