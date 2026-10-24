"""Energy leaves: structural models.

電気代・都市ガス代 — 燃料費（原料費）調整の再現:
  ΔE_t (index pts) = c + Σ_f β_f Δavg3(fuel_f)_{t-lag} + γ·Δ(再エネ賦課金 − 値引き単価)_t
  avg3 = 3-month average of BOJ import prices (yen); lag=3 → fuel months t-5..t-3 set
  the CPI of month t (CPI month = 検針月 = usage month + 1). γ≈1/(yen per index pt).
  Subsidies and the levy are explicit regressors, so the history needs no cleaning and
  announced future subsidies enter the forecast directly.
ガソリン・灯油 — 石油製品小売市況調査（週次）:
  observed months use the weekly survey (CPI gasoline ≈ monthly mean of surveys,
  kerosene ≈ survey nearest the 12th); later months hold the latest weekly price
  (a price cap in policy.csv pins it when the cap binds).
プロパンガス — AR(1) + 輸入LPG価格（円）のラグ.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import policy, upstream

FUELS = {"electricity": ["imp_crude", "imp_lng", "imp_coal"],
         "city_gas": ["imp_lng", "imp_crude"]}
SUBSIDY = {"electricity": "subsidy_elec", "city_gas": "subsidy_gas"}
LAG = 3


def _ols(y, X):
    A = np.c_[np.ones(len(X)), X]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


def _fuel_design(boj: pd.DataFrame, fuels, idx) -> pd.DataFrame:
    return pd.DataFrame({f: boj[f].rolling(3).mean().shift(LAG).diff()
                         for f in fuels}).reindex(idx)


def _extend_boj(boj: pd.DataFrame, until: pd.Timestamp, fuels) -> pd.DataFrame:
    """Missing recent fuel months: hold the latest value (only needed for h beyond
    what the fuel lag already pins)."""
    idx = pd.date_range(boj.index[0], max(until, boj.index[-1]), freq="MS")
    b = boj.reindex(idx)
    for f in fuels:
        b[f] = b[f].ffill()
    return b


def utility(leaf: str, L: pd.Series, boj: pd.DataFrame, months, asof, start="2010-01-01"):
    """Forecast index levels of electricity / city gas for `months` (after L's last obs)."""
    fuels = FUELS[leaf]
    months = pd.DatetimeIndex(months)
    idx = pd.date_range(L.index[0], months[-1], freq="MS")
    b = _extend_boj(boj, months[-1], fuels)
    X = _fuel_design(b, fuels, idx)
    pol = -policy.monthly_value(SUBSIDY[leaf], idx, asof)
    if leaf == "electricity":
        pol = pol + policy.monthly_value("levy", idx, asof)
    X["pol"] = pol.diff()
    y = L.reindex(idx).diff()
    ev = policy.event_months(leaf, asof)
    tr = pd.concat([y.rename("y"), X], axis=1).loc[start:L.index[-1]].dropna()
    tr = tr[~tr.index.isin(ev)]
    coef = _ols(tr["y"].values, tr.drop(columns="y").values)
    resid = tr["y"].values - np.c_[np.ones(len(tr)), tr.drop(columns="y").values] @ coef
    lvl = L.iloc[-1]
    out = {}
    for m in months:
        x = X.loc[m].fillna(0).values
        lvl = lvl + coef[0] + x @ coef[1:]
        out[m] = lvl
    info = dict(coef=dict(zip(["const"] + list(X.columns), coef)), rmse=float(np.sqrt(np.mean(resid ** 2))))
    return pd.Series(out), info


def oil(leaf: str, L: pd.Series, weekly: pd.DataFrame, months, asof):
    """Gasoline / kerosene from the weekly retail survey."""
    col = {"gasoline": "gasoline_weekly", "kerosene": "kerosene_weekly"}[leaf]
    mode = "mean" if leaf == "gasoline" else "mid"
    months = pd.DatetimeIndex(months)
    w = weekly[col].dropna()
    if w.empty:
        return pd.Series(L.iloc[-1], index=months), {}
    # calibrate CPI mom on weekly mom (full months only)
    wm = upstream.weekly_to_cpi_month(w.to_frame(), mode)[col]
    d = pd.concat([np.log(L).diff().rename("y"), np.log(wm).diff().rename("x")], axis=1)
    d = d.loc["2010":L.index[-1]].dropna()
    slope = float((d["x"] * d["y"]).sum() / (d["x"] ** 2).sum()) if len(d) > 24 else 1.0
    last_obs = w.index[-1]
    cap = policy.monthly_value("cap", months, asof) if leaf == "gasoline" else None
    lvl = L.iloc[-1]
    prev_w = wm.get(L.index[-1], np.nan)
    out = {}
    for m in months:
        g = w[(w.index >= m) & (w.index < m + pd.offsets.MonthBegin(1))]
        if len(g):
            cur = g.mean() if mode == "mean" else g.iloc[(abs(g.index - (m + pd.Timedelta(days=11)))).argmin()]
            if len(g) < 3 and m == months[0] and mode == "mean":
                cur = 0.5 * cur + 0.5 * w.iloc[-1]
        else:
            cur = w.iloc[-1]
        if cap is not None and not np.isnan(cap.get(m, np.nan)):
            cur = min(cur, cap[m]) if cur > cap[m] * 1.03 else cur
        if np.isnan(prev_w):
            prev_w = cur
        lvl = lvl * np.exp(slope * np.log(cur / prev_w))
        prev_w = cur
        out[m] = lvl
    return pd.Series(out), dict(slope=slope, last_weekly=float(w.iloc[-1]), last_date=str(last_obs.date()))


def propane(L: pd.Series, boj: pd.DataFrame, months, start="2005-01-01"):
    months = pd.DatetimeIndex(months)
    idx = pd.date_range(L.index[0], months[-1], freq="MS")
    y = np.log(L).diff().reindex(idx)
    lp = np.log(boj["imp_lpg"]).diff().reindex(idx)
    feats = pd.DataFrame({"y1": y.shift(1), "l2": lp.shift(2), "l3": lp.shift(3), "l4": lp.shift(4)})
    tr = pd.concat([y.rename("y"), feats], axis=1).loc[start:L.index[-1]].dropna()
    tr = tr[~tr.index.isin(policy.event_months("propane"))]
    coef = _ols(tr["y"].values, tr.drop(columns="y").values)
    lvl, ylast = L.iloc[-1], y.loc[L.index[-1]]
    out = {}
    for m in months:
        x = feats.loc[m].copy()
        x["y1"] = ylast
        x = x.fillna(0).values
        g = coef[0] + x @ coef[1:]
        lvl *= np.exp(g)
        ylast = g
        out[m] = lvl
    return pd.Series(out), {}
