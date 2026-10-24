"""E2: one global gradient-boosting model across leaves (LightGBM), direct per horizon.

Target: dev_{o+h} = log m/m − 5-year seasonal median, for every non-policy, non-energy
leaf. Pooling the leaves gives many more rows than any single series. Features:
leaf id, calendar month of the target, the seasonal median itself, recent deviations
(1, 3, 12 months), y/y of the leaf, and upstream changes (import/CGPI food, all
items, USD/JPY). Clean history only (policy.masked months dropped).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import config, e1, policy

LEAVES = [l.key for l in config.LEAVES
          if not l.policy_only and l.block not in ("B5",)]
UP = ["imp_food", "cgpi_food", "imp_all", "cgpi_all", "usdjpy"]
PARAMS = dict(objective="l2", learning_rate=0.03, num_leaves=15, min_data_in_leaf=30,
              bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8, lambda_l2=1.0,
              verbose=-1, seed=7, num_threads=2)
ROUNDS = 300


def _panel(L: pd.DataFrame, boj: pd.DataFrame, end: pd.Timestamp):
    idx = pd.date_range(L.index[0], end, freq="MS")
    up = None
    if boj is not None and len(boj):
        up = boj[[c for c in UP if c in boj]].reindex(idx).ffill()
        up = pd.DataFrame({f"{c}_{k}": np.log(up[c]) - np.log(up[c]).shift(k)
                           for c in up.columns for k in (3, 12)})
    out = {}
    for key in LEAVES:
        if key not in L:
            continue
        y = np.log(L[key]).diff().reindex(idx)
        mask = policy.masked(key, idx)
        S = e1.seasonal(y, mask)
        dev = (y - S).where(~mask)
        dv = dev.ffill(limit=2)
        f = pd.DataFrame({"dev1": dv, "dev3": dv.rolling(3, min_periods=2).mean(),
                          "dev12": dv.rolling(12, min_periods=8).mean(),
                          "yoy": np.log(L[key]).reindex(idx).diff(12)}, index=idx)
        if up is not None:
            f = f.join(up)
        out[key] = (f, dev, S)
    return out


def forecast_dev(L: pd.DataFrame, boj: pd.DataFrame, T: pd.Timestamp, months, horizons=(1, 2, 3)):
    """Returns {leaf: {month: dev_hat}}."""
    import lightgbm as lgb

    months = pd.DatetimeIndex(months)
    P = _panel(L.loc[:T], boj, months[-1])
    res = {k: {} for k in P}
    cats = {k: i for i, k in enumerate(LEAVES)}
    for h, m in zip(horizons, months):
        Xs, ys = [], []
        for k, (f, dev, S) in P.items():
            o = f.index[(f.index >= e1.TRAIN_START) & (f.index <= T - pd.offsets.MonthBegin(h))]
            X = f.loc[o].copy()
            tgt_m = o + pd.offsets.MonthBegin(h)
            X["S"] = S.reindex(tgt_m).values
            X["month"] = tgt_m.month
            X["leaf"] = cats[k]
            yv = dev.reindex(tgt_m).values
            ok = ~np.isnan(yv)
            Xs.append(X[ok])
            ys.append(yv[ok])
        X = pd.concat(Xs)
        y = np.concatenate(ys)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dset = lgb.Dataset(X, y, categorical_feature=["leaf", "month"], free_raw_data=False)
            model = lgb.train(PARAMS, dset, num_boost_round=ROUNDS)
        for k, (f, dev, S) in P.items():
            x = f.loc[[T]].copy()
            x["S"] = S.get(m, np.nan)
            x["month"] = m.month
            x["leaf"] = cats[k]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res[k][m] = float(model.predict(x)[0])
    return res
