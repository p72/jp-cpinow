"""Forecast engine: leaves → five published aggregates (index, y/y), h = 1..3."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config, cpi, e1, energy, fresh, policy, tokyo
from .data import Dataset

H = 3
ENSEMBLE: dict = {"B1": 0.25, "B2": 0.25, "B6": 0.25, "B9": 0.25}  # weight on E2 by block (SKILL.md)


@dataclass
class Forecast:
    T: pd.Timestamp                      # latest national month
    asof: pd.Timestamp | None
    leaf_levels: pd.DataFrame            # history + forecast months
    targets: pd.DataFrame                # rows = months T+1..T+3, cols = target keys (y/y %)
    levels: pd.DataFrame                 # target index levels (anchored)
    info: dict = field(default_factory=dict)

    def contributions(self, target: str = "headline") -> pd.DataFrame:
        """Leaf contributions (pt) to the target's y/y in each month."""
        ls = config.leaves_of(target)
        w = pd.Series({l.key: config.leaf_weight(l) for l in ls})
        L = self.leaf_levels[w.index]
        rows = {}
        for m in self.targets.index.union(self.leaf_levels.index[-15:]):
            p = m - pd.DateOffset(years=1)
            if p not in L.index or m not in L.index:
                continue
            base = (L.loc[p] * w).sum()
            rows[m] = (L.loc[m] - L.loc[p]) * w / base * 100
        return pd.DataFrame(rows).T


def run(ds: Dataset, scenario: str = "base", use_tokyo: bool = True, use_veg: bool = True,
        modules: dict | None = None, e2w=None, rollforward: bool = False) -> Forecast:
    """modules: optional overrides {leaf_key: callable} (used by ablations).
    e2w: weight on E2 (LightGBM) deviations — float, {block: w}, or None (= ENSEMBLE).
    rollforward: when T+1 was bridged (Tokyo / weekly vegetables), re-forecast T+2..T+3
    of E1 leaves from the bridged T+1 (its deviation then feeds the later months).
    Off by default: it made h=2/3 worse in both backtest periods (SKILL.md)."""
    T = ds.T
    asof = ds.asof or pd.Timestamp.today().normalize()
    months = pd.date_range(T + pd.offsets.MonthBegin(1), periods=H, freq="MS")
    Ln = ds.leaves("nat")
    Lt = ds.leaves("tky") if use_tokyo and len(ds.tky) else None
    fut = {}
    info = {"leaf": {}, "tokyo": {}}
    e2w = ENSEMBLE if e2w is None else e2w
    wmap = e2w if isinstance(e2w, dict) else {b: float(e2w) for b in config.BLOCKS}
    d2 = None
    if any(v > 0 for v in wmap.values()):
        try:
            from . import e2
            d2 = e2.forecast_dev(Ln, ds.boj, T, months)
        except Exception as ex:
            print(f"[e2] failed: {ex}")
    for leaf in config.LEAVES:
        s = Ln[leaf.key].loc[:T].dropna()
        if modules and leaf.key in modules:
            lv, inf = modules[leaf.key](leaf, s, ds, months, asof)
        elif leaf.key in energy.FUELS and len(ds.boj):
            lv, inf = energy.utility(leaf.key, s, ds.boj, months, asof)
        elif leaf.key in ("gasoline", "kerosene"):
            lv, inf = energy.oil(leaf.key, s, ds.weekly, months, asof)
        elif leaf.key == "propane" and len(ds.boj):
            lv, inf = energy.propane(s, ds.boj, months)
        else:
            lv, inf = e1.forecast_leaf(leaf, s, ds.boj, months, asof)
            inf["e1_path"] = True
            w2 = wmap.get(leaf.block, 0.0)
            if d2 is not None and leaf.key in d2 and w2 > 0:
                g = pd.Series({m: inf["S"][m] + (1 - w2) * inf["dev"][m] + w2 * d2[leaf.key][m]
                               for m in months})
                g = g + policy.future_events(leaf.key, months, asof)
                lv = s.iloc[-1] * np.exp(g.cumsum())
        if leaf.key == "fresh_veg" and ds.veg is not None and use_veg:
            g = np.log(pd.concat([s.iloc[[-1]], lv])).diff().iloc[1:]
            rm = inf.get("rmse", {}).get(1)
            g2, vinf = fresh.blend(s, ds.veg, g, rm)
            # Tokyo's figure, once out, beats the weekly survey: use the survey only before it
            if Lt is not None and leaf.key in Lt:
                tk = set(Lt[leaf.key].dropna().index)
                g2 = g2.where(~g2.index.isin(tk), g)
                vinf = {m: v for m, v in vinf.items() if m not in tk}
            if vinf:
                lv = s.iloc[-1] * np.exp(g2.cumsum())
                info["veg"] = vinf
                if rm:  # the bridge reduced the error of the blended months
                    inf = dict(inf, rmse={1: rm * np.sqrt(1 - max(v["w"] for v in vinf.values()))})
        # Tokyo bridge for months already published in Tokyo
        if Lt is not None:
            g = np.log(pd.concat([s.iloc[[-1]], lv])).diff().iloc[1:]
            changed = False
            for m in months:
                rm = inf.get("rmse", {}).get(1) if isinstance(inf.get("rmse"), dict) else inf.get("rmse")
                b = tokyo.bridge(leaf, s, Lt[leaf.key] if leaf.key in Lt else None, m, g[m], rm)
                if b is not None:
                    g[m] = b[0]
                    info["tokyo"][(leaf.key, m)] = {"w": b[1], "beta": b[2]}
                    changed = True
            if changed:
                lv = s.iloc[-1] * np.exp(g.cumsum())
        m1 = months[0]
        bridged = (leaf.key, m1) in info["tokyo"] or (leaf.key == "fresh_veg" and m1 in info.get("veg", {}))
        if rollforward and bridged and inf.get("e1_path") and e1.opts(leaf.key)["dev"]:
            s_ext = pd.concat([s, lv.loc[[m1]]])
            lv2, _ = e1.forecast_leaf(leaf, s_ext, ds.boj, months[1:], asof, horizons=(1, 2))
            lv = pd.concat([lv.loc[[m1]], lv2])
        fut[leaf.key] = lv
        info["leaf"][leaf.key] = inf
    LL = pd.concat([Ln.loc[:T], pd.DataFrame(fut)])
    levels, yy = {}, {}
    for key, code, _ in config.TARGETS:
        a = cpi.aggregate(LL, key, "nat", anchor=ds.nat[code].loc[:T])
        levels[key] = a
        yy[key] = cpi.yoy(a)
    levels = pd.DataFrame(levels)
    targets = pd.DataFrame(yy).loc[months]
    return Forecast(T=T, asof=ds.asof, leaf_levels=LL, targets=targets, levels=levels, info=info)
