"""News decomposition: why did the forecast change between two information sets?

Starting from the old data set, each source is swapped for its new version one at a
time (in release order) and the engine re-run; each step's change in the y/y
forecast is that source's "news". The steps add up exactly to the total revision
(sequential attribution, order-dependent). Policy-table updates are the last step.
"""
from __future__ import annotations

from dataclasses import replace

import pandas as pd

from . import forecast

STEPS = [  # (label, Dataset fields)
    ("全国CPI", ["nat"]),
    ("東京都区部CPI", ["tky"]),
    ("日銀 輸入物価・企業物価・為替", ["boj"]),
    ("ガソリン・灯油（週次）", ["weekly"]),
    ("野菜小売価格（週次）", ["veg"]),
    ("政策・制度の更新", ["asof"]),
]


def decompose(old, new, target: str = "core") -> pd.DataFrame:
    """old/new: Dataset (cut). Rows = sources, columns = target months of `new`."""
    base_fc = forecast.run(old)
    if base_fc.T != new.T:
        # a new national month changes the horizon itself: report only the total
        nf = forecast.run(new)
        return pd.DataFrame({"統計": ["全国CPI（新しい月）"], **{
            f"{m:%Y-%m}": [nf.targets.loc[m, target] - base_fc.targets[target].get(m, float("nan"))]
            for m in nf.targets.index}})
    months = base_fc.targets.index
    cur = old
    prev = base_fc.targets[target]
    rows = []
    for label, fields in STEPS:
        upd = {f: getattr(new, f) for f in fields}
        nxt = replace(cur, **upd)
        same = all(_same(getattr(cur, f), getattr(nxt, f)) for f in fields)
        if same:
            continue
        fc = forecast.run(nxt).targets[target]
        d = (fc - prev).reindex(months)
        if (d.abs() > 1e-9).any():
            rows.append({"統計": label, **{f"{m:%Y-%m}": d[m] for m in months}})
        cur, prev = nxt, fc
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame([{"統計": "（変化なし）", **{f"{m:%Y-%m}": 0.0 for m in months}}])
    total = {"統計": "合計", **{c: df[c].sum() for c in df.columns if c != "統計"}}
    return pd.concat([df, pd.DataFrame([total])], ignore_index=True)


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is b
    if isinstance(a, pd.DataFrame):
        return a.shape == b.shape and a.equals(b)
    return a == b
