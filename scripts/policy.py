"""Policy / institutional events (scripts/policy.csv).

Every row has an `announced` date. `load(asof)` keeps only rows that were public on
`asof`, so pseudo-real-time backtests never see later decisions (extensions, cuts).
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import pandas as pd

PATH = os.path.join(os.path.dirname(__file__), "policy.csv")


@lru_cache(maxsize=None)
def _raw() -> pd.DataFrame:
    df = pd.read_csv(PATH, comment="#", dtype=str).fillna("")
    for c in ("start", "end"):
        df[c] = pd.to_datetime(df[c], format="%Y-%m", errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["announced"] = pd.to_datetime(df["announced"], errors="coerce")
    return df


def load(asof=None, scenario: str = "base") -> pd.DataFrame:
    df = _raw()
    df = df[df["scenario"].isin(["base", scenario])]
    if asof is not None:
        a = pd.Timestamp(asof)
        df = df[df["announced"].isna() | (df["announced"] <= a)]
    return df


def _months(row) -> pd.DatetimeIndex:
    end = row["end"] if pd.notna(row["end"]) else pd.Timestamp("2035-12-01")
    return pd.date_range(row["start"], end, freq="MS")


def monthly_value(kind: str, index: pd.DatetimeIndex, asof=None, scenario="base") -> pd.Series:
    """Monthly series of a 'level' kind (subsidy_elec / subsidy_gas / levy / cap); 0 or NaN where none."""
    df = load(asof, scenario)
    df = df[df["kind"] == kind]
    fill = np.nan if kind == "cap" else 0.0
    s = pd.Series(fill, index=index, dtype=float)
    for _, r in df.iterrows():
        m = _months(r)
        s.loc[s.index.isin(m)] = r["value"]
    if kind == "levy" and not df.empty:   # next fiscal year not announced yet: hold the last rate
        last = df.loc[df["end"].idxmax()]
        s.loc[s.index > last["end"]] = last["value"]
    return s


def assumptions(months, asof=None) -> list[str]:
    """Plain-language policy assumptions behind a forecast for `months` (CPI months).
    Shown on the report and in README so that a missing policy update is visible."""
    months = pd.DatetimeIndex(months)
    lo, hi = months[0], months[-1]
    df = load(asof)
    out = []
    for kind, name in [("subsidy_elec", "電気・ガス料金支援")]:
        rows = df[df["kind"] == kind]
        if rows.empty:
            continue
        last_end = rows["end"].max()
        active = [m for m in months if monthly_value(kind, months, asof).get(m, 0) > 0]
        if last_end >= hi:
            out.append(f"{name}：予測期間中は発表済みの値引き単価を反映")
        elif last_end >= lo - pd.offsets.MonthBegin(1):
            stop = last_end + pd.offsets.MonthBegin(1)
            span = f"{lo:%-m}月分" if last_end == lo else f"{lo:%-m}〜{last_end:%-m}月分"
            had = f"{span}は反映、" if active else ""
            out.append(f"{name}：{had}{stop:%Y年%-m月}分（{(stop - pd.offsets.MonthBegin(1)):%-m}月使用分）以降は"
                       "未発表のため値引きなしで予測（延長されれば電気・ガス代は下振れ）")
        else:
            out.append(f"{name}：予測期間中の支援は未発表（値引きなしで予測）")
    caps = df[df["kind"] == "cap"]
    for _, r in caps.iterrows():
        if pd.notna(r["end"]) and r["end"] < lo:
            continue
        if pd.isna(r["end"]) or r["end"] >= hi:
            out.append(f"ガソリン：{r['value']:.0f}円程度の上限目安が予測期間中も続くと仮定（終了時期は未定）")
        else:
            out.append(f"ガソリン：上限目安は{r['end']:%Y年%-m月}まで、その後は市場価格で予測")
    yrs = sorted({m.year for m in months if m.month == 5})
    for y in yrs:
        lev = df[(df["kind"] == "levy") & (df["start"] == pd.Timestamp(y, 5, 1))]
        if lev.empty:
            out.append(f"再エネ賦課金：{y}年度の単価が未発表（前年度据え置きで予測）")
    return out


def events(leaf: str, asof=None, scenario="base") -> pd.DataFrame:
    """Event rows for a leaf (rows with empty leaf apply to every leaf)."""
    df = load(asof, scenario)
    df = df[(df["kind"] == "event") & ((df["leaf"] == leaf) | (df["leaf"] == ""))]
    return df


def event_months(leaf: str, asof=None) -> set:
    out = set()
    for _, r in events(leaf, asof).iterrows():
        out |= set(_months(r))
    return out


def masked(leaf: str, index: pd.DatetimeIndex, covid: bool = True) -> pd.Series:
    """True where the leaf's history should not be used for training.
    covid=False ignores the `mask` rows (COVID / travel-subsidy windows) and keeps
    only the policy-event months masked."""
    df = _raw()
    df = df[(df["kind"] == "mask") & ((df["leaf"] == leaf) | (df["leaf"] == "")) & covid]
    m = pd.Series(False, index=index)
    for _, r in df.iterrows():
        m |= m.index.isin(_months(r))
    m |= m.index.isin(event_months(leaf))
    return m


def future_events(leaf: str, months, asof=None, scenario="base") -> pd.Series:
    """Known impacts (log-points) on future months; NaN-valued events count as 0."""
    s = pd.Series(0.0, index=pd.DatetimeIndex(months))
    for _, r in events(leaf, asof, scenario).iterrows():
        if np.isnan(r["value"]):
            continue
        for m in _months(r):
            if m in s.index:
                s[m] += np.log1p(r["value"] / 100)
    return s
