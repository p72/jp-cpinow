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
    return s


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


def masked(leaf: str, index: pd.DatetimeIndex) -> pd.Series:
    """True where the leaf's history should not be used for training."""
    df = _raw()
    df = df[(df["kind"] == "mask") & ((df["leaf"] == leaf) | (df["leaf"] == ""))]
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
