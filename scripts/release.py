"""Release calendar: what was public on a given date (pseudo-real-time cut).

Typical publication days (good enough to reproduce the information set):
  全国CPI          month m → around the 20th of m+1 (Friday)
  東京都区部CPI    month m → around the 26th of m (中旬速報, last Friday)
  日銀 輸入物価等  month m → around the 11th of m+1
  ドル円 月中平均  month m → 1st of m+1
  石油製品小売市況（週次） survey date d → d + 2 days (Wednesday)
"""
from __future__ import annotations

import os

import pandas as pd

NAT_DAY, TKY_DAY, BOJ_DAY = 20, 26, 11


def _avail_month(asof, lag_months: int, day: int) -> pd.Timestamp:
    """Latest reference month published on `asof` for a (lag, day) rule."""
    a = pd.Timestamp(asof)
    m = a.to_period("M").to_timestamp()
    lag = lag_months + (1 if a.day < day else 0)
    return m - pd.offsets.MonthBegin(lag) if lag else m


def last_nat(asof) -> pd.Timestamp:
    return _avail_month(asof, 1, NAT_DAY)


def last_tky(asof) -> pd.Timestamp:
    return _avail_month(asof, 0, TKY_DAY)


def last_boj(asof) -> pd.Timestamp:
    return _avail_month(asof, 1, BOJ_DAY)


def last_fx(asof) -> pd.Timestamp:
    return _avail_month(asof, 1, 1)


def origins(T: pd.Timestamp) -> dict:
    """Three nowcast points with the same latest national month T:
    a = right after national CPI(T), c = after Tokyo(T+1), b = after BOJ(T+1)."""
    T = pd.Timestamp(T)
    m1 = T + pd.offsets.MonthBegin(1)
    m2 = T + pd.offsets.MonthBegin(2)
    return {"a": m1 + pd.Timedelta(days=NAT_DAY),       # 21st of T+1
            "c": m1 + pd.Timedelta(days=TKY_DAY + 1),   # 28th of T+1
            "b": m2 + pd.Timedelta(days=BOJ_DAY)}       # 12th of T+2


def upcoming(asof, n: int = 8) -> pd.DataFrame:
    """Next releases after `asof` from scripts/schedule.csv (hand-maintained)."""
    path = os.path.join(os.path.dirname(__file__), "schedule.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, dtype=str).fillna("")
    df["date"] = pd.to_datetime(df["date"])
    return df[df["date"] > pd.Timestamp(asof)].sort_values("date").head(n)
