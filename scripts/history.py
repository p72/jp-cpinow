"""How the forecast for each target month evolved.

Two sources, kept apart on every chart (as in jp-gdpnow):
  record  — what the automatic runs actually said (vintage.runs, filled markers)
  recon   — a pseudo history: the engine re-run on past release dates (release.origins:
            after national CPI / Tokyo CPI / BOJ prices) with today's data cut by the
            release calendar (hollow markers). Reference only: it is not what was published
            at the time, and later data revisions leak in.
Recon results are cached in $JP_CPINOW_HOME/recon.csv by (date, model), so each run only
computes dates it has not seen.
"""
from __future__ import annotations

import os

import pandas as pd

from . import config, forecast, release, vintage


def update_dates(T_latest: pd.Timestamp, n_months: int = 6, until=None) -> list[pd.Timestamp]:
    """Release-driven nowcast dates for the last `n_months` national months up to T_latest."""
    until = pd.Timestamp(until or pd.Timestamp.now())
    out = []
    for k in range(n_months - 1, -1, -1):
        T = T_latest - pd.offsets.MonthBegin(k)
        out += [d for d in release.origins(T).values() if d <= until]
    return sorted(out)


def _cache_path() -> str:
    return os.path.join(vintage.home(), "recon.csv")


def reconstruct(ds, model: str, n_months: int = 6, until=None, verbose: bool = False) -> pd.DataFrame:
    """Rows: date, T, month, h, target, yoy (pseudo real-time, on `ds` = today's full data)."""
    path = _cache_path()
    cache = pd.read_csv(path, parse_dates=["date", "T", "month"]) if os.path.exists(path) else pd.DataFrame()
    if len(cache):
        cache = cache[cache["model"] == model]
    have = set(cache["date"]) if len(cache) else set()
    rows = []
    for d in update_dates(ds.T, n_months, until):
        if d in have:
            continue
        cut = ds.cut(d)
        try:
            fc = forecast.run(cut)
        except Exception as ex:  # one failing date should not stop the report
            print(f"[history] {d:%Y-%m-%d}: {ex}")
            continue
        for h, m in enumerate(fc.targets.index, start=1):
            for k in fc.targets.columns:
                rows.append(dict(date=d, T=fc.T, month=m, h=h, target=k,
                                 yoy=float(fc.targets.loc[m, k]), model=model))
        if verbose:
            print(f"[history] 再構成 {d:%Y-%m-%d}")
    new = pd.DataFrame(rows)
    allr = pd.concat([cache, new], ignore_index=True) if len(new) else cache
    if len(new):
        old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
        keep = old[old["model"] != model] if len(old) else old
        pd.concat([keep, allr.astype({"date": str, "T": str, "month": str})]).to_csv(path, index=False)
    return allr


def records(model: str | None = None) -> pd.DataFrame:
    """Actual logged runs, one per day and target month (the day's last run)."""
    try:
        r = vintage.runs()
    except Exception:
        return pd.DataFrame()
    if r.empty:
        return r
    if model:
        r = r[r["model"].str.split("-").str[0] == model]
    r = r.copy()
    r["date"] = pd.to_datetime(r["run_at"]).dt.normalize()
    r["month"] = pd.to_datetime(r["month"])
    r = r.sort_values("run_at").groupby(["date", "month", "target"], as_index=False).last()
    return r[["date", "month", "h", "target", "yoy"]]


def table(rec: pd.DataFrame, recon: pd.DataFrame, actual: pd.DataFrame | None, target: str = "core",
          months=None) -> pd.DataFrame:
    """Wide table for one series: rows = dates (record or recon), columns = target months,
    plus the published value as the last row when it is out."""
    parts = []
    for src, df in (("再構成", recon), ("記録", rec)):
        if df is None or df.empty:
            continue
        x = df[df["target"] == target].copy()
        x["種類"] = src
        parts.append(x[["date", "種類", "month", "yoy"]])
    if not parts:
        return pd.DataFrame()
    x = pd.concat(parts)
    if months is not None:
        x = x[x["month"].isin(pd.DatetimeIndex(months))]
    # a real record wins over a reconstruction on the same day
    x = x.sort_values("種類").drop_duplicates(["date", "month"], keep="last")
    w = x.pivot_table(index=["date", "種類"], columns="month", values="yoy", aggfunc="last").sort_index()
    w.columns = [f"{m:%Y-%m}" for m in w.columns]
    w.index = [f"{d:%Y-%m-%d}（{s}）" for d, s in w.index]
    if actual is not None:
        row = {c: actual[target].get(pd.Timestamp(c + "-01"), float("nan")) for c in w.columns}
        if any(v == v for v in row.values()):
            w.loc["公表値"] = pd.Series(row)
    return w


def short(key: str) -> str:
    return config.SHORT[key]
