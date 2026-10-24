"""Vintage store (SQLite): every value as it was published, plus a log of forecasts.

Location: $JP_CPINOW_HOME/vintages.sqlite (default ~/.jp-cpinow).

Tables
  obs(series, ref_date, vintage, value)  -- append-only, a row only when a value is
                                           new or revised (diff storage)
  runs(run_at, asof, T, target, month, h, yoy, model, components)
                                         -- one row per target × month per forecast run
Series names carry a prefix per source ("nat:0001", "tky:0161", "boj:imp_crude", "wk:gasoline_weekly").
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import date, datetime

import numpy as np
import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS obs (
  series   TEXT NOT NULL,
  ref_date TEXT NOT NULL,
  vintage  TEXT NOT NULL,
  value    REAL,
  PRIMARY KEY (series, ref_date, vintage)
);
CREATE INDEX IF NOT EXISTS obs_series_vintage ON obs(series, vintage);
CREATE TABLE IF NOT EXISTS runs (
  run_at TEXT NOT NULL, asof TEXT, T TEXT, target TEXT, month TEXT, h INTEGER,
  yoy REAL, model TEXT, components TEXT
);
"""


def home() -> str:
    d = os.environ.get("JP_CPINOW_HOME", os.path.expanduser("~/.jp-cpinow"))
    os.makedirs(d, exist_ok=True)
    return d


def connect(path: str | None = None) -> sqlite3.Connection:
    con = sqlite3.connect(path or os.path.join(home(), "vintages.sqlite"))
    con.executescript(SCHEMA)
    return con


def _latest(con, series: str, asof: str) -> dict:
    q = """SELECT ref_date, value FROM obs o WHERE series=? AND vintage = (
             SELECT MAX(vintage) FROM obs i WHERE i.series=o.series
             AND i.ref_date=o.ref_date AND i.vintage<=?)"""
    return dict(con.execute(q, (series, asof)).fetchall())


def save_snapshot(df: pd.DataFrame, vintage=None, prefix: str = "",
                  con: sqlite3.Connection | None = None, rtol: float = 1e-9) -> int:
    """Store the columns of `df` (index = ref dates) as published on `vintage`
    (default today). Only new or changed values are written. Returns rows added."""
    vintage = str(pd.Timestamp(vintage or date.today()).date())
    own = con is None
    con = con or connect()
    added = 0
    for col in df.columns:
        s = df[col].dropna()
        name = prefix + str(col)
        prev = _latest(con, name, vintage)
        rows = []
        for ref, v in s.items():
            key = str(pd.Timestamp(ref).date())
            old = prev.get(key)
            if old is None or not np.isclose(old, v, rtol=rtol, atol=0):
                rows.append((name, key, vintage, float(v)))
        con.executemany("INSERT OR REPLACE INTO obs VALUES (?,?,?,?)", rows)
        added += len(rows)
    con.commit()
    if own:
        con.close()
    return added


def load_asof(asof, prefix: str, con: sqlite3.Connection | None = None) -> pd.DataFrame:
    """Wide DataFrame of the `prefix` series as they stood on `asof`."""
    asof = str(pd.Timestamp(asof).date())
    own = con is None
    con = con or connect()
    names = [r[0] for r in con.execute(
        "SELECT DISTINCT series FROM obs WHERE series LIKE ?", (prefix + "%",))]
    out = {}
    for name in names:
        d = _latest(con, name, asof)
        if d:
            out[name[len(prefix):]] = pd.Series(d, dtype=float)
    if own:
        con.close()
    df = pd.DataFrame(out)
    if len(df):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def log_run(fc, model_name: str, con: sqlite3.Connection | None = None):
    own = con is None
    con = con or connect()
    now = datetime.now().isoformat(timespec="seconds")
    comp = fc.contributions("headline")
    rows = []
    for h, m in enumerate(fc.targets.index, start=1):
        c = json.dumps({k: round(float(v), 3) for k, v in comp.loc[m].items()}) if m in comp.index else ""
        for k in fc.targets.columns:
            rows.append((now, str(fc.asof.date()) if fc.asof is not None else None, str(fc.T.date()),
                         k, str(m.date()), h, float(fc.targets.loc[m, k]), model_name,
                         c if k == "headline" else ""))
    con.executemany("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?)", rows)
    con.commit()
    if own:
        con.close()


def runs(con: sqlite3.Connection | None = None) -> pd.DataFrame:
    own = con is None
    con = con or connect()
    df = pd.read_sql("SELECT * FROM runs ORDER BY run_at", con)
    if own:
        con.close()
    df["run_at"] = pd.to_datetime(df["run_at"])
    return df


def last_run_before(before, con: sqlite3.Connection | None = None):
    r = runs(con)
    if r.empty:
        return None
    r = r[r["run_at"].dt.normalize() < pd.Timestamp(before).normalize()]
    return None if r.empty else r["run_at"].iloc[-1]
