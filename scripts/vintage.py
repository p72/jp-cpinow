"""Vintage store (SQLite): every value as it was published, plus a log of forecasts.

Location: $JP_CPINOW_HOME/vintages.sqlite (default ~/.jp-cpinow).

Tables
  obs(series, ref_date, vintage, value)  -- append-only, a row only when a value is
                                           new or revised (diff storage)
  runs(run_at, asof, T, target, month, h, yoy, model, components)
                                         -- one row per target × month per forecast run
  news(run_at, model, target, month, basis, item, contrib)
                                         -- news decomposition of each run vs the previous one
Series names carry a prefix per source ("nat:0001", "tky:0161", "boj:imp_crude", "wk:gasoline_weekly").
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

JST = ZoneInfo("Asia/Tokyo")


def now_jst() -> datetime:
    """Run timestamps and vintage dates are always Japan time (GitHub runs in UTC by default)."""
    return datetime.now(JST).replace(tzinfo=None)

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
CREATE TABLE IF NOT EXISTS news (
  run_at TEXT NOT NULL, model TEXT, target TEXT, month TEXT, basis TEXT, item TEXT, contrib REAL
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
    vintage = str(pd.Timestamp(vintage or now_jst()).date())
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


def log_run(fc, model_name: str, con: sqlite3.Connection | None = None) -> str:
    """Store the forecast; returns the run_at key (also used by log_news)."""
    own = con is None
    con = con or connect()
    now = now_jst().isoformat(timespec="seconds")
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
    return now


def log_news(run_at: str, model_name: str, target: str, basis: str, nw: pd.DataFrame | None,
             con: sqlite3.Connection | None = None):
    """nw: news.decompose output (rows = 統計 incl. 合計, columns = YYYY-MM months)."""
    own = con is None
    con = con or connect()
    rows = []
    if nw is None or nw.empty:
        rows.append((run_at, model_name, target, None, basis, "（内訳なし）", 0.0))
    else:
        for _, r in nw[nw["統計"] != "合計"].iterrows():
            for c in nw.columns:
                if c != "統計" and r[c] == r[c]:
                    rows.append((run_at, model_name, target, c, basis, r["統計"], float(r[c])))
    con.executemany("INSERT INTO news VALUES (?,?,?,?,?,?,?)", rows)
    con.commit()
    if own:
        con.close()


def news_log(con: sqlite3.Connection | None = None) -> pd.DataFrame:
    own = con is None
    con = con or connect()
    df = pd.read_sql("SELECT * FROM news ORDER BY run_at", con)
    if own:
        con.close()
    return df


def arrivals(prev_run_at, run_at, con: sqlite3.Connection | None = None) -> pd.DataFrame:
    """Values stored after the day of `prev_run_at` up to the day of `run_at`, per series:
    new reference dates vs revisions of existing ones. prev_run_at=None → everything."""
    own = con is None
    con = con or connect()
    hi = str(pd.Timestamp(run_at).date())
    lo = str(pd.Timestamp(prev_run_at).date()) if prev_run_at is not None else "0000-00-00"
    df = pd.read_sql("SELECT series, ref_date, vintage FROM obs WHERE vintage > ? AND vintage <= ?",
                     con, params=(lo, hi))
    out = []
    for s, g in df.groupby("series"):
        older = pd.read_sql("SELECT DISTINCT ref_date FROM obs WHERE series = ? AND vintage <= ?",
                            con, params=(s, lo))
        known = set(older["ref_date"])
        refs = sorted(set(g["ref_date"]))
        out.append(dict(series=s, new=[d for d in refs if d not in known],
                        revised=[d for d in refs if d in known], new_series=not known))
    if own:
        con.close()
    return pd.DataFrame(out, columns=["series", "new", "revised", "new_series"])


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
