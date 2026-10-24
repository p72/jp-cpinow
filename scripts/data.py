"""Collect every input into one `Dataset`, and cut it to what was public on a date."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from . import config, cpi, release, upstream


@dataclass
class Dataset:
    nat: pd.DataFrame          # national CPI, official codes (index levels)
    tky: pd.DataFrame          # Tokyo-ku CPI, official codes
    boj: pd.DataFrame          # upstream monthly (BOJ)
    weekly: pd.DataFrame       # enecho weekly retail prices
    veg: pd.DataFrame | None = None   # MAFF weekly vegetable retail prices (2021-)
    yoy: pd.DataFrame | None = None   # published y/y of the five targets (national)
    asof: pd.Timestamp | None = None

    # ------------------------------------------------------------ derived
    def leaves(self, area: str = "nat") -> pd.DataFrame:
        raw = self.nat if area == "nat" else self.tky
        return cpi.leaf_index(raw, area)

    @property
    def T(self) -> pd.Timestamp:
        return self.nat["0001"].dropna().index[-1]

    def cut(self, asof) -> "Dataset":
        a = pd.Timestamp(asof)
        nat = self.nat.loc[: release.last_nat(a)]
        tky = self.tky.loc[: release.last_tky(a)]
        b = self.boj.copy()
        b.loc[b.index > release.last_boj(a), [c for c in b.columns if c != "usdjpy"]] = np.nan
        if "usdjpy" in b:
            b.loc[b.index > release.last_fx(a), "usdjpy"] = np.nan
        w = self.weekly[self.weekly.index + pd.Timedelta(days=2) <= a]
        v = None if self.veg is None else self.veg[self.veg.index + pd.Timedelta(days=9) <= a]
        yy = None if self.yoy is None else self.yoy.loc[: release.last_nat(a)]
        return replace(self, nat=nat, tky=tky, boj=b.dropna(how="all"), weekly=w, veg=v, yoy=yy, asof=a)


def fetch(verbose: bool = True) -> Dataset:
    def log(m):
        if verbose:
            print(m)
    nat = cpi.fetch_raw(config.AREA_NAT)
    log(f"[data] 全国CPI {nat.index[-1]:%Y-%m}")
    try:
        yy = cpi.fetch_yoy(config.AREA_NAT)
    except Exception as ex:
        print(f"[data] 前年同月比（公表値） failed: {ex}")
        yy = None
    tky = cpi.fetch_raw(config.AREA_TKY)
    log(f"[data] 東京都区部CPI {tky.index[-1]:%Y-%m}")
    try:
        b = upstream.get_boj()
        log(f"[data] 日銀 {b.dropna(how='all').index[-1]:%Y-%m}")
    except Exception as ex:
        print(f"[data] BOJ failed: {ex}")
        b = pd.DataFrame()
    try:
        w = upstream.enecho_weekly()
        log(f"[data] 石油製品小売市況 {w.index[-1]:%Y-%m-%d}")
    except Exception as ex:
        print(f"[data] enecho failed: {ex}")
        w = pd.DataFrame(columns=["gasoline_weekly", "kerosene_weekly"])
    try:
        v = upstream.maff_veg_weekly()
        log(f"[data] 食品価格動向調査（野菜） {v.index[-1]:%Y-%m-%d}")
    except Exception as ex:
        print(f"[data] maff failed: {ex}")
        v = None
    return Dataset(nat=nat, tky=tky, boj=b, weekly=w, veg=v, yoy=yy)


def to_frames(ds: Dataset) -> dict:
    """Flat frames for the vintage store."""
    f = {"nat:": ds.nat, "tky:": ds.tky, "boj:": ds.boj, "wk:": ds.weekly}
    if ds.veg is not None:
        f["veg:"] = ds.veg
    if ds.yoy is not None:
        f["yoy:"] = ds.yoy
    return f


def from_frames(frames: dict) -> Dataset:
    v, yy = frames.get("veg:"), frames.get("yoy:")
    return Dataset(nat=frames["nat:"], tky=frames["tky:"], boj=frames["boj:"],
                   weekly=frames["wk:"], veg=v if v is not None and len(v) else None,
                   yoy=yy if yy is not None and len(yy) else None)
