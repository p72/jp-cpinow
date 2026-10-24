"""Upstream prices that lead CPI components.

- 日本銀行 企業物価指数（輸入物価・円ベース／国内企業物価）, ドル円（月中平均）
- 資源エネルギー庁 石油製品小売市況調査（週次, 1990-）: レギュラーガソリン・灯油の全国平均
"""
from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd

from .sources import boj, http_get

BOJ_SERIES = {  # name -> (db, code)
    "imp_crude": ("PR01", "PRCG20_2600550010"),    # 輸入物価 円ベース 原油
    "imp_lng": ("PR01", "PRCG20_2600550011"),      # 液化天然ガス
    "imp_coal": ("PR01", "PRCG20_2600550009"),     # 一般炭
    "imp_lpg": ("PR01", "PRCG20_2600550006"),      # 液化石油ガス
    "imp_kerosene": ("PR01", "PRCG20_2600550004"),
    "imp_food": ("PR01", "PRCG20_2600120001"),     # 飲食料品・食料用農水産物
    "imp_all": ("PR01", "PRCG20_2600000000"),      # 輸入物価 総平均（円ベース）
    "cgpi_food": ("PR01", "PRCG20_2200120001"),    # 国内企業物価 飲食料品
    "cgpi_all": ("PR01", "PRCG20_2200000000"),     # 国内企業物価 総平均
    "usdjpy": ("FM08", "FXERM09"),                 # ドル円 月中平均
}

ENECHO_PAGE = "https://www.enecho.meti.go.jp/statistics/petroleum_and_lpgas/pl007/results.html"
ENECHO_BASE = "https://www.enecho.meti.go.jp/statistics/petroleum_and_lpgas/pl007/"


def get_boj(start: str = "198001") -> pd.DataFrame:
    out = {}
    by_db: dict = {}
    for name, (db, code) in BOJ_SERIES.items():
        by_db.setdefault(db, []).append((name, code))
    for db, items in by_db.items():
        try:
            df = boj().get_series(db, ",".join(c for _, c in items), start_date=start)
        except Exception as ex:  # one bad code should not kill the rest
            print(f"[upstream] BOJ {db} failed: {ex}")
            continue
        for name, code in items:
            if code in df.columns:
                out[name] = pd.to_numeric(df[code], errors="coerce")
    df = pd.DataFrame(out)
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()
    return df.sort_index()


def enecho_weekly() -> pd.DataFrame:
    """Weekly national retail prices: regular gasoline (円/L), kerosene 店頭 (円/18L).
    Index = survey date (Monday)."""
    html = http_get(ENECHO_PAGE).decode("utf-8", "replace")
    m = re.findall(r'href="([^"]*/xlsx/\d{6}s5\.xlsx)"', html)
    if not m:
        raise RuntimeError("enecho: s5.xlsx link not found")
    url = m[0] if m[0].startswith("http") else "https://www.enecho.meti.go.jp" + m[0]
    raw = http_get(url)
    out = {}
    for sheet, name in [("レギュラー", "gasoline_weekly"), ("灯油店頭", "kerosene_weekly")]:
        df = pd.read_excel(io.BytesIO(raw), sheet_name=sheet, header=None)
        d = pd.to_datetime(df.iloc[1:, 1], errors="coerce")
        v = pd.to_numeric(df.iloc[1:, 2], errors="coerce")
        out[name] = pd.Series(v.values, index=d.values).dropna()
        out[name] = out[name][out[name].index.notna()]
    w = pd.DataFrame(out).sort_index()
    w.index.name = "date"
    return w


MAFF_VEG_PAGE = "https://www.maff.go.jp/j/zyukyu/anpo/kouri/k_yasai/h22index.html"
MAFF_VEG = {"キャベツ": 10, "ねぎ": 11, "レタス": 8, "ばれいしょ": 6, "たまねぎ": 12,
            "きゅうり": 10, "トマト": 22, "にんじん": 8, "はくさい": 5, "だいこん": 5}  # CPI weights


def _wareki(s: str):
    m = re.match(r"(令和|平成)(\d+|元)年(\d+)月(\d+)日", str(s))
    if not m:
        return pd.NaT
    y = 1 if m.group(2) == "元" else int(m.group(2))
    y += 2018 if m.group(1) == "令和" else 1988
    return pd.Timestamp(y, int(m.group(3)), int(m.group(4)))


def maff_veg_weekly() -> pd.DataFrame:
    """農水省 食品価格動向調査（野菜, 週次, 2021-04-）: 10品目の小売価格(円/kg) と
    CPIウエイトによる幾何平均の合成指数 `veg_index`."""
    html = http_get(MAFF_VEG_PAGE).decode("utf-8", "replace")
    m = re.findall(r'href="([^"]*attach/xls/[^"]+\.xlsx)"', html)
    if not m:
        raise RuntimeError("maff: xlsx link not found")
    url = m[0]
    if url.startswith("./"):
        url = MAFF_VEG_PAGE.rsplit("/", 1)[0] + url[1:]
    raw = pd.read_excel(io.BytesIO(http_get(url)), sheet_name="価格", header=None)
    cols = raw.iloc[1].tolist()
    body = raw.iloc[2:].copy()
    body.index = [_wareki(x) for x in body.iloc[:, 0]]
    body = body[body.index.notna()]
    df = pd.DataFrame({c: pd.to_numeric(body.iloc[:, j], errors="coerce")
                       for j, c in enumerate(cols) if c in MAFF_VEG})
    lw = pd.Series(MAFF_VEG)[df.columns]
    lg = np.log(df)
    # chain-linked geometric index so that missing items (はくさい・だいこん early on) do not break it
    d = lg.diff()
    wsum = d.notna().mul(lw).sum(axis=1)
    g = (d.fillna(0).mul(lw).sum(axis=1) / wsum.replace(0, np.nan)).fillna(0)
    df["veg_index"] = 100 * np.exp(g.cumsum())
    df.index.name = "date"
    return df.sort_index()


def weekly_to_cpi_month(w: pd.DataFrame, mode: str = "mid") -> pd.DataFrame:
    """Monthly value from weekly prices. CPI prices are collected in the week that
    contains the 12th ('mid'); 'mean' = average of all surveys in the month."""
    if mode == "mean":
        return w.groupby(w.index.to_period("M")).mean().to_timestamp()
    rows = {}
    for p, g in w.groupby(w.index.to_period("M")):
        target = pd.Timestamp(p.year, p.month, 12)
        i = (abs(g.index - target)).argmin()
        rows[p.to_timestamp()] = g.iloc[i]
    return pd.DataFrame(rows).T.sort_index()
