"""Partition of the CPI basket into leaves, blocks and the five published aggregates.

Every leaf is either an official CPI class/item (`code`) or a residual
(`parent` minus `minus` codes). Leaves do not overlap and add up to 総合, so the
five targets are exact weighted sums of leaves and can never contradict each other.

Weights are the 2025-base 1万分比 weights (scripts/weights.csv, from the e-Stat
statistical tables). Residual weights are parent − Σ children.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd

ESTAT_CPI = "0004052037"          # 消費者物価指数（2025年基準）, connected back to 1970
AREA_NAT, AREA_TKY = "00000", "13100"

# five targets: (key, official code, label)
TARGETS = [
    ("headline", "0001", "総合"),
    ("core", "0161", "コア（生鮮食品を除く総合）"),
    ("corecore", "0178", "コアコア（生鮮食品及びエネルギーを除く総合）"),
    ("western", "0168", "欧米型コア（食料（酒類を除く）及びエネルギーを除く総合）"),
    ("services", "0220", "サービス"),
]
SHORT = {"headline": "総合", "core": "コア", "corecore": "コアコア",
         "western": "欧米型コア", "services": "サービス"}

BLOCKS = {
    "B1": "生鮮食品", "B2": "食料（生鮮・外食・酒類を除く）", "B3": "外食", "B4": "酒類",
    "B5": "エネルギー", "B6": "財（食料・エネルギーを除く）", "B7": "家賃",
    "B8": "公共サービス（外食・家賃を除く）", "B9": "一般サービス（外食・家賃を除く）",
}
# which blocks each target contains
TARGET_BLOCKS = {
    "headline": list(BLOCKS),
    "core": ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9"],
    "corecore": ["B2", "B3", "B4", "B6", "B7", "B8", "B9"],
    "western": ["B4", "B6", "B7", "B8", "B9"],
    "services": ["B3", "B7", "B8", "B9"],
}


@dataclass(frozen=True)
class Leaf:
    key: str
    label: str
    block: str
    code: str | None = None             # official code (leaf = that series)
    plus: tuple = ()                     # residual: Σ plus − Σ minus (codes)
    minus: tuple = ()
    kind: str = "default"                # forecasting module hint
    policy_only: bool = False            # deterministic (level held + policy steps)


LEAVES = [
    # B1 生鮮食品
    Leaf("fresh_veg", "生鮮野菜", "B1", code="0159", kind="fresh"),
    Leaf("fresh_fish", "生鮮魚介", "B1", code="0158", kind="fresh"),
    Leaf("fresh_fruit", "生鮮果物", "B1", plus=("0157",), minus=("0159", "0158"), kind="fresh"),
    # B2 食料（生鮮・外食・酒類を除く）
    Leaf("rice", "米類", "B2", code="0004", kind="food"),
    Leaf("food_other", "生鮮・外食・酒類・米類を除く食料", "B2",
         plus=("0002",), minus=("0157", "0042", "0041", "0004"), kind="food"),
    # B3 外食
    Leaf("dining", "一般外食", "B3", code="0043", kind="dining"),
    Leaf("school_lunch", "学校給食", "B3", code="0044", policy_only=True),
    # B4 酒類
    Leaf("alcohol", "酒類", "B4", code="0041"),
    # B5 エネルギー
    Leaf("electricity", "電気代", "B5", code="3500", kind="electricity"),
    Leaf("city_gas", "都市ガス代", "B5", code="3600", kind="city_gas"),
    Leaf("propane", "プロパンガス", "B5", code="3612", kind="propane"),
    Leaf("kerosene", "灯油", "B5", code="3701", kind="kerosene"),
    Leaf("gasoline", "ガソリン", "B5", code="7301", kind="gasoline"),
    # B6 財（食料・エネルギーを除く） = 財 − (食料 − 外食) − エネルギー
    Leaf("water", "上下水道料", "B6", code="0059", policy_only=True),
    Leaf("tobacco", "たばこ", "B6", code="0155", policy_only=True),
    Leaf("handset", "携帯電話機", "B6", code="7446"),
    Leaf("goods_other", "その他の財", "B6", plus=("0202", "0042"),
         minus=("0002", "0167", "0059", "0155", "7446"), kind="goods"),
    # B7 家賃（公営家賃を含む）
    Leaf("rent_private", "民営家賃", "B7", code="0047", kind="rent"),
    Leaf("rent_imputed", "持家の帰属家賃", "B7", code="0050", kind="rent"),
    Leaf("rent_public", "公営・都市再生機構・公社家賃", "B7", code="0048", policy_only=True),
    # B8 公共サービス（学校給食・公営家賃を除く）
    Leaf("public_svc", "公共サービス（外食・家賃を除く）", "B8",
         plus=("0221",), minus=("0243", "0222"), policy_only=True),
    # B9 一般サービス（他のサービス）
    Leaf("lodging", "宿泊料", "B9", code="9300", kind="travel"),
    Leaf("package_tour", "外国パック旅行費", "B9", code="9305", kind="travel"),
    Leaf("mobile_fee", "通信料（携帯電話）", "B9", code="7430"),
    Leaf("svc_other", "その他の一般サービス", "B9", plus=("0232",),
         minus=("9300", "9305", "7430"), kind="services"),
]
LEAF = {l.key: l for l in LEAVES}

# official aggregates used to check that the leaves reproduce them
CHECK_CODES = {"headline": "0001", "core": "0161", "corecore": "0178",
               "western": "0168", "services": "0220"}


def all_codes() -> list[str]:
    s = {c for _, c, _ in TARGETS} | set(CHECK_CODES.values())
    for l in LEAVES:
        if l.code:
            s.add(l.code)
        s |= set(l.plus) | set(l.minus)
    return sorted(s)


@lru_cache(maxsize=None)
def weights(area: str = "nat") -> pd.Series:
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "weights.csv"), dtype={"code": str})
    return df.set_index("code")[f"w_{area}"].astype(float)


def leaf_weight(leaf: Leaf, area: str = "nat") -> float:
    w = weights(area)
    if leaf.code:
        return float(w[leaf.code])
    return float(sum(w[c] for c in leaf.plus) - sum(w[c] for c in leaf.minus))


def leaves_of(target: str) -> list[Leaf]:
    bl = set(TARGET_BLOCKS[target])
    return [l for l in LEAVES if l.block in bl]
