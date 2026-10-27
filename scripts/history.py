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


# ---------------------------------------------------------------- data/HISTORY.md
GROUP_LABELS = [  # (series prefix, label, weekly?)
    ("nat:", "全国CPI", False), ("tky:", "東京都区部CPI", False),
    ("boj:imp_", "輸入物価（日銀）", False), ("boj:cgpi_", "国内企業物価（日銀）", False),
    ("boj:usdjpy", "ドル円", False),
    ("wk:", "ガソリン・灯油（週次）", True), ("veg:", "野菜小売価格（週次）", True),
]


def describe_arrivals(arr: pd.DataFrame, keep_revisions: int = 6) -> str:
    """Compact Japanese description, e.g. '全国CPI 9月分、ガソリン・灯油（週次） 10/20調査'."""
    if arr is None or arr.empty:
        return "なし"
    groups: dict = {}
    for _, r in arr.iterrows():
        s = r["series"]
        if s.startswith("yoy:"):          # published y/y arrives with the national CPI
            continue
        hit = next(((lab, wk) for p, lab, wk in GROUP_LABELS if s.startswith(p)), (s, False))
        g = groups.setdefault(hit[0], {"weekly": hit[1], "new": set(), "rev": set()})
        g["new"].update(r["new"])
        g["rev"].update(r["revised"])
    parts, n_old = [], 0
    for lab, g in groups.items():
        new, rev = sorted(g["new"]), sorted(g["rev"] - g["new"])
        latest = max(new + rev) if (new or rev) else None
        recent = [d for d in rev if latest and pd.Timestamp(d) > pd.Timestamp(latest) - pd.DateOffset(months=keep_revisions)]
        n_old += len(rev) - len(recent)
        fmt = (lambda d: f"{pd.Timestamp(d):%-m/%-d}調査") if g["weekly"] else (lambda d: f"{pd.Timestamp(d).month}月")
        bits = []
        if new:
            bits.append("・".join(fmt(d) for d in new[-3:]) + ("" if g["weekly"] else "分"))
        if recent and not g["weekly"]:
            bits.append("・".join(fmt(d) for d in recent[-3:]) + "改定")
        if bits:
            parts.append(f"{lab} " + "／".join(bits))
    if n_old:
        parts.append(f"（ほか過去分の改定 {n_old}件）")
    return "、".join(parts) if parts else "なし"


def build_md(model: str, out_dir: str | None = None) -> str:
    """Write data/HISTORY.md (+ news_log.csv) from the vintage store; returns the md path."""
    out_dir = out_dir or vintage.home()
    con = vintage.connect()
    runs = vintage.runs(con=con)
    runs = runs[runs["model"].str.split("-").str[0] == model].copy()
    news = vintage.news_log(con=con)
    news.to_csv(os.path.join(out_dir, "news_log.csv"), index=False, encoding="utf-8-sig")
    keys = [k for k, _, _ in config.TARGETS]
    lines = ["# 予測の変化の記録（自動生成）", "",
             f"モデル {model} の記録（リアルタイム）。対象月ごと、新しい順。値は全国CPIの前年同月比（%）。",
             "「変化」はコアの前回の記録からの変化。「主な要因」はコアの予測の改定をデータの種類ごとに分けたもの（%pt）。",
             "「入ったデータ」は前回の記録から今回までにデータベースに入った新しい月・改定。",
             "記録の日時は自動実行の時刻で、GitHub の混雑により統計の公表日より遅れることがある。", ""]
    run_keys = sorted(runs["run_at"].unique())
    prev_of = {rk: (run_keys[i - 1] if i else None) for i, rk in enumerate(run_keys)}
    arr_cache: dict = {}
    for month in sorted(runs["month"].unique(), reverse=True):
        g = runs[runs["month"] == month]
        m = pd.Timestamp(month)
        lines += [f"## {m.year}年{m.month}月分", "",
                  "| 記録日時 | 総合 | **コア** | 変化 | コアコア | 欧米型コア | サービス | 主な要因（コアへの寄与） | 入ったデータ |",
                  "|---|---:|---:|---:|---:|---:|---:|---|---|"]
        rows, prev_core = [], None
        for rk, gr in g.groupby("run_at", sort=True):
            v = gr.set_index("target")["yoy"]
            key = pd.Timestamp(rk).isoformat(timespec="seconds")
            nw = news[(news["run_at"] == key) & (news["target"] == "core") & (news["month"] == f"{m:%Y-%m}")]
            nw = nw[nw["contrib"].abs() >= 0.005].sort_values("contrib", key=abs, ascending=False)
            p = prev_of[rk]
            if p is None:
                drivers, got = "初回", "初回（全データ取得）"
            else:
                drivers = "、".join(f"{i} {c:+.2f}" for i, c in zip(nw["item"], nw["contrib"])) or "変化なし"
                if p not in arr_cache:
                    arr_cache[p] = describe_arrivals(vintage.arrivals(p, rk, con=con))
                got = arr_cache[p]
            change = "—" if prev_core is None else f"{v.get('core') - prev_core:+.1f}"
            cells = [f"{v.get(k):+.1f}" if k in v else "" for k in keys]
            cells[1] = f"**{cells[1]}**"
            rows.append(f"| {pd.Timestamp(rk):%Y-%m-%d %H:%M} | {cells[0]} | {cells[1]} | {change} | "
                        f"{cells[2]} | {cells[3]} | {cells[4]} | {drivers} | {got} |")
            prev_core = v.get("core")
        lines += rows[::-1] + [""]
    con.close()
    path = os.path.join(out_dir, "HISTORY.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def backfill(model: str):
    """Compute the news decomposition for logged runs that have none (from the stored vintages)."""
    from . import data, news as nws
    con = vintage.connect()
    runs = vintage.runs(con=con)
    runs = runs[runs["model"].str.split("-").str[0] == model]
    have = set(vintage.news_log(con=con)["run_at"])

    def ds_at(t):
        fr = {p: vintage.load_asof(pd.Timestamp(t).date(), p, con=con)
              for p in ("nat:", "tky:", "boj:", "wk:", "veg:", "yoy:")}
        return data.from_frames(fr).cut(pd.Timestamp(t).normalize() + pd.Timedelta(hours=23))

    prev = None
    for rk in sorted(runs["run_at"].unique()):
        key = pd.Timestamp(rk).isoformat(timespec="seconds")
        if key not in have:
            if prev is None:
                vintage.log_news(key, model, "core", "初回", None, con=con)
            else:
                nw = nws.decompose(ds_at(prev), ds_at(rk), "core")
                vintage.log_news(key, model, "core", f"前回 {pd.Timestamp(prev).isoformat(timespec='seconds')}",
                                 nw, con=con)
            print(f"[history] news {key}")
        prev = rk
    con.close()
