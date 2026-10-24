"""Command line entry.

python -m scripts.cli forecast [--out DIR] [--only-if-new] [--no-save] [--asof YYYY-MM-DD]
python -m scripts.cli backtest [--start 2015-01-01] [--out DIR]
python -m scripts.cli bands                  # rebuild scripts/error_bands.csv (slow)
python -m scripts.cli history

Every `forecast` run (without --asof / --no-save) stores today's data snapshot and the
result in $JP_CPINOW_HOME/vintages.sqlite (default ~/.jp-cpinow).
"""
from __future__ import annotations

import argparse
import os
import shutil
from datetime import date

import numpy as np
import pandas as pd

from . import backtest, bands, config, cpi, data, forecast, news, plots, vintage

MODEL = "v1"


def _load_vintage(asof) -> data.Dataset:
    frames = {p: vintage.load_asof(asof, p) for p in ("nat:", "tky:", "boj:", "wk:", "veg:", "yoy:")}
    if frames["nat:"].empty:
        raise SystemExit(f"no stored vintage on or before {asof}")
    return data.from_frames(frames).cut(pd.Timestamp(asof) + pd.Timedelta(hours=23))


def _inputs(ds: data.Dataset, fc) -> list:
    rows = []
    t = ds.tky
    if len(t) and "0161" in t:
        ty = cpi.yoy(t["0161"]).dropna()
        rows.append((f"東京都区部 コア（{ty.index[-1].month}月）", f"{ty.iloc[-1]:+.1f}%"))
    el = fc.leaf_levels["electricity"]
    g = (el / el.shift(1) - 1).loc[fc.targets.index] * 100
    rows.append(("電気代 予測（前月比）", "　".join(f"{m.month}月 {v:+.1f}%" for m, v in g.items())))
    if len(ds.weekly):
        w = ds.weekly["gasoline_weekly"].dropna()
        rows.append(("レギュラーガソリン（週次）", f"{w.iloc[-1]:.1f}円/L（{w.index[-1]:%-m/%-d}）"))
    if ds.veg is not None and len(ds.veg):
        v = ds.veg["veg_index"].dropna()
        mm = v.groupby(v.index.to_period("M")).mean()
        if len(mm) > 1:
            rows.append((f"野菜小売価格（{mm.index[-1].month}月, 週次平均）", f"前月比 {(mm.iloc[-1] / mm.iloc[-2] - 1) * 100:+.1f}%"))
    b = ds.boj
    if "usdjpy" in b:
        u = b["usdjpy"].dropna()
        rows.append((f"ドル円（{u.index[-1].month}月平均）", f"{u.iloc[-1]:.1f}円"))
    for c, lab in [("imp_crude", "輸入物価 原油（円）"), ("imp_lng", "輸入物価 LNG（円）"), ("imp_food", "輸入物価 飲食料品（円）")]:
        if c in b:
            s = b[c].dropna()
            rows.append((f"{lab}（{s.index[-1].month}月）", f"前年比 {(s.iloc[-1] / s.iloc[-13] - 1) * 100:+.1f}%"))
    return rows


def _result_table(fc, point: str, n_hist: int = 3, text: bool = True) -> pd.DataFrame:
    """Rows = the last `n_hist` published months (実績) + T+1..T+3 (予測); columns = 5 series.
    text=True: printable strings with 90% intervals; False: numbers + 区分/90%区間 columns."""
    act = fc.recent(n_hist)
    rows = []
    for m in act.index:
        r = {"月": f"{m:%Y-%m}", "区分": "実績"}
        for k in fc.targets.columns:
            r[config.SHORT[k]] = act.loc[m, k]
        rows.append(r)
    for h, m in enumerate(fc.targets.index, start=1):
        r = {"月": f"{m:%Y-%m}", "区分": f"予測 h={h}"}
        for k in fc.targets.columns:
            v = fc.targets.loc[m, k]
            lo, hi = bands.interval(k, h, point, v)
            if text:
                r[config.SHORT[k]] = f"{v:+.1f} [{lo:+.1f}, {hi:+.1f}]"
            else:
                r[config.SHORT[k]] = v
                r[f"{config.SHORT[k]}_90%下限"] = round(lo, 2)
                r[f"{config.SHORT[k]}_90%上限"] = round(hi, 2)
        rows.append(r)
    df = pd.DataFrame(rows).set_index(["月", "区分"])
    if text:
        df = df.apply(lambda c: [v if isinstance(v, str) else f"{v:+.1f}" for v in c])
    return df


README = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md")
MARK_START, MARK_END = "<!-- forecast:start -->", "<!-- forecast:end -->"


def readme_table(fc, point: str, asof) -> str:
    """Markdown block for README: rows = 5 series, columns = 3 published months + h=1..3."""
    act = fc.recent(3)
    head = ["前年同月比（%）"] + [f"{m.year}年{m.month}月<br>実績" for m in act.index] \
        + [f"**{m.year}年{m.month}月**<br>予測 h={h}" for h, m in enumerate(fc.targets.index, start=1)]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" + "---:|" * (len(head) - 1)]
    for k, _, label in config.TARGETS:
        cells = [config.SHORT[k] if k != "core" else f"**{config.SHORT[k]}**"]
        cells += [f"{act.loc[m, k]:+.1f}" for m in act.index]
        for h, m in enumerate(fc.targets.index, start=1):
            v = fc.targets.loc[m, k]
            lo, hi = bands.interval(k, h, point, v)
            band = f" <sub>[{lo:+.1f}, {hi:+.1f}]</sub>" if lo == lo else ""
            cells.append(f"**{v:+.1f}**{band}")
        lines.append("| " + " | ".join(cells) + " |")
    note = (f"データ時点 {pd.Timestamp(asof):%Y-%m-%d}（最新の公表月 {fc.T.year}年{fc.T.month}月）。"
            "実績は総務省の公表値、［ ］は90%区間（政策の不確実性は含まない）。自動実行のたびに更新。")
    return "\n".join([MARK_START, "", *lines, "", note, "", MARK_END])


def update_readme(fc, point: str, asof, path: str = README) -> bool:
    if not os.path.exists(path):
        return False
    text = open(path, encoding="utf-8").read()
    if MARK_START not in text or MARK_END not in text:
        return False
    i, j = text.index(MARK_START), text.index(MARK_END) + len(MARK_END)
    new = text[:i] + readme_table(fc, point, asof) + text[j:]
    if new != text:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new)
    return True


def cmd_forecast(a):
    if a.asof:
        ds = _load_vintage(a.asof)
    else:
        ds = data.fetch()
        if not a.no_save:
            n = sum(vintage.save_snapshot(df, prefix=p) for p, df in data.to_frames(ds).items())
            print(f"[vintage] {date.today()} snapshot: {n} new/revised values -> {vintage.home()}")
            if a.only_if_new and n == 0:
                print("[forecast] no new data since the last snapshot -> nothing to do")
                return None
        ds = ds.cut(pd.Timestamp.now())
    fc = forecast.run(ds)
    point = bands.point_of(ds.asof, fc.T)
    print(f"最新の公表月（全国）: {fc.T:%Y-%m}　時点 {point}")
    tab = _result_table(fc, point)
    print(tab.to_string())

    nw, nw_label = None, ""
    try:
        prev = None if a.asof else vintage.last_run_before(date.today())
        if prev is not None:
            old = _load_vintage(prev.date())
            nw = news.decompose(old, ds, "core")
            nw_label = f"{prev:%-m/%-d}→{ds.asof:%-m/%-d}"
            print(f"ニュース分解（コア, {nw_label}）"); print(nw.round(2).to_string(index=False))
    except Exception as ex:
        print(f"[news] failed: {ex}")

    if not a.asof and not a.no_save:
        vintage.log_run(fc, MODEL)
        vintage.runs().to_csv(os.path.join(vintage.home(), "runs.csv"), index=False, encoding="utf-8-sig")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        stamp = f"{pd.Timestamp(ds.asof):%Y%m%d}"
        _result_table(fc, point, text=False).to_csv(os.path.join(a.out, "forecast.csv"), encoding="utf-8-sig")
        if update_readme(fc, point, ds.asof):
            print("[readme] 予測の表を更新")
        fc.contributions("headline").round(3).to_csv(os.path.join(a.out, "forecast_contributions.csv"),
                                                    encoding="utf-8-sig")
        if nw is not None:
            nw.round(3).to_csv(os.path.join(a.out, "forecast_news.csv"), index=False, encoding="utf-8-sig")
        try:
            runs = vintage.runs()
        except Exception:
            runs = None
        rp = os.path.join(a.out, f"report_{fc.targets.index[0]:%Y%m}_{stamp}.png")
        plots.plot_report(fc, rp, runs=runs, news=nw, news_label=nw_label, inputs=_inputs(ds, fc),
                          model_name=MODEL)
        shutil.copyfile(rp, os.path.join(a.out, "latest.png"))
        print(f"[report] {rp}")
    return fc


def cmd_backtest(a):
    ds = data.fetch()
    bt = backtest.run(ds, start=a.start, verbose=True)
    sc = backtest.scores(bt)
    for p in ("a", "c", "b"):
        print(f"--- point {p} (MAE, %pt)"); print(backtest.table(sc, "model", p).to_string())
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        bt.to_csv(os.path.join(a.out, "backtest.csv"), index=False, encoding="utf-8-sig")
        sc.round(4).to_csv(os.path.join(a.out, "backtest_scores.csv"), index=False, encoding="utf-8-sig")
    return bt


def cmd_bands(a):
    bt = cmd_backtest(a)
    print(bands.build(bt).round(3).to_string(index=False))


def cmd_history(a):
    r = vintage.runs()
    if r.empty:
        raise SystemExit("no logged runs yet")
    print(r.pivot_table(index=["run_at", "month"], columns="target", values="yoy").to_string())


def main(argv=None):
    p = argparse.ArgumentParser(prog="jp-cpinow")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("forecast")
    f.add_argument("--asof", help="re-run on the stored vintage of this date")
    f.add_argument("--no-save", action="store_true")
    f.add_argument("--only-if-new", action="store_true")
    f.add_argument("--out")
    for name in ("backtest", "bands"):
        b = sub.add_parser(name)
        b.add_argument("--start", default="2015-01-01")
        b.add_argument("--out")
    sub.add_parser("history")
    a = p.parse_args(argv)
    {"forecast": cmd_forecast, "backtest": cmd_backtest, "bands": cmd_bands,
     "history": cmd_history}[a.cmd](a)


if __name__ == "__main__":
    main()
