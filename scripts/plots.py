"""One-page 16:9 report (white background, Japanese labels via the estat-api skill's jp_font)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import bands, config, policy, release
from .sources import estat

# categorical slots (fixed order) — reference palette
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID = "#1f1f1f", "#52514e", "#8a8a8a", "#e5e5e5"
BLUE, ORANGE = SLOTS[0], SLOTS[1]

GROUPS = [  # (label, leaf blocks) for the contribution chart, stacked in this order
    ("生鮮食品", ["B1"]),
    ("エネルギー", ["B5"]),
    ("食料（生鮮を除く）・外食・酒類", ["B2", "B3", "B4"]),
    ("財（食料・エネルギーを除く）", ["B6"]),
    ("サービス（家賃・外食を除く）", ["B8", "B9"]),
    ("家賃", ["B7"]),
]


def _font():
    """IPAex Gothic from matplotlib-fontja (same look locally and on GitHub Actions);
    else the estat-api skill's font finder."""
    try:
        import matplotlib_fontja  # noqa: F401  (registers and selects IPAexGothic)
        matplotlib.rcParams["font.family"] = "IPAexGothic"
        return True
    except Exception:
        pass
    try:
        return estat().setup_japanese_font() is not None
    except Exception:
        return False


def _fmt(v):
    return "－" if v != v else f"{v + 0.0:+.1f}"


def plot_report(fc, path: str, runs: pd.DataFrame | None = None, news: pd.DataFrame | None = None,
                news_label: str = "", inputs: list | None = None, actual: pd.DataFrame | None = None,
                model_name: str = "v1"):
    jp = _font()
    asof = pd.Timestamp(fc.asof or pd.Timestamp.today()).normalize()
    point = bands.point_of(asof, fc.T)
    months = list(fc.targets.index)

    fig = plt.figure(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(3, 2, height_ratios=[0.25, 0.40, 0.35], width_ratios=[1.0, 1.0],
                          left=0.06, right=0.98, top=0.955, bottom=0.075, hspace=0.45, wspace=0.13)

    # ------------------------------------------------------------------ header
    ax = fig.add_subplot(gs[0, :]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0, 1.0, "日本版CPI Now　全国消費者物価指数（前年同月比）の予測", fontsize=17,
            color=INK, va="top", weight="bold")
    m1 = months[0]
    core = fc.targets.loc[m1, "core"]
    lo, hi = bands.interval("core", 1, point, core)
    ax.text(0, 0.62, f"{m1.year}年{m1.month}月分 コア（生鮮食品を除く総合）", fontsize=11, color=INK2, va="bottom")
    ax.text(0, 0.02, f"{core + 0.0:+.1f}%", fontsize=40, color=BLUE, va="bottom", weight="bold")
    if lo == lo:
        ax.text(0.135, 0.08, f"90%区間\n{lo:+.1f} 〜 {hi:+.1f}", fontsize=9.5, color=MUTED,
                va="bottom", linespacing=1.3)
    # 5 × 3 table
    x0, cw = 0.25, 0.115
    ax.text(x0, 0.80, "系列", fontsize=10, color=INK2, weight="bold", va="center")
    for j, m in enumerate(months):
        ax.text(x0 + 0.13 + (j + 1) * cw, 0.80, f"{m.year % 100}年{m.month}月（h={j + 1}）", fontsize=10,
                color=INK2, weight="bold", ha="right", va="center")
    ax.plot([x0, x0 + 0.13 + 3 * cw], [0.73, 0.73], color=INK2, lw=0.8)
    for i, (k, code, _) in enumerate(config.TARGETS):
        y = 0.62 - i * 0.135
        ax.text(x0, y, config.SHORT[k], fontsize=11, color=INK, va="center",
                weight="bold" if k == "core" else "normal")
        for j, m in enumerate(months):
            v = fc.targets.loc[m, k]
            l, h_ = bands.interval(k, j + 1, point, v)
            xr = x0 + 0.13 + (j + 1) * cw
            ax.text(xr - 0.052, y, f"{v + 0.0:+.1f}", fontsize=12, color=INK, ha="right", va="center",
                    weight="bold")
            if l == l:
                ax.text(xr, y, f"[{l:+.1f}, {h_:+.1f}]", fontsize=8.5, color=MUTED, ha="right", va="center")
    sub = [f"データ時点 {asof:%Y-%m-%d}　最新の公表月 {fc.T.year}年{fc.T.month}月（全国）　モデル {model_name}",
           "［ ］は90%区間（擬似リアルタイム・バックテストの誤差分位点。政策の不確実性は含まない）"]
    ax.text(x0 + 0.13 + 3 * cw + 0.03, 0.02, "\n".join(sub), fontsize=9, color=INK2, va="bottom",
            linespacing=1.6)
    if news is not None and len(news) > 1:
        col = f"{m1:%Y-%m}"
        if col in news:
            body = news[news["統計"] != "合計"].copy()
            body = body.reindex(body[col].abs().sort_values(ascending=False).index).head(4)
            tot = float(news.loc[news["統計"] == "合計", col].iloc[0])
            lines = [f"コア{m1.month}月の改定 {news_label}：{tot:+.2f}pt"]
            lines += [f"  {r['統計']} {r[col]:+.2f}" for _, r in body.iterrows()]
            ax.text(1.0, 0.98, "\n".join(lines), fontsize=9.5, color=INK2, va="top", ha="right",
                    linespacing=1.45)

    # --------------------------------------------------------- contributions
    ax = fig.add_subplot(gs[1, 0])
    comp = fc.contributions("headline")
    hist_m = [m for m in comp.index if m <= fc.T][-6:]
    show = hist_m + months
    comp = comp.loc[show]
    x = np.arange(len(show))
    pos = np.zeros(len(show)); neg = np.zeros(len(show))
    for (lab, bl), color in zip(GROUPS, SLOTS):
        keys = [l.key for l in config.LEAVES if l.block in bl]
        v = comp[keys].sum(axis=1).values
        base = np.where(v >= 0, pos, neg)
        ax.bar(x, v, bottom=base, color=color, width=0.62, label=lab, edgecolor="white", linewidth=1)
        pos += np.where(v >= 0, v, 0); neg += np.where(v < 0, v, 0)
    tot = fc.levels["headline"]
    yy = (tot.round(1) / tot.round(1).shift(12) - 1) * 100
    yv = yy.reindex(show).round(1).values
    ax.plot(x, yv, color=INK, lw=2, marker="o", ms=6, label="総合（前年比）", zorder=5)
    for i, v in enumerate(yv):
        if i in (len(hist_m) - 1,) or i >= len(hist_m):
            ax.annotate(f"{v:+.1f}", (x[i], v), textcoords="offset points", xytext=(0, 9), ha="center",
                        fontsize=9, color=INK)
    ax.axvspan(len(hist_m) - 0.5, len(show) - 0.5, color="#f2f2f0", zorder=0)
    ax.text(len(hist_m) - 0.4, ax.get_ylim()[1], "予測", fontsize=9, color=INK2, va="top")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"{m.year % 100}/{m.month}" for m in show], fontsize=9.5)
    ax.set_title("総合の前年比と寄与度（%pt）", fontsize=12, color=INK, loc="left")
    ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left", bbox_to_anchor=(0, -0.1))

    # ------------------------------------------------------------- evolution
    ax = fig.add_subplot(gs[2, 0])
    ax.set_title(f"{m1.year}年{m1.month}月分の予測の推移（塗り＝実際の記録）", fontsize=12, color=INK, loc="left")
    drawn = False
    if runs is not None and not runs.empty:
        r = runs[runs["month"] == str(m1.date())].copy()
        r["d"] = pd.to_datetime(r["run_at"]).dt.normalize()
        for k, color in [("core", BLUE), ("headline", INK2), ("corecore", SLOTS[2])]:
            s = r[r["target"] == k].groupby("d")["yoy"].last()
            if len(s):
                ax.plot(s.index, s.values, color=color, lw=2, marker="o", ms=7, label=config.SHORT[k])
                ax.annotate(f"{s.iloc[-1]:+.1f}", (s.index[-1], s.iloc[-1]), textcoords="offset points",
                            xytext=(8, 0), va="center", fontsize=9, color=INK)
                drawn = True
    if drawn:
        nat_rel = m1 + pd.offsets.MonthBegin(1) + pd.Timedelta(days=release.NAT_DAY - 1)
        ax.axvline(nat_rel, color=MUTED, lw=0.8, ls=":")
        ax.text(nat_rel, ax.get_ylim()[0], " 全国CPI公表", fontsize=8.5, color=MUTED, va="bottom")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%-m/%-d"))
        ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=3)
    else:
        ax.text(0.5, 0.5, "記録はまだありません（自動実行のたびに追加されます）", ha="center", va="center",
                fontsize=10, color=MUTED, transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
    ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # ---------------------------------------------------------- right column
    ax = fig.add_subplot(gs[1:, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    y = 1.0
    ax.text(0, y, "主な入力データ", fontsize=13, color=INK, weight="bold", va="bottom")
    y -= 0.04
    for row in inputs or []:
        ax.text(0, y, row[0], fontsize=10, color=INK, va="top")
        ax.text(0.55, y, row[1], fontsize=10, color=INK2, va="top")
        y -= 0.034
        ax.plot([0, 1], [y + 0.006, y + 0.006], color=GRID, lw=0.6)
    # policy calendar for the horizon
    y -= 0.02
    ax.text(0, y, "予測期間中の制度要因（policy.csv）", fontsize=12, color=INK, weight="bold", va="top")
    y -= 0.04
    pol = policy.load(asof)
    lo_m, hi_m = months[0], months[-1]
    shown = 0
    for _, r in pol.iterrows():
        if r["kind"] in ("mask",) or pd.isna(r["start"]):
            continue
        end = r["end"] if pd.notna(r["end"]) else hi_m
        if r["start"] > hi_m or end < lo_m - pd.offsets.MonthBegin(1):
            continue
        if r["kind"] == "levy" and r["start"] < lo_m:
            continue
        rng = f"{r['start']:%Y-%m}" + (f"〜{r['end']:%Y-%m}" if pd.notna(r["end"]) and r["end"] != r["start"] else "")
        txt = r["note"][:38]
        ax.text(0, y, f"・{rng}　{txt}", fontsize=9, color=INK2, va="top")
        y -= 0.03
        shown += 1
        if shown >= 6:
            break
    nxt = [m for m in months]
    if shown == 0:
        ax.text(0, y, "（なし）", fontsize=9, color=MUTED, va="top"); y -= 0.03
    # upcoming releases
    up = release.upcoming(asof, 7)
    if not up.empty:
        y -= 0.02
        ax.text(0, y, "今後の公表予定", fontsize=12, color=INK, weight="bold", va="top")
        y -= 0.04
        for _, rw in up.iterrows():
            star = "全国" in rw["items"]
            st = dict(fontsize=9, va="top", color=BLUE if star else INK, weight="bold" if star else "normal")
            ax.text(0, y, rw["items"], **st)
            ax.text(0.55, y, rw["month"], **st)
            ax.text(0.72, y, rw["display"], **st)
            y -= 0.03

    fig.text(0.06, 0.035, "出所：総務省「消費者物価指数」（e-Stat）、日本銀行「企業物価指数」・外国為替市況、"
             "資源エネルギー庁「石油製品小売市況調査」、農林水産省「食品価格動向調査」を加工して作成。",
             fontsize=8.5, color=MUTED)
    fig.text(0.06, 0.012, "個人のモデル推計（公的機関の見通しではない）。"
             "このサービスは、政府統計総合窓口(e-Stat)のAPI機能を使用していますが、サービスの内容は国によって保証されたものではありません。",
             fontsize=8.5, color=MUTED)
    fig.savefig(path, dpi=120, facecolor="white")
    plt.close(fig)
