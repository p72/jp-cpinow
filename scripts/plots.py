"""One-page 16:9 report (white background, Japanese labels via the estat-api skill's jp_font)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import bands, config, cpi, policy, release
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
                news_label: str = "", inputs: list | None = None,
                model_name: str = "v1", recon: pd.DataFrame | None = None):
    jp = _font()
    asof = pd.Timestamp(fc.asof or pd.Timestamp.today()).normalize()
    point = bands.point_of(asof, fc.T)
    months = list(fc.targets.index)

    fig = plt.figure(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(3, 2, height_ratios=[0.31, 0.37, 0.32], width_ratios=[1.0, 1.0],
                          left=0.06, right=0.98, top=0.955, bottom=0.075, hspace=0.62, wspace=0.13)

    # ------------------------------------------------------------------ header
    ax = fig.add_subplot(gs[0, :]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0, 1.0, "日本版CPI Now　全国消費者物価指数（前年同月比）の予測", fontsize=17,
            color=INK, va="top", weight="bold")
    m1 = months[0]
    core = fc.targets.loc[m1, "core"]
    lo, hi = bands.interval("core", 1, point, core)
    ax.text(0, 0.56, f"{m1.year}年{m1.month}月分 コア（生鮮食品を除く総合）", fontsize=11, color=INK2, va="bottom")
    ax.text(0, -0.04, f"{core + 0.0:+.1f}%", fontsize=40, color=BLUE, va="bottom", weight="bold")
    if lo == lo:
        ax.text(0.135, 0.02, f"90%区間\n{lo:+.1f} 〜 {hi:+.1f}", fontsize=9.5, color=MUTED,
                va="bottom", linespacing=1.3)
    # table: recent official figures (3 months up to T) + forecasts (h=1..3)
    hist = [fc.T - pd.offsets.MonthBegin(k) for k in (2, 1, 0)]
    act = fc.recent(3)
    x0, xl, ca, cf = 0.25, 0.345, 0.06, 0.12      # label, first value column, widths
    xa = [xl + (j + 1) * ca for j in range(3)]             # right edges, official
    xf = [xa[-1] + 0.02 + (j + 1) * cf for j in range(3)]  # right edges, forecast
    ax.add_patch(plt.Rectangle((xa[-1] + 0.01, -0.05), xf[-1] - xa[-1], 0.86,
                               color="#f2f2f0", zorder=0, transform=ax.transAxes))
    ax.text(x0, 0.70, "前年同月比（%）", fontsize=9, color=INK2, va="center")
    for j, m in enumerate(hist):
        ax.text(xa[j], 0.70, f"実績\n{m.year % 100}年{m.month}月", fontsize=9.5, color=INK2, ha="right",
                va="center", linespacing=1.3)
    for j, m in enumerate(months):
        ax.text(xf[j], 0.70, f"予測 h={j + 1}\n{m.year % 100}年{m.month}月", fontsize=9.5,
                color=BLUE, weight="bold", ha="right", va="center", linespacing=1.3)
    ax.plot([x0, xf[-1]], [0.585, 0.585], color=INK2, lw=0.8)
    for i, (k, code, _) in enumerate(config.TARGETS):
        y = 0.50 - i * 0.11
        ax.text(x0, y, config.SHORT[k], fontsize=11, color=INK, va="center",
                weight="bold" if k == "core" else "normal")
        for j, m in enumerate(hist):
            v = act.loc[m, k] if m in act.index else np.nan
            ax.text(xa[j], y, _fmt(v), fontsize=11, color=INK2, ha="right", va="center")
        for j, m in enumerate(months):
            v = fc.targets.loc[m, k]
            l, h_ = bands.interval(k, j + 1, point, v)
            ax.text(xf[j] - 0.055, y, f"{v + 0.0:+.1f}", fontsize=12, color=INK, ha="right", va="center",
                    weight="bold")
            if l == l:
                ax.text(xf[j], y, f"[{l:+.1f}, {h_:+.1f}]", fontsize=8.5, color=MUTED, ha="right", va="center")
    ax.text(1.0, 1.0, f"データ時点 {asof:%Y-%m-%d}　モデル {model_name}", fontsize=9, color=INK2,
            va="top", ha="right")
    ax.text(x0, -0.09, "実績は総務省の公表値（前年同月比）。予測の［ ］は90%区間（擬似リアルタイム検証の誤差分位点。"
            "政策の不確実性は含まない）", fontsize=8.5, color=MUTED, va="top")

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
    ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.09),
              columnspacing=1.2, handlelength=1.4)

    # ------------------------------------------------------------- evolution
    ax = fig.add_subplot(gs[2, 0])
    ax.set_title(f"{m1.year}年{m1.month}月分の予測の推移（白抜き＝再構成・参考、塗り＝実際の記録）",
                 fontsize=12, color=INK, loc="left")
    drawn = False
    for k, color in [("headline", INK2), ("corecore", SLOTS[2]), ("core", BLUE)]:
        last = None
        if recon is not None and not recon.empty:
            q = recon[(recon["month"] == m1) & (recon["target"] == k)].sort_values("date")
            if runs is not None and not runs.empty:   # reconstructed points only before the first record
                first = pd.to_datetime(runs["run_at"]).min().normalize()
                q = q[q["date"] < first]
            if len(q):
                ax.plot(q["date"], q["yoy"], color=color, lw=1.3, ls="--", marker="o", ms=6,
                        mfc="white", mew=1.4)
                last = (q["date"].iloc[-1], q["yoy"].iloc[-1])
                drawn = True
        if runs is not None and not runs.empty:
            r = runs[runs["month"] == str(m1.date())].copy()
            r["d"] = pd.to_datetime(r["run_at"]).dt.normalize()
            sr = r[r["target"] == k].groupby("d")["yoy"].last()
            if len(sr):
                ax.plot(sr.index, sr.values, color=color, lw=2, marker="o", ms=7, label=config.SHORT[k])
                last = (sr.index[-1], sr.iloc[-1])
                drawn = True
            elif last is not None:
                ax.plot([], [], color=color, lw=2, marker="o", ms=7, label=config.SHORT[k])
        if last is not None:
            ax.annotate(f"{last[1]:+.1f}", last, textcoords="offset points", xytext=(8, 0), va="center",
                        fontsize=9, color=INK)
    if drawn:
        nat_rel = m1 + pd.offsets.MonthBegin(1) + pd.Timedelta(days=release.NAT_DAY - 1)
        start = m1 - pd.offsets.MonthBegin(2) + pd.Timedelta(days=release.NAT_DAY - 2)
        ax.set_xlim(min(start, asof - pd.Timedelta(days=2)), nat_rel + pd.Timedelta(days=3))
        lo_y, hi_y = ax.get_ylim()
        mid = (lo_y + hi_y) / 2
        half = max((hi_y - lo_y) / 2 + 0.1, 0.5)
        ax.set_ylim(mid - half, mid + half)
        ax.axvline(nat_rel, color=MUTED, lw=0.8, ls=":")
        ax.text(nat_rel, mid - half, " 全国CPI公表", fontsize=8.5, color=MUTED, va="bottom", ha="right")
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.FR, interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%-m/%-d"))
        ax.set_ylabel("前年比（%）", fontsize=9)
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
    if news is not None and len(news) > 1 and f"{m1:%Y-%m}" in news:
        col = f"{m1:%Y-%m}"
        body = news[news["統計"] != "合計"].copy()
        body = body.reindex(body[col].abs().sort_values(ascending=False).index).head(3)
        tot = float(news.loc[news["統計"] == "合計", col].iloc[0])
        y -= 0.02
        ax.text(0, y, f"前回からの改定（コア{m1.month}月, {news_label}）：{tot:+.2f}pt", fontsize=11,
                color=INK, weight="bold", va="top")
        y -= 0.035
        for _, r in body.iterrows():
            ax.text(0, y, f"・{r['統計']}  {r[col]:+.2f}", fontsize=9, color=INK2, va="top")
            y -= 0.028
    # policy calendar for the horizon
    y -= 0.02
    ax.text(0, y, "予測の前提（政策）と制度要因（policy.csv）", fontsize=12, color=INK, weight="bold", va="top")
    y -= 0.04
    import textwrap
    for a_ in policy.assumptions(months, asof):
        for k_, ln in enumerate(textwrap.wrap(a_, 46)):
            ax.text(0, y, ("▶ " if k_ == 0 else "　 ") + ln, fontsize=9, color=INK,
                    va="top")
            y -= 0.027
    pol = policy.load(asof)
    lo_m, hi_m = months[0], months[-1]
    shown = 0
    seen = set()
    for _, r in pol.iterrows():
        if r["kind"] in ("mask",) or pd.isna(r["start"]):
            continue
        end = r["end"] if pd.notna(r["end"]) else hi_m
        if r["kind"].startswith("subsidy") and end + pd.offsets.MonthBegin(1) >= lo_m and end < hi_m:
            nxt = end + pd.offsets.MonthBegin(1)
            later = pol[(pol["kind"] == r["kind"]) & (pol["start"] == nxt)]
            key = (f"{nxt:%Y-%m}", "end")
            if later.empty and key not in seen:
                seen.add(key)
                ax.text(0, y, f"・{nxt:%Y-%m}　電気・ガス料金支援の終了（値引きなし）", fontsize=9, color=INK2, va="top")
                y -= 0.03
                shown += 1
        if r["start"] > hi_m or end < lo_m:
            continue
        if r["kind"] == "levy" and r["start"] < lo_m:
            continue
        rng = f"{r['start']:%Y-%m}" + (f"〜{r['end']:%Y-%m}" if pd.notna(r["end"]) and r["end"] != r["start"] else "")
        txt = r["note"][:38]
        if (rng, txt) in seen:
            continue
        seen.add((rng, txt))
        ax.text(0, y, f"・{rng}　{txt}", fontsize=9, color=INK2, va="top")
        y -= 0.03
        shown += 1
        if shown >= 4:
            break
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


def plot_history(rec: pd.DataFrame, recon: pd.DataFrame, actual: pd.DataFrame | None, months, path: str,
                 asof=None):
    """One panel per series: how the forecast for each target month moved over time.
    Hollow/dashed = reconstruction (reference), filled/solid = actual records, ★ = published."""
    _font()
    months = list(pd.DatetimeIndex(months))
    fig, axes = plt.subplots(len(config.TARGETS), 1, figsize=(12, 3.0 * len(config.TARGETS)), sharex=True)
    fig.patch.set_facecolor("white")
    first_rec = rec["date"].min() if rec is not None and len(rec) else None
    for ax, (k, _, label) in zip(axes, config.TARGETS):
        labels = []
        for m, color in zip(months, SLOTS):
            pts = []
            if recon is not None and len(recon):
                q = recon[(recon["month"] == m) & (recon["target"] == k)].sort_values("date")
                if first_rec is not None:
                    q = q[q["date"] < first_rec]
                if len(q):
                    ax.plot(q["date"], q["yoy"], color=color, lw=1.2, ls="--", marker="o", ms=5,
                            mfc="white", mew=1.3)
                    pts.append((q["date"].iloc[-1], q["yoy"].iloc[-1]))
            if rec is not None and len(rec):
                r = rec[(rec["month"] == m) & (rec["target"] == k)].sort_values("date")
                if len(r):
                    ax.plot(r["date"], r["yoy"], color=color, lw=2, marker="o", ms=6)
                    pts.append((r["date"].iloc[-1], r["yoy"].iloc[-1]))
            pub = None
            if actual is not None and m in actual.index and actual.loc[m, k] == actual.loc[m, k]:
                pub_d = m + pd.offsets.MonthBegin(1) + pd.Timedelta(days=release.NAT_DAY - 1)
                pub = (pub_d, float(actual.loc[m, k]))
                ax.plot([pub_d], [pub[1]], marker="*", ms=13, color=color, mec=INK, mew=0.6, ls="none")
            if pub is not None:
                ax.annotate(f"{m.month}月 {pub[1]:+.1f}（公表）", pub, textcoords="offset points",
                            xytext=(-6, -11), ha="right", va="center", fontsize=8, color=INK2,
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
            elif pts:
                labels.append((pts[-1][0], pts[-1][1], f"{m.month}月 {pts[-1][1]:+.1f}"))
        # right-edge labels of the open target months: spread them so they never overlap
        if labels:
            lo_y, hi_y = ax.get_ylim()
            gap = (hi_y - lo_y) * 0.09
            labels.sort(key=lambda t: t[1])
            ys = [labels[0][1]]
            for _, yv, _ in labels[1:]:
                ys.append(max(yv, ys[-1] + gap))
            for (xv, yv, txt), yl in zip(labels, ys):
                ax.annotate(txt, (xv, yv), xytext=(xv + pd.Timedelta(days=4), yl), textcoords="data",
                            va="center", fontsize=8.5, color=INK, annotation_clip=False)
            ax.set_ylim(lo_y, max(hi_y, ys[-1] + gap / 2))
        ax.set_title(f"{config.SHORT[k]}　対象月ごとの予測の推移", fontsize=11, color=INK, loc="left")
        ax.set_ylabel("前年比（%）", fontsize=9)
        ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, lw=2, label=f"{m.year % 100}年{m.month}月分") for m, c in zip(months, SLOTS)]
    handles += [Line2D([], [], color=INK2, lw=1.2, ls="--", marker="o", mfc="white", label="再構成（参考）"),
                Line2D([], [], color=INK2, lw=2, marker="o", label="実際の記録"),
                Line2D([], [], color=INK2, marker="*", ms=12, ls="none", label="公表値")]
    axes[0].legend(handles=handles, frameon=False, fontsize=8.5, ncol=5, loc="lower left",
                   bbox_to_anchor=(0, 1.15))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%-m/%-d"))
    stamp = f"データ時点 {pd.Timestamp(asof or pd.Timestamp.today()):%Y-%m-%d}　" if asof is not None else ""
    fig.text(0.01, 0.005, stamp + "再構成＝過去の公表日時点に切り戻したデータでモデルを再実行した参考値"
             "（当時の実際の予測ではない）。★＝総務省の公表値（全国CPI公表日に表示）。", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.015, 0.93, 0.97))
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)
