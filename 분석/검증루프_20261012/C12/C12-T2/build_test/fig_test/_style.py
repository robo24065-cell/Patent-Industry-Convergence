# -*- coding: utf-8 -*-
"""보고서 그림 공통 스타일 v3 (최종 인쇄 크기 그대로 그림).

규칙: 폭 17cm(6.69in) 고정, 높이는 그림마다 cm로 지정, 300dpi, bbox 자르기 없음(물리 크기 보존),
그림 안 최소 글자 7pt(MINPT), 제목은 그림 안에 넣지 않음(캡션은 문서에서), Malgun Gothic, 하이픈 마이너스.
"""
import os
import itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text

OUT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(OUT, "..", "..", "모델경쟁_20260927"))
SR = os.path.join(BASE, "_shared_report")
FA = os.path.join(SR, "fusion_advantage")

ACC = "#1B6AA5"      # 강조색(제공 데이터·모형)
ACC_L = "#A9C8E2"    # 강조색 옅은 단계
ACC_XL = "#DCE9F4"
INK = "#262626"      # 본문 글자
G1 = "#595959"       # 보조 글자·진한 회색 마크
G2 = "#8C8C8C"       # 회색 마크(기준선·대조군)
G3 = "#BFBFBF"
G4 = "#E6E6E6"       # 격자·면
WHITE = "#FFFFFF"

CM = 1 / 2.54
W_IN = 6.69          # 본문 폭 17cm. md2docx FIG_NATIVE=1의 최대 642px(=6.69in×96)와 같아 축소되지 않는다.
MINPT = 7.0          # 그림 안 최소 글자 크기(pt)
FS = 7.4             # 기본 본문 글자
FS_S = 7.2           # 작은 글자(주석·눈금) - 7pt 이상
FS_B = 8.0           # 패널 라벨·상자 머리

plt.rcParams.update({
    "font.family": "Malgun Gothic",
    "axes.unicode_minus": False,
    "font.size": FS,
    "axes.titlesize": FS_B,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": FS,
    "axes.labelcolor": INK,
    "axes.edgecolor": G2,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": G1,
    "ytick.color": G1,
    "xtick.labelsize": FS_S,
    "ytick.labelsize": FS_S,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.fontsize": FS_S,
    "legend.frameon": False,
    "text.color": INK,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.facecolor": WHITE,
})


def figure(h_cm):
    """폭 17cm × 높이 h_cm 그림."""
    return plt.figure(figsize=(W_IN, h_cm * CM))


def cm_axes(fig, x0, y0, w, h, **kw):
    """그림 왼쪽 아래 기준 cm 좌표로 축을 놓는다."""
    fw, fh = fig.get_size_inches() / CM
    return fig.add_axes([x0 / fw, y0 / fh, w / fw, h / fh], **kw)


# ---------------- 검사 ----------------
_BOXED = []   # (Text, (x0, y0, x1, y1) 데이터 좌표, ax) : 도식 상자 안에 있어야 하는 글자


def boxed(t, ax, x0, y0, x1, y1):
    _BOXED.append((t, (x0, y0, x1, y1), ax))
    return t


def _drawn_texts(fig):
    out = []
    for t in fig.texts:
        out.append(t)
    for ax in fig.axes:
        out += list(ax.texts)
        if not ax.axison:          # axis("off") 축은 눈금·축 이름을 그리지 않는다
            continue
        for tt in (ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label):
            out.append(tt)
        for axis in (ax.xaxis, ax.yaxis):
            for tk in axis._update_ticks():
                out += [tk.label1, tk.label2]
        lg = ax.get_legend()
        if lg is not None:
            out += list(lg.get_texts()) + [lg.get_title()]
    for lg in fig.legends:
        out += list(lg.get_texts()) + [lg.get_title()]
    return [t for t in out if isinstance(t, Text) and t.get_visible() and t.get_text().strip()]


def check(fig, name, allow_overlap=()):
    """최소 글자 크기, 그림 밖으로 잘림, 글자끼리 겹침, 상자 넘침을 검사해 문제 목록을 돌려준다."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    texts = _drawn_texts(fig)
    probs = []
    ext = {}
    for t in texts:
        if t.get_fontsize() < MINPT - 1e-6:
            probs.append(f"작은 글자 {t.get_fontsize():.1f}pt: {t.get_text()[:30]!r}")
        e = t.get_window_extent(r)
        ext[id(t)] = e
        if e.x0 < -0.5 or e.y0 < -0.5 or e.x1 > W + 0.5 or e.y1 > H + 0.5:
            probs.append(f"잘림: {t.get_text()[:30]!r} ({e.x0:.0f},{e.y0:.0f})-({e.x1:.0f},{e.y1:.0f}) / {W:.0f}x{H:.0f}")
    for a, b in itertools.combinations(texts, 2):
        ea, eb = ext[id(a)], ext[id(b)]
        ix = min(ea.x1, eb.x1) - max(ea.x0, eb.x0)
        iy = min(ea.y1, eb.y1) - max(ea.y0, eb.y0)
        if ix > 0.8 and iy > 0.8:
            key = (a.get_text(), b.get_text())
            if any(k in key[0] and k2 in key[1] or k in key[1] and k2 in key[0] for k, k2 in allow_overlap):
                continue
            probs.append(f"겹침: {a.get_text()[:25]!r} <> {b.get_text()[:25]!r}")
    for t, (x0, y0, x1, y1), ax in _BOXED:
        if t.figure is not fig:
            continue
        e = t.get_window_extent(r)
        p0 = ax.transData.transform((x0, y0)); p1 = ax.transData.transform((x1, y1))
        if e.x0 < p0[0] - 0.5 or e.x1 > p1[0] + 0.5 or e.y0 < p0[1] - 0.5 or e.y1 > p1[1] + 0.5:
            probs.append(f"상자 넘침: {t.get_text()[:30]!r}")
    return probs


def save(fig, name, allow_overlap=()):
    """물리 크기 그대로 저장(bbox 자르기 없음). 검사 결과를 출력한다."""
    probs = check(fig, name, allow_overlap)
    p = os.path.join(OUT, name + ".png")
    fig.savefig(p, dpi=300)
    w, h = fig.get_size_inches() / CM
    plt.close(fig)
    print(f"[{name}] {w:.2f} x {h:.2f} cm, 문제 {len(probs)}건")
    for s in probs:
        print("   -", s)
    return p


# v2 방식(잘라서 저장, 6.3pt 주석) - 원고에 쓰지 않는 fig06·fig07 재현용으로만 남긴다.
def note(fig, text, y=0.01, size=6.3):
    fig.text(0.01, y, text, ha="left", va="bottom", fontsize=size, color=G1, wrap=True)


def save_legacy(fig, name):
    p = os.path.join(OUT, name + ".png")
    fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return p


def csv(df, name):
    p = os.path.join(OUT, name + ".csv")
    df.to_csv(p, index=False, encoding="utf-8-sig")
    return p
