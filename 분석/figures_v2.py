# -*- coding: utf-8 -*-
"""
결과 표현 세트 — 심사 '창의성 40점: 결과 표현의 창의성' 대응
  F1 사분면 진단 지도 (명명된 4유형)
  F2 산업 궤적 — 전반기→후반기 지도 위 이동
  F3 예측 vs 실제 (모형 성능)
  F4 산업 진단 카드 (상위 6개)
  F5 검정 여정 요약 (148개 가설 → 살아남은 것)
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
FUS = os.path.join(BASE, "분석", "fusion")
V4 = os.path.join(BASE, "분석", "v4")
OUT = os.path.join(BASE, "분석", "figures")
os.makedirs(OUT, exist_ok=True)
log = lambda m: print(m, flush=True)

NAMES = {"15220": "신발부품", "20202": "합성수지", "20411": "일반도료", "20423": "화장품",
         "20491": "감광재료", "21220": "한의약품", "23992": "연마재", "24191": "도금강재",
         "24219": "비철금속", "25914": "금속압형", "25922": "도금업", "25924": "절삭가공",
         "25932": "일반철물", "26112": "비메모리", "26223": "연성PCB", "26224": "실장기판",
         "26291": "전자축전기", "26322": "컴퓨터모니터", "26323": "컴퓨터프린터", "26422": "이동전화기",
         "26529": "기타음향", "27211": "레이더항행", "28122": "전기회로", "28511": "주방전기",
         "28902": "탄소절연", "29131": "액체펌프", "29132": "기체펌프", "29133": "탭밸브",
         "29150": "산업오븐", "29162": "승강기", "29210": "농업기계", "29229": "가공공작기계",
         "29241": "건설기계", "29280": "산업로봇", "30393": "자동차의자", "31202": "철도부품",
         "31312": "무인항공기", "33301": "체력단련", "33309": "기타운동"}

P = pd.read_csv(os.path.join(FUS, "predictive_agg.csv"), encoding="utf-8-sig",
                dtype={"ksic": str, "mid": str})
P = P.dropna(subset=["padi_e", "cn_growth_e", "dd_late"]).copy()
P["name"] = P.ksic.map(NAMES).fillna(P.ksic)
mx, my = P.padi_e.median(), P.cn_growth_e.median()

# 4유형 명명
def quad(r):
    hi_p, hi_c = r.padi_e >= mx, r.cn_growth_e >= my
    if hi_p and hi_c: return "이중압박형"
    if hi_p and not hi_c: return "내부이탈형"
    if not hi_p and hi_c: return "외부추격형"
    return "안정형"


P["type"] = P.apply(quad, axis=1)
COL = {"이중압박형": "#d62728", "내부이탈형": "#ff7f0e", "외부추격형": "#1f77b4", "안정형": "#2ca02c"}
DESC = {"이중압박형": "기술자산 이탈 + 중국 추격 동시",
        "내부이탈형": "자체 기술 포기가 빠름",
        "외부추격형": "중국 추격이 빠름",
        "안정형": "양축 모두 낮음"}

# ── F1 사분면 진단 지도 ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 8.5))
ax.axvspan(mx, P.padi_e.max() * 1.05, ymin=0, ymax=1, color="#fff5f5", zorder=0)
ax.axhspan(my, P.cn_growth_e.max() * 1.05, xmin=0, xmax=1, color="#f5f9ff", zorder=0)
for t, g in P.groupby("type"):
    ax.scatter(g.padi_e, g.cn_growth_e, s=np.abs(g.dd_late) * 7, c=COL[t],
               edgecolors="k", linewidths=.7, alpha=.8, label=f"{t} (n={len(g)})", zorder=3)
for _, r in P.iterrows():
    ax.annotate(r["name"], (r.padi_e, r.cn_growth_e), fontsize=7.5,
                xytext=(4, 4), textcoords="offset points", zorder=4)
ax.axvline(mx, color="gray", ls="--", lw=1, zorder=1)
ax.axhline(my, color="gray", ls="--", lw=1, zorder=1)
for (x, y, t, ha, va) in [(0.97, 0.97, "이중압박형", "right", "top"),
                          (0.97, 0.03, "내부이탈형", "right", "bottom"),
                          (0.03, 0.97, "외부추격형", "left", "top"),
                          (0.03, 0.03, "안정형", "left", "bottom")]:
    ax.text(x, y, t, transform=ax.transAxes, ha=ha, va=va, fontsize=13,
            color=COL[t], alpha=.35, fontweight="bold")
ax.set_xlabel("전반기(2008~16) 특허 조기포기율  →  내부 기술자산 이탈", fontsize=11)
ax.set_ylabel("전반기 중국 출원 증가율(%)  →  외부 경쟁 압력", fontsize=11)
ax.set_title("산업 취약성 4유형 진단 지도  (원 크기 = 실제 후반기 최대낙폭)", fontsize=14, pad=12)
ax.legend(loc="upper left", fontsize=9, framealpha=.9)
ax.grid(alpha=.2, zorder=0)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "F1_quadrant_map.png"), dpi=145); plt.close(fig)
log("F1 사분면 진단 지도")

# ── F2 산업 궤적 ────────────────────────────────────────────────
comp = pd.read_csv(os.path.join(FUS, "competition_panel.csv"), encoding="utf-8-sig", dtype={"ksic": str})
early = comp[(comp.yr >= 2008) & (comp.yr <= 2012)].groupby("ksic").cn_growth.mean()
late = comp[(comp.yr >= 2013) & (comp.yr <= 2016)].groupby("ksic").cn_growth.mean()
T = P[["ksic", "name", "padi_e", "dd_late", "type"]].copy()
T["cn1"] = T.ksic.map(early); T["cn2"] = T.ksic.map(late)
T = T.dropna(subset=["cn1", "cn2"])
fig, ax = plt.subplots(figsize=(11, 7.5))
for _, r in T.iterrows():
    ax.annotate("", xy=(r.padi_e, r.cn2), xytext=(r.padi_e, r.cn1),
                arrowprops=dict(arrowstyle="->", color=COL[r["type"]], lw=1.6, alpha=.75))
    ax.scatter(r.padi_e, r.cn2, s=45, c=COL[r["type"]], edgecolors="k", linewidths=.5, zorder=3)
    if abs(r.cn2 - r.cn1) > 12 or r.cn2 > 35:
        ax.annotate(r["name"], (r.padi_e, r.cn2), fontsize=7.5, xytext=(4, 2),
                    textcoords="offset points")
ax.axhline(0, color="k", lw=.8)
ax.set_xlabel("전반기 특허 조기포기율 (PADI)")
ax.set_ylabel("중국 출원 증가율 (%)")
ax.set_title("중국 추격 압력의 변화 궤적  (화살표: 2008~12 평균 → 2013~16 평균)", fontsize=13)
ax.legend(handles=[mpatches.Patch(color=c, label=t) for t, c in COL.items()], fontsize=9)
ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "F2_trajectory.png"), dpi=145); plt.close(fig)
log("F2 산업 궤적")

# ── F3 예측 vs 실제 ─────────────────────────────────────────────
import statsmodels.formula.api as smf
f = smf.ols("dd_late ~ padi_e + cn_growth_e", data=P).fit()
P["pred"] = f.predict(P)
fig, ax = plt.subplots(figsize=(8.5, 7.5))
for t, g in P.groupby("type"):
    ax.scatter(g.pred, g.dd_late, s=70, c=COL[t], edgecolors="k", linewidths=.6, label=t, alpha=.85)
lim = [min(P.pred.min(), P.dd_late.min()) - 4, max(P.pred.max(), P.dd_late.max()) + 4]
ax.plot(lim, lim, "k--", lw=1, alpha=.6, label="완전 예측선")
for _, r in P.iterrows():
    if abs(r.pred - r.dd_late) > 16 or r.dd_late < -60:
        ax.annotate(r["name"], (r.pred, r.dd_late), fontsize=7.5,
                    xytext=(4, 3), textcoords="offset points")
ax.set_xlabel("모형 예측 낙폭 (%)  ← 전반기 정보만 사용")
ax.set_ylabel("실제 후반기 최대낙폭 (%)")
ax.set_title(f"예측 성능  (R² = {f.rsquared:.3f}, n = {len(P)})", fontsize=13)
ax.legend(fontsize=9); ax.grid(alpha=.25)
ax.set_xlim(lim); ax.set_ylim(lim)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "F3_pred_actual.png"), dpi=145); plt.close(fig)
log(f"F3 예측 vs 실제 (R²={f.rsquared:.3f})")

# ── F4 산업 진단 카드 ───────────────────────────────────────────
P["risk"] = (P.padi_e.rank(pct=True) + P.cn_growth_e.rank(pct=True)) / 2
top = P.nlargest(6, "risk")
fig = plt.figure(figsize=(13, 8))
gs = GridSpec(2, 3, figure=fig, hspace=.35, wspace=.25)
for i, (_, r) in enumerate(top.iterrows()):
    ax = fig.add_subplot(gs[i // 3, i % 3])
    bars = ["포기율\n(백분위)", "중국압력\n(백분위)", "실제낙폭\n(백분위)"]
    vals = [P.padi_e.rank(pct=True)[r.name] * 100,
            P.cn_growth_e.rank(pct=True)[r.name] * 100,
            (1 - P.dd_late.rank(pct=True)[r.name]) * 100]
    ax.barh(bars, vals, color=[COL[r["type"]], "#4c72b0", "#999999"], alpha=.85)
    ax.set_xlim(0, 100)
    for j, v in enumerate(vals):
        ax.text(v + 2, j, f"{v:.0f}", va="center", fontsize=9)
    ax.set_title(f"{r['name']} ({r.ksic})\n{r['type']} · 실제낙폭 {r.dd_late:.0f}%",
                 fontsize=11, color=COL[r["type"]], fontweight="bold")
    ax.grid(axis="x", alpha=.25); ax.set_axisbelow(True)
fig.suptitle("취약성 상위 6개 산업 진단 카드", fontsize=15, y=.98)
fig.savefig(os.path.join(OUT, "F4_industry_cards.png"), dpi=145, bbox_inches="tight"); plt.close(fig)
log("F4 산업 진단 카드")

# ── F5 검정 여정 ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
steps = ["사전등록\n가설 148개", "순방향\n(IP→경기)", "역방향\n(경기→IP)", "신호분해\n(시간축)",
         "횡단면\n(빈도)", "횡단면\n(심도)", "예측검증\n(시간분할)"]
vals = [148, 0, 0, 0, 0, 1, 1]
cols = ["#888888", "#d62728", "#d62728", "#d62728", "#d62728", "#2ca02c", "#2ca02c"]
notes = ["전부 사전등록", "FDR 유의 0/120", "FDR 유의 0/28", "산업내 AUC 0.51",
         "정의 민감, 기각", "r=-0.52 강건", "r=-0.478, R²=0.33"]
ax.bar(range(len(steps)), [1] * len(steps), color=cols, alpha=.8, width=.6)
for i, (s, n) in enumerate(zip(steps, notes)):
    ax.text(i, .5, n, ha="center", va="center", fontsize=9, color="white", fontweight="bold", rotation=0)
ax.set_xticks(range(len(steps))); ax.set_xticklabels(steps, fontsize=9.5)
ax.set_yticks([]); ax.set_ylim(0, 1.3)
for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
ax.set_title("검정 여정 — 무엇을 기각하고 무엇이 살아남았는가", fontsize=14, pad=14)
ax.text(.5, 1.15, "빨강 = 기각    초록 = 생존", transform=ax.transAxes, ha="center", fontsize=10, color="#555")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "F5_journey.png"), dpi=145); plt.close(fig)
log("F5 검정 여정")

P[["ksic", "name", "type", "padi_e", "cn_growth_e", "dd_late", "pred", "risk"]].sort_values(
    "risk", ascending=False).to_csv(os.path.join(OUT, "industry_diagnosis.csv"),
                                    index=False, encoding="utf-8-sig")
log(f"\n4유형 분포: {P.type.value_counts().to_dict()}")
log(f"저장 → {OUT}")
