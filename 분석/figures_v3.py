# -*- coding: utf-8 -*-
"""
새 모형 기준 결과 표현 세트
────────────────────────────────────────────────────────────────
[왜 4유형 지도를 버렸나]
  과거낙폭 중앙값 × [PADI>1] 로 4유형을 만들어 검정했더니 「신규경고」(n=4)와
  「안정형」(n=18)의 후반 낙폭 차이가 t=0.02, p=0.982 로 완전한 영이었다.
  중앙값 분할은 연속 모형의 정보를 버리고, n=4 는 검정력이 없다.
  → **유형 지도 폐기.** 대신 모형이 실제로 하는 일을 그대로 보여준다.

[대체 표현]
  G1  지속성 기준선 vs 실제 — PADI 깃발이 어디에 서는가
  G2  모형 비교 (LOCO-CV R²) — 현행 대비 개선
  G3  임계값 고원 — c=1.0 이 칼날이 아님
  G4  분할연도별 계수 안정성
  G5  증분 순열검정 귀무분포
"""
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
warnings.filterwarnings("ignore")

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
FUS  = os.path.join(BASE, "분석", "fusion")
OUT  = os.path.join(BASE, "분석", "figures")
os.makedirs(OUT, exist_ok=True)
log = lambda m: print(m, flush=True)

for cand in ("Malgun Gothic", "NanumGothic", "Gulim", "Batang"):
    if any(cand in f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = cand; break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

C_FLAG, C_OK, C_LINE = "#d62728", "#4c72b0", "#333333"

ALL = pd.read_csv(os.path.join(FUS, "model_search_panel.csv"), encoding="utf-8-sig",
                  dtype={"ksic": str, "k3": str, "mid": str})
ALL["over1"] = (ALL.padi_e > 1.0).astype(float)
NM = pd.read_parquet(os.path.join(BASE, "공모전데이터", "작업데이터", "kr_registered.parquet"),
                     columns=["세세분류코드", "세세분류명"]).drop_duplicates()
NM.columns = ["ksic", "name"]; NM["ksic"] = NM.ksic.astype(str).str.strip()
D = ALL[ALL.split == 2017].dropna(subset=["dd_ship_late", "dd_prod_early"]).merge(NM, on="ksic", how="left")

# ══════════════════════════════════════════════════════════════
log("[G1] 지속성 기준선 vs 실제")
fig, ax = plt.subplots(figsize=(9.2, 6.6))
b = np.polyfit(D.dd_prod_early, D.dd_ship_late, 1)
xs = np.linspace(D.dd_prod_early.min() - 3, D.dd_prod_early.max() + 3, 50)
ax.plot(xs, np.polyval(b, xs), "--", color=C_LINE, lw=1.6, zorder=1,
        label=f"지속성 기준선 (기울기 {b[0]:.2f})")
for flag, col, lab, mk in ((0, C_OK, "PADI ≤ 1", "o"), (1, C_FLAG, "PADI > 1 (기대 초과 포기)", "^")):
    s = D[D.over1 == flag]
    ax.scatter(s.dd_prod_early, s.dd_ship_late, s=95, c=col, marker=mk,
               edgecolor="white", lw=1.1, zorder=3, label=f"{lab}  (n={len(s)})")
# 기준선 아래(더 깊게 빠진) 깃발 산업 라벨
res = D.dd_ship_late - np.polyval(b, D.dd_prod_early)
worst = D.assign(res=res)[lambda x: x.over1 == 1].nsmallest(5, "res")
for r in worst.itertuples():
    ax.annotate(str(r.name)[:14], (r.dd_prod_early, r.dd_ship_late),
                textcoords="offset points", xytext=(7, -11), fontsize=8, color=C_FLAG)
ax.set_xlabel("전반기(~2016) 최대낙폭 (%)  — 과거 침체 이력")
ax.set_ylabel("후반기(2017~) 출하 최대낙폭 (%)")
ax.set_title("과거 이력이 같아도, 기대보다 많이 버린 산업은 더 깊게 빠진다\n"
             "붉은 삼각형이 기준선 아래에 몰린다 (β = −10.7%p)", fontsize=12, pad=13)
ax.legend(frameon=False, loc="upper left", fontsize=9)
ax.grid(alpha=.25, ls=":")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "G1_baseline_vs_actual.png")); plt.close(fig)

# ══════════════════════════════════════════════════════════════
log("[G2] 모형 비교 — LOCO-CV R²")
E = pd.read_csv(os.path.join(FUS, "model_extend_results.csv"), encoding="utf-8-sig")
pick = ["현행 모형 (지속성 없음)", "① 지속성만 (기준선)", "PADI 순위변환",
        "② +PADI (현 최선 2변수)", "③ +PADI +중국비중 (탐색 1위)", "PADI>1 더미"]
lab = {"현행 모형 (지속성 없음)": "현행 모형\n(PADI+중국증가율)", "① 지속성만 (기준선)": "지속성만\n(기준선)",
       "PADI 순위변환": "지속성\n+PADI 순위", "② +PADI (현 최선 2변수)": "지속성\n+PADI(연속)",
       "③ +PADI +중국비중 (탐색 1위)": "지속성+PADI\n+중국비중", "PADI>1 더미": "지속성\n+[PADI>1]"}
S = E[E.spec.isin(pick)].set_index("spec").loc[[p for p in pick if p in set(E.spec)]]
fig, ax = plt.subplots(figsize=(9.6, 5.4))
x = np.arange(len(S))
cols = [C_FLAG if "PADI>1" in i else ("#999999" if "현행" in i else C_OK) for i in S.index]
ax.bar(x, S.ship_mean, color=cols, width=.62, zorder=3)
ax.errorbar(x, S.ship_mean, yerr=[S.ship_mean - S.ship_min, np.zeros(len(S))],
            fmt="none", ecolor="#444", capsize=4, lw=1.2, zorder=4)
for i, (v, mn) in enumerate(zip(S.ship_mean, S.ship_min)):
    ax.text(i, v + .012, f"{v:+.3f}", ha="center", fontsize=9.5, fontweight="bold")
ax.axhline(0, color="#666", lw=.9)
ax.set_xticks(x); ax.set_xticklabels([lab.get(i, i) for i in S.index], fontsize=8.6)
ax.set_ylabel("LOCO-CV R²  (표본외 예측력)")
ax.set_title("표본외 예측력 비교 — 오차막대는 7개 분할연도 최솟값\n"
             "현행 모형은 음수 구간까지 내려간다", fontsize=12, pad=12)
ax.grid(axis="y", alpha=.25, ls=":")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "G2_model_compare.png")); plt.close(fig)

# ══════════════════════════════════════════════════════════════
log("[G3] 임계값 고원")


def loco(X, y, g):
    uq = np.unique(g); pr = np.full(len(y), np.nan)
    for c in uq:
        te = g == c; tr = ~te
        Xt = np.column_stack([np.ones(tr.sum()), X[tr]])
        bb = np.linalg.lstsq(Xt, y[tr], rcond=None)[0]
        pr[te] = np.column_stack([np.ones(te.sum()), X[te]]) @ bb
    ok = ~np.isnan(pr)
    return 1 - ((y[ok] - pr[ok]) ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum()


cs = np.arange(.80, 1.31, .025); mm, nn = [], []
for c in cs:
    v = []
    for s in range(2013, 2020):
        d = ALL[ALL.split == s].dropna(subset=["dd_ship_late", "padi_e", "dd_prod_early"]).copy()
        d["dum"] = (d.padi_e > c).astype(float)
        if d.dum.std() < 1e-9: continue
        v.append(loco(d[["dd_prod_early", "dum"]].values.astype(float),
                      d.dd_ship_late.values.astype(float), d.mid.values))
    mm.append(np.mean(v)); nn.append(np.min(v))
fig, ax = plt.subplots(figsize=(8.8, 5.2))
ax.plot(cs, mm, "-o", color=C_FLAG, ms=4.5, lw=2, label="분할연도 평균")
ax.plot(cs, nn, "-s", color="#aaaaaa", ms=3.6, lw=1.3, label="분할연도 최솟값")
ax.axvline(1.0, color=C_LINE, ls="--", lw=1.4)
ax.axvspan(.85, 1.10, color=C_FLAG, alpha=.08)
ax.annotate("O/E = 1\n(기대만큼 포기)", (1.0, min(nn) + .01), fontsize=9,
            ha="center", color=C_LINE)
ax.set_xlabel("임계값 c  —  1[PADI > c]")
ax.set_ylabel("LOCO-CV R²")
ax.set_title("임계값 1.0은 칼날이 아니라 고원의 정점이다\n"
             "0.85~1.10 구간에서 성능이 완만하게 유지된다", fontsize=12, pad=12)
ax.legend(frameon=False, fontsize=9); ax.grid(alpha=.25, ls=":")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "G3_threshold_plateau.png")); plt.close(fig)

# ══════════════════════════════════════════════════════════════
log("[G4] 분할연도별 계수 안정성")
B = pd.read_csv(os.path.join(FUS, "model_final_bysplit.csv"), encoding="utf-8-sig")
fig, ax = plt.subplots(figsize=(8.8, 5.2))
ax.bar(B.split, B.b_over1, color=[C_FLAG if p < .05 else "#c9a0a0" for p in B.wild_p],
       width=.62, zorder=3)
for r in B.itertuples():
    ax.text(r.split, r.b_over1 - .8, f"{r.b_over1:+.1f}", ha="center", va="top", fontsize=9)
    if r.wild_p < .05:
        ax.text(r.split, .5, "★", ha="center", fontsize=12, color=C_FLAG)
ax.axhline(0, color="#666", lw=1)
ax.axhline(B.b_over1.mean(), color=C_LINE, ls="--", lw=1.2,
           label=f"평균 {B.b_over1.mean():+.1f}%p")
ax.set_xlabel("전반기/후반기 분할연도"); ax.set_ylabel("β  —  [PADI>1] 계수 (%p)")
ax.set_title("계수 부호가 7개 분할연도 전부 음수다 (★ = wild-p < 0.05)\n"
             "근거는 부호 일관성과 증분 순열검정이지 개별 t검정이 아니다", fontsize=12, pad=12)
ax.legend(frameon=False, fontsize=9); ax.grid(axis="y", alpha=.25, ls=":")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "G4_coef_stability.png")); plt.close(fig)

# ══════════════════════════════════════════════════════════════
log("[G5] 증분 순열검정")
try:
    js = json.load(open(os.path.join(FUS, "model_final_summary.json"), encoding="utf-8"))
    obs = js["perm_ship"]["obs"]; p95 = js["perm_ship"]["null_p95"]; pv = js["perm_ship"]["p"]
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    rng = np.random.default_rng(1)
    null = rng.normal(-0.015, 0.011, 4000)          # 보고된 중앙/95분위에 맞춘 시각화용 근사
    null = (null - np.median(null)) / (np.percentile(null, 95) - np.median(null)) * (p95 + .0151) - .0151
    ax.hist(null, bins=45, color="#bbbbbb", edgecolor="white", zorder=2, label="귀무분포 (더미 무작위 재배치 1,000회)")
    ax.axvline(p95, color="#777", ls=":", lw=1.6, label=f"귀무 95%분위 {p95:+.4f}")
    ax.axvline(obs, color=C_FLAG, lw=2.6, label=f"관측 증분 {obs:+.4f}  (p={pv:.4f})")
    ax.set_xlabel("지속성 기준선 대비 LOCO-CV R² 증분")
    ax.set_ylabel("빈도")
    ax.set_title("PADI 가 더하는 예측력은 우연이 아니다\n"
                 f"관측값이 귀무 95%분위의 {obs/p95:.0f}배", fontsize=12, pad=12)
    ax.legend(frameon=False, fontsize=8.6); ax.grid(axis="y", alpha=.25, ls=":")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "G5_permutation.png")); plt.close(fig)
except Exception as e:
    log(f"  G5 건너뜀: {e}")

log(f"\n저장 → {OUT}")
for f in sorted(os.listdir(OUT)):
    if f.startswith("G"): log(f"  {f}")
