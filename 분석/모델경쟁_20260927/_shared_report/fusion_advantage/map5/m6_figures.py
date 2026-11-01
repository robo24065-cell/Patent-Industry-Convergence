# -*- coding: utf-8 -*-
r"""map5 6단계: 보고서용 그림 4장(PNG) + 그림 데이터 CSV. 입력은 m2–m5 산출물만."""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 8,
                     "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
BLUE, ORANGE, GRAY, INK = "#2a78d6", "#eb6834", "#b5b4ae", "#0b0b0b"
div = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#f0a3a2", "#c93a39"])
seq = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#cde2fb", "#86b6ef", "#2a78d6", "#104281"])
CCS = ["KR", "US", "JP", "CN", "EP"]
prof = pd.read_csv(os.path.join(OUT, "m2_ksic_profile.csv"), dtype={"ksic": str}).set_index("ksic")

# F1 5개국 RTA 지형도 (P3 2017–19)
f1 = prof[[f"rta_{c}_P3" for c in CCS]].copy(); f1.columns = CCS
f1 = f1.loc[f1.KR.sort_values(ascending=False).index]
f1.insert(0, "name", prof.name.reindex(f1.index))
f1.to_csv(os.path.join(OUT, "fig1_rta_map_P3.csv"), encoding="utf-8-sig")
fig, ax = plt.subplots(figsize=(6.2, 11))
v = np.log2(f1[CCS].values)
im = ax.imshow(v, cmap=div, norm=TwoSlopeNorm(0, -2, 2), aspect="auto")
ax.set_xticks(range(5)); ax.set_xticklabels(CCS); ax.xaxis.tick_top()
ax.set_yticks(range(len(f1))); ax.set_yticklabels([f"{k} {n[:16]}" for k, n in zip(f1.index, f1.name)], fontsize=6.5)
for i in range(v.shape[0]):
    for j in range(5):
        ax.text(j, i, f"{f1[CCS].values[i, j]:.2f}", ha="center", va="center", fontsize=5.5, color=INK if abs(v[i, j]) < 1.2 else "white")
for s in ax.spines.values(): s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("RTA (log2, 0 = 5개국 평균)")
ax.set_title("그림 A. 50개 KSIC × 5개국 현시기술비교우위(RTA, 출원 2017–19, 동일가중)\n한국 RTA 내림차순", fontsize=9, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_rta_map_P3.png"), dpi=200); plt.close(fig)

# F2 기술(특허 RTA) × 수출(RCA) 사분면
q = pd.read_csv(os.path.join(OUT, "m4_quadrant_table.csv"), dtype={"ksic": str}).set_index("ksic")
f2 = q[["name", "pat_rta_P2", "exp_rca_P2", "kr_origin_abroad_rta_P3", "quad_tech_export_P2", "cw_group"]].dropna(subset=["exp_rca_P2"])
f2.to_csv(os.path.join(OUT, "fig2_quadrant_tech_export.csv"), encoding="utf-8-sig")
fig, ax = plt.subplots(figsize=(7.5, 6))
col = np.where(f2.quad_tech_export_P2.str.contains("괴리"), ORANGE, BLUE)
ax.axvline(1, color=GRAY, lw=1); ax.axhline(1, color=GRAY, lw=1)
ax.scatter(f2.pat_rta_P2, f2.exp_rca_P2, s=34, c=col, edgecolor="white", linewidth=1.5, zorder=3)
ax.set_xscale("log"); ax.set_yscale("log")
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
for axis, tk in [(ax.xaxis, [0.2, 0.3, 0.5, 1, 1.5, 2]), (ax.yaxis, [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 3])]:
    axis.set_major_locator(FixedLocator(tk)); axis.set_major_formatter(FixedFormatter([str(t) for t in tk])); axis.set_minor_locator(NullLocator())
lab = f2.index[(f2.quad_tech_export_P2.str.contains("괴리")) | (f2.exp_rca_P2 > 2.5) | (f2.pat_rta_P2 < 0.45) | (f2.exp_rca_P2 < 0.05)]
for k in lab:
    ax.annotate(f"{k} {f2.at[k, 'name'][:10]}", (f2.at[k, "pat_rta_P2"], f2.at[k, "exp_rca_P2"]), fontsize=6, xytext=(3, 2), textcoords="offset points", color="#52514e")
for (x, y, t) in [(1.7, 2.9, "기술 강·수출 강"), (1.7, 0.03, "기술 강·수출 약"), (0.25, 2.9, "기술 약·수출 강"), (0.25, 0.03, "기술 약·수출 약")]:
    ax.text(x, y, t, fontsize=8, color="#52514e", ha="center")
ax.set_xlabel("한국 특허 RTA (5개 관청, 출원 2015–17)"); ax.set_ylabel("한국 수출 RCA (KR·US·JP·CN·DE, 2015–17)")
ax.set_title("그림 B. 같은 KSIC 5자리로 이은 기술 포지션과 수출 포지션 (48개 KSIC)\n주황 = 괴리 사분면, Spearman ρ = 0.36 [0.03, 0.63]", fontsize=9, loc="left")
for s in ["top", "right"]: ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_quadrant_tech_export.png"), dpi=200); plt.close(fig)

# F3 선행-후행 패널 β (L=1): 원 계열 vs 우선권 경로 제거
a = pd.read_csv(os.path.join(OUT, "m3_panel_leadlag.csv")); a["series"] = "원 계열"
b = pd.read_csv(os.path.join(OUT, "m3b_panel_leadlag_origin_purged.csv")); b["series"] = "우선권 경로 제거"
f3 = pd.concat([a, b]); f3 = f3[(f3.period == "dev_2004_2018") & (f3.L == 1)].copy()
f3["pair"] = f3.lead + "→" + f3.follow
f3[["pair", "series", "beta", "se", "p_cluster", "p_perm", "q_bh_within_period", "n"]].to_csv(os.path.join(OUT, "fig3_leadlag_L1.csv"), index=False, encoding="utf-8-sig")
order = ["US→KR", "JP→KR", "CN→KR", "EP→KR", "KR→US", "KR→JP", "KR→CN", "KR→EP"]
fig, ax = plt.subplots(figsize=(7.5, 3.8))
x = np.arange(len(order)); w = 0.36
for i, (s, c) in enumerate([("원 계열", BLUE), ("우선권 경로 제거", ORANGE)]):
    d = f3[f3.series == s].set_index("pair").reindex(order)
    ax.bar(x + (i - 0.5) * w, d.beta, w - 0.04, color=c, label=s, zorder=3)
    ax.errorbar(x + (i - 0.5) * w, d.beta, yerr=1.96 * d.se, fmt="none", ecolor="#52514e", lw=0.8, zorder=4)
ax.axhline(0, color="#8a8984", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(order)
ax.set_ylabel("β (1년 시차, Δlog 국가 내 비중)"); ax.legend(frameon=False, fontsize=7)
ax.set_title("그림 C. 1년 선행 계수: 우선권(해외 재출원) 경로를 빼면 한국→US·EP 선행이 줄어든다\n개발 2004–18, 막대 = KSIC 군집 SE 95% CI(유의성 판정은 순열 p·BH 기준, fig3_leadlag_L1.csv)", fontsize=8.5, loc="left")
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#e6e5e1", lw=0.6, zorder=0)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_leadlag_L1.png"), dpi=200); plt.close(fig)

# F4 산업 대시보드 백분위 히트맵
D = pd.read_csv(os.path.join(OUT, "m5_industry_dashboard.csv"), dtype={"ksic": str}).set_index("ksic")
cols = [("kr_rta_P3", "한국 RTA"), ("kr_origin_abroad_rta_P3", "해외출원 RTA"), ("T3_breakout_cell_rate", "T3 부상셀"), ("T1_index_share", "T1 진입대기"),
        ("N9_exp_pos_rate", "N9 R&D성과"), ("N6_top10_share", "N6 첫특허"), ("N7_top10_share", "N7 R&D진입"),
        ("N8_exp_disputes_per1000", "N8 분쟁"), ("T4_lapse_risk_3y", "T4 포기")]
P = pd.DataFrame({lab: D[c].rank(pct=True) for c, lab in cols})
P = P.loc[D.kr_rta_P3.sort_values(ascending=False).index]
P.insert(0, "name", D.name.reindex(P.index)); P.to_csv(os.path.join(OUT, "fig4_dashboard_percentiles.csv"), encoding="utf-8-sig")
fig, ax = plt.subplots(figsize=(7.2, 11))
vals = P.drop(columns="name").values
ax.imshow(vals, cmap=seq, vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(vals.shape[1])); ax.set_xticklabels(P.columns[1:], rotation=40, ha="left", fontsize=7); ax.xaxis.tick_top()
ax.set_yticks(range(len(P))); ax.set_yticklabels([f"{k} {n[:16]}" for k, n in zip(P.index, P.name)], fontsize=6.5)
ax.axvline(1.5, color="white", lw=3)
for s in ax.spines.values(): s.set_visible(False)
ax.set_title("그림 D. 50개 산업 대시보드(초안)\n지형도 2열 + 트랙 미래 산출물 7열, 50개 산업 내 백분위(진할수록 높음, 순위 비교용)", fontsize=8.5, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_dashboard_percentiles.png"), dpi=200); plt.close(fig)
print("figures done")
