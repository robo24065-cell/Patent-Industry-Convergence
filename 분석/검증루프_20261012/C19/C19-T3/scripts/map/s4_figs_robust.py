# [C19-T3 사본] 원본: C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared_report\fusion_advantage\split34\s4_figs_robust.py — 경로만 바꿈(OUT/HERE→C19_OUT, META→C19_META, 상대 MC→원래 폴더)
# -*- coding: utf-8 -*-
r"""split34 4단계: 보강 검정 + 보고서용 그림 데이터.
(i)  2개 창 패널 그룹 안 정합: KR 출원 성장 ↔ KOSIS 성장, 창 [2008-09→2012-13], [2012-13→2016-17], 그룹×창 고정효과, 순열 5000.
(ii) 무학습 신호 점검(L=2, 개발): 자기 모멘텀 − 그룹 모멘텀으로 향후 3년 KR/5국 출원 성장의 그룹 안 순서 정확도.
(iii) 그림 데이터: 소형다중 추세(fig_smallmultiples_long.csv), 분화 지수 표(t4_divergence_index.csv), PNG.
개발 기간 2005–2017만 그림 본선에 쓰고, 2018–2022는 '사후' 열로 따로 둔다."""
import os as _os
_C19_OUT = _os.environ['C19_OUT']; _C19_META = _os.environ['C19_META']
import os, sys, json
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
HERE = _C19_OUT
MC = os.path.abspath(os.path.join(r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared_report\fusion_advantage\split34", r"..\..\.."))
BASE = os.path.abspath(os.path.join(MC, r"..\.."))
sys.path.insert(0, os.path.join(MC, r"_shared_report\label_validity"))
from lv_common import ksic_to_group
rng = np.random.default_rng(11)
P = pd.read_parquet(os.path.join(HERE, "panel_ksic_cc_year.parquet"))
P = P[(P.year >= 2000) & (P.year <= 2022)]
K50 = sorted(P.ksic.unique()); G = pd.Series({k: ksic_to_group(k) for k in K50}); gs = G.value_counts()
K34 = [k for k in K50 if gs[G[k]] > 1]
YR = list(range(2000, 2023))
tot = P.groupby(["ksic", "year"]).n.sum().unstack().reindex(index=K50, columns=YR, fill_value=0)
kr = P[P.cc == "KR"].groupby(["ksic", "year"]).n.sum().unstack().reindex(index=K50, columns=YR, fill_value=0)
kro = P.groupby(["ksic", "year"]).n_kro.sum().unstack().reindex(index=K50, columns=YR, fill_value=0)
ko = pd.read_csv(os.path.join(BASE, r"공모전데이터\작업데이터\kosis_광업제조업_50개산업.csv"), dtype={"세세분류코드": str})
NAME = ko.drop_duplicates("세세분류코드").set_index("세세분류코드").산업명
k10 = ko[(ko.기준 == "10차") & (ko.마스킹 == 0)]
XS = {nm: k10[k10.항목코드 == c].pivot_table(index="세세분류코드", columns="연도", values="값", aggfunc="sum").reindex(K50)
      for c, nm in (("T02", "종사자수"), ("T04", "출하액"), ("T01", "사업체수"))}
lm = lambda A, ys: np.log(A[ys].mean(axis=1) + 1)
res = {}

# (i) 2개 창 패널
W = [([2008, 2009], [2012, 2013]), ([2012, 2013], [2016, 2017])]
out = []
for pn, A in (("KR출원", kr), ("KR기원", kro), ("5국", tot)):
    for xn, X in XS.items():
        rows = []
        for wi, (a, b) in enumerate(W):
            gp = (lm(A, b) - lm(A, a)) / 4
            gx = (np.log(X[b].mean(axis=1)) - np.log(X[a].mean(axis=1))) / 4
            rows.append(pd.DataFrame({"k": K34, "w": wi, "g": G[K34].values, "p": gp[K34].values, "x": gx[K34].values}))
        d = pd.concat(rows).dropna()
        c = pd.factorize(d.g + "_" + d.w.astype(str))[0]
        dmv = lambda v: v - (np.bincount(c, weights=v) / np.bincount(c))[c]
        pv, xv = d.p.values, d.x.values
        r = np.corrcoef(dmv(pv), dmv(xv))[0, 1]
        cells = [np.where(c == x)[0] for x in np.unique(c)]
        sims = []
        for _ in range(5000):
            sp = pv.copy()
            for ix in cells:
                sp[ix] = pv[rng.permutation(ix)]
            sims.append(np.corrcoef(dmv(sp), dmv(xv))[0, 1])
        out.append({"특허": pn, "외부": xn, "n(KSIC×창)": len(d), "그룹×창 안 r": round(r, 3), "순열p(단측)": round(float(np.mean(np.array(sims) >= r - 1e-12)), 4)})
T5 = pd.DataFrame(out); T5.to_csv(os.path.join(HERE, "t2c_panel_two_windows.csv"), index=False, encoding="utf-8-sig")
print(T5.to_string())


# (ii) 무학습 신호 L=2
def pair_acc(pred, y, g):
    hit = n = 0
    for x in np.unique(g):
        ix = np.where(g == x)[0]
        for i in range(len(ix)):
            for j in range(i + 1, len(ix)):
                a, b = ix[i], ix[j]
                if y[a] != y[b]:
                    dp = pred[a] - pred[b]; hit += 0.5 if dp == 0 else float(np.sign(dp) == np.sign(y[a] - y[b])); n += 1
    return hit / n


out = []
for tn, A in (("KR", kr), ("5국", tot), ("KR기원", kro)):
    for L in (2, 0):
        for t0 in range(2006, 2020):
            e = t0 - L
            y = (lm(A, [t0 + 1, t0 + 2, t0 + 3]) - lm(A, [t0 - 2, t0 - 1, t0]))[K34].values
            mom = (lm(A, [e - 1, e]) - lm(A, [e - 3, e - 2]))[K34].values
            out.append({"목표": tn, "L": L, "t0": t0, "구간": "개발" if t0 + 3 <= 2017 else "사후분석",
                        "쌍순서_자기모멘텀(지속)": pair_acc(mom, y, G[K34].values), "쌍순서_역모멘텀(평균회귀)": pair_acc(-mom, y, G[K34].values)})
T6 = pd.DataFrame(out)
T6s = T6.groupby(["목표", "L", "구간"]).agg(원점수=("t0", "size"), 지속=("쌍순서_자기모멘텀(지속)", "mean"), 평균회귀=("쌍순서_역모멘텀(평균회귀)", "mean"),
                                         평균회귀_0_5초과=("쌍순서_역모멘텀(평균회귀)", lambda s: f"{int((s > 0.5).sum())}/{len(s)}")).round(3)
T6.to_csv(os.path.join(HERE, "t3c_nonfitted_momentum_by_origin.csv"), index=False, encoding="utf-8-sig")
T6s.to_csv(os.path.join(HERE, "t3c_nonfitted_momentum_summary.csv"), encoding="utf-8-sig")
print(T6s.to_string())

# (iii) 소형다중 long 데이터
rows = []
for k in K34:
    for y in YR:
        if y < 2005:
            continue
        rows.append({"grp": G[k], "ksic": k, "name": NAME.get(k, ""), "year": y, "구간": "개발" if y <= 2017 else "사후",
                     "n_5국": int(tot.at[k, y]), "n_KR": int(kr.at[k, y]), "n_KR기원": int(kro.at[k, y]),
                     "KR출원_지수2005": round(100 * (kr.at[k, y] + 1) / (kr.at[k, 2005] + 1), 1),
                     "5국_지수2005": round(100 * (tot.at[k, y] + 1) / (tot.at[k, 2005] + 1), 1),
                     "한국기원점유": round(kro.at[k, y] / max(tot.at[k, y], 1), 4),
                     "KOSIS_종사자_지수2008": round(100 * XS["종사자수"].at[k, y] / XS["종사자수"].at[k, 2008], 1) if y in XS["종사자수"].columns and pd.notna(XS["종사자수"].at[k, y]) else None,
                     "KOSIS_출하_지수2008": round(100 * XS["출하액"].at[k, y] / XS["출하액"].at[k, 2008], 1) if y in XS["출하액"].columns and pd.notna(XS["출하액"].at[k, y]) else None})
SM = pd.DataFrame(rows); SM.to_csv(os.path.join(HERE, "fig_smallmultiples_long.csv"), index=False, encoding="utf-8-sig")

# 분화 지수 표 (개발 2005–07 → 2015–17)
M = pd.read_csv(os.path.join(HERE, "ksic_metrics_dev.csv"), dtype={0: str}, index_col=0)
M.index = M.index.astype(str)
rows = []
for g in sorted(G[K34].unique()):
    m = M[M.grp == g]
    gk = [k for k in K34 if G[k] == g]
    grp_kr = (np.log(kr.loc[gk, [2015, 2016, 2017]].sum().mean() + 1) - np.log(kr.loc[gk, [2005, 2006, 2007]].sum().mean() + 1)) / 10
    dev = (m.g_KR - grp_kr).abs()
    # 그룹 안 쌍: KR 출원 성장 순서와 KOSIS 사업체·종사자 성장 순서 일치율
    def agree(xc):
        h = n = 0
        for i in range(len(gk)):
            for j in range(i + 1, len(gk)):
                a, b = m.loc[gk[i]], m.loc[gk[j]]
                if pd.notna(a[xc]) and pd.notna(b[xc]):
                    h += float(np.sign(a.gw_KR_lag0 - b.gw_KR_lag0) == np.sign(a[xc] - b[xc])); n += 1
        return f"{int(h)}/{n}"
    top, bot = m.g_KR.idxmax(), m.g_KR.idxmin()
    rows.append({"연계표코드": g, "구성원수": len(gk), "구성원": " · ".join(f"{k} {str(NAME.get(k,''))[:10]}" for k in gk),
                 "그룹_KR출원_연성장%": round(100 * grp_kr, 1),
                 "분화지수_평균절대편차%p": round(100 * dev.mean(), 1),
                 "KR출원성장_최고": f"{top} {100*m.g_KR[top]:+.1f}%", "KR출원성장_최저": f"{bot} {100*m.g_KR[bot]:+.1f}%",
                 "격차%p": round(100 * (m.g_KR.max() - m.g_KR.min()), 1),
                 "부호갈림": bool((m.g_KR > 0.005).any() and (m.g_KR < -0.005).any()),
                 "한국기원점유_범위%": f"{100*m['한국기원점유_2015_17'].min():.1f}–{100*m['한국기원점유_2015_17'].max():.1f}",
                 "KOSIS종사자성장_격차%p": round(100 * (m.x_g_종사자수.max() - m.x_g_종사자수.min()), 1),
                 "순서일치_KR출원vs사업체": agree("x_g_사업체수"), "순서일치_KR출원vs종사자": agree("x_g_종사자수")})
DI = pd.DataFrame(rows).sort_values("분화지수_평균절대편차%p", ascending=False)
DI.to_csv(os.path.join(HERE, "t4_divergence_index.csv"), index=False, encoding="utf-8-sig")
print(DI.drop(columns=["구성원"]).to_string())
tot_pairs = [tuple(map(int, s.split("/"))) for s in DI["순서일치_KR출원vs사업체"]]
tot_pairs2 = [tuple(map(int, s.split("/"))) for s in DI["순서일치_KR출원vs종사자"]]
res["쌍순서일치_사업체"] = f"{sum(a for a,b in tot_pairs)}/{sum(b for a,b in tot_pairs)}"
res["쌍순서일치_종사자"] = f"{sum(a for a,b in tot_pairs2)}/{sum(b for a,b in tot_pairs2)}"
print(res)
json.dump(res, open(os.path.join(HERE, "t4_meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# PNG 소형다중: KR 출원 지수(2005=100), 개발 기간, 그룹별 구성원 선
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in ("Malgun Gothic", "NanumGothic"):
    if any(f in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f; break
plt.rcParams["axes.unicode_minus"] = False
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
groups = list(DI.연계표코드)
fig, axes = plt.subplots(3, 4, figsize=(16, 11), sharex=True)
for ax, g in zip(axes.flat, groups):
    gk = [k for k in K34 if G[k] == g]
    for i, k in enumerate(gk):
        s = SM[(SM.ksic == k) & (SM.year <= 2017)]
        ax.plot(s.year, s.KR출원_지수2005, color=PAL[i], lw=2, label=f"{k} {str(NAME.get(k,''))[:9]}")
        ax.text(2017.3, s.KR출원_지수2005.iloc[-1], k, color="#333333", fontsize=8, va="center")
    ax.axhline(100, color="#bbbbbb", lw=0.8)
    ax.set_title(f"{g} (공식 연계표 1개 코드)", fontsize=10)
    ax.grid(axis="y", color="#eeeeee"); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_xlim(2005, 2019)
axes.flat[-1].axis("off")
axes.flat[-1].text(0, 0.6, "KR 출원 지수(2005=100)\n개발 기간 2005–2017\n같은 패널 = 공식 IPC-KSIC 연계표가\n구분하지 못하는 세세분류\n출처: 특허-산업 융합데이터(제공)", fontsize=10, va="top")
fig.suptitle("연계표 한 코드 안에서 갈라지는 세세분류 특허 추세 (분화 지수 큰 순)", fontsize=13)
fig.subplots_adjust(left=0.04, right=0.98, top=0.93, bottom=0.04, wspace=0.18, hspace=0.25)
fig.savefig(os.path.join(HERE, "fig_smallmultiples_KR.png"), dpi=130)
print("png saved")
