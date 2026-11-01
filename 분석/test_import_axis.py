# -*- coding: utf-8 -*-
"""
제3축 검정 — 수입침투율 (KOSIS 업종별 수입점유비)
사전 등록: 가설 — 수입침투율이 높거나 빠르게 상승한 산업일수록 침체가 깊다 (β<0)
한계 명시: KSIC 중분류(2자리) 해상도 → 우리 유효 클러스터(13개 중분류)와 거의 동일 단위.
          within-중분류 변동이 0이므로 클러스터 부트스트랩에서 검정력이 크게 떨어진다.
"""
import os, re, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
FUS = os.path.join(BASE, "분석", "fusion")
AX = os.path.join(BASE, "공모전데이터", "외부데이터", "제3축")
rng = np.random.default_rng(41)
log = lambda m: print(m, flush=True)

imp = pd.read_csv(os.path.join(AX, "수입점유비_업종별.csv"), encoding="utf-8-sig")
imp["yr"] = imp.PRD_DE.astype(int)
imp["val"] = pd.to_numeric(imp.DT, errors="coerce")
imp["mid"] = imp.C1.str.extract(r"C(\d{2})")[0]
imp = imp.dropna(subset=["val", "mid"])
log(f"수입점유비: 업종 {imp.mid.nunique()}개 × {imp.yr.min()}~{imp.yr.max()}")

SY = 2017
e = imp[imp.yr < SY].groupby("mid").agg(imp_lvl=("val", "mean")).reset_index()
# 전반기 상승 속도 (연간 기울기)
sl = []
for m, g in imp[imp.yr < SY].groupby("mid"):
    g = g.sort_values("yr")
    if len(g) >= 4:
        sl.append({"mid": m, "imp_slope": np.polyfit(g.yr, g.val, 1)[0]})
e = e.merge(pd.DataFrame(sl), on="mid", how="left")
log(f"전반기(~{SY-1}) 수입침투율: 평균 {e.imp_lvl.mean():.1f}%, 기울기 평균 {e.imp_slope.mean():+.3f}%p/년")
log("  상위3: " + str(e.nlargest(3, "imp_lvl")[["mid", "imp_lvl"]].round(1).to_dict("records")))
log("  하위3: " + str(e.nsmallest(3, "imp_lvl")[["mid", "imp_lvl"]].round(1).to_dict("records")))

A = pd.read_csv(os.path.join(FUS, "predictive_agg.csv"), encoding="utf-8-sig",
                dtype={"ksic": str, "mid": str})
A = A.dropna(subset=["dd_late", "padi_e", "cn_growth_e"]).merge(e, on="mid", how="left")
log(f"\n결합 표본 {len(A)}산업, 수입침투율 매칭 {A.imp_lvl.notna().sum()}개")


def boot(x, y, groups, n=3000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y)); x, y, g = x[ok], y[ok], np.asarray(groups)[ok]
    r0 = np.corrcoef(x, y)[0, 1]; bs = []
    uq = np.unique(g)
    for _ in range(n):
        pick = rng.choice(uq, len(uq), replace=True)
        i = np.concatenate([np.where(g == c)[0] for c in pick])
        if len(i) > 3 and np.std(x[i]) > 1e-9 and np.std(y[i]) > 1e-9:
            bs.append(np.corrcoef(x[i], y[i])[0, 1])
    return r0, np.percentile(bs, [2.5, 97.5])


log("\n[1] 단순 상관 (후반기 낙폭)")
for v, lab in [("imp_lvl", "수입침투율 수준"), ("imp_slope", "수입침투율 상승속도")]:
    d = A.dropna(subset=[v])
    r, ci = boot(d[v], d.dd_late, d.mid)
    log(f"  {lab:18s} r={r:+.3f} 클러스터CI[{ci[0]:+.3f},{ci[1]:+.3f}] "
        f"{'유의' if (ci[0]>0 or ci[1]<0) else '비유의'}")

log("\n[2] 기존 2축과의 독립성")
for v in ("imp_lvl", "imp_slope"):
    d = A.dropna(subset=[v])
    log(f"  corr({v}, PADI)={d[v].corr(d.padi_e):+.3f}  corr({v}, 중국증가율)={d[v].corr(d.cn_growth_e):+.3f}")

log("\n[3] 3축 모형 (2축 + 수입침투율)")
base_d = A.dropna(subset=["imp_lvl", "imp_slope"])
b = smf.ols("dd_late ~ padi_e + cn_growth_e", data=base_d).fit()
for v, lab in [("imp_lvl", "수입침투율 수준"), ("imp_slope", "상승속도")]:
    f = smf.ols(f"dd_late ~ padi_e + cn_growth_e + {v}", data=base_d).fit(
        cov_type="cluster", cov_kwds={"groups": base_d.mid})
    vif = 1 / (1 - smf.ols(f"{v} ~ padi_e + cn_growth_e", data=base_d).fit().rsquared)
    log(f"  +{lab:14s} ΔR²={f.rsquared-b.rsquared:+.3f} β={f.params[v]:+7.3f} "
        f"t={f.tvalues[v]:+5.2f} p={f.pvalues[v]:.3f} VIF={vif:.2f}")

log("\n[4] 결론")
d = A.dropna(subset=["imp_lvl"])
r, ci = boot(d.imp_lvl, d.dd_late, d.mid)
if ci[0] <= 0 <= ci[1]:
    log("  수입침투율은 클러스터 CI가 0을 포함 → 제3축으로 채택하지 않음")
    log("  (원인: KSIC 2자리 해상도 = 유효 클러스터 단위와 동일해 within 변동 0)")
else:
    log("  수입침투율 제3축 후보로 채택 가능")
A.to_csv(os.path.join(FUS, "with_import.csv"), index=False, encoding="utf-8-sig")
