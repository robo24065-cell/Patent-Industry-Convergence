# -*- coding: utf-8 -*-
"""
선행성 검정 결과의 '좌측절단(left-censoring)' 진단

이륙연도 비교에서 무역이 특허보다 먼저라는 (가설과 반대) 결과가 나왔다.
그러나 두 계열의 관측 시작점이 대칭이 아니다:
  - 특허측 중국비중은 2000년에 평균 2.8% → 진짜 이륙 前 기저가 표본 안에 있다.
  - 무역측 중국비중은 2000년에 이미 상당히 높을 수 있다(1990년대에 이미 이륙).
    이 경우 2000~2003 '기저'가 기저가 아니라 이미 상승 중인 구간이고,
    이륙연도가 기계적으로 표본 시작쪽으로 당겨진다.

이 스크립트는 그 절단이 실제로 있는지, 있다면 결과를 얼마나 좌우하는지 본다.
결과를 각색하지 않는다.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
F = os.path.join(BASE, "분석", "fusion")
log = lambda m: print(m, flush=True)

Q = pd.read_csv(os.path.join(F, "leadlag_panel.csv"), encoding="utf-8-sig",
                dtype={"ksic": str, "k3": str, "mid": str})
Q = Q[Q.in39.astype(str).isin(["True", "true", "1"])].copy()
tk = pd.read_csv(os.path.join(F, "leadlag_takeoff.csv"), encoding="utf-8-sig",
                 dtype={"unit": str})
tk = tk[(tk.level == "ksic5") & (tk.frac == "중간점")].dropna(subset=["gap"])

log("=" * 72)
log("[1] 2000년 시점의 두 계열 수준 — 기저가 진짜 기저인가")
s0 = Q[Q.year.between(2000, 2003)].groupby("ksic").agg(
    pat0=("cn_pat_share", "mean"), imp0=("cn_share_imp", "mean"))
s9 = Q[Q.year.between(2019, 2022)].groupby("ksic").agg(
    pat9=("cn_pat_share", "mean"), imp9=("cn_share_imp", "mean"))
s = s0.join(s9)
s["pat_rise"] = s.pat9 - s.pat0
s["imp_rise"] = s.imp9 - s.imp0
# 기저가 최종수준 대비 얼마나 낮은가 = 이륙 前 구간을 관측했는지의 지표
s["pat_base_frac"] = s.pat0 / s.pat9
s["imp_base_frac"] = s.imp0 / s.imp9
log(f"  특허 중국비중: 2000~03 평균 {s.pat0.mean():.1f}% → 2019~22 평균 {s.pat9.mean():.1f}%"
    f"  (기저/정점 = {s.pat_base_frac.mean():.2f})")
log(f"  무역 중국비중: 2000~03 평균 {s.imp0.mean():.1f}% → 2019~22 평균 {s.imp9.mean():.1f}%"
    f"  (기저/정점 = {s.imp_base_frac.mean():.2f})")
log("  → 기저/정점 비율이 높을수록 '이륙이 표본 시작 前에 이미 진행됨'을 뜻한다.")

log("")
log("[2] 무역측 이륙연도가 표본 시작(2000~2003)에 몰려 있는가")
early = tk[tk.t_imp <= 2003.99]
log(f"  무역 이륙연도 ≤2003.99 인 산업: {len(early)}/{len(tk)} "
    f"({len(early)/len(tk):.0%})")
log(f"  특허 이륙연도 ≤2003.99 인 산업: {(tk.t_pat <= 2003.99).sum()}/{len(tk)}")
log("  (무역측만 시작점에 뭉쳐 있으면 절단 편의의 전형적 징후)")

log("")
log("[3] 절단 의심 산업을 제외하고 재검정")
log("    기준: 무역 기저(2000~03)가 정점(2019~22)의 60% 미만인 산업만 사용")
log("          = 관측구간 안에서 실제로 이륙한 산업")
keep = s[(s.imp_base_frac < 0.60) & (s.imp_rise > 0)].index
sub = tk[tk.unit.isin(keep)]
log(f"    잔존 산업 {len(sub)}개")
if len(sub) >= 5:
    d = sub.gap
    t = stats.ttest_1samp(d, 0)
    w = stats.wilcoxon(d) if len(d) >= 6 else (np.nan, np.nan)
    log(f"    격차(무역−특허) 중앙값 {d.median():+.2f}년, 평균 {d.mean():+.2f}년, "
        f"특허 선행 비율 {(d > 0).mean():.1%}")
    log(f"    t={t.statistic:+.2f} p={t.pvalue:.4f}"
        + (f" | Wilcoxon p={w[1]:.4f}" if np.isfinite(w[1]) else ""))
else:
    log("    표본부족")

log("")
log("[4] 절단에 둔감한 대안 — '정점 도달 시점' 비교")
log("    각 계열이 자기 최종수준의 90%에 처음 도달한 연도 (기저 추정 불필요)")


def reach(g, col, frac=0.90):
    g = g.dropna(subset=[col]).sort_values("year")
    if len(g) < 8:
        return np.nan
    hi = g[g.year.between(2019, 2022)][col].mean()
    if not np.isfinite(hi) or hi <= 0:
        return np.nan
    thr = frac * hi
    y, v = g.year.values.astype(float), g[col].values.astype(float)
    i = np.where(v >= thr)[0]
    return y[i[0]] if len(i) else np.nan


rows = []
for u, g in Q.groupby("ksic"):
    rows.append({"ksic": u, "rp": reach(g, "cn_pat_share"),
                 "ri": reach(g, "cn_share_imp")})
r = pd.DataFrame(rows).dropna()
r["gap"] = r.ri - r.rp
log(f"    유효 {len(r)}개, 격차(무역−특허) 중앙값 {r.gap.median():+.2f}년, "
    f"평균 {r.gap.mean():+.2f}년, 특허 선행 비율 {(r.gap > 0).mean():.1%}")
tt = stats.ttest_1samp(r.gap, 0)
log(f"    t={tt.statistic:+.2f} p={tt.pvalue:.4f}")

log("")
log("[5] 결론 재료")
log(f"    - 원 검정(중간점): 격차 중앙값 {tk.gap.median():+.2f}년 (특허 선행 비율 "
    f"{(tk.gap > 0).mean():.0%})")
log(f"    - 무역계열 기저/정점 평균 {s.imp_base_frac.mean():.2f} vs 특허 "
    f"{s.pat_base_frac.mean():.2f}")
r.to_csv(os.path.join(F, "leadlag_diag_reach.csv"), index=False, encoding="utf-8-sig")
s.round(3).to_csv(os.path.join(F, "leadlag_diag_levels.csv"), encoding="utf-8-sig")
log("    저장: leadlag_diag_levels.csv, leadlag_diag_reach.csv")
