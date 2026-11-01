# -*- coding: utf-8 -*-
"""
'중국 이륙(takeoff) 연도' 비교 — 특허가 무역보다 먼저인가?

아이디어: 두 계열 모두 S자 곡선을 그린다.
  특허측: cn_pat_share  = 5개국 출원 중 중국 비중(%)
  무역측: cn_share_imp  = 한국 수입 중 중국 비중(%)
각 산업에서 '기저(2000~2003 평균) → 정점(2019~2022 평균)' 의 중간점을 처음 넘은 연도를
이륙연도로 정의하고, (무역 이륙연도 − 특허 이륙연도) 의 분포를 본다.
양수면 특허가 먼저다.

강건성: 중간점 대신 25%/75% 분위 통과연도로도 반복.
표본: (a) n=39 원자료, (b) k3 집계 n=23 두 버전 모두.
쌍대비교이므로 산업별 수준차·추세차가 자동 상쇄된다.

절대 각색하지 않는다.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
F = os.path.join(BASE, "분석", "fusion")
log = lambda m: print(m, flush=True)

BASE_YRS = (2000, 2003)
PEAK_YRS = (2019, 2022)

Q = pd.read_csv(os.path.join(F, "leadlag_panel.csv"), encoding="utf-8-sig",
                dtype={"ksic": str, "k3": str, "mid": str})
tg = pd.read_csv(os.path.join(F, "predictive_agg.csv"), encoding="utf-8-sig",
                 dtype={"ksic": str, "k3": str})
Q = Q[Q.in39.astype(str).isin(["True", "true", "1"])].copy()
log(f"패널: {len(Q)}행, 산업 {Q.ksic.nunique()}개, {Q.year.min()}~{Q.year.max()}")


def takeoff(g, col, frac):
    """기저→정점 구간의 frac 지점을 처음(그리고 이후 유지되며) 넘은 연도. 선형보간."""
    g = g.dropna(subset=[col]).sort_values("year")
    if len(g) < 8:
        return np.nan
    b = g[g.year.between(*BASE_YRS)][col]
    p = g[g.year.between(*PEAK_YRS)][col]
    if len(b) == 0 or len(p) == 0:
        return np.nan
    lo, hi = b.mean(), p.mean()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo <= 1e-9:
        return np.nan                      # 상승하지 않은 산업은 이륙연도 정의 불가
    thr = lo + frac * (hi - lo)
    y, v = g.year.values.astype(float), g[col].values.astype(float)
    idx = np.where(v >= thr)[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    if i == 0:
        return y[0]
    # 선형보간으로 소수점 연도
    v0, v1 = v[i - 1], v[i]
    if v1 == v0:
        return y[i]
    return y[i - 1] + (thr - v0) / (v1 - v0) * (y[i] - y[i - 1])


def run(df, unit, tag):
    log("")
    log("=" * 72)
    log(f"[{tag}]  단위 {df[unit].nunique()}개")
    res = []
    for frac, fl in [(0.25, "25%"), (0.50, "중간점"), (0.75, "75%")]:
        rows = []
        for u, g in df.groupby(unit):
            tp = takeoff(g, "cn_pat_share", frac)
            ti = takeoff(g, "cn_share_imp", frac)
            rows.append({unit: u, "t_pat": tp, "t_imp": ti,
                         "gap": ti - tp if np.isfinite(tp) and np.isfinite(ti) else np.nan})
        r = pd.DataFrame(rows)
        d = r.gap.dropna()
        if len(d) < 3:
            log(f"  {fl:6s}: 유효 {len(d)}개 — 검정 불가"); continue
        w = stats.wilcoxon(d) if len(d) >= 6 else (np.nan, np.nan)
        t = stats.ttest_1samp(d, 0)
        log(f"  {fl:6s} 통과연도 격차(무역−특허): 유효 {len(d)}개, "
            f"중앙값 {d.median():+.2f}년, 평균 {d.mean():+.2f}년")
        log(f"          특허가 먼저인 비율 {(d > 0).mean():.1%}  "
            f"| t={t.statistic:+.2f} p={t.pvalue:.4f}"
            + (f" | Wilcoxon p={w[1]:.4f}" if np.isfinite(w[1]) else ""))
        r["frac"] = fl
        res.append(r)
    return pd.concat(res) if res else pd.DataFrame()


# (a) n=39
a = run(Q, "ksic", "(a) n=39 원자료 · KSIC 5자리")

# (b) k3 집계 n=23
Q3 = Q[Q.k3.notna()].groupby(["k3", "year"], as_index=False).agg(
    n_apply=("n_apply", "sum"), n_all=("n_all", "sum"),
    cn_imp_usd=("cn_imp_usd", "sum"), wld_imp_usd=("wld_imp_usd", "sum"))
Q3["cn_pat_share"] = Q3.n_apply / Q3.n_all * 100
Q3["cn_share_imp"] = np.where(Q3.wld_imp_usd > 0,
                              Q3.cn_imp_usd / Q3.wld_imp_usd * 100, np.nan)
b = run(Q3, "k3", "(b) k3 집계 n=23")

out = []
if len(a):
    a2 = a.rename(columns={"ksic": "unit"}); a2["level"] = "ksic5"; out.append(a2)
if len(b):
    b2 = b.rename(columns={"k3": "unit"}); b2["level"] = "k3"; out.append(b2)
if out:
    o = pd.concat(out)
    o.to_csv(os.path.join(F, "leadlag_takeoff.csv"), index=False, encoding="utf-8-sig")
    log("")
    log(f"저장 leadlag_takeoff.csv ({len(o)}행)")
    log("")
    log("중간점 기준 산업별 이륙연도 (n=39):")
    m = o[(o.level == "ksic5") & (o.frac == "중간점")].dropna(subset=["gap"])
    m = m.merge(tg[["ksic", "k3"]].rename(columns={"ksic": "unit"}), on="unit", how="left")
    for _, r in m.sort_values("gap", ascending=False).iterrows():
        log(f"    {r.unit}  특허 {r.t_pat:7.1f}  무역 {r.t_imp:7.1f}  격차 {r.gap:+6.1f}년")
