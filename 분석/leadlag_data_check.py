# -*- coding: utf-8 -*-
"""
선행성(lead-lag) 검정용 데이터 결합 가능성 점검 + 예비 교차상관
  좌: 분석/fusion/cn_apply_yearly.csv        (KSIC 5자리 × 연도, 중국 출원건수)
  우: 분석/fusion/cn_import_yearly.csv       (KSIC 2자리 × 연도, 대중 수입침투율 %p)
      분석/fusion/cn_import_yearly_ksic5.csv (KSIC 5자리 × 연도, 대중 수입액 USD, HS6 기반)

교차상관 규약: x(t-k) 와 y(t) 의 상관. k>0 이면 x가 y보다 k년 선행.
  두 방향으로 대칭이므로(역방향은 부호 k만 뒤집힘) '선행 증거'는
  k>0 쪽 상관이 k<0 쪽보다 유의하게 큰 '비대칭'으로만 주장할 수 있다.

절대 각색하지 않는다.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
F = os.path.join(BASE, "분석", "fusion")
log = lambda m: print(m, flush=True)

PAT_END = 2022          # 2023~2025 는 공개지연 절단 → 제외 (2024년 건수가 2023의 68%)

pat = pd.read_csv(os.path.join(F, "cn_apply_yearly.csv"), encoding="utf-8-sig",
                  dtype={"ksic": str})
allp = pd.read_csv(os.path.join(F, "apply_yearly_all.csv"), encoding="utf-8-sig",
                   dtype={"ksic": str})
imp2 = pd.read_csv(os.path.join(F, "cn_import_yearly.csv"), encoding="utf-8-sig",
                   dtype={"mid": str})
pat = pat[pat.year <= PAT_END].copy()
pat["mid"] = pat.ksic.str[:2]
tot = (allp[allp.year <= PAT_END].groupby(["ksic", "year"], as_index=False)
       .n_apply.sum().rename(columns={"n_apply": "n_all"}))
pat = pat.merge(tot, on=["ksic", "year"], how="left")


def xcorr(df, unit, xcol, ycol, lags=range(-4, 5), twoway=True):
    """x(t-k) vs y(t). twoway=True 면 산업·연도 이중 제거(공통충격 흡수)."""
    out = []
    for k in lags:
        z = df.sort_values([unit, "year"]).copy()
        z["xl"] = z.groupby(unit)[xcol].shift(k)
        z = z[[unit, "year", "xl", ycol]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(z) < 25:
            out.append((k, np.nan, len(z))); continue
        for c in ("xl", ycol):
            z[c] = z[c] - z.groupby(unit)[c].transform("mean")
            if twoway:
                z[c] = z[c] - z.groupby("year")[c].transform("mean")
        out.append((k, np.corrcoef(z.xl, z[ycol])[0, 1], len(z)))
    return out


def show(rows, tag):
    log(f"  {tag}")
    for k, r, n in rows:
        mark = ""
        if not np.isnan(r) and abs(r) > 0.15:
            mark = "   [특허 선행]" if k > 0 else ("   [수입 선행]" if k < 0 else "   [동시]")
        log(f"    k={k:+d}: r={r:+.3f} (n={n}){mark}")
    ok = [r for k, r, n in rows if k > 0 and not np.isnan(r)]
    no = [r for k, r, n in rows if k < 0 and not np.isnan(r)]
    if ok and no:
        log(f"    → 비대칭도: mean r(k>0)={np.mean(ok):+.3f} vs mean r(k<0)={np.mean(no):+.3f} "
            f"(차이 {np.mean(ok)-np.mean(no):+.3f})")


log("=" * 72)
log("[1] 결합 가능성 - KSIC 2자리(중분류) 축")
pm = (pat.groupby(["mid", "year"], as_index=False)
      .agg(cn_apply=("n_apply", "sum"), all_apply=("n_all", "sum")))
pm["cn_pat_share"] = pm.cn_apply / pm.all_apply * 100
mp_, mi_ = set(pm.mid), set(imp2.mid)
log(f"  특허측 중분류 {len(mp_)}개 / 수입측 {len(mi_)}개 / 교집합 {len(mp_ & mi_)}개")
log(f"  교집합: {sorted(mp_ & mi_)}   (특허측 미매칭 {sorted(mp_ - mi_)})")
P = pm.merge(imp2, on=["mid", "year"], how="inner").sort_values(["mid", "year"])
log(f"  결합 패널 {len(P)}행 = 중분류 {P.mid.nunique()} × 연도 {P.year.min()}~{P.year.max()}")
mid39 = set(pd.read_csv(os.path.join(F, "predictive_agg.csv"), encoding="utf-8-sig",
                        dtype={"mid": str}).mid.astype(str).str.zfill(2))
log(f"  기준모형(39산업) 중분류 {len(mid39)}개 중 결합 성공 {len(mid39 & set(P.mid))}개")

P["d_imp"] = P.groupby("mid").cn_imp_pen.diff()
P["g_pat"] = P.groupby("mid").cn_apply.transform(lambda s: np.log(s).diff())
P["d_share"] = P.groupby("mid").cn_pat_share.diff()
log("")
log("[2] 예비 교차상관 (중분류, 산업·연도 이중제거)")
show(xcorr(P, "mid", "g_pat", "d_imp"), "x=중국출원 증가율 → y=대중 침투율 변화(Δ%p)")
show(xcorr(P, "mid", "d_share", "d_imp"), "x=중국 출원비중 변화 → y=대중 침투율 변화(Δ%p)")

# ── KSIC5 해상도 ───────────────────────────────────────────────
f5 = os.path.join(F, "cn_import_yearly_ksic5.csv")
if not os.path.exists(f5):
    log("\n[3] KSIC5 파일 미생성 - build_cn_import_yearly.py 재실행 필요")
    raise SystemExit

log("")
log("=" * 72)
log("[3] 결합 가능성 - KSIC 5자리(세세분류) 축, HS6 기반")
i5 = pd.read_csv(f5, encoding="utf-8-sig", dtype={"ksic": str})
k5p, k5i = set(pat.ksic), set(i5.ksic)
log(f"  특허측 {len(k5p)} / 수입측 {len(k5i)} / 교집합 {len(k5p & k5i)}")
log(f"  미매칭(특허측): {sorted(k5p - k5i)}")
Q = pat.merge(i5, on=["ksic", "year"], how="inner").sort_values(["ksic", "year"])
Q["mid"] = Q.ksic.str[:2]
log(f"  결합 패널 {len(Q)}행 = 산업 {Q.ksic.nunique()} × 연도 {Q.year.min()}~{Q.year.max()}")
tgt39 = set(pd.read_csv(os.path.join(F, "predictive_agg.csv"), encoding="utf-8-sig",
                        dtype={"ksic": str}).ksic)
log(f"  기준모형 39산업 중 결합 성공 {len(tgt39 & set(Q.ksic))}개")

# 해상도 함정 회피 확인: 중분류 내 변동이 살아있는가
for yy in (2012, 2019):
    y0 = Q[Q.year == yy]
    if len(y0) > 5:
        s = y0.groupby("mid").cn_share_imp.std().dropna()
        log(f"  {yy}년 중분류 내 '수입 중 중국비중' 표준편차: 평균 {s.mean():.2f}%p, "
            f"0인 중분류 {int((s.fillna(0) < 1e-9).sum())}/{len(s)}")

Q["d_imp"] = Q.groupby("ksic").cn_share_imp.diff()
Q["g_impv"] = Q.groupby("ksic").cn_imp_usd.transform(
    lambda s: np.log(s.clip(lower=1)).diff())
Q["g_pat"] = Q.groupby("ksic").n_apply.transform(
    lambda s: np.log(s.clip(lower=1)).diff())
Q["cn_pat_share"] = Q.n_apply / Q.n_all * 100
Q["d_pshare"] = Q.groupby("ksic").cn_pat_share.diff()

log("")
log("[4] 예비 교차상관 (세세분류, 산업·연도 이중제거)")
show(xcorr(Q, "ksic", "g_pat", "d_imp"), "x=중국출원 증가율 → y=수입 중 중국비중 변화(Δ%p)")
show(xcorr(Q, "ksic", "g_pat", "g_impv"), "x=중국출원 증가율 → y=대중 수입액 증가율")
show(xcorr(Q, "ksic", "d_pshare", "d_imp"), "x=중국 출원비중 변화 → y=수입 중 중국비중 변화")

log("")
log("[5] 분포시차 회귀 (산업FE + 연도FE, mid 클러스터)")
R = Q.copy()
for k in range(1, 5):
    R[f"gp_l{k}"] = R.groupby("ksic").g_pat.shift(k)
R = R.replace([np.inf, -np.inf], np.nan)
d = R.dropna(subset=["d_imp", "gp_l1", "gp_l2", "gp_l3", "g_pat"])
try:
    m = smf.ols("d_imp ~ g_pat + gp_l1 + gp_l2 + gp_l3 + C(ksic) + C(year)",
                data=d).fit(cov_type="cluster", cov_kwds={"groups": d.mid})
    for v in ["g_pat", "gp_l1", "gp_l2", "gp_l3"]:
        log(f"    {v:8s} β={m.params[v]:+.4f}  t={m.tvalues[v]:+.2f}  p={m.pvalues[v]:.3f}")
    log(f"    n={int(m.nobs)}, R²={m.rsquared:.3f}, 클러스터 {d.mid.nunique()}개")
except Exception as e:
    log(f"    회귀 실패: {e}")
