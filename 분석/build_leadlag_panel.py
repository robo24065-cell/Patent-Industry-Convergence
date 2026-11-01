# -*- coding: utf-8 -*-
"""
선행성(lead-lag) 검정용 패널 구축 + 검정
  가설: 중국의 '특허 출원 급증'이 '대중 수입침투 상승'보다 시간적으로 앞선다.

입력
  분석/fusion/cn_apply_yearly.csv    중국 출원 (KSIC 5자리 × 연도)  [기존 재사용]
  분석/fusion/apply_yearly_all.csv   5개국 전체 출원 (정규화용)     [기존 재사용]
  분석/fusion/cn_import_yearly.csv   대중 수입침투 (KSIC 2자리)     [기존 재사용]
  분석/fusion/hs6_import_yearly.csv  HS6 수입액 (Comtrade)          [신규 수집]
  분석/fusion/ksic_hs6_map.csv       KSIC5 ↔ HS6 대조표             [기존 재사용]

산출
  분석/fusion/cn_import_yearly_ksic5.csv  대중 수입 (KSIC 5자리 × 연도)
  분석/fusion/leadlag_panel.csv           특허×수입 결합 패널 (검정 준비 완료)
  분석/fusion/leadlag_crosscorr.csv       교차상관 결과표

설계 원칙
  1) 시간분할: 횡단면으로 접는 스칼라 변수는 전반기(≤2016) 값만 사용한다.
     단 패널 자체는 선행성 식별을 위해 전 구간을 쓴다(종속변수 dd_late 와 무관한 별도 검정).
  2) 해상도: 2자리 축은 유효 클러스터(mid)와 단위가 같아 within 변동이 0 → 5자리 축이 본선.
  3) 모든 검정은 n=39 원자료 / k3 집계 n=23 두 버전 모두 보고.

교차상관 규약: corr( x(t-k), y(t) ). k>0 이면 x가 y보다 k년 선행.
  역방향과 대칭이므로 '선행 증거'는 k>0 쪽이 k<0 쪽보다 큰 '비대칭'으로만 주장 가능.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
F = os.path.join(BASE, "분석", "fusion")
log = lambda m: print(m, flush=True)

PAT_END = 2022        # 2023~ 는 특허 공개지연 절단 (2024 건수가 2023의 68%)
TRD_END = 2024

# ══════════════════════════════════════════════════════════════════
# 1. 특허 측 — 중국 출원 (KSIC5 × 연도) + 5개국 대비 점유율
# ══════════════════════════════════════════════════════════════════
pat = pd.read_csv(os.path.join(F, "cn_apply_yearly.csv"), encoding="utf-8-sig",
                  dtype={"ksic": str})
allp = pd.read_csv(os.path.join(F, "apply_yearly_all.csv"), encoding="utf-8-sig",
                   dtype={"ksic": str})
pat = pat[pat.year <= PAT_END].copy()
tot = (allp[allp.year <= PAT_END].groupby(["ksic", "year"], as_index=False)
       .n_apply.sum().rename(columns={"n_apply": "n_all"}))
pat = pat.merge(tot, on=["ksic", "year"], how="left")
pat["cn_pat_share"] = pat.n_apply / pat.n_all * 100
pat["mid"] = pat.ksic.str[:2]
log(f"[1] 특허: {len(pat)}행, KSIC5 {pat.ksic.nunique()}개, "
    f"{pat.year.min()}~{pat.year.max()}, 중국출원 {pat.n_apply.sum():,}건")

# ══════════════════════════════════════════════════════════════════
# 2. 수입 측 [B] — HS6 → KSIC5 (5자리 해상도, 본선)
# ══════════════════════════════════════════════════════════════════
hs_f = os.path.join(F, "hs6_import_yearly.csv")
mp = pd.read_csv(os.path.join(F, "ksic_hs6_map.csv"), encoding="utf-8-sig",
                 dtype={"hs6": str, "k5": str})
hs = pd.read_csv(hs_f, encoding="utf-8-sig", dtype={"hs6": str})
hs["hs6"] = hs.hs6.str.zfill(6)

# HS6가 여러 KSIC5 에 걸치면 균등 분할(1/n). 83%가 1:1 이라 영향은 제한적이지만
# 배타적 HS6 가 0인 산업쌍은 완전히 동일한 계열이 된다 → 아래에서 명시적으로 진단.
w = mp.groupby("hs6").k5.transform("nunique")
mp = mp.assign(wgt=1.0 / w)

piv = hs.pivot_table(index=["hs6", "year"], columns="partner",
                     values="value", aggfunc="sum").reset_index()
for c in ("CN", "WLD"):
    if c not in piv:
        piv[c] = np.nan
m = mp[["k5", "hs6", "wgt"]].drop_duplicates().merge(piv, on="hs6", how="inner")
m["CN"] = m.CN.fillna(0) * m.wgt
m["WLD"] = m.WLD.fillna(0) * m.wgt
B = (m.groupby(["k5", "year"], as_index=False)
       .agg(cn_imp_usd=("CN", "sum"), wld_imp_usd=("WLD", "sum"),
            n_hs6=("hs6", "nunique")))
B["cn_share_imp"] = np.where(B.wld_imp_usd > 0,
                             B.cn_imp_usd / B.wld_imp_usd * 100, np.nan)
B = B.rename(columns={"k5": "ksic"}).sort_values(["ksic", "year"])
B["ksic_level"] = 5
B = B[B.year <= TRD_END]
B.to_csv(os.path.join(F, "cn_import_yearly_ksic5.csv"), index=False,
         encoding="utf-8-sig")
log(f"[2] 수입(KSIC5): {len(B)}행, 산업 {B.ksic.nunique()}개, "
    f"{B.year.min()}~{B.year.max()}")

# ── 해상도 진단: 중분류 내 변동이 실제로 살아있는가 ────────────────
tg = pd.read_csv(os.path.join(F, "predictive_agg.csv"), encoding="utf-8-sig",
                 dtype={"ksic": str})
t39 = set(tg.ksic)
b39 = B[B.ksic.isin(t39)].copy()
b39["mid"] = b39.ksic.str[:2]
log(f"    39산업 중 5자리 수입계열 확보 {b39.ksic.nunique()}개 "
    f"(미확보 {sorted(t39 - set(B.ksic))})")
for yy in (2010, 2016):
    y0 = b39[b39.year == yy]
    s = y0.groupby("mid").cn_share_imp.std()
    multi = s[y0.groupby("mid").ksic.nunique() > 1]
    log(f"    {yy}년 중분류 내 '수입 중 중국비중' 표준편차: 평균 {multi.mean():.2f}%p "
        f"(복수산업 보유 중분류 {len(multi)}개, 편차 0 인 곳 "
        f"{int((multi.fillna(0) < 1e-9).sum())}개)")

# 완전 중복(배타적 HS6=0 → 동일 계열) 진단
piv16 = b39[b39.year == 2016].set_index("ksic").cn_share_imp.round(6)
dup = piv16[piv16.duplicated(keep=False)].sort_values()
log(f"    ※ 2016년 값이 다른 산업과 완전히 같은 산업(=배타적 HS6 없음) {len(dup)}개: "
    f"{list(dup.index)}")
log(f"    → 실질 고유 수입계열 수 = {piv16.nunique()}개 / 명목 {len(piv16)}개")

# ══════════════════════════════════════════════════════════════════
# 3. 수입 측 [A] — KOSIS 2자리 (개념적으로 정확한 '국내공급 대비 침투율')
# ══════════════════════════════════════════════════════════════════
imp2 = pd.read_csv(os.path.join(F, "cn_import_yearly.csv"), encoding="utf-8-sig",
                   dtype={"mid": str})
imp2["mid"] = imp2.mid.str.zfill(2)
log(f"[3] 수입(KSIC2): {len(imp2)}행, 중분류 {imp2.mid.nunique()}개, "
    f"{imp2.year.min()}~{imp2.year.max()}")

# ══════════════════════════════════════════════════════════════════
# 4. 결합
# ══════════════════════════════════════════════════════════════════
# (a) 5자리 축
Q = pat.merge(B, on=["ksic", "year"], how="inner").sort_values(["ksic", "year"])
Q["mid"] = Q.ksic.str[:2]
log(f"[4a] 5자리 결합 패널 {len(Q)}행 = 산업 {Q.ksic.nunique()} × "
    f"{Q.year.min()}~{Q.year.max()}")
log(f"     기준모형 39산업 중 결합 성공 {len(t39 & set(Q.ksic))}개")

# (b) 2자리 축
pm = (pat.groupby(["mid", "year"], as_index=False)
      .agg(cn_apply=("n_apply", "sum"), all_apply=("n_all", "sum")))
pm["cn_pat_share"] = pm.cn_apply / pm.all_apply * 100
P = pm.merge(imp2, on=["mid", "year"], how="inner").sort_values(["mid", "year"])
mid39 = set(tg.ksic.str[:2])
log(f"[4b] 2자리 결합 패널 {len(P)}행 = 중분류 {P.mid.nunique()} × "
    f"{P.year.min()}~{P.year.max()}")
log(f"     39산업의 중분류 {len(mid39)}개 중 결합 성공 {len(mid39 & set(P.mid))}개")

# ── 파생변수 ───────────────────────────────────────────────────────
Q["g_pat"] = Q.groupby("ksic").n_apply.transform(lambda s: np.log(s.clip(lower=1)).diff())
Q["d_pshare"] = Q.groupby("ksic").cn_pat_share.diff()
Q["d_imp"] = Q.groupby("ksic").cn_share_imp.diff()
Q["g_impv"] = Q.groupby("ksic").cn_imp_usd.transform(lambda s: np.log(s.clip(lower=1)).diff())
Q["in39"] = Q.ksic.isin(t39)
# k3 집계용 키
k3map = tg.set_index("ksic").k3.astype(str).to_dict()
Q["k3"] = Q.ksic.map(k3map)
Q.to_csv(os.path.join(F, "leadlag_panel.csv"), index=False, encoding="utf-8-sig")
log(f"[5] 저장 leadlag_panel.csv : {len(Q)}행")

P["g_pat"] = P.groupby("mid").cn_apply.transform(lambda s: np.log(s.clip(lower=1)).diff())
P["d_pshare"] = P.groupby("mid").cn_pat_share.diff()
P["d_imp"] = P.groupby("mid").cn_imp_pen.diff()
P.to_csv(os.path.join(F, "leadlag_panel_mid.csv"), index=False, encoding="utf-8-sig")


# ══════════════════════════════════════════════════════════════════
# 5. 교차상관 (산업FE + 연도FE 제거 = 공통충격 흡수)
# ══════════════════════════════════════════════════════════════════
def xcorr(df, unit, xcol, ycol, lags=range(-4, 5)):
    out = []
    for k in lags:
        z = df.sort_values([unit, "year"]).copy()
        z["xl"] = z.groupby(unit)[xcol].shift(k)
        z = z[[unit, "year", "xl", ycol]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(z) < 25 or z[unit].nunique() < 3:
            out.append((k, np.nan, len(z))); continue
        for c in ("xl", ycol):
            z[c] = z[c] - z.groupby(unit)[c].transform("mean")
            z[c] = z[c] - z.groupby("year")[c].transform("mean")
        sd = z.xl.std() * z[ycol].std()
        out.append((k, np.corrcoef(z.xl, z[ycol])[0, 1] if sd > 0 else np.nan, len(z)))
    return out


def show(rows, tag, store, key):
    log(f"  {tag}")
    for k, r, n in rows:
        mk = ""
        if not np.isnan(r) and abs(r) > 0.15:
            mk = "  [특허 선행]" if k > 0 else ("  [수입 선행]" if k < 0 else "  [동시]")
        log(f"    k={k:+d}: r={r:+.3f} (n={n}){mk}")
        store.append({"spec": key, "k": k, "r": r, "n": n})
    a = [r for k, r, n in rows if k > 0 and not np.isnan(r)]
    b = [r for k, r, n in rows if k < 0 and not np.isnan(r)]
    if a and b:
        log(f"    → 비대칭도 mean r(k>0)={np.mean(a):+.3f} vs mean r(k<0)={np.mean(b):+.3f}"
            f"  (차이 {np.mean(a) - np.mean(b):+.3f})")


store = []
log("")
log("=" * 72)
log("[6] 교차상관 — (a) n=39 원자료 (KSIC5)")
Q39 = Q[Q.in39]
show(xcorr(Q39, "ksic", "g_pat", "d_imp"), "x=중국출원 증가율 → y=수입 중 중국비중 Δ%p",
     store, "39_gpat_dimp")
show(xcorr(Q39, "ksic", "g_pat", "g_impv"), "x=중국출원 증가율 → y=대중 수입액 증가율",
     store, "39_gpat_gimpv")
show(xcorr(Q39, "ksic", "d_pshare", "d_imp"), "x=중국 출원비중 Δ → y=수입 중 중국비중 Δ",
     store, "39_dpsh_dimp")

log("")
log("[7] 교차상관 — (b) k3 집계 n=23")
Q3 = Q[Q.in39 & Q.k3.notna()].groupby(["k3", "year"], as_index=False).agg(
    n_apply=("n_apply", "sum"), n_all=("n_all", "sum"),
    cn_imp_usd=("cn_imp_usd", "sum"), wld_imp_usd=("wld_imp_usd", "sum"))
Q3["cn_share_imp"] = np.where(Q3.wld_imp_usd > 0, Q3.cn_imp_usd / Q3.wld_imp_usd * 100, np.nan)
Q3["cn_pat_share"] = Q3.n_apply / Q3.n_all * 100
Q3 = Q3.sort_values(["k3", "year"])
Q3["g_pat"] = Q3.groupby("k3").n_apply.transform(lambda s: np.log(s.clip(lower=1)).diff())
Q3["d_imp"] = Q3.groupby("k3").cn_share_imp.diff()
Q3["g_impv"] = Q3.groupby("k3").cn_imp_usd.transform(lambda s: np.log(s.clip(lower=1)).diff())
Q3["d_pshare"] = Q3.groupby("k3").cn_pat_share.diff()
log(f"  k3 단위 {Q3.k3.nunique()}개 × {Q3.year.min()}~{Q3.year.max()}")
show(xcorr(Q3, "k3", "g_pat", "d_imp"), "x=중국출원 증가율 → y=수입 중 중국비중 Δ%p",
     store, "k3_gpat_dimp")
show(xcorr(Q3, "k3", "g_pat", "g_impv"), "x=중국출원 증가율 → y=대중 수입액 증가율",
     store, "k3_gpat_gimpv")

log("")
log("[8] 교차상관 — (c) KSIC2 축, 개념상 정확한 침투율 (2010~2025, 참고)")
show(xcorr(P, "mid", "g_pat", "d_imp"), "x=중국출원 증가율 → y=대중 침투율 Δ%p",
     store, "mid_gpat_dimp")

pd.DataFrame(store).to_csv(os.path.join(F, "leadlag_crosscorr.csv"), index=False,
                           encoding="utf-8-sig")


# ══════════════════════════════════════════════════════════════════
# 6. 분포시차 회귀 (산업FE + 연도FE, mid 클러스터)
# ══════════════════════════════════════════════════════════════════
def dlreg(df, unit, ycol, xcol, cluster, tag, nlag=3):
    R = df.sort_values([unit, "year"]).copy()
    for k in range(1, nlag + 1):
        R[f"x_l{k}"] = R.groupby(unit)[xcol].shift(k)
    R = R.replace([np.inf, -np.inf], np.nan)
    need = [ycol, xcol] + [f"x_l{k}" for k in range(1, nlag + 1)]
    d = R.dropna(subset=need + [cluster])
    if len(d) < 40:
        log(f"  {tag}: 표본부족 n={len(d)}"); return
    rhs = " + ".join([xcol] + [f"x_l{k}" for k in range(1, nlag + 1)])
    try:
        mm = smf.ols(f"{ycol} ~ {rhs} + C({unit}) + C(year)", data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d[cluster]})
    except Exception as e:
        log(f"  {tag}: 실패 {e}"); return
    log(f"  {tag}  (n={int(mm.nobs)}, 단위 {d[unit].nunique()}, "
        f"클러스터 {d[cluster].nunique()})")
    tot_l = 0.0
    for v in [xcol] + [f"x_l{k}" for k in range(1, nlag + 1)]:
        lab = "동시(k=0)" if v == xcol else f"시차 {v[-1]}년"
        log(f"    {lab:10s} β={mm.params[v]:+.4f}  t={mm.tvalues[v]:+.2f}  "
            f"p={mm.pvalues[v]:.3f}")
        if v != xcol:
            tot_l += mm.params[v]
    log(f"    시차합(1~{nlag}년) Σβ={tot_l:+.4f} vs 동시 β={mm.params[xcol]:+.4f}")


log("")
log("=" * 72)
log("[9] 분포시차 회귀")
Q39c = Q39.copy()
dlreg(Q39c, "ksic", "d_imp", "g_pat", "mid", "(a) n=39 · y=수입 중 중국비중 Δ%p")
dlreg(Q39c, "ksic", "g_impv", "g_pat", "mid", "(a) n=39 · y=대중 수입액 증가율")
Q3c = Q3.copy()
Q3c["cl"] = Q3c.k3.astype(str).str[:2]
dlreg(Q3c, "k3", "d_imp", "g_pat", "cl", "(b) k3 n=23 · y=수입 중 중국비중 Δ%p")
dlreg(Q3c, "k3", "g_impv", "g_pat", "cl", "(b) k3 n=23 · y=대중 수입액 증가율")
P2 = P.copy(); P2["cl"] = P2.mid
dlreg(P2, "mid", "d_imp", "g_pat", "cl", "(c) KSIC2 · y=대중 침투율 Δ%p")

log("")
log("완료")
