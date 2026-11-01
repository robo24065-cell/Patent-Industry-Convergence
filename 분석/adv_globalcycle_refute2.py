# -*- coding: utf-8 -*-
"""
적대적 검증 2 — 세 가지 결정적 확인
  S1. over1(PADI 깃발) 은 전체표본에서 예측에 기여하는가 (한계 LOCO)
  S2. 저자의 플라시보는 '무작위 절반'이었다. 그러나 동조도 상위층은
      24행이 k3 8개에서 나온다(중복 3배). 구조를 맞춘 플라시보로 다시 비교한다.
  S3. H-B 를 실제 지목 대상(26x 반도체·디스플레이)으로 좁혔을 때의 상호작용
  S4. 분할연도별 원자료 — '7분할 평균'이 감춘 시간 추세
"""
import os, sys, glob, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
exec(open(os.path.join(r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전", "분석",
                       "adv_globalcycle_refute.py"), encoding="utf-8").read().split("log(\"=\"*80)")[0])

SPLITS = list(range(2013, 2020))
DVCOL = "dd_ship_late"
PANELS = {s: panel_for(s) for s in SPLITS}
NEW8 = [26111, 26121, 26211, 26212, 26519, 27199, 27301, 28111]
SEMI5 = [26111, 26121, 26211, 26212, 26519]
log = lambda m: print(m, flush=True)

log("="*80); log("[S1] over1 의 한계 기여 — 전체 47표본"); log("="*80)
a, b = [], []
for s in SPLITS:
    d = PANELS[s]; y = d[DVCOL].values; g = d.mids.values
    a.append(loco(d[["dd_prod_early"]].values, y, g))
    b.append(loco(d[["dd_prod_early", "over1"]].values, y, g))
log(f"  {'split':>6s} {'LOCO(지속성만)':>13s} {'LOCO(+over1)':>13s} {'Δ':>7s}")
for i, s in enumerate(SPLITS):
    log(f"  {s:>6d} {a[i]:+13.3f} {b[i]:+13.3f} {b[i]-a[i]:+7.3f}")
log(f"  {'평균':>6s} {np.mean(a):+13.3f} {np.mean(b):+13.3f} {np.mean(b)-np.mean(a):+7.3f}")

log("\n"+"="*80)
log("[S2] ★ 구조 맞춘 플라시보 — 동조도 상위층은 k3 8개 / 24행 (중복 3.0배)")
log("="*80)
d17 = PANELS[2017]
k3sz = d17.groupby("k3s").size()
tgt_k3, tgt_rows = 8, 24
log(f"  관측 동조도 상위층: k3 {tgt_k3}개, 행 {tgt_rows}, mid 5개, LOCO 평균 -0.033")
log("  귀무 1: 저자 방식 = 47행에서 무작위 절반 (k3 중복 구조 무시)")
log("  귀무 2: k3 블록 8개를 무작위 추출 (관측 층과 같은 중복 구조)")
rng = np.random.default_rng(777)
allk3 = sorted(d17.k3s.unique())


def loco_mean_rows(mask_fn):
    v = []
    for s in SPLITS:
        d = PANELS[s]; dd = mask_fn(d)
        if len(dd) < 12 or dd.mids.nunique() < 4 or dd.over1.nunique() < 2: return np.nan
        L = loco(dd[["dd_prod_early", "over1"]].values, dd[DVCOL].values, dd.mids.values)
        if np.isnan(L): return np.nan
        v.append(L)
    return float(np.mean(v)) if v else np.nan


n1, n2 = [], []
for _ in range(1500):
    idx = rng.permutation(len(d17))[:tgt_rows]
    ks = set(d17.iloc[idx].ksic)
    m = loco_mean_rows(lambda d: d[d.ksic.isin(ks)])
    if not np.isnan(m): n1.append(m)
for _ in range(1500):
    sel = set(rng.choice(allk3, tgt_k3, replace=False).tolist())
    m = loco_mean_rows(lambda d: d[d.k3s.isin(sel)])
    if not np.isnan(m): n2.append(m)
n1, n2 = np.array(n1), np.array(n2)
obs = -0.033
for lab, arr in [("귀무1 무작위 24행", n1), ("귀무2 무작위 k3블록 8개", n2)]:
    log(f"    {lab:22s} 중위 {np.median(arr):+.3f} | 5% {np.percentile(arr,5):+.3f} | 25% {np.percentile(arr,25):+.3f} "
        f"| 유효 {len(arr)} → 관측 {obs:+.3f} 는 하위 {(arr<=obs).mean()*100:5.1f}%")

log("\n"+"="*80)
log("[S3] ★ H-B 를 실제 지목 대상(26x 반도체·디스플레이)으로 좁힌 검정")
log("="*80)
for lab, mk in [("semi5더미", lambda d: d.ksic.isin(SEMI5).astype(float)),
                ("k3=261·262·265더미", lambda d: d.k3s.isin(["261", "262", "265"]).astype(float)),
                ("mid26더미", lambda d: (d.mids == "26").astype(float))]:
    bs, ts, ls = [], [], []
    for s in SPLITS:
        d = PANELS[s].copy(); d["_m"] = mk(d); d["_mix"] = d._m*d.over1
        X = d[["dd_prod_early", "over1", "_m", "_mix"]].values; y = d[DVCOL].values
        bb, se_c, se_h, G = ols_cluster(X, y, d.k3s.values)
        bs.append(bb[4]); ts.append(bb[4]/se_c[4])
        ls.append(loco(X, y, d.mids.values))
    log(f"  {lab:20s} β_ix {np.mean(bs):+7.2f} (k3클러스터 t̄ {np.mean(ts):+5.2f}) LOCO 평균 {np.nanmean(ls):+.3f}"
        f"  [β_ix>0 = 그 집단에서 PADI 신호 약화 = H-B 방향]")

log("\n  집단별 over1 원자료 대조 (split=2017, n 작음 — 방향·크기만)")
d = PANELS[2017]
for lab, sel in [("26x 반도체디스플레이(261/262/265) 전체", d.k3s.isin(["261", "262", "265"])),
                 ("신규8 중 semi5", d.ksic.isin(SEMI5)),
                 ("신규8 중 기타3", d.ksic.isin([27199, 27301, 28111])),
                 ("그 외", ~d.k3s.isin(["261", "262", "265"]))]:
    dd = d[sel]
    if dd.over1.nunique() < 2:
        log(f"    {lab:34s} n={len(dd):2d} over1=1 {int(dd.over1.sum()):2d} → 대조 불가(한쪽뿐), 낙폭 평균 {dd[DVCOL].mean():+6.1f}")
        continue
    log(f"    {lab:34s} n={len(dd):2d} | over1=1 {dd.loc[dd.over1==1,DVCOL].mean():+6.1f}({int(dd.over1.sum())}개) "
        f"vs over1=0 {dd.loc[dd.over1==0,DVCOL].mean():+6.1f} → 차이 {dd.loc[dd.over1==1,DVCOL].mean()-dd.loc[dd.over1==0,DVCOL].mean():+6.1f}")

log("\n"+"="*80)
log("[S4] 분할연도별 원자료 — '7분할 평균'이 감춘 것")
log("="*80)
log(f"  {'split':>6s} {'LOCO':>7s} {'β_over1':>8s} {'OLS t':>6s} {'k3클t':>6s} {'y평균':>7s} {'y sd':>6s} {'over1=1':>7s}")
for s in SPLITS:
    d = PANELS[s]; X = d[["dd_prod_early", "over1"]].values; y = d[DVCOL].values
    L = loco(X, y, d.mids.values)
    bb, se_c, se_h, G = ols_cluster(X, y, d.k3s.values)
    log(f"  {s:>6d} {L:+7.3f} {bb[2]:+8.2f} {bb[2]/se_h[2]:+6.2f} {bb[2]/se_c[2]:+6.2f} "
        f"{y.mean():+7.1f} {y.std(ddof=1):6.1f} {int(d.over1.sum()):7d}")
log("\n완료")
