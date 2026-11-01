# -*- coding: utf-8 -*-
"""
새 모델 기준 진단 체계 + 방어 재검정
────────────────────────────────────────────────────────────────
새 모형: 후반기 출하 최대낙폭 = α + β₁·전반기 낙폭 + β₂·1[PADI>1]
         LOCO-CV R² 0.542, 증분 순열 p<0.0001

[진단 축 교체]
  구 축: PADI(연속) × 중국출원증가율  → '이중압박' 곱셈 가정이 검정에서 기각됨(+0.514 < +0.542)
  신 축: **과거 침체 이력(전반기 낙폭)** × **[PADI > 1]**

  이 조합이 정책적으로 훨씬 유용하다. 특히:
    「신규경고」= 과거 이력만 보면 안전한데 기술자산이 이탈 중인 산업
                 → 기존 정보(과거 낙폭)로는 절대 잡을 수 없는 산업이며,
                   우리 지표가 더하는 가치를 **명단으로** 보여준다.

[검정]
  Q1. 「신규경고」 산업이 실제로 후반기에 더 깊게 빠졌는가? (유형의 실효성)
  Q2. 새 사양에서도 재무·산업구조 대체설명이 차단되는가? (방어 재검정)
"""
import os, json, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
FUS  = os.path.join(BASE, "분석", "fusion")
SPLITS = list(range(2013, 2020))
rng = np.random.default_rng(2468)
log = lambda m: print(m, flush=True)
out = {}

ALL = pd.read_csv(os.path.join(FUS, "model_search_panel.csv"), encoding="utf-8-sig",
                  dtype={"ksic": str, "k3": str, "mid": str})
ALL["over1"] = (ALL.padi_e > 1.0).astype(float)
NM = pd.read_parquet(os.path.join(BASE, "공모전데이터", "작업데이터", "kr_registered.parquet"),
                     columns=["세세분류코드", "세세분류명"]).drop_duplicates()
NM.columns = ["ksic", "name"]
NM["ksic"] = NM.ksic.astype(str).str.strip()


def wildp(d, y, cols, var, B=2999, seed=0):
    r = np.random.default_rng(seed)
    fml = f"{y} ~ " + " + ".join(cols)
    f = smf.ols(fml, data=d).fit()
    t = f.params[var] / f.bse[var]
    rest = [c for c in cols if c != var]
    f0 = smf.ols(f"{y} ~ " + (" + ".join(rest) if rest else "1"), data=d).fit()
    r0, fit0 = f0.resid.values, f0.fittedvalues.values
    g = d["mid"].values; uq = np.unique(g); ts = []
    for _ in range(B):
        wm = dict(zip(uq, r.choice([-1., 1.], size=len(uq))))
        d2 = d.copy(); d2[y] = fit0 + r0 * np.array([wm[x] for x in g])
        try:
            fb = smf.ols(fml, data=d2).fit(); ts.append(fb.params[var] / fb.bse[var])
        except Exception:
            pass
    return f, float((np.abs(np.array(ts)) >= abs(t)).mean())


# ══════════════════════════════════════════════════════════════
log("[1] 새 4유형 진단 — 분할연도 2017 기준")
D = ALL[ALL.split == 2017].dropna(subset=["dd_ship_late", "dd_prod_early", "over1"]).copy()
med = D.dd_prod_early.median()
log(f"  과거 낙폭 중앙값 {med:+.1f}% 를 기준선으로 사용")


def typ(r):
    hist_bad = r.dd_prod_early < med          # 과거에 더 깊게 빠졌다
    if hist_bad and r.over1: return "만성취약"
    if hist_bad and not r.over1: return "순환형"
    if not hist_bad and r.over1: return "신규경고"
    return "안정형"


D["유형"] = D.apply(typ, axis=1)
D = D.merge(NM, on="ksic", how="left")
cnt = D.유형.value_counts()
log(f"  분포: {cnt.to_dict()}")

log("\n  [유형별 실제 후반기 낙폭 — 유형이 실효성이 있는가]")
g = D.groupby("유형").agg(n=("ksic", "size"), 과거낙폭=("dd_prod_early", "mean"),
                        후반낙폭=("dd_ship_late", "mean")).reset_index()
g["악화"] = g.후반낙폭 - g.과거낙폭
for r in g.sort_values("후반낙폭").itertuples():
    log(f"    {r.유형:8s} n={r.n:2d}  과거 {r.과거낙폭:+7.1f}%  후반 {r.후반낙폭:+7.1f}%  "
        f"악화 {r.악화:+6.1f}%p")

log("\n  [★ Q1 결정적 — 과거가 같은데 PADI만 다른 두 유형 비교]")
a = D[D.유형 == "신규경고"].dd_ship_late
b = D[D.유형 == "안정형"].dd_ship_late
ha = D[D.유형 == "신규경고"].dd_prod_early
hb = D[D.유형 == "안정형"].dd_prod_early
log(f"    신규경고 (n={len(a)}): 과거 {ha.mean():+.1f}% → 후반 {a.mean():+.1f}%")
log(f"    안정형   (n={len(b)}): 과거 {hb.mean():+.1f}% → 후반 {b.mean():+.1f}%")
log(f"    과거 차이 {ha.mean()-hb.mean():+.1f}%p  vs  후반 차이 {a.mean()-b.mean():+.1f}%p")
if len(a) > 2 and len(b) > 2:
    from scipy import stats
    t, p = stats.ttest_ind(a, b, equal_var=False)
    log(f"    후반 낙폭 차이 t={t:+.2f} p={p:.4f} "
        f"{'★ 과거가 비슷한데 PADI 로 갈린다' if p < .05 else '(비유의)'}")
    out["q1"] = {"newwarn_late": float(a.mean()), "stable_late": float(b.mean()),
                 "newwarn_early": float(ha.mean()), "stable_early": float(hb.mean()),
                 "t": float(t), "p": float(p)}

log("\n  [신규경고 산업 명단 — 정책적으로 가장 가치 있는 산출물]")
for r in D[D.유형 == "신규경고"].sort_values("dd_ship_late").itertuples():
    log(f"    {r.ksic} {str(r.name)[:22]:24s} PADI={r.padi_e:.3f} "
        f"과거 {r.dd_prod_early:+6.1f}% → 후반 {r.dd_ship_late:+6.1f}%")

# ══════════════════════════════════════════════════════════════
log("\n[2] ★ Q2 방어 재검정 — 새 사양에서 대체설명이 차단되는가")
S = pd.read_csv(os.path.join(FUS, "ctrl_structure.csv"), encoding="utf-8-sig", dtype={"ksic": str})
SV = [v for v in ["ln_str_capital", "str_va_ratio", "ln_str_estab_size",
                  "ln_str_labor_prod", "str_wage_share", "ln_str_capital_pw"] if v in S.columns]
F = pd.read_csv(os.path.join(FUS, "ctrl_financial.csv"), encoding="utf-8-sig", dtype={"ksic": str})
FV = [c for c in ["fin_debt_ratio", "fin_borrow_dep", "fin_opm", "fin_sales_growth"] if c in F.columns]
E = D.merge(S[["ksic"] + SV], on="ksic", how="left").merge(F[["ksic"] + FV], on="ksic", how="left")

BASECOLS = ["dd_prod_early", "over1"]
f0, p0 = wildp(E.dropna(subset=BASECOLS + ["dd_ship_late"]), "dd_ship_late", BASECOLS, "over1", seed=1)
log(f"  기준(새 모형)          β_[PADI>1]={f0.params.over1:+7.2f} R²={f0.rsquared:.3f} wild-p={p0:.4f}")

rows = []
for nm, ctrl in (("재무4", FV), ("구조6", SV), ("재무4+구조6", FV + SV)):
    d = E.dropna(subset=BASECOLS + ["dd_ship_late"] + ctrl)
    if len(d) < 20: continue
    f, p = wildp(d, "dd_ship_late", BASECOLS + ctrl, "over1", seed=2)
    shr = 1 - abs(f.params.over1 / f0.params.over1)
    log(f"  +{nm:18s} β_[PADI>1]={f.params.over1:+7.2f} R²={f.rsquared:.3f} wild-p={p:.4f} "
        f"(축소 {shr*100:+5.1f}%) {'유의' if p < .05 else '비유의'}")
    rows.append({"ctrl": nm, "beta": f.params.over1, "wild_p": p, "shrink": shr, "n": len(d)})

log("\n  [플라시보 — 통제가 무작위보다 더 죽이는가]")
for nm, ctrl in (("재무4", FV), ("구조6", SV)):
    d = E.dropna(subset=BASECOLS + ["dd_ship_late"] + ctrl)
    if len(d) < 20: continue
    bS = smf.ols("dd_ship_late ~ " + " + ".join(BASECOLS + ctrl), data=d).fit().params.over1
    bs = []
    for _ in range(3000):
        d2 = d.copy()
        for j, v in enumerate(ctrl):
            d2[f"z{j}"] = rng.permutation(d[v].values)
        cs = " + ".join(f"z{j}" for j in range(len(ctrl)))
        try:
            bs.append(smf.ols(f"dd_ship_late ~ {' + '.join(BASECOLS)} + {cs}",
                              data=d2).fit().params.over1)
        except Exception:
            pass
    bs = np.array(bs); tail = float((np.abs(bs) <= abs(bS)).mean())
    log(f"    {nm:12s} 관측 β={bS:+7.2f} | 플라시보 중앙 {np.median(bs):+7.2f} "
        f"| ★꼬리 p={tail:.4f} {'공유분산' if tail < .10 else '차단 성공'}")
    out[f"placebo_{nm}"] = {"tail": tail, "beta": float(bS)}

log("\n[3] 전 분할연도에서 방어 재검정 (구조6)")
for s in SPLITS:
    d = ALL[ALL.split == s].merge(S[["ksic"] + SV], on="ksic", how="left")
    d = d.dropna(subset=["dd_ship_late"] + BASECOLS + SV)
    if len(d) < 20: continue
    fb, pb = wildp(d, "dd_ship_late", BASECOLS, "over1", B=999, seed=s)
    fc, pc = wildp(d, "dd_ship_late", BASECOLS + SV, "over1", B=999, seed=s)
    log(f"  {s}: 기준 β={fb.params.over1:+7.2f}(p={pb:.3f}) → +구조6 β={fc.params.over1:+7.2f}"
        f"(p={pc:.3f}) 축소 {(1-abs(fc.params.over1/fb.params.over1))*100:+5.1f}%")

D.to_csv(os.path.join(FUS, "diag_v2_types.csv"), index=False, encoding="utf-8-sig")
with open(os.path.join(FUS, "diag_v2_summary.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
log(f"\n저장 → {FUS}")
