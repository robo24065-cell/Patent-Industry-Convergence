# -*- coding: utf-8 -*-
"""
ctrl_financial.csv 검증
  [A] 해상도 함정 실증 — k3해상도(21계열) vs mid해상도(13계열) 비교
  [B] fin_icr_low 취약성 점검 (변동이 소수 산업에 몰려있는지)
  [C] leave-one-mid-out 로 β_padi / β_ctrl 안정성
  [D] 재무통제 전부 동시 투입 시 β_padi
"""
import os, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
FUS = os.path.join(BASE, "분석", "fusion")
log = lambda m: print(m, flush=True)

Z = pd.read_csv(os.path.join(FUS, "ctrl_financial.csv"), encoding="utf-8-sig",
                dtype={"ksic": str, "k3": str, "mid": str})
P = pd.read_csv(os.path.join(FUS, "predictive_agg.csv"), encoding="utf-8-sig",
                dtype={"ksic": str, "k3": str, "mid": str})
D = P.merge(Z.drop(columns=["k3", "mid"]), on="ksic", how="left")
D = D.dropna(subset=["dd_late", "padi_e", "cn_growth_e"]).reset_index(drop=True)
log(f"병합 {D.shape}")

FIN = ["fin_debt_ratio", "fin_equity_ratio", "fin_current_ratio", "fin_borrow_dep",
       "fin_opm", "fin_icr", "fin_int_rate", "fin_rnd_sales", "fin_sales_growth",
       "fin_icr_low"]
MIDC = [c + "_mid" for c in FIN if c + "_mid" in D.columns]

A = D.groupby("k3").agg(**{**{"dd_late": ("dd_late", "first"), "padi_e": ("padi_e", "mean"),
                             "cn_growth_e": ("cn_growth_e", "mean"), "mid": ("mid", "first")},
                           **{c: (c, "first") for c in FIN + MIDC}}).reset_index()

# ── [A] 해상도 함정 실증 ──────────────────────────────────────────────────
log("\n[A] 해상도 함정 — k3해상도 vs mid(2자리)해상도 비교")
log("     통제변수를 2자리로만 만들면 클러스터(mid)와 단위가 같아져 within 변동이 0")
for c in ["fin_debt_ratio", "fin_opm", "fin_borrow_dep"]:
    cm = c + "_mid"
    # within-mid 표준편차 (k3 단위)
    w3 = A.groupby("mid")[c].std().fillna(0).mean()
    w2 = A.groupby("mid")[cm].std().fillna(0).mean()
    log(f"  {c:18s} within-mid 평균SD: k3판={w3:7.3f} / mid판={w2:7.3f} "
        f"| 고유값 k3판={A[c].nunique():2d} mid판={A[cm].nunique():2d}")

log("\n  같은 지표를 두 해상도로 넣었을 때 (n=23, cluster mid)")
for c in ["fin_debt_ratio", "fin_opm", "fin_borrow_dep"]:
    for suf, lab in [("", "k3해상도"), ("_mid", "mid해상도")]:
        v = c + suf
        f = smf.ols(f"dd_late ~ padi_e + cn_growth_e + {v}", data=A).fit(
            cov_type="cluster", cov_kwds={"groups": A.mid})
        log(f"    {lab:9s} {c:18s} β_ctrl={f.params[v]:+8.3f} p={f.pvalues[v]:.3f} "
            f"| β_padi={f.params.padi_e:+7.2f} p={f.pvalues.padi_e:.4f} R²={f.rsquared:.3f}")

# ── [B] fin_icr_low 취약성 ───────────────────────────────────────────────
log("\n[B] fin_icr_low(이자보상비율<100 연도비중) 분포 — 변동이 소수에 몰려있는가")
vc = A[["k3", "fin_icr_low"]].sort_values("fin_icr_low", ascending=False)
log(f"  k3 23개 중 0이 아닌 것: {int((A.fin_icr_low > 0).sum())}개")
log("  " + vc[vc.fin_icr_low > 0].to_string(index=False).replace("\n", "\n  "))
log("  → 사실상 4개 산업 더미. 유의하나 소수 관측에 의존 → 보조지표로만 취급 권고")

# ── [C] leave-one-mid-out ────────────────────────────────────────────────
log("\n[C] leave-one-mid-out 안정성 (n=39, cluster mid)")
for c in ["fin_borrow_dep", "fin_opm", "fin_icr_low", "fin_sales_growth"]:
    bp, bc, pc = [], [], []
    for m in sorted(D.mid.unique()):
        d = D[D.mid != m]
        if d.mid.nunique() < 3:
            continue
        f = smf.ols(f"dd_late ~ padi_e + cn_growth_e + {c}", data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d.mid})
        bp.append(f.params.padi_e); bc.append(f.params[c]); pc.append(f.pvalues[c])
    log(f"  {c:18s} β_padi [{min(bp):+7.2f},{max(bp):+7.2f}] 부호일관={all(b<0 for b in bp)} "
        f"| β_ctrl [{min(bc):+8.3f},{max(bc):+8.3f}] p<0.05 비율={np.mean(np.array(pc)<0.05):.2f}")

# ── [D] 재무통제 총동원 ──────────────────────────────────────────────────
log("\n[D] 재무통제 총동원 — PADI 가 '돈 없는 산업' 대리변수인지 최종 점검")
sets = {
    "핵심4종(부채/차입금의존/영업이익률/이자보상<100비중)":
        ["fin_debt_ratio", "fin_borrow_dep", "fin_opm", "fin_icr_low"],
    "유동성포함6종": ["fin_debt_ratio", "fin_borrow_dep", "fin_opm", "fin_icr_low",
                  "fin_current_ratio", "fin_int_rate"],
    "전체9종+": FIN,
}
for lab, cs in sets.items():
    for tag, d, g in [("n=39", D, D.mid), ("n=23", A, A.mid)]:
        if len(cs) + 3 >= len(d):
            log(f"  {lab} / {tag}: 자유도 부족 — 생략")
            continue
        f = smf.ols("dd_late ~ padi_e + cn_growth_e + " + " + ".join(cs), data=d).fit(
            cov_type="cluster", cov_kwds={"groups": g})
        log(f"  {lab:48s} {tag} β_padi={f.params.padi_e:+7.2f} "
            f"z={f.tvalues.padi_e:+5.2f} p={f.pvalues.padi_e:.4f} R²={f.rsquared:.3f}")

# 잔차화(orthogonalize): PADI 에서 재무요인 전부 제거한 뒤에도 예측력 남는가
log("\n  [D-2] PADI 를 재무지표 전부에 회귀해 잔차만 남긴 뒤 재검정")
for tag, d, g in [("n=39", D, D.mid), ("n=23", A, A.mid)]:
    r = smf.ols("padi_e ~ " + " + ".join(FIN), data=d).fit()
    d2 = d.copy(); d2["padi_resid"] = r.resid
    f = smf.ols("dd_late ~ padi_resid + cn_growth_e", data=d2).fit(
        cov_type="cluster", cov_kwds={"groups": g})
    log(f"    {tag} PADI~재무 R²={r.rsquared:.3f} (재무가 PADI 설명력 {r.rsquared*100:.1f}%) "
        f"→ β_resid={f.params.padi_resid:+7.2f} p={f.pvalues.padi_resid:.4f}")
