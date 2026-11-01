# -*- coding: utf-8 -*-
"""
v3 검증 — 사전 등록(2026-08-12) 대로 전 지표를 동일 프로토콜로 검정하고 전부 보고
  핵심 검정: 산업FE + 시간FE 패널 회귀
      prod_yoy_{t+h} = β·indicator_t + α_i + γ_t + ε      (SE: 산업 클러스터)
    → 산업FE가 '산업 정체성', 시간FE가 '전산업 공통충격'을 제거하므로
      β는 순수 '산업내 시계열' 효과. 우리가 실패했던 바로 그 성분.
  가설: 조기경보가 실재하면 β < 0 (지표↑ → 이후 생산성장↓) 이고 h>0에서 유의
  다중검정: BH-FDR (지표 5종 × 호라이즌 6개 = 30개 가설)
"""
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
V3 = os.path.join(BASE, "분석", "v3")
MACRO = os.path.join(BASE, "공모전데이터", "외부데이터", "경기지표")
PSTART = 2008
IND = ["padi", "padi_val", "padi_sme", "padi_big", "aai"]
HOR = [0, 1, 2, 3, 6, 9]      # 개월
log = lambda m: print(m, flush=True)

# ── 경기변수(월별): 생산·출하·재고 ──────────────────────────────
k = pd.read_csv(os.path.join(MACRO, "kosis_광공업생산지수_월별.csv"), encoding="utf-8-sig", dtype=str)
k["val"] = pd.to_numeric(k.값, errors="coerce")
k = k.dropna(subset=["val"])
k["mi"] = (k.연월.str[:4].astype(int) - PSTART) * 12 + k.연월.str[4:6].astype(int) - 1
k["k3"] = k.산업코드.str[1:]
k = k[k.k3.str.len() == 3]
piv = k.pivot_table(index=["k3", "mi"], columns="항목명", values="val", aggfunc="mean").reset_index()
piv.columns.name = None
ren = {c: ("prod" if "생산" in c else "ship" if "출하" in c else "inv" if "재고" in c else c)
       for c in piv.columns if c not in ("k3", "mi")}
piv = piv.rename(columns=ren)
log(f"경기변수: {piv.k3.nunique()}개 산업군 × {piv.mi.nunique()}개월, 항목 {[c for c in piv.columns if c not in ('k3','mi')]}")

results = []
for lvl in ("k3", "k5"):
    p = pd.read_parquet(os.path.join(V3, f"padi_v3_{lvl}.parquet"))
    p["k3"] = p.unit.astype(str).str[:3]
    m = p.merge(piv, on=["k3", "mi"], how="inner").sort_values(["unit", "mi"])
    m = m.replace([np.inf, -np.inf], np.nan)
    # 종속변수: 생산 전년동월비, 재고순환(재고YoY − 출하YoY)
    m["prod_yoy"] = m.groupby("unit")["prod"].pct_change(12) * 100
    m["ship_yoy"] = m.groupby("unit")["ship"].pct_change(12) * 100
    m["inv_yoy"] = m.groupby("unit")["inv"].pct_change(12) * 100
    m["invcyc"] = m.inv_yoy - m.ship_yoy
    m["ti"] = m.mi.astype(int)
    log(f"\n{'='*66}\n[트랙 {lvl}] {m.unit.nunique()}단위 × {len(m):,}행")

    for tgt in ("prod_yoy", "invcyc"):
        rows = []
        for ind in IND:
            for h in HOR:
                d = m.copy()
                d["y"] = d.groupby("unit")[tgt].shift(-h)
                d = d.dropna(subset=["y", ind])
                # 지표 표준화(산업 내) — 계수 비교 가능하게
                d["x"] = d.groupby("unit")[ind].transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
                if d.unit.nunique() < 5 or len(d) < 200:
                    continue
                try:
                    fit = smf.ols("y ~ x + C(unit) + C(ti)", data=d).fit(
                        cov_type="cluster", cov_kwds={"groups": d.unit})
                    rows.append({"target": tgt, "ind": ind, "h": h,
                                 "beta": fit.params["x"], "t": fit.tvalues["x"],
                                 "p": fit.pvalues["x"], "n": len(d)})
                except Exception as e:
                    log(f"    회귀 실패 {ind} h={h}: {str(e)[:60]}")
        r = pd.DataFrame(rows)
        if r.empty:
            continue
        # BH-FDR
        r = r.sort_values("p").reset_index(drop=True)
        mm = len(r)
        r["q"] = (r.p * mm / (r.index + 1)).cummin().clip(upper=1)
        r["sig"] = np.where(r.q < 0.10, "*", "")
        results.append(r.assign(track=lvl))
        log(f"\n  [종속: {tgt}]  산업FE+시간FE, 산업클러스터 SE, BH-FDR q<0.10")
        show = r.sort_values(["ind", "h"])[["ind", "h", "beta", "t", "p", "q", "sig", "n"]]
        log(show.round(4).to_string(index=False))
        best = r.iloc[0]
        log(f"    최소 p: {best['ind']} h={int(best.h)} β={best.beta:+.4f} t={best.t:+.2f} p={best.p:.4f} q={best.q:.3f}")

allr = pd.concat(results, ignore_index=True)
allr.to_csv(os.path.join(V3, "panel_fe_results.csv"), index=False, encoding="utf-8-sig")

log(f"\n{'='*66}\n[종합] 전체 {len(allr)}개 가설 중 FDR q<0.10 유의: {int((allr.q < 0.10).sum())}개")
sigs = allr[allr.q < 0.10].sort_values("q")
if len(sigs):
    log(sigs[["track", "target", "ind", "h", "beta", "t", "q"]].round(4).to_string(index=False))
    neg = sigs[(sigs.target == "prod_yoy") & (sigs.beta < 0) & (sigs.h > 0)]
    log(f"\n  >>> 가설 방향(β<0, h>0, 생산) 부합 유의건: {len(neg)}개")
else:
    log("  유의한 결과 없음")
log(f"\n저장 → {V3}")
