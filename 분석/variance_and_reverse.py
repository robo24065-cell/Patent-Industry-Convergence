# -*- coding: utf-8 -*-
"""
문헌조사 §2.2 + §2.7 구현
  [A] 분산분해 + 신뢰도비율 — "FE가 무엇을 지웠는가"를 수치화 [진단, p값 없음]
      ICC = between/(between+within), within 신뢰도비율로 감쇠편의 상한 산출
      pooled/between/RE/산업FE/이중FE 5종 추정량 병렬표
  [B] 역방향 그레인저 — 경기 → IP 유지결정 [탐색, 사전선언]
      가설(Geroski-Walters 1995): 산출 변화가 혁신활동을 그레인저 인과
      검정: PADI_{t} = f(생산YoY_{t-h}) + 산업FE + 시간FE, h=1..12개월
      방향 예측: 생산 악화(YoY↓) → 이후 포기 증가(PADI↑) 즉 β<0
"""
import os, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
V3 = os.path.join(BASE, "분석", "v3")
MACRO = os.path.join(BASE, "공모전데이터", "외부데이터", "경기지표")
OUT = os.path.join(BASE, "분석", "stress")
os.makedirs(OUT, exist_ok=True)
PSTART = 2008
log = lambda m: print(m, flush=True)

# ── 데이터 결합 ─────────────────────────────────────────────────
k = pd.read_csv(os.path.join(MACRO, "kosis_광공업생산지수_월별.csv"), encoding="utf-8-sig", dtype=str)
k["val"] = pd.to_numeric(k.값, errors="coerce"); k = k.dropna(subset=["val"])
k["mi"] = (k.연월.str[:4].astype(int) - PSTART) * 12 + k.연월.str[4:6].astype(int) - 1
k["k3"] = k.산업코드.str[1:]
k = k[k.k3.str.len() == 3]
piv = k.pivot_table(index=["k3", "mi"], columns="항목명", values="val", aggfunc="mean").reset_index()
piv.columns.name = None
piv = piv.rename(columns={c: ("prod" if "생산" in c else "ship" if "출하" in c else "inv" if "재고" in c else c)
                          for c in piv.columns if c not in ("k3", "mi")})
piv = piv.sort_values(["k3", "mi"])
piv["yoy"] = piv.groupby("k3")["prod"].pct_change(12) * 100

p = pd.read_parquet(os.path.join(V3, "padi_v3_k5.parquet"))
p["k3"] = p.unit.astype(str).str[:3]
m = p.merge(piv[["k3", "mi", "prod", "yoy"]], on=["k3", "mi"], how="inner")
m = m.replace([np.inf, -np.inf], np.nan).sort_values(["unit", "mi"])
m["ti"] = m.mi.astype(int)
m = m.dropna(subset=["padi"])
log(f"패널: {m.unit.nunique()}산업 × {m.mi.nunique()}월 = {len(m):,}행")

# ══════════════════════════════════════════════════════════════
log("\n" + "=" * 68)
log("[A] 분산분해 — 신호는 어느 축에 있는가")
for ind in ("padi", "padi_sme", "aai"):
    d = m.dropna(subset=[ind])
    gm = d[ind].mean()
    bmeans = d.groupby("unit")[ind].mean()
    ns = d.groupby("unit").size()
    # between: 산업평균의 분산 (가중), within: 산업내 편차의 분산
    var_b = float(((bmeans - gm) ** 2 * ns).sum() / ns.sum())
    var_w = float(d.groupby("unit")[ind].transform(lambda s: s - s.mean()).pow(2).mean())
    icc = var_b / (var_b + var_w)
    log(f"  {ind:10s} between={var_b:.4f}  within={var_w:.4f}  **ICC={icc:.3f}**  "
        f"(총분산의 {icc*100:.0f}%가 산업간 차이)")

# 신뢰도비율: within 분산 중 표본잡음 비중 추정 (포아송 근사: Var(O/E) ≈ padi/E)
d = m.dropna(subset=["padi", "E_r"])
d = d[d.E_r > 0]
noise_var = float((d.padi / d.E_r).mean())          # 셀별 추정 표본오차 분산의 평균
within_var = float(d.groupby("unit").padi.transform(lambda s: s - s.mean()).pow(2).mean())
rel = max(0.0, (within_var - noise_var)) / within_var
log(f"\n  within 총분산 {within_var:.4f} 중 표본잡음 {noise_var:.4f} "
    f"→ **신뢰도비율 {rel:.3f}**")
log(f"  → 감쇠편의 보정 시 계수 상한 = 관측β / {rel:.3f} = 관측β × {1/max(rel,1e-6):.2f}")

# 5종 추정량 병렬표 (종속: 12개월 후 생산 YoY)
log("\n[A-2] 추정량 5종 병렬 — 같은 데이터, 다른 분산 활용")
d2 = m.copy()
d2["y"] = d2.groupby("unit")["yoy"].shift(-3)
d2 = d2.dropna(subset=["y", "padi"])
bet = d2.groupby("unit").agg(y=("y", "mean"), x=("padi", "mean")).reset_index()
rows = []
f1 = smf.ols("y ~ x", data=bet).fit()
rows.append(("between (산업평균만)", f1.params["x"], f1.tvalues["x"], f1.pvalues["x"], len(bet)))
f2 = smf.ols("y ~ padi", data=d2).fit(cov_type="cluster", cov_kwds={"groups": d2.unit})
rows.append(("pooled OLS", f2.params["padi"], f2.tvalues["padi"], f2.pvalues["padi"], len(d2)))
f3 = smf.ols("y ~ padi + C(unit)", data=d2).fit(cov_type="cluster", cov_kwds={"groups": d2.unit})
rows.append(("산업FE", f3.params["padi"], f3.tvalues["padi"], f3.pvalues["padi"], len(d2)))
f4 = smf.ols("y ~ padi + C(ti)", data=d2).fit(cov_type="cluster", cov_kwds={"groups": d2.unit})
rows.append(("시간FE", f4.params["padi"], f4.tvalues["padi"], f4.pvalues["padi"], len(d2)))
f5 = smf.ols("y ~ padi + C(unit) + C(ti)", data=d2).fit(cov_type="cluster", cov_kwds={"groups": d2.unit})
rows.append(("이중FE (기존 결과)", f5.params["padi"], f5.tvalues["padi"], f5.pvalues["padi"], len(d2)))
est = pd.DataFrame(rows, columns=["추정량", "β", "t", "p", "n"])
log(est.round(4).to_string(index=False))
log("  → between이 크고 이중FE에서 사라지면 '신호는 between에 있었다'는 증거")
est.to_csv(os.path.join(OUT, "estimator_comparison.csv"), index=False, encoding="utf-8-sig")

# ══════════════════════════════════════════════════════════════
log("\n" + "=" * 68)
log("[B] 역방향 그레인저 — 경기 → IP 유지결정 (사전선언: β<0 예측)")
log("   PADI_t = β·생산YoY_{t-h} + 산업FE + 시간FE, 산업 클러스터 SE")
res = []
for ind in ("padi", "padi_sme", "padi_big", "aai"):
    for h in (1, 3, 6, 9, 12, 18, 24):
        d = m.copy()
        d["x"] = d.groupby("unit")["yoy"].shift(h)      # 과거 경기
        d = d.dropna(subset=["x", ind])
        if len(d) < 300:
            continue
        d["yv"] = d.groupby("unit")[ind].transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
        try:
            f = smf.ols("yv ~ x + C(unit) + C(ti)", data=d).fit(
                cov_type="cluster", cov_kwds={"groups": d.unit})
            res.append({"ind": ind, "h(개월전)": h, "beta": f.params["x"],
                        "t": f.tvalues["x"], "p": f.pvalues["x"], "n": len(d)})
        except Exception as e:
            log(f"    실패 {ind} h={h}: {str(e)[:50]}")
r = pd.DataFrame(res)
# BH-FDR
r = r.sort_values("p").reset_index(drop=True)
r["q"] = (r.p * len(r) / (r.index + 1)).cummin().clip(upper=1)
r["sig"] = np.where(r.q < 0.10, "*", "")
log(r.sort_values(["ind", "h(개월전)"]).round(4).to_string(index=False))
nsig = int((r.q < 0.10).sum())
log(f"\n  FDR q<0.10 유의: {nsig}/{len(r)}개")
if nsig:
    s = r[r.q < 0.10].sort_values("q")
    log("  유의 결과:")
    log(s[["ind", "h(개월전)", "beta", "t", "q"]].round(4).to_string(index=False))
    neg = s[s.beta < 0]
    log(f"  >>> 가설 방향(β<0: 경기악화→포기증가) 부합: {len(neg)}/{len(s)}")
r.to_csv(os.path.join(OUT, "reverse_granger.csv"), index=False, encoding="utf-8-sig")
log(f"\n저장 → {OUT}")
