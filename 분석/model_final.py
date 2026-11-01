# -*- coding: utf-8 -*-
"""
최종 모델 확정 — 후보 좁히기 + 계수 추론
────────────────────────────────────────────────────────────────
탐색·확장 결과 요약 (LOCO-CV R², 분할연도 7개 평균 / 최소):
  지속성만                    +0.482 / +0.322   ← 진짜 기준선
  +PADI(연속)                 +0.524 / +0.397
  +PADI+중국비중              +0.534 / +0.369
  +[PADI>1] 더미              +0.542 / +0.463   ← 최고이자 가장 안정
  현행 모형(지속성 없음)        +0.040 / −0.169

임계값 민감도: c=0.85~1.10 에서 +0.50~+0.54 의 완만한 고원. 칼날 아님. 정점이 c=1.0.
  → O/E=1('기대만큼 포기')은 이론적으로 자연스러운 기준점이고 데이터도 그곳을 가리킨다.

이 스크립트가 하는 일
  [1] 더미 기반 조합을 좁게 재탐색 (중국·층화 축을 더할 가치가 있는지)
  [2] 최종 후보의 계수·표준오차·wild cluster bootstrap p
  [3] 분할연도별 계수 안정성
  [4] 지속성 대비 증분의 순열검정
"""
import os, json, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
FUS  = os.path.join(BASE, "분석", "fusion")
SPLITS = list(range(2013, 2020))
rng = np.random.default_rng(1234)
log = lambda m: print(m, flush=True)
out = {}

ALL = pd.read_csv(os.path.join(FUS, "model_search_panel.csv"), encoding="utf-8-sig",
                  dtype={"ksic": str, "k3": str, "mid": str})
PA = pd.read_csv(os.path.join(FUS, "predictive_agg.csv"), encoding="utf-8-sig",
                 dtype={"ksic": str})[["ksic", "padi_lo_e"]]
ALL = ALL.merge(PA, on="ksic", how="left")
ALL["over1"] = (ALL.padi_e > 1.0).astype(float)
ALL["over1_lo"] = (ALL.padi_lo_e > 1.0).astype(float)


def loco(X, y, g):
    uq = np.unique(g); pr = np.full(len(y), np.nan)
    for c in uq:
        te = g == c; tr = ~te
        if tr.sum() < X.shape[1] + 3: return np.nan
        Xt = np.column_stack([np.ones(tr.sum()), X[tr]])
        try:
            b = np.linalg.lstsq(Xt, y[tr], rcond=None)[0]
        except Exception:
            return np.nan
        pr[te] = np.column_stack([np.ones(te.sum()), X[te]]) @ b
    ok = ~np.isnan(pr)
    sse = ((y[ok] - pr[ok]) ** 2).sum(); sst = ((y[ok] - y[ok].mean()) ** 2).sum()
    return float(1 - sse / sst) if sst > 0 else np.nan


def evalspec(cols, dv="ship"):
    y_ = f"dd_{dv}_late"; v = []
    for s in SPLITS:
        d = ALL[ALL.split == s].dropna(subset=[y_] + cols)
        if len(d) < 15 or d.mid.nunique() < 6: continue
        X = d[cols].values.astype(float)
        if np.any(np.std(X, axis=0) < 1e-12): continue
        r = loco(X, d[y_].values.astype(float), d.mid.values)
        if not np.isnan(r): v.append(r)
    return (np.mean(v), np.min(v), len(v)) if v else (np.nan, np.nan, 0)


log("[1] 더미 기반 좁은 재탐색 — 축을 더할 가치가 있는가")
CANDS = {
    "지속성만":                          ["dd_prod_early"],
    "지속성 + [PADI>1]":                 ["dd_prod_early", "over1"],
    "지속성 + [PADI>1] + 중국비중":       ["dd_prod_early", "over1", "cn_share_e"],
    "지속성 + [PADI>1] + 중국증가율":     ["dd_prod_early", "over1", "cn_growth_e"],
    "지속성 + [PADI>1] + PADI연속":       ["dd_prod_early", "over1", "padi_e"],
    "지속성 + [저가치PADI>1]":            ["dd_prod_early", "over1_lo"],
    "지속성 + [PADI>1] + [저가치>1]":     ["dd_prod_early", "over1", "over1_lo"],
    "지속성 + [PADI>1] + 한국상대지위":    ["dd_prod_early", "over1", "kr_rel_e"],
}
log(f"  {'사양':32s} {'출하 평균':>10s}{'출하 최소':>10s}{'생산 평균':>10s}{'k':>3s}")
rows = []
for nm, cols in CANDS.items():
    a = evalspec(cols, "ship"); b = evalspec(cols, "prod")
    log(f"  {nm:32s} {a[0]:+10.3f}{a[1]:+10.3f}{b[0]:+10.3f}{len(cols):3d}")
    rows.append({"spec": nm, "cols": "+".join(cols), "k": len(cols),
                 "ship_mean": a[0], "ship_min": a[1], "prod_mean": b[0]})
R = pd.DataFrame(rows)
best = R.sort_values(["ship_mean"], ascending=False).iloc[0]
log(f"\n  → 최고: {best.spec}  (출하 LOCO {best.ship_mean:+.3f}, 최소 {best.ship_min:+.3f})")

# ══════════════════════════════════════════════════════════════
log("\n[2] 최종 후보 계수 — 분할연도별")


def wild_p(d, ycol, cols, var, B=2999, seed=0):
    r = np.random.default_rng(seed)
    fml = f"{ycol} ~ " + " + ".join(cols)
    f = smf.ols(fml, data=d).fit()
    t = f.params[var] / f.bse[var]
    rest = [c for c in cols if c != var]
    f0 = smf.ols(f"{ycol} ~ " + (" + ".join(rest) if rest else "1"), data=d).fit()
    r0, fit0 = f0.resid.values, f0.fittedvalues.values
    g = d["mid"].values; uq = np.unique(g); ts = []
    for _ in range(B):
        wm = dict(zip(uq, r.choice([-1., 1.], size=len(uq))))
        d2 = d.copy(); d2[ycol] = fit0 + r0 * np.array([wm[x] for x in g])
        try:
            fb = smf.ols(fml, data=d2).fit(); ts.append(fb.params[var] / fb.bse[var])
        except Exception:
            pass
    return float(f.params[var]), float((np.abs(np.array(ts)) >= abs(t)).mean())


COLS = ["dd_prod_early", "over1"]
log(f"  종속변수: 후반기 출하지수 최대낙폭 | 설명: {COLS}")
log(f"  {'분할':>5s} {'n':>3s} {'β_지속성':>10s} {'β_[PADI>1]':>12s} {'wild-p':>8s} {'R²':>7s} {'LOCO':>8s}")
rr = []
for s in SPLITS:
    d = ALL[ALL.split == s].dropna(subset=["dd_ship_late"] + COLS)
    f = smf.ols("dd_ship_late ~ dd_prod_early + over1", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d.mid})
    b, p = wild_p(d, "dd_ship_late", COLS, "over1", seed=s)
    lr = loco(d[COLS].values.astype(float), d.dd_ship_late.values.astype(float), d.mid.values)
    log(f"  {s:5d} {len(d):3d} {f.params.dd_prod_early:+10.3f} {b:+12.2f} {p:8.4f} "
        f"{f.rsquared:7.3f} {lr:+8.3f}")
    rr.append({"split": s, "b_persist": f.params.dd_prod_early, "b_over1": b,
               "wild_p": p, "r2": f.rsquared, "loco": lr, "n": len(d),
               "n_flag": int(d.over1.sum())})
S = pd.DataFrame(rr)
log(f"\n  β_[PADI>1] 평균 {S.b_over1.mean():+.2f}  (범위 {S.b_over1.min():+.2f} ~ {S.b_over1.max():+.2f})")
log(f"  부호 음수 비율 {(S.b_over1 < 0).mean():.0%}  |  wild-p<0.05: {(S.wild_p < .05).sum()}/{len(S)}")
log(f"  해당 산업 수 평균 {S.n_flag.mean():.1f} / 39")
out["final_coef"] = S.to_dict("records")

# ══════════════════════════════════════════════════════════════
log("\n[3] ★ 지속성 대비 증분의 순열검정 (더미를 산업 간 무작위 재배치)")
for dv in ("ship", "prod"):
    y_ = f"dd_{dv}_late"
    obs = evalspec(["dd_prod_early", "over1"], dv)[0] - evalspec(["dd_prod_early"], dv)[0]
    nd = []
    for _ in range(1000):
        ds = []
        for s in SPLITS:
            d = ALL[ALL.split == s].dropna(subset=[y_, "over1", "dd_prod_early"]).copy()
            d["over1"] = rng.permutation(d.over1.values)
            y = d[y_].values.astype(float); g = d.mid.values
            b = loco(d[["dd_prod_early"]].values.astype(float), y, g)
            p_ = loco(d[["dd_prod_early", "over1"]].values.astype(float), y, g)
            if not (np.isnan(b) or np.isnan(p_)): ds.append(p_ - b)
        if ds: nd.append(np.mean(ds))
    nd = np.array(nd); pv = float((nd >= obs).mean())
    log(f"  {dv}: 관측 증분 {obs:+.4f} | 귀무 중앙 {np.median(nd):+.4f} "
        f"95%분위 {np.percentile(nd,95):+.4f} | ★p={pv:.4f} {'유의' if pv<.05 else '비유의'}")
    out[f"perm_{dv}"] = {"obs": float(obs), "p": pv,
                         "null_p95": float(np.percentile(nd, 95))}

R.to_csv(os.path.join(FUS, "model_final_candidates.csv"), index=False, encoding="utf-8-sig")
S.to_csv(os.path.join(FUS, "model_final_bysplit.csv"), index=False, encoding="utf-8-sig")
with open(os.path.join(FUS, "model_final_summary.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
log(f"\n저장 → {FUS}")
