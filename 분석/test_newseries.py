# -*- coding: utf-8 -*-
"""
신규 계열 검정 — 가동률지수 · 내수/수출 분해
────────────────────────────────────────────────────────────────
[A] 가동률지수를 새 종속변수 후보로 (소분류 69개, 현행과 동일 해상도)
    가동률은 '설비 대비 얼마나 돌렸나'라 유휴 정도를 직접 잰다.
    생산·출하지수와 다른 정보를 담는지, PADI 예측력이 있는지 본다.

[B] ★ 내수/수출 분해 — 정합성 검정
    지금까지 우리 신호는 서로 다른 두 층화에서 같은 방향을 가리켰다:
      · 국내전용 특허 층에만 신호 (패밀리 보유 특허는 무신호)
      · 중소기업 층에만 신호 (대기업 층은 무신호)
    이 둘이 진짜라면 **세 번째 층화**도 같은 방향이어야 한다:
      · PADI 는 수출 출하보다 **내수 출하**의 낙폭을 더 잘 예측해야 한다

    검정 설계: 짝지은 비교. 같은 산업에서 (내수 낙폭 − 수출 낙폭) 을 종속변수로 놓는다.
              PADI 가 이 **차이**를 예측하면, 내수 쪽에 신호가 몰린다는 직접 증거다.
    한계 선언: 이 계열은 중분류(2자리) 해상도라 n=13 이다. 검정력이 매우 낮다.
              방향과 크기를 보되 유의성에 기대지 않는다.
"""
import os, json, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
FUS  = os.path.join(BASE, "분석", "fusion")
EXT  = os.path.join(BASE, "공모전데이터", "외부데이터", "경기지표")
SPLITS = list(range(2013, 2020))
rng = np.random.default_rng(9090)
log = lambda m: print(m, flush=True)
out = {}


def maxdd(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if len(v) < 12 or np.nanmax(v) <= 0: return np.nan
    return float(np.min(v / np.maximum.accumulate(v) - 1.0) * 100)


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


ALL = pd.read_csv(os.path.join(FUS, "model_search_panel.csv"), encoding="utf-8-sig",
                  dtype={"ksic": str, "k3": str, "mid": str})
ALL["over1"] = (ALL.padi_e > 1.0).astype(float)

# ══════════════════════════════════════════════════════════════
log("[A] 가동률지수 — 새 종속변수 후보")
U = pd.read_csv(os.path.join(EXT, "kosis_가동률지수_소분류.csv"), encoding="utf-8-sig", dtype=str)
U["k3"] = U.C1.astype(str).str.extract(r"^C(\d{3})$")[0]
U = U.dropna(subset=["k3"]).copy()
U["yr"] = U.PRD_DE.str[:4].astype(int)
U["v"] = pd.to_numeric(U.DT, errors="coerce")
log(f"  소분류 {U.k3.nunique()}개 | {U.PRD_DE.min()}~{U.PRD_DE.max()}")

rows = []
for s in SPLITS:
    late = U[U.yr >= s].groupby("k3").v.apply(maxdd).rename("dd_util_late")
    early = U[(U.yr < s) & (U.yr >= 2008)].groupby("k3").v.apply(maxdd).rename("dd_util_early")
    d = ALL[ALL.split == s].merge(pd.concat([late, early], axis=1).reset_index(), on="k3", how="left")
    rows.append(d)
AU = pd.concat(rows, ignore_index=True)
m = AU[AU.split == 2017]
log(f"  매칭 {m.dd_util_late.notna().sum()}/39 | 가동률 낙폭 평균 {m.dd_util_late.mean():+.1f}%")
log(f"  기존 계열과 상관: 생산 {m.dd_util_late.corr(m.dd_prod_late):+.3f} "
    f"출하 {m.dd_util_late.corr(m.dd_ship_late):+.3f}")

log("\n  [LOCO-CV — 가동률 낙폭을 종속변수로]")
for nm, cols in (("지속성만(가동률 전반기)", ["dd_util_early"]),
                 ("지속성 + [PADI>1]", ["dd_util_early", "over1"]),
                 ("지속성(생산) + [PADI>1]", ["dd_prod_early", "over1"]),
                 ("PADI + 중국증가율 (현행)", ["padi_e", "cn_growth_e"])):
    v = []
    for s in SPLITS:
        d = AU[AU.split == s].dropna(subset=["dd_util_late"] + cols)
        if len(d) < 15 or d.mid.nunique() < 6: continue
        X = d[cols].values.astype(float)
        if np.any(np.std(X, axis=0) < 1e-12): continue
        r = loco(X, d.dd_util_late.values.astype(float), d.mid.values)
        if not np.isnan(r): v.append(r)
    log(f"    {nm:28s} LOCO 평균 {np.mean(v):+.3f} 최소 {np.min(v):+.3f} (n_split={len(v)})")
    out[f"util_{nm}"] = float(np.mean(v)) if v else None

# ══════════════════════════════════════════════════════════════
log("\n[B] ★ 내수/수출 분해 — 세 번째 층화 정합성 검정")
X2 = pd.read_csv(os.path.join(EXT, "kosis_내수수출출하지수.csv"), encoding="utf-8-sig", dtype=str)
X2["mid"] = X2.C1.astype(str).str.extract(r"^C(\d{2})$")[0]
X2 = X2.dropna(subset=["mid"]).copy()
X2["yr"] = X2.PRD_DE.str[:4].astype(int)
X2["v"] = pd.to_numeric(X2.DT, errors="coerce")
X2["kind"] = np.where(X2.ITM_NM.str.contains("내수"), "dom", "exp")
log(f"  중분류 {X2.mid.nunique()}개 | {X2.PRD_DE.min()}~{X2.PRD_DE.max()}")

SPLIT = 2017
dd = []
for (m_, k), g in X2[X2.yr >= SPLIT].groupby(["mid", "kind"]):
    dd.append({"mid": m_, "kind": k, "dd": maxdd(g.sort_values("PRD_DE").v)})
for (m_, k), g in X2[(X2.yr < SPLIT) & (X2.yr >= 2008)].groupby(["mid", "kind"]):
    dd.append({"mid": m_, "kind": k + "_e", "dd": maxdd(g.sort_values("PRD_DE").v)})
DD = pd.DataFrame(dd).pivot(index="mid", columns="kind", values="dd").reset_index()
DD.columns.name = None

# PADI 를 중분류로 집계 (위험집합 가중)
P = ALL[ALL.split == SPLIT].copy()
A = P.groupby("mid").agg(padi_e=("padi_e", "mean"), over1=("over1", "mean"),
                         cn_share_e=("cn_share_e", "mean")).reset_index()
M = A.merge(DD, on="mid", how="inner").dropna(subset=["dom", "exp"])
M["diff"] = M.dom - M["exp"]          # 음수면 내수가 더 깊게 빠졌다
log(f"  결합 중분류 {len(M)}개")
log(f"  내수 낙폭 평균 {M.dom.mean():+.1f}%  수출 낙폭 평균 {M['exp'].mean():+.1f}%  "
    f"차이 평균 {M['diff'].mean():+.1f}%p")

log("\n  [상관 — PADI 는 어느 쪽 낙폭과 더 붙는가]")
for c in ("dom", "exp", "diff"):
    lab = {"dom": "내수 낙폭", "exp": "수출 낙폭", "diff": "내수−수출 차이"}[c]
    r = M.padi_e.corr(M[c]); rs = M.padi_e.corr(M[c], method="spearman")
    ro = M.over1.corr(M[c])
    log(f"    {lab:14s} corr(PADI연속)={r:+.3f} (Spearman {rs:+.3f}) | corr([PADI>1]비율)={ro:+.3f}")

log("\n  [짝지은 검정 — 차이를 종속변수로]")
for x in ("padi_e", "over1"):
    d = M.dropna(subset=[x, "diff"])
    f = smf.ols(f"diff ~ {x}", data=d).fit()
    # 순열검정 (n 작아 t 분포 신뢰 어려움)
    obs = f.params[x]
    nd = [smf.ols(f"diff ~ {x}", data=d.assign(**{x: rng.permutation(d[x].values)})).fit().params[x]
          for _ in range(5000)]
    nd = np.array(nd)
    pv = float((np.abs(nd) >= abs(obs)).mean())
    lab = "PADI(연속)" if x == "padi_e" else "[PADI>1] 비율"
    log(f"    {lab:16s} β={obs:+8.2f}  통상p={f.pvalues[x]:.4f}  순열p={pv:.4f}  R²={f.rsquared:.3f}")
    out[f"decomp_{x}"] = {"beta": float(obs), "perm_p": pv, "r2": float(f.rsquared)}

log("\n  [판정]")
bd = out.get("decomp_padi_e", {}).get("beta", np.nan)
log(f"    β<0 이면 'PADI 높을수록 내수가 수출보다 더 깊게 빠진다' = 기존 층화와 정합")
log(f"    관측 β = {bd:+.2f} → {'정합 ✔' if bd < 0 else '역방향 ✘'}")

AU.to_csv(os.path.join(FUS, "panel_with_util.csv"), index=False, encoding="utf-8-sig")
M.to_csv(os.path.join(FUS, "decomp_dom_exp.csv"), index=False, encoding="utf-8-sig")
with open(os.path.join(FUS, "newseries_summary.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
log(f"\n저장 → {FUS}")
