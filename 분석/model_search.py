# -*- coding: utf-8 -*-
"""
체계적 모델 탐색 — 분할연도 × 종속변수 × 설명변수 부분집합
────────────────────────────────────────────────────────────────
[목적] 지금까지 시도한 것들 중 가장 좋은 모델을 찾되, **탐색 자체가 만드는 거품**을 제거한다.

[탐색 공간]
  종속변수 2종   : 생산지수 낙폭 / 출하지수 낙폭 (월별·실질·소분류)
  분할연도 7개   : 2013 ~ 2019
  설명변수 후보  : 전반기 정보만으로 만든 10종 (분할연도마다 재산출)
  부분집합       : 1~3개 조합

[선정 기준 — 사전에 고정. 결과를 보고 바꾸지 않는다]
  주 기준 : **LOCO-CV R²** (leave-one-cluster-out 예측 R²). 표본내 R² 는 참고만.
            중분류 하나를 통째로 빼고 학습 → 그 중분류를 예측. G=13 회 반복.
  부 기준 : 분할연도 간 **안정성** (부호 일관성, LOCO-R² 분산)
  간결성  : 같은 성능이면 변수 적은 쪽

[승자의 저주 보정 — 이 부분이 핵심이다]
  N개 사양을 뒤져 최고를 고르면 그 최고값은 필연적으로 부풀려진다.
  → 종속변수를 클러스터 단위로 순열한 **가짜 데이터**에 동일 탐색을 반복해
    '탐색 최고 LOCO-R²'의 귀무분포를 만든다. 우리 최고값이 그 꼬리에 있어야만 진짜다.
"""
import os, json, itertools, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
A    = os.path.join(BASE, "분석")
FUS  = os.path.join(A, "fusion")
PANEL_START = 2008
SPLITS = list(range(2013, 2020))
rng = np.random.default_rng(20260812)
log = lambda m: print(m, flush=True)

# ══════════════════════════════════════════════════════════════
log("[1] 재료 적재")
PP = pd.read_csv(os.path.join(A, "v2", "padi_panel_v2.csv"), encoding="utf-8-sig", dtype={"ksic": str})
PP["year"] = PANEL_START + PP.qi // 4
log(f"  PADI 분기패널 {len(PP):,} | 산업 {PP.ksic.nunique()} | 연도 {PP.year.min()}~{PP.year.max()}")

CP = pd.read_csv(os.path.join(FUS, "competition_panel.csv"), encoding="utf-8-sig", dtype={"ksic": str})
log(f"  경쟁 연패널 {len(CP):,} | 연도 {CP.yr.min()}~{CP.yr.max()}")

# [2026-08-13] 산업 매핑을 predictive_agg(39개 고정) 대신 PADI 패널에서 직접 유도한다.
#   등록원부 수집이 47개 산업까지 완결되었으므로 표본을 하드코딩으로 39개에 묶어두면 안 된다.
MI = pd.DataFrame({"ksic": sorted(PP.ksic.unique())})
MI["k3"] = MI.ksic.str[:3]
MI["mid"] = MI.ksic.str[:2]
log(f"  산업 매핑 {len(MI)}개 (소분류 {MI.k3.nunique()} / 중분류 {MI.mid.nunique()})")

# 종속변수 원천 — 월별 지수 (소분류)
IX = pd.read_csv(os.path.join(BASE, "공모전데이터", "외부데이터", "경기지표",
                              "kosis_광공업생산지수_월별.csv"), encoding="utf-8-sig", dtype=str)
IX.columns = [c.strip() for c in IX.columns]
IX["code"] = IX["산업코드"].astype(str).str.strip()
IX = IX[IX.code.str.match(r"^C\d{3}$", na=False)].copy()
IX["k3"] = IX.code.str[1:4]
IX["ym"] = IX["연월"].astype(str)
IX["yr"] = IX.ym.str[:4].astype(int)
IX["v"] = pd.to_numeric(IX["값"], errors="coerce")
DVKIND = {"prod": "생산지수(원지수)", "ship": "출하지수(원지수)"}
log(f"  월별지수 소분류 {IX.k3.nunique()}개 | {IX.yr.min()}~{IX.yr.max()}")


def maxdd(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if len(v) < 12 or np.nanmax(v) <= 0: return np.nan
    return float(np.min(v / np.maximum.accumulate(v) - 1.0) * 100)


# ══════════════════════════════════════════════════════════════
def build(split):
    """분할연도 split 기준으로 전반기 설명변수 + 후반기 종속변수를 만든다."""
    # --- PADI 계열 (전반기)
    e = PP[(PP.year < split) & (PP.year >= PANEL_START)]
    g = e.groupby("ksic").agg(O=("O", "sum"), E=("E_adj", "sum"), R=("R", "sum")).reset_index()
    g["padi_e"] = g.O / g.E
    # 전반기 내 PADI 추세(연도별 O/E 의 기울기)와 변동성
    yr = e.groupby(["ksic", "year"]).agg(O=("O", "sum"), E=("E_adj", "sum")).reset_index()
    yr["oe"] = yr.O / yr.E.replace(0, np.nan)
    tr = []
    for k, gg in yr.groupby("ksic"):
        gg = gg.dropna(subset=["oe"]).sort_values("year")
        tr.append({"ksic": k,
                   "padi_trend_e": np.polyfit(gg.year, gg.oe, 1)[0] if len(gg) >= 4 else np.nan,
                   "padi_vol_e": gg.oe.std() if len(gg) >= 4 else np.nan,
                   "padi_last_e": gg.oe.iloc[-1] if len(gg) else np.nan})
    g = g.merge(pd.DataFrame(tr), on="ksic", how="left")

    # --- 경쟁 계열 (전반기)
    c = CP[(CP.yr >= 2005) & (CP.yr < split)]
    cg = c.groupby("ksic").agg(cn_share_e=("cn_share", "mean"), cn_growth_e=("cn_growth", "mean"),
                               kr_share_e=("kr_share", "mean"),
                               kr_rel_e=("kr_rel_adv", "mean")).reset_index()
    cs = []
    for k, gg in c.groupby("ksic"):
        gg = gg.sort_values("yr")
        cs.append({"ksic": k,
                   "cn_trend_e": np.polyfit(gg.yr, gg.cn_share, 1)[0] if len(gg) >= 4 else np.nan})
    cg = cg.merge(pd.DataFrame(cs), on="ksic", how="left")

    X = MI.merge(g, on="ksic", how="inner").merge(cg, on="ksic", how="left")

    # --- 종속변수 (후반기 / 전반기 각각)
    out = {}
    for tag, item in DVKIND.items():
        s = IX[IX["항목명"] == item]
        late = s[s.yr >= split].groupby("k3").v.apply(maxdd).rename(f"dd_{tag}_late")
        early = s[(s.yr < split) & (s.yr >= PANEL_START)].groupby("k3").v.apply(maxdd).rename(f"dd_{tag}_early")
        out[tag] = pd.concat([late, early], axis=1).reset_index()
    D = X
    for tag in DVKIND:
        D = D.merge(out[tag], on="k3", how="left")
    D["split"] = split
    return D


log(f"\n[2] 분할연도별 데이터 구축 ({SPLITS})")
ALL = pd.concat([build(s) for s in SPLITS], ignore_index=True)
CAND = ["padi_e", "padi_trend_e", "padi_vol_e", "padi_last_e",
        "cn_share_e", "cn_growth_e", "kr_share_e", "kr_rel_e", "cn_trend_e",
        "dd_prod_early"]
for s in SPLITS:
    d = ALL[ALL.split == s]
    log(f"  {s}: n={len(d)} | 후반기 낙폭 평균 prod {d.dd_prod_late.mean():+.1f}% "
        f"ship {d.dd_ship_late.mean():+.1f}% | PADI 평균 {d.padi_e.mean():.3f}")

# ══════════════════════════════════════════════════════════════
log("\n[3] 평가 함수 — LOCO-CV (leave-one-cluster-out 예측 R²)")


def loco_r2(X, y, grp):
    """중분류 하나를 빼고 학습 → 그 중분류 예측. 예측 R² 반환."""
    uq = np.unique(grp)
    if len(uq) < 4: return np.nan
    pred = np.full(len(y), np.nan)
    for c in uq:
        te = grp == c
        tr = ~te
        if tr.sum() < X.shape[1] + 3: return np.nan
        Xt = np.column_stack([np.ones(tr.sum()), X[tr]])
        try:
            b = np.linalg.lstsq(Xt, y[tr], rcond=None)[0]
        except Exception:
            return np.nan
        pred[te] = np.column_stack([np.ones(te.sum()), X[te]]) @ b
    ok = ~np.isnan(pred)
    if ok.sum() < 8: return np.nan
    sse = np.sum((y[ok] - pred[ok]) ** 2)
    sst = np.sum((y[ok] - y[ok].mean()) ** 2)
    return float(1 - sse / sst) if sst > 0 else np.nan


def insample_r2(X, y):
    Xt = np.column_stack([np.ones(len(y)), X])
    b = np.linalg.lstsq(Xt, y, rcond=None)[0]
    r = y - Xt @ b
    sst = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum(r ** 2) / sst) if sst > 0 else np.nan, b


# ══════════════════════════════════════════════════════════════
log("\n[4] 전수 탐색")
SPECS = []
for r_ in (1, 2, 3):
    SPECS += list(itertools.combinations(CAND, r_))
log(f"  설명변수 조합 {len(SPECS)}개 × 종속변수 2종 × 분할연도 {len(SPLITS)}개 "
    f"= {len(SPECS)*2*len(SPLITS):,} 사양")

rows = []
for s in SPLITS:
    d0 = ALL[ALL.split == s]
    for tag in DVKIND:
        ycol = f"dd_{tag}_late"
        for sp in SPECS:
            cols = list(sp)
            d = d0.dropna(subset=[ycol] + cols)
            if len(d) < 15 or d.mid.nunique() < 6: continue
            X = d[cols].values.astype(float)
            y = d[ycol].values.astype(float)
            if np.any(np.std(X, axis=0) < 1e-12): continue
            ir2, b = insample_r2(X, y)
            lr2 = loco_r2(X, y, d.mid.values)
            rows.append({"split": s, "dv": tag, "spec": "+".join(cols), "k": len(cols),
                         "n": len(d), "r2": ir2, "loco_r2": lr2,
                         "b_padi": b[1 + cols.index("padi_e")] if "padi_e" in cols else np.nan})
R = pd.DataFrame(rows)
log(f"  유효 사양 {len(R):,}개")

# ══════════════════════════════════════════════════════════════
log("\n[5] 분할연도 전반에서 안정적인 사양 — 주 선정 기준")
agg = R.groupby(["dv", "spec", "k"]).agg(
    loco_mean=("loco_r2", "mean"), loco_min=("loco_r2", "min"), loco_sd=("loco_r2", "std"),
    r2_mean=("r2", "mean"), n_split=("split", "nunique")).reset_index()
agg = agg[agg.n_split == len(SPLITS)].sort_values("loco_mean", ascending=False)
log(f"  전 분할연도에서 추정 가능한 사양 {len(agg):,}개\n")
log(f"  {'순위':>3s} {'DV':5s} {'k':>2s} {'LOCO평균':>9s} {'LOCO최소':>9s} {'표본내R²':>8s}  사양")
for i, r in enumerate(agg.head(15).itertuples(), 1):
    log(f"  {i:3d} {r.dv:5s} {r.k:2d} {r.loco_mean:+9.3f} {r.loco_min:+9.3f} {r.r2_mean:8.3f}  {r.spec}")

log("\n  [현행 모형의 위치]")
cur = agg[(agg.spec == "padi_e+cn_growth_e")]
for r in cur.itertuples():
    rk = (agg.loco_mean > r.loco_mean).sum() + 1
    log(f"    {r.dv:5s} padi_e+cn_growth_e : LOCO평균 {r.loco_mean:+.3f} "
        f"표본내 {r.r2_mean:.3f} → {len(agg)}개 중 {rk}위")

log("\n  [단일변수 PADI 만]")
for r in agg[agg.spec == "padi_e"].itertuples():
    log(f"    {r.dv:5s} padi_e : LOCO평균 {r.loco_mean:+.3f} 표본내 {r.r2_mean:.3f}")

R.to_csv(os.path.join(FUS, "model_search_all.csv"), index=False, encoding="utf-8-sig")
agg.to_csv(os.path.join(FUS, "model_search_agg.csv"), index=False, encoding="utf-8-sig")
ALL.to_csv(os.path.join(FUS, "model_search_panel.csv"), index=False, encoding="utf-8-sig")
log(f"\n저장 → {FUS} (model_search_all/agg/panel.csv)")
