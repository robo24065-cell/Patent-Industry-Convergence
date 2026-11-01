# -*- coding: utf-8 -*-
"""
§2.1 생존 결과 스트레스 테스트 — corr(산업평균 PADI, 침체빈도) = +0.53 이 진짜인가
문헌 지적(문헌조사 §2.1): 39개 세세분류는 소수의 중분류에 중첩(nested)되어 유효 N이 훨씬 작다.
  T1 클러스터 부트스트랩 (중분류/대분류 단위) → CI 재계산
  T2 Leave-one-industry-out → 한두 산업이 끌고 있는가
  T3 Spearman 순위상관 (외곽치 강건)
  T4 추가 통제: 특허집약도, 평균 특허연령, 출원인 집중도(HHI)
  T5 침체빈도 대안 정의 3종
  T6 수렴타당도: 경기민감도 베타(Berman-Pfleeger식)와의 상관
"""
import os, re
import numpy as np
import pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
V2, V3 = os.path.join(BASE, "분석", "v2"), os.path.join(BASE, "분석", "v3")
MACRO = os.path.join(BASE, "공모전데이터", "외부데이터", "경기지표")
OUT = os.path.join(BASE, "분석", "stress")
os.makedirs(OUT, exist_ok=True)
PSTART = 2008
rng = np.random.default_rng(11)
log = lambda m: print(m, flush=True)

# ── 경기변수 ────────────────────────────────────────────────────
k = pd.read_csv(os.path.join(MACRO, "kosis_광공업생산지수_월별.csv"), encoding="utf-8-sig", dtype=str)
k["val"] = pd.to_numeric(k.값, errors="coerce")
k = k.dropna(subset=["val"])
k["mi"] = (k.연월.str[:4].astype(int) - PSTART) * 12 + k.연월.str[4:6].astype(int) - 1
k["k3"] = k.산업코드.str[1:]
k = k[k.k3.str.len() == 3]
piv = k.pivot_table(index=["k3", "mi"], columns="항목명", values="val", aggfunc="mean").reset_index()
piv.columns.name = None
piv = piv.rename(columns={c: ("prod" if "생산" in c else "ship" if "출하" in c else "inv" if "재고" in c else c)
                          for c in piv.columns if c not in ("k3", "mi")})

# 전산업 생산(제조업 전체 대용: 모든 k3 평균 성장률)로 경기민감도 베타 산출
piv = piv.sort_values(["k3", "mi"])
piv["yoy"] = piv.groupby("k3")["prod"].pct_change(12) * 100
mkt = piv.groupby("mi").yoy.mean().rename("mkt")
piv = piv.merge(mkt, on="mi", how="left")

# ── PADI 패널 (월별 v3) ─────────────────────────────────────────
p = pd.read_parquet(os.path.join(V3, "padi_v3_k5.parquet"))
p["k3"] = p.unit.astype(str).str[:3]
m = p.merge(piv[["k3", "mi", "prod", "inv", "ship", "yoy", "mkt"]], on=["k3", "mi"], how="inner")
m = m.replace([np.inf, -np.inf], np.nan).sort_values(["unit", "mi"])
m["inv_yoy"] = m.groupby("unit")["inv"].pct_change(12) * 100
m["ship_yoy"] = m.groupby("unit")["ship"].pct_change(12) * 100

# ── 침체 정의 3종 ───────────────────────────────────────────────
def onsets(g, kind):
    if kind == "prod5":                          # 원래 정의: 생산YoY 0이상 → -5%미만 전환
        return ((g.yoy < -5) & (g.yoy.shift(1) >= 0)).astype(int)
    if kind == "prod0":                          # 완화: 0 아래로 전환
        return ((g.yoy < 0) & (g.yoy.shift(1) >= 0)).astype(int)
    if kind == "ship5":                          # 출하 기준
        return ((g.ship_yoy < -5) & (g.ship_yoy.shift(1) >= 0)).astype(int)
    raise ValueError


for kind in ("prod5", "prod0", "ship5"):
    m[f"on_{kind}"] = 0
    for u, g in m.groupby("unit"):
        g = g.sort_values("mi")
        m.loc[g.index, f"on_{kind}"] = onsets(g, kind).values

# ── 산업 단위 집계 + 통제변수 ───────────────────────────────────
agg = m.groupby("unit").agg(
    padi=("padi", "mean"),
    on_prod5=("on_prod5", "mean"), on_prod0=("on_prod0", "mean"), on_ship5=("on_ship5", "mean"),
    yoy_sd=("yoy", "std"), size=("R_r", "mean"), n=("mi", "size")).reset_index().dropna()

# 경기민감도 베타 (산업 yoy ~ 시장 yoy)
betas = {}
for u, g in m.dropna(subset=["yoy", "mkt"]).groupby("unit"):
    if len(g) > 40 and g.mkt.std() > 1e-9:
        betas[u] = np.polyfit(g.mkt.values, g.yoy.values, 1)[0]
agg["beta_cyc"] = agg.unit.map(betas)

# 특허 속성 통제 (제공데이터에서)
import duckdb
d = duckdb.connect()
pq = os.path.join(BASE, "공모전데이터", "작업데이터", "kr_registered.parquet").replace(chr(92), "/")
attr = d.execute(f"""SELECT trim(CAST(세세분류코드 AS VARCHAR)) unit,
    count(*) n_pat, count(DISTINCT 출원인명) n_app,
    max(cnt)*1.0/count(*) top_share
  FROM (SELECT 세세분류코드, 출원인명, count(*) OVER (PARTITION BY 세세분류코드, 출원인명) cnt
        FROM read_parquet('{pq}'))
  GROUP BY 1""").fetchdf()
agg = agg.merge(attr, on="unit", how="left")
agg["hhi_proxy"] = agg.top_share
agg["pat_per_app"] = agg.n_pat / agg.n_app.replace(0, np.nan)
agg["mid"] = agg.unit.str[:2]      # 중분류
agg["maj"] = agg.unit.str[:1]      # 대분류(제조 C 내부라 사실상 동일 → 중분류가 실질 클러스터)
log(f"산업 {len(agg)}개 | 중분류 {agg.mid.nunique()}개 | 클러스터당 평균 {len(agg)/agg.mid.nunique():.1f}개")


def boot_ci(x, y, groups=None, n=4000):
    """groups 지정 시 클러스터 부트스트랩"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y)); x, y = x[ok], y[ok]
    g = np.asarray(groups)[ok] if groups is not None else None
    r0 = np.corrcoef(x, y)[0, 1]
    bs = []
    if g is None:
        for _ in range(n):
            i = rng.integers(0, len(x), len(x))
            if np.std(x[i]) > 1e-9 and np.std(y[i]) > 1e-9:
                bs.append(np.corrcoef(x[i], y[i])[0, 1])
    else:
        uq = np.unique(g)
        for _ in range(n):
            pick = rng.choice(uq, len(uq), replace=True)
            idx = np.concatenate([np.where(g == c)[0] for c in pick])
            if len(idx) > 3 and np.std(x[idx]) > 1e-9 and np.std(y[idx]) > 1e-9:
                bs.append(np.corrcoef(x[idx], y[idx])[0, 1])
    return r0, np.percentile(bs, [2.5, 97.5]), len(bs)


log("\n" + "=" * 68)
log("[T1] 클러스터 부트스트랩 — 유효 N 축소 반영")
for tgt, lab in [("on_prod5", "침체빈도(생산-5%)"), ("on_prod0", "침체빈도(생산<0)"), ("on_ship5", "침체빈도(출하-5%)")]:
    r, ci, nb = boot_ci(agg.padi, agg[tgt])
    rc, cic, _ = boot_ci(agg.padi, agg[tgt], groups=agg.mid)
    v1 = "유의" if ci[0] > 0 else "비유의"
    v2 = "유의" if cic[0] > 0 else "**비유의**"
    log(f"  {lab:22s} r={r:+.3f} | 단순CI[{ci[0]:+.3f},{ci[1]:+.3f}] {v1} | 중분류클러스터CI[{cic[0]:+.3f},{cic[1]:+.3f}] {v2}")

log("\n[T3] Spearman 순위상관 (외곽치 강건)")
for tgt, lab in [("on_prod5", "침체빈도(생산-5%)"), ("on_prod0", "침체빈도(생산<0)"), ("on_ship5", "침체빈도(출하)")]:
    rs = agg.padi.corr(agg[tgt], method="spearman")
    log(f"  {lab:22s} Spearman ρ = {rs:+.3f}")

log("\n[T2] Leave-one-industry-out (원 정의 기준)")
base = np.corrcoef(agg.padi, agg.on_prod5)[0, 1]
loo = []
for i in range(len(agg)):
    s = agg.drop(agg.index[i])
    loo.append((agg.unit.iloc[i], np.corrcoef(s.padi, s.on_prod5)[0, 1]))
loo = pd.DataFrame(loo, columns=["dropped", "r"]).sort_values("r")
log(f"  전체 r={base:+.3f} | LOO 범위 [{loo.r.min():+.3f}, {loo.r.max():+.3f}] | 표준편차 {loo.r.std():.4f}")
log(f"  영향 최대(제거 시 r 최소): {loo.iloc[0].dropped} → {loo.iloc[0].r:+.3f}")
log(f"  Leave-one-중분류-out:")
for mid in sorted(agg.mid.unique()):
    s = agg[agg.mid != mid]
    if len(s) > 5:
        rr = np.corrcoef(s.padi, s.on_prod5)[0, 1]
        flag = " ←크게 하락" if rr < base - 0.15 else ""
        log(f"    {mid} 제외(n={len(s)}): r={rr:+.3f}{flag}")

log("\n[T4] 통제변수 추가 후 부분상관")
def partial(x, y, Z):
    Z = np.column_stack([np.ones(len(x))] + [np.asarray(z, float) for z in Z])
    ok = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(Z).any(axis=1)
    x, y, Z = np.asarray(x)[ok], np.asarray(y)[ok], Z[ok]
    bx = np.linalg.lstsq(Z, x, rcond=None)[0]; by = np.linalg.lstsq(Z, y, rcond=None)[0]
    return np.corrcoef(x - Z @ bx, y - Z @ by)[0, 1], ok.sum()

ctrls = {
    "log규모": [np.log(agg["size"] + 1)],
    "+특허집약도": [np.log(agg["size"] + 1), np.log(agg.n_pat + 1)],
    "+출원인집중도": [np.log(agg["size"] + 1), np.log(agg.n_pat + 1), agg.hhi_proxy],
    "+경기민감도β": [np.log(agg["size"] + 1), np.log(agg.n_pat + 1), agg.hhi_proxy, agg.beta_cyc],
}
for name, Z in ctrls.items():
    pr, nn = partial(agg.padi.values, agg.on_prod5.values, Z)
    log(f"  {name:16s} 부분상관 = {pr:+.3f} (n={nn})")

log("\n[T6] 수렴타당도 — 경기민감도 베타와의 상관")
r6, ci6, _ = boot_ci(agg.padi, agg.beta_cyc)
log(f"  corr(PADI, 경기민감도β) = {r6:+.3f} CI[{ci6[0]:+.3f},{ci6[1]:+.3f}]")
r7, ci7, _ = boot_ci(agg.padi, agg.yoy_sd)
log(f"  corr(PADI, 생산변동성)  = {r7:+.3f} CI[{ci7[0]:+.3f},{ci7[1]:+.3f}]")

agg.to_csv(os.path.join(OUT, "stress_agg.csv"), index=False, encoding="utf-8-sig")
loo.to_csv(os.path.join(OUT, "loo.csv"), index=False, encoding="utf-8-sig")
log(f"\n저장 → {OUT}")
