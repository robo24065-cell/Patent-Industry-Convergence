# -*- coding: utf-8 -*-
# verify_5 (반박 검증) 3: 대안 B C-RCA-PKR ρ·Δ·CI와 지형도 pkr_H를 pyarrow+pandas 독립 코드로 재계산(duckdb 미사용).
import sys, json, glob
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy.stats import spearmanr
BASE = r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"
MC = BASE + "/분석/모델경쟁_20260927"; W = BASE + "/분석/검증루프_20261012/C15/C15-T3"
C5 = MC + "/_shared_report/fusion_advantage/c5_R10"; MAP5 = MC + "/_shared_report/fusion_advantage/map5"
CCS = ["KR", "US", "JP", "CN", "EP"]
km = pd.read_csv(MC + "/T3_기술셀부상예측/ESSENTIAL_R7/conc_ksic_map.csv", dtype=str); K = sorted(km.ksic)
flt = [("is_inv", "=", True), ("appl_year", ">=", 2015), ("appl_year", "<=", 2019)]
prov = pd.concat([pq.read_table(f, columns=["cc", "appl_year", "kr_origin", "ksic_prov"], filters=flt).to_pandas() for f in glob.glob(C5 + "/alt_assign/alt_*.parquet")])
arms = {}
for a in ("pure", "reg", "stored"):
    arms[a] = pd.concat([pq.read_table(f, columns=["cc", "appl_year", "kr_origin", f"k_{a}"], filters=flt).to_pandas().rename(columns={f"k_{a}": "k"})
                         for f in glob.glob(W + "/cache/pkr_alt_*.parquet")])
ek = pd.read_csv(MAP5 + "/m4_exports_ksic_5rep.csv", dtype={"k5": str})
v = ek[ek.year.between(2015, 2017)].pivot_table(index="k5", columns="rep", values="v", aggfunc="sum")
sh = v / v.sum(); RCA = (sh.KR / sh.mean(1)).reindex(K)
def abroad(df, kcol, y0=2017, y1=2019):
    d = df[(df.cc != "KR") & df.appl_year.between(y0, y1) & df[kcol].notna()]
    a = d.groupby(kcol).size().reindex(K); k = d[d.kr_origin].groupby(kcol).size().reindex(K).fillna(0)
    return (k / a) / (k.sum() / a.sum())
def rta(cnt):
    s = cnt / cnt.sum(); return s.div(s.mean(1), axis=0)
def counts(df_for, kcol, y0=2017, y1=2019):
    o = {}
    for cc in CCS:
        if cc == "KR": d = prov[(prov.cc == "KR") & prov.appl_year.between(y0, y1)]; o[cc] = d.groupby("ksic_prov").size()
        else: d = df_for[(df_for.cc == cc) & df_for.appl_year.between(y0, y1)]; o[cc] = d.groupby(kcol).size()
    return pd.DataFrame(o).reindex(K).fillna(0)
xp = abroad(prov, "ksic_prov"); rt_prov = rta(counts(prov, "ksic_prov"))
out = {}
for a, df in arms.items():
    xa = abroad(df, "k")
    D = pd.concat([xp, xa, RCA], axis=1).dropna(); n = len(D); x1, x2, y = D.iloc[:, 0].values, D.iloc[:, 1].values, D.iloc[:, 2].values
    rp = spearmanr(x1, y)[0]; ra = spearmanr(x2, y)[0]
    rng = np.random.default_rng(20261005)
    for _ in range(3): rng.integers(0, n, (2000, n))   # c2 순서 cweq→txt→free 다음 pkr
    I = rng.integers(0, n, (2000, n))
    bd = np.array([spearmanr(x1[i], y[i])[0] - spearmanr(x2[i], y[i])[0] for i in I]); ba = np.array([spearmanr(x2[i], y[i])[0] for i in I])
    rng2 = np.random.default_rng(99); I2 = rng2.integers(0, n, (2000, n))
    bd2 = np.array([spearmanr(x1[i], y[i])[0] - spearmanr(x2[i], y[i])[0] for i in I2])
    rt = rta(counts(df, "k")); ld = rt.idxmax(1); lp = rt_prov.idxmax(1)
    out[a] = dict(n=n, rho_prov=rp, rho_pkr=ra, d=rp - ra, ci_d=[float(np.nanpercentile(bd, 2.5)), float(np.nanpercentile(bd, 97.5))],
                  ci_pkr=[float(np.nanpercentile(ba, 2.5)), float(np.nanpercentile(ba, 97.5))], ci_d_seed99=[float(np.nanpercentile(bd2, 2.5)), float(np.nanpercentile(bd2, 97.5))],
                  land_kr_gt1=int((rt.KR > 1).sum()), land_kr_leader=int((ld == "KR").sum()), land_kept=int(((ld == "KR") & (lp == "KR")).sum()),
                  land_rho_KR=float(spearmanr(rt.KR, rt_prov.KR)[0]), land_min_rho=float(min(spearmanr(rt[c], rt_prov[c])[0] for c in CCS)),
                  leaders={c: int((ld == c).sum()) for c in CCS})
    print(a, out[a], flush=True)
json.dump(out, open(W + "/verify_5/v5_rho.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
