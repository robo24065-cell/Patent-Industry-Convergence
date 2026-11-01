# -*- coding: utf-8 -*-
"""verify_0 v1: C03-T2 발견(A1 CI 0 포함 = MCO 균등분할 번짐 산물, A1c 12/11/3/22, rho 0.336 [0.013,0.608]) 반박 시도.
lib.py/s5를 import하지 않고 원천 파일에서 독립 재구현. 원자료 읽기 전용."""
import os, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr

AN = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
MC = os.path.join(AN, "모델경쟁_20260927")
T2 = os.path.join(AN, "검증루프_20261012", "C03", "C03-T2")
OUT = os.path.join(T2, "verify_0")
REP = {410: "KR", 842: "US", 392: "JP", 156: "CN", 276: "DE"}
IC = {"854231", "854232", "854233", "854239"}

mp = pd.read_csv(os.path.join(AN, "fusion", "ksic_hs6_map.csv"), dtype=str); mp["hs6"] = mp.hs6.str.zfill(6)
codes = set(mp.hs6)
m = mp.drop_duplicates(["k5", "hs6"]).copy(); m["w"] = 1 / m.groupby("hs6").k5.transform("nunique")

a = pd.read_csv(os.path.join(MC, "_shared", "ext", "R2ind_comtrade_exports_5c_hs6_2004_2021.csv"), dtype={"hs6": str})
b = pd.read_csv(os.path.join(MC, "_shared", "ext", "R2N3_comtrade_exports_5c_hs6.csv"), dtype={"hs6": str})
ex = pd.concat([a, b], ignore_index=True); ex["hs6"] = ex.hs6.str.zfill(6); ex["rep"] = ex.reporter.map(REP)
ex = ex.drop_duplicates(["year", "reporter", "hs6"])
dl = pd.read_csv(os.path.join(T2, "cache", "s4_downloads_parsed.csv"), dtype={"hs6": str}); dl["hs6"] = dl.hs6.str.zfill(6)
c12 = pd.read_csv(os.path.join(T2, "cache", "unsd_corr_17_12.csv"), dtype=str)
c07 = pd.read_csv(os.path.join(T2, "cache", "unsd_corr_17_07.csv"), dtype=str)
VER = {2008: "H3", 2009: "H3", 2010: "H3", 2015: "H4", 2016: "H4", 2017: "H5", 2021: "H5"}

pa = pd.read_csv(os.path.join(MC, "_shared_report", "fusion_advantage", "map5", "m2_panel_appl.csv"), dtype={"ksic": str})
def rta(y0, y1):
    n = pa[pa.year.between(y0, y1) & pa.valid].groupby(["ksic", "cc"]).n_inv.sum().unstack()
    s = n / n.sum(); return s.div(s.mean(1), axis=0).KR
RTA2, RTA1 = rta(2015, 2017), rta(2008, 2010)


def trans(corr, mode):
    t = corr[["old", "new"]].drop_duplicates().copy()
    t["nT"] = t.groupby("old").new.transform("nunique")
    if mode == "equal":
        t = t[t.new.isin(codes)]; t["f"] = 1 / t.nT
    elif mode == "cont":
        cont = t.groupby("old").new.apply(lambda s: s.name in set(s))
        t["c"] = t.old.map(cont)
        t = pd.concat([t[t.c & (t.new == t.old)].assign(f=1.0), t[~t.c].assign(f=lambda d: 1 / d.nT)])
        t = t[t.new.isin(codes)]
    elif mode == "noMCO":  # A1 그대로이되, 8542가 아닌 구 코드에서 IC(854231–39)로 가는 MCO 연결만 제거 후 남은 대상에 균등
        t = t[~(t.new.isin(IC) & ~t.old.str.startswith("8542"))].copy()
        t["nT"] = t.groupby("old").new.transform("nunique")
        t = t[t.new.isin(codes)]; t["f"] = 1 / t.nT
    elif mode == "contMCOonly":  # 연속성 규칙을 MCO 관련 구 코드에만 적용, 나머지는 A1 균등
        cont = t.groupby("old").new.apply(lambda s: s.name in set(s))
        mco = t.groupby("old").new.apply(lambda s: len(set(s) & IC) > 0 and not s.name.startswith("8542"))
        t["c"] = t.old.map(cont) & t.old.map(mco)
        t = pd.concat([t[t.c & (t.new == t.old)].assign(f=1.0), t[~t.c].assign(f=lambda d: 1 / d.nT)])
        t = t[t.new.isin(codes)]
    return t[["old", "new", "f"]]


def rows(y, mode):
    e = ex[ex.year == y][["year", "rep", "hs6", "export_usd"]].rename(columns={"export_usd": "v"})
    if mode is None or VER[y] == "H5":
        return e
    corr = c12 if VER[y] == "H4" else c07
    add = dl[(dl.year == y) & dl.key.str.startswith("OLD" + VER[y])][["year", "rep", "hs6", "v"]]
    o = pd.concat([e, add]).drop_duplicates(["year", "rep", "hs6"])
    z = o.merge(trans(corr, mode), left_on="hs6", right_on="old"); z["v"] = z.v * z.f
    return z.groupby(["year", "rep", "new"]).v.sum().reset_index().rename(columns={"new": "hs6"})


def rca(mode, years):
    r = pd.concat([rows(y, mode) for y in years]).merge(m[["k5", "hs6", "w"]], on="hs6")
    v = r.assign(v=r.v * r.w).groupby(["k5", "rep"]).v.sum().unstack()
    s = v / v.sum(); return s.div(s.mean(1), axis=0).KR, v


def ci(x, y, seed=42, B=2000):
    d = pd.concat([x, y], axis=1).dropna(); X, Y = d.iloc[:, 0].values, d.iloc[:, 1].values
    rng = np.random.default_rng(seed)
    bs = [spearmanr(X[i], Y[i]).statistic for i in (rng.integers(0, len(X), len(X)) for _ in range(B))]
    return round(spearmanr(X, Y).statistic, 4), round(float(np.nanpercentile(bs, 2.5)), 4), round(float(np.nanpercentile(bs, 97.5)), 4), len(X)


def quad(r2):
    x = RTA2; y = r2.reindex(x.index)
    q = np.where(x.isna() | y.isna(), "NA", np.where(x >= 1, np.where(y >= 1, "SS", "SW"), np.where(y >= 1, "WS", "WW")))
    return pd.Series(q, index=x.index)


res = {}; R2 = {}; V2 = {}
for lab, mode in [("B", None), ("A1", "equal"), ("A1c", "cont"), ("A1_noMCO", "noMCO"), ("A1_contMCOonly", "contMCOonly")]:
    r2, v2 = rca(mode, [2015, 2016, 2017]); R2[lab] = r2; V2[lab] = v2
    q = quad(r2); vc = q.value_counts().to_dict()
    rho = ci(RTA2, r2)
    seeds = [ci(RTA2, r2, seed=s, B=2000)[1] for s in range(1, 21)]
    res[lab] = dict(SS=vc.get("SS", 0), SW=vc.get("SW", 0), WS=vc.get("WS", 0), WW=vc.get("WW", 0), NA=vc.get("NA", 0),
                    rho=rho, lo_seed1_20_min=min(seeds), lo_seed1_20_max=max(seeds), lo_frac_le0=float(np.mean(np.array(seeds) <= 0)),
                    SW_list=" ".join(sorted(q.index[q == "SW"])))
    print(lab, res[lab], flush=True)

# MCO 번짐 크기: 851770·851712 등 KR 2015–16 값 중 IC로 간 몫
t = trans(c12, "equal")
for o in ["851770", "851712", "853710", "847330", "852990"]:
    g = t[t.old == o]; res[f"share_to_IC_{o}"] = round(float(g[g.new.isin(IC)].f.sum()), 3); res[f"share_kept_{o}"] = round(float(g[g.new == o].f.sum()), 3)
mco_old = c12[c12.new.isin(IC) & ~c12.old.str.startswith("8542")].old.unique()
res["n_old_codes_MCO_HS12"] = int(len(mco_old))
res["n_old_codes_MCO_HS07"] = int(c07[c07.new.isin(IC) & ~c07.old.str.startswith("8542")].old.unique().size)
# 반도체 KSIC 값 비율
for lab in ["A1", "A1c", "A1_noMCO"]:
    for k in ["26111", "26422", "28902"]:
        res[f"{lab}_{k}_KRratio"] = round(float(V2[lab].loc[k, "KR"] / V2["B"].loc[k, "KR"]), 3)
        res[f"{lab}_{k}_othersratio"] = round(float(V2[lab].loc[k].drop("KR").sum() / V2["B"].loc[k].drop("KR").sum()), 3)
        res[f"{lab}_{k}_rca"] = round(float(R2[lab][k]), 3)
res["B_26111_rca"] = round(float(R2["B"]["26111"]), 3)
json.dump(res, open(os.path.join(OUT, "v1_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}, ensure_ascii=False, indent=1))
