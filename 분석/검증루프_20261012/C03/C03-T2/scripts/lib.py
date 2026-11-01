# -*- coding: utf-8 -*-
"""C03-T2 공용: 경로·원본 로더·사분면/RCA 재구현 (원 m4_industry_quadrant.py 논리 독립 재작성)."""
import os, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
AN = os.path.join(ROOT, "분석")
MC = os.path.join(AN, "모델경쟁_20260927")
EXT = os.path.join(MC, "_shared", "ext")
MAP5 = os.path.join(MC, "_shared_report", "fusion_advantage", "map5")
SPLIT34 = os.path.join(MC, "_shared_report", "fusion_advantage", "split34")
W = os.path.join(AN, "검증루프_20261012", "C03", "C03-T2")
CACHE = os.path.join(W, "cache")
DL = os.path.join(W, "download")
REP = {410: "KR", 842: "US", 392: "JP", 156: "CN", 276: "DE"}
REPS = ["KR", "US", "JP", "CN", "DE"]
B3 = ["15220", "20491", "28902", "29133", "31312", "33309"]


def load_map():
    mp = pd.read_csv(os.path.join(AN, "fusion", "ksic_hs6_map.csv"), dtype={"k5": str, "hs6": str, "io": str})
    mp["hs6"] = mp.hs6.str.zfill(6)
    return mp


def load_ex():
    a = pd.read_csv(os.path.join(EXT, "R2ind_comtrade_exports_5c_hs6_2004_2021.csv"), dtype={"hs6": str})
    b = pd.read_csv(os.path.join(EXT, "R2N3_comtrade_exports_5c_hs6.csv"), dtype={"hs6": str})
    a["src"] = "R2ind"; b["src"] = "R2N3"
    ex = pd.concat([a, b], ignore_index=True)
    ex["hs6"] = ex.hs6.str.zfill(6)
    ex["rep"] = ex.reporter.map(REP)
    return ex


def patent_rta():
    pa = pd.read_csv(os.path.join(MAP5, "m2_panel_appl.csv"), dtype={"ksic": str})

    def rta(y0, y1):
        n = pa[pa.year.between(y0, y1) & pa.valid].groupby(["ksic", "cc"]).n_inv.sum().unstack()
        s = n / n.sum()
        return s.div(s.mean(1), axis=0), n
    r2, n2 = rta(2015, 2017); r1, n1 = rta(2008, 2010); r4, n4 = rta(2020, 2022)
    return dict(P2=r2.KR, P1=r1.KR, P4=r4.KR, nKR_P2=n2.KR)


def ksic_value(ex_rows, wmap):
    """ex_rows: year, rep, hs6, v ; wmap: k5, hs6, w → k5, rep, year, v"""
    return ex_rows.merge(wmap[["k5", "hs6", "w"]], on="hs6").assign(v=lambda d: d.v * d.w).groupby(["k5", "rep", "year"]).v.sum().reset_index()


def rca_from(ek, years, denom=None):
    """denom None: 48 KSIC 합(원본). denom: DataFrame index rep, columns year → 보고국 전체 수출"""
    v = ek[ek.year.isin(years)].groupby(["k5", "rep"]).v.sum().unstack()
    if denom is None:
        s = v / v.sum()
    else:
        tot = denom.loc[:, years].sum(1)
        s = v / tot.reindex(v.columns)
    return s.div(s.mean(1), axis=0)["KR"], v


def quad(x, y):
    return pd.Series(np.where(x.isna() | y.isna(), "NA",
                     np.where(x >= 1, np.where(y >= 1, "SS", "SW"), np.where(y >= 1, "WS", "WW"))), index=x.index)


def sp_ci(a, b, B=2000, seed=42):
    rng = np.random.default_rng(seed)
    d = pd.concat([a, b], axis=1).dropna(); x, y = d.iloc[:, 0].values, d.iloc[:, 1].values
    r = spearmanr(x, y).statistic
    bs = [spearmanr(x[i], y[i]).statistic for i in (rng.integers(0, len(x), len(x)) for _ in range(B))]
    return dict(rho=round(float(r), 3), lo=round(float(np.nanpercentile(bs, 2.5)), 3), hi=round(float(np.nanpercentile(bs, 97.5)), 3), n=int(len(x)))


def t3_pct():
    d = pd.read_csv(os.path.join(MAP5, "m5_industry_dashboard.csv"), dtype={"ksic": str}, encoding="utf-8-sig").set_index("ksic")
    return d.T3_breakout_cell_rate_pct


def evaluate(rta, rca_p2, rca_p1, rca_p4, label):
    q = quad(rta["P2"].reindex(rca_p2.index.union(rta["P2"].index)), rca_p2.reindex(rca_p2.index.union(rta["P2"].index)))
    q = q.reindex(rta["P2"].index)
    cnt = q.value_counts().to_dict()
    r = sp_ci(rta["P2"], rca_p2)
    rg = sp_ci(np.log(rta["P2"] / rta["P1"]), np.log(rca_p2 / rca_p1))
    t3 = t3_pct()
    sw = sorted(q.index[q == "SW"])
    lst = sorted([k for k in sw if t3.get(k, np.nan) >= 2 / 3])
    out = dict(variant=label, SS=cnt.get("SS", 0), SW=cnt.get("SW", 0), WS=cnt.get("WS", 0), WW=cnt.get("WW", 0), NA=cnt.get("NA", 0),
               rho=r["rho"], rho_lo=r["lo"], rho_hi=r["hi"], n=r["n"], rho_chg=rg["rho"], rho_chg_lo=rg["lo"], rho_chg_hi=rg["hi"],
               list42=" ".join(lst), list42_n=len(lst), list42_out=len(set(B3) - set(lst)), list42_in=len(set(lst) - set(B3)),
               SW_list=" ".join(sw))
    if rca_p4 is not None:
        q4 = quad(rta["P4"], rca_p4.reindex(rta["P4"].index))
        m = q != "NA"
        out["persist"] = round(float((q[m] == q4[m]).mean()), 3)
    return out, q
