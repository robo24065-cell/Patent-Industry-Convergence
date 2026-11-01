# -*- coding: utf-8 -*-
r"""C07-T1 (d) 민감도: split34 s2_decomp(SP-VAR)·s4_figs_robust(i)(SP-FE)·s5_cases(SP-CASE)의 KOSIS 부분을 복사해 입력·창만 바꾼다.
원 스크립트·원 산출물은 읽기만. 출력: sensitivity_313.csv, cache\s3_*.json
변형: B0 원판 / V1a 끝 창 2015–16(파일 값) / V1a_x 끝 창 2015–16을 KOSIS에만 / V2a 9→10 비-1:1 코드 제외 / V3 출하 실질화(옵션, 디플레이터 csv 있을 때)."""
import os, sys, json, time
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
B = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
MC = os.path.join(B, r"분석\모델경쟁_20260927")
S34 = os.path.join(MC, r"_shared_report\fusion_advantage\split34")
H = os.path.join(B, r"분석\검증루프_20261012\C07\C07-T1")
sys.path.insert(0, os.path.join(MC, r"_shared_report\label_validity"))
from lv_common import ksic_to_group
NP = 5000
P0 = pd.read_parquet(os.path.join(S34, "panel_ksic_cc_year.parquet"))
KO = pd.read_csv(os.path.join(B, r"공모전데이터\작업데이터\kosis_광업제조업_50개산업.csv"), dtype={"세세분류코드": str})
MAP = pd.read_csv(os.path.join(H, "ksic9_10_map_check.csv"), dtype={"ksic10": str})
NON11 = set(MAP[~MAP.type_9to10.isin(["1:1", "1:1(코드 변경)"])].ksic10)
DEF = None
if os.path.exists(os.path.join(H, r"cache\deflator_case.csv")):
    DEF = pd.read_csv(os.path.join(H, r"cache\deflator_case.csv"), dtype={"ksic": str})  # ksic, year, defl(2015=100)


def _codes(g): return pd.factorize(np.asarray(g))[0]
def _dm(v, c): return v - (np.bincount(c, weights=v) / np.bincount(c))[c]


def run(var):
    excl = NON11 if var == "V2a" else set()
    # ---------- s2 부분 ----------
    rng = np.random.default_rng(20261005)
    P = P0[(P0.year >= 2005) & (P0.year <= 2017)]
    K50 = sorted(P.ksic.unique()); G = pd.Series({k: ksic_to_group(k) for k in K50})
    Kuse = [k for k in K50 if k not in excl]
    gs = G[Kuse].value_counts(); K34 = sorted([k for k in Kuse if gs[G[k]] > 1])
    years = list(range(2005, 2018))
    tot = P.groupby(["ksic", "year"]).n.sum().unstack().reindex(index=K50, columns=years, fill_value=0)
    kro = P.groupby(["ksic", "year"]).n_kro.sum().unstack().reindex(index=K50, columns=years, fill_value=0)
    bycc = {c: P[P.cc == c].groupby(["ksic", "year"]).n.sum().unstack().reindex(index=K50, columns=years, fill_value=0) for c in ["KR", "US", "JP", "CN", "EP"]}
    lg = lambda A, a, b: (np.log(A[b].mean(axis=1) + 1) - np.log(A[a].mean(axis=1) + 1)) / (np.mean(b) - np.mean(a))
    E, L = [2005, 2006, 2007], [2015, 2016, 2017]
    M = pd.DataFrame(index=K50); M["grp"] = G
    M["g_5국"] = lg(tot, E, L)
    for c in bycc: M[f"g_{c}"] = lg(bycc[c], E, L)
    M["g_KR기원"] = lg(kro, E, L)
    M["한국기원점유_2015_17"] = kro[L].sum(axis=1) / tot[L].sum(axis=1)
    M["한국기원점유_변화pp"] = 100 * (M["한국기원점유_2015_17"] - kro[E].sum(axis=1) / tot[E].sum(axis=1))
    M["CN출원국점유_2015_17"] = bycc["CN"][L].sum(axis=1) / tot[L].sum(axis=1)
    M["gw_KR_lag0"] = lg(bycc["KR"], [2008, 2009], [2016, 2017])
    k10 = KO[(KO.기준 == "10차") & (KO.마스킹 == 0)]
    endw, span = ([2015, 2016], 7.0) if var.startswith("V1a") else ([2016, 2017], 8.0)
    for code, nm in (("T04", "출하액"), ("T02", "종사자수"), ("T01", "사업체수")):
        Wd = k10[k10.항목코드 == code].pivot_table(index="세세분류코드", columns="연도", values="값", aggfunc="sum").reindex(K50)
        if var == "V3" and code == "T04":
            if DEF is None:
                return None
            dp = DEF.pivot_table(index="ksic", columns="year", values="defl").reindex(K50)
            Wd = Wd / dp.reindex(columns=Wd.columns) * 100
        M[f"x_g_{nm}"] = (np.log(Wd[endw].mean(axis=1)) - np.log(Wd[[2008, 2009]].mean(axis=1))) / span
    if var == "V1a":
        M["gw_KR_lag0"] = lg(bycc["KR"], [2008, 2009], [2015, 2016])
    S = M.loc[K34]
    def perm_within(v, g):
        v = pd.Series(v).astype(float).values; g = np.asarray(g); ok = ~np.isnan(v); v, c = v[ok], _codes(g[ok])
        sst = ((v - v.mean()) ** 2).sum(); obs = ((_dm(v, c)) ** 2).sum() / sst
        sims = np.array([((_dm(v, rng.permutation(c))) ** 2).sum() / sst for _ in range(NP)])
        return obs, float(np.mean(sims)), float((sims <= obs + 1e-12).mean()), int(ok.sum())
    PAT = ["g_5국", "g_KR", "g_US", "g_JP", "g_CN", "g_EP", "g_KR기원", "한국기원점유_2015_17", "한국기원점유_변화pp", "CN출원국점유_2015_17"]
    for v in PAT:  # rng 소비 순서를 원 스크립트와 맞춘다
        if var in ("B0", "V2a"):
            perm_within(S[v], S.grp)
        else:
            perm_within(S[v], S.grp)
    out = {"var": var, "n34": len(K34), "n_groups": int(G[K34].nunique())}
    for v in ["x_g_출하액", "x_g_종사자수", "x_g_사업체수"]:
        obs, null, p, n = perm_within(S[v], S.grp)
        out[f"VAR_{v}"] = (round(obs, 4), round(null, 4), round(p, 4), n)
    # ---------- s4 (i) 부분 ----------
    rng2 = np.random.default_rng(11)
    P2 = P0[(P0.year >= 2000) & (P0.year <= 2022)]
    YR = list(range(2000, 2023))
    kr = P2[P2.cc == "KR"].groupby(["ksic", "year"]).n.sum().unstack().reindex(index=K50, columns=YR, fill_value=0)
    XS = {nm: k10[k10.항목코드 == c].pivot_table(index="세세분류코드", columns="연도", values="값", aggfunc="sum").reindex(K50)
          for c, nm in (("T02", "종사자수"), ("T04", "출하액"), ("T01", "사업체수"))}
    if var == "V3":
        dp = DEF.pivot_table(index="ksic", columns="year", values="defl").reindex(K50)
        XS["출하액"] = XS["출하액"] / dp.reindex(columns=XS["출하액"].columns) * 100
    lm = lambda A, ys: np.log(A[ys].mean(axis=1) + 1)
    if var.startswith("V1a"):
        Wp = [([2008, 2009], [2012, 2013], 4.0), ([2012, 2013], [2015, 2016], 3.0)]
    else:
        Wp = [([2008, 2009], [2012, 2013], 4.0), ([2012, 2013], [2016, 2017], 4.0)]
    Wpat = Wp if var == "V1a" else [([2008, 2009], [2012, 2013], 4.0), ([2012, 2013], [2016, 2017], 4.0)]
    for xn, X in XS.items():
        rows = []
        for wi, ((a, b, s), (pa, pb, ps)) in enumerate(zip(Wp, Wpat)):
            gp = (lm(kr, pb) - lm(kr, pa)) / ps
            gx = (np.log(X[b].mean(axis=1)) - np.log(X[a].mean(axis=1))) / s
            rows.append(pd.DataFrame({"k": K34, "w": wi, "g": G[K34].values, "p": gp[K34].values, "x": gx[K34].values}))
        d = pd.concat(rows).dropna()
        c = pd.factorize(d.g + "_" + d.w.astype(str))[0]
        dmv = lambda v: v - (np.bincount(c, weights=v) / np.bincount(c))[c]
        pv, xv = d.p.values, d.x.values
        r = np.corrcoef(dmv(pv), dmv(xv))[0, 1]
        cells = [np.where(c == x)[0] for x in np.unique(c)]
        sims = []
        for _ in range(NP):
            sp = pv.copy()
            for ix in cells:
                sp[ix] = pv[rng2.permutation(ix)]
            sims.append(np.corrcoef(dmv(sp), dmv(xv))[0, 1])
        out[f"FE_{xn}"] = (round(float(r), 4), round(float(np.mean(np.array(sims) >= r - 1e-12)), 4), int(len(d)))
    # ---------- 사례(s5) ----------
    cs = {}
    for k in ["26211", "26212", "26111", "29280", "26112", "26121", "26129", "26223", "26224", "26291", "29210", "29229", "29241"]:
        cs[k] = {nm: (None if k not in M.index or pd.isna(M.loc[k, f"x_g_{nm}"]) else round(100 * float(M.loc[k, f"x_g_{nm}"]), 3))
                 for nm in ("출하액", "종사자수", "사업체수")}
        cs[k]["KR출원_창"] = round(100 * float(M.loc[k, "gw_KR_lag0"]), 3)
        cs[k]["in_K34"] = k in K34
    out["cases"] = cs
    return out


if __name__ == "__main__":
    T = time.time()
    vars_ = sys.argv[1:] or ["B0", "V1a", "V1a_x", "V2a"]
    res = {}
    for v in vars_:
        r = run(v)
        res[v] = r
        print(v, json.dumps({k: x for k, x in (r or {}).items() if k != "cases"}, ensure_ascii=False), round(time.time() - T), "s", flush=True)
        json.dump(r, open(os.path.join(H, rf"cache\s3_{v}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
