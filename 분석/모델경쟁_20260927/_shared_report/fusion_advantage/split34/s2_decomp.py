# -*- coding: utf-8 -*-
r"""split34 2단계 (개발 기간 2005–2017만): 연계표가 한 코드로 묶는 34개 KSIC(11개 그룹)
(1) 그룹 안 세세분류의 특허 추세·한국 점유·성장률 분산 분해(그룹 간 vs 그룹 안) + 무작위 묶음 순열 귀무
(2) 그룹 안 차이가 외부 산업 지표(KOSIS 5자리 출하·종사자·사업체, HS6 무역)에서도 실재하는지:
    그룹 평균을 뺀 값끼리의 상관(그룹 안 순열 검정). 연계표 해상도는 그룹 안 차이를 정의상 0으로 만든다.
    + 34개 전체 단면 상관: 제공 5자리 계열 vs 연계표 해상도 계열(그룹 합계를 구성원 모두에 배정, 연계표에 유리한 상한)
무역: HS6가 그 KSIC에만 걸린 코드(배타 HS6)만 쓴다. 공유 HS6는 같은 몫 분할이라 그룹 안 차이를 인위적으로 지운다.
출력은 이 폴더에만."""
import os, sys, json, time
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
HERE = os.path.dirname(os.path.abspath(__file__))
MC = os.path.abspath(os.path.join(HERE, r"..\..\.."))
BASE = os.path.abspath(os.path.join(MC, r"..\.."))
EXT = os.path.join(MC, r"_shared\ext")
sys.path.insert(0, os.path.join(MC, r"_shared_report\label_validity"))
from lv_common import ksic_to_group
T0 = time.time()
rng = np.random.default_rng(20261005)
NP = 5000
Y0, Y1 = 2005, 2017  # 개발 기간(가장 엄격한 T1 F=2018 이전)

P = pd.read_parquet(os.path.join(HERE, "panel_ksic_cc_year.parquet"))
P = P[(P.year >= Y0) & (P.year <= Y1)]
K50 = sorted(P.ksic.unique())
G = pd.Series({k: ksic_to_group(k) for k in K50})
gs = G.value_counts()
K34 = sorted([k for k in K50 if gs[G[k]] > 1])
ko = pd.read_csv(os.path.join(BASE, r"공모전데이터\작업데이터\kosis_광업제조업_50개산업.csv"), dtype={"세세분류코드": str})
NAME = ko.drop_duplicates("세세분류코드").set_index("세세분류코드").산업명
assert len(K34) == 34 and G[K34].nunique() == 11
years = list(range(Y0, Y1 + 1))
tot = P.groupby(["ksic", "year"]).n.sum().unstack().reindex(index=K50, columns=years, fill_value=0)
kro = P.groupby(["ksic", "year"]).n_kro.sum().unstack().reindex(index=K50, columns=years, fill_value=0)
bycc = {c: P[P.cc == c].groupby(["ksic", "year"]).n.sum().unstack().reindex(index=K50, columns=years, fill_value=0) for c in ["KR", "US", "JP", "CN", "EP"]}


def lg(A, a, b):  # 연평균 로그 성장률: 구간 평균(a 연도들) → (b 연도들)
    return (np.log(A[b].mean(axis=1) + 1) - np.log(A[a].mean(axis=1) + 1)) / (np.mean(b) - np.mean(a))


E, L = [2005, 2006, 2007], [2015, 2016, 2017]
M = pd.DataFrame(index=K50)
M["grp"] = G
M["name"] = NAME.reindex(K50)
M["n5_2015_17"] = tot[L].mean(axis=1)
M["g_5국"] = lg(tot, E, L)
for c in bycc:
    M[f"g_{c}"] = lg(bycc[c], E, L)
M["g_KR기원"] = lg(kro, E, L)
M["한국기원점유_2015_17"] = kro[L].sum(axis=1) / tot[L].sum(axis=1)
M["한국기원점유_변화pp"] = 100 * (M["한국기원점유_2015_17"] - kro[E].sum(axis=1) / tot[E].sum(axis=1))
M["KR출원국점유_2015_17"] = bycc["KR"][L].sum(axis=1) / tot[L].sum(axis=1)
M["CN출원국점유_2015_17"] = bycc["CN"][L].sum(axis=1) / tot[L].sum(axis=1)
# 같은 창(KOSIS 2008–09 → 2016–17)과 3년 선행 창 특허 성장
for nm, A in (("5국", tot), ("KR", bycc["KR"]), ("KR기원", kro)):
    M[f"gw_{nm}_lag0"] = lg(A, [2008, 2009], [2016, 2017])
    M[f"gw_{nm}_lag3"] = lg(A, [2005, 2006], [2013, 2014])

# ---- 외부 지표 ----
k10 = ko[(ko.기준 == "10차") & (ko.마스킹 == 0)]
X = pd.DataFrame(index=K50)
for code, nm in (("T04", "출하액"), ("T02", "종사자수"), ("T01", "사업체수"), ("T06", "부가가치")):
    W = k10[k10.항목코드 == code].pivot_table(index="세세분류코드", columns="연도", values="값", aggfunc="sum").reindex(K50)
    X[f"x_g_{nm}"] = (np.log(W[[2016, 2017]].mean(axis=1)) - np.log(W[[2008, 2009]].mean(axis=1))) / 8.0
mp = pd.read_csv(os.path.join(BASE, r"분석\fusion\ksic_hs6_map.csv"), dtype=str).drop_duplicates(["k5", "hs6"])
mp["nk"] = mp.groupby("hs6").k5.transform("nunique")
mpx = mp[mp.nk == 1][["k5", "hs6"]]
X["n_배타HS6"] = mpx.groupby("k5").size().reindex(K50).fillna(0)
ex = pd.read_csv(os.path.join(EXT, "trade_kr_export_hs6_uv_2000_2021.csv"), dtype={"hs6": str}, usecols=["year", "hs6", "export_usd"])
im = pd.read_csv(os.path.join(BASE, r"분석\fusion\hs6_import_yearly.csv"), dtype={"hs6": str})
im = im[im.partner == "WLD"]
c5 = pd.concat([pd.read_csv(os.path.join(EXT, f), dtype={"hs6": str}, usecols=["year", "reporter", "hs6", "export_usd"])
                for f in ("R2ind_comtrade_exports_5c_hs6_2004_2021.csv", "R2N3_comtrade_exports_5c_hs6.csv")])


def trade_panel(d, col):
    z = d.merge(mpx, on="hs6")
    return z.pivot_table(index="k5", columns="year", values=col, aggfunc="sum").reindex(K50)


EXP = trade_panel(ex, "export_usd")
IMP = trade_panel(im.rename(columns={"value": "v"}), "v")
X["x_g_수출"] = (np.log(EXP[L].mean(axis=1)) - np.log(EXP[E].mean(axis=1))) / 10
X["x_g_수입"] = (np.log(IMP[L].mean(axis=1)) - np.log(IMP[E].mean(axis=1))) / 10
X["x_g_수출_w"] = (np.log(EXP[[2016, 2017]].mean(axis=1)) - np.log(EXP[[2008, 2009]].mean(axis=1))) / 8
c5k = c5.merge(mpx, on="hs6")
krx = c5k[c5k.reporter == 410].pivot_table(index="k5", columns="year", values="export_usd", aggfunc="sum").reindex(K50)
allx = c5k.pivot_table(index="k5", columns="year", values="export_usd", aggfunc="sum").reindex(K50)
X["x_한국수출점유_2015_17"] = krx[L].sum(axis=1) / allx[L].sum(axis=1)
X["x_한국수출점유_변화pp"] = 100 * (X["x_한국수출점유_2015_17"] - krx[E].sum(axis=1) / allx[E].sum(axis=1))
M = M.join(X)
M.to_csv(os.path.join(HERE, "ksic_metrics_dev.csv"), encoding="utf-8-sig")


# ---- (1) 분산 분해 ----
def _codes(g):
    return pd.factorize(np.asarray(g))[0]


def _dm(v, c):
    m = np.bincount(c, weights=v) / np.bincount(c)
    return v - m[c]


def within_share(v, g):
    v = pd.Series(v).astype(float).values; g = np.asarray(g); ok = ~np.isnan(v); v, c = v[ok], _codes(g[ok])
    sst = ((v - v.mean()) ** 2).sum()
    return (((_dm(v, c)) ** 2).sum() / sst if sst > 0 else np.nan), int(ok.sum())


def perm_within(v, g, n=NP):
    v = pd.Series(v).astype(float).values; g = np.asarray(g); ok = ~np.isnan(v); v, c = v[ok], _codes(g[ok])
    sst = ((v - v.mean()) ** 2).sum()
    obs = ((_dm(v, c)) ** 2).sum() / sst
    sims = np.array([((_dm(v, rng.permutation(c))) ** 2).sum() / sst for _ in range(n)])
    return obs, float(np.mean(sims)), float((sims <= obs + 1e-12).mean())


PAT = ["g_5국", "g_KR", "g_US", "g_JP", "g_CN", "g_EP", "g_KR기원", "한국기원점유_2015_17", "한국기원점유_변화pp", "CN출원국점유_2015_17"]
EXTV = ["x_g_출하액", "x_g_종사자수", "x_g_사업체수", "x_g_부가가치", "x_g_수출", "x_g_수입", "x_한국수출점유_2015_17", "x_한국수출점유_변화pp"]
rows = []
S = M.loc[K34]
for v in PAT + EXTV:
    obs, null, p = perm_within(S[v], S.grp)
    w50, n50 = within_share(M[v], M.grp)  # 50개 전체(단독 16개는 그룹 안 분산 0)
    rows.append({"지표": v, "구분": "특허(제공)" if v in PAT else "외부", "n(34중)": int(S[v].notna().sum()),
                 "그룹안_분산비중_34": round(obs, 3), "무작위묶음_기대": round(null, 3), "p(더_동질)": round(p, 4),
                 "그룹안_분산비중_50전체": round(w50, 3),
                 "그룹안_SD": round(float((S[v] - S.groupby("grp")[v].transform("mean")).dropna().std(ddof=0)), 4),
                 "전체_SD": round(float(S[v].std(ddof=0)), 4)})
D1 = pd.DataFrame(rows)
D1.to_csv(os.path.join(HERE, "t1_variance_decomposition.csv"), index=False, encoding="utf-8-sig")
print(D1.to_string()); print(round(time.time() - T0), "s")

# 연간 동조성: 같은 그룹 쌍 vs 다른 그룹 쌍의 Δlog(5국 합계) 상관
dl = np.log(tot.loc[K34] + 1).diff(axis=1).iloc[:, 1:]
C = np.corrcoef(dl.values)
same, diff = [], []
for i in range(34):
    for j in range(i + 1, 34):
        (same if G[K34[i]] == G[K34[j]] else diff).append(C[i, j])
coh = {"같은그룹쌍_n": len(same), "같은그룹쌍_평균상관": round(float(np.mean(same)), 3), "같은그룹쌍_중앙값": round(float(np.median(same)), 3),
       "같은그룹쌍_음의상관비율": round(float(np.mean(np.array(same) < 0)), 3),
       "다른그룹쌍_n": len(diff), "다른그룹쌍_평균상관": round(float(np.mean(diff)), 3)}
print(coh)


# ---- (2) 그룹 안 정합: 특허 차이 ↔ 외부 차이 ----
def dm(s, g):
    return s - s.groupby(g).transform("mean")


def within_corr(a, b, g, n=NP):
    d = pd.DataFrame({"a": a, "b": b, "g": g}).dropna()
    d = d[d.g.map(d.g.value_counts()) > 1]  # 그룹 안 비교 가능한 구성원 2개 이상
    if len(d) < 6:
        return None
    c = _codes(d.g); av, bv = d.a.values.astype(float), d.b.values.astype(float)
    da, db = _dm(av, c), _dm(bv, c)
    r = float(np.corrcoef(da, db)[0, 1]); rho = float(spearmanr(da, db)[0])
    gidx = [np.where(c == x)[0] for x in np.unique(c)]
    sims = np.empty(n)
    for i in range(n):
        ap = av.copy()
        for ix in gidx:
            ap[ix] = av[rng.permutation(ix)]
        sims[i] = np.corrcoef(_dm(ap, c), db)[0, 1]
    bs = []
    for _ in range(2000):  # 그룹 단위 부트스트랩
        pick = rng.integers(0, len(gidx), len(gidx))
        ix = np.concatenate([gidx[j] for j in pick])
        cc = np.concatenate([np.full(len(gidx[j]), k) for k, j in enumerate(pick)])
        xa, xb = _dm(av[ix], cc), _dm(bv[ix], cc)
        if xa.std() > 0 and xb.std() > 0:
            bs.append(np.corrcoef(xa, xb)[0, 1])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"n": len(d), "n그룹": int(len(gidx)), "그룹안_r": round(r, 3), "그룹안_rho": round(rho, 3),
            "순열p(단측,r>0)": round(float((sims >= r - 1e-12).mean()), 4), "r_95CI_그룹부트": f"[{lo:.3f}, {hi:.3f}]",
            "같은부호_비율": round(float((np.sign(da) == np.sign(db)).mean()), 3)}


pairs = []
for pv in ["gw_5국_lag0", "gw_KR_lag0", "gw_KR기원_lag0", "gw_5국_lag3", "gw_KR_lag3", "gw_KR기원_lag3"]:
    for xv in ["x_g_출하액", "x_g_종사자수", "x_g_사업체수", "x_g_부가가치"]:
        pairs.append((pv, xv))
for pv in ["g_5국", "g_KR", "g_KR기원"]:
    for xv in ["x_g_수출", "x_g_수입"]:
        pairs.append((pv, xv))
pairs += [("한국기원점유_2015_17", "x_한국수출점유_2015_17"), ("한국기원점유_변화pp", "x_한국수출점유_변화pp"),
          ("KR출원국점유_2015_17", "x_한국수출점유_2015_17")]
rows = []
for pv, xv in pairs:
    r = within_corr(M.loc[K34, pv], M.loc[K34, xv], M.loc[K34, "grp"])
    if r is None:
        continue
    # 34개 전체 단면: 제공 5자리 vs 연계표 해상도(그룹값을 구성원 모두에 배정)
    d = M.loc[K34, [pv, xv, "grp"]].dropna()
    r_full = float(pearsonr(d[pv], d[xv])[0])
    r.update({"특허지표": pv, "외부지표": xv, "34단면_r_제공": round(r_full, 3)})
    rows.append(r)
D2 = pd.DataFrame(rows)
D2.to_csv(os.path.join(HERE, "t2_within_group_external.csv"), index=False, encoding="utf-8-sig")
print(D2.to_string()); print(round(time.time() - T0), "s")

# ---- (2b) 34개 단면: 제공 5자리 계열 vs 연계표 해상도 상한(그룹 합계를 구성원에 배정) ----
grp_tot = {nm: A.loc[K50].groupby(G).sum() for nm, A in (("5국", tot), ("KR", bycc["KR"]), ("KR기원", kro))}
rows = []
for nm, A in (("5국", tot), ("KR", bycc["KR"]), ("KR기원", kro)):
    Ag = grp_tot[nm].reindex(G[K34].values).set_axis(K34)
    for lag, (a, b) in (("lag0", ([2008, 2009], [2016, 2017])), ("lag3", ([2005, 2006], [2013, 2014]))):
        gp, gc = lg(A.loc[K34], a, b), lg(Ag, a, b)
        for xv in ["x_g_출하액", "x_g_종사자수", "x_g_사업체수", "x_g_부가가치"]:
            d = pd.DataFrame({"p": gp, "c": gc, "x": M.loc[K34, xv]}).dropna()
            rp, rc = pearsonr(d.p, d.x)[0], pearsonr(d.c, d.x)[0]
            bs = []
            for _ in range(2000):
                q = d.iloc[rng.integers(0, len(d), len(d))]
                if q.c.std() > 0:
                    bs.append(np.corrcoef(q.p, q.x)[0, 1] - np.corrcoef(q.c, q.x)[0, 1])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rows.append({"특허": nm, "창": lag, "외부지표": xv, "n": len(d), "r_제공5자리": round(rp, 3), "r_연계표해상도(그룹합,상한)": round(rc, 3),
                         "차이": round(rp - rc, 3), "차이_95CI": f"[{lo:.3f}, {hi:.3f}]"})
D3 = pd.DataFrame(rows)
D3.to_csv(os.path.join(HERE, "t2b_cross_section_prov_vs_cw.csv"), index=False, encoding="utf-8-sig")
print(D3.to_string())
json.dump({"연간동조성_Δlog5국": coh, "K34": K34, "groups": {g: [k for k in K34 if G[k] == g] for g in sorted(G[K34].unique())},
           "배타HS6_보유_34중": int((M.loc[K34, "n_배타HS6"] > 0).sum()), "permutations": NP},
          open(os.path.join(HERE, "t1_meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("done", round(time.time() - T0), "s")
