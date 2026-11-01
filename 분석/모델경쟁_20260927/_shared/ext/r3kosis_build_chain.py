# -*- coding: utf-8 -*-
"""품목 연쇄 연결(chain-linking)로 분류 개정(KSIC8→9→10) 단절을 없앤 KSIC50 연간 성장률 패널.
1) 인접 연도쌍(y-1,y)마다 품목을 짝짓는다: ①코드+이름 일치 ②정규화 이름 일치 ③같은 코드(같은 분류기반)
   ④같은 중분류 안 유사도≥0.85(difflib). 한 품목은 한 번만 짝지음.
2) KSIC10 5자리는 2016~ 품목코드에서 시작해 짝 사슬을 따라 과거로 전파.
3) 성장률은 두 해 모두 값이 있는 '짝지은 품목'만으로 계산(구성 변화 제거).
산출: r3kosis_item_chain_links.parquet, r3kosis_ksic50_items_chain.parquet
"""
import os, re, json, difflib, unicodedata
import pandas as pd, numpy as np

EXT = os.path.dirname(os.path.abspath(__file__))
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
K50 = [x["code"] for x in json.load(open(os.path.join(BASE, "공모전데이터", "외부데이터", "KSIC해설",
                                                      "ksic50_definitions.json"), encoding="utf-8"))]
def norm(s):
    s = unicodedata.normalize("NFKC", str(s)).lower()
    return re.sub(r"[\s·ㆍ,.\-()/\[\]{}'\"’‘“”:;~_]+", "", s)

L = pd.read_parquet(os.path.join(EXT, "r3kosis_items_annual_long.parquet"))
L = L[(L.div2.str.isnumeric()) & (L.div2.astype(int).between(10, 34))]
W = L.pivot_table(index=["year", "code8", "item_nm"], columns="meas", values="val", aggfunc="first").reset_index()
W["n"] = W.item_nm.map(norm)
basis = lambda y: "k10" if y >= 2016 else ("k9" if y >= 2007 else "k8")
years = sorted(W.year.unique())

links = []
for y0, y1 in zip(years[:-1], years[1:]):
    A = W[W.year == y0][["code8", "n", "item_nm"]].copy(); B = W[W.year == y1][["code8", "n", "item_nm"]].copy()
    usedA, usedB, pairs = set(), set(), []
    def add(a, b, m):
        if a in usedA or b in usedB: return
        usedA.add(a); usedB.add(b); pairs.append((a, b, m))
    # ① 코드+이름
    m1 = A.merge(B, on=["code8", "n"])
    for c in m1.code8: add(c, c, "code_name")
    # ② 이름(유일할 때)
    An = A[~A.code8.isin(usedA)]; Bn = B[~B.code8.isin(usedB)]
    ua = An.groupby("n").code8.agg(list); ub = Bn.groupby("n").code8.agg(list)
    for n in set(ua.index) & set(ub.index):
        a, b = ua[n], ub[n]
        if len(a) == 1 and len(b) == 1: add(a[0], b[0], "name")
        else:  # 같은 중분류로 좁혀 1:1이면 채택
            for x in a:
                cand = [z for z in b if z[:2] == x[:2] and z not in usedB]
                if len(cand) == 1: add(x, cand[0], "name_div")
    # ③ 같은 분류기반이면 같은 코드
    if basis(y0) == basis(y1):
        rest = set(A.code8) - usedA
        for c in rest & (set(B.code8) - usedB): add(c, c, "code_only")
    # ④ 유사도
    An = A[~A.code8.isin(usedA)]; Bn = B[~B.code8.isin(usedB)]
    for _, ra in An.iterrows():
        cb = Bn[(Bn.code8.str[:2] == ra.code8[:2]) & (~Bn.code8.isin(usedB))]
        if cb.empty: continue
        sc = cb.n.map(lambda z: difflib.SequenceMatcher(None, ra.n, z).ratio())
        if sc.max() >= 0.85:
            add(ra.code8, cb.code8.loc[sc.idxmax()], f"fuzzy")
    for a, b, m in pairs: links.append((y0, a, y1, b, m))
LK = pd.DataFrame(links, columns=["y0", "code0", "y1", "code1", "method"])

# KSIC10 전파: 2016 이상은 코드 앞 5자리, 과거는 사슬 역추적
k10 = {(y, c): c[:5] for y, c in W[W.year >= 2016][["year", "code8"]].itertuples(index=False)}
for y0 in sorted([y for y in years if y < 2016], reverse=True):
    sub = LK[LK.y0 == y0]
    for c0, c1 in zip(sub.code0, sub.code1):
        v = k10.get((y0 + (sub.y1.iloc[0] - y0), c1))
        if v: k10[(y0, c0)] = v
W["k10"] = [k10.get((y, c)) for y, c in zip(W.year, W.code8)]
LK["k10"] = [k10.get((y, c)) for y, c in zip(LK.y1, LK.code1)]
LK.to_parquet(os.path.join(EXT, "r3kosis_item_chain_links.parquet"), index=False)
print("연도쌍 짝짓기 방법별 건수\n", LK.groupby(["y1", "method"]).size().unstack(fill_value=0).to_string())

# 연결률(생산액 가중) — 사슬 전파 후
rate = W.dropna(subset=["prod"]).groupby("year").apply(
    lambda g: g.loc[g.k10.notna(), "prod"].sum() / g["prod"].sum(), include_groups=False).rename("prod_share_k10")
print("\n사슬 전파 후 제조업 생산액 중 KSIC10 연결 비율\n", rate.round(3).to_string())
rate.to_csv(os.path.join(EXT, "r3kosis_item_link_rate_chain.csv"), encoding="utf-8-sig")

# 짝지은 품목 성장률
V = W.set_index(["year", "code8"])
out = []
for (y0, y1), sub in LK[LK.k10.isin(K50)].groupby(["y0", "y1"]):
    a = V.loc[[(y0, c) for c in sub.code0]].reset_index(drop=True)
    b = V.loc[[(y1, c) for c in sub.code1]].reset_index(drop=True)
    df = pd.DataFrame({"ksic": sub.k10.values})
    for m in ["prod", "ship", "inv", "estab"]:
        df[m + "0"] = a[m].values; df[m + "1"] = b[m].values
    for k, g in df.groupby("ksic"):
        r = {"ksic": k, "year": y1, "n_pairs": len(g)}
        for m in ["prod", "ship", "inv"]:
            ok = g[m + "0"].gt(0) & g[m + "1"].gt(0)
            r[f"g_{m}"] = np.log(g.loc[ok, m + "1"].sum() / g.loc[ok, m + "0"].sum()) if ok.any() else np.nan
            r[f"n_ok_{m}"] = int(ok.sum())
        ok = g.ship0.gt(0) & g.ship1.gt(0) & g.inv0.notna() & g.inv1.notna()
        if ok.any():
            r["inv_ship0"] = g.loc[ok, "inv0"].sum() / g.loc[ok, "ship0"].sum()
            r["inv_ship1"] = g.loc[ok, "inv1"].sum() / g.loc[ok, "ship1"].sum()
            r["d_inv_ship"] = r["inv_ship1"] - r["inv_ship0"]
        out.append(r)
C = pd.DataFrame(out)
# 해당 연도 KSIC10 전체 생산 대비 짝지은 품목 생산 비중(=성장률 대표성)
tot = W[W.k10.isin(K50)].groupby(["k10", "year"])["prod"].sum().rename("prod_all")
C = C.merge(tot, left_on=["ksic", "year"], right_index=True, how="left")
mp = []
for (y0, y1), sub in LK[LK.k10.isin(K50)].groupby(["y0", "y1"]):
    b = V.loc[[(y1, c) for c in sub.code1]].reset_index(drop=True)
    mp.append(pd.DataFrame({"ksic": sub.k10.values, "year": y1, "p": b["prod"].values}))
mp = pd.concat(mp).groupby(["ksic", "year"]).p.sum().rename("prod_paired")
C = C.merge(mp, left_on=["ksic", "year"], right_index=True, how="left")
C["paired_share"] = C.prod_paired / C.prod_all
# 연쇄지수(2003=100에서 앞으로만 누적 → 원점 이후 정보 없음). 결측 연도는 끊김 표시
C = C.sort_values(["ksic", "year"])
C["chain_prod_idx"] = C.groupby("ksic").g_prod.transform(lambda s: 100 * np.exp(s.fillna(0).cumsum()))
C["scope_break"] = C.year.eq(2007)  # 2003~2006 5명이상 → 2007~ 10명이상
C["avail_rule"] = "자료연도 Y는 Y+2년 1~2월 공표(LST_CHN_DE) → 원점 t0 말 기준 Y≤t0-2만 사용"
C.to_parquet(os.path.join(EXT, "r3kosis_ksic50_items_chain.parquet"), index=False)
cov = C.groupby("year").agg(ksic_n=("ksic", "nunique"), g_prod_ok=("g_prod", lambda s: int(s.notna().sum())),
                            d_inv_ship_ok=("d_inv_ship", lambda s: int(s.notna().sum())),
                            paired_share_med=("paired_share", "median"))
print("\nKSIC50 연쇄 성장률 커버리지\n", cov.round(3).to_string())
cov.to_csv(os.path.join(EXT, "r3kosis_ksic50_chain_coverage.csv"), encoding="utf-8-sig")
