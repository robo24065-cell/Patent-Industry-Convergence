# -*- coding: utf-8 -*-
"""연간 품목편 → (연도, 품목) 긴 패널 + KSIC10 5자리 연결 + KSIC50 집계.
- 2016~2024 자료는 품목코드 앞 5자리가 KSIC10 → 직접 연결.
- 2003~2015 자료(KSIC 8·9차 기반 코드)는 '품목명 정규화 일치'로 KSIC10 품목에 다리를 놓는다
  (참조: 2016~2019 KSIC10 품목표의 품목명 → KSIC10 5자리). 후보가 여럿이면 같은 2자리(중분류) 우선,
  그래도 여럿이면 모호(ambiguous)로 두고 제외.
산출: r3kosis_items_annual_long.parquet, r3kosis_item_ksic10_bridge.csv, r3kosis_ksic50_items_annual.parquet
"""
import os, re, glob, unicodedata, json
import pandas as pd, numpy as np

EXT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(EXT, "r3kosis_raw", "items_annual")
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
K50 = [x["code"] for x in json.load(open(os.path.join(BASE, "공모전데이터", "외부데이터", "KSIC해설",
                                                      "ksic50_definitions.json"), encoding="utf-8"))]
MEAS = {"T01": "estab", "T10": "estab", "T02": "prod", "T20": "prod",
        "T03": "ship", "T05": "ship", "T30": "ship", "T04": "inv", "T06": "inv", "T40": "inv"}

def norm(s):
    s = unicodedata.normalize("NFKC", str(s)).lower()
    return re.sub(r"[\s·ㆍ,.\-()/\[\]{}'\"’‘“”:;~_]+", "", s)

rows = []
for f in sorted(glob.glob(os.path.join(RAW, "DT_*.parquet"))):
    d = pd.read_parquet(f)
    d = d[d["C1"].astype(str) == "00"].copy()
    code = d["C2"].astype(str)
    d = d[code.str.len() >= 8].copy()
    d["code8"] = d["C2"].astype(str).str[-8:]
    d["item_nm"] = d["C2_NM"].astype(str)
    d["meas"] = d["ITM_ID"].map(MEAS)
    d["val"] = pd.to_numeric(d["DT"], errors="coerce")
    d["masked"] = d["DT"].astype(str).str.strip().eq("X")
    d["year"] = d["PRD_DE"].astype(int)
    rows.append(d[["year", "tbl", "scope", "code8", "item_nm", "meas", "val", "masked"]])
L = pd.concat(rows, ignore_index=True)
L["ksic_raw5"] = L["code8"].str[:5]
L["div2"] = L["code8"].str[:2]

# --- KSIC10 참조 사전 (2016~2019) ---
ref = L[(L.year >= 2016) & (L.year <= 2019)][["code8", "item_nm"]].drop_duplicates()
ref["k10"] = ref.code8.str[:5]; ref["n"] = ref.item_nm.map(norm)
ref_map = ref.groupby("n")["k10"].agg(lambda s: sorted(set(s))).to_dict()
# 2016 이후 코드→k10 (2020~2024도 KSIC10 코드인지 이름으로 확인)
items = L[["year", "code8", "item_nm", "div2", "ksic_raw5"]].drop_duplicates()

def link(r):
    if r.year >= 2016:
        return r.ksic_raw5, "direct_k10"
    c = ref_map.get(norm(r.item_nm))
    if not c:
        return None, "unmatched"
    if len(c) == 1:
        return c[0], "name_exact"
    same = [k for k in c if k[:2] == r.div2]
    if len(same) == 1:
        return same[0], "name_exact_div"
    return None, "ambiguous"

lk = items.apply(lambda r: pd.Series(link(r), index=["k10", "link"]), axis=1)
items = pd.concat([items, lk], axis=1)
items.to_csv(os.path.join(EXT, "r3kosis_item_ksic10_bridge.csv"), index=False, encoding="utf-8-sig")
L = L.merge(items[["year", "code8", "k10", "link"]], on=["year", "code8"], how="left")
L.to_parquet(os.path.join(EXT, "r3kosis_items_annual_long.parquet"), index=False)

# --- 연결률 (생산액 가중, 제조업 C만) ---
P = L[(L.meas == "prod")]
P = P[~P.code8.str[:2].isin(["05", "06", "07", "08"]) & (P.code8.str[:2].astype(int) >= 10) & (P.code8.str[:2].astype(int) <= 34)]
rate = P.assign(ok=P.k10.notna()).groupby("year").apply(
    lambda g: pd.Series({"items": len(g), "items_linked": int(g.ok.sum()),
                         "prod_share_linked": g.loc[g.ok, "val"].sum() / g["val"].sum()}), include_groups=False)
print("연도별 제조업 품목 연결률(생산액 가중)\n", rate.round(3).to_string())
rate.to_csv(os.path.join(EXT, "r3kosis_item_link_rate_by_year.csv"), encoding="utf-8-sig")

# --- KSIC50 집계 ---
S = L[L.k10.isin(K50)]
agg = S.pivot_table(index=["k10", "year"], columns="meas", values="val", aggfunc="sum").reset_index()
nit = S[S.meas == "prod"].groupby(["k10", "year"]).agg(
    n_items=("code8", "nunique"),
    n_items_pos=("val", lambda s: int((s > 0).sum())),
    n_masked=("masked", "sum")).reset_index()
hhi = S[(S.meas == "prod") & (S.val > 0)].groupby(["k10", "year"])["val"].apply(
    lambda s: float(((s / s.sum()) ** 2).sum())).rename("item_hhi").reset_index()
A = agg.merge(nit, on=["k10", "year"], how="left").merge(hhi, on=["k10", "year"], how="left")
A["inv_ship"] = A["inv"] / A["ship"]
sc = L[["year", "scope"]].drop_duplicates().groupby("year")["scope"].first()
A["scope"] = A.year.map(sc)
A["ksic_basis"] = np.where(A.year >= 2016, "KSIC10 코드 직접", "KSIC8·9 품목명→KSIC10 다리")
A = A.rename(columns={"k10": "ksic"})
A.to_parquet(os.path.join(EXT, "r3kosis_ksic50_items_annual.parquet"), index=False)
cov = A.groupby("year").agg(ksic_n=("ksic", "nunique"), ksic_prod_pos=("prod", lambda s: int((s > 0).sum())),
                            ksic_inv_ok=("inv_ship", lambda s: int(s.notna().sum())))
print("\nKSIC50 연도별 커버리지\n", cov.to_string())
cov.to_csv(os.path.join(EXT, "r3kosis_ksic50_items_coverage_by_year.csv"), encoding="utf-8-sig")
