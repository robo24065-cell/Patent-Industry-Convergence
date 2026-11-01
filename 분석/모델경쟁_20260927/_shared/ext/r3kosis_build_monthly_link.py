# -*- coding: utf-8 -*-
"""월별 표 → KSIC50 연결표·연간 요약.
- 생산능력·가동률(DT_1F32001, 소분류 3자리): KSIC50 앞 3자리로 연결.
- 제조업 국내공급지수(DT_1MS2001/2003): 3자리(C261~C265)가 있으면 그것, 없으면 2자리 중분류.
- 품목별 물량(DT_1F02012, 66품목): 연간 품목편(KSIC10, 2016~) 품목명과 정규화 일치/포함으로 KSIC10 부여 + 수작업 확인.
모든 지수는 2020=100 재기준 현재 빈티지 → 수준값은 누수. 성장률·비율만 산출한다.
"""
import os, re, json, unicodedata
import pandas as pd, numpy as np

EXT = os.path.dirname(os.path.abspath(__file__))
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
KD = json.load(open(os.path.join(BASE, "공모전데이터", "외부데이터", "KSIC해설", "ksic50_definitions.json"), encoding="utf-8"))
K50 = [x["code"] for x in KD]; KNM = {x["code"]: x["codenmKr"] for x in KD}
def norm(s):
    s = unicodedata.normalize("NFKC", str(s)).lower()
    return re.sub(r"[\s·ㆍ,.\-()/\[\]{}'\"’‘“”:;~_]+", "", s)

# ---------- 1) 생산능력·가동률 ----------
C = pd.read_parquet(os.path.join(EXT, "r3kosis_capacity_util_monthly.parquet"))
C["year"] = C.PRD_DE.str[:4].astype(int); C["mon"] = C.PRD_DE.str[4:].astype(int)
codes = set(C.C1)
link = []
for k in K50:
    c3 = "C" + k[:3]; c2 = "C" + k[:2]
    link.append({"ksic": k, "ksic_nm": KNM[k], "cap_code": c3 if c3 in codes else (c2 if c2 in codes else None),
                 "cap_level": "소분류3" if c3 in codes else ("중분류2" if c2 in codes else "없음")})
LC = pd.DataFrame(link)
Y = C.groupby(["C1", "ITM_ID", "year"]).agg(v=("DT_num", "mean"), nmon=("DT_num", "count")).reset_index()
Y = Y[Y.nmon == 12].pivot_table(index=["C1", "year"], columns="ITM_ID", values="v").reset_index()
Y.columns = ["code", "year", "cap_idx_avg", "util_idx_avg"]
Y = Y.sort_values(["code", "year"])
for h in (1, 3):
    Y[f"g{h}_cap"] = Y.groupby("code").cap_idx_avg.transform(lambda s: np.log(s / s.shift(h)))
    Y[f"g{h}_util"] = Y.groupby("code").util_idx_avg.transform(lambda s: np.log(s / s.shift(h)))
Y = Y.drop(columns=["cap_idx_avg", "util_idx_avg"])  # 수준값(2020=100)은 저장하지 않는다(누수 방지)
Y["avail_note"] = "연평균은 이듬해 1월 말 공표 → 원점 t0 말 기준 year≤t0-1 권장(성장률만)"
CP = LC.merge(Y, left_on="cap_code", right_on="code", how="left").drop(columns="code")
CP.to_parquet(os.path.join(EXT, "r3kosis_ksic50_capacity_util_annual.parquet"), index=False)

# ---------- 2) 국내공급지수 ----------
D1 = pd.read_parquet(os.path.join(EXT, "r3kosis_domsupply_ind_monthly.parquet"))
dcodes = set(D1.C1)
LD = []
for k in K50:
    c3 = "C" + k[:3]; c2 = "C" + k[:2]
    LD.append({"ksic": k, "ds_code": c3 if c3 in dcodes else (c2 if c2 in dcodes else None),
               "ds_level": "소분류3" if c3 in dcodes else ("중분류2" if c2 in dcodes else "없음")})
LD = pd.DataFrame(LD)
D1["year"] = D1.PRD_DE.str[:4].astype(int)
YD = D1.groupby(["C1", "ITM_ID", "year"]).agg(v=("DT_num", "mean"), nmon=("DT_num", "count")).reset_index()
YD = YD[YD.nmon == 12].pivot_table(index=["C1", "year"], columns="ITM_ID", values="v").reset_index()
YD = YD.rename(columns={"C1": "code", "T1": "dom_supply", "T2": "dom_made", "T3": "imported"}).sort_values(["code", "year"])
for c in ["dom_supply", "dom_made", "imported"]:
    for h in (1, 3):
        YD[f"g{h}_{c}"] = YD.groupby("code")[c].transform(lambda s: np.log(s / s.shift(h)))
YD["g3_import_minus_dom"] = YD["g3_imported"] - YD["g3_dom_made"]
YD = YD.drop(columns=["dom_supply", "dom_made", "imported"])
DS = LD.merge(YD, left_on="ds_code", right_on="code", how="left").drop(columns="code")
DS.to_parquet(os.path.join(EXT, "r3kosis_ksic50_domsupply_annual.parquet"), index=False)

# ---------- 3) 품목별 월간 물량 → KSIC10 ----------
V = pd.read_parquet(os.path.join(EXT, "r3kosis_item_volume_monthly.parquet"))
LA = pd.read_parquet(os.path.join(EXT, "r3kosis_items_annual_long.parquet"))
ref = LA[LA.year.between(2016, 2024)][["code8", "item_nm"]].drop_duplicates()
ref["n"] = ref.item_nm.map(norm); ref["k10"] = ref.code8.str[:5]
vi = V[["C1", "C1_NM", "UNIT_NM"]].drop_duplicates("C1")
rows = []
for c, nm, u in vi.itertuples(index=False):
    n = norm(re.sub(r"\(.*?\)", "", nm))
    ex = ref[ref.n == n]; how = "exact"
    if ex.empty:
        ex = ref[ref.n.str.contains(re.escape(n))]; how = "contains"
    ks = sorted(set(ex.k10))
    # 월별표 코드 앞 3자리 = KSIC 소분류 → 같은 소분류 후보만 남김
    ks3 = [k for k in ks if k[:3] == c[:3]] or ks
    rows.append({"item_code": c, "item_nm": nm, "unit": u, "match": how if ks else "none",
                 "k10_cands": "|".join(ks3), "ksic10": ks3[0] if len(ks3) == 1 else None,
                 "in_ksic50": (ks3[0] in K50) if len(ks3) == 1 else any(k in K50 for k in ks3)})
IV = pd.DataFrame(rows)
IV.to_csv(os.path.join(EXT, "r3kosis_item_volume_ksic10_link.csv"), index=False, encoding="utf-8-sig")
print(IV.to_string())
print("\n월별 품목 66개 중 KSIC10 단일 연결:", int(IV.ksic10.notna().sum()), "/ KSIC50에 속함:", int(IV.in_ksic50.sum()),
      "→ KSIC50 코드", sorted(set(IV.loc[IV.in_ksic50 & IV.ksic10.notna(), "ksic10"])))
print("\n생산능력·가동률 연결 수준:", LC.cap_level.value_counts().to_dict(), "미연결:", LC.loc[LC.cap_level == "없음", "ksic"].tolist())
print("국내공급지수 연결 수준:", LD.ds_level.value_counts().to_dict())
LC.merge(LD, on="ksic").to_csv(os.path.join(EXT, "r3kosis_ksic50_monthly_link.csv"), index=False, encoding="utf-8-sig")
