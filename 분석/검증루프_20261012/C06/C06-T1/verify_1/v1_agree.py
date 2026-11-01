# -*- coding: utf-8 -*-
# verify_1: DART(2026) vs NPS(2019) 5자리 일치 독립 재계산. 읽기만. 쓰기: verify_1/v1_result.json
import os, sys, json
sys.dont_write_bytecode = True
import pandas as pd, numpy as np
MC = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927"
R = os.path.join(MC, r"_shared_report\label_origin_R10")
OUT = os.path.dirname(os.path.abspath(__file__))
KSIC50 = ['15220','20202','20411','20423','20491','21220','23992','24191','24219','25914','25922','25924','25932','26111','26112','26121','26129','26211','26212','26223','26224','26291','26322','26323','26422','26429','26519','26529','27199','27211','27212','27301','28111','28122','28511','28902','29131','29132','29133','29150','29162','29210','29229','29241','29280','30393','31202','31312','33301','33309']
S50 = set(KSIC50)
reg = pd.read_parquet(os.path.join(R, r"confirm_R12\cache\reg12.parquet"))
reg["applicantCd"] = reg.applicantCd.astype(str)
LB = pd.read_parquet(os.path.join(R, r"confirm_R12\cache\labels.parquet"), columns=["appl_no", "split"])
LB["appl_no"] = LB.appl_no.astype(str)
pf = pd.read_parquet(os.path.join(R, r"cache\pat_firm.parquet")); pf["appl_no"] = pf.appl_no.astype(str); pf["applicantCd"] = pf.applicantCd.astype(str)
ev = set(LB.appl_no[LB.split == "eval"])
Ucd = set(pf.applicantCd[pf.appl_no.isin(ev)])
J = {"cols": list(reg.columns), "reg_rows": len(reg), "U": len(Ucd)}
def clean(s): return s.astype(str).str.replace(r"\D", "", regex=True).where(s.notna())
reg["d"] = clean(reg.dart); reg["n"] = clean(reg.nps)
reg["d5"] = reg.d.where(reg.d.str.len() == 5); reg["n5"] = reg.n.where(reg.n.str.len() == 5)
U = reg[reg.applicantCd.isin(Ucd)].drop_duplicates("applicantCd")
def agree(df, tag):
    m = df.d5.notna() & df.n5.notna(); x = df[m]
    J[tag] = {"n": int(len(x)), "eq5": round(float((x.d5 == x.n5).mean()), 4) if len(x) else None,
              "eq4": round(float((x.d5.str[:4] == x.n5.str[:4]).mean()), 4) if len(x) else None,
              "eq3": round(float((x.d5.str[:3] == x.n5.str[:3]).mean()), 4) if len(x) else None,
              "eq2": round(float((x.d5.str[:2] == x.n5.str[:2]).mean()), 4) if len(x) else None}
agree(U[U.d5.isin(S50)], "G1_DART5in50_withNPS")          # 원 발견의 조건
agree(U[U.n5.isin(S50)], "NPS5in50_withDART")             # 반대 방향 조건
agree(U[U.d5.isin(S50) & U.n5.isin(S50)], "both_in50")
agree(U, "U_all_both5")                                    # 조건 없이 U 전체
agree(reg.drop_duplicates("applicantCd"), "reg_all_both5")  # 등록표 전체
# 조건부 비율: DART in 50 기업 중 NPS도 50 안 / NPS in 50 중 DART도 50 안
g1 = U[U.d5.isin(S50) & U.n5.notna()]; n5 = U[U.n5.isin(S50) & U.d5.notna()]
J["G1_nps_in50_share"] = round(float(g1.n5.isin(S50).mean()), 4)
J["NPS5_dart_in50_share"] = round(float(n5.d5.isin(S50).mean()), 4)
# G1 수와 NPS 결측
J["G1_n"] = int(U.d5.isin(S50).sum()); J["G1_nps_notna"] = int((U.d5.isin(S50) & U.n.notna()).sum())
J["G1_nps_len"] = U[U.d5.isin(S50)].n.str.len().value_counts(dropna=False).astype(int).to_dict()
J["NPS5_n"] = int(U.n5.isin(S50).sum())
# 기존 산출 대조: C06-T1 firm_U.parquet의 nps
FU = pd.read_parquet(os.path.join(os.path.dirname(OUT), r"cache\firm_U.parquet"), columns=["applicantCd", "nps", "dart"])
FU["applicantCd"] = FU.applicantCd.astype(str)
cmp_ = U[["applicantCd", "n", "d"]].merge(FU, on="applicantCd", how="outer")
J["firmU_match_nps"] = round(float(((cmp_.n == cmp_.nps) | (cmp_.n.isna() & cmp_.nps.isna())).mean()), 4)
J["firmU_rows"] = len(FU)
json.dump(J, open(os.path.join(OUT, "v1_result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(J, ensure_ascii=True, default=str)[:3000])
