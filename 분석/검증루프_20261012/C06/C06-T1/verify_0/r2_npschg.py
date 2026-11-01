# -*- coding: utf-8 -*-
r"""C06-T1 verify_0 r2: NPS 원 스냅샷(_shared\ext firm_nps_wkpl_*.parquet)에서 G1(DART5 평가 811사)의 업종 코드(indcd 원값) 변경을 직접 다시 센다.
k5(이름 매핑)는 쓰지 않고 indcd 원값과 indnm 원문 두 가지로. 쓰기: verify_0\r2_result.json"""
import os, sys, json, re, sqlite3
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"; MC = os.path.join(BASE, r"분석\모델경쟁_20260927"); EXT = os.path.join(MC, r"_shared\ext")
R12 = os.path.join(MC, r"_shared_report\label_origin_R10\confirm_R12"); LO = os.path.join(MC, r"_shared_report\label_origin_R10\cache")
KSIC50 = set(['15220', '20202', '20411', '20423', '20491', '21220', '23992', '24191', '24219', '25914', '25922', '25924',
          '25932', '26111', '26112', '26121', '26129', '26211', '26212', '26223', '26224', '26291', '26322', '26323',
          '26422', '26429', '26519', '26529', '27199', '27211', '27212', '27301', '28111', '28122', '28511', '28902',
          '29131', '29132', '29133', '29150', '29162', '29210', '29229', '29241', '29280', '30393', '31202', '31312',
          '33301', '33309'])
SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
def norm(s):
    s = SUF.sub("", str(s or "")).upper(); return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", s)
reg = pd.read_parquet(os.path.join(R12, "cache", "reg12.parquet")); reg["applicantCd"] = reg.applicantCd.astype(str)
LB = pd.read_parquet(os.path.join(R12, "cache", "labels.parquet"), columns=["appl_no", "split"]); LB = LB[LB.split == "eval"]; LB["appl_no"] = LB.appl_no.astype(str)
pf = pd.read_parquet(os.path.join(LO, "pat_firm.parquet")); pf["appl_no"] = pf.appl_no.astype(str); pf["applicantCd"] = pf.applicantCd.astype(str)
evf = set(pf.merge(LB, on="appl_no").applicantCd)
G1 = set(reg[reg.dart.isin(KSIC50) & reg.applicantCd.isin(evf)].applicantCd)
lk = pd.read_parquet(os.path.join(EXT, "firm_nps_applicant_link.parquet")); lk = lk[lk.nps_hit]; lk["applicantCd"] = lk.applicantCd.astype(str)
lk["key"] = lk.biz6 + "|" + lk.nname; A2K = dict(zip(lk.applicantCd, lk.key)); K2A = {}
for a in G1:
    if a in A2K: K2A.setdefault(A2K[a], []).append(a)
SN = ["201912", "202012", "202112", "202212", "202312", "202412", "202512", "202608"]
H = []
for s in SN:
    w = pd.read_parquet(os.path.join(EXT, f"firm_nps_wkpl_{s}.parquet"), columns=["name", "biz6", "indcd", "indnm", "n_emp", "form"])
    if s >= "202512": w = w[w.form.astype(str).str.strip() == "1"]
    w["key"] = w.biz6.fillna("") + "|" + w.name.map(norm); w = w[w.key.isin(K2A)].sort_values("n_emp", ascending=False).drop_duplicates("key")
    ic = w.indcd.fillna("").astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
    w["ic"] = ic.where(~ic.isin(["000000", "999999"])); w["nm"] = w.indnm.fillna("").str.replace(r"\s", "", regex=True).replace("", np.nan); w["snap"] = s
    H.append(w[["key", "snap", "ic", "nm"]])
H = pd.concat(H)
g = H.groupby("key")
R = {"G1": len(G1), "G1_with_nps_key": sum(len(v) for v in K2A.values()), "keys_seen": int(H.key.nunique()),
     "ic_changed_keys": int(g.ic.nunique().gt(1).sum()), "nm_changed_keys": int(g.nm.nunique().gt(1).sum()),
     "snaps_per_key_median": float(g.snap.nunique().median())}
R["ic_change_rate"] = round(R["ic_changed_keys"] / R["keys_seen"], 4); R["nm_change_rate"] = round(R["nm_changed_keys"] / R["keys_seen"], 4)
# 2019 업종명 vs 2026 업종명 다른 비율(양 끝 둘 다 있는 key)
a = H[H.snap == "201912"].set_index("key"); b = H[H.snap == "202608"].set_index("key"); com = a.index.intersection(b.index)
R["ends_both"] = len(com); R["ends_nm_diff"] = int((a.loc[com, "nm"] != b.loc[com, "nm"]).sum()); R["ends_ic_diff"] = int((a.loc[com, "ic"] != b.loc[com, "ic"]).sum())
json.dump(R, open(os.path.join(HERE, "r2_result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(R))
