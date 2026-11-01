# -*- coding: utf-8 -*-
r"""R12 confirm — 등록업종 표(PREREG.txt). NPS 가입자 수는 lo_prep.py NPS 부분 복사로 재계산(nps 코드가 firm_reg와 같아야 함),
KIND는 fp_tests.py 규칙 복사. 읽기만. 쓰기: cache/reg12.parquet(applicantCd, nps, dart, kind3, n_emp), res_reg.json"""
import os, sys, re, json, time, glob
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); CACHE = os.path.join(HERE, "cache")
MC = os.path.abspath(os.path.join(HERE, r"..\..\..")); BASE = os.path.abspath(os.path.join(MC, r"..\.."))
LO = os.path.join(MC, r"_shared_report\label_origin_R10\cache"); EXT = os.path.join(MC, r"_shared\ext")
KSIC_DEF = os.path.join(BASE, r"공모전데이터\외부데이터\KSIC해설\ksic50_definitions.json")
_T = time.time()
def log(*a): print(*a, f"[{round(time.time()-_T)}s]", flush=True)
J = {}
reg = pd.read_parquet(os.path.join(LO, "firm_reg.parquet"))
# ---- NPS (lo_prep.py 그대로, n_emp 유지)
SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
def norm(s):
    s = SUF.sub("", str(s or "")).upper()
    return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", s)
def nname(s): return re.sub(r"[\s·ㆍ,\.\(\)]", "", str(s or ""))
nm2code = {}
for fn in sorted(glob.glob(os.path.join(BASE, r"공모전데이터\외부데이터\산업분류별\*.csv"))):
    dd = pd.read_csv(fn, encoding="cp949", dtype=str, usecols=[2, 3]); dd.columns = ["code", "name"]
    dd = dd[dd.code.str.match(r"^[A-Z]\d{5}$", na=False)]
    for c, n in zip(dd.code, dd.name): nm2code.setdefault(nname(n), c[1:])
for r in json.load(open(KSIC_DEF, encoding="utf-8")): nm2code.setdefault(nname(r["codenmKr"]), r["code"])
SNAP = "201912"
lk = pd.read_parquet(os.path.join(EXT, "firm_nps_applicant_link.parquet")); lk = lk[lk.nps_hit]
lk["key"] = lk.biz6 + "|" + lk.nname
A2K = dict(zip(lk.applicantCd, lk.key)); Kset = set(A2K.values())
ALL = ["201512", "201612", "201712", "201812", "201912", "202012", "202112", "202212", "202312", "202412", "202512", "202608"]
ident = []
for s in ALL:
    w = pd.read_parquet(os.path.join(EXT, f"firm_nps_wkpl_{s}.parquet"), columns=["name", "biz6", "adpt_dt"])
    w["key"] = w.biz6.fillna("") + "|" + w.name.map(norm); w["ba"] = w.biz6.fillna("") + "|" + w.adpt_dt.fillna("").astype(str)
    ident.append(w.loc[w.key.isin(Kset) & w.adpt_dt.notna(), ["ba", "key"]].drop_duplicates())
ident = pd.concat(ident).drop_duplicates(); nK = ident.groupby("ba").key.nunique(); ident = ident[ident.ba.map(nK) == 1].set_index("ba").key
W = pd.read_parquet(os.path.join(EXT, f"firm_nps_wkpl_{SNAP}.parquet"), columns=["name", "biz6", "adpt_dt", "indcd", "indnm", "n_emp", "stat"])
W["key"] = W.biz6.fillna("") + "|" + W.name.map(norm); W["ba"] = W.biz6.fillna("") + "|" + W.adpt_dt.fillna("").astype(str)
have = set(W.key)
al = W[W.ba.isin(ident.index)].copy(); al["K"] = al.ba.map(ident); al = al[(al.key != al.K) & ~al.K.isin(have)]; al["key"] = al.K
W2 = pd.concat([W, al.drop(columns="K")], ignore_index=True).sort_values("n_emp", ascending=False).drop_duplicates("key")
W2["nps"] = W2.indnm.map(lambda s: nm2code.get(nname(s)))
nps = pd.DataFrame({"applicantCd": list(A2K.keys()), "key": list(A2K.values())}).merge(W2[["key", "nps", "n_emp"]], on="key")[["applicantCd", "nps", "n_emp"]]
nps["nps"] = nps.nps.astype(str).str.replace(r"\D", "", regex=True).where(nps.nps.notna()); nps["applicantCd"] = nps.applicantCd.astype(str)
chk = reg[["applicantCd", "nps"]].merge(nps, on="applicantCd", how="outer", suffixes=("", "_re"))
J["nps_recompute_match"] = float(((chk.nps == chk.nps_re) | (chk.nps.isna() & chk.nps_re.isna())).mean())
J["nps_rows"] = [int(reg.nps.notna().sum()), int(nps.nps.notna().sum())]
log("NPS", J)
reg = reg.merge(nps[["applicantCd", "n_emp"]].drop_duplicates("applicantCd"), on="applicantCd", how="left")
if J["nps_recompute_match"] < 1.0: reg["n_emp"] = np.nan
# ---- KIND (fp_tests.py 그대로)
nn = lambda s: re.sub(r"[\s,·\.;]", "", str(s)).replace("및", "")
L = pd.read_csv(os.path.join(MC, r"_shared\ext\r7_data_ksic_ksco_revision_link_raw.csv"), encoding="cp949", dtype=str)
L = L[L["분류종류"] == "KSIC"]
names = pd.concat([L[["신분류코드", "신분류항목명"]].set_axis(["code", "nm"], axis=1), L[["구분류코드", "구분류항목명"]].set_axis(["code", "nm"], axis=1)])
names = names[names.code.str.fullmatch(r"\d{2,4}", na=False)]; names["k"] = names.nm.map(nn)
n2c = names.groupby("k").code.agg(lambda s: s.value_counts().index[0])
K = pd.read_parquet(os.path.join(MC, r"_ideas_R7\cache\kind_linked.parquet"))
K["ind"] = K["업종"].map(nn).map(n2c); K["jurir"] = K.jurir_no.astype(str).str.replace("-", "").str.strip()
rows = []
with open(os.path.join(BASE, r"공모전데이터\외부데이터\Corporate_20260720\CORP_APPLICANT.txt"), encoding="utf-8") as fh:
    next(fh)
    for line in fh:
        p = line.rstrip("\r\n").split("¶")
        if len(p) >= 5: rows.append((p[0].strip(), p[3].replace("-", "").strip()))
CA = pd.DataFrame(rows, columns=["applicantCd", "jurir"]); CA = CA[CA.jurir.str.len() == 13]
KK = K[(K.jurir.str.len() == 13) & K.ind.notna()].merge(CA, on="jurir")
KK["pre"] = KK.listing_date <= "2019-12-31"
KK = pd.concat([KK[KK.pre].sort_values("listing_date", ascending=False), KK[~KK.pre].sort_values("listing_date")]).drop_duplicates("applicantCd")
kind = KK[KK.ind.str.len() >= 3][["applicantCd", "ind"]].rename(columns={"ind": "kind3"}); kind["kind3"] = kind.kind3.str[:3]
reg = reg.merge(kind, on="applicantCd", how="outer")
J["kind_applicants"] = int(reg.kind3.notna().sum()); J["n_emp_known"] = int(reg.n_emp.notna().sum())
reg.to_parquet(os.path.join(CACHE, "reg12.parquet"), index=False)
json.dump(J, open(os.path.join(HERE, "res_reg.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
log("done", J)
