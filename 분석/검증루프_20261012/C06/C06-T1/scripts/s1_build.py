# -*- coding: utf-8 -*-
r"""C06-T1 s1: 모집단 U(평가 문서 보유 법인) 기업표·문서표 구축 + 데이터 점검 원값. 읽기만(원자료·R12·label_origin_R10). 쓰기: C06-T1\cache, out\s1_*.json
DART 연결 3경로(법인번호·사업자번호·이름), NPS 스냅샷 업종 이력(2015-12~2026-08), 지역(제공 출원인 주소), 업력·특허 수."""
import os, sys, re, json, glob, time, sqlite3
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); CACHE = os.path.join(HERE, "cache"); OUT = os.path.join(HERE, "out")
os.makedirs(CACHE, exist_ok=True); os.makedirs(OUT, exist_ok=True)
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
MC = os.path.join(BASE, r"분석\모델경쟁_20260927"); EXT = os.path.join(MC, r"_shared\ext")
LO = os.path.join(MC, r"_shared_report\label_origin_R10\cache"); R12 = os.path.join(MC, r"_shared_report\label_origin_R10\confirm_R12")
KSIC_DEF = os.path.join(BASE, r"공모전데이터\외부데이터\KSIC해설\ksic50_definitions.json")
_T = time.time(); LOG = open(os.path.join(OUT, "s1_log.txt"), "w", encoding="utf-8")
def log(*a):
    s = " ".join(str(x) for x in a) + f" [{round(time.time()-_T)}s]"; print(s.encode("ascii", "replace").decode(), flush=True); LOG.write(s + "\n"); LOG.flush()
J = {}
SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
def norm(s):
    s = SUF.sub("", str(s or "")).upper(); return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", s)
def nname(s): return re.sub(r"[\s·ㆍ,\.\(\)]", "", str(s or ""))

# ---- 평가 문서 × 법인
LB = pd.read_parquet(os.path.join(R12, "cache", "labels.parquet"), columns=["appl_no", "ay", "split", "PROV", "code", "H-mg-3", "H-sc-1", "C-30-20", "D-100"])
LB["appl_no"] = LB.appl_no.astype(str); EV = LB[LB.split == "eval"].drop(columns="split")
pf = pd.read_parquet(os.path.join(LO, "pat_firm.parquet"))
X = pf.merge(EV, on="appl_no"); U = pd.DataFrame({"applicantCd": X.applicantCd.unique()})
J["U_firms"] = len(U); J["U_docs_pairs"] = len(X); log("U", J)
X.to_parquet(os.path.join(CACHE, "docs_U.parquet"), index=False)
reg = pd.read_parquet(os.path.join(R12, "cache", "reg12.parquet"))
U = U.merge(reg, on="applicantCd", how="left")

# ---- 특허 수·업력(전 기간 KR 문서)
kl = pd.read_parquet(os.path.join(LO, "kr_lo.parquet"), columns=["appl_no", "ay"]); kl["appl_no"] = kl.appl_no.astype(str)
pa = pf[pf.applicantCd.isin(set(U.applicantCd))].merge(kl, on="appl_no")
g = pa.groupby("applicantCd"); U = U.merge(pd.DataFrame({"first_ay": g.ay.min(), "n_pat_all": g.size()}).reset_index(), on="applicantCd", how="left")
U = U.merge(X.groupby("applicantCd").size().rename("n_pat_eval").reset_index(), on="applicantCd", how="left")
U["age_proxy"] = 2019 - U.first_ay
# 제공 KSIC 최빈(평가 문서; 동률은 코드 작은 것)
KSIC50 = ['15220', '20202', '20411', '20423', '20491', '21220', '23992', '24191', '24219', '25914', '25922', '25924',
          '25932', '26111', '26112', '26121', '26129', '26211', '26212', '26223', '26224', '26291', '26322', '26323',
          '26422', '26429', '26519', '26529', '27199', '27211', '27212', '27301', '28111', '28122', '28511', '28902',
          '29131', '29132', '29133', '29150', '29162', '29210', '29229', '29241', '29280', '30393', '31202', '31312',
          '33301', '33309']
t = X[X.PROV >= 0].groupby(["applicantCd", "PROV"]).size().rename("c").reset_index().sort_values(["applicantCd", "c", "PROV"], ascending=[True, False, True])
mode = t.drop_duplicates("applicantCd").set_index("applicantCd").PROV
U["prov_mode"] = U.applicantCd.map(mode).map(lambda i: KSIC50[int(i)] if pd.notna(i) else None)

# ---- 지역: 제공 KR 출원인 주소(단독 출원 문서)
import pyarrow.parquet as pq
kp = os.path.join(BASE, r"공모전데이터\작업데이터\kr_patents.parquet"); nmz = pq.read_schema(kp).names
K = pd.read_parquet(kp, columns=[nmz[0], nmz[8], nmz[9]]); K.columns = ["appl_no", "aname", "addr"]; K["appl_no"] = K.appl_no.astype(str)
J["kr_patents_appl_overlap_eval"] = float(EV.appl_no.isin(set(K.appl_no)).mean())
K = K[K.appl_no.isin(set(X.appl_no)) & ~K.aname.fillna("|").str.contains(r"\|")]
K["sido"] = K.addr.fillna("").str.strip().str.split().str[0].str[:2]
xr = X[["applicantCd", "appl_no"]].merge(K[["appl_no", "sido"]], on="appl_no")
xr = xr[xr.sido.str.len() == 2]
t = xr.groupby(["applicantCd", "sido"]).size().rename("c").reset_index().sort_values(["applicantCd", "c", "sido"], ascending=[True, False, True])
U["sido"] = U.applicantCd.map(t.drop_duplicates("applicantCd").set_index("applicantCd").sido)
J["sido_known"] = float(U.sido.notna().mean()); log("sido", J["sido_known"])
del K, xr

# ---- CORP_APPLICANT
rows = []
with open(os.path.join(BASE, r"공모전데이터\외부데이터\Corporate_20260720\CORP_APPLICANT.txt"), encoding="utf-8") as fh:
    next(fh)
    for line in fh:
        p = line.rstrip("\r\n").split("¶")
        if len(p) >= 5: rows.append((p[0].strip(), p[1].strip(), p[3].replace("-", "").strip(), re.sub(r"\D", "", p[4])))
CA = pd.DataFrame(rows, columns=["applicantCd", "aname", "jurir", "bizr"])
J["CA_rows"] = len(CA); J["CA_dup_code"] = int(CA.applicantCd.duplicated().sum())
U = U.merge(CA.drop_duplicates("applicantCd"), on="applicantCd", how="left")
U["jur2"] = U.jurir.where(U.jurir.str.len() == 13).str[:2]
J["U_has_jurir13"] = float((U.jurir.str.len() == 13).mean()); J["U_has_bizr10"] = float((U.bizr.str.len() == 10).mean())

# ---- DART 원천
c = sqlite3.connect("file:" + os.path.join(BASE, r"공모전데이터\외부데이터\DART\dart_company.sqlite") + "?mode=ro", uri=True)
D = pd.read_sql("select corp_code, corp_name, jurir_no, bizr_no, induty_code, est_dt, corp_cls, ts from company", c); c.close()
D["jurir"] = D.jurir_no.fillna("").str.replace("-", "").str.strip(); D["bizr"] = D.bizr_no.fillna("").str.replace(r"\D", "", regex=True)
D["ind"] = D.induty_code.fillna("").str.strip()
J["dart_n"] = len(D); J["dart_cls"] = D.corp_cls.value_counts().to_dict(); J["dart_ts"] = [D.ts.min(), D.ts.max()]
J["dart_ind_len"] = D.ind.str.len().value_counts().sort_index().to_dict()
J["dart_jurir13"] = int((D.jurir.str.len() == 13).sum()); J["dart_bizr10"] = int((D.bizr.str.len() == 10).sum())
dj = D[D.jurir.str.len() == 13]; dup = dj[dj.jurir.duplicated(keep=False)]
J["dart_dup_jurir_rows"] = len(dup); J["dart_dup_jurir_groups"] = int(dup.jurir.nunique())
J["dart_dup_jurir_code_conflict"] = int(dup[dup.ind != ""].groupby("jurir").ind.nunique().gt(1).sum())
db = D[D.bizr.str.len() == 10]; J["dart_dup_bizr_groups"] = int(db.bizr[db.bizr.duplicated()].nunique())
# lo_prep 규칙(ind<>'' 후 jurir 중복 첫 행) 그대로
Dd = D[D.ind != ""].copy(); Dd = Dd[Dd.jurir.str.len() == 13].drop_duplicates("jurir")
lj = CA[CA.jurir.str.len() == 13].merge(Dd[["jurir", "ind", "corp_cls", "est_dt", "corp_code"]], on="jurir").drop_duplicates("applicantCd")
lj = lj.rename(columns={"ind": "dart_j", "corp_cls": "cls_j", "est_dt": "est_j", "corp_code": "cc_j"})[["applicantCd", "dart_j", "cls_j", "est_j", "cc_j"]]
U = U.merge(lj, on="applicantCd", how="left")
J["recon_dart_eq_reg"] = float(((U.dart == U.dart_j.str.replace(r"\D", "", regex=True)) | (U.dart.isna() & U.dart_j.isna())).mean())
# 사업자번호 경로
Db = D[(D.ind != "") & (D.bizr.str.len() == 10)]
Db = Db[~Db.bizr.duplicated(keep=False)]
lb = CA[CA.bizr.str.len() == 10].merge(Db[["bizr", "ind", "corp_cls", "corp_code"]], on="bizr").drop_duplicates("applicantCd")
U = U.merge(lb.rename(columns={"ind": "dart_b", "corp_cls": "cls_b", "corp_code": "cc_b"})[["applicantCd", "dart_b", "cls_b", "cc_b"]], on="applicantCd", how="left")
# 이름 경로(정규화 상호 유일)
D["nn"] = D.corp_name.map(norm); Dn = D[(D.ind != "") & (D.nn != "")]; Dn = Dn[~Dn.nn.duplicated(keep=False)]
CA["nn"] = CA.aname.map(norm)
ln = CA[CA.nn != ""].merge(Dn[["nn", "ind", "corp_cls", "corp_code"]], on="nn").drop_duplicates("applicantCd")
U = U.merge(ln.rename(columns={"ind": "dart_n", "corp_cls": "cls_n", "corp_code": "cc_n"})[["applicantCd", "dart_n", "cls_n", "cc_n"]], on="applicantCd", how="left")
log("DART links", {k: int(U[k].notna().sum()) for k in ("dart_j", "dart_b", "dart_n")})

# ---- NPS 스냅샷 업종 이력 (lo_prep 키, 별칭 없이 정확 키)
nm2code = {}
for fn in sorted(glob.glob(os.path.join(BASE, r"공모전데이터\외부데이터\산업분류별\*.csv"))):
    dd = pd.read_csv(fn, encoding="cp949", dtype=str, usecols=[2, 3]); dd.columns = ["code", "name"]
    dd = dd[dd.code.str.match(r"^[A-Z]\d{5}$", na=False)]
    for cc_, n in zip(dd.code, dd.name): nm2code.setdefault(nname(n), cc_[1:])
for r in json.load(open(KSIC_DEF, encoding="utf-8")): nm2code.setdefault(nname(r["codenmKr"]), r["code"])
# 이름 매핑의 출처 파일(10차 1FS1001 vs 11차 1FS1101) 기록
src = {}
for fn in sorted(glob.glob(os.path.join(BASE, r"공모전데이터\외부데이터\산업분류별\*.csv"))):
    dd = pd.read_csv(fn, encoding="cp949", dtype=str, usecols=[2, 3]); dd.columns = ["code", "name"]
    dd = dd[dd.code.str.match(r"^[A-Z]\d{5}$", na=False)]
    tag = "10" if "1FS1001" in os.path.basename(fn) else ("11" if "1FS1101" in os.path.basename(fn) else os.path.basename(fn)[:12])
    for n in dd.name: src.setdefault(nname(n), set()).add(tag)
lk = pd.read_parquet(os.path.join(EXT, "firm_nps_applicant_link.parquet")); lk = lk[lk.nps_hit]
lk["key"] = lk.biz6 + "|" + lk.nname; A2K = dict(zip(lk.applicantCd.astype(str), lk.key))
Uk = U[["applicantCd"]].assign(key=U.applicantCd.map(A2K)).dropna(); Kset = set(Uk.key)
ALL = ["201512", "201612", "201712", "201812", "201912", "202012", "202112", "202212", "202312", "202412", "202512", "202608"]
hist = []
for s in ALL:
    w = pd.read_parquet(os.path.join(EXT, f"firm_nps_wkpl_{s}.parquet"), columns=["name", "biz6", "indcd", "indnm", "n_emp", "ldong", "form"])
    if s >= "202512": w = w[w.form.astype(str).str.strip() == "1"]
    w["key"] = w.biz6.fillna("") + "|" + w.name.map(norm); w = w[w.key.isin(Kset)].sort_values("n_emp", ascending=False).drop_duplicates("key")
    ic = w.indcd.fillna("").astype(str).str.strip().str.replace(r"\D", "", regex=True)
    ic = ic.where(ic.str.len() == 0, ic.str.zfill(6)); w["indcd_c"] = ic.where(~ic.isin(["", "000000", "999999"]))
    w["k5"] = w.indnm.map(lambda x: nm2code.get(nname(x)) if isinstance(x, str) and x.strip() else None)
    w["k5src"] = w.indnm.map(lambda x: "/".join(sorted(src.get(nname(x), []))) if isinstance(x, str) else "")
    w["snap"] = s; hist.append(w[["key", "snap", "indcd_c", "k5", "k5src", "n_emp", "ldong"]])
H = pd.concat(hist, ignore_index=True).merge(Uk, on="key")
H.to_parquet(os.path.join(CACHE, "nps_hist_U.parquet"), index=False)
J["nps_hist_rows"] = len(H); J["nps_hist_firms"] = int(H.applicantCd.nunique())
log("NPS hist", len(H))
U.to_parquet(os.path.join(CACHE, "firm_U.parquet"), index=False)
json.dump(J, open(os.path.join(OUT, "s1_meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
log("done", len(U))
