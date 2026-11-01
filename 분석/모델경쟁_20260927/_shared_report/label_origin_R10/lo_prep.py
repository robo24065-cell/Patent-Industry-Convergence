# -*- coding: utf-8 -*-
r"""R10 label_origin — 준비. 읽기만(동결 캐시·원본), 쓰기는 이 폴더 cache/ 만.
출력: cache/kr_lo.parquet (KR 특허 1행: appl_no, app, ay, ksic, sg, mg, sc, k_free, fam_k, fam_n, fam_agree_any)
      cache/firm_reg.parquet (applicantCd, nps5, dart5), cache/pat_firm.parquet (applicantCd, appl_no)"""
import os, sys, re, json, time
sys.dont_write_bytecode = True
import numpy as np, pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
MC = os.path.join(BASE, r"분석\모델경쟁_20260927")
OUT = os.path.join(MC, r"_shared_report\label_origin_R10"); CACHE = os.path.join(OUT, "cache"); os.makedirs(CACHE, exist_ok=True)
T9 = os.path.join(MC, r"T1_산업대기실\TXT_R9\cache"); R8 = os.path.join(MC, r"_ideas_R8\cache"); EXT = os.path.join(MC, r"_shared\ext")
KSIC_DEF = os.path.join(BASE, r"공모전데이터\외부데이터\KSIC해설\ksic50_definitions.json")
_T = time.time()
def log(*a): print(*a, f"[{round(time.time()-_T)}s]", flush=True)
SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
def norm(s):
    s = SUF.sub("", str(s or "")).upper()
    return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", s)
def nname(s): return re.sub(r"[\s·ㆍ,\.\(\)]", "", str(s or ""))

# ---- KR 특허
kr = pd.read_parquet(os.path.join(R8, "kr.parquet"), columns=["appl_no", "appl_year", "ksic", "applicant"])
import pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa
RT = pq.read_table(os.path.join(R8, "rec.parquet"), columns=["cc", "appl_no", "ksic", "sc", "mg", "sg"])
rk = RT.filter(pc.equal(RT["cc"], "KR")).select(["appl_no", "sc", "mg", "sg"]).to_pandas()
kr = kr.merge(rk, on="appl_no", how="left")
kr = kr.merge(pd.read_parquet(os.path.join(T9, "kr_txtfree.parquet")), on="appl_no", how="left")
kr["app"] = kr.applicant.map(lambda s: norm(str(s).split("|")[0]))
log("KR", len(kr), "sg 결측", kr.sg.isna().mean().round(4), "k_free 결측", kr.k_free.isna().mean().round(4))

# ---- 해외 패밀리 라벨
fl = pd.read_parquet(os.path.join(R8, "fam_link.parquet"))
fl["appl_no"] = fl.appl_no.astype(str)
RT2 = RT.filter(pc.not_equal(RT["cc"], "KR")).select(["cc", "appl_no", "ksic"])
RT2 = RT2.filter(pc.is_in(pc.cast(RT2["appl_no"], pa.string()), value_set=pa.array(fl.appl_no.unique())))
fr = RT2.to_pandas().rename(columns={"ksic": "fk"}); fr["appl_no"] = fr.appl_no.astype(str)
log("rec 해외 후보", len(fr), fr.cc.value_counts().to_dict())
j = fl.merge(fr, on=["cc", "appl_no"], how="left")
log("fam_link", len(fl), "해외 라벨 붙음", j.fk.notna().mean().round(4))
j = j.dropna(subset=["fk"])
pri = {"US": 0, "EP": 1, "JP": 2, "CN": 3}; j["pr"] = j.cc.map(pri)
cnt = j.groupby(["kr_appl_no", "fk"]).agg(c=("fk", "size"), p=("pr", "min")).reset_index()
cnt = cnt.sort_values(["kr_appl_no", "c", "p"], ascending=[True, False, True]).drop_duplicates("kr_appl_no")
nf = j.groupby("kr_appl_no").size().rename("fam_n")
kr["kr_s"] = kr.appl_no.astype(str)
kr = kr.merge(cnt[["kr_appl_no", "fk"]].rename(columns={"kr_appl_no": "kr_s", "fk": "fam_k"}), on="kr_s", how="left")
kr = kr.merge(nf.rename_axis("kr_s").reset_index(), on="kr_s", how="left")
anyag = j.merge(kr[["kr_s", "ksic"]], left_on="kr_appl_no", right_on="kr_s").assign(a=lambda d: d.fk == d.ksic).groupby("kr_s").a.max()
kr = kr.merge(anyag.rename("fam_agree_any").reset_index(), on="kr_s", how="left")
# 국가별 일치율(문서 단위)
jj = j.merge(kr[["kr_s", "ksic"]], left_on="kr_appl_no", right_on="kr_s")
p0 = {"fam_docs_linked": int(len(jj)), "kr_with_family": int(kr.fam_k.notna().sum()),
      "doc_agree_by_cc": jj.assign(a=jj.fk == jj.ksic).groupby("cc").a.mean().round(4).to_dict(),
      "doc_n_by_cc": jj.cc.value_counts().to_dict(),
      "kr_vs_majority_agree": float((kr.fam_k == kr.ksic)[kr.fam_k.notna()].mean()),
      "kr_vs_any_agree": float(kr.fam_agree_any[kr.fam_k.notna()].mean())}
log(p0)
kr.drop(columns=["kr_s", "applicant"]).rename(columns={"appl_year": "ay"}).to_parquet(os.path.join(CACHE, "kr_lo.parquet"), index=False)
del RT, RT2, fr, j, jj

# ---- 법인 출원인 코드 ↔ 특허 (s1_firm 규칙: registry, 코드 1로 시작). 원점 제한 없음(전 기간)
af = pd.read_parquet(os.path.join(MC, r"N4_특허행태_기업폐업조기경보\cache\app_firm.parquet"), columns=["applicantCd", "appl_no", "src"])
af = af[(af.src == "registry") & af.applicantCd.astype(str).str.startswith("1")][["applicantCd", "appl_no"]].drop_duplicates()
af["applicantCd"] = af.applicantCd.astype(str); af["appl_no"] = af.appl_no.astype(str)
af.to_parquet(os.path.join(CACHE, "pat_firm.parquet"), index=False)
log("법인 쌍", len(af))

# ---- 등록업종 (s1_firm.py 2019 규칙 복사)
import glob
nm2code = {}
for f in sorted(glob.glob(os.path.join(BASE, r"공모전데이터\외부데이터\산업분류별\*.csv"))):
    d = pd.read_csv(f, encoding="cp949", dtype=str, usecols=[2, 3]); d.columns = ["code", "name"]
    d = d[d.code.str.match(r"^[A-Z]\d{5}$", na=False)]
    for c, n in zip(d.code, d.name): nm2code.setdefault(nname(n), c[1:])
for r in json.load(open(KSIC_DEF, encoding="utf-8")): nm2code.setdefault(nname(r["codenmKr"]), r["code"])
import sqlite3
rows = []
with open(os.path.join(BASE, r"공모전데이터\외부데이터\Corporate_20260720\CORP_APPLICANT.txt"), encoding="utf-8") as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("¶")
        if len(p) >= 5: rows.append((p[0].strip(), p[3].replace("-", "").strip()))
CA = pd.DataFrame(rows, columns=["applicantCd", "jurir"])
c = sqlite3.connect("file:" + os.path.join(BASE, r"공모전데이터\외부데이터\DART\dart_company.sqlite") + "?mode=ro", uri=True)
Dd = pd.read_sql("select jurir_no, induty_code from company where induty_code<>''", c); c.close()
Dd["jurir"] = Dd.jurir_no.str.replace("-", "").str.strip(); Dd = Dd[Dd.jurir.str.len() == 13].drop_duplicates("jurir")
dart = CA[CA.jurir.str.len() == 13].merge(Dd[["jurir", "induty_code"]], on="jurir")[["applicantCd", "induty_code"]].drop_duplicates("applicantCd")
dart.columns = ["applicantCd", "dart"]
SNAP = "201912"
lk = pd.read_parquet(os.path.join(EXT, "firm_nps_applicant_link.parquet")); lk = lk[lk.nps_hit]
lk["key"] = lk.biz6 + "|" + lk.nname
A2K = dict(zip(lk.applicantCd, lk.key)); Kset = set(A2K.values())
ALL = ["201512", "201612", "201712", "201812", "201912", "202012", "202112", "202212", "202312", "202412", "202512", "202608"]
ident = []
for s in ALL:
    d = pd.read_parquet(os.path.join(EXT, f"firm_nps_wkpl_{s}.parquet"), columns=["name", "biz6", "adpt_dt"])
    d["key"] = d.biz6.fillna("") + "|" + d.name.map(norm); d["ba"] = d.biz6.fillna("") + "|" + d.adpt_dt.fillna("").astype(str)
    ident.append(d.loc[d.key.isin(Kset) & d.adpt_dt.notna(), ["ba", "key"]].drop_duplicates())
ident = pd.concat(ident).drop_duplicates(); nK = ident.groupby("ba").key.nunique(); ident = ident[ident.ba.map(nK) == 1].set_index("ba").key
W = pd.read_parquet(os.path.join(EXT, f"firm_nps_wkpl_{SNAP}.parquet"), columns=["name", "biz6", "adpt_dt", "indcd", "indnm", "n_emp", "stat"])
W["key"] = W.biz6.fillna("") + "|" + W.name.map(norm); W["ba"] = W.biz6.fillna("") + "|" + W.adpt_dt.fillna("").astype(str)
have = set(W.key)
al = W[W.ba.isin(ident.index)].copy(); al["K"] = al.ba.map(ident); al = al[(al.key != al.K) & ~al.K.isin(have)]; al["key"] = al.K
W2 = pd.concat([W, al.drop(columns="K")], ignore_index=True).sort_values("n_emp", ascending=False).drop_duplicates("key")
W2["nps"] = W2.indnm.map(lambda s: nm2code.get(nname(s)))
nps = pd.DataFrame({"applicantCd": list(A2K.keys()), "key": list(A2K.values())}).merge(W2[["key", "nps"]], on="key")[["applicantCd", "nps"]]
reg = nps.merge(dart, on="applicantCd", how="outer")
for col in ("nps", "dart"):
    reg[col] = reg[col].astype(str).str.replace(r"\D", "", regex=True).where(reg[col].notna())
reg["applicantCd"] = reg.applicantCd.astype(str)
reg.to_parquet(os.path.join(CACHE, "firm_reg.parquet"), index=False)
p0["n_firm_nps"] = int(reg.nps.notna().sum()); p0["n_firm_dart"] = int(reg.dart.notna().sum())
json.dump(p0, open(os.path.join(OUT, "res_p0.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
log("done", p0["n_firm_nps"], p0["n_firm_dart"])
