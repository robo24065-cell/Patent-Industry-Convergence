# -*- coding: utf-8 -*-
"""국민연금 가입사업장(연말 스냅샷 2015-11 ~ 2025-12, 2026-08) ↔ 특허 출원인 연결 + 50 KSIC 산업 패널.
연결키: 사업자등록번호 앞 6자리(NPS 공개분) + 정규화 상호. 특허 쪽 사업자번호는 CORP_APPLICANT(출원인코드↔사업자번호)
        와 사업자 권리목록(business_no)에서 가져온다. 제공 데이터 내용은 외부로 보내지 않는다(모두 로컬 결합).
업종: NPS 업종코드는 국세청 6자리 업종코드 → 50 KSIC와 '업종명 정규화 완전일치'로 연결(50/50 확인).
출력:
  firm_nps_applicant_link.parquet  출원인코드 ↔ (biz6, 정규화상호) 매칭 결과
  firm_nps_firm_panel.parquet      출원인코드 × 스냅샷: 가입자수 합, 사업장수, 신규취득·상실, 최초적용일, 탈퇴여부
  firm_nps_ksic50_panel.parquet    KSIC50 × 스냅샷: 사업장수·가입자수·당해 신규적용(탄생)·탈퇴 (NPS 전체 사업장)
  firm_nps_link_rates.csv          연계율
"""
import os, re, glob, json, sqlite3
import pandas as pd, numpy as np

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
EXT = os.path.join(BASE, r"분석\모델경쟁_20260927\_shared\ext")
SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)


def norm(s):
    s = SUF.sub("", str(s or "")).upper()
    return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", s)


# ---------- 특허 출원인 → 사업자번호 ----------
rows = []
with open(os.path.join(BASE, r"공모전데이터\외부데이터\Corporate_20260720\CORP_APPLICANT.txt"), encoding="utf-8") as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("¶")
        if len(p) >= 5:
            rows.append((p[0].strip(), p[1].strip(), p[3].strip(), p[4].strip()))
ca = pd.DataFrame(rows, columns=["applicantCd", "aname", "corpno", "bizno"])
ca["bizno"] = ca.bizno.str.replace(r"\D", "", regex=True)
bc = sqlite3.connect(f"file:{os.path.join(BASE, r'수집파이프라인/business_rights.sqlite')}?mode=ro", uri=True)
rl = pd.read_sql("select distinct r.applicant_cd applicantCd, r.business_no, q.applicant_name from right_list r left join applicant_query q using(applicant_cd) where r.business_no is not null", bc)
rl["bizno2"] = rl.business_no.str.replace(r"\D", "", regex=True)
rl = rl[rl.bizno2.str.len() == 10].drop_duplicates("applicantCd")
ca = ca.merge(rl[["applicantCd", "bizno2", "applicant_name"]], on="applicantCd", how="outer")
ca["bizno"] = ca.bizno.where(ca.bizno.fillna("").str.len() == 10, ca.bizno2)
ca["aname"] = ca.aname.fillna(ca.applicant_name)
ca = ca[ca.bizno.fillna("").str.len() == 10].copy()
ca["biz6"] = ca.bizno.str[:6]
ca["nname"] = ca.aname.map(norm)
ca = ca[["applicantCd", "aname", "bizno", "biz6", "nname"]].drop_duplicates("applicantCd")
print("사업자번호 있는 출원인코드", len(ca))

# ---------- NPS 스냅샷 ----------
snaps = sorted(glob.glob(os.path.join(EXT, "firm_nps_wkpl_*.parquet")))
N = []
for fp in snaps:
    d = pd.read_parquet(fp, columns=["ym", "name", "biz6", "stat", "form", "indcd", "indnm", "adpt_dt", "rrg_dt", "n_emp", "n_new", "n_loss", "snap"])
    N.append(d)
N = pd.concat(N, ignore_index=True)
N["nname"] = N.name.map(norm)
print("NPS 행", len(N))

# ---------- 매칭 ----------
keys = N[["biz6", "nname"]].drop_duplicates()
keys["nps_hit"] = True
m = ca.merge(keys, on=["biz6", "nname"], how="left")
m["nps_hit"] = m.nps_hit.fillna(False).astype(bool)
m.to_parquet(os.path.join(EXT, "firm_nps_applicant_link.parquet"), index=False)

# 출원인 × 스냅샷 패널
hitN = N.merge(m.loc[m.nps_hit, ["applicantCd", "biz6", "nname"]], on=["biz6", "nname"])
hitN["adpt_dt"] = hitN.adpt_dt.fillna("9999-12-31")
hitN["rrg_dt"] = hitN.rrg_dt.fillna("")
agg = hitN.groupby(["applicantCd", "snap"]).agg(
    n_emp=("n_emp", "sum"), n_wkpl=("name", "size"), n_new=("n_new", "sum"), n_loss=("n_loss", "sum"),
    active=("stat", lambda s: (s == "1").any()), first_adpt=("adpt_dt", "min"), last_rrg=("rrg_dt", "max"),
    indcd=("indcd", lambda s: s.mode().iat[0] if len(s) else None)).reset_index()
agg.to_parquet(os.path.join(EXT, "firm_nps_firm_panel.parquet"), index=False)

# ---------- 50 KSIC 산업 패널 (업종명 정규화 일치) ----------
d50 = json.load(open(os.path.join(BASE, r"공모전데이터\외부데이터\KSIC해설\ksic50_definitions.json"), encoding="utf-8"))
nn = lambda s: re.sub(r"[\s·,및]", "", str(s))
kmap = {nn(x["codenmKr"]): x["code"] for x in d50}
N["ksic"] = N.indnm.map(lambda s: kmap.get(nn(s)))
cmap = N.loc[N.ksic.notna(), ["indcd", "indnm", "ksic"]].drop_duplicates()
cmap.to_csv(os.path.join(EXT, "firm_nps_indcd_ksic50_map.csv"), index=False, encoding="utf-8-sig")
K = N[N.ksic.notna()].copy()
K["snap_year"] = K.snap.str[:4].astype(int)
K["adpt_year"] = pd.to_numeric(K.adpt_dt.str[:4], errors="coerce")
kp = K.groupby(["ksic", "snap"]).agg(n_wkpl=("name", "size"), n_active=("stat", lambda s: (s == "1").sum()),
                                      n_emp=("n_emp", "sum"), n_new_hire=("n_new", "sum"), n_loss=("n_loss", "sum"),
                                      n_corp=("form", lambda s: (s == "1").sum())).reset_index()
born = K[K.adpt_year == K.snap_year].groupby(["ksic", "snap"]).size().rename("n_born_in_year")
kp = kp.merge(born, on=["ksic", "snap"], how="left").fillna({"n_born_in_year": 0})
kp.to_parquet(os.path.join(EXT, "firm_nps_ksic50_panel.parquet"), index=False)

# ---------- 연계율 ----------
pf = pd.read_parquet(os.path.join(BASE, r"분석\모델경쟁_20260927\_ideas_R2\firm\cache\pat_firm.parquet"), columns=["rgst_no", "applicantCd", "ksic", "applDate"])
pf["ay"] = pf.applDate.astype(str).str[:4].astype(int)
pf["has_biz"] = pf.applicantCd.isin(set(ca.applicantCd))
pf["nps"] = pf.applicantCd.isin(set(m.loc[m.nps_hit, "applicantCd"]))
dom_corp = pf.applicantCd.astype(str).str[0].isin(["1", "2"])  # 국내 법인 출원인코드(1·2로 시작)
out = []
def add(k, v, r=""):
    out.append((k, v, r)); print(k, v, r)
tot = pf.rgst_no.nunique()
add("KR 등록특허(제공범위)", tot)
add("출원인코드 1·2 시작(국내법인 추정) 특허", pf.loc[dom_corp, "rgst_no"].nunique(), f"{pf.loc[dom_corp,'rgst_no'].nunique()/tot:.3f}")
add("사업자번호 확보 출원인의 특허", pf.loc[pf.has_biz, "rgst_no"].nunique(), f"{pf.loc[pf.has_biz,'rgst_no'].nunique()/tot:.3f}")
add("NPS 사업장 매칭 출원인의 특허", pf.loc[pf.nps, "rgst_no"].nunique(), f"{pf.loc[pf.nps,'rgst_no'].nunique()/tot:.3f}")
add("  └ 사업자번호 확보 특허 대비", "", f"{pf.loc[pf.nps,'rgst_no'].nunique()/pf.loc[pf.has_biz,'rgst_no'].nunique():.3f}")
add("사업자번호 확보 출원인 수", int(pf.loc[pf.has_biz, "applicantCd"].nunique()))
add("NPS 매칭 출원인 수(제공범위 특허 보유)", int(pf.loc[pf.nps, "applicantCd"].nunique()), f"{pf.loc[pf.nps,'applicantCd'].nunique()/pf.loc[pf.has_biz,'applicantCd'].nunique():.3f}")
for lo, hi in ((2005, 2012), (2013, 2016), (2017, 2022)):
    r = pf[pf.ay.between(lo, hi)]
    add(f"출원 {lo}-{hi} 특허 중 NPS 매칭", r.loc[r.nps, "rgst_no"].nunique(), f"{r.loc[r.nps,'rgst_no'].nunique()/r.rgst_no.nunique():.3f}")
k = pf.groupby("ksic").apply(lambda d: d.loc[d.nps, "rgst_no"].nunique() / d.rgst_no.nunique())
add("50 KSIC별 NPS 매칭 특허비중 최소/중앙/최대", f"{k.min():.3f}/{k.median():.3f}/{k.max():.3f}", f"{k.shape[0]}개 KSIC")
dfin = pd.read_parquet(os.path.join(EXT, "firm_dart_fin_panel.parquet"), columns=["corp_code"])
add("NPS 50 KSIC 업종명 일치", len(set(cmap.ksic)), "50개 중")
add("NPS 50 KSIC 사업장(2016-11 스냅샷)", int(kp.loc[kp.snap == "2016-12", "n_wkpl"].sum()))
pd.DataFrame(out, columns=["항목", "값", "비율"]).to_csv(os.path.join(EXT, "firm_nps_link_rates.csv"), index=False, encoding="utf-8-sig")
k.rename("nps_share").to_csv(os.path.join(EXT, "firm_nps_link_by_ksic.csv"), encoding="utf-8-sig")
