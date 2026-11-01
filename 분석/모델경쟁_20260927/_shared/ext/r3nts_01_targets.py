# 국세청 상태조회 대상 사업자번호 목록 작성 (읽기 전용 소스)
import sqlite3, pandas as pd, re
from pathlib import Path
def ro(p): return sqlite3.connect(Path(p).as_uri()+"?mode=ro", uri=True)
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
EXT = BASE + r"\분석\모델경쟁_20260927\_shared\ext"
# 1) CORP_APPLICANT
rows = []
with open(BASE + r"\공모전데이터\외부데이터\Corporate_20260720\CORP_APPLICANT.txt", encoding="utf-8", errors="replace") as f:
    next(f)
    for l in f:
        p = l.rstrip("\r\n").split("¶")
        if len(p) < 5: continue
        rows.append((p[0].strip(), p[1].strip(), p[3].strip(), re.sub(r"\D", "", p[4])))
ca = pd.DataFrame(rows, columns=["applicantCd", "aname", "corp_no", "bizno"])
ca["src"] = "CORP_APPLICANT"
# 2) right_list business_no (특허 보유 법인 포트폴리오)
c = ro(BASE + "/수집파이프라인/business_rights.sqlite")
rl = pd.read_sql("select applicant_cd as applicantCd, business_no, corp_rgstno, "
                 "sum(substr(rgst_no,1,2)='10') as n_pat_all from right_list group by 1,2,3", c)
aq = pd.read_sql("select applicant_cd as applicantCd, applicant_name as aname, n_in_fusion from applicant_query", c)
rl["bizno"] = rl.business_no.fillna("").str.replace(r"\D", "", regex=True)
rl = rl.merge(aq, on="applicantCd", how="left")
# 3) 제공범위 KR 등록특허 출원인코드
r = ro(BASE + "/수집파이프라인/registry_full.sqlite")
reg = pd.read_sql("select applicantCd, count(distinct rgst_no) n_prov_rgst from applicant group by 1", r)
link = pd.concat([ca[["applicantCd", "aname", "bizno"]].assign(src="CA"),
                  rl[["applicantCd", "aname", "bizno"]].assign(src="RL")])
link = link[link.bizno.str.len() == 10].drop_duplicates(["applicantCd", "bizno"])
link = link.merge(reg, on="applicantCd", how="left")
link["n_prov_rgst"] = link.n_prov_rgst.fillna(0).astype(int)
link.to_parquet(EXT + r"\r3nts_work\applicant_bizno_link.parquet", index=False)
t = link.groupby("bizno").agg(n_appcd=("applicantCd", "nunique"), n_prov_rgst=("n_prov_rgst", "sum"),
                              in_rl=("src", lambda s: (s == "RL").any())).reset_index()
t["tier"] = 3
t.loc[t.in_rl, "tier"] = 2
t.loc[t.n_prov_rgst > 0, "tier"] = 1
t = t.sort_values(["tier", "n_prov_rgst"], ascending=[True, False])
t.to_parquet(EXT + r"\r3nts_work\targets.parquet", index=False)
print("CORP_APPLICANT 행", len(ca), "사업자번호10자리", (ca.bizno.str.len() == 10).sum())
print("right_list 출원인", rl.applicantCd.nunique(), "bizno보유", (rl.bizno.str.len() == 10).sum())
print("출원인코드-bizno 쌍", len(link), "출원인코드", link.applicantCd.nunique())
print("대상 사업자번호", len(t)); print(t.tier.value_counts().sort_index())
print("제공범위 등록특허 보유 출원인코드(bizno 있음)", link.loc[link.n_prov_rgst > 0, "applicantCd"].nunique(),
      "/ 제공범위 출원인코드 전체", len(reg))
