# -*- coding: utf-8 -*-
"""제공 KR 출원(등록·미등록) → 출원인코드 → NPS 기업키(biz6+정규화상호).
- 등록특허: 등록원부 applicant 테이블의 출원인코드(정확).
- 미등록 포함 전체 출원: kr_patents 출원인명 → CORP_APPLICANT 정규화 상호(유일한 것만) → 출원인코드.
- 출원인코드 → 기업키: _shared/ext/firm_nps_applicant_link.parquet (nps_hit).
출력 cache/app_firm.parquet: key, applicantCd, appl_no, appl_date, known_date(공개·공고·등록 중 최초일), ksic
모두 로컬 결합. 외부 호출 없음."""
import os, sqlite3, time
import pandas as pd, numpy as np
from i00_common import *

t = time.time()
# CORP_APPLICANT 이름 → 코드(유일)
rows = []
with open(os.path.join(BASE, r"공모전데이터\외부데이터\Corporate_20260720\CORP_APPLICANT.txt"), encoding="utf-8") as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("¶")
        if len(p) >= 2:
            rows.append((p[0].strip(), p[1].strip()))
ca = pd.DataFrame(rows, columns=["applicantCd", "aname"])
ca["n"] = ca.aname.map(norm)
ca = ca[ca.n.str.len() >= 2]
amb = ca.groupby("n").applicantCd.nunique()
ca = ca[ca.n.map(amb) == 1].drop_duplicates("n")
n2c = dict(zip(ca.n, ca.applicantCd))
print("CORP_APPLICANT 유일 상호", len(n2c), round(time.time() - t), flush=True)

k = pd.read_parquet(os.path.join(BASE, r"공모전데이터\작업데이터\kr_patents.parquet"),
                    columns=["출원번호", "출원일", "공개일자", "등록일자", "출원인명", "세세분류코드"])
k.columns = ["appl_no", "appl_date", "pub_date", "rgst_date", "names", "ksic"]
k["ksic"] = k.ksic.astype(str)
k["nm"] = k.names.fillna("").str.split("|")
k = k.explode("nm")
k["applicantCd"] = k.nm.map(norm).map(n2c)
kn = k.dropna(subset=["applicantCd"])[["appl_no", "appl_date", "pub_date", "rgst_date", "ksic", "applicantCd"]].copy()
kn["src"] = "name"
print("이름 매칭 출원-출원인 쌍", len(kn), "출원", kn.appl_no.nunique(), round(time.time() - t), flush=True)

# 등록원부 정확 코드
c = sqlite3.connect(f"file:{os.path.join(BASE, '수집파이프라인', 'registry_full.sqlite')}?mode=ro", uri=True)
rp = pd.read_sql("select p.applNo appl_no, p.applDate appl_date, p.opnDate opn, p.pubDate pub, p.rgstDate rgst_date, p.ksic, a.applicantCd "
                 "from patent p join applicant a on p.rgst_no=a.rgst_no", c)
for _c in ["opn", "pub"]:
    rp[_c] = pd.to_numeric(rp[_c].astype(str).str.replace(r"\D", "", regex=True).str[:8], errors="coerce")
rp["pub_date"] = rp[["opn", "pub"]].min(axis=1)
rp = rp[["appl_no", "appl_date", "pub_date", "rgst_date", "ksic", "applicantCd"]]
rp["src"] = "registry"
print("등록원부 쌍", len(rp), round(time.time() - t), flush=True)

allp = pd.concat([rp, kn], ignore_index=True)
for col in ["appl_date", "pub_date", "rgst_date"]:
    allp[col] = pd.to_numeric(allp[col].astype(str).str.replace(r"\D", "", regex=True).str[:8], errors="coerce")
allp["known_date"] = allp[["pub_date", "rgst_date"]].min(axis=1)
allp = allp.sort_values("src", ascending=False).drop_duplicates(["appl_no", "applicantCd", "ksic"])  # registry 우선

lk = pd.read_parquet(os.path.join(EXT, "firm_nps_applicant_link.parquet"))
lk = lk[lk.nps_hit]
lk["key"] = lk.biz6 + "|" + lk.nname
allp = allp.merge(lk[["applicantCd", "key"]], on="applicantCd", how="left")
print("기업키 연결 비율(출원 기준)", round(allp.dropna(subset=["key"]).appl_no.nunique() / allp.appl_no.nunique(), 3))
allp["ay"] = (allp.appl_date // 10000).astype("Int64")
allp[["key", "applicantCd", "appl_no", "appl_date", "ay", "known_date", "ksic", "src"]].to_parquet(os.path.join(CACHE, "app_firm.parquet"), index=False)
tot = k.appl_no.nunique()
print("제공 KR 출원", tot, "코드 부여", allp.appl_no.nunique(), round(allp.appl_no.nunique() / tot, 3),
      "기업키", allp.dropna(subset=["key"]).appl_no.nunique(), round(allp.dropna(subset=["key"]).appl_no.nunique() / tot, 3), round(time.time() - t))
