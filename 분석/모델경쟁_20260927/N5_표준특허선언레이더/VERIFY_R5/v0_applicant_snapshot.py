# -*- coding: utf-8 -*-
"""누수 재탐색: 제공 KR 출원인명이 '원출원인'인지 '현재 권리자(사후 스냅샷)'인지 — 26429 등록특허 중 권리이전 건으로 확인."""
import sys, os, sqlite3, json
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from n5lib import norm, BASE
import pandas as pd
k = pd.read_parquet(os.path.join(HERE, "cache", "kr_26429.parquet"))
c = sqlite3.connect(f"file:{os.path.join(BASE, '수집파이프라인', 'registry_full.sqlite')}?mode=ro", uri=True)
pt = pd.read_sql("select rgst_no, applNo from patent where ksic='26429'", c)
ap = pd.read_sql("select a.rgst_no, a.seq, a.applicantName from applicant a join patent p on a.rgst_no=p.rgst_no where p.ksic='26429'", c)
ow = pd.read_sql("select o.rgst_no, o.seq, o.ownerName, o.finalOwnerYn from owner o join patent p on o.rgst_no=p.rgst_no where p.ksic='26429'", c)
c.close()
a0 = ap.sort_values("seq").groupby("rgst_no").applicantName.first().map(norm)
fo = ow[ow.finalOwnerYn == "Y"].sort_values("seq").groupby("rgst_no").ownerName.first().map(norm)
d = pt.assign(orig=pt.rgst_no.map(a0), final=pt.rgst_no.map(fo)).dropna()
d = d.merge(k[["appl_no", "app", "ay"]], left_on="applNo", right_on="appl_no")
tr = d[d.orig != d.final]
res = dict(n_registered_matched=len(d), n_transferred=len(tr),
           match_orig=float((tr.app == tr.orig).mean()), match_final=float((tr.app == tr.final).mean()),
           match_neither=float(((tr.app != tr.orig) & (tr.app != tr.final)).mean()),
           nontransfer_match=float((d[d.orig == d.final].app == d[d.orig == d.final].orig).mean()))
print(json.dumps(res, ensure_ascii=False, indent=1))
json.dump(res, open(os.path.join(HERE, "v0_applicant_snapshot.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
