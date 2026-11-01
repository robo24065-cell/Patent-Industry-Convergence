# -*- coding: utf-8 -*-
"""C07 사전 점검: 권리목록(business_rights, 2026 스냅샷)의 시점 정합.
원점까지 등록된 자사 제공범위 특허 중 권리목록에 그 출원인코드로 남은 비율, 조회 실패율을 y별로 본다(dev 원점만)."""
import os, numpy as np, pandas as pd
from n7lib import *
D = pd.read_parquet(os.path.join(CACHE, "dev_panel.parquet"))
R = load_registry(); pt = R["pt"]
c = ro(BRS)
qs = pd.read_sql("select applicant_cd, status from applicant_query", c)
ok = set(qs[qs.status == "OK"].applicant_cd); fail = set(qs[qs.status != "OK"].applicant_cd)
rl = pd.read_sql("select applicant_cd, rgst_no from right_list where substr(rgst_no,1,2)='10'", c)
rl_pairs = set(zip(rl.applicant_cd, rl.rgst_no))
rows = []
for t0 in sorted(D.t0.unique()):
    B = D[D.t0 == t0]; end = t0 * 10000 + 1231
    p = pt[(pt.rgstDate <= end) & pt.applicantCd.isin(set(B.applicantCd))][["rgst_no", "applicantCd"]].drop_duplicates()
    p["kept"] = [(a, r) in rl_pairs for a, r in zip(p.applicantCd, p.rgst_no)]
    p = p[p.applicantCd.isin(ok)]
    kr = p.groupby("applicantCd").kept.mean()
    for yv in [0, 1]:
        Bs = B[B.y == yv]
        rows.append(dict(t0=t0, y=yv, n=len(Bs), query_fail_rate=Bs.applicantCd.isin(fail).mean(), query_missing=(~Bs.applicantCd.isin(ok | fail)).mean(),
                         kept_rate_firm_mean=Bs.applicantCd.map(kr).mean(), kept_rate_patent=p[p.applicantCd.isin(set(Bs.applicantCd))].kept.mean()))
T = pd.DataFrame(rows)
print(T.round(4).to_string(index=False))
T.to_csv(os.path.join(OUT, "rl_leak_check.csv"), index=False, encoding="utf-8-sig")
w = T.pivot(index="t0", columns="y", values=["kept_rate_patent", "query_fail_rate"])
gap = (w["kept_rate_patent"][1] - w["kept_rate_patent"][0]).abs()
ratio = (w["query_fail_rate"][0] / w["query_fail_rate"][1].replace(0, np.nan))
print("보유율 차이(%p)", (gap * 100).round(2).to_dict()); print("조회실패 비율(y0/y1)", ratio.round(2).to_dict())
leak = bool((gap >= 0.03).any() or (ratio >= 1.5).any() or (ratio <= 1 / 1.5).any())
print("C07 누수 판정:", leak)
