# -*- coding: utf-8 -*-
"""권리목록(business_rights.right_list) 시점 정합 점검: 제공범위 등록특허(등록원부, 원점까지 등록) 중 권리목록에 그 출원인코드로 남아 있는 비율을
결과(y)·원점 이후 포함 전 기간 권리자 변경 여부·소멸 여부별로 본다. 1에 가깝지 않고 y·양도와 함께 움직이면 '현재 권리자' 스냅샷 → 누수."""
import os
import pandas as pd, numpy as np
from n4lib import *
R = load_registry(); pt = R["pt"]; ow, sur = load_registry_extra()
D = pd.read_parquet(os.path.join(CACHE, "base_dev.parquet"))
c = ro(BRS)
q = pd.read_sql("select applicant_cd, status from applicant_query", c).set_index("applicant_cd").status
rl = pd.read_sql("select applicant_cd, rgst_no from right_list where substr(rgst_no,1,2)='10'", c)
out = []
for t0 in [2017, 2019, 2021]:
    d = D[D.t0 == t0].copy()
    d["q"] = d.applicantCd.map(q).fillna("none")
    for (yy, qq), n in d.groupby(["y", "q"]).size().items():
        out.append(dict(t0=t0, part="query_status", y=yy, grp=qq, n=n, found_rate=np.nan))
    d = d[d.q == "OK"]
    end = t0 * 10000 + 1231
    p = pt[pt.applicantCd.isin(set(d.applicantCd)) & (pt.rgstDate <= end)][["rgst_no", "applicantCd", "ext_d"]]
    m = p.merge(rl[rl.applicant_cd.isin(set(d.applicantCd))].rename(columns={"applicant_cd": "applicantCd"}).assign(f=1),
                on=["rgst_no", "applicantCd"], how="left")
    m["f"] = m.f.fillna(0); m["y"] = m.applicantCd.map(d.set_index("applicantCd").y)
    m["owner_changed_ever"] = m.rgst_no.isin(set(ow.rgst_no)); m["extinct_ever"] = m.ext_d.notna()
    for k, g in m.groupby(["y", "owner_changed_ever", "extinct_ever"]):
        out.append(dict(t0=t0, part="found_in_rightlist", y=k[0], grp=f"owner_changed={k[1]},extinct={k[2]}", n=len(g), found_rate=g.f.mean()))
    for yy, g in m.groupby("y"):
        out.append(dict(t0=t0, part="found_in_rightlist", y=yy, grp="ALL", n=len(g), found_rate=g.f.mean()))
O = pd.DataFrame(out)
O.to_csv(os.path.join(OUT, "rl_leak_check.csv"), index=False, encoding="utf-8-sig")
print(O.round(3).to_string(index=False))
