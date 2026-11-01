# -*- coding: utf-8 -*-
"""누수 재탐색(N4): 모집단 조건 = 사업자번호(CORP_APPLICANT 2026-07 사후 목록) → 국세청 등록. 원점 시점 특허 활동별 연결률이 다른가?
대상: 국내법인 출원인코드(1로 시작) 중 t0까지 제공범위 KR 등록특허 보유. 레이블은 쓰지 않는다."""
import sys, os, json
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); TR = os.path.dirname(HERE); sys.path.insert(0, TR)
import n4lib
n4lib.CACHE = os.path.join(HERE, "cache")
from n4lib import load_app_firm, load_registry, load_nts
import numpy as np, pandas as pd
AF = load_app_firm(); AF = AF[(AF.applicantCd.astype(str).str[0] == "1") & AF.known_date.notna()]
R = load_registry(); ns = load_nts()
res = {}
for t0 in [2017, 2021, 2023]:
    end = t0 * 10000 + 1231
    kn = AF[AF.known_date <= end]
    p = R["pt"][R["pt"].rgstDate <= end]
    firms = set(kn.applicantCd) & set(p.applicantCd)
    g = kn[kn.applicantCd.isin(firms)].groupby("applicantCd")
    f = pd.DataFrame({"last": g.known_date.max(), "n": g.appl_no.nunique()})
    f["yrs"] = t0 - f["last"] // 10000
    live = p[~(p.ext_d <= end)].groupby("applicantCd").rgst_no.nunique()
    f["has_live"] = f.index.map(live).fillna(0) > 0
    f["linked"] = f.index.isin(set(ns.applicantCd))
    f["yb"] = pd.cut(f.yrs, [-1, 0, 2, 5, 10, 100], labels=["0", "1-2", "3-5", "6-10", ">10"])
    f["nb"] = pd.cut(f.n, [0, 1, 3, 10, 1e9], labels=["1", "2-3", "4-10", ">10"])
    res[t0] = dict(n_firms=len(f), link_rate=float(f.linked.mean()),
                   by_yrs_since_last=f.groupby("yb", observed=True).linked.agg(["mean", "size"]).round(4).to_dict("index"),
                   by_has_live=f.groupby("has_live").linked.agg(["mean", "size"]).round(4).to_dict("index"),
                   by_n_app=f.groupby("nb", observed=True).linked.agg(["mean", "size"]).round(4).to_dict("index"))
    print(t0, json.dumps(res[t0], ensure_ascii=False, default=str), flush=True)
json.dump(res, open(os.path.join(HERE, "v5_pop_link.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
