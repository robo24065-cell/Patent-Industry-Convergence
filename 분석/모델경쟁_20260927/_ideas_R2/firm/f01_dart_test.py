# -*- coding: utf-8 -*-
"""OpenDART fnlttMultiAcnt 동작 확인(소량 3콜). 키는 파일에서 읽고 출력하지 않는다."""
import os, re, json, time, urllib.request
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
txt = open(os.path.join(BASE, "API키_기록.txt"), encoding="utf-8").read()
sec = txt.split("[금융감독원 OpenDART]")[1]
KEY = re.search(r"키:\s*([0-9a-f]{40})", sec).group(1)

def call(codes, year):
    u = ("https://opendart.fss.or.kr/api/fnlttMultiAcnt.json?crtfc_key=" + KEY +
         "&corp_code=" + ",".join(codes) + f"&bsns_year={year}&reprt_code=11011")
    with urllib.request.urlopen(u, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))

import pandas as pd
d = pd.read_parquet(os.path.join(BASE, r"분석\모델경쟁_20260927\_ideas_R2\firm\cache\dart_firms.parquet"))
pf = pd.read_parquet(os.path.join(BASE, r"분석\모델경쟁_20260927\_ideas_R2\firm\cache\pat_firm.parquet"))
codes = sorted(pf.corp_code.dropna().unique())
E = d[d.corp_code.isin(codes) & (d.corp_cls == "E")].corp_code.head(100).tolist()
K = d[d.corp_code.isin(codes) & (d.corp_cls == "K")].corp_code.head(100).tolist()
for nm, cc in [("K100", K), ("E100", E)]:
    j = call(cc, 2018)
    st = j.get("status"); lst = j.get("list", [])
    print(nm, st, j.get("message"), "rows", len(lst), "firms", len({x["corp_code"] for x in lst}))
    if lst:
        print(sorted({x["account_nm"] for x in lst}))
        print({k: lst[0][k] for k in lst[0] if k not in ()})
    time.sleep(1.0)
