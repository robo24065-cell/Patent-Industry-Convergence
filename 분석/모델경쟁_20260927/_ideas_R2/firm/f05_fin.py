# -*- coding: utf-8 -*-
"""DART fnlttMultiAcnt 원시행 → 기업×회계연도 주요계정 (별도재무 OFS 우선, 없으면 연결 CFS).
같은 연도 값이 여러 보고서에 있으면 당기(thstrm) > 전기 > 전전기 순으로 채택(최초 보고 값 우선).
출력: cache/fin.parquet (corp_code, year, rev, op, ni, assets, liab, equity, capital, cur_assets, fs)"""
import os, json
import numpy as np, pandas as pd
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
C = os.path.join(BASE, r"분석\모델경쟁_20260927\_ideas_R2\firm\cache")
RAW = os.path.join(BASE, r"분석\모델경쟁_20260927\_shared\ext\R2firm_dart_fnltt_raw.jsonl")
ACC = {"매출액": "rev", "영업이익": "op", "당기순이익(손실)": "ni", "자산총계": "assets", "부채총계": "liab",
       "자본총계": "equity", "자본금": "capital", "유동자산": "cur_assets", "이익잉여금": "re"}
rows = []
for line in open(RAW, encoding="utf-8"):
    x = json.loads(line)
    a = ACC.get(x["account_nm"])
    if a is None:
        continue
    by = int(x["bsns_year"])
    for k, off, pr in (("thstrm_amount", 0, 0), ("frmtrm_amount", 1, 1), ("bfefrmtrm_amount", 2, 2)):
        v = (x.get(k) or "").replace(",", "").strip()
        if v in ("", "-"):
            continue
        try:
            v = float(v)
        except ValueError:
            continue
        rows.append((x["corp_code"], by - off, x["fs_div"], a, v, pr))
D = pd.DataFrame(rows, columns=["corp_code", "year", "fs", "acc", "v", "pr"])
D = D.sort_values("pr").drop_duplicates(["corp_code", "year", "fs", "acc"])
W = D.pivot_table(index=["corp_code", "year", "fs"], columns="acc", values="v").reset_index()
W["fsr"] = (W.fs != "OFS").astype(int)
W = W.sort_values("fsr").drop_duplicates(["corp_code", "year"]).drop(columns="fsr")
assert W.year.max() <= 2023
W.to_parquet(os.path.join(C, "fin.parquet"), index=False)
print(W.groupby("year").agg(firms=("corp_code", "size"), rev_ok=("rev", lambda s: s.notna().sum()), ofs=("fs", lambda s: (s == "OFS").mean())))
