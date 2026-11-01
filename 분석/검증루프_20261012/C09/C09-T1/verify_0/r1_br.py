# -*- coding: utf-8 -*-
# 반박 검증 보조: business_rights 직접(mode=ro) 현대 코드 범위 안 비율 + 이름 연도표(출원연도 열 수정)
import sys, os, json, sqlite3
sys.dont_write_bytecode = True; sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
OUT = os.path.join(ROOT, r"분석\검증루프_20261012\C09\C09-T1\verify_0")
kp = pd.read_parquet(os.path.join(ROOT, r"공모전데이터\작업데이터\kr_patents.parquet"), columns=["출원번호", "출원연도", "출원인명"])
kp["a"] = kp["출원번호"].astype(str).str.replace("-", "", regex=False)
k10 = kp[kp.a.str[:2] == "10"]
res = {}
nm = k10[k10["출원인명"].isin(["현대자동차주식회사", "기아 주식회사"])].groupby(["출원연도", "출원인명"]).size().unstack(fill_value=0)
res["name_year"] = {int(k): v for k, v in nm.loc[2014:2020].to_dict("index").items()}
c = sqlite3.connect("file:" + os.path.join(ROOT, r"수집파이프라인\business_rights.sqlite").replace("\\", "/") + "?mode=ro", uri=True)
h = pd.DataFrame(c.execute("""select distinct rgst_no, appl_no, substr(appl_date,1,4), json_extract(raw,'$.applicantInfo') from right_list
  where applicant_cd='119980045675' and substr(rgst_no,1,2)='10' and rgst_date<='20250822'""").fetchall(), columns=["r", "a", "ay", "ai"]).drop_duplicates("r")
h["a"] = h.a.astype(str).str.replace("-", "", regex=False)
h["in"] = h.a.isin(set(k10.a)); h["kia"] = h.ai.fillna("").str.contains("기아")
g = h.groupby("ay").agg(n=("r", "size"), n_in=("in", "sum"), solo=("kia", lambda s: int((~s).sum())), joint=("kia", "sum"))
g["share"] = (g.n_in / g.n).round(4)
res["br_hy"] = g.loc["2014":"2020"].reset_index().astype(str).values.tolist()
json.dump(res, open(os.path.join(OUT, "r1_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(res, ensure_ascii=False, default=str))
