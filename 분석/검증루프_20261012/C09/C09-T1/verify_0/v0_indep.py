# -*- coding: utf-8 -*-
# verify_0: 독립 재계산 — duckdb·주 경로 캐시 미사용. sqlite3 직접 집계 + 작업데이터 kr_patents.parquet(다른 원천 파일)
import sys, os, json, time, sqlite3
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
t0 = time.time()
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
OUT = os.path.join(ROOT, r"분석\검증루프_20261012\C09\C09-T1\verify_0")
kp = pd.read_parquet(os.path.join(ROOT, r"공모전데이터\작업데이터\kr_patents.parquet"), columns=["출원번호", "출원연도", "출원인명", "세세분류코드"])
kp["출원번호"] = kp["출원번호"].astype(str).str.replace("-", "", regex=False)
res = {"kr_patents_rows": len(kp), "kr_patents_applno_len": kp["출원번호"].str.len().value_counts().to_dict()}
inset = set(kp["출원번호"])
c = sqlite3.connect("file:" + os.path.join(ROOT, r"수집파이프라인\business_rights.sqlite").replace("\\", "/") + "?mode=ro", uri=True)
# (i) 출원연도별: 특허 단위(rgst_no 중복 제거)
rows = c.execute("""select distinct r.rgst_no, r.appl_no, substr(r.appl_date,1,4) from right_list r
  where substr(r.rgst_no,1,2)='10' and r.rgst_date<='20250822'
  and r.applicant_cd in (select applicant_cd from applicant_query where status='OK')""").fetchall()
df = pd.DataFrame(rows, columns=["rgst_no", "appl_no", "ay"]).drop_duplicates("rgst_no")
df["in"] = df.appl_no.isin(inset)
g = df.groupby("ay").agg(n=("rgst_no", "size"), n_in=("in", "sum"))
g["share"] = (g.n_in / g.n).round(4)
res["i_share"] = g.loc["2008":"2023"].reset_index().to_dict("records")
print("i", round(time.time() - t0), flush=True)
# 현대 코드
HY = "119980045675"
h = pd.DataFrame(c.execute(f"""select rgst_no, appl_no, substr(appl_date,1,4) ay, instr(json_extract(raw,'$.applicantInfo'),'기아') kia
  from right_list where applicant_cd='{HY}' and substr(rgst_no,1,2)='10' and rgst_date<='20250822'""").fetchall(), columns=["rgst_no", "appl_no", "ay", "kia"])
h["in"] = h.appl_no.isin(inset)
hg = h.groupby("ay").agg(n=("rgst_no", "size"), n_in=("in", "sum"), joint_kia=("kia", lambda s: int((s > 0).sum())))
hg["share"] = (hg.n_in / hg.n).round(4)
res["hy_code"] = hg.loc["2012":"2021"].reset_index().to_dict("records")
# 현대·기아 이름(작업데이터 원천)
k10 = kp[kp["출원번호"].str[:2] == "10"]
nm = k10[k10["출원인명"].isin(["현대자동차주식회사", "기아 주식회사"])].groupby(["출원연도", "출원인명"]).size().unstack(fill_value=0)
res["hy_kia_name"] = nm.loc[2012:2023].reset_index().to_dict("records")
h16 = k10[(k10["출원인명"] == "현대자동차주식회사") & (k10["출원연도"] == 2016)]
res["hy2016_ksic_top5"] = h16["세세분류코드"].value_counts().head(5).to_dict()
# 코드 기준 급감(독립 구현): 상위 300(2010–16 범위 안 특허 수, 권리자 코드 단위 쌍)
pairs = pd.DataFrame(c.execute("""select applicant_cd, rgst_no, appl_no, substr(appl_date,1,4) from right_list
  where substr(rgst_no,1,2)='10' and rgst_date<='20250822' and applicant_cd in (select applicant_cd from applicant_query where status='OK')""").fetchall(),
                     columns=["cd", "rgst_no", "appl_no", "ay"])
pairs["ay"] = pd.to_numeric(pairs.ay, errors="coerce"); pairs["in"] = pairs.appl_no.isin(inset)
base = pairs[pairs["in"] & pairs.ay.between(2010, 2016)].cd.value_counts().head(300).index
P = pairs[pairs.cd.isin(base)].groupby(["cd", "ay"]).agg(n=("rgst_no", "size"), n_in=("in", "sum")).reset_index()
P["s"] = P.n_in / P.n
drops = []
for cd, grp in P.groupby("cd"):
    s = grp.set_index("ay")
    for t in range(2011, 2023):
        pv = [t - 3, t - 2, t - 1]
        if not all(p in s.index for p in pv):
            continue
        if s.loc[t, "s"] if t in s.index else 0.0 < 0:
            pass
        st = s.s.get(t, 0.0)
        if st < 0.25 * s.loc[pv, "s"].mean() and s.loc[pv, "n_in"].sum() >= 50:
            drops.append((cd, t, int(s.n.get(t, 0)), round(float(s.loc[pv, "n"].mean()), 1))); break
res["code_dropouts"] = drops; res["n_code_dropouts"] = len(drops)
res["sec"] = round(time.time() - t0)
json.dump(res, open(os.path.join(OUT, "v0_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(res, ensure_ascii=False, default=str)[:5000])
