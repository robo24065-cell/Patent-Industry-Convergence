# -*- coding: utf-8 -*-
# 반박 시도 4단계: 같은 출원인 코드 안에서 제공 이름이 문서마다 갈리는지(이름 시점이 코드별 단일인가), 갈린다면 공개일/등록일 기준인가. 읽기 전용.
import sys, os, json, sqlite3, time
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd, duckdb
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
META = os.path.join(ROOT, r"분석\모델경쟁_20260927\_shared\meta\*.parquet").replace("\\", "/")
OUT = os.path.join(ROOT, r"분석\검증루프_20261012\C08\C08-T1\verify_0")
con = duckdb.connect(); con.sql("set memory_limit='3GB'; set threads=2; set enable_progress_bar=false")
kr = con.sql(f"""select appl_no, any_value(appl_year) yr, any_value(trim(applicant)) raw, any_value(pub_date) pub, any_value(rgst_date) rgd
 from read_parquet('{META}') where cc='KR' and substr(appl_no,1,2)='10' group by 1""").df()
db = sqlite3.connect("file:" + os.path.join(ROOT, r"수집파이프라인\registry_full.sqlite") + "?mode=ro", uri=True)
ap = pd.read_sql("select a.applicantCd cd, a.applicantName nm, p.applNo from applicant a join patent p on a.rgst_no=p.rgst_no", db); db.close()
ap["applNo"] = ap.applNo.astype(str).str.replace("-", "")
m = ap.merge(kr, left_on="applNo", right_on="appl_no")
r = {}
targets = {"두산": "두산에너빌리티|두산중공업", "HD현대인프라코어": "인프라코어", "한국조선해양": "한국조선해양 주식회사|에이치디한국조선해양",
           "포스코홀딩스": "포스코홀딩스", "포스코퓨처엠": "포스코퓨처엠|포스코케미칼", "레조낙": "레조낙|쇼와 덴코", "기아": "^기아"}
for k, pat in targets.items():
    codes = m[m.nm.str.contains(pat, regex=True, na=False)].cd.value_counts().head(2).index.tolist()
    for c in codes:
        s = m[(m.cd == c)]
        s = s[s.raw.str.contains(pat.replace("^", ""), regex=True, na=False) | (k in ["포스코홀딩스"]) & s.raw.str.contains("포스코", na=False)]
        if not len(s): continue
        s = s.assign(pubyr=s.pub.astype(str).str[:4], rgyr=s.rgd.astype(str).str[:4])
        tab = s.groupby(["raw"]).agg(n=("appl_no", "size"), appl_min=("yr", "min"), appl_max=("yr", "max"),
                                     pub_min=("pubyr", "min"), pub_max=("pubyr", "max"), rg_min=("rgyr", "min"), rg_max=("rgyr", "max"))
        r[f"{k}|{c}|{s.nm.iloc[0]}"] = tab.reset_index().to_dict("records")
# 코드 안 이름 수 분포(코드당 V정규화 전 원문 이름 개수) — 문서 비중
g = m.groupby("cd").raw.nunique()
r["codes_total"] = int(g.size); r["codes_with_ge2_raw_names"] = int((g > 1).sum())
r["docs_in_codes_with_ge2_raw"] = round(float(m.cd.isin(g.index[g > 1]).mean()), 4)
print(json.dumps(r, ensure_ascii=False, indent=1))
json.dump(r, open(os.path.join(OUT, "v2_vintage_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
