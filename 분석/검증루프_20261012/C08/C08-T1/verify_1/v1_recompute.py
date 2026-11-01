# -*- coding: utf-8 -*-
# verify_1: C08-T1 발견(P3 KSIC 안 1위 출원인 몫) 독립 재계산. 원자료 meta parquet 읽기만. 주 경로 cache/common 미사용.
import sys, os, json, re, unicodedata
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
import duckdb, pandas as pd, numpy as np
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
META = os.path.join(ROOT, r"분석\모델경쟁_20260927\_shared\meta\*.parquet").replace("\\", "/")
OUT = os.path.join(ROOT, r"분석\검증루프_20261012\C08\C08-T1\verify_1")
con = duckdb.connect(); con.sql("set memory_limit='3GB'; set threads=2; set enable_progress_bar=false")
inv = """case when cc='KR' then substr(appl_no,1,2)='10'
 when cc='CN' then length(appl_no) in (12,14) and substr(appl_no,5,1) in ('1','8')
 when cc='JP' then length(appl_no) in (8,17) else true end"""
d = con.sql(f"""select ksic, cc, nullif(trim(split_part(coalesce(applicant,''),'|',1)),'') a, count(*) n, count(distinct appl_no) nu
 from read_parquet('{META}') where appl_year between 2017 and 2019 and ({inv}) group by 1,2,3""").df()
# 단순 정규화 V1: NFKC, 소문자, 한국 법인 표기·영문 접미어 일부, 공백·구두점 제거 (주 경로 N2와 다름)
KO = ["주식회사", "(주)", "㈜", "유한회사", "재단법인", "사단법인"]
LAT = re.compile(r"\b(co|ltd|inc|corp|corporation|company|limited|llc|gmbh|ag|kabushiki kaisha)\b\.?")
def v1(s):
    if s is None or (isinstance(s, float)): return None
    s = unicodedata.normalize("NFKC", s).lower()
    for t in KO: s = s.replace(t, "")
    s = LAT.sub("", s)
    s = re.sub(r"[\W_]+", "", s)
    return s or None
d["k"] = [v1(x) for x in d.a.values]
res = {}
def conc(d, col):
    g = d[d.k.notna()].groupby(["ksic", "cc", "k"])[col].sum().reset_index()
    tot = g.groupby(["ksic", "cc"])[col].transform("sum")
    g["sh"] = g[col] / tot
    g = g.sort_values(["ksic", "cc", "sh"], ascending=[True, True, False])
    top = g.drop_duplicates(["ksic", "cc"]).set_index(["ksic", "cc"])
    hhi = g.assign(s2=g.sh ** 2).groupby(["ksic", "cc"]).s2.sum()
    top["effn"] = 1 / hhi
    top["miss"] = d[d.k.isna()].groupby(["ksic", "cc"])[col].sum().reindex(top.index).fillna(0) / d.groupby(["ksic", "cc"])[col].sum().reindex(top.index)
    return top.reset_index()
for col in ["n", "nu"]:
    T = conc(d, col)
    res[col] = dict(med=T.groupby("cc").sh.median().round(4).to_dict(), effn_med=T.groupby("cc").effn.median().round(1).to_dict(),
                    n_ge30=T[T.sh >= 0.3].groupby("cc").size().to_dict(),
                    kr_ge30=T[(T.cc == "KR") & (T.sh >= 0.3)][["ksic", "k", "sh", "effn"]].round(4).values.tolist(),
                    kr_ge25=T[(T.cc == "KR") & (T.sh >= 0.25)][["ksic", "k", "sh"]].round(4).values.tolist(),
                    kr_top10=T[T.cc == "KR"].nlargest(10, "sh")[["ksic", "k", "sh"]].round(4).values.tolist(),
                    nksic=T.groupby("cc").size().to_dict())
    T.to_csv(os.path.join(OUT, f"v1_top1_P3_{col}.csv"), index=False, encoding="utf-8-sig")
# 그룹 병합 민감도: KR 엘지/LG 계열·삼성 계열·현대 계열 병합 시 1위 몫
def grp(k):
    if k is None: return None
    for p, gname in [("엘지", "LG계"), ("lg", "LG계"), ("삼성", "삼성계"), ("samsung", "삼성계"), ("현대", "현대계"), ("기아", "현대계"), ("포스코", "포스코계"), ("posco", "포스코계"), ("에스케이", "SK계"), ("sk", "SK계")]:
        if k.startswith(p): return gname
    return k
dk = d[d.cc == "KR"].copy(); dk["k"] = [grp(x) for x in dk.k.values]
TG = conc(dk, "n")
res["KR_group_merge"] = dict(med=round(float(TG.sh.median()), 4), n_ge30=int((TG.sh >= 0.3).sum()),
                            ge30=TG[TG.sh >= 0.3][["ksic", "k", "sh"]].round(4).values.tolist())
# 원 파일과 대조
C = pd.read_csv(os.path.join(ROOT, r"분석\검증루프_20261012\C08\C08-T1\concentration_by_cell.csv"), dtype={"ksic": str})
c3 = C[C.period == "P3"].set_index(["ksic", "cc"])
T = conc(d, "n").set_index(["ksic", "cc"])
j = T.join(c3[["n_docs", "top1_share", "top1_name"]])
j["ndocs_v1"] = d.groupby(["ksic", "cc"]).n.sum().reindex(j.index)
res["max_abs_diff_top1"] = round(float((j.sh - j.top1_share).abs().max()), 4)
res["n_cells_diff_gt_0.01"] = int(((j.sh - j.top1_share).abs() > 0.01).sum())
res["ndocs_equal"] = bool((j.ndocs_v1 == j.n_docs).all())
res["diff_cells"] = j[(j.sh - j.top1_share).abs() > 0.01].reset_index()[["ksic", "cc", "k", "sh", "top1_name", "top1_share"]].round(4).values.tolist()
# KR 공동출원 비중(셀별) — '|' 포함 비율, 1위 4개 KSIC
jt = con.sql(f"""select ksic, avg((applicant like '%|%')::int) joint from read_parquet('{META}')
 where cc='KR' and appl_year between 2017 and 2019 and substr(appl_no,1,2)='10' and ksic in ('28511','29132','26212','24191') group by 1""").df()
res["KR_joint_share_4"] = jt.round(4).values.tolist()
# 4개 셀에서 '어느 이름이든 포함'(공동출원 포함) 기준 1위 몫
for ks, pat in [("28511", "엘지전자"), ("29132", "엘지전자"), ("26212", "엘지디스플레이"), ("24191", "포스코")]:
    r = con.sql(f"""select avg((split_part(applicant,'|',1) like '%{pat}%')::int) first_, avg((applicant like '%{pat}%')::int) any_
     from read_parquet('{META}') where cc='KR' and appl_year between 2017 and 2019 and substr(appl_no,1,2)='10' and ksic='{ks}'""").fetchone()
    res[f"KR_{ks}_{pat}_first_any"] = [round(r[0], 4), round(r[1], 4)]
json.dump(res, open(os.path.join(OUT, "v1_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
