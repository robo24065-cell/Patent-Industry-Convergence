# -*- coding: utf-8 -*-
# verify_0 (C08-T1 new_result: KR 출원인 필드) 반박 시도 1단계: 구분자·공동출원 선택 규칙. 읽기 전용.
import sys, os, json, sqlite3, re, unicodedata, time
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd, duckdb
t0 = time.time()
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
META = os.path.join(ROOT, r"분석\모델경쟁_20260927\_shared\meta\*.parquet").replace("\\", "/")
OUT = os.path.join(ROOT, r"분석\검증루프_20261012\C08\C08-T1\verify_0")
con = duckdb.connect(); con.sql("set memory_limit='3GB'; set threads=2; set enable_progress_bar=false")
res = {}
# 1) 구분자 존재 여부 (국가별)
q = f"""select cc, count(*) n,
 sum(case when contains(applicant,'|') then 1 else 0 end) n_pipe,
 sum(case when contains(applicant,';') then 1 else 0 end) n_semicolon,
 sum(case when contains(applicant,',') then 1 else 0 end) n_comma,
 sum(case when applicant is null or trim(applicant)='' then 1 else 0 end) n_null
 from read_parquet('{META}') group by 1 order by 1"""
d = con.sql(q).df(); print(d); res["separators"] = d.to_dict("records")
ex = con.sql(f"select applicant from read_parquet('{META}') where cc='KR' and (contains(applicant,'|') or contains(applicant,';') or contains(applicant,',')) limit 15").df()
print(ex); res["kr_sep_examples"] = ex.applicant.tolist()

# 2) 공동출원
def V(s):
    if not isinstance(s, str): return None
    s = unicodedata.normalize("NFKC", s).lower()
    for t in ["주식회사", "(주)", "㈜", "유한회사", "재단법인", "사단법인"]:
        s = s.replace(t, "")
    s = re.sub(r"[^\w]|_", "", s)
    return s or None
kr = con.sql(f"""select distinct appl_no, appl_year yr, trim(applicant) raw from read_parquet('{META}')
  where cc='KR' and substr(appl_no,1,2)='10' and appl_year between 2015 and 2019""").df()
print("kr 2015-19 rows(distinct appl_no,raw)", len(kr), "unique appl", kr.appl_no.nunique())
dup = kr.groupby("appl_no").raw.nunique(); res["appl_with_multi_raw_across_ksic"] = int((dup > 1).sum())
kr = kr.drop_duplicates("appl_no")
db = sqlite3.connect("file:" + os.path.join(ROOT, r"수집파이프라인\registry_full.sqlite") + "?mode=ro", uri=True)
ap = pd.read_sql("select rgst_no, seq, applicantCd cd, applicantName nm, applicantNatl natl, rpstrYn rp from applicant", db)
pt = pd.read_sql("select rgst_no, applNo from patent", db); db.close()
pt["applNo"] = pt.applNo.astype(str).str.replace("-", "")
ap = ap.merge(pt, on="rgst_no")
ap = ap[ap.applNo.isin(set(kr.appl_no))]
cnt = ap.groupby("applNo").size()
res["n_unique_appl_2015_19"] = int(len(kr)); res["n_linked"] = int(cnt.size); res["n_joint"] = int((cnt > 1).sum())
res["joint_share_of_all"] = round(res["n_joint"] / len(kr), 4); res["joint_share_of_linked"] = round(res["n_joint"] / cnt.size, 4)
print({k: res[k] for k in ["n_unique_appl_2015_19", "n_linked", "n_joint", "joint_share_of_all", "joint_share_of_linked"]})
J = ap[ap.applNo.isin(cnt.index[cnt > 1])].copy()
J["v"] = J.nm.map(V)
J = J.merge(kr[["appl_no", "raw"]], left_on="applNo", right_on="appl_no")
J["vp"] = J.raw.map(V)
J["hit"] = J.v == J.vp
g = J.sort_values(["applNo", "seq"]).groupby("applNo")
first = g.head(1).set_index("applNo"); last = g.tail(1).set_index("applNo")
anyhit = g.hit.any()
cdmin = J.sort_values(["applNo", "cd"]).groupby("applNo").head(1).set_index("applNo")
cdmax = J.sort_values(["applNo", "cd"]).groupby("applNo").tail(1).set_index("applNo")
nmmax = J.sort_values(["applNo", "nm"]).groupby("applNo").tail(1).set_index("applNo")
nmmin = J.sort_values(["applNo", "nm"]).groupby("applNo").head(1).set_index("applNo")
r = dict(n=int(anyhit.size), match_any=round(float(anyhit.mean()), 4),
         match_seq_first=round(float(first.hit.mean()), 4), match_seq_last=round(float(last.hit.mean()), 4),
         match_cd_min=round(float(cdmin.hit.mean()), 4), match_cd_max=round(float(cdmax.hit.mean()), 4),
         match_name_max=round(float(nmmax.hit.mean()), 4), match_name_min=round(float(nmmin.hit.mean()), 4))
# 2명 공동출원으로 한정, 둘 중 하나와 맞는 문서에서 규칙 일치율
n_ap = J.groupby("applNo").size()
two = n_ap.index[n_ap == 2]
sel = [a for a in two if anyhit.get(a, False)]
r["two_applicant_docs"] = int(len(two)); r["two_with_hit"] = int(len(sel))
for nm_, T in [("seq_first", first), ("seq_last", last), ("cd_min", cdmin), ("cd_max", cdmax), ("name_max", nmmax), ("name_min", nmmin)]:
    r["cond_two_" + nm_] = round(float(T.loc[sel].hit.mean()), 4)
# 맞는 출원인의 성격: 법인/개인(코드 첫 자리 1=법인? 4=개인), 국적
hitrows = J[J.hit]
r["hit_cd_prefix"] = hitrows.cd.astype(str).str[:1].value_counts().to_dict()
r["all_cd_prefix"] = J.cd.astype(str).str[:1].value_counts().to_dict()
r["hit_rp"] = hitrows.rp.value_counts(dropna=False).to_dict(); r["all_rp"] = J.rp.value_counts(dropna=False).to_dict()
# 규칙 후보: '법인 코드(1...) 우선, 그 안에서 seq 최소'
J["corp"] = J.cd.astype(str).str[:1] == "1"
pc = J.sort_values(["applNo", "corp", "seq"], ascending=[True, False, True]).groupby("applNo").head(1).set_index("applNo")
r["match_corp_then_seq"] = round(float(pc.hit.mean()), 4)
pc2 = J.sort_values(["applNo", "corp", "seq"], ascending=[True, False, False]).groupby("applNo").head(1).set_index("applNo")
r["match_corp_then_lastseq"] = round(float(pc2.hit.mean()), 4)
# 국내 우선
J["dom"] = J.natl.astype(str).str.contains("대한민국|KR")
pd_ = J.sort_values(["applNo", "dom", "seq"], ascending=[True, False, True]).groupby("applNo").head(1).set_index("applNo")
r["match_domestic_then_seq"] = round(float(pd_.hit.mean()), 4)
# 연도별 first/last 일치율 (규칙이 시기마다 바뀌는지)
yr = kr.set_index("appl_no").yr
r["by_year_first_last"] = {int(y): (round(float(first.hit[first.index.map(yr) == y].mean()), 3), round(float(last.hit[last.index.map(yr) == y].mean()), 3)) for y in range(2015, 2020)}
res["joint"] = r
print(json.dumps(r, ensure_ascii=False, indent=1))
J.drop(columns=["appl_no"]).to_parquet(os.path.join(OUT, "v2_joint_tmp.parquet"))
json.dump(res, open(os.path.join(OUT, "v2_probe_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print("done", round(time.time() - t0))
