# -*- coding: utf-8 -*-
# 4단계: 데이터 검증 (a)–(d). 읽기 전용(원자료·sqlite mode=ro), 쓰기 cache\s4_*.
import sys, os, json, time, sqlite3
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import numpy as np, pandas as pd, duckdb
t0 = time.time()
con = duckdb.connect(); con.sql("set memory_limit='3GB'; set threads=2; set enable_progress_bar=false; set preserve_insertion_order=false")
FIRST = "nullif(trim(split_part(coalesce(applicant,''),'|',1)),'')"
res = {}
names = pd.read_parquet(os.path.join(CACHE, "s1_names.parquet"))
A = pd.read_parquet(os.path.join(CACHE, "s1_agg.parquet")); A["ksic"] = A.ksic.astype(str)

# ---- (a) 공식 공개 통계(특허청 2018 출원 실적, 2019-01-22 발표 인용 기사) 대 제공 KR 2018 출원 ----
off = {"삼성전자": 5761, "엘지전자": 4558, "엘지화학": 4169, "현대자동차": 2680, "한국전자통신연구원": 1892}
pat = {"삼성전자": "삼성전자", "엘지전자": "엘지전자", "엘지화학": "엘지화학", "현대자동차": "현대자동차", "한국전자통신연구원": "한국전자통신연구원"}
kr18 = con.sql(f"""select {FIRST} raw, count(distinct appl_no) n from read_parquet('{META}')
  where cc='KR' and appl_year=2018 and substr(appl_no,1,2)='10' group by 1""").df()
kr18["n2"] = [norm_n2(x) if isinstance(x, str) else None for x in kr18.raw]
tot18 = int(kr18.n.sum())
rows = []
for k, v in off.items():
    exact = kr18[kr18.n2 == norm_n2(pat[k] + " 주식회사")] if k != "한국전자통신연구원" else kr18[kr18.n2 == "한국전자통신연구원"]
    loose = kr18[kr18.raw.fillna("").str.replace(" ", "").str.contains(pat[k])]
    rows.append(dict(company=k, official_2018_all_fields=v, provided_n2=int(exact.n.sum()), provided_loose=int(loose.n.sum()),
                     n_raw_variants_loose=int(loose.raw.nunique()), variants=" ; ".join(loose.sort_values("n", ascending=False).raw.head(6).astype(str)),
                     coverage=round(exact.n.sum() / v, 3)))
ra = pd.DataFrame(rows)
ra["rank_official"] = ra.official_2018_all_fields.rank(ascending=False).astype(int); ra["rank_provided"] = ra.provided_n2.rank(ascending=False).astype(int)
ra.to_csv(os.path.join(CACHE, "s4_a_official_vs_provided.csv"), index=False, encoding="utf-8-sig")
# 제공 KR 2018 전체 상위 10(KSIC 중복 제거 고유 출원)
t10 = kr18.groupby("n2").n.sum().sort_values(ascending=False).head(10)
res["a"] = dict(kr_2018_invention_unique_total=tot18, table=ra.to_dict("records"),
                provided_top10_2018=[(k, int(v)) for k, v in t10.items()])
print("a done", round(time.time() - t0), flush=True)

# ---- (b) 결측률·이름 변형 ----
mis = con.sql(f"""select cc, appl_year yr, count(*) n, sum(case when {FIRST} is null then 1 else 0 end) n_miss
  from read_parquet('{META}') where appl_year between 2005 and 2024 group by 1,2 order by 1,2""").df()
mis["miss_rate"] = mis.n_miss / mis.n
mis.to_csv(os.path.join(CACHE, "s4_b_missing_by_cc_year.csv"), index=False, encoding="utf-8-sig")
piv = mis.pivot(index="yr", columns="cc", values="miss_rate")
res["b_missing_rate"] = {c: {int(y): round(float(piv.at[y, c]), 4) for y in [2005, 2008, 2010, 2015, 2017, 2019, 2021, 2023]} for c in CCS}
# 결측의 KSIC 편중(P3 발명특허): KSIC별 결측률 범위
P3 = A[A.yr.between(2017, 2019) & A.is_inv]
mk = P3.assign(m=(P3.a2 < 0) * P3.n_rows).groupby(["cc", "ksic"]).agg(n=("n_rows", "sum"), m=("m", "sum"))
mk["r"] = mk.m / mk.n
res["b_missing_P3_by_ksic"] = {c: dict(overall=round(float(mk.loc[c].m.sum() / mk.loc[c].n.sum()), 4), min=round(float(mk.loc[c].r.min()), 4),
                                       max=round(float(mk.loc[c].r.max()), 4), max_ksic=str(mk.loc[c].r.idxmax())) for c in CCS}
# 병합률: 원문 이름 수 → N2 키 수
mg = names[names.raw.notna()].groupby("cc").agg(raw=("raw", "nunique"), n2=("id_n2", "nunique"))
mg["merge_rate"] = 1 - mg.n2 / mg.raw
res["b_merge"] = mg.round(4).to_dict("index")
# 삼성전자·LG디스플레이 변형 (국가별)
pats = {"samsung_elec": r"(?i)samsung\s*electronics|삼성전자|三星电子|サムスン\s*エレクトロニクス|三星電子",
        "samsung_display": r"(?i)samsung\s*display|삼성디스플레이|三星显示|サムスン\s*ディスプレイ",
        "lg_display": r"(?i)lg\s*display|엘지디스플레이|乐金显示|エルジー\s*ディスプレイ|ＬＧ\s*ディスプレイ"}
var = []
for key, rg in pats.items():
    s = names[names.raw.fillna("").str.contains(rg, regex=True)]
    for c, g in s.groupby("cc"):
        var.append(dict(firm=key, cc=c, n_raw=int(g.raw.nunique()), n_n2=int(g.id_n2.nunique()), docs=int(g.n.sum()),
                        top_n2_share=round(float(g.groupby("id_n2").n.sum().max() / g.n.sum()), 4),
                        top_raw=" ; ".join(g.sort_values("n", ascending=False).raw.head(4).astype(str))))
vv = pd.DataFrame(var); vv.to_csv(os.path.join(CACHE, "s4_b_name_variants.csv"), index=False, encoding="utf-8-sig")
res["b_variants"] = vv.drop(columns="top_raw").to_dict("records")
print("b done", round(time.time() - t0), flush=True)

# ---- (c) 출원인 코드 연결 (KR 등록특허 ↔ registry_full.applicant) ----
kr = con.sql(f"""select ksic, appl_year yr, rgst_no, appl_no, {FIRST} raw from read_parquet('{META}')
  where cc='KR' and substr(appl_no,1,2)='10'""").df()
kr["rgst_no"] = kr.rgst_no.astype("string")
db = sqlite3.connect("file:" + os.path.join(ROOT, r"수집파이프라인\registry_full.sqlite") + "?mode=ro", uri=True)
ap = pd.read_sql("select rgst_no, seq, applicantCd, applicantName from applicant", db)
pt = pd.read_sql("select rgst_no, applNo from patent", db)
db.close()
ap = ap.sort_values(["rgst_no", "seq"]).drop_duplicates("rgst_no")
ap["rgst_no"] = ap.rgst_no.astype(str)
pt["applNo"] = pt.applNo.astype(str).str.replace("-", "")
m_by_appl = pt.merge(ap, on="rgst_no")[["applNo", "applicantCd", "applicantName"]].drop_duplicates("applNo").set_index("applNo")
kr["acd"] = kr.appl_no.astype(str).map(m_by_appl.applicantCd)
kr["registered"] = kr.rgst_no.notna() & (kr.rgst_no.astype(str).str.len() > 3)
kr["linked"] = kr.acd.notna()
sub = kr[kr.registered]
def tvd(a, b):
    a = a / a.sum(); b = b / b.sum(); i = a.index.union(b.index)
    return round(float(0.5 * (a.reindex(i).fillna(0) - b.reindex(i).fillna(0)).abs().sum()), 4)
L = sub[sub.linked]; U = sub[~sub.linked]
res["c"] = dict(kr_registered_rows=int(len(sub)), link_rate_registered=round(float(sub.linked.mean()), 4),
                link_rate_unregistered=round(float(kr[~kr.registered].linked.mean()), 4),
                tvd_ksic_linked_vs_unlinked=tvd(L.ksic.value_counts(), U.ksic.value_counts()) if len(U) else None,
                tvd_year_linked_vs_unlinked=tvd(L.yr.value_counts(), U.yr.value_counts()) if len(U) else None,
                unlinked_year_dist=U.yr.value_counts().sort_index().tail(8).to_dict(),
                link_rate_by_period={p: round(float(sub[sub.yr.between(a, b)].linked.mean()), 4) for p, (a, b) in
                                     {"2005-07": (2005, 2007), "2008-10": (2008, 2010), "2015-17": (2015, 2017), "2017-19": (2017, 2019), "2020-22": (2020, 2022)}.items()})
# 코드 기준 1위와 N2 이름 기준 1위 (KR 등록특허, 출원 2017–19, KSIC별)
s3 = sub[sub.yr.between(2017, 2019) & sub.linked].copy()
s3["n2"] = [norm_n2(x) if isinstance(x, str) else None for x in s3.raw]
s3 = s3.drop_duplicates(["ksic", "appl_no"])
tc = s3.groupby(["ksic", "acd"]).size().rename("n").reset_index().sort_values(["ksic", "n"], ascending=[True, False]).drop_duplicates("ksic").set_index("ksic")
tn = s3.groupby(["ksic", "n2"]).size().rename("n").reset_index().sort_values(["ksic", "n"], ascending=[True, False]).drop_duplicates("ksic").set_index("ksic")
s3["acd_of_n2top"] = s3.n2.values == tn.n2.reindex(s3.ksic).values
code_of_top = s3[s3.acd_of_n2top].groupby("ksic").acd.agg(lambda x: x.value_counts().index[0])
same = (code_of_top.reindex(tc.index) == tc.acd)
tot = s3.groupby("ksic").size()
res["c_top1_code_vs_name"] = dict(n_ksic=int(len(tc)), same_top1=int(same.sum()), differ=sorted(tc.index[~same]),
                                  share_diff_max=round(float(((tc.n - tn.n.reindex(tc.index)).abs() / tot.reindex(tc.index)).max()), 4))
# 같은 이름(N2) 하나에 코드 여러 개 / 같은 코드에 N2 여러 개
nm_per_code = s3.groupby("acd").n2.nunique(); code_per_nm = s3.groupby("n2").acd.nunique()
res["c_mapping"] = dict(codes_with_multi_n2=int((nm_per_code > 1).sum()), n2_with_multi_codes=int((code_per_nm > 1).sum()),
                        docs_in_multi_code_n2=round(float(s3.n2.isin(code_per_nm.index[code_per_nm > 1]).mean()), 4))
print("c done", round(time.time() - t0), flush=True)

# ---- (d) 개명·합병: 연도별 이름 등장 ----
chk = {"기아자동차": "기아자동차", "기아(신)": r"^기아 ?주식회사|^기아주식회사", "삼성테크윈": "삼성테크윈", "한화테크윈": "한화테크윈",
       "한화에어로스페이스": "한화에어로스페이스", "한화정밀기계": "한화정밀기계", "한화세미텍": "한화세미텍", "엘지이노텍": "엘지이노텍",
       "에스케이하이닉스": r"에스케이하이닉스|에스케이 하이닉스", "주식회사 하이닉스반도체": "하이닉스반도체",
       "삼성모바일디스플레이": "삼성모바일디스플레이", "삼성에스디아이": "삼성에스디아이", "HD현대(구 현대중공업)": "현대중공업|에이치디현대"}
dd = con.sql(f"""select appl_year yr, {FIRST} raw, count(distinct appl_no) n from read_parquet('{META}') where cc='KR' group by 1,2""").df()
out = []
for k, rg in chk.items():
    s = dd[dd.raw.fillna("").str.contains(rg, regex=True)]
    if len(s):
        yy = s.groupby("yr").n.sum()
        out.append(dict(name=k, docs=int(yy.sum()), first_year=int(yy.index.min()), last_year=int(yy.index.max()),
                        by_period={p: int(yy[(yy.index >= a) & (yy.index <= b)].sum()) for p, (a, b) in
                                   {"<=2010": (1990, 2010), "2011-15": (2011, 2015), "2016-20": (2016, 2020), "2021-25": (2021, 2025)}.items()}))
    else:
        out.append(dict(name=k, docs=0))
res["d"] = out
# registry 현재 출원인명 대 제공 이름: 같은 출원번호에서 N2가 다른 비율 (출원 연도별)
kr2 = kr[kr.linked].copy()
kr2["nm_reg"] = kr2.appl_no.astype(str).map(m_by_appl.applicantName)
smp = kr2.drop_duplicates("appl_no")
smp = smp.assign(n2p=[norm_n2(x) if isinstance(x, str) else None for x in smp.raw], n2r=[norm_n2(x) if isinstance(x, str) else None for x in smp.nm_reg])
smp["diff"] = smp.n2p != smp.n2r
res["d_name_vs_registry"] = {p: round(float(smp[smp.yr.between(a, b)]["diff"].mean()), 4) for p, (a, b) in
                             {"2005-07": (2005, 2007), "2008-10": (2008, 2010), "2015-17": (2015, 2017), "2017-19": (2017, 2019), "2020-22": (2020, 2022)}.items()}
ex = smp[smp["diff"]].groupby(["raw", "nm_reg"]).size().sort_values(ascending=False).head(15)
res["d_top_diff_pairs"] = [(a, b, int(n)) for (a, b), n in ex.items()]
print("d done", round(time.time() - t0), flush=True)
json.dump(res, open(os.path.join(CACHE, "s4_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
