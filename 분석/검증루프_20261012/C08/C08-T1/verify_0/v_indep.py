# -*- coding: utf-8 -*-
# verify_0: 독립 재계산 (주 경로 cache·common.py 미사용). duckdb SQL로 원자료에서 직접 집계, 이름 정규화는 SQL 정규식(별도 규칙).
# 대상: S1의 KR 1위 수, ρ(P2 RTA vs RCA), C2620 사례 + 주 판정 반박 시도. 쓰기: verify_0 폴더만.
import sys, os, json
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
import duckdb, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
MC = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927"
META = (MC + r"\_shared\meta\*.parquet").replace("\\", "/")
EXP = MC + r"\_shared_report\fusion_advantage\map5\m4_exports_ksic_5rep.csv"
con = duckdb.connect(); con.sql("set memory_limit='3GB'; set threads=2; set enable_progress_bar=false; set preserve_insertion_order=false")
# 별도 정규화 V: 소문자, 법인 접미어 일부 제거, 비문자 제거
con.sql(r"""create macro vnorm(s) as regexp_replace(regexp_replace(lower(trim(s)),
  '(주식회사|\(주\)|㈜|株式会社|有限公司|股份|co\.?|ltd\.?|inc\.?|corporation|corp\.?|gmbh|limited|llc|kabushiki kaisha|company)', '', 'g'), '[^\p{L}\p{N}]', '', 'g')""")
con.sql(f"""create temp table d as select ksic::varchar ksic, cc, appl_year yr, appl_no,
  case when cc='KR' then substr(appl_no,1,2)='10' when cc='CN' then length(appl_no) in (12,14) and substr(appl_no,5,1) in ('1','8')
       when cc='JP' then length(appl_no) in (8,17) else true end inv,
  nullif(vnorm(split_part(coalesce(applicant,''),'|',1)),'') a,
  nullif(trim(split_part(coalesce(applicant,''),'|',1)),'') raw
  from read_parquet('{META}') where appl_year between 2005 and 2019""")
CCS = ["KR", "US", "JP", "CN", "EP"]


def rank(x):
    return pd.Series(x).rank().values


def spearman(x, y):
    a, b = rank(x), rank(y); return float(np.corrcoef(a, b)[0, 1])


def counts(y0, y1, inv=True, uniq=False, drop_top=0, only_kr=False, merge=None):
    """KSIC×cc 문서 수; drop_top=k이면 KSIC×cc마다 상위 k 출원인 제거"""
    ex = "count(distinct appl_no)" if uniq else "count(*)"
    a = "a" if merge is None else f"case when a in ({','.join(repr(m) for m in merge[1:])}) then '{merge[0]}' else a end"
    q = f"""with b as (select ksic, cc, {a} a, {ex} n from d where yr between {y0} and {y1} {'and inv' if inv else ''} group by 1,2,3),
      r as (select *, row_number() over (partition by ksic, cc order by case when a is null then 1 else 0 end, n desc, a) rk from b)
      select ksic, cc, sum(n) n from r where not (a is not null and rk <= {drop_top} {"and cc='KR'" if only_kr else ''}) group by 1,2"""
    return con.sql(q).df().pivot(index="ksic", columns="cc", values="n").reindex(columns=CCS).fillna(0)


def rta(n):
    s = n / n.sum(); return s.div(s.mean(1), axis=0)


out = {}
for lab, kw in {"S0": {}, "S1": dict(drop_top=1), "S1_uniq": dict(drop_top=1, uniq=True), "S0_uniq": dict(uniq=True),
                "S1_top2": dict(drop_top=2), "S1_KRonly": dict(drop_top=1, only_kr=True)}.items():
    r3 = rta(counts(2017, 2019, **kw)); lead = r3.idxmax(1)
    r2 = rta(counts(2015, 2017, **kw)).KR
    ek = pd.read_csv(EXP, dtype={"k5": str}); v = ek[ek.year.isin([2015, 2016, 2017])].groupby(["k5", "rep"]).v.sum().unstack()
    s = v / v.sum(); rca = s.div(s.mean(1), axis=0).KR
    dd = pd.concat([r2, rca], axis=1, keys=["r", "c"]).dropna()
    rng = np.random.default_rng(7); bs = []
    for _ in range(1000):
        i = rng.integers(0, len(dd), len(dd)); bs.append(spearman(dd.r.values[i], dd.c.values[i]))
    out[lab] = dict(kr_leader=int((lead == "KR").sum()), kr_rta_gt1=int((r3.KR > 1).sum()), kr_leaders=sorted(r3.index[lead == "KR"]),
                    oled_26212={c: round(float(r3.at["26212", c]), 3) for c in CCS}, lead_26212=lead["26212"],
                    rho=round(spearman(dd.r.values, dd.c.values), 3), rho_ci=[round(float(np.nanpercentile(bs, 2.5)), 3), round(float(np.nanpercentile(bs, 97.5)), 3)])
    print(lab, out[lab], flush=True)

# C2620 사례: 2008–09 → 2016–17, 전 유형 고유 출원, KR, 1위는 두 창 합산
def gw(drop_top=0, merge=None, top_window="both"):
    a = "a" if merge is None else f"case when a in ({','.join(repr(m) for m in merge[1:])}) then '{merge[0]}' else a end"
    yrs = "yr in (2008,2009,2016,2017)" if top_window == "both" else "yr in (2016,2017)" if top_window == "late" else "yr in (2008,2009)"
    q = f"""with b as (select ksic, {a} a, count(distinct case when yr in (2008,2009) then appl_no end) n0, count(distinct case when yr in (2016,2017) then appl_no end) n1,
               count(distinct case when {yrs} then appl_no end) nt
             from d where cc='KR' and ksic in ('26211','26212','26111') and yr in (2008,2009,2016,2017) group by 1,2),
      r as (select *, row_number() over (partition by ksic order by case when a is null then 1 else 0 end, nt desc, a) rk from b)
      select ksic, sum(n0) n0, sum(n1) n1 from r where not (a is not null and rk <= {drop_top}) group by 1"""
    x = con.sql(q).df().set_index("ksic")
    return {k: round(100 * float((np.log(x.n1[k] / 2 + 1) - np.log(x.n0[k] / 2 + 1)) / 8), 1) for k in x.index}


SM = ["삼성디스플레이", "삼성모바일디스플레이", "삼성sdi", "삼성에스디아이"]
out["C2620"] = {"S0": gw(0), "S1": gw(1), "S1_top2": gw(2), "S1_merge_samsung_display": gw(1, merge=SM),
                "S1_top_by_late_window": gw(1, top_window="late"), "S1_top_by_early_window": gw(1, top_window="early")}
print("C2620", out["C2620"], flush=True)
# KR OLED P3 상위 출원인 몫 (이 경로의 정규화)
top = con.sql("""select a, count(*) n from d where cc='KR' and inv and ksic='26212' and yr between 2017 and 2019 group by 1 order by 2 desc limit 6""").df()
tot = con.sql("select count(*) from d where cc='KR' and inv and ksic='26212' and yr between 2017 and 2019").fetchone()[0]
out["KR_26212_P3_top"] = [(r.a, int(r.n), round(r.n / tot, 3)) for r in top.itertuples()]
topu = con.sql("""select a, count(*) n from d where cc='US' and inv and ksic='26212' and yr between 2017 and 2019 group by 1 order by 2 desc limit 4""").df()
totu = con.sql("select count(*) from d where cc='US' and inv and ksic='26212' and yr between 2017 and 2019").fetchone()[0]
out["US_26212_P3_top"] = [(r.a, int(r.n), round(r.n / totu, 3)) for r in topu.itertuples()]
# 반박 시도: OLED KR 1위 상실이 '한국 기업 1곳'의 효과인지 — US·EP에서 한국계 출원인(삼성·LG) 제외 시 (KR 기원 해외출원 효과)
kor = "(a like '%samsung%' or a like '%lgdisplay%' or a like '%lgelectronics%' or a like '%lgchem%' or a like '%삼성%' or a like '%엘지%' or a like '%三星%' or a like '%乐金%' or a like '%サムスン%' or a like '%ｌｇ%')"
n_all = con.sql(f"select ksic, cc, count(*) n, sum(case when {kor} then 1 else 0 end) nk from d where inv and yr between 2017 and 2019 group by 1,2").df()
pv = n_all.pivot(index="ksic", columns="cc", values="n").reindex(columns=CCS).fillna(0)
pk = n_all.pivot(index="ksic", columns="cc", values="nk").reindex(columns=CCS).fillna(0)
pk2 = pk.copy(); pk2["KR"] = 0
r_nokor_abroad = rta(pv - pk2)
out["refute_korean_firms_abroad_removed"] = dict(oled={c: round(float(r_nokor_abroad.at["26212", c]), 3) for c in CCS},
                                                 kr_leader=int((r_nokor_abroad.idxmax(1) == "KR").sum()),
                                                 korean_share_of_US_26212=round(float(pk.at["26212", "US"] / pv.at["26212", "US"]), 3))
json.dump(out, open(os.path.join(HERE, "v_indep_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
