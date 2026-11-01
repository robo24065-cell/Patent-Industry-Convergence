# -*- coding: utf-8 -*-
"""C09-T3 반박 검증(r1): 공개연도 기준 출원인 결측·US 발명자 이름·KSIC 치우침 주장을
트랙 base_rows 가 아니라 공용 원천 _shared/meta/*.parquet 에서 직접 다시 계산한다.
읽기 전용. 출력은 이 폴더(verify_1)에만 r1_* 로.
"""
import os, sys, json, time, duckdb
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))  # 분석
M = os.path.join(ROOT, '모델경쟁_20260927', '_shared', 'meta', '*.parquet').replace('\\', '/')
t = time.time()
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false")
con.execute(f"set temp_directory='{os.path.join(HERE, 'r1_tmp').replace(chr(92), '/')}'")
# 행 단위: 공개일 = 두 날짜 중 이른 것(문자열 연도 비교 대신 정수 연도 min)
con.execute(f"""create temp table r as select cast(cc as varchar) cc, appl_no, appl_year::int ay, cast(ksic as varchar) ksic,
   case when pub_date is null or length(pub_date)<4 then null else try_cast(substr(pub_date,1,4) as int) end pyr,
   case when rgst_date is null or length(rgst_date)<4 then null else try_cast(substr(rgst_date,1,4) as int) end ryr,
   applicant, trim(split_part(coalesce(applicant,''),'|',1)) a1
 from read_parquet('{M}') where appl_year >= 2000""")
con.execute("""create temp table r2 as select *, case when pyr is null then ryr when ryr is null then pyr else least(pyr,ryr) end py from r""")
print('rows', con.sql("select count(*) from r2").fetchone(), round(time.time()-t), flush=True)
# 문서 단위
con.execute("""create temp table d as select cc, appl_no, min(ay) ay, min(py) py, count(distinct a1) n_a1_var,
   bool_and(a1='') miss_all, bool_or(a1='') miss_any, bool_and(applicant is null) rawnull_all, any_value(a1) a1
 from r2 group by cc, appl_no""")
con.execute("delete from d where py is null or py<2000 or py>2024")
R = {}
R['docs_by_cc'] = con.sql("select cc, count(*) n, sum((n_a1_var>1)::int) a1_inconsistent, sum((miss_all<>miss_any)::int) miss_inconsistent from d group by cc order by cc").df().to_dict('records')
pub = con.sql("select cc, py as year, count(*) n, avg(miss_all::int) miss, avg(rawnull_all::int) rawnull from d group by 1,2 order by 1,2").df()
pub.to_csv(os.path.join(HERE, 'r1_miss_pub.csv'), index=False)
app = con.sql("select cc, ay as year, count(*) n, avg(miss_all::int) miss from d where ay<=2024 group by 1,2 order by 1,2").df()
app.to_csv(os.path.join(HERE, 'r1_miss_appl.csv'), index=False)
# US 이름 형태: 독립 법인 표지(간단한 대문자 토큰) + 'LAST, FIRST' 형태 비율
con.execute(r"""create temp table us as select py, ay, a1,
   regexp_matches(upper(a1), '(\bINC\b|\bCORP|\bLTD\b|\bLLC\b|\bCO\.|\bCOMPANY\b|\bGMBH\b|KABUSHIKI|KAISHA|UNIVERSIT|\bAG\b|\bS\.A\.|\bN\.V\.|\bB\.V\.|\bLIMITED\b|INSTITUTE|\bPLC\b|\bLLP\b|\bL\.P\.)') corp2,
   regexp_matches(a1, '^[^,]+, [^,]+$') commaname
 from d where cc='US' and a1<>''""")
usn = con.sql("select py as year, count(*) n, avg(corp2::int) corp2, avg(commaname::int) comma_form from us group by 1 order by 1").df()
usn.to_csv(os.path.join(HERE, 'r1_us_nameform_pub.csv'), index=False)
usa = con.sql("select ay as year, count(*) n, avg(corp2::int) corp2 from us where ay between 2008 and 2016 group by 1 order by 1").df()
R['us_corp2_by_appl'] = usa.to_dict('records')
R['us_sample_2008'] = con.sql("select a1 from us where py=2008 using sample 15 rows (reservoir, 7)").df().a1.tolist()
R['us_sample_2018'] = con.sql("select a1 from us where py=2018 using sample 15 rows (reservoir, 7)").df().a1.tolist()
# KSIC 치우침: 문서×KSIC 쌍(원래 방식) TVD + 출원연도 층화 TVD + 무작위 반분 기준선
con.execute("""create temp table dk as select distinct r2.cc, r2.appl_no, r2.ksic, d.ay, d.miss_all miss from r2 join d using(cc, appl_no)""")
def tvd(df):
    pv = df.pivot_table(index='ksic', columns='miss', values='n', aggfunc='sum').fillna(0)
    if True not in pv or False not in pv: return None
    return float(0.5 * (pv[True] / pv[True].sum() - pv[False] / pv[False].sum()).abs().sum())
for cc, a, b in [('CN', 2005, 2009), ('JP', 2019, 2020), ('US', 2000, 2004)]:
    df = con.sql(f"select ay, ksic, miss, count(*) n from dk where cc='{cc}' and ay between {a} and {b} group by 1,2,3").df()
    out = dict(pooled=tvd(df), miss_docs=int(df[df.miss].n.sum()), nonmiss_docs=int(df[~df.miss].n.sum()))
    strat = {}; wsum = 0; wtot = 0
    for y, g in df.groupby('ay'):
        v = tvd(g); mr = g[g.miss].n.sum() / g.n.sum()
        strat[int(y)] = dict(tvd=v, miss_rate=round(float(mr), 3), n=int(g.n.sum()))
        if v is not None:
            w = min(g[g.miss].n.sum(), g[~g.miss].n.sum()); wsum += v * w; wtot += w
    out['by_year'] = strat; out['stratified_weighted'] = wsum / wtot if wtot else None
    # 연도 구성만으로 생기는 TVD: 결측 쪽 연도 분포로 비결측 쪽을 재가중
    ym = df[df.miss].groupby('ay').n.sum(); yn = df[~df.miss].groupby('ay').n.sum()
    nm = df[~df.miss].copy(); nm['w'] = nm.ay.map((ym / ym.sum()) / (yn / yn.sum())); nm['n'] = nm.n * nm.w
    rw = pd.concat([df[df.miss], nm[['ay', 'ksic', 'miss', 'n']]])
    out['year_reweighted_tvd'] = tvd(rw)
    # 무작위 반분 기준선(같은 크기)
    base = con.sql(f"select ksic, (hash(appl_no) % 2 = 0) miss, count(*) n from dk where cc='{cc}' and ay between {a} and {b} group by 1,2").df()
    out['random_split_tvd'] = tvd(base)
    R[f'tvd_{cc}_{a}_{b}'] = out
    print(cc, json.dumps({k: v for k, v in out.items() if k != 'by_year'}), flush=True)
# 공개연도 기준 JP 2021–22 결측의 KSIC 치우침(최종 원점 창)
df = con.sql("select py as ay, ksic, miss, count(*) n from dk join (select cc, appl_no, py from d) using(cc, appl_no) where cc='JP' and py between 2021 and 2022 group by 1,2,3").df()
R['tvd_JP_pub2021_22'] = dict(pooled=tvd(df), by_year={int(y): tvd(g) for y, g in df.groupby('ay')})
json.dump(R, open(os.path.join(HERE, 'r1_results.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print('done', round(time.time()-t), flush=True)
