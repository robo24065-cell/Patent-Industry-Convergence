import os
CACHE=os.environ.get("T3_CACHE","cache"); MAXYEAR=int(os.environ.get("T3_MAXYEAR","2018")); T0MAX=int(os.environ.get("T3_T0MAX","2015"))
os.makedirs(CACHE,exist_ok=True)
# 추가 피처: 출원인 포트폴리오 모멘텀(KR 출원인의 전체 KR 출원 증가율), 대형 출원인 비중, 1위 출원인 비중,
#           IPC 서브클래스 모멘텀, 공동분류 이웃 셀 모멘텀.  모두 t0 이하 연도만 사용.
import duckdb, time, sys
t=time.time()
con=duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false")
src=open('build_panel.py',encoding='utf-8').read()
macro=src[src.index('con.execute(r"""create macro'):src.index('con.execute(rf"""create view b')]
exec(macro)
con.execute(rf"""create temp table a as select ksic, mgn(ipc_raw) mg, cc, year, appl_no, any_value(app1) app1, any_value(ipc_all) ipc_all
 from read_parquet('{CACHE}/base_rows.parquet') where mgn(ipc_raw) is not null group by all""")
con.execute("create temp table kr as select * from a where cc='KR'")
# 출원인별 연도별 KR 출원수(전 셀 통합, 출원 단위 dedupe)
con.execute("create temp table ay as select app1, year, count(distinct appl_no) n from kr where app1 is not null and app1<>'' group by all")
# 이웃: (ksic, mg) - mg2 공동분류 출원수 (연도별, 전 국가)
con.execute(r"""create temp table nb as select ksic, mg, mgn(x) mg2, year, count(*) w from (select ksic, mg, year, unnest(string_split(ipc_all,'|')) x from a where ipc_all is not null) where mgn(x) is not null and mgn(x)<>mg group by all""")
con.execute("create temp table cy as select ksic, mg, cc, year, count(*) n from a group by all")
out=[]
for t0 in range(2005,T0MAX+1):
    q=f"""
    with am as (select app1, sum(case when year between {t0-2} and {t0} then n else 0 end) w, sum(case when year between {t0-5} and {t0-3} then n else 0 end) p from ay where year between {t0-5} and {t0} group by app1),
    c as (select k.ksic, k.mg, k.appl_no, k.app1, am.w, am.p from kr k left join am using(app1) where k.year between {t0-2} and {t0}),
    top as (select ksic, mg, max(cnt)*1.0/sum(cnt) top1_sh from (select ksic, mg, app1, count(*) cnt from c group by all) group by all)
    select c.ksic, c.mg, {t0} t0,
       avg(ln((w+1)/(p+1))) app_mom, avg(ln(w+1)) app_size, avg(case when w>=100 then 1 else 0 end) big_sh, any_value(top.top1_sh) top1_sh
    from c left join top using(ksic,mg) group by c.ksic, c.mg"""
    df=con.sql(q).df()
    # 이웃 모멘텀: 창 내 공동분류 가중치로 이웃 mg(같은 ksic)의 KR/전체 모멘텀 가중평균
    q2=f"""
    with w as (select ksic, mg, mg2, sum(w) w from nb where year between {t0-2} and {t0} group by all),
    g as (select ksic, mg,
            ln((sum(case when year between {t0-2} and {t0} and cc='KR' then n else 0 end)+1)/(sum(case when year between {t0-5} and {t0-3} and cc='KR' then n else 0 end)+1)) kmom,
            ln((sum(case when year between {t0-2} and {t0} then n else 0 end)+1)/(sum(case when year between {t0-5} and {t0-3} then n else 0 end)+1)) tmom
          from cy where year between {t0-5} and {t0} group by all),
    gs as (select ksic, left(mg,4) sub,
            ln((sum(case when year between {t0-2} and {t0} and cc='KR' then n else 0 end)+1)/(sum(case when year between {t0-5} and {t0-3} and cc='KR' then n else 0 end)+1)) sub_kmom,
            ln((sum(case when year between {t0-2} and {t0} then n else 0 end)+1)/(sum(case when year between {t0-5} and {t0-3} then n else 0 end)+1)) sub_tmom
          from cy where year between {t0-5} and {t0} group by all)
    select w.ksic, w.mg, sum(w.w*g.kmom)/sum(w.w) nb_kmom, sum(w.w*g.tmom)/sum(w.w) nb_tmom, ln(sum(w.w)+1) nb_w, count(*) nb_deg
    from w join g on w.ksic=g.ksic and w.mg2=g.mg group by w.ksic, w.mg"""
    nbdf=con.sql(q2).df()
    sub=con.sql(q2.split('select w.ksic')[0].replace('with w as','with w0 as')+" select * from gs").df() if False else None
    q3=f"""select ksic, left(mg,4) sub,
            ln((sum(case when year between {t0-2} and {t0} and cc='KR' then n else 0 end)+1)/(sum(case when year between {t0-5} and {t0-3} and cc='KR' then n else 0 end)+1)) sub_kmom,
            ln((sum(case when year between {t0-2} and {t0} and cc<>'KR' then n else 0 end)+1)/(sum(case when year between {t0-5} and {t0-3} and cc<>'KR' then n else 0 end)+1)) sub_fmom,
            ln((sum(case when year between {t0-2} and {t0} and cc='KR' then n else 0 end)+1)/(sum(case when year between {t0-2} and {t0} then n else 0 end)+1)) sub_krsh
          from cy where year between {t0-5} and {t0} group by all"""
    sdf=con.sql(q3).df()
    df=df.merge(nbdf,on=['ksic','mg'],how='outer'); df['t0']=t0
    df['sub']=df.mg.str[:4]
    df=df.merge(sdf,on=['ksic','sub'],how='left')
    out.append(df); print(t0,len(df),round(time.time()-t,1),flush=True)
import pandas as pd
pd.concat(out,ignore_index=True).to_parquet(f'{CACHE}/feat2.parquet')
