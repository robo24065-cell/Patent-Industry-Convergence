# -*- coding: utf-8 -*-
"""verify_0 V-A: (A) SAB 비율·빼도 skill을 duckdb SQL만으로 독립 재계산(작업 코드·cache 미사용; 원천 base_rows·meta·registry·동결 예측에서 직접).
반박 시도: 등호(≥ 대 >), D≤0 양성, 등록 연결 키(meta rgst_no 대 registry patent.applNo=출원번호)."""
import os, sys, json, sqlite3, time
import duckdb, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
R3 = ROOT + r"\분석\모델경쟁_20260927\T3_기술셀부상예측\R3"
BR = (R3 + r"\cache_pub_final\base_rows.parquet").replace('\\', '/')
META = (ROOT + r"\분석\모델경쟁_20260927\_shared\meta\*.parquet").replace('\\', '/')
PRED = (R3 + r"\preds\FINAL_R3_t0_2021.parquet").replace('\\', '/')
RF = ROOT + r"\수집파이프라인\registry_full.sqlite"
OUTD = os.path.dirname(os.path.abspath(__file__))
t = time.time()
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2; set enable_progress_bar=false")
sq = sqlite3.connect(f"file:{RF}?mode=ro", uri=True)
app = pd.read_sql("select rgst_no, seq, applicantCd, applicantName from applicant", sq)
pat = pd.read_sql("select rgst_no, applNo from patent", sq)
con.register('rf_app', app); con.register('rf_pat', pat)
con.execute(r"""create macro mg(s) as (case
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{1,4}/') then substr(replace(s,' ',''),1,4)||' '||cast(cast(regexp_extract(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z]([0-9]{1,4})/',1) as int) as varchar)
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{3,6}$') then substr(replace(s,' ',''),1,4)||' '||cast(cast(substr(replace(s,' ',''),5,length(replace(s,' ',''))-6) as int) as varchar)
  else null end)""")
con.execute(r"""create macro nm(s) as regexp_replace(
  replace(replace(replace(replace(replace(
    lower(replace(replace(replace(replace(replace(replace(replace(coalesce(s,''),'&amp;','&'),'&lt;','<'),'&gt;','>'),'&quot;','"'),'&#39;',''''),'&apos;',''''),'&nbsp;',chr(160))),
  '주식회사',''),'(주)',''),'㈜',''),'(株)',''),'（주）',''),
  '[\s\.,·\-\(\)\[\]''"“”‘’&/]+','','g')""")
con.execute(f"""create table P as select cell, yb, p, cast(p_base as double) pb from read_parquet('{PRED}')""")
con.execute(f"""create table docs as select distinct ksic||'|'||mg(ipc_raw) cell, appl_no, year, app1
   from read_parquet('{BR}') where cc='KR' and year between 2019 and 2024 and mg(ipc_raw) is not null""")
con.execute("create table d2 as select d.*, case when year<=2021 then 'past' else 'fut' end w from docs d join P using(cell)")
con.execute("create table D as select cell, sum(w='fut')::int - sum(w='past')::int D, sum(w='past') kw, sum(w='fut') kf from d2 group by cell")
res = {}
res['n_cells_P'] = con.execute("select count(*), sum(yb) from P").fetchall()[0]
res['pos_D_le0'] = con.execute("select count(*) from P left join D using(cell) where yb=1 and coalesce(D,0)<=0").fetchone()[0]
res['dup_cell_appl_year'] = con.execute("select count(*) - count(distinct (cell, appl_no, year)) from d2").fetchone()[0]
res['app1_multi_per_doc'] = con.execute("select count(*) from (select cell, appl_no, year, count(distinct app1) c from d2 group by all) where c>1").fetchone()[0]
# 등록 연결: (K1) meta rgst_no, (K2) registry applNo = 출원번호
con.execute(f"""create table k1 as select appl_no, any_value(nullif(trim(rgst_no),'')) rgst_no from read_parquet('{META}') where cc='KR' group by appl_no""")
con.execute("create table k2 as select applNo appl_no, min(rgst_no) rgst_no from rf_pat group by applNo")
con.execute("create table A as select rgst_no, seq, coalesce(nullif(applicantCd,''), 'NOCD:'||nm(applicantName)) code, nm(applicantName) n1 from rf_app")
AUC = """with r as (select yb, (rank() over (order by {p}) + (count(*) over (partition by {p}) - 1)/2.0) rk from P where cell not in (select cell from EX)),
 s as (select sum(yb) n1, count(*)-sum(yb) n0, sum(case when yb=1 then rk else 0 end) sr from r) select (sr - n1*(n1+1)/2.0)/(n1*n0) from s"""
def skill_excl(excl_sql):
    con.execute("create or replace temp table EX as " + excl_sql)
    a = con.execute(AUC.format(p='p')).fetchone()[0]; b = con.execute(AUC.format(p='pb')).fetchone()[0]
    return (a - b) / (1 - b), con.execute("select count(*) from EX").fetchone()[0]
def sab(credit_table, op='>='):
    q = f"""with g as (select cell, ky, sum(case when w='fut' then 1 else -1 end) dl from {credit_table} group by cell, ky),
             mx as (select cell, max(dl) mx from g group by cell)
             select P.cell from P join D using(cell) left join mx using(cell) where yb=1 and D>0 and coalesce(mx,0) {op} 0.5*D"""
    return q
# (i) 이름
con.execute("create table cr_i as select distinct cell, appl_no, year, w, 'N:'||nm(app1) ky from d2")
for op in ['>=', '>']:
    sk, n = skill_excl(sab('cr_i', op)); res[f'i_{op}'] = dict(n_sab=n, rate=n / 504, skill_excl=sk)
for kn in ['k1', 'k2']:
    con.execute(f"create or replace table dk as select d2.*, {kn}.rgst_no from d2 left join {kn} using(appl_no)")
    res[f'reg_share_{kn}'] = con.execute("select avg((rgst_no is not null)::int) from dk").fetchone()[0]
    # (ii) 문서별 코드: 이름 일치 코드(seq 최소) 없으면 seq 최소; 이름→코드: 등록 문서(공개 2016–24 전체) 배정 코드의 최빈(몫≥0.5)
    con.execute(f"""create or replace table alld as select distinct appl_no, app1 from read_parquet('{BR}') where cc='KR' and year between 2016 and 2024 and mg(ipc_raw) is not null""")
    con.execute(f"""create or replace table alld2 as select a.appl_no, nm(a.app1) kname, {kn}.rgst_no from alld a left join {kn} using(appl_no)""")
    con.execute("""create or replace table codeof as select appl_no, kname, code from (
        select x.appl_no, x.kname, A.code, row_number() over (partition by x.appl_no order by (A.n1 = x.kname) desc, A.seq) rn
        from alld2 x join A using(rgst_no)) where rn=1""")
    con.execute("""create or replace table n2c as select kname, code from (
        select kname, code, n, n*1.0/sum(n) over (partition by kname) sh, row_number() over (partition by kname order by n desc, code) rn
        from (select kname, code, count(*) n from codeof group by all)) where rn=1 and sh>=0.5""")
    con.execute("""create or replace table cr_iii as
        select distinct dk.cell, dk.appl_no, dk.year, dk.w, 'C:'||A.code ky from dk join A using(rgst_no)
        union
        select distinct dk.cell, dk.appl_no, dk.year, dk.w, coalesce('C:'||n2c.code, 'N:'||nm(dk.app1)) ky from dk left join n2c on n2c.kname = nm(dk.app1) where dk.rgst_no is null""")
    con.execute("""create or replace table cr_ii as
        select distinct dk.cell, dk.appl_no, dk.year, dk.w, 'C:'||c.code ky from dk join codeof c using(appl_no) where dk.rgst_no is not null
        union
        select distinct dk.cell, dk.appl_no, dk.year, dk.w, coalesce('C:'||n2c.code, 'N:'||nm(dk.app1)) ky from dk left join n2c on n2c.kname = nm(dk.app1) where dk.rgst_no is null""")
    for tb in ['cr_ii', 'cr_iii']:
        for op in ['>=', '>']:
            sk, n = skill_excl(sab(tb, op)); res[f'{tb[3:]}_{kn}_{op}'] = dict(n_sab=n, rate=n / 504, skill_excl=sk)
res['sec'] = round(time.time() - t, 1)
json.dump(res, open(os.path.join(OUTD, 'vA_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
for k, v in res.items(): print(k, v)
