# -*- coding: utf-8 -*-
"""verify_0 (a): N7 모집단·레이블을 sqlite SQL 조인 경로로 재구성(pandas 판 s1과 다른 경로) → n7_label_rebuild.csv와 대조."""
import sqlite3, sys, io, json
import pandas as pd, pyarrow.parquet as pq
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
SH = ROOT + r"\분석\모델경쟁_20260927\_shared\ext"
OUT = ROOT + r"\분석\검증루프_20261012\C06\C06-T3"
con = sqlite3.connect(":memory:", uri=True)
con.execute("ATTACH DATABASE ? AS reg", ("file:" + ROOT + r"\수집파이프라인\registry_full.sqlite?mode=ro",))
con.execute("ATTACH DATABASE ? AS br", ("file:" + ROOT + r"\수집파이프라인\business_rights.sqlite?mode=ro",))
# CORP_APPLICANT 원문을 그대로 적재(숫자 추출은 SQL에서)
rows = []
with open(ROOT + r"\공모전데이터\외부데이터\Corporate_20260720\CORP_APPLICANT.txt", encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("¶")
        if len(p) >= 5: rows.append((p[0].strip(), p[4]))
con.execute("create table ca(cd text, bizraw text)"); con.executemany("insert into ca values(?,?)", rows)
con.execute("create table ca10 as select cd, replace(replace(trim(bizraw),'-',''),' ','') bz from ca")
con.execute("delete from ca10 where length(bz)<>10 or bz glob '*[^0-9]*'")
st = pd.read_parquet(SH + r"\r3nts_status.parquet", columns=['bizno', 'registered', 'end_dt'])
st['end'] = st.end_dt.dt.strftime('%Y%m%d'); st['registered'] = st.registered.astype(int)
st[['bizno', 'registered', 'end']].to_sql('nts', con, index=False)
nt = pq.read_table(SH + r"\ntis_project_raw.parquet", columns=['year', 'corp_rgst_no']).to_pandas()
nt['year'] = nt.year.astype(int); nt[['year', 'corp_rgst_no']].to_sql('ntis', con, index=False)
con.execute("""create table nfirst as select bz, min(year) fy from (
   select replace(replace(replace(corp_rgst_no,'-',''),' ',''),'.','') bz, year from ntis where year>=2010 and corp_rgst_no is not null)
   where length(bz)=10 and bz not glob '*[^0-9]*' group by bz""")
con.execute("""create table popcd as select a.applicantCd cd, count(distinct a.rgst_no) n_reg from reg.applicant a join reg.patent p on a.rgst_no=p.rgst_no
  where substr(a.applicantCd,1,1)='1' and cast(p.rgstDate as integer) <= 20221231 group by a.applicantCd""")
con.execute("create index i1 on popcd(cd)"); con.execute("create index i2 on ca10(cd)"); con.execute("create index i3 on nts(bizno)"); con.execute("create index i4 on nfirst(bz)")
con.execute("create table need as select cd from popcd where cd not in (select cd from ca10)"); con.execute("create index i5 on need(cd)")
con.execute("""create table rlbz as select r.applicant_cd cd, replace(r.business_no,'-','') bz from br.right_list r join need n on n.cd=r.applicant_cd
   where length(replace(r.business_no,'-',''))=10""")
con.execute("""create table bzmap as select cd, bz, 'CA' route from ca10
   union all select cd, min(bz), 'RL' from rlbz group by cd having count(distinct bz)=1""")
con.execute("create index i6 on bzmap(cd)")
q = """select p.cd, m.bz, m.route, f.fy,
   case when f.fy in (2023,2024) then 1 else 0 end y, case when f.fy=2023 then 1 else 0 end y1
from popcd p join bzmap m on p.cd=m.cd join nts s on s.bizno=m.bz
left join nfirst f on f.bz=m.bz
where s.registered=1 and (s.end is null or s.end > '20221231') and (f.fy is null or f.fy > 2022)"""
d = pd.read_sql(q, con)
o = {'n_all_routes': len(d), 'route': d.route.value_counts().to_dict(), 'pos': int(d.y.sum()), 'pos_y1': int(d.y1.sum())}
lab = pd.read_csv(OUT + r"\n7_label_rebuild.csv", dtype={'cd': str, 'bizno': str})
fr = pd.read_parquet(ROOT + r"\분석\모델경쟁_20260927\N7_특허기업_정부RnD첫수행\final_test_pred_r5.parquet"); fr['cd'] = fr.applicantCd.astype(str)
m = fr[['cd', 'y', 'y1']].merge(d, on='cd', how='outer', suffixes=('_off', '_sql'), indicator=True)
o['vs_official'] = m._merge.value_counts().to_dict()
b = m[m._merge == 'both']
o['y_agree'] = float((b.y_off == b.y_sql).mean()); o['y1_agree'] = float((b.y1_off == b.y1_sql).mean())
o['sql_only_routes'] = m[m._merge == 'right_only'].route.value_counts().to_dict()
o['sql_only_examples'] = m[m._merge == 'right_only'].head(10)[['cd', 'bz', 'route', 'fy']].astype(str).values.tolist()
m2 = lab[['cd', 'y_rb', 'y1_rb']].merge(d[['cd', 'y', 'y1']], on='cd', how='inner')
o['vs_pandas_rebuild'] = dict(common=len(m2), y_agree=float((m2.y_rb == m2.y).mean()), y1_agree=float((m2.y1_rb == m2.y1).mean()))
json.dump(o, open(OUT + r"\verify_0\v_n7_sqlite_out.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(o, ensure_ascii=False, indent=1, default=str))
