# s1_A: C03-A 해외(US·EP) 문서 → KR 문서 패밀리 연결표. 2025-01-01 이후 공개는 월별 건수만(SQL 집계), 행은 적재하지 않는다.
import sys, json, time
import duckdb, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
T = time.time()
B = r'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전'
W = B + '/분석/검증루프_20261012/C03/C03-T4'
META = B + '/분석/모델경쟁_20260927/_shared/meta/*.parquet'
KRP = B + '/공모전데이터/작업데이터/kr_patents.parquet'
DOCS = B + '/분석/검증루프_20261012/C02/C02-T4/cache/docs.parquet'
d = duckdb.connect(); d.execute("set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false")
log = {}
# C02 docs 재사용 점검
log['docs_rows'] = d.sql(f"select count(*) from read_parquet('{DOCS}')").fetchone()[0]
log['docs_firms_prefix1'] = d.sql(f"select count(distinct cd) from read_parquet('{DOCS}') where substr(cd,1,1)='1'").fetchone()[0]
assert log['docs_rows'] == 1008540 and log['docs_firms_prefix1'] == 40162
d.execute(f"create table kr as select 출원번호 kr_appl, 공개번호 pb, substr(등록번호,1,9) r9 from read_parquet('{KRP}')")
d.execute(f"""create table fo as select cc, appl_no, pub_date, prio_cc, family, ksic fk from read_parquet('{META}')
              where cc in ('US','EP')""")
log['fo_rows_by_cc_pub_lt2025'] = d.sql("select cc, count(*), avg((family like '%KR%')::int), avg((prio_cc like '%KR%')::int) from fo where pub_date<'20250101' group by 1 order by 1").fetchall()
d.execute("""create table tok as select distinct cc, appl_no, pub_date, fk, t from (
              select cc, appl_no, pub_date, fk, unnest(string_split(family,'|')) t from fo where family like '%KR%') where t like 'KR%'""")
d.execute("""create table tk as select *, case when regexp_full_match(t,'KR(10|20)[0-9]{11}') then 'P13'
                                         when regexp_full_match(t,'KR(19|20)[0-9]{9}') then 'P11'
                                         when regexp_full_match(t,'KR(10|20)[0-9]{7}') then 'R9' else 'other' end typ,
              case when regexp_full_match(t,'KR(10|20)[0-9]{11}') then substr(t,3)
                   when regexp_full_match(t,'KR(19|20)[0-9]{9}') then '10'||substr(t,3) end pub13,
              case when regexp_full_match(t,'KR(10|20)[0-9]{7}') then substr(t,3) end r9 from tok""")
log['token_types_lt2025'] = d.sql("select typ, count(*) from tk where pub_date<'20250101' group by 1 order by 1").fetchall()
# P13 토큰이 출원번호로도 맞는지(오독 진단)
log['P13_match_pubno_vs_applno_lt2025'] = d.sql("""select sum((pub13 in (select pb from kr where pb<>''))::int), sum((pub13 in (select kr_appl from kr))::int), count(*)
                                                  from tk where typ='P13' and pub_date<'20250101'""").fetchone()
d.execute("""create table lk as
   select tk.cc, tk.appl_no, tk.pub_date, tk.fk, kr.kr_appl, 'PUB' meth from tk join kr on tk.pub13=kr.pb where tk.typ in ('P13','P11') and kr.pb<>''
   union all
   select tk.cc, tk.appl_no, tk.pub_date, tk.fk, kr.kr_appl, 'REG' meth from tk join kr on tk.r9=kr.r9 where tk.typ='R9' and kr.r9<>''""")
d.execute(f"create table dc as select cd, appl_no, ksic kk, vis from read_parquet('{DOCS}') where substr(cd,1,1)='1'")
# 잠금 구간(2025-01 이후 공개): 월별 건수만
lock = d.sql("""select substr(lk.pub_date,1,6) ym, lk.cc, lk.meth, count(distinct lk.cc||lk.appl_no) n_foreign_linked,
                count(distinct case when dc.cd is not null then lk.cc||lk.appl_no end) n_foreign_linked_corp
                from lk left join dc on lk.kr_appl=dc.appl_no where lk.pub_date>='20240101' group by 1,2,3 order by 1,2,3""").df()
lock.to_csv(W + '/cache/s1_A_monthly_linked_2024_2026.csv', index=False, encoding='utf-8-sig')
allm = d.sql("""select substr(pub_date,1,6) ym, cc, count(*) n_all, sum((prio_cc like '%KR%')::int) n_prkr, sum((family like '%KR%')::int) n_famkr
                from fo where pub_date>='20240101' group by 1,2 order by 1,2""").df()
allm.to_csv(W + '/cache/s1_A_monthly_all_2024_2026.csv', index=False, encoding='utf-8-sig')
# 개발 행(2025 이전 공개)만 적재
L = d.sql("select * from lk where pub_date<'20250101'").df()
assert L.pub_date.max() < '20250101'
L.to_parquet(W + '/cache/A_links_lt2025.parquet', index=False)
F = d.sql("select cc, appl_no, pub_date, prio_cc, fk, (family like '%KR%') famkr from fo where pub_date<'20250101' and (prio_cc like '%KR%' or family like '%KR%')").df()
F.to_parquet(W + '/cache/A_foreign_krrel_lt2025.parquet', index=False)
log['links_lt2025_by_meth'] = L.groupby('meth').size().to_dict()
log['links_lt2025_label_agree_fk_kk'] = {}
D = d.sql("select appl_no, any_value(kk) kk from dc group by 1").df().set_index('appl_no').kk
L['kk'] = L.kr_appl.map(D)
for m_, g in L.groupby('meth'):
    gg = g.dropna(subset=['kk'])
    log['links_lt2025_label_agree_fk_kk'][m_] = [len(gg), round(float((gg.fk.astype(str) == gg.kk.astype(str)).mean()), 4)]
log['elapsed'] = round(time.time() - T, 1)
json.dump(log, open(W + '/cache/s1_A_log.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(log, ensure_ascii=False, default=str))
