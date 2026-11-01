# C13-T4 s0: 구조 탐색(결과 아님). 공개일 < 2025-01-01 문서만 집계한다. 2025 이후 행은 건수도 보지 않는다.
import duckdb, json, sys
sys.stdout.reconfigure(encoding='utf-8')
B = r'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전'
META = B + '/분석/모델경쟁_20260927/_shared/meta/*.parquet'
OUT = B + '/분석/검증루프_20261012/C13/C13-T4/cache/s0_probe.json'
con = duckdb.connect(); con.execute('set threads=2'); con.execute("set memory_limit='3GB'")
res = {}
# (1) 문서당 KSIC 수 분포(나라별, 공개연도 2015-2024)
q = f"""
with d as (select cc, appl_no, min(substr(pub_date,1,4)) py, count(distinct ksic) nk
           from read_parquet('{META}') where pub_date is not null and pub_date < '20250101' and pub_date >= '20150101'
           group by cc, appl_no)
select cc, count(*) n, avg((nk>=2)::int) multi_rate, max(nk) maxk from d group by cc order by cc"""
res['multi_ksic'] = con.execute(q).fetchdf().to_dict('records')
# (2) 공동출원인(구분자 |) 비율, 나라별
q = f"""
with d as (select cc, appl_no, any_value(applicant) a from read_parquet('{META}')
           where pub_date >= '20150101' and pub_date < '20250101' group by cc, appl_no)
select cc, count(*) n, avg((a like '%|%')::int) multi_app, avg((a is null or a='')::int) miss_app from d group by cc order by cc"""
res['multi_app'] = con.execute(q).fetchdf().to_dict('records')
json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
