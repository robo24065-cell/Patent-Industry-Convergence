# C02-T2 s10: KR 등록 사실 누락 중 T1 최종 진입창(출원 2020–2023) 해당 건수와 공개일 보유(공개 경로로 레이블 포착 가능) 비율 (읽기 전용)
import os, sys, json, duckdb
sys.stdout.reconfigure(encoding='utf-8')
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
META = os.path.join(ROOT, r"분석\모델경쟁_20260927\_shared\meta\*.parquet").replace("\\", "/")
BR = os.path.join(ROOT, r"수집파이프라인\business_rights.sqlite").replace("\\", "/")
W = os.path.join(ROOT, r"분석\검증루프_20261012\C02\C02-T2")
con = duckdb.connect(); con.sql("set threads=2"); con.sql("set memory_limit='3GB'"); con.sql("install sqlite; load sqlite")
con.sql(f"create temp table kr as select appl_no, any_value(rgst_no) rgst_no, any_value(appl_year) ay, any_value(pub_date) pd from read_parquet('{META}') where cc='KR' and substr(appl_no,1,2)='10' group by appl_no")
con.sql(f"create temp table br as select appl_no, min(rgst_date) rd from sqlite_scan('{BR}', 'right_list') where substr(rgst_no,1,2)='10' group by appl_no")
q = con.sql("""select k.ay, count(*) n_missing_reg, sum((k.pd is not null and k.pd<>'')::int) n_with_pub,
   (select count(*) from kr k2 where k2.ay=k.ay and k2.rgst_no is not null and k2.rgst_no<>'') n_reg_present
   from kr k join br b using(appl_no) where (k.rgst_no is null or k.rgst_no='') and b.rd<='20250822' and k.ay between 2016 and 2024 group by 1 order by 1""").df()
print(q.to_string())
json.dump(q.to_dict('records'), open(os.path.join(W, 'cache', 's10_t1.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
