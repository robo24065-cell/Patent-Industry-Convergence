# C02-T2 s3: 등록 사실 누락(rgst 공란)이 T3 KR 공개연도 계열(least(pub_date, rgst_date))에 주는 영향 + pub_date 의미 확인
import os, json, time, sys, sqlite3
import duckdb, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
t = time.time()
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
META = os.path.join(ROOT, r"분석\모델경쟁_20260927\_shared\meta\*.parquet").replace("\\", "/")
BR = os.path.join(ROOT, r"수집파이프라인\business_rights.sqlite").replace("\\", "/")
REG = os.path.join(ROOT, r"수집파이프라인\registry_full.sqlite").replace("\\", "/")
OUT = os.path.join(ROOT, r"분석\검증루프_20261012\C02\C02-T2\cache")
con = duckdb.connect(); con.sql("set threads=2"); con.sql("set memory_limit='3GB'"); con.sql("install sqlite; load sqlite")
con.sql(f"""create temp table kr as select appl_no, any_value(rgst_no) rgst_no, any_value(rgst_date) rgst_date, any_value(appl_year) appl_year,
             any_value(ksic) ksic, any_value(pub_date) pub_date, any_value(main_ipc) main_ipc
            from read_parquet('{META}') where cc='KR' group by appl_no""")
con.sql(f"""create temp table br as select appl_no, min(rgst_date) rgst_date from sqlite_scan('{BR}', 'right_list')
            where substr(rgst_no,1,2)='10' group by appl_no""")
con.sql(f"create temp table rg as select applNo appl_no, opnDate, pubDate, rgstDate from sqlite_scan('{REG}', 'patent')")
res = {}
# pub_date 의미: 등록특허에서 제공 pub_date = 등록원부 opnDate(공개) 인지 pubDate(등록공고) 인지
res['pub_semantics'] = con.sql("""select count(*) n, sum((k.pub_date=r.opnDate)::int) eq_opn, sum((k.pub_date=r.pubDate)::int) eq_regpub,
     sum((k.pub_date is null)::int) pub_null, sum((r.opnDate is null)::int) opn_null
   from kr k join rg r on r.appl_no=k.appl_no""").fetchall()
# KR 공개연도(T3 정의) 계열
py = con.sql("""select left(coalesce(least(pub_date, rgst_date), pub_date, rgst_date),4) y, count(*) n,
        count(*) filter (where main_ipc is not null) n_ipc from kr group by 1 order by 1""").df()
py.to_csv(os.path.join(OUT, 's3_kr_pubyear.csv'), index=False)
# 등록 누락 문서(제공 rgst 공란 & 법인 포트폴리오에서 등록 확인): 공개일 유무, 실제 첫 공개연도
miss = con.sql("""select left(b.rgst_date,6) rym, count(*) n, sum((k.pub_date is null)::int) no_pub,
        sum((k.pub_date is not null and k.pub_date < b.rgst_date)::int) pub_before_rgst
     from kr k join br b on b.appl_no=k.appl_no where k.rgst_no is null and b.rgst_date between '20231101' and '20250822' group by 1 order by 1""").df()
miss.to_csv(os.path.join(OUT, 's3_missing_reg_pub.csv'), index=False)
# 누락분의 'T3 공개연도'가 실제로 어디로 가는지: 실제 = least(pub_date, br.rgst_date), 제공 = pub_date(없으면 탈락)
mv = con.sql("""select left(least(coalesce(k.pub_date,'99999999'), b.rgst_date),4) true_y, coalesce(left(k.pub_date,4),'탈락') prov_y, count(*) n
     from kr k join br b on b.appl_no=k.appl_no where k.rgst_no is null and b.rgst_date <= '20250822' group by 1,2 order by 1,2""").df()
mv.to_csv(os.path.join(OUT, 's3_missing_move.csv'), index=False)
# 법인 커버리지(제공 등록특허 중 br 에 있는 비율) — 전체 추정 배율
res['br_cover_of_prov_reg_by_year'] = con.sql("""select left(k.rgst_date,4) y, count(*) n, sum((b.appl_no is not null)::int) in_br
     from kr k left join br b on b.appl_no=k.appl_no where k.rgst_no is not null and substr(k.rgst_no,1,2)='10' and k.rgst_date>='2019' group by 1 order by 1""").fetchall()
# 미등록·미공개 문서(날짜 없음) 출원연도별
res['no_date_by_appl_year'] = con.sql("""select appl_year, count(*) from kr where pub_date is null and rgst_date is null and appl_year>=2015 group by 1 order by 1""").fetchall()
con.close()
json.dump(res, open(os.path.join(OUT, 's3_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(res); print(py[py.y >= '2015'].to_string()); print(miss.to_string()); print(mv.to_string()); print(round(time.time() - t, 1))
