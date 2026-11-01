# C02-T2 s2: 법인 포트폴리오(business_rights, 2026 스냅샷)의 KR 특허 등록을 제공 KR과 출원번호로 대조 → 등록월별 제공 반영률
# 목적: 2024 등록 급감이 '등록 사실 누락(출원은 있으나 rgst_no 공란)'인지 '출원 자체 미수록'인지 '실제 감소'인지 구분.
import os, json, time, sys
import duckdb, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
t = time.time()
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
META = os.path.join(ROOT, r"분석\모델경쟁_20260927\_shared\meta\*.parquet").replace("\\", "/")
BR = os.path.join(ROOT, r"수집파이프라인\business_rights.sqlite").replace("\\", "/")
OUT = os.path.join(ROOT, r"분석\검증루프_20261012\C02\C02-T2\cache")
con = duckdb.connect(); con.sql("set threads=2"); con.sql("set memory_limit='3GB'")
con.sql("install sqlite; load sqlite")
con.sql(f"""create temp table kr as select appl_no, any_value(rgst_no) rgst_no, any_value(rgst_date) rgst_date, any_value(appl_year) appl_year,
             any_value(ksic) ksic, any_value(pub_date) pub_date
            from read_parquet('{META}') where cc='KR' group by appl_no""")
con.sql(f"""create temp table br as select distinct rgst_no, rgst_date, appl_no, appl_date from sqlite_scan('{BR}', 'right_list')
            where substr(rgst_no,1,2)='10' and rgst_date>='20190101'""")
res = {}
res['fmt'] = con.sql("select length(appl_no) l, count(*) from kr group by 1").fetchall() + [('br',)] + con.sql("select length(appl_no) l, count(*) from br group by 1").fetchall()
m = con.sql("""select substr(b.rgst_date,1,6) ym, count(*) n_br,
                 count(*) filter (where k.appl_no is not null and k.rgst_no is not null) in_prov_reg,
                 count(*) filter (where k.appl_no is not null and k.rgst_no is null) in_prov_noreg,
                 count(*) filter (where k.appl_no is not null and k.rgst_no is not null and substr(k.rgst_date,1,6)<>substr(b.rgst_date,1,6)) reg_month_mismatch
               from br b left join kr k on k.appl_no=b.appl_no group by 1 order by 1""").df()
m['share_in_prov'] = (m.in_prov_reg + m.in_prov_noreg) / m.n_br
m['share_noreg_among_prov'] = m.in_prov_noreg / (m.in_prov_reg + m.in_prov_noreg)
m.to_csv(os.path.join(OUT, 's2_br_month.csv'), index=False)
# 제공에 출원은 있으나 rgst_no 공란인 KR 특허(출원연도별), 그중 br에서 등록 확인된 수
res['prov_noreg_but_br_registered_by_ym_year'] = con.sql("""select substr(b.rgst_date,1,4) y, count(*) from br b join kr k on k.appl_no=b.appl_no where k.rgst_no is null group by 1 order by 1""").fetchall()
# 등록일 형식 점검: br 등록일 vs 제공 등록일 차이
res['date_diff_days'] = con.sql("""select quantile_cont(abs(date_diff('day', strptime(b.rgst_date,'%Y%m%d'), strptime(k.rgst_date,'%Y%m%d'))), [0.5,0.99]) from br b join kr k on k.appl_no=b.appl_no where k.rgst_no is not null""").fetchall()
con.close()
json.dump(res, open(os.path.join(OUT, 's2_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(res); print(m.to_string()); print(round(time.time() - t, 1))
