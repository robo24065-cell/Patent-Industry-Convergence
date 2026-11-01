# -*- coding: utf-8 -*-
# 테이블별 날짜 범위·행 수(메타데이터). 창 안 사건 수는 세지 않는다.
import sqlite3, json
DB = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True); cur = con.cursor()
q = {
 'pay_rows': "select count(*) from pay",
 'pay_payDate_max': "select max(payDate) from pay where payDate glob '[12][0-9][0-9][0-9]*'",
 'pay_payDate_min': "select min(payDate) from pay where payDate glob '[12][0-9][0-9][0-9]*'",
 'pay_top_dates': "select payDate from pay where payDate glob '[12][0-9][0-9][0-9]*' order by payDate desc limit 5",
 'right_rows': "select count(*) from right",
 'right_max': "select max(rgstCsDate) from right where rgstCsDate glob '[12][0-9][0-9][0-9]*'",
 'patent_rows': "select count(*) from patent",
 'patent_rgst_max': "select max(rgstDate) from patent where rgstDate glob '[12][0-9]*'",
 'unknown_ts_max': "select max(ts) from unknown_keys",
 'unknown_ts_min': "select min(ts) from unknown_keys",
 'pay_date_fmt': "select payDate from pay limit 3",
}
o = {k: cur.execute(v).fetchall() for k, v in q.items()}
# 월별 납부 행 수(2025-01 이후) — 납부 기록이 언제까지 차 있는지(수집 마감) 확인용. 사건 아님.
o['pay_monthly_2024on'] = cur.execute("select substr(payDate,1,6) m, count(*) from pay where payDate >= '20240101' group by m order by m").fetchall()
print(json.dumps(o, ensure_ascii=False))
