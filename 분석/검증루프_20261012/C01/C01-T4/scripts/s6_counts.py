# s6: 등록원부 2026-08 스냅샷의 연도별 '건수'만(분쟁 예고등록, 권리이전) — 잠금 가능 구간 존재 확인용. 레이블·모형 없음.
import os, sqlite3, json
import pandas as pd
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
OUTD = os.path.join(ROOT, r"분석\검증루프_20261012\C01\C01-T4\cache")
con = sqlite3.connect("file:" + os.path.join(ROOT, r"수집파이프라인\registry_full.sqlite") + "?mode=ro", uri=True)
res = {}
q = """select substr(rgstCsDate,1,4) y, rgstCsName, count(*) from right
       where rgstCsName in ('무효(심판)예고등록','권리범위(심판)예고등록') and rgstCsDate>='2019' group by 1,2 order by 1,2"""
res["dispute_by_year"] = pd.read_sql(q, con).values.tolist()
q2 = """select substr(rgstCsDate,1,6) ym, count(*) from right where rgstCsName in ('무효(심판)예고등록','권리범위(심판)예고등록') and rgstCsDate>='2025' group by 1 order by 1"""
res["dispute_monthly_2025_26"] = pd.read_sql(q2, con).values.tolist()
q3 = """select substr(ownerRgstCsDate,1,4) y, count(*) n, sum((ownerName like '%주식회사%' or ownerName like '%(주)%' or ownerName like '%㈜%')::int) n_corp
        from owner where ownerRgstCsName is not null and ownerRgstCsDate>='2015' group by 1 order by 1"""
try:
    res["transfer_by_year"] = pd.read_sql(q3, con).values.tolist()
except Exception:
    q3 = q3.replace("::int", "")
    res["transfer_by_year"] = pd.read_sql("""select substr(ownerRgstCsDate,1,4) y, count(*) n,
        sum(case when ownerName like '%주식회사%' or ownerName like '%(주)%' or ownerName like '%㈜%' then 1 else 0 end) n_corp
        from owner where ownerRgstCsName is not null and ownerRgstCsDate>='2015' group by 1 order by 1""", con).values.tolist()
res["owner_event_max_date"] = pd.read_sql("select max(ownerRgstCsDate) from owner", con).iloc[0, 0]
con.close()
json.dump(res, open(os.path.join(OUTD, "s6_counts.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
