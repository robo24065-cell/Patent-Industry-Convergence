# -*- coding: utf-8 -*-
"""KR 출원인 구성(제공 meta, 첫 출원인): (ksic, 출원연도, 출원인) 고유 출원 수 + 출원인 유형(대학·연구기관·기업·개인) 규칙 분류.
출원인명은 출원 시 기재값(현재 권리자 아님). 출력 cache/kr_ap.parquet"""
import duckdb
from n1lib import CACHE, META
con = duckdb.connect(); con.execute("SET memory_limit='3GB'; SET threads=2")
con.execute(f"""COPY (
 WITH a AS (SELECT ksic, CAST(appl_year AS INT) y, trim(split_part(applicant,'|',1)) ap, appl_no FROM read_parquet('{META}')
            WHERE cc='KR' AND appl_year BETWEEN 1995 AND 2021 AND applicant IS NOT NULL)
 SELECT ksic, y, ap,
   CASE WHEN regexp_matches(ap, '대학|산학협력|학교법인|University|UNIVERSITY|College') THEN 'U'
        WHEN regexp_matches(ap, '연구원|연구소|연구재단|Institute|INSTITUTE|진흥원|공사$|공단') THEN 'I'
        WHEN regexp_matches(ap, '주식회사|\\(주\\)|㈜|유한회사|회사|Co\\.|CO\\.|Ltd|LTD|Inc|INC|Corp|CORP|GmbH|AG$|S\\.A|LLC|Limited|KAISHA|Kaisha|주$|인코포레이티드|가부시키|가부시끼|코포레이션|게엠베하|리미티드|엘엘씨|에스\.에이|아게$|비\.브이|컴퍼니|카이샤|인크|코오퍼레이티브|에스에이$|엔\.브이|에이비$|에스\.피\.에이') THEN 'C'
        ELSE 'P' END typ,
   COUNT(DISTINCT appl_no) n FROM a GROUP BY 1,2,3,4) TO '{CACHE}/kr_ap.parquet' (FORMAT parquet)""")
print(con.sql(f"select typ, count(*) aps, sum(n) n from '{CACHE}/kr_ap.parquet' group by 1").df())
print(con.sql(f"select ap, typ, sum(n) n from '{CACHE}/kr_ap.parquet' group by 1,2 order by 3 desc limit 15").df())
