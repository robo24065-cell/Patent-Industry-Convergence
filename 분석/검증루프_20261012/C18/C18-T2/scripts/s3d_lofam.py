# C18-T2 s3d: 표 3 '패밀리 재연결 US 392,048 / JP 216,762 / EP 150,230쌍'과 표 2 '같은 발명 라벨 일치 US 87.3%, JP 87.0%, EP 90.5%'를
# label_origin_R10/lo_fam.py 규칙 그대로 P0·V1·V1E 메타로 재계산 → cache/s3d_lofam.json (원본 res_fam_check.json과 대조)
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib18 import *
res = {'orig': json.load(open(os.path.join(LO, 'res_fam_check.json'), encoding='utf-8'))['links']}
for V in ('P0', 'V1', 'V1E'):
    M = META if V == 'P0' else f'{ddb(C)}/meta_{V}.parquet'
    con = con3()
    con.sql(f"""CREATE TABLE kr AS SELECT appl_no AS kr_appl_no, substr(rgst_no,1,9) AS r9, ksic AS kk FROM read_parquet('{M}') WHERE cc='KR' AND rgst_no IS NOT NULL AND length(rgst_no)>=9""")
    con.sql(f"""CREATE TABLE tok AS SELECT DISTINCT cc, appl_no, ksic AS fk, substr(t,3) AS r9 FROM (
            SELECT cc, appl_no, ksic, unnest(string_split(family,'|')) AS t FROM read_parquet('{M}')
            WHERE cc<>'KR' AND family LIKE '%KR%') WHERE regexp_full_match(t, 'KR(10|20)[0-9]{{7}}')""")
    con.sql("CREATE TABLE fr AS SELECT tok.cc, tok.appl_no, kr.kr_appl_no, tok.fk, kr.kk FROM tok JOIN kr USING (r9)")
    res[V] = con.sql("SELECT cc, count(*), count(DISTINCT appl_no), count(DISTINCT kr_appl_no), avg((fk=kk)::INT) FROM fr GROUP BY cc ORDER BY cc").fetchall()
    con.close()
jdump(res, 's3d_lofam.json')
for k, v in res.items(): out(k, v)
