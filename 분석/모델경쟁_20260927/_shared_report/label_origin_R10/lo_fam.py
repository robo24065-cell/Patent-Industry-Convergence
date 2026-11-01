# -*- coding: utf-8 -*-
r"""R10 — 패밀리 연결 재구축(P0 보정). R8 fam_link는 해외 문서 패밀리 필드의 'KR10YYYYNNNNNNN'(KR 공개번호 형식이
KR 출원번호와 같은 13자리)을 출원번호로 읽어 US·EP·JP 연결이 틀린 것으로 보인다(예시 확인: 서로 무관한 발명).
여기서는 해외 문서 패밀리 필드의 KR 등록번호 토큰(KR + 10/20 + 7자리)을 KR 문서 등록번호 앞 9자리와 맞춘다.
CN은 패밀리 필드가 KR 출원번호(우선권)라 기존 방식도 맞을 수 있어, 두 방식을 모두 기록한다.
출력: cache/fam_reg.parquet (cc, appl_no, kr_appl_no, fk, title_f), res_fam_check.json, 표본 대조 fam_sample.csv"""
import os, sys, json, time
sys.dont_write_bytecode = True
import duckdb, pandas as pd
MC = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927"
OUT = os.path.join(MC, r"_shared_report\label_origin_R10")
META = os.path.join(MC, r"_shared\meta\*.parquet").replace("\\", "/")
_T = time.time()
con = duckdb.connect(); con.sql("SET threads=2; SET memory_limit='3GB'; SET preserve_insertion_order=false")
con.sql(f"""CREATE TABLE kr AS SELECT appl_no AS kr_appl_no, substr(rgst_no,1,9) AS r9, ksic AS kk, title AS kt, applicant AS ka
            FROM read_parquet('{META}') WHERE cc='KR' AND rgst_no IS NOT NULL AND length(rgst_no)>=9""")
con.sql(f"""CREATE TABLE tok AS SELECT DISTINCT cc, appl_no, ksic AS fk, title AS ft, applicant AS fa, substr(t,3) AS r9 FROM (
            SELECT cc, appl_no, ksic, title, applicant, unnest(string_split(family,'|')) AS t FROM read_parquet('{META}')
            WHERE cc<>'KR' AND family LIKE '%KR%') WHERE regexp_full_match(t, 'KR(10|20)[0-9]{{7}}')""")
con.sql("CREATE TABLE fr AS SELECT tok.cc, tok.appl_no, kr.kr_appl_no, tok.fk, kr.kk, tok.ft, kr.kt, tok.fa, kr.ka FROM tok JOIN kr USING (r9)")
res = {"links": con.sql("SELECT cc, count(*), count(DISTINCT appl_no), count(DISTINCT kr_appl_no), avg((fk=kk)::INT) FROM fr GROUP BY cc ORDER BY cc").fetchall()}
# 한 해외 문서가 여러 KR 문서에 붙는 경우(패밀리 확장) 비중
res["multi_kr_per_foreign"] = con.sql("SELECT avg((n>1)::INT) FROM (SELECT cc, appl_no, count(DISTINCT kr_appl_no) n FROM fr GROUP BY 1,2)").fetchall()
# 명칭 대조 표본(사람이 읽는 용도)
con.sql(f"COPY (SELECT cc, kk, fk, left(kt,50) kt, left(ft,70) ft, left(ka,20) ka, left(fa,30) fa FROM fr USING SAMPLE 60 ROWS) TO '{os.path.join(OUT,'fam_sample.csv').replace(chr(92),'/')}' (HEADER, DELIMITER ',')")
con.sql(f"COPY (SELECT cc, appl_no, kr_appl_no, fk FROM fr) TO '{os.path.join(OUT,'cache','fam_reg.parquet').replace(chr(92),'/')}' (FORMAT parquet)")
# 기존 R8 방식(출원번호 13자리 토큰) 국가별 일치율 재계산 — 비교용
con.sql(f"""CREATE TABLE old AS SELECT DISTINCT cc, appl_no, fk, substr(t,3) AS kr_appl_no FROM (
            SELECT cc, appl_no, ksic AS fk, unnest(string_split(family,'|')) AS t FROM read_parquet('{META}') WHERE cc<>'KR' AND family LIKE '%KR%')
            WHERE regexp_full_match(t, 'KR(10|20)[0-9]{{11}}')""")
res["old_r8_method"] = con.sql("""SELECT o.cc, count(*), avg((o.fk=k.kk)::INT) FROM old o JOIN
      (SELECT appl_no kr_appl_no, ksic kk FROM read_parquet('""" + META + """') WHERE cc='KR') k USING (kr_appl_no) GROUP BY 1 ORDER BY 1""").fetchall()
json.dump(res, open(os.path.join(OUT, "res_fam_check.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(res, round(time.time() - _T))
