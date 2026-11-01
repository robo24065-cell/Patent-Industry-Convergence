# -*- coding: utf-8 -*-
"""C01-T3 s00: 등록원부 right 사건명·사유 코드표(전 기간 빈도)만 본다. 시험 레이블·점수는 보지 않는다.
목적: 권리범위확인심판의 적극/소극 판정 규칙을 PREREG에 쓰기 전에 어떤 사유 문자열이 있는지 확인.
출력: s00_codebook.txt (사유 문자열 빈도)"""
import sqlite3, os
import pandas as pd
OUT = os.path.dirname(os.path.abspath(__file__))
REG = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
c = sqlite3.connect(f"file:{REG}?mode=ro", uri=True)
a = pd.read_sql("select rgstCsName, count(*) n from right where rgstCsName like '%예고%' or rgstCsName like '%심결%' or rgstCsName like '%확정%' or rgstCsName like '%말소%' group by 1 order by 2 desc", c)
b = pd.read_sql("select rgstCsName, rgstCsReason, count(*) n from right where rgstCsName like '%확정%' group by 1,2 order by 3 desc", c)
d = pd.read_sql("select rgstCsName, rgstCsReason, count(*) n from right where rgstCsName like '%말소%' and rgstCsName not in ('소멸등록','권리의 말소등록') group by 1,2 order by 3 desc limit 40", c)
pp = pd.read_sql("select rgstCsName, count(*) n, sum(rgstCsPupos is not null and rgstCsPupos<>'') n_pupos from right group by 1 order by 2 desc limit 60", c)
with open(os.path.join(OUT, "s00_codebook.txt"), "w", encoding="utf-8") as f:
    f.write("== 사건명(예고·심결·확정·말소 포함) 빈도\n" + a.to_string() + "\n\n")
    f.write("== 확정 사건 사유 빈도\n" + b.to_string() + "\n\n")
    f.write("== 말소 관련 사유(상위 40)\n" + d.to_string() + "\n\n")
    f.write("== 사건명별 rgstCsPupos 채움 수(상위 60)\n" + pp.to_string() + "\n")
print("ok")
