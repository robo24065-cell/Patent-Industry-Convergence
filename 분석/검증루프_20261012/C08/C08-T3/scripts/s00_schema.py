# -*- coding: utf-8 -*-
# 등록원부 스키마·테이블별 최대 날짜(메타데이터만; 결과 건수 집계 안 함)
import sqlite3, json, sys
DB = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
cur = con.cursor()
out = {}
for (t,) in cur.execute("select name from sqlite_master where type='table'").fetchall():
    cols = [r[1] for r in cur.execute(f"pragma table_info('{t}')").fetchall()]
    out[t] = dict(cols=cols)
print(json.dumps(out, ensure_ascii=False, indent=0)[:6000])
