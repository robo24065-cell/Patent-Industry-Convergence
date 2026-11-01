# s0: 등록원부 스키마·사건 유형 건수 탐색(잠금 구간은 연도·월별 count만)
import sys, sqlite3
sys.stdout.reconfigure(encoding='utf-8')
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
con = sqlite3.connect('file:' + B + r'\수집파이프라인\registry_full.sqlite?mode=ro', uri=True)
tabs = [r[0] for r in con.execute("select name from sqlite_master where type='table'")]
print(tabs)
for t in ['elsLcnsList', 'noelsLcnsList', 'right', 'cofpr']:
    if t in tabs:
        print(t, [r[1] for r in con.execute(f'pragma table_info("{t}")')], con.execute(f'select count(*) from "{t}"').fetchone())
        print(con.execute(f'select * from "{t}" limit 2').fetchall())
print('right types (count, min, max date):')
for r in con.execute('select rgstCsName, count(*), min(rgstCsDate), max(rgstCsDate) from "right" group by 1 order by 2 desc'):
    print(r)
