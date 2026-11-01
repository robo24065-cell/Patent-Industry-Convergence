# 반박 검증: P4 월별 건수 독립 재계산 (sqlite mode=ro, 월별 집계만)
import sqlite3, sys, collections
sys.stdout.reconfigure(encoding='utf-8')
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
con = sqlite3.connect('file:' + B + r'\수집파이프라인\registry_full.sqlite?mode=ro', uri=True)
print(con.execute("select sql from sqlite_master where name='owner'").fetchone()[0])
# 사유명 분포 (2024-2026) — 공백/변형 확인
rows = con.execute("select ownerRgstCsName, substr(ownerRgstCsDate,1,4) y, count(*) from owner where ownerRgstCsDate>='20220101' group by 1,2").fetchall()
d = collections.defaultdict(dict)
for n,y,c in rows: d[repr(n)][y]=c
for n in sorted(d, key=lambda k:-sum(d[k].values()))[:40]: print(n, d[n])
ACC=('양도','일부양도','지분양도','신탁계약에 의한 양도')
q = "select substr(ownerRgstCsDate,1,4), substr(ownerRgstCsDate,5,2), count(*) from owner where ownerRgstCsName in (?,?,?,?) and ownerRgstCsDate>='20150101' group by 1,2"
m = {(y,mm):c for y,mm,c in con.execute(q, ACC)}
def s(y,a=1,b=7): return sum(m.get((str(y),f'{i:02d}'),0) for i in range(a,b+1))
import statistics as st
print({y:s(y) for y in range(2015,2027)})
print('P4b', s(2026), s(2025), s(2026)/s(2025))
ma = st.median([m.get(('2026',f'{i:02d}'),0) for i in (5,6,7)]); mb = st.median([m.get(('2026',f'{i:02d}'),0) for i in (1,2,3,4)])
print('P4a', ma, mb, ma/mb)
base = (s(2022)+s(2023)+s(2024))/3; print('P4c', base, s(2026)/base)
print('monthly 2025-2026', {f'{y}-{mm}':c for (y,mm),c in sorted(m.items()) if y>='2025'})
# trimmed variant
q2 = q.replace("ownerRgstCsName in (?,?,?,?)", "trim(ownerRgstCsName) in (?,?,?,?)")
m2 = {(y,mm):c for y,mm,c in con.execute(q2, ACC)}
print('trim 2025/2026 jan-jul', sum(m2.get(('2025',f'{i:02d}'),0) for i in range(1,8)), sum(m2.get(('2026',f'{i:02d}'),0) for i in range(1,8)))
# 등록번호-날짜 고유 (공동권리자 중복 제거)
q3 = "select substr(ownerRgstCsDate,1,4), count(distinct rgst_no||ownerRgstCsDate) from owner where ownerRgstCsName in (?,?,?,?) and substr(ownerRgstCsDate,5,2) between '01' and '07' and ownerRgstCsDate>='20220101' group by 1"
print('distinct pat-date', con.execute(q3, ACC).fetchall())
print('max date', con.execute("select max(ownerRgstCsDate) from owner").fetchone())
