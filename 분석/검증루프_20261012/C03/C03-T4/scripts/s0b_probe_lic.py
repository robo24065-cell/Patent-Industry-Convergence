# s0b: 실시권 표 건수(연도별 count, 잠금 구간은 월별 count만)
import sys, sqlite3
sys.stdout.reconfigure(encoding='utf-8')
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
con = sqlite3.connect('file:' + B + r'\수집파이프라인\registry_full.sqlite?mode=ro', uri=True)
for t, p in [('elsLcnsList', 'elsLcns'), ('noelsLcnsList', 'noelsLcns')]:
    print(t, 'nonnull name', con.execute(f'select count(*) from {t} where {p}PsnNm is not null').fetchone(),
          'distinct rgst', con.execute(f'select count(distinct rgst_no) from {t} where {p}PsnNm is not null').fetchone())
    print(' RgstCsNm', con.execute(f'select {p}RgstCsNm, count(*) from {t} where {p}PsnNm is not null group by 1 order by 2 desc limit 8').fetchall())
    print(' start len', con.execute(f'select length({p}StartDate), count(*) from {t} where {p}PsnNm is not null group by 1').fetchall())
    print(' by year (start):', con.execute(f"select substr({p}StartDate,1,4) y, count(*) from {t} where {p}PsnNm is not null group by 1 order by 1").fetchall())
    print(' sample', con.execute(f"select {p}StartDate, {p}EndDate, {p}RgstCsNm, {p}RkNo, {p}RgnLcns from {t} where {p}PsnNm is not null and {p}StartDate < '20200101' limit 3").fetchall())
