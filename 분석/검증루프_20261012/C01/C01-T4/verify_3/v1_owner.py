# verify_3: owner 사건 재계산 (읽기전용)
import sqlite3, sys, json
p=r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite'
out=r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C01\C01-T4\verify_3\v1_owner.txt'
con=sqlite3.connect('file:'+p+'?mode=ro',uri=True); c=con.cursor()
L=[]
def q(t,s): L.append('## '+t); [L.append(str(r)) for r in c.execute(s).fetchall()]
q('reason 2025+', "select ownerRgstCsName,count(*) from owner where ownerRgstCsDate>='2025' group by 1 order by 2 desc")
q('reason 2015-2024', "select ownerRgstCsName,count(*) from owner where ownerRgstCsDate>='2015' and ownerRgstCsDate<'2025' group by 1 order by 2 desc limit 30")
q('monthly 2024-2026 named', "select substr(ownerRgstCsDate,1,6),count(*), sum(case when ownerRgstCsName in ('양도','일부양도','지분양도','신탁계약에 의한 양도') then 1 else 0 end) from owner where ownerRgstCsName is not null and ownerRgstCsDate>='2024' group by 1 order by 1")
q('null reason sample', "select rgst_no,seq,ownerRgstCsDate,ownerName,ownerRgstCsReason,finalOwnerYn from owner where ownerRgstCsName is null and ownerRgstCsDate>='2025' limit 8")
q('null reason by year', "select substr(ownerRgstCsDate,1,4),count(*) from owner where ownerRgstCsName is null group by 1 order by 1 desc limit 6")
q('acquire-type by year (corp)', "select substr(ownerRgstCsDate,1,4),count(*),sum(case when ownerName like '%주식회사%' or ownerName like '%(주)%' or ownerName like '%㈜%' then 1 else 0 end), count(distinct rgst_no) from owner where ownerRgstCsName in ('양도','일부양도','지분양도','신탁계약에 의한 양도') and ownerRgstCsDate>='2012' group by 1 order by 1")
q('right max date', "select max(rgstCsDate) from right")
open(out,'w',encoding='utf-8').write('\n'.join(L))
