# s0: 2025 이전 owner/right 사건 범주 확인(읽기 전용). 잠금창 행은 SQL에서 제외.
import sqlite3, sys
sys.stdout.reconfigure(encoding='utf-8')
p=r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite'
c=sqlite3.connect('file:'+p+'?mode=ro',uri=True).cursor()
q=lambda s: c.execute(s).fetchall()
print('OWNER names', q("select ownerRgstCsName,count(*) from owner where ownerRgstCsDate<'20250101' group by 1 order by 2 desc limit 30"))
print('RIGHT names', q("select rgstCsName,count(*) from right where rgstCsDate<'20250101' group by 1 order by 2 desc limit 45"))
print('OWNER reason', q("select ownerRgstCsReason,count(*) from owner where ownerRgstCsDate<'20250101' group by 1 order by 2 desc limit 12"))
print('RIGHT 말소/경정 purpose', q("select rgstCsName, rgstCsPupos,count(*) from right where rgstCsDate<'20250101' and (rgstCsName like '%말소%' or rgstCsName like '%경정%' or rgstCsName like '%정정%' or rgstCsName like '%변경%') group by 1,2 order by 3 desc limit 25"))
print('seq>0 null name sample', q("select rgst_no,seq,finalOwnerYn,ownerName,ownerNatl,ownerRgstCsDate from owner where seq>0 and ownerRgstCsName is null and ownerRgstCsDate<'20250101' limit 5"))
print('seq>0 null name n by reason', q("select count(*) from owner where seq>0 and ownerRgstCsName is null and ownerRgstCsDate<'20250101'"))
print('owner natl', q("select ownerNatl,count(*) from owner where ownerRgstCsName in ('양도','일부양도','지분양도','신탁계약에 의한 양도') and ownerRgstCsDate<'20250101' group by 1 order by 2 desc limit 8"))
