import sqlite3,sys
sys.stdout.reconfigure(encoding='utf-8')
DB=r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
con=sqlite3.connect('file:'+DB+'?mode=ro',uri=True)
q=lambda s: con.execute(s).fetchall()
# examples of patents with transfer events before 2019
ex=q("select rgst_no from owner where ownerRgstCsName is not null and ownerRgstCsDate<'20190101' limit 5")
for (r,) in ex:
    print('----',r, q(f"select rgstDate,applDate from patent where rgst_no='{r}'"))
    for row in q(f"select seq,finalOwnerYn,ownerName,ownerNatl,ownerRgstCsDate,ownerRgstCsName,ownerRgstCsReason from owner where rgst_no='{r}' order by seq"): print(row)
    for row in q(f"select seq,rgstCsDate,rgstCsName,rgstCsReason from right where rgst_no='{r}' and rgstCsDate<'20190101' order by seq"): print('  R',row)
    for row in q(f"select seq,applicantName,applicantNatl,applicantCd from applicant where rgst_no='{r}' order by seq"): print('  A',row)
# seq0 owner: date == rgstDate?
print(q("select sum(o.ownerRgstCsDate=p.rgstDate), count(*), sum(o.ownerRgstCsName is not null) from owner o join patent p using(rgst_no) where o.seq=0"))
# owner rows with null name but date != rgstDate (co-owners at registration?)
print(q("select count(*) from owner o join patent p using(rgst_no) where o.ownerRgstCsName is null and o.ownerRgstCsDate<>p.rgstDate"))
print(q("select count(*) from owner o join patent p using(rgst_no) where o.ownerRgstCsName is not null and o.ownerRgstCsDate<=p.rgstDate"))
print('null dates', q("select count(*) from owner where ownerRgstCsName is not null and ownerRgstCsDate is null"))
# transfer event years < 2019
for r in q("select substr(ownerRgstCsDate,1,4) y,count(*),count(distinct rgst_no) from owner where ownerRgstCsName is not null and ownerRgstCsDate<'20190101' group by 1"): print(r)
