import sqlite3,sys
sys.stdout.reconfigure(encoding='utf-8')
DB=r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
con=sqlite3.connect('file:'+DB+'?mode=ro',uri=True)
q=lambda s: con.execute(s).fetchall()
for r in q("select ownerRgstCsName,count(*) from owner where ownerRgstCsName is not null and ownerRgstCsDate<'20190101' group by 1 order by 2 desc"): print(r)
print('patents without owner rows', q("select count(*), sum(substr(rgstDate,1,4)<'2019') from patent where rgst_no not in (select rgst_no from owner)"))
print(q("select substr(rgstDate,1,4),count(*) from patent where rgst_no not in (select rgst_no from owner) group by 1"))
print('applicantCd prefix'); [print(r) for r in q("select substr(applicantCd,1,1),applicantNatl,count(*) from applicant group by 1,2 order by 3 desc limit 25")]
print([r for r in q("select applicantName from applicant where substr(applicantCd,1,1)='2' limit 15")])
print([r for r in q("select applicantName from applicant where substr(applicantCd,1,1)='3' limit 15")])
print([r for r in q("select applicantName from applicant where substr(applicantCd,1,1)='5' limit 8")])
print([r for r in q("select applicantName from applicant where substr(applicantCd,1,1)='6' limit 8")])
print([r for r in q("select applicantName from applicant where substr(applicantCd,1,1)='7' limit 8")])
print('licenses', q("select count(*) from elsLcnsList where elsLcnsStartDate is not null"), q("select count(*) from noelsLcnsList where noelsLcnsStartDate is not null"))
print(q("select * from noelsLcnsList where noelsLcnsStartDate is not null limit 3"))
print('cofpr', q("select cofprNatl,count(*) from cofpr where cofprNatl is not null group by 1 order by 2 desc limit 8"))
print('pay sample', q("select * from pay where rgst_no='1012334040000'"))
