# s0: 등록원부 구조 확인(읽기 전용). 결과창 레이블은 만들지 않는다.
import sqlite3, io, os
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
OUTD = os.path.join(ROOT, r"분석\검증루프_20261012\C01\C01-T4\cache")
o = io.open(os.path.join(OUTD, "s0_probe.txt"), "w", encoding="utf-8")
con = sqlite3.connect("file:" + os.path.join(ROOT, r"수집파이프라인\registry_full.sqlite") + "?mode=ro", uri=True)
cur = con.cursor()
for t in ["applicant", "owner", "patent", "pay", "right"]:
    o.write(str(cur.execute(f"select sql from sqlite_master where name='{t}'").fetchall()) + "\n")
o.write("applicant sample " + str(cur.execute("select * from applicant limit 3").fetchall()) + "\n")
o.write("owner sample " + str(cur.execute("select * from owner limit 4").fetchall()) + "\n")
o.write("== right 소멸/회복 사유\n")
for r in cur.execute("select rgstCsName, rgstCsReason, count(*) from right where rgstCsName like '%소멸%' or rgstCsName like '%회복%' group by 1,2 order by 3 desc limit 25").fetchall():
    o.write(str(r) + "\n")
o.write("== owner 사건명\n")
for r in cur.execute("select ownerRgstCsName, count(*) from owner group by 1 order by 2 desc limit 15").fetchall():
    o.write(str(r) + "\n")
o.write("patent n/distinct/ksic null " + str(cur.execute("select count(*), count(distinct rgst_no), sum(ksic is null) from patent").fetchall()) + "\n")
o.write("applicant code prefix " + str(cur.execute("select substr(applicantCd,1,1), count(*), count(distinct rgst_no) from applicant group by 1").fetchall()) + "\n")
o.write("applicant per patent " + str(cur.execute("select k, count(*) from (select rgst_no, count(*) k from applicant group by 1) group by 1 order by 1 limit 6").fetchall()) + "\n")
o.write("pay rows / patents with pay " + str(cur.execute("select count(*), count(distinct rgst_no) from pay").fetchall()) + "\n")
o.close()
