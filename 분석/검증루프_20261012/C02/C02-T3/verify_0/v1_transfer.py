import sqlite3, time
t=time.time()
db=r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
c=sqlite3.connect("file:"+db.replace("\\","/")+"?mode=ro", uri=True)
q=lambda s: c.execute(s).fetchall()
N=656360
print("ownerRgstCsName all", q("select ownerRgstCsName, count(*), count(distinct rgst_no) from owner group by 1 order by 2 desc"))
c.execute("""create temp table o as select rgst_no, count(*) n,
  sum(case when ownerRgstCsName is not null and trim(ownerRgstCsName)<>'' then 1 else 0 end) nr,
  sum(case when finalOwnerYn='N' then 1 else 0 end) nn,
  count(distinct ownerName) nnames
  from owner group by rgst_no""")
r=q("select count(*), sum(n>1), sum(nr>0), sum(n>1 and nr=0), sum(n>1 and nr=0 and nn=0), sum(nr>0 or nn>0), sum(n=1 and nr>0) from o")[0]
print("owner patents, n>1, has_reason, n>1&no_reason, n>1&no_reason&noN, reason_or_N, n=1&reason", r)
print("pct n>1", r[1]/N*100, "pct reason", r[2]/N*100, "pct reason_or_N", r[5]/N*100)
# n>1 & no reason: are rows all finalOwnerYn=Y with distinct names? (co-owners)
print("n>1 noreason noN: nnames dist", q("select nnames=n, count(*) from o where n>1 and nr=0 and nn=0 group by 1"))
print("n>1 noreason: nn dist", q("select nn>0, count(*) from o where n>1 and nr=0 group by 1"))
# right table event names
print("right rgstCsName", q("select rgstCsName, count(*), count(distinct rgst_no) from right group by 1 order by 2 desc limit 40"))
print("time", time.time()-t)
