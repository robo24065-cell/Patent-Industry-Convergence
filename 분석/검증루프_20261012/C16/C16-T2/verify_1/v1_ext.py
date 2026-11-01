import sys, sqlite3, json
sys.stdout.reconfigure(encoding='utf-8')
db=sys.argv[1]
c=sqlite3.connect(f"file:{db}?mode=ro", uri=True)
out={}
out['ext_events']=c.execute("select rgstCsName, count(*), min(rgstCsDate), max(rgstCsDate) from right where rgstCsName like '%연장%' group by rgstCsName").fetchall()
# patents whose current expiry exceeds applDate+20y
q="""select count(*) from patent where cndrtExptnDate is not null and length(cndrtExptnDate)=8 and applDate is not null and length(applDate)=8
 and cast(cndrtExptnDate as int) > cast(applDate as int)+200000+1"""
out['n_exp_gt_ad20']=c.execute(q).fetchone()[0]
for L in ['20220101','20221231']:
    q=f"""select count(*) from patent where length(cndrtExptnDate)=8 and length(applDate)=8
      and cast(applDate as int)+200000 <= {L} and cast(cndrtExptnDate as int) > {L}"""
    out['n_alive_only_by_ext_'+L]=c.execute(q).fetchone()[0]
    q=f"""select count(distinct p.rgst_no) from patent p join right r on p.rgst_no=r.rgst_no where r.rgstCsName like '%연장%' and r.rgstCsDate > '{L}'
      and length(p.applDate)=8 and cast(p.applDate as int)+200000 <= {L} and cast(p.cndrtExptnDate as int) > {L}"""
    out['n_ext_registered_after_'+L]=c.execute(q).fetchone()[0]
out['n_patent']=c.execute("select count(*) from patent").fetchone()[0]
print(json.dumps(out,ensure_ascii=False,indent=1))
