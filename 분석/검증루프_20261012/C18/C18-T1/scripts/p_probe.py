import sqlite3, pathlib, pyarrow.parquet as pq, json
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
OUT = B/'분석/검증루프_20261012/C18/C18-T1/cache/p_probe.json'
ro = lambda p: sqlite3.connect(pathlib.Path(p).as_uri()+'?mode=ro', uri=True)
o = {}
for f in ['kr_registered.parquet','kr_patents.parquet']:
    t = pq.read_table(B/'공모전데이터/작업데이터'/f).slice(0,5).to_pylist()
    o[f+'_cols'] = list(t[0].keys()); o[f+'_rows'] = [{k:str(v)[:60] for k,v in r.items()} for r in t[:3]]
c = ro(B/'수집파이프라인/registry_full.sqlite')
o['rf_cols'] = [r[1] for r in c.execute('pragma table_info(patent)')]
o['rf_sample'] = [list(map(str,r)) for r in c.execute("select rgst_no, applNo, rgstDate, applDate, claimCount, ksic, lastDspst from patent limit 3")]
o['rf_len'] = c.execute("select length(rgst_no), substr(rgst_no,1,2), count(*) from patent group by 1,2").fetchall()
c = ro(B/'수집파이프라인/registry.sqlite')
o['reg_status'] = c.execute("select source, status, count(*) from registry group by 1,2").fetchall()
o['pending_n'] = c.execute("select count(*), sum(substr(rgst_no,1,2)='10') from pending").fetchone()
c = ro(B/'수집파이프라인/business_rights.sqlite')
o['rl_sample'] = [list(map(str,r)) for r in c.execute("select applicant_cd, rgst_no, rgst_date, appl_no, claim_count from right_list limit 3")]
o['aq_cols'] = [r[1] for r in c.execute('pragma table_info(applicant_query)')]
json.dump(o, open(OUT,'w',encoding='utf-8'), ensure_ascii=False, indent=1)
