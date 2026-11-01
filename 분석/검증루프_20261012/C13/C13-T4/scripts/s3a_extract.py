# C13-T4 s3a: X1' 원자료 추출(CN·EP, 공개일 2010-01-01~2024-12-31). 2025 이후 행은 읽지 않는다(쿼리 조건 + assert).
import sys, json
import pandas as pd
sys.path.insert(0, __import__('os').path.dirname(__file__))
from lib13 import *
con = con_duck()
log = {}
for cc in ['CN', 'EP']:
    docs = con.execute(f"""
        select appl_no, min(cast(pub_date as integer)) pub, any_value(ksic) ksic, any_value(main_ipc) main_ipc,
               any_value(applicant) applicant
        from read_parquet('{META}')
        where cc='{cc}' and pub_date >= '20100101' and pub_date < '20250101' and length(pub_date)=8
        group by appl_no""").fetchdf()
    assert docs.pub.max() < CUT and docs.pub.min() >= 20100101
    docs['sc'] = docs.main_ipc.fillna('').str.upper().str.extract(r'^([A-H]\d\d[A-Z])', expand=False)
    raw = docs[['appl_no', 'applicant']].copy()
    raw['nm'] = raw.applicant.fillna('').str.split(r'[|;]')
    raw = raw.explode('nm')
    raw['nm'] = raw.nm.str.strip()
    raw = raw[raw.nm.notna() & (raw.nm != '') & (~raw.nm.str.lower().isin(['nan', 'none', '-', 'null']))]
    u = pd.DataFrame({'nm': raw.nm.unique()})
    u['key'] = u.nm.map(norm_key)
    u['org'] = [is_org(x, cc) for x in u.nm]
    raw = raw.merge(u, on='nm')
    app = raw[raw.org & (raw.key.str.len() >= 2)][['appl_no', 'key']].drop_duplicates()
    app = app.merge(docs[['appl_no', 'pub', 'ksic']], on='appl_no')
    docs.drop(columns=['applicant']).to_parquet(f'{C}/docs_{cc}.parquet', index=False)
    app.to_parquet(f'{C}/app_{cc}.parquet', index=False)
    nper = app.groupby('appl_no').key.nunique()
    log[cc] = {'docs': len(docs), 'name_rows': len(raw), 'uniq_names': len(u), 'org_name_share': float(u.org.mean()),
               'org_row_share': float(raw.org.mean()), 'app_rows': len(app), 'docs_with_org': int(nper.size),
               'docs_multi_org': int((nper >= 2).sum()), 'sc_missing': float(docs.sc.isna().mean()),
               'org_name_sample': u[u.org].nm.sample(8, random_state=1).tolist(),
               'nonorg_name_sample': u[~u.org].nm.sample(8, random_state=1).tolist()}
    out(cc, log[cc])
    del docs, raw, app, u
json.dump(log, open(f'{C}/s3a_log.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
