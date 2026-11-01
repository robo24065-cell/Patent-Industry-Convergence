# 문서·출원인 캐시(공개일 < 2025-01-01). 출력 cache/docs_{cc}.parquet, cache/app_{cc}.parquet, cache/s0_log.json
import time, json, re
import numpy as np, pandas as pd
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib11 import *

t = time.time()
con = con_duck()
log = {}
ks = sorted(r[0] for r in con.execute(f"select distinct ksic from read_parquet('{META}')").fetchall())
assert len(ks) == 50, len(ks)
K2I = {k: i for i, k in enumerate(ks)}
json.dump(ks, open(C + r'\ksic_list.json', 'w'))
tf = con.execute(f"select appl_no, k_free from read_parquet('{TXTFREE}')").df().set_index('appl_no').k_free
for cc in ['KR', 'CN', 'EP']:
    d = con.execute(f"""select appl_no, cast(pub_date as varchar) pub_s, cast(appl_date as varchar) ad_s, ksic, main_ipc, applicant
        from read_parquet('{META}') where cc='{cc}' and pub_date is not null and pub_date < '{CUT}' """).df()
    n0 = len(d)
    d['pub'] = pd.to_numeric(d.pub_s.str.slice(0, 8), errors='coerce')
    d['ad'] = pd.to_numeric(d.ad_s.str.slice(0, 8), errors='coerce')
    bad = int(d.pub.isna().sum())
    d = d[d.pub.notna()].copy(); d['pub'] = d.pub.astype(np.int64)
    assert d.pub.max() < CUT
    ip = d.main_ipc.fillna('').str.upper().str.replace(r'\s+', ' ', regex=True).str.strip()
    d['sc'] = ip.str.slice(0, 4).where(ip.str.len() >= 4)
    ex = ip.str.extract(r'^([A-H]\d\d[A-Z])\s?0*(\d+)(?:/(\d+))?', expand=True)
    d['mg'] = (ex[0] + ' ' + ex[1]).where(ex[0].notna() & ex[1].notna())
    d['sg'] = (d['mg'] + '/' + ex[2]).where(d['mg'].notna() & ex[2].notna())
    d['k'] = d.ksic.map(K2I).astype(np.int16)
    if cc == 'KR':
        d['kt'] = d.appl_no.map(tf)
    d['did'] = np.arange(len(d), dtype=np.int64)
    # 출원인 펼치기
    rows = []
    for did, a in zip(d.did.values, d.applicant.values):
        for nm in split_app(a):
            rows.append((did, nm))
    A = pd.DataFrame(rows, columns=['did', 'name'])
    A['key'] = A.name.map(norm_key)
    A['akey'] = A.name.map(aggr_key)
    if cc == 'KR':
        A['corp'] = A.name.str.contains(KR_CORP)
    A = A[A.key.str.len() >= 2]
    A = A.drop_duplicates(['did', 'key'])
    n_noapp = int((~d.did.isin(A.did)).sum())
    cols = ['did', 'appl_no', 'pub', 'ad', 'k', 'sc', 'mg', 'sg'] + (['kt'] if cc == 'KR' else [])
    d[cols].to_parquet(C + rf'\docs_{cc}.parquet', index=False)
    A[['did', 'key', 'akey'] + (['corp'] if cc == 'KR' else [])].to_parquet(C + rf'\app_{cc}.parquet', index=False)
    yr = (d.pub // 10000)
    log[cc] = dict(rows=n0, bad_pub=bad, docs=len(d), app_rows=len(A), docs_no_app=n_noapp,
                   sc_missing=float(d.sc.isna().mean()), sg_missing=float(d.sg.isna().mean()),
                   pub_before_appl=int((d.ad.notna() & (d.pub < d.ad)).sum()),
                   by_year={int(k): int(v) for k, v in yr.value_counts().sort_index().items()},
                   noapp_by_year={int(k): int(v) for k, v in yr[~d.did.isin(A.did)].value_counts().sort_index().items()},
                   kt_cov=(float(d.kt.notna().mean()) if cc == 'KR' else None))
    out(cc, log[cc]['docs'], log[cc]['app_rows'], round(time.time() - t, 1))
json.dump(log, open(C + r'\s0_log.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
out('done', round(time.time() - t, 1))
