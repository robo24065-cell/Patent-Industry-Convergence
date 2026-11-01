# 문서·출원인·서브클래스 캐시(공개일 < 2025-01-01만). 출력 cache/docs_{CN,EP,KR}.parquet, app_{CN,EP}.parquet, dsc_{CN,EP}.parquet, agg5.parquet, s0_log.json
import time, json
import numpy as np, pandas as pd
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib12 import *

t = time.time()
con = con_duck()
log = {}
ks = sorted(r[0] for r in con.execute(f"select distinct ksic from read_parquet('{META}')").fetchall())
assert len(ks) == 50
K2I = {k: i for i, k in enumerate(ks)}
json.dump(ks, open(C + r'\ksic_list.json', 'w'))
tf = con.execute(f"select appl_no, k_free from read_parquet('{TXTFREE}')").df().drop_duplicates('appl_no').set_index('appl_no').k_free

SCS = set()
for cc in ['KR', 'CN', 'EP']:
    d = con.execute(f"""select appl_no, cast(pub_date as varchar) pub_s, cast(appl_date as varchar) ad_s, ksic, main_ipc, ipc_all, applicant
        from read_parquet('{META}') where cc='{cc}' and pub_date is not null and pub_date < '{CUT}' """).df()
    n0 = len(d)
    d['pub'] = pd.to_numeric(d.pub_s.str.slice(0, 8), errors='coerce')
    d['ad'] = pd.to_numeric(d.ad_s.str.slice(0, 8), errors='coerce')
    bad = int(d.pub.isna().sum())
    d = d[d.pub.notna()].copy(); d['pub'] = d.pub.astype(np.int64)
    assert d.pub.max() < CUT
    d['sc'], d['mg'], d['sg'] = parse_ipc(d.main_ipc)
    d['k'] = d.ksic.map(K2I).astype(np.int16)
    d['did'] = np.arange(len(d), dtype=np.int64)
    if cc == 'KR':
        d['kt'] = d.appl_no.map(tf)
        d[['did', 'appl_no', 'pub', 'ad', 'k', 'sc', 'mg', 'sg', 'kt']].to_parquet(C + r'\docs_KR.parquet', index=False)
        log[cc] = dict(rows=n0, bad_pub=bad, docs=len(d), sc_missing=float(d.sc.isna().mean()), kt_cov=float(d.kt.notna().mean()))
        out(cc, len(d), round(time.time() - t, 1))
        continue
    # 전체 IPC 서브클래스(보조 포함) 긴 표
    ia = d[['did', 'ipc_all']].dropna()
    ia = ia.assign(x=ia.ipc_all.str.upper().str.split('|')).explode('x')
    ia['sc2'] = ia.x.str.strip().str.slice(0, 4)
    ia = ia[ia.sc2.str.match(SC_RE, na=False)][['did', 'sc2']].drop_duplicates()
    SCS |= set(d.sc.dropna().unique()) | set(ia.sc2.unique())
    # 출원인
    names = d.applicant.values
    rows_d, rows_n = [], []
    for did, a in zip(d.did.values, names):
        for nm in split_app(a):
            rows_d.append(did); rows_n.append(nm)
    A = pd.DataFrame({'did': np.array(rows_d, np.int64), 'name': rows_n})
    un = pd.Series(A.name.unique())
    km = dict(zip(un, un.map(norm_key))); am = dict(zip(un, un.map(aggr_key)))
    A['key'] = A.name.map(km); A['akey'] = A.name.map(am)
    A = A[A.key.str.len() >= 2].drop_duplicates(['did', 'key'])
    n_noapp = int((~d.did.isin(A.did)).sum())
    d[['did', 'appl_no', 'pub', 'ad', 'k', 'sc', 'mg', 'sg']].to_parquet(C + rf'\docs_{cc}.parquet', index=False)
    A[['did', 'key', 'akey']].to_parquet(C + rf'\app_{cc}.parquet', index=False)
    ia.to_parquet(C + rf'\dsc_{cc}.parquet', index=False)
    yr = d.pub // 10000
    log[cc] = dict(rows=n0, bad_pub=bad, docs=len(d), app_rows=len(A), docs_no_app=n_noapp,
                   sc_missing=float(d.sc.isna().mean()), sg_missing=float(d.sg.isna().mean()),
                   pub_before_appl=int((d.ad.notna() & (d.pub < d.ad)).sum()), ad_missing=int(d.ad.isna().sum()),
                   by_year={int(k): int(v) for k, v in yr.value_counts().sort_index().items()},
                   noapp_by_year={int(k): int(v) for k, v in yr[~d.did.isin(A.did)].value_counts().sort_index().items()},
                   sc_missing_by_year={int(k): float(v) for k, v in d.sc.isna().groupby(yr).mean().items()})
    out(cc, log[cc]['docs'], log[cc]['app_rows'], round(time.time() - t, 1))
    del d, A, ia

# 5개국 KSIC×IPC 집계(추세 피처용; 출원인 불필요)
g = con.execute(f"""select cc, cast(substr(cast(pub_date as varchar),1,4) as int) y, main_ipc, ksic, count(*) n
    from read_parquet('{META}') where pub_date is not null and pub_date < '{CUT}' group by 1,2,3,4""").df()
g['sc'], g['mg'], g['sg'] = parse_ipc(g.main_ipc)
g['k'] = g.ksic.map(K2I).astype(np.int16)
g = g.groupby(['cc', 'y', 'sc', 'mg', 'sg', 'k'], dropna=False).n.sum().reset_index()
assert g.y.max() <= 2024
g.to_parquet(C + r'\agg5.parquet', index=False)
log['agg5'] = {cc: int(v) for cc, v in g.groupby('cc').n.sum().items()}
SCL = sorted(SCS)
json.dump(SCL, open(C + r'\sc_list.json', 'w'))
log['n_sc'] = len(SCL)
json.dump(log, open(C + r'\s0_log.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
out('done', round(time.time() - t, 1), log['agg5'])
