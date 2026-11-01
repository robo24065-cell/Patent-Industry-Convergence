# 재개 점검: 직전 시도가 남긴 캐시가 s0_extract.py 출력과 맞는지, C11 캐시와 대조. 출력 cache/c0_cachecheck.json
import json, time
import numpy as np, pandas as pd
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib12 import *

C11 = B + r'\분석\검증루프_20261012\C11\C11-T4\cache'
res = {}
log = json.load(open(C + r'\s0_log.json', encoding='utf-8'))
for cc in ['KR', 'CN', 'EP']:
    d = pd.read_parquet(C + rf'\docs_{cc}.parquet')
    o = pd.read_parquet(C11 + rf'\docs_{cc}.parquet', columns=['appl_no', 'pub', 'k'])
    r = dict(n=len(d), n_log=log[cc]['docs'], did_ok=bool((d.did.values == np.arange(len(d))).all()),
             max_pub=int(d.pub.max()), appl_dup=int(d.appl_no.duplicated().sum()),
             n_c11=len(o), same_appl_set=bool(set(d.appl_no) == set(o.appl_no)) if cc != 'CN' else None)
    if cc == 'CN':
        r['same_appl_set'] = bool(len(np.setdiff1d(d.appl_no.values.astype(str), o.appl_no.values.astype(str))) == 0)
    m = d[['appl_no', 'k', 'pub']].merge(o, on='appl_no', suffixes=('', '_o'))
    r['k_agree'] = float((m.k == m.k_o).mean()); r['pub_agree'] = float((m.pub == m.pub_o).mean())
    if cc != 'KR':
        a = pd.read_parquet(C + rf'\app_{cc}.parquet')
        a11 = pd.read_parquet(C11 + rf'\app_{cc}.parquet', columns=['did', 'key'])
        r['app_rows'] = len(a); r['app_rows_log'] = log[cc]['app_rows']; r['app_rows_c11'] = len(a11)
        r['nkeys'] = int(a.key.nunique())
        ds = pd.read_parquet(C + rf'\dsc_{cc}.parquet')
        r['dsc_rows'] = len(ds); r['dsc_docs'] = int(ds.did.nunique())
        # 메인 서브클래스가 ipc_all 서브클래스 집합에 포함되는 비율
        mm = d[['did', 'sc']].dropna().merge(ds.rename(columns={'sc2': 'sc'}), on=['did', 'sc'])
        r['main_in_all'] = float(len(mm) / d.sc.notna().sum())
        # 월별 메인 IPC 결측(2020-)
        ym = d.pub // 100
        mr = d.sc.isna().groupby(ym).mean()
        r['sc_miss_month_gt2pct'] = {int(k): round(float(v), 4) for k, v in mr.items() if k >= 201801 and v > 0.02}
    else:
        r['kt_cov'] = float(d.kt.notna().mean())
    res[cc] = r
    out(cc, json.dumps(r, ensure_ascii=False)[:600])
ag = pd.read_parquet(C + r'\agg5.parquet')
res['agg5'] = {cc: int(v) for cc, v in ag.groupby('cc').n.sum().items()}
res['agg5_maxy'] = int(ag.y.max())
json.dump(res, open(C + r'\c0_cachecheck.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
out(res['agg5'])
