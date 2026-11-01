# s3: 필수성 사전 점검(개발 원점만). 공개 배정 FREE-IPC(H-sg-3), H-sc-1, TXT-FREE로 사건 재정의 → 제공 사건과 자카드·재현율.
import sys, json, time
sys.path.insert(0, r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C02\C02-T4\scripts')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from lib_cd import *
T0 = time.time()
own, pat, app, docs, orig = load()
assert own.d.max() < '20250101'
lab = pd.read_parquet(C + r'\kr_publab_src.parquet')
lab['mg'] = lab.sg.str.split('/').str[0]; lab['sc'] = lab.sg.str[:4]
lab = lab.set_index('appl_no')
P = pd.read_parquet(C + r'\dev_pos.parquet')
EP = set(zip(P.t0, P.cd, P.k))
prov_doc = docs.ksic.where(docs.ksic.isin(KSIC50))


def hard(t0, start, nmin):
    sub = lab[(lab.appl_year <= t0)]
    lv = ['sg', 'mg', 'sc'][['sg', 'mg', 'sc'].index(start):]
    out = pd.Series(np.nan, index=lab.index, dtype=object)
    todo = pd.Series(True, index=lab.index)
    for l in lv:
        s = sub[sub[l].notna()]
        tab = s.groupby([l, 'k_free']).size().reset_index(name='n')
        tot = tab.groupby(l).n.sum()
        mode = tab.sort_values([l, 'n', 'k_free'], ascending=[True, False, True]).drop_duplicates(l).set_index(l).k_free
        ok = tot[tot >= nmin].index
        mp = mode[mode.index.isin(ok)]
        v = lab[l].map(mp)
        hit = todo & v.notna()
        out[hit] = v[hit]; todo &= ~hit
    return out


rows, agree = [], []
for t0 in DEV:
    labs = {'TXT-FREE': lab.k_free, 'FREE-IPC(H-sg-3)': hard(t0, 'sg', 3), 'H-sc-1': hard(t0, 'sc', 1)}
    Pt = P[P.t0 == t0]
    Et = set(zip(Pt.cd, Pt.k))
    for nm, L in labs.items():
        dl = docs.appl_no.map(L)
        pl = dict(zip(pat.rgst_no, pat.appl_no.map(L)))
        st, pos, blocked, F = build(t0, own, docs, app, orig, dl, pl)
        EA = set(zip(pos.cd, pos.k))
        inter = len(EA & Et); uni = len(EA | Et)
        rows.append(dict(t0=t0, assign=nm, events_public=len(EA), events_provided=len(Et), intersect=inter,
                         recall_of_provided=round(inter / max(len(Et), 1), 4), jaccard=round(inter / max(uni, 1), 4),
                         risk_units_public=st['risk_units'], pat_label_agree=round(float((pat.appl_no.map(L) == pat.ksic).mean()), 4),
                         pat_unassigned=round(float(pat.appl_no.map(L).isna().mean()), 4)))
        print(rows[-1], round(time.time() - T0), flush=True)
R = pd.DataFrame(rows)
pool = R.groupby('assign')[['events_public', 'events_provided', 'intersect']].sum()
pool['recall_of_provided'] = (pool.intersect / pool.events_provided).round(4)
pool['jaccard'] = (pool.intersect / (pool.events_public + pool.events_provided - pool.intersect)).round(4)
pool = pool.reset_index(); pool['t0'] = 'pooled_2012_2022'
out = pd.concat([pool, R], ignore_index=True)
out['verdict_rule'] = 'jaccard<0.5 for all three => 필수성 지지'
out.to_csv(W + r'\essentiality_dev.csv', index=False, encoding='utf-8-sig')
print(pool.to_string())
print('done', round(time.time() - T0))
