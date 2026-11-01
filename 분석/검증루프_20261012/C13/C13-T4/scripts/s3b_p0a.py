# C13-T4 s3b: X1' P0(a) (PREREG 변경 1). 입력 cache/app_{CN,EP}.parquet (공개 < 2025).
import sys, json, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from lib13 import *


def pair_first(app):
    m = app[app.appl_no.isin(app.groupby('appl_no').key.nunique().pipe(lambda s: s[s >= 2]).index)]
    e = m.merge(m[['appl_no', 'key']], on='appl_no', suffixes=('', '_g'))
    e = e[e.key != e.key_g]
    return e  # 순서쌍 (key, key_g), pub


def modal_at(app, lo, hi):
    h = app[(app.pub > lo) & (app.pub <= hi)]
    m = h.groupby(['key', 'ksic']).size().rename('n').reset_index().sort_values(['key', 'n', 'ksic'], ascending=[True, False, True])
    return m.drop_duplicates('key').set_index('key').ksic


res = {}
rows = []
for cc in ['CN', 'EP']:
    app = pd.read_parquet(f'{C}/app_{cc}.parquet')
    assert app.pub.max() < CUT
    e = pair_first(app)
    first = e.groupby(['key', 'key_g']).pub.min().rename('first').reset_index()
    for t0 in range(2016, 2025):
        end = t0 * 10000 + 1231
        h3 = app[(app.pub > end - 30000) & (app.pub <= end)]
        cnt = h3.groupby('key').appl_no.nunique()
        act = cnt[cnt >= 5].index
        r = {'cc': cc, 't0': t0, 'risk': int(len(act))}
        if t0 <= 2023:
            modal = modal_at(app, end - 50000, end)
            lo, hi = (t0 + 1) * 10000 + 101, (t0 + 1) * 10000 + 630
            nw = first[(first['first'] >= lo) & (first['first'] <= hi) & first.key.isin(act)].copy()
            nw['mf'] = nw.key.map(modal); nw['mg'] = nw.key_g.map(modal)
            st = nw[nw.mg.notna() & (nw.mg != nw.mf)]
            r.update({'win_docs_multi_org': int(e[(e.pub >= lo) & (e.pub <= hi)].appl_no.nunique()),
                      'pos_loose': int(nw.key.nunique()), 'pos_strict': int(st.key.nunique()),
                      'pairs_strict': int(len(st)), 'partner_modal_defined': float(nw.mg.notna().mean()) if len(nw) else None})
            r['rate_loose'] = r['pos_loose'] / r['risk']; r['rate_strict'] = r['pos_strict'] / r['risk']
            # 같은 (f, g) 첫 관계는 정의상 한 번. 원점 간 창 서로소.
        rows.append(r); out(r)
df = pd.DataFrame(rows)
df.to_csv(f'{C}/s3b_p0a_by_origin.csv', index=False, encoding='utf-8-sig')
dev = df[df.t0 <= 2023]
for cc in ['CN', 'EP', 'ALL']:
    d = dev if cc == 'ALL' else dev[dev.cc == cc]
    lk = df[(df.t0 == 2024) & ((df.cc == cc) | (cc == 'ALL'))]
    # 잠금 예상 = 나라별 위험 × 나라별 2022·23 평균 기저율의 합
    exp_s = exp_l = 0.0
    for c2 in (['CN', 'EP'] if cc == 'ALL' else [cc]):
        b = dev[(dev.cc == c2) & dev.t0.isin([2022, 2023])]
        rk = int(df[(df.cc == c2) & (df.t0 == 2024)].risk.iloc[0])
        exp_s += rk * b.rate_strict.mean(); exp_l += rk * b.rate_loose.mean()
    res[cc] = {'dev_pos_strict_sum': int(d.pos_strict.sum()), 'dev_pos_loose_sum': int(d.pos_loose.sum()),
               'lock_risk': int(lk.risk.sum()), 'lock_expected_strict': float(exp_s), 'lock_expected_loose': float(exp_l)}
res['P0a_pass'] = bool(res['ALL']['dev_pos_strict_sum'] >= 1000 and res['ALL']['lock_expected_strict'] >= 1000)
json.dump(res, open(f'{C}/s3b_p0a.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
out(json.dumps(res, ensure_ascii=False, indent=1))
