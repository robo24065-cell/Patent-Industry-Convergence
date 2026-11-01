# -*- coding: utf-8 -*-
"""(B단계) 내부 휴면 대리변수(dm_)가 국세청 폐업(nt_)을 얼마나 잡는지 — L>=2016 평가행(O16·O18)에서만 검증.
개발 원점 평가행(결과 창 끝 <= 2021-07)만 읽는다. 결과: probe/p7_dm_nts.csv"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import r3lib, r3ext
from sklearn.metrics import roc_auc_score
R = r3lib.R
rows = []
for o in ('O16', 'O18'):
    d = R._read_frame(o, 'ev.parquet', ['pid', 'L', 'ksic', 'acd', 'atype', 'y'])
    d = r3ext.add(d, ['dm_opay_yrs', 'dm_oreg_yrs', 'dm_opay_n1y', 'nt_closed', 'nt_cl1y'])
    for L, g in d.groupby('L'):
        g = g[g.nt_closed.notna() & g.dm_opay_yrs.notna()]
        r = dict(origin=o, L=str(L), n_nts_linked=len(g), closed_rate=round(g.nt_closed.mean(), 4), cl1y_rate=round(g.nt_cl1y.mean(), 4))
        for c in ('dm_opay_yrs', 'dm_oreg_yrs'):
            r[f'auc_{c}_closed'] = round(roc_auc_score(g.nt_closed, g[c]), 4)
            r[f'auc_{c}_cl1y'] = round(roc_auc_score(g.nt_cl1y, g[c]), 4) if g.nt_cl1y.nunique() > 1 else np.nan
        r['auc_neg_n1y_closed'] = round(roc_auc_score(g.nt_closed, -g.dm_opay_n1y), 4)
        dorm = g.dm_opay_yrs > 1.0
        r['dorm1y_share'] = round(dorm.mean(), 4)
        r['closed_among_dorm1y'] = round(g.nt_closed[dorm].mean(), 4)
        r['closed_among_active'] = round(g.nt_closed[~dorm].mean(), 4)
        r['lapse_dorm1y'] = round(g.y[dorm].mean(), 4); r['lapse_active'] = round(g.y[~dorm].mean(), 4)
        # 전체 평가행(국세청 미연결 포함)의 휴면 여부별 포기율
        a = d[(d.L == L) & d.dm_opay_yrs.notna()]
        r['all_dorm1y_share'] = round((a.dm_opay_yrs > 1).mean(), 4)
        r['all_lapse_dorm1y'] = round(a.y[a.dm_opay_yrs > 1].mean(), 4); r['all_lapse_active'] = round(a.y[a.dm_opay_yrs <= 1].mean(), 4)
        rows.append(r); print(r, flush=True)
pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'p7_dm_nts.csv'), index=False, encoding='utf-8-sig')
