# -*- coding: utf-8 -*-
"""3라운드 A단계 외부데이터 연계율·시점정합 점검(개발 원점 평가행 = 기준일 L의 생존 특허).
출력: ext_link_A.csv (원점×데이터별 연계율), ext_link_A_ksic.csv (KSIC별 연계 특허 수, O18)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r3lib, r3ext, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
R = r3lib.R
rows, kk = [], []
G = {'ETSI SEP(자체)': 'sep_self', 'ETSI SEP(해외 패밀리)': 'sep_fam', 'ETSI SEP(자체 또는 패밀리)': 'sep_any',
     '국민연금: 사업자번호 연결': 'np_pres', 'KPSS: US 패밀리 xi 보유': 'kp_has', 'ECOS 생산자물가(기본분류)': 'pr_ppi_g3',
     'ECOS 수출물가': 'pr_exp_g3', 'OpenAlex 기업논문': 'oa_all_rg', '내부 납부행동(출원인)': 'ap_late1y'}
cols = list(G.values()) + ['np_act', 'np_exit']
for o, _, Ls in R.ORIGINS:
    d = R._read_frame(o, 'ev.parquet', ['pid', 'L', 'ksic', 'next_annl', 'acd', 'age', 'atype', 'y'])
    d = r3ext.add(d, cols)
    for L in Ls:
        m = d.L.astype(str).str[:10] == L
        e = d[m]
        for nm, c in G.items():
            v = e[c]
            if c in ('sep_self', 'sep_fam', 'sep_any', 'kp_has'):
                lk = (v == 1)
            elif c == 'np_pres':
                lk = v.notna()
            else:
                lk = v.notna()
            r = dict(origin=o, L=L, data=nm, col=c, n=len(e), linked=int(lk.sum()), rate=round(float(lk.mean()), 4),
                     ksic_linked=int(e.loc[lk, 'ksic'].nunique()))
            if c in ('sep_self', 'sep_fam', 'sep_any', 'kp_has') and lk.sum() > 0:
                r['y_linked'] = round(float(e.y[lk].mean()), 4); r['y_other'] = round(float(e.y[~lk].mean()), 4)
            if c == 'np_pres' and lk.sum() > 0:
                r['y_linked'] = round(float(e.y[v == 1].mean()), 4) if (v == 1).any() else np.nan
                r['y_other'] = round(float(e.y[v == 0].mean()), 4) if (v == 0).any() else np.nan
                ex = e.np_exit == 1
                r['np_exit_n'] = int(ex.sum()); r['y_np_exit'] = round(float(e.y[ex].mean()), 4) if ex.any() else np.nan
            ok = v.notna()
            if ok.sum() > 1000 and v[ok].nunique() > 1:
                r['uni_auc'] = round(float(roc_auc_score(e.y[ok], v[ok])), 4)
            rows.append(r)
    if o == 'O18':
        e = d[d.L.astype(str).str[:10] == '2018-01-01']
        kk = e.groupby('ksic').agg(n=('pid', 'size'), sep_any=('sep_any', 'sum'), kp_has=('kp_has', 'sum'),
                                   np_linked=('np_pres', lambda s: int(s.notna().sum())), pr_ppi=('pr_ppi_g3', lambda s: int(s.notna().sum())),
                                   pr_exp=('pr_exp_g3', lambda s: int(s.notna().sum())))
    print(o, 'done', R.peak_mem_gb(), flush=True)
pd.DataFrame(rows).to_csv(os.path.join(r3lib.R3, 'ext_link_A.csv'), index=False, encoding='utf-8-sig')
kk.to_csv(os.path.join(r3lib.R3, 'ext_link_A_ksic.csv'), encoding='utf-8-sig')
print(pd.DataFrame(rows).to_string())
