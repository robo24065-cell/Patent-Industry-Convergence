# -*- coding: utf-8 -*-
"""연계율 탐침: ETSI SEP·KPSS·NPS가 T4 특허 모집단(pt)에 몇 건 붙는가 (읽기 전용)"""
import os, pickle, numpy as np, pandas as pd
T4 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXT = os.path.join(os.path.dirname(T4), '_shared', 'ext')
D = pickle.load(open(os.path.join(T4, 'R2', 'cache', 'loaded_r2dev.pkl'), 'rb'))
pt = D['pt']
print('pt', len(pt), pt.applNo.str.len().value_counts().to_dict())
s = pd.read_parquet(os.path.join(EXT, 'trade_etsi_sep_link.parquet'))
k = s[s.cc == 'KR']
print('ETSI trade KR rows', len(k), k.appl_no.str.len().value_counts().to_dict(), k.appl_no.head(3).tolist())
m = pt.applNo.isin(set(k.appl_no))
print('ETSI trade 연결 pt', int(m.sum()), round(m.mean(), 4))
print(k.first_refl.dt.year.value_counts().sort_index().to_dict())
print('decl year', k.first_decl.dt.year.value_counts().sort_index().to_dict())
q = pd.read_parquet(os.path.join(EXT, 'quality_etsi_sep_link.parquet'))
qk = q[q.cc == 'KR']
print('ETSI quality KR', len(qk), 'pt 연결', int(pt.applNo.isin(set(qk.appl_no)).sum()))
kp = pd.read_parquet(os.path.join(EXT, 'quality_kpss_family_link.parquet'))
kk = kp[kp.cc == 'KR']
print('KPSS KR rows', len(kk), 'uniq appl', kk.appl_no.nunique(), 'pt 연결', int(pt.applNo.isin(set(kk.appl_no)).sum()))
print(kk.issue_date.astype(str).str[:4].value_counts().sort_index().tail(12).to_dict())
lk = pd.read_parquet(os.path.join(EXT, 'firm_nps_applicant_link.parquet'))
print('NPS link acd', len(lk), 'pt acd 연결', round(pt.acd.isin(set(lk.applicantCd)).mean(), 4), 'nps_hit', round(pt.acd.isin(set(lk[lk.nps_hit].applicantCd)).mean(), 4))
print(pt.atype.value_counts().to_dict())
for L in ['2016-01-01', '2018-01-01']:
    Ld = np.datetime64(L)
    al = (pt.rd.values <= Ld)
    print(L, 'ETSI 선언<=L', int((m & al).sum()))
