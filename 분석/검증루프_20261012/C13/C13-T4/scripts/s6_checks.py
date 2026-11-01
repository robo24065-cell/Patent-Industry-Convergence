# C13-T4 s6: 데이터 점검(분포·결측·중복·연결 선택 편향·시점). 공개 < 2025 자료만.
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from lib13 import *
res = {}
for cc in ['CN', 'EP']:
    d = pd.read_parquet(f'{C}/docs_{cc}.parquet', columns=['appl_no', 'pub', 'ksic', 'sc'])
    a = pd.read_parquet(f'{C}/app_{cc}.parquet')
    assert d.appl_no.is_unique and d.pub.max() < CUT
    d['y'] = d.pub // 10000
    nor = a.groupby('appl_no').key.nunique()
    d['norg'] = d.appl_no.map(nor).fillna(0)
    g = d[d.y >= 2015].groupby('y').agg(docs=('appl_no', 'size'), org_doc=('norg', lambda s: float((s >= 1).mean())),
                                         multi_org=('norg', lambda s: float((s >= 2).mean())), sc_na=('sc', lambda s: float(s.isna().mean())),
                                         ksic_na=('ksic', lambda s: float(s.isna().mean())))
    res[cc] = {'by_year': g.reset_index().to_dict('records'), 'dup_appl_no': int(d.appl_no.duplicated().sum())}
    # 상반기 문서 수: 2024-H1 / 2023-H1
    h = lambda y0: int(((d.pub >= y0 * 10000 + 101) & (d.pub <= y0 * 10000 + 630)).sum())
    res[cc]['H1_2024_over_2023'] = h(2024) / h(2023)
# 평가 원점 단위
P = pd.read_parquet(f'{C}/u_prov_2023.parquet')
res['eval'] = {'n': len(P), 'pos': int(P.y.sum()), 'cn_share_units': float((P.cc == 'CN').mean()), 'cn_share_pos': float((P[P.y == 1].cc == 'CN').mean()),
               'imp_na_share': float(P.x_imp_na.mean()), 'pos_rate_imp_ok': float(P[P.x_imp_na == 0].y.mean()), 'pos_rate_imp_na': float(P[P.x_imp_na == 1].y.mean()),
               'top5_ksic_share_pos': float(P[P.y == 1].modal.value_counts(normalize=True).head(5).sum()),
               'top5_ksic_share_risk': float(P.modal.value_counts(normalize=True).head(5).sum()),
               'n_ksic': int(P.modal.nunique())}
# 같은 그룹 의심 관계(키 앞 4글자 같음) 비율: 2024-H1 창 새 이종 관계
A = []
for cc in ['CN', 'EP']:
    a = pd.read_parquet(f'{C}/app_{cc}.parquet'); a['u'] = cc + ':' + a.key; A.append(a)
A = pd.concat(A)
m = A[A.appl_no.isin(A.groupby('appl_no').u.nunique().pipe(lambda s: s[s >= 2]).index)]
e = m.merge(m[['appl_no', 'u']], on='appl_no', suffixes=('', '_g')); e = e[e.u != e.u_g]
fst = e.groupby(['u', 'u_g']).pub.min().rename('first').reset_index()
nw = fst[(fst['first'] >= 20240101) & (fst['first'] <= 20240630) & fst.u.isin(P.u)]
same4 = (nw.u.str[3:7] == nw.u_g.str[3:7])
res['ties_2024H1'] = {'new_ties_from_risk': len(nw), 'same_prefix4_share': float(same4.mean()),
                      'univ_partner_share': float(nw.u_g.str.contains('大学|学院|UNIV|INSTITUT|研究').mean())}
# 양성 기업 vs 비양성: 규모 SMD
x1 = P[P.y == 1].b_n3; x0 = P[P.y == 0].b_n3
res['eval']['smd_b_n3_pos_vs_neg'] = float((x1.mean() - x0.mean()) / np.sqrt((x1.var() + x0.var()) / 2))
json.dump(res, open(f'{C}/s6_checks.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
out(json.dumps(res, ensure_ascii=False, indent=1, default=float))
