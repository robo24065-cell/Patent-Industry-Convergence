# C18-T2 verify_0 r2: 이중 수록 쌍 독립 정의 + 표2/표3 패밀리 재연결 재계산 (P0/V1/V1E/정규화키 dedup/짧은형 유지)
import os, json, pandas as pd, numpy as np
D = os.path.dirname(os.path.abspath(__file__))
ue = pd.read_pickle(os.path.join(D, 'us_ep.pkl')); kr = pd.read_pickle(os.path.join(D, 'kr.pkl')); tk = pd.read_pickle(os.path.join(D, 'tok.pkl'))
res = {}
us = ue[ue.cc == 'US']; ep = ue[ue.cc == 'EP']
res['us_len'] = us.appl_no.str.len().value_counts().sort_index().to_dict()
res['ep_len'] = ep.appl_no.str.len().value_counts().sort_index().to_dict()
res['ep_examples'] = ep.appl_no.sample(8, random_state=1).tolist()
# 정규화 키: 숫자·점 이외 제거 후 앞 0 제거
def nk(s):
    return s.str.replace(r'[^0-9.]', '', regex=True).str.lstrip('0')
for nm, df in [('US', us), ('EP', ep)]:
    u = df[['appl_no']].drop_duplicates().assign(k=lambda x: nk(x.appl_no))
    g = u.groupby('k').appl_no.nunique()
    res[f'{nm}_normkey_groups_gt1'] = int((g > 1).sum()); res[f'{nm}_normkey_max'] = int(g.max())
    res[f'{nm}_rows_dup_appl_no_exact'] = int(df.appl_no.duplicated().sum())
# US 7자리 → 8자리
u8 = set(us.appl_no[us.appl_no.str.len() == 8])
us7 = us.appl_no[(us.appl_no.str.len() == 7) & us.appl_no.str.zfill(8).isin(u8)]
res['US_7in8_rows'] = int(len(us7)); res['US_7in8_distinct'] = int(us7.nunique())
# EP: 짧은형 s, 긴형 '0'+s
epa = set(ep.appl_no)
eps = ep.appl_no[ep.appl_no.map(lambda s: ('0' + s) in epa)]
res['EP_short_rows'] = int(len(eps)); res['EP_short_distinct'] = int(eps.nunique())
res['EP_short_len'] = eps.str.len().value_counts().to_dict()
# 같은 특허 점검(제목·출원일)
def same(df, short, mk):
    a = df[df.appl_no.isin(set(short))][['appl_no', 'title', 'appl_date', 'ksic']].drop_duplicates('appl_no')
    a['L'] = a.appl_no.map(mk)
    b = df[['appl_no', 'title', 'appl_date', 'ksic']].drop_duplicates('appl_no').rename(columns={'appl_no': 'L'})
    m = a.merge(b, on='L', suffixes=('_s', '_l'))
    n = lambda s: s.fillna('').str.lower().str.split().str.join(' ')
    return {'pairs': int(len(m)), 'title_eq': float((n(m.title_s) == n(m.title_l)).mean()), 'date_eq': float((m.appl_date_s == m.appl_date_l).mean()),
            'ksic_eq': float((m.ksic_s == m.ksic_l).mean()), 'appl_year_s': m.appl_date_s.str[:4].value_counts().sort_index().to_dict()}
res['US_same'] = same(us, us7, lambda s: s.zfill(8))
res['EP_same'] = same(ep, eps, lambda s: '0' + s)
# 연결
tk = tk.drop_duplicates(['cc', 'appl_no', 'fk', 'r9'])
def links(t):
    fr = t.merge(kr, on='r9')
    o = {}
    for cc, g in fr.groupby('cc'):
        o[cc] = [int(len(g)), int(g.appl_no.nunique()), int(g.kr_appl_no.nunique()), float((g.fk == g.kk).mean())]
    return o, fr
drop_us7 = (tk.cc == 'US') & tk.appl_no.isin(set(us7))
drop_ep9 = (tk.cc == 'EP') & tk.appl_no.isin(set(eps))
res['P0'], fr0 = links(tk)
res['V1'], _ = links(tk[~drop_us7])
res['V1E'], _ = links(tk[~drop_us7 & ~drop_ep9])
# 반대 규칙: 긴형(8자리·10자) 행을 빼고 짧은형 유지
long_us = set(us7.str.zfill(8)); long_ep = set('0' + eps)
res['V1E_keep_short'], _ = links(tk[~((tk.cc == 'US') & tk.appl_no.isin(long_us)) & ~((tk.cc == 'EP') & tk.appl_no.isin(long_ep))])
# 정규화 키 기준 고유 (cc, 키, KR문서, fk) 연결
fr0 = fr0.assign(k=nk(fr0.appl_no))
z = fr0.drop_duplicates(['cc', 'k', 'kr_appl_no', 'fk', 'kk'])
res['normkey_dedup'] = {cc: [int(len(g)), float((g.fk == g.kk).mean())] for cc, g in z.groupby('cc')}
z2 = fr0.drop_duplicates(['cc', 'k', 'kr_appl_no'])
res['normkey_dedup_docpair'] = {cc: [int(len(g)), float((g.fk == g.kk).mean())] for cc, g in z2.groupby('cc')}
json.dump(res, open(os.path.join(D, 'b_fam.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
for k, v in res.items(): print(k, v)
