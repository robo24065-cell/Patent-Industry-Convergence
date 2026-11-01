# C18-T2 verify_0 r2: EP 정규화 키 쌍 7,357 전체(짧은 쪽 제거) 규칙으로 V1E 재계산 + '0'+s 외 444쌍 성격
import os, json, pandas as pd
D = os.path.dirname(os.path.abspath(__file__))
ue = pd.read_pickle(os.path.join(D, 'us_ep.pkl')); kr = pd.read_pickle(os.path.join(D, 'kr.pkl')); tk = pd.read_pickle(os.path.join(D, 'tok.pkl'))
nk = lambda s: s.str.replace(r'[^0-9.]', '', regex=True).str.lstrip('0')
res = {}
ep = ue[ue.cc == 'EP'][['appl_no', 'title', 'appl_date', 'ksic']].drop_duplicates('appl_no')
ep['k'] = nk(ep.appl_no); ep['L'] = ep.appl_no.str.len()
g = ep[ep.duplicated('k', keep=False)].sort_values(['k', 'L'])
assert g.groupby('k').size().max() == 2
sh = g.groupby('k').head(1); lo = g.groupby('k').tail(1)
pp = sh.merge(lo, on='k', suffixes=('_s', '_l'))
pp['kind'] = pp.L_s.astype(str) + '-' + pp.L_l.astype(str)
pp['zero1'] = ('0' + pp.appl_no_s) == pp.appl_no_l
n = lambda s: s.fillna('').str.lower().str.split().str.join(' ')
pp['teq'] = n(pp.title_s) == n(pp.title_l); pp['deq'] = pp.appl_date_s == pp.appl_date_l
res['pairs'] = int(len(pp)); res['eq_len_pairs'] = int((pp.L_s == pp.L_l).sum())
res['by_kind'] = pp.groupby(['kind', 'zero1']).agg(n=('k', 'size'), teq=('teq', 'mean'), deq=('deq', 'mean')).reset_index().to_dict('records')
res['ex_nonzero1'] = pp.loc[~pp.zero1, ['appl_no_s', 'appl_no_l', 'teq', 'deq']].head(8).to_dict('records')
us = ue[ue.cc == 'US']; u8 = set(us.appl_no[us.appl_no.str.len() == 8])
us7 = set(us.appl_no[(us.appl_no.str.len() == 7) & us.appl_no.str.zfill(8).isin(u8)])
ep9 = set(pp.appl_no_s)
tk = tk.drop_duplicates(['cc', 'appl_no', 'fk', 'r9'])
t = tk[~((tk.cc == 'US') & tk.appl_no.isin(us7)) & ~((tk.cc == 'EP') & tk.appl_no.isin(ep9))]
fr = t.merge(kr, on='r9')
res['V1E_7357'] = {cc: [int(len(x)), int(x.appl_no.nunique()), int(x.kr_appl_no.nunique()), float((x.fk == x.kk).mean())] for cc, x in fr.groupby('cc')}
json.dump(res, open(os.path.join(D, 'c_ep.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
for k, v in res.items(): print(k, v)
