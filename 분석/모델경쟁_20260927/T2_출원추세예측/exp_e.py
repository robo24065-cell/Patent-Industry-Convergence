# v36 후보: 외부데이터(KOSIS 광업제조업조사 KR 세세분류) 성장 피처를 전역 Ridge 에 추가
from lab2 import *
import final_run as FR
KOS = pd.read_csv(os.path.join(FR.T, '..', '..', '..', '공모전데이터', '작업데이터', 'kosis_광업제조업_50개산업.csv'), encoding='utf-8-sig', dtype={'세세분류코드': str})
KOS = KOS[KOS.기준 == '10차']
ks = list(D['idx'].get_level_values(0).unique()); kidx = np.array([ks.index(k) for k in D['idx'].get_level_values(0)])
def gmat(item):
    """연도 t 성장 = log S_t − log S_{t−1}; 2010 결측 → 2011 성장은 (2011−2009)/2, 2010 성장은 결측. t<2009 결측. 자료는 ≤t 만 사용"""
    p = KOS[KOS.항목코드 == item].pivot_table(index='세세분류코드', columns='연도', values='값').reindex(ks)
    G = np.full((len(ks), len(D['years'])), np.nan)
    yrs = sorted(p.columns)
    for a, b in zip(yrs[:-1], yrs[1:]):
        if b - FR.Y0 < G.shape[1]:
            G[:, b - FR.Y0] = (np.log(p[b].clip(lower=1)) - np.log(p[a].clip(lower=1))) / (b - a)
    return G[kidx]
EXT = {it: gmat(it) for it in ['T02', 'T04', 'T06']}
orig_feats = FR._ridge_feats
def feats_ext(E, c, items=('T02', 'T04')):
    F = orig_feats(E, c)
    na = np.zeros(len(F))
    for it in items:
        v = EXT[it][:, c]; na = np.maximum(na, ~np.isfinite(v)); F[f'ext_{it}'] = np.nan_to_num(v, nan=0.0)
    F['ext_na'] = na
    return F
for items in [('T02',), ('T04',), ('T02', 'T04'), ('T02', 'T04', 'T06')]:
    FR._ridge_feats = lambda E, c, items=items: feats_ext(E, c, items)
    R = member_cache('ridge_lead', dict(BASE['ridge_lead'], ext='-'.join(items)) ) if False else None
    out = {}
    for o in ORIG_ALL:
        out[o] = FR.ridge_lead(FR.truncate(D, o), o, H, **BASE['ridge_lead'])
    s = rep(rows_from(out), f'ridge+ext{items}')
    M = {k: member_cache(k, v) for k, v in BASE.items() if k != 'ridge_lead'}
    rep(rows_from(combine([out, M['flow_um'], M['topdown']])), f'  ens w/ ridge+ext{items}')
    import pickle; pickle.dump(out, open(os.path.join(FR.CACHE, f"mem_ridge_ext_{'-'.join(items)}.pkl"), 'wb'))
FR._ridge_feats = orig_feats
