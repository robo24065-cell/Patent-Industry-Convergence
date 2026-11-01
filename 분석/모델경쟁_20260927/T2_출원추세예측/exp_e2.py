# v36 수정: 외부 성장 피처를 연도별 산업 간 평균 제거(기간효과 제거), 결측은 0(=평균 산업), 결측 지시변수 삭제
from exp_e import *
def feats_ext2(E, c, items):
    F = orig_feats(E, c)
    for it in items:
        v = EXT[it][:, c].copy()
        if np.isfinite(v).any():
            v = v - np.nanmean(v[::5])  # 산업 50개 기준 평균(행은 산업×5국 복제이므로 한 국가분)
        F[f'ext_{it}'] = np.nan_to_num(v, nan=0.0)
    return F
M = {k: member_cache(k, v) for k, v in BASE.items() if k != 'ridge_lead'}
for items in [('T02',), ('T04',), ('T02', 'T04')]:
    FR._ridge_feats = lambda E, c, items=items: feats_ext2(E, c, items)
    out = {o: FR.ridge_lead(FR.truncate(D, o), o, H, **BASE['ridge_lead']) for o in ORIG_ALL}
    rep(rows_from(out), f'ridge+ext2{items}')
    rep(rows_from(combine([out, M['flow_um'], M['topdown']])), f'  ens w/ ridge+ext2{items}')
    pickle.dump(out, open(os.path.join(FR.CACHE, f"mem_ridge_ext2_{'-'.join(items)}.pkl"), 'wb'))
FR._ridge_feats = orig_feats
