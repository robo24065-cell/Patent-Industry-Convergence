# v8: 공동출원 파트너 노출(범위안 공동출원 코드 + 범위밖 applicantInfo 공동출원인 이름→코드) 추가. 시드 잡음 측정 포함.
from exp import *
from feats import partner_feats
t0 = time.time()
def comb(cut, cap):
    a = pd.read_parquet(CACHE/f'panel_c{cut}_{cap}_c2009_base_.parquet')
    b = pd.read_parquet(CACHE/f'panel_c{cut}_{cap}_c2009_os51_.parquet')
    for c in ['t_mean', 't_max', 't_top3', 't_hi', 't_rw', 't_nbr', 'in50']: a['o' + c] = b[c].values
    pp = CACHE/f'part_c{cut}_{cap}.parquet'
    if pp.exists(): p = pd.read_parquet(pp)
    else:
        d, c = build_panel(cut, 5, cap=None if cap == 'None' else cap)
        p = partner_feats(d[['fidx', 'k']].copy(), c)[['p_n', 'p_k', 'p_share']]; p.to_parquet(pp, index=False)
    for c in p.columns: a[c] = p[c].values
    return a
tr = pd.concat([comb(c, CAP_TR) for c in (2002, 2003, 2004)], ignore_index=True); te = comb(2009, 'None')
base = json.load(open(T1/'baseline_v0.json'))['fixed']
OS = ['ot_mean', 'ot_max', 'ot_top3', 'ot_hi', 'ot_rw', 'ot_nbr', 'oin50']
c7 = BASE + AGG + MOM + NBR + KMOM + OS; c8 = c7 + PART
print('partner coverage: firms with >=1 partner (valid)', (te.drop_duplicates('firm').p_n > 0).mean().round(3), ' newcomer', (te[te.newc].drop_duplicates('firm').p_n > 0).mean().round(3))
out = {}
for nm, cols in [('v7', c7), ('v8', c8)]:
    ms = [metrics(te, predict(fit_xgb(tr, cols, seed=sd), te, cols)) for sd in range(3)]
    out[nm] = {k: (np.mean([m[k] for m in ms]), np.std([m[k] for m in ms])) for k in ms[0]}
    print(nm, ' '.join(f'{k}={v[0]:.4f}±{v[1]:.4f}' for k, v in out[nm].items()))
met = {k: v[0] for k, v in out['v8'].items()}
lb_add('v8', '+공동출원 파트너 노출(파트너 수·k 보유 파트너 수·비율), 3시드 평균', '공동출원 파트너가 이미 있는 산업으로 따라 들어간다', met, base, '?',
       f"동일 3시드 v7 평균 auc_all={out['v7']['auc_all'][0]:.4f} auc_new={out['v7']['auc_new'][0]:.4f}; 시드 sd auc_all={out['v8']['auc_all'][1]:.4f} auc_new={out['v8']['auc_new'][1]:.4f}")
print(time.time() - t0)
