# v7: 두 분류기 관점 결합 — 50-class 노출(v5) + open-set 51-class 노출/50산업 신호비율(v6)을 함께
from exp import *
t0 = time.time()
def comb(cut, cap):
    a = pd.read_parquet(CACHE/f'panel_c{cut}_{cap}_c2009_base_.parquet')
    b = pd.read_parquet(CACHE/f'panel_c{cut}_{cap}_c2009_os51_.parquet')
    assert (a.firm.values == b.firm.values).all() and (a.k.values == b.k.values).all()
    for c in ['t_mean', 't_max', 't_top3', 't_hi', 't_rw', 't_nbr', 'in50']:
        a['o' + c] = b[c].values
    return a
tr = pd.concat([comb(c, CAP_TR) for c in (2002, 2003, 2004)], ignore_index=True); te = comb(2009, 'None')
base = json.load(open(T1/'baseline_v0.json'))['fixed']
OS = ['ot_mean', 'ot_max', 'ot_top3', 'ot_hi', 'ot_rw', 'ot_nbr', 'oin50']
cols = BASE + AGG + MOM + NBR + KMOM + OS
s = predict(fit_xgb(tr, cols), te, cols); met = metrics(te, s)
print('v7', fmt(met), time.time() - t0)
te[['firm', 'k', 'y', 'newc']].assign(s=s).to_parquet(CACHE/'v7_scores.parquet', index=False)
lb_add('v7', 'v5 피처 + open-set 51-class 노출 7개(평균·최대·상위3·고신뢰·최근성·확산·50산업 신호비율) 병용', '두 분류기 관점은 상보적: 상대 배분(50)과 절대 산업성(51)을 함께 쓰면 v5보다 낫다', met, base, '?', '')
