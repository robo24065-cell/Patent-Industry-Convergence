# C18-T4 s2: 점추정(팔별 AUC, 층 안 AUC), 최선 공개 팔, 구조 진단, 이전 바퀴 값 재현 대조. 출력 cache/s2_point.json
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib18 import *
from sklearn.metrics import roc_auc_score
t = time.time()
R = {}
SETS = [('K1', v, None) for v in [2020, 2021, 2022, 2023, 2024]] + [('K2', 2023, None), ('K3', 2023, None)] + \
       [('K4', v, L) for v in [2021, 2022, 2023] for L in ['xgb', 'lr']] + \
       [('K1', v, 'lr') for v in [2020, 2021, 2022, 2023, 2024] if os.path.exists(C + rf'\fr_K1_{v}_lr.parquet')]  # 변경 1: K1 경량 LR
for K, v, L in SETS:
    F = pd.read_parquet(C + (rf'\fr_{K}_{v}_{L}.parquet' if L else rf'\fr_{K}_{v}.parquet'))
    arms = ['base', 'prov'] + PUB_ARMS[K]
    w = F.w.values.astype(np.float64)
    r = {'n': len(F), 'pos': int(F.y.sum()), 'wpos': float(w[F.y == 1].sum()), 'auc': {}, 'auc_strat': {}}
    for m in arms:
        r['auc'][m] = AUCPrep(F[m].values, F.y.values).auc(w)
        r['auc_strat'][m] = StratAUC(F[m].values, F.y.values, F.k.values).auc(w)
    # 검산: sklearn 가중 AUC, 층 AUC를 층별 직접 계산
    r['chk_sklearn_prov'] = float(roc_auc_score(F.y.values, F.prov.values, sample_weight=w))
    num = den = 0.0
    for kk, g in F.groupby('k'):
        if g.y.nunique() < 2:
            continue
        wp = g.w[g.y == 1].sum(); wn = g.w[g.y == 0].sum()
        num += roc_auc_score(g.y.values, g.prov.values, sample_weight=g.w.values) * wp * wn; den += wp * wn
    r['chk_strat_prov'] = float(num / den)
    # 구조 진단
    pos = F[F.y == 1]
    ksh = pos.groupby('k').w.sum() / pos.w.sum()
    r['G_eff_pos'] = float(1 / (ksh ** 2).sum())
    r['top10_k_pos_share'] = float(ksh.sort_values(ascending=False).head(10).sum())
    r['n_k_with_pos'] = int((ksh > 0).sum())
    fk = F.groupby('f').k.nunique()
    r['firms'] = int(F.f.nunique()); r['firms_multi_k'] = int((fk > 1).sum())
    r['median_k_per_firm_ge2units'] = float(F.groupby('f').filter(lambda g: len(g) >= 2).groupby('f').k.nunique().median()) if K == 'K4' else 1.0
    if K == 'K4':
        sh = F.groupby(['f', 'k']).size().groupby(level=0).apply(lambda s: s.max() / s.sum())
        n_f = F.groupby('f').size()
        r['median_modal_k_share_firms_ge2'] = float(sh[n_f[n_f >= 2].index].median())
    if K == 'K1':
        r['firms_per_s_median'] = float(F.groupby('s').f.nunique().median())
        r['s_per_firm_median'] = float(F.groupby('f').s.nunique().median())
        r['ks_cells'] = int(F.groupby(['k', 's']).ngroups)
        r['cc_EP_share_pos'] = float(pos.cc.mean())
    r['firms_per_k_median'] = float(F.groupby('k').f.nunique().median())
    R[f'{K}_{v}' + (f'_{L}' if L else '')] = r
    out(K, v, L, json.dumps({m: round(x, 5) for m, x in r['auc'].items()}), round(time.time() - t, 1))
# 최선 공개 팔(개발 원점 평균 AUC, 후보·학습기별)
best = {}
for K, L in [('K1', None), ('K2', None), ('K3', None), ('K4', 'xgb'), ('K4', 'lr'), ('K1', 'lr')]:
    keys = [k for k in R if k.startswith(K + '_') and ((L is None and k.count('_') == 1) or (L is not None and k.endswith('_' + L)))]
    if not keys:
        continue
    mean = {m: float(np.mean([R[k]['auc'][m] for k in keys])) for m in ['base', 'prov'] + PUB_ARMS[K]}
    b = max(PUB_ARMS[K], key=lambda m: mean[m])
    best[K + ('_' + L if L else '')] = {'best': b, 'mean_auc': mean, 'origins': keys}
R['best_public'] = best
# 이전 바퀴 값 재현 대조
R['repro'] = {
    'K1_2024_prov_vs_C12(0.9369)': R['K1_2024']['auc']['prov'],
    'K2_prov_vs_C13(0.85959)': R['K2_2023']['auc']['prov'], 'K2_hsc_vs_C13(0.85890)': R['K2_2023']['auc']['hsc'],
    'K4_2023_xgb_prov_vs_C17(0.65576)': R['K4_2023_xgb']['auc']['prov'], 'K4_2023_xgb_off_vs_C17(0.64966)': R['K4_2023_xgb']['auc']['off'],
}
R['sec'] = round(time.time() - t, 1)
json.dump(R, open(C + r'\s2_point.json', 'w'), indent=1, ensure_ascii=False)
out(json.dumps(best, indent=1)); out(json.dumps(R['repro'], indent=1)); out('done', R['sec'])
