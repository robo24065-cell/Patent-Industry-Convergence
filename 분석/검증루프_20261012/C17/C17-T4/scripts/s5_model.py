# C17-T4 s5: 원점별 팔 학습·평가. 인자: <Y 2021|2022|2023> <learner xgb|lr> <arms, 쉼표> [save_pred 0|1]
# P0 단계: 잠금 단위는 쓰지 않는다(평가 D < 2025-01-01 검사).
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib17 import *
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

Y = int(sys.argv[1]); LEARN = sys.argv[2]; ARMS = sys.argv[3].split(','); SAVE = len(sys.argv) > 4 and sys.argv[4] == '1'
SEED_M = int(sys.argv[5]) if len(sys.argv) > 5 else SEED   # 반박용 모형 시드(기본은 본 실행과 같음)
assert Y in (2021, 2022, 2023), 'P0 단계는 개발 원점만'
t_start = time.time()
F = pd.read_parquet(C + r'\feat_units.parquet')
tr = F[(F.D >= 20160101) & (F.D <= (Y - 1) * 10000 + 1115)]
ev = F[(F.D >= Y * 10000 + 101) & (F.D <= Y * 10000 + 1231)]
assert ev.D.max() < LOCK_START and ev.y.notna().all() and tr.y.notna().all()
tr = tr.sample(min(250000, len(tr)), random_state=SEED).reset_index(drop=True)
ev = ev.reset_index(drop=True)
TY = 2023 if Y == 2023 else Y   # 공개표 기준 연도 = 원점
# KOSIS 출하액 증가율
ko = pd.read_csv(B + r'\공모전데이터\작업데이터\kosis_광업제조업_50개산업.csv', dtype={'세세분류코드': str})
ko = ko[ko.항목코드 == 'T04']
KG = {}
for (k, base), g in ko.groupby(['세세분류코드', '기준']):
    v = g.set_index('연도').값
    for y in v.index:
        if y - 1 in v.index and v[y] > 0 and v[y - 1] > 0:
            if base == '11차' or (k, y) not in KG:
                KG[(k, y)] = float(np.log(v[y] / v[y - 1]))


def te(trk, trs, try_, k, s, m=50):
    df = pd.DataFrame({'k': trk.values, 's': trs.values, 'y': try_.values})
    pri = df.groupby('s').y.mean()
    g = df.groupby(['k', 's']).y.agg(['sum', 'count'])
    rate = (g['sum'] + m * pri.reindex(g.index.get_level_values(1)).values) / (g['count'] + m)
    idx = pd.MultiIndex.from_arrays([k.values, s.values])
    out_ = rate.reindex(idx).values
    return np.where(np.isnan(out_), pri.reindex(s.values).values, out_)


NUM = ['late_prev', 'n_pet', 'n_expx', 'n_stat', 'n_exp_prior', 'pay_prev_lead', 'org', 'n_app', 'fam_n', 'fam_cc', 'pend', 'gyear']


def base_X(d):
    X = d[NUM].astype(float).copy()
    X['s3'] = (d.s == 3).astype(float)
    for col, cats in [('ent_prev', ['L', 'S', 'M']), ('prio_g', ['none', 'US', 'JP', 'KR', 'CN', 'DE', 'other']), ('sec', list('ABCDEFGH'))]:
        for c in cats:
            X[f'{col}_{c}'] = (d[col] == c).astype(float)
    X['sc_rate'] = te(tr.sc.fillna('NA'), tr.s, tr.y, d.sc.fillna('NA'), d.s)
    return X


XB_tr, XB_ev = base_X(tr), base_X(ev)


def null_map(r):
    nc = pd.read_parquet(C + r'\null_cells.parquet')
    rng = np.random.default_rng(SEED + 1000 * r)
    order = np.lexsort((rng.random(len(nc)), nc.cell.values))
    cs = nc.cell.values[order]
    starts = np.r_[0, np.flatnonzero(cs[1:] != cs[:-1]) + 1, len(nc)]
    lab = nc.ksic.values
    perm = np.empty(len(nc), dtype=object)
    for a, b in zip(starts[:-1], starts[1:]):
        idx = order[a:b]
        perm[idx] = lab[rng.permutation(idx)]
    return pd.Series(perm, index=nc.appl8.values)


def labels(arm, d):
    if arm == 'prov':
        return d.k_prov
    if arm == 'free':
        return d[f'k_free_{TY}']
    if arm == 'hsc':
        return d[f'k_hsc_{TY}']
    if arm == 'off':
        return d.k_off
    if arm.startswith('null'):
        return d.appl8.map(NULLS[arm])
    raise ValueError(arm)


NULLS = {a: null_map(int(a[4:])) for a in ARMS if a.startswith('null')}


def ctx_X(arm, d, ktr, kd):
    X = pd.DataFrame(index=d.index)
    cats = sorted(ktr.unique())
    kd2 = kd.where(kd.isin(cats), 'UNK')
    for c in cats:
        X[f'k_{c}'] = (kd2 == c).astype(float)
    X['k_rate'] = te(ktr, tr.s, tr.y, kd2, d.s)
    sm = pd.Series((tr.ent_prev.isin(['S', 'M'])).astype(float).values).groupby(ktr.values).mean()
    X['k_small'] = kd2.map(sm).astype(float).fillna(float(tr.ent_prev.isin(['S', 'M']).mean())).values
    t0y = (d.t0 // 10000).values - 2
    X['k_kosis'] = [KG.get((k, int(y)), np.nan) for k, y in zip(kd2.values, t0y)]
    X['k_kosis_na'] = np.isnan(X.k_kosis.values).astype(float)
    X['k_kosis'] = X.k_kosis.fillna(0.0)
    return X


def fit_pred(Xtr, ytr, Xev):
    if LEARN == 'xgb':
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                              min_child_weight=20, n_jobs=2, random_state=SEED_M, tree_method='hist', eval_metric='logloss')
        m.fit(Xtr.values, ytr)
        return m.predict_proba(Xev.values)[:, 1]
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr.values)
    m = LogisticRegression(C=1.0, max_iter=2000)
    m.fit(sc.transform(Xtr.values), ytr)
    return m.predict_proba(sc.transform(Xev.values))[:, 1]


res = {'Y': Y, 'learner': LEARN, 'n_tr': len(tr), 'n_ev': len(ev), 'pos_ev': int(ev.y.sum()), 'auc': {}}
preds = pd.DataFrame({'appl8': ev.appl8, 'firm': ev.firm, 'k_prov': ev.k_prov, 's': ev.s, 'y': ev.y})
for arm in ARMS:
    t1 = time.time()
    if arm == 'base':
        Xtr, Xev = XB_tr, XB_ev
    else:
        ktr = labels(arm, tr).fillna('UNK').astype(str); kev = labels(arm, ev).fillna('UNK').astype(str)
        Xtr = pd.concat([XB_tr, ctx_X(arm, tr, ktr, ktr)], axis=1)
        Xev = pd.concat([XB_ev, ctx_X(arm, ev, ktr, kev)], axis=1)
    p = fit_pred(Xtr, tr.y.values, Xev)
    res['auc'][arm] = float(roc_auc_score(ev.y.values, p))
    if SAVE:
        preds[arm] = p
    out(Y, LEARN, arm, round(res['auc'][arm], 5), f'{time.time() - t1:.0f}s')
res['sec'] = round(time.time() - t_start, 1)
tag = f'{Y}_{LEARN}_{ARMS[0]}' + (f'-{ARMS[-1]}' if len(ARMS) > 1 else '') + (f'_seed{SEED_M}' if SEED_M != SEED else '')
json.dump(res, open(C + rf'\s5_{tag}.json', 'w'), indent=1)
if SAVE:
    preds.to_parquet(C + rf'\pred_{Y}_{LEARN}' + (f'_seed{SEED_M}' if SEED_M != SEED else '') + '.parquet', index=False)
