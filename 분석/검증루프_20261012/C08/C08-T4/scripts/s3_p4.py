# C08-T4 s3: P4 전진 평가(M0 vs M1). 인자: 모형 시드(20261012|20261013|20261014) 또는 'logit'. 예측을 cache\p4_pred_<arg>.parquet로 저장.
# P5(c) 모형 수준 보고용으로 인자 'pub' → 시드 20261012에서 M1의 KSIC 피처를 공개 배정판(TXT, CONC, HSC1)으로 바꾼 모형, 'shuf' → KSIC 라벨 섞음 5회.
import sys, os, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
from lib8 import *
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
ARG = sys.argv[1]
T = time.time()
M0 = ['age_y', 'util', 'days_to_due', 'overdue', 'n_due_win', 'next_ann', 'step', 'pat_late_n', 'pat_pay_n', 'last_lag', 'max_span',
      'own_late_r3', 'own_pay3', 'own_aband3', 'own_alive', 'app_type', 'n_app', 'n_transfer', 'n_lic', 'ipc_sec', 'nopay']
KF = ['k_late1', 'k_trend', 'k_aband1']
EVAL = [2018, 2019, 2020, 2021, 2022]
D = pd.concat([pd.read_parquet(os.path.join(C, f'feat_{Y}.parquet')) for Y in range(2015, 2023)], ignore_index=True)


def ksic_shuffle(D, r):
    """KSIC 라벨을 특허 간 무작위로 섞은 뒤(전 원점 같은 순열) 산업 피처를 다시 계산하는 대신, 라벨→피처 값 표를 원점별로 재매핑."""
    rng = np.random.default_rng(20261012 + r)
    ks = D.drop_duplicates('rgst_no')[['rgst_no', 'ksic']]
    perm = dict(zip(ks.rgst_no, ks.ksic.values[rng.permutation(len(ks))]))
    out = D.copy()
    newk = out.rgst_no.map(perm)
    for f in KF:
        tab = out.groupby(['origin', 'ksic'])[f'{f}_prov'].first()
        out[f'{f}_prov'] = pd.MultiIndex.from_arrays([out.origin, newk]).map(tab).values
    return out


def fit_pred(X, y, w, Xe, seed):
    m = xgb.XGBClassifier(tree_method='hist', max_depth=4, n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                          min_child_weight=20, n_jobs=2, random_state=seed, eval_metric='logloss')
    m.fit(X, y, sample_weight=w)
    return m.predict_proba(Xe)[:, 1]


def fit_logit(X, y, w, Xe):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    def prep(Z):
        Z = Z.copy()
        for c in ['app_type', 'ipc_sec']:
            for v in sorted(D[c].unique()):
                Z[f'{c}_{v}'] = (Z[c] == v).astype(float)
            Z = Z.drop(columns=c)
        for c in ['pat_late_n', 'pat_pay_n', 'n_app', 'n_transfer', 'n_lic']:
            Z[c] = np.log1p(Z[c])
        Z['last_lag_na'] = Z.last_lag.isna().astype(float); Z['last_lag'] = Z.last_lag.clip(-400, 400)
        Z['days_to_due'] = Z.days_to_due.clip(-400, 4000)
        return Z
    pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=1.0, max_iter=500))
    pipe.fit(prep(X), y, logisticregression__sample_weight=w)
    return pipe.predict_proba(prep(Xe))[:, 1]


variants = {}
if ARG in ('20261012', '20261013', '20261014', 'logit'):
    variants = {'M0': (M0, 'prov', D), 'M1': (M0 + [f'{f}_prov' for f in KF], 'prov', D)}
elif ARG == 'pub':
    variants = {f'M1_{a}': (M0 + [f'{f}_{a}' for f in KF], a, D) for a in ['TXT', 'CONC', 'HSC1']}
elif ARG == 'shuf':
    for r in range(5):
        variants[f'M1_shuf{r}'] = (M0 + [f'{f}_prov' for f in KF], 'prov', ksic_shuffle(D, r))
seed = int(ARG) if ARG.isdigit() else 20261012
preds, LOG = [], {}
for v in EVAL:
    tr = D[D.origin < v]
    rng = np.random.default_rng(20261012)
    keep = (tr.y.values == 1) | (rng.random(len(tr)) < 0.3)
    idx_tr = np.where(keep)[0]
    w = np.where(tr.y.values[idx_tr] == 1, 1.0, 1 / 0.3)
    ev = D[D.origin == v]
    out = ev[['rgst_no', 'origin', 'y', 'cd']].copy()
    for nm, (cols, _, DD) in variants.items():
        trd = DD[DD.origin < v].iloc[idx_tr]; evd = DD[DD.origin == v]
        if ARG == 'logit':
            out[nm] = fit_logit(trd[cols], trd.y.values, w, evd[cols])
        else:
            out[nm] = fit_pred(trd[cols], trd.y.values, w, evd[cols], seed)
        LOG[f'{v}_{nm}_auc'] = float(roc_auc_score(evd.y, out[nm]))
    preds.append(out)
    print(v, {k: round(x, 4) for k, x in LOG.items() if k.startswith(str(v))}, len(idx_tr), round(time.time() - T, 1), flush=True)
P = pd.concat(preds, ignore_index=True)
P.to_parquet(os.path.join(C, f'p4_pred_{ARG}.parquet'), index=False)
LOG['sec'] = round(time.time() - T, 1)
json.dump(LOG, open(os.path.join(C, f'p4_log_{ARG}.json'), 'w', encoding='utf-8'), indent=1)
