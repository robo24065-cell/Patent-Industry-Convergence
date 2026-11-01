# C09-T4 s4: N4 지연납부 2개 피처(late_share_2y·delay_mean_2y) 원판·R1·R2·R3 재계산 → 동결 ens 재학습·채점(진단).
#  입력(읽기 전용): N4 cache/final_panel.parquet·dev_panel.parquet(동결 실행이 만든 패널), cache/reg_py.parquet, final_spec_r4.json
#  n4lib.predict / cluster_boot / f_C04 계산을 그대로 옮겼다. 출력은 이 폴더 cache/n4_* 에만.
#  python s4_n4.py feats | final | dev
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
from c9lib import *
SPEC = json.load(open(os.path.join(N4D, 'final_spec_r4.json'), encoding='utf-8'))
FE, M0F = SPEC['features'], SPEC['baseline_M0']
SEEDS = tuple(SPEC['model']['seeds'])
XGB_PARAMS = SPEC['model']['xgb_params']
LATE = ['late_share_2y', 'delay_mean_2y']
RULES = ['orig', 'R1', 'R2', 'R3']


def _add_years(d, n):
    d = d.astype('Int64'); y = d // 10000; md = d % 10000
    md = md.where(md != 229, 228)
    return (y + n) * 10000 + md


def _to_dt(x):
    return pd.to_datetime(x.astype('Int64').astype(str), format='%Y%m%d', errors='coerce')


def feats():
    hol = load_hol()
    py = pd.read_parquet(os.path.join(N4D, 'cache', 'reg_py.parquet'))
    out = {}
    log = {}
    for mode in ['final', 'dev']:
        P = pd.read_parquet(os.path.join(N4D, 'cache', f'{mode}_panel.parquet'), columns=['applicantCd', 't0'] + LATE)
        res = {r: pd.DataFrame(index=P.index, columns=LATE, dtype=float) for r in RULES}
        for t0, B in P.groupby('t0'):
            end = t0 * 10000 + 1231; y2 = end - 20000
            q = py[(py.d <= end) & py.applicantCd.isin(set(B.applicantCd))]
            w = q[(q.d > y2) & (q.statAnnl >= 4) & q.rgstDate.notna()].copy()
            w['due'] = _add_years(w.rgstDate, (w.statAnnl - 1).astype(int))
            dd = _to_dt(w.due).values.astype('datetime64[D]'); pdd = _to_dt(w.d).values.astype('datetime64[D]')
            w['delay'] = (pdd - dd).astype('timedelta64[D]').astype(float)
            ok = ~np.isnat(dd)
            for r in ['R1', 'R2']:
                adj = dd.copy(); adj[ok] = adjust_due(dd[ok], r, hol)
                w['delay' + r] = (pdd - adj).astype('timedelta64[D]').astype(float)
            a = B['applicantCd']
            for r in RULES:
                if r == 'orig':
                    late = (w.delay > 0).astype(float); dpos = w.delay.clip(lower=0)
                elif r == 'R3':
                    late = (w.delay > 3).astype(float); dpos = w.delay.where(w.delay > 3, 0.0)
                else:
                    late = (w['delay' + r] > 0).astype(float); dpos = w['delay' + r].clip(lower=0)
                late = late.where(w.delay.notna()); dpos = dpos.where(w.delay.notna())
                g = pd.DataFrame({'a': w.applicantCd.values, 'late': late.values, 'dpos': dpos.values}).groupby('a')
                res[r].loc[B.index, 'late_share_2y'] = a.map(g.late.mean()).astype(float).values
                res[r].loc[B.index, 'delay_mean_2y'] = a.map(g.dpos.mean()).astype(float).values
        rep = {}
        for c in LATE:
            x = P[c].to_numpy(float); y = res['orig'][c].to_numpy(float)
            same = (np.isnan(x) & np.isnan(y)) | (np.abs(x - y) <= 1e-6 * np.maximum(1, np.abs(x)))
            rep[c] = float(1 - same.mean())
        chg = {r: {c: float(((res[r][c].fillna(-9) - res['orig'][c].fillna(-9)).abs() > 1e-12).mean()) for c in LATE} for r in ['R1', 'R2', 'R3']}
        log[mode] = dict(rows=len(P), repro_mismatch=rep, changed_share=chg)
        for r in RULES:
            res[r].astype('float64').to_parquet(os.path.join(C, f'n4_late_{mode}_{r}.parquet'))
        print(mode, rep, chg, flush=True)
    json.dump(log, open(os.path.join(C, 'n4_feat_log.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


def fit_pred(Xtr, ytr, Xte, model, seed=0):
    if model == 'xgb':
        from xgboost import XGBClassifier
        m = XGBClassifier(random_state=seed, **XGB_PARAMS)
        m.fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1], m
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    m = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1], m


def rank01(p):
    return pd.Series(p).rank(pct=True).values


def ens_models(tr, feats):
    X = tr[feats].astype(float); y = tr.y
    ms = [fit_pred(X, y, X.iloc[:1], 'xgb', s)[1] for s in SEEDS]
    ml = fit_pred(X, y, X.iloc[:1], 'logit')[1]
    return ms, ml


def ens_pred(ms, ml, Xte):
    px = np.mean([m.predict_proba(Xte)[:, 1] for m in ms], axis=0)
    pl = ml.predict_proba(Xte)[:, 1]
    return 0.5 * rank01(px) + 0.5 * rank01(pl)


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, p)


def skill(a, b):
    return (a - b) / (1 - b)


def paired_boot(y, pms, pb, cl, n, seed):
    """n4lib.cluster_boot 와 같은 재표집(같은 시드·같은 순서)으로 여러 모형 skill 과 차이를 함께 계산"""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); cl = pd.Series(cl).astype(str).values
    u, inv = np.unique(cl, return_inverse=True)
    idx_by = pd.Series(np.arange(len(y))).groupby(inv).apply(lambda s: s.values).values
    S = {k: [] for k in pms}
    for _ in range(n):
        pick = rng.integers(0, len(u), len(u))
        i = np.concatenate([idx_by[j] for j in pick])
        if y[i].min() == y[i].max():
            continue
        a0 = auc(y[i], pb[i])
        for k, p in pms.items():
            S[k].append(skill(auc(y[i], p[i]), a0))
    S = {k: np.array(v) for k, v in S.items()}
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    out = {f'skill_ci_{k}': q(v) for k, v in S.items()}
    for k in pms:
        if k != 'orig':
            out[f'delta_ci_{k}'] = q(S[k] - S['orig'])
    out['n_boot'] = len(S['orig'])
    return out


def gain_rank(m, feats):
    g = m.get_booster().get_score(importance_type='total_gain')
    s = pd.Series({f: g.get(f'f{i}', g.get(f, 0.0)) for i, f in enumerate(feats)}).sort_values(ascending=False)
    r = s.rank(ascending=False, method='min')
    return {c: int(r[c]) for c in LATE}, s.head(10).index.tolist()


def run(mode):
    t = time.time()
    D = pd.read_parquet(os.path.join(N4D, 'cache', f'{mode}_panel.parquet'))
    if mode == 'final':
        TASKS = [(list(range(2015, 2022)), 2023)]; NB = 1000
    else:
        TASKS = [(list(range(2015, v - 1)), v) for v in [2017, 2018, 2019, 2020, 2021]]; NB = 300
    alt = {r: pd.read_parquet(os.path.join(C, f'n4_late_{mode}_{r}.parquet')) for r in RULES}
    out = []
    for tr_t0, te_t0 in TASKS:
        trm = D.t0.isin(tr_t0).values; tem = (D.t0 == te_t0).values
        te = D[tem]; y = te.y.values
        Xtr0 = D.loc[trm, M0F].astype(float); Xte0 = te[M0F].astype(float)
        p0 = np.mean([fit_pred(Xtr0, D.y[trm], Xte0, 'xgb', s)[0] for s in SEEDS], axis=0)
        preds, imp = {}, {}
        for r in RULES:
            DD = D.copy()
            if r != 'orig':
                DD[LATE] = alt[r][LATE].values
            tr = DD[trm]; Xte = DD.loc[tem, FE].astype(float)
            ms, ml = ens_models(tr, FE)
            pc = ens_pred(ms, ml, Xte)
            preds[r] = pc
            gr, top10 = gain_rank(ms[0], FE)
            # 지연 피처 묶음 순열(기업 단위, 시드 0) → ens AUC 하락
            rng = np.random.default_rng(0); perm = rng.permutation(len(Xte))
            Xp = Xte.copy(); Xp[LATE] = Xte[LATE].values[perm]
            dperm = auc(y, pc) - auc(y, ens_pred(ms, ml, Xp))
            imp[r] = dict(gain_rank=gr, top10=top10, perm_auc_drop=float(dperm))
        a0 = auc(y, p0)
        bt = paired_boot(y, preds, p0, te.bizno.values, NB, te_t0)
        res = dict(t0=te_t0, n=int(len(te)), pos=int(y.sum()), auc_M0=float(a0),
                   auc={r: float(auc(y, p)) for r, p in preds.items()}, skill={r: float(skill(auc(y, p), a0)) for r, p in preds.items()},
                   importance=imp, **bt)
        out.append(res)
        print(mode, te_t0, {r: round(v, 4) for r, v in res['skill'].items()}, 'AUC', {r: round(v, 4) for r, v in res['auc'].items()}, f'{time.time() - t:.0f}s', flush=True)
    json.dump(out, open(os.path.join(C, f'n4_rescore_{mode}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    if sys.argv[1] == 'feats': feats()
    else: run(sys.argv[1])
