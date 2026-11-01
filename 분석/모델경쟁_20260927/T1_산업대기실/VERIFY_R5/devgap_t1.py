# T1 레드팀 — 개발(0.347) → 최종(0.473) 격차 점검: 같은 강화 기준선(SBB, 최근 활동 포함)을 개발 원점에서도 학습해 대비. 읽기 전용, 산출 devgap_t1.json
import sys, json, pathlib, time
import numpy as np, pandas as pd
from scipy.stats import rankdata
import xgboost as xgb
V = pathlib.Path(__file__).resolve().parent; R3 = V.parent/'R3'
sk = lambda a, b: (a - b) / (1 - b)
def auc(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
F = ['lsize', 'lin', 'lout', 'inshare', 'dens', 'kpop', 'k', 'newc', 'n_last1', 'gap_last', 'now1', 'now3', 'now_grow', 'k_grow', 'k_newent', 'vis_frac']
out = {}
for c in (2010, 2012, 2015, 2019):
    t = time.time()
    FC = R3/'cache_final'/'o2019' if c == 2019 else R3/'cache'/f'o{c}'
    tr = pd.read_parquet(FC/'prep_tr.parquet', columns=F + ['y_reg', 'y_pub', 'cut']); tr['y'] = np.maximum(tr.y_reg.values, tr.y_pub.values)
    assert tr.cut.max() + 4 <= max(c, 2019) and (c == 2019 or tr.cut.max() + 4 <= c)
    rs = np.random.RandomState(2019); kp = (tr.y.values == 1) | (rs.rand(len(tr)) < 0.25); w = np.where(tr.y.values == 1, 1.0, 4.0)[kp]
    m = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, tree_method='hist', nthread=2, random_state=0)
    m.fit(tr.loc[kp, F].values.astype(np.float32), tr.y.values[kp], sample_weight=w); del tr
    o = pd.read_parquet(R3/'runs'/('FROZEN_final' if c == 2019 else 'FROZEN_dev')/f'o{c}.parquet', columns=['firm', 'px', 'pl', 'prior_any'] + ([] if c == 2019 else ['y_u']))
    if c == 2019: o['y_u'] = pd.read_parquet(V/'labels_c2019_official.parquet').y_u.values
    te = pd.read_parquet(FC/'prep_te.parquet', columns=F)
    bs = pd.read_parquet(FC/'baseline_scores.parquet')
    keep = ~o.prior_any.values; y = o.y_u.values[keep].astype(int); n = keep.sum()
    s = 0.9 * rankdata(o.px.values[keep]) / n + 0.1 * rankdata(o.pl.values[keep]) / n
    sbb = m.predict_proba(te[F].values[keep].astype(np.float32))[:, 1]
    bst = max(auc(y, bs[c_].values[keep]) for c_ in bs.columns); am = auc(y, s); ab = auc(y, sbb)
    act = te.now3.values[keep] > 0
    out[str(c)] = dict(pairs=int(n), pos=int(y.sum()), rate=float(y.mean()), auc_model=am, auc_official_strongest=bst, auc_SBB=ab,
                       skill_official=sk(am, bst), skill_vs_SBB=sk(am, ab), SBB_skill_vs_official=sk(ab, bst),
                       share_pairs_inactive_now3_0=float(1 - act.mean()), rate_inactive=float(y[~act].mean()), rate_active=float(y[act].mean()),
                       skill_official_active_only=sk(auc(y[act], s[act]), max(auc(y[act], bs[c_].values[keep][act]) for c_ in bs.columns)), secs=round(time.time() - t))
    print(c, json.dumps(out[str(c)], ensure_ascii=False), flush=True)
json.dump(out, open(V/'devgap_t1.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
