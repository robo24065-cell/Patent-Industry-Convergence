# -*- coding: utf-8 -*-
"""T1 묶음별 기준선: (1) M0 고정 기준선 6변형 = 동결 base_scores_mode('final')와 같은 학습(로지스틱 4개는 결정적이라 공식 baseline_scores 값 그대로,
XGB 2개 xgb_y·xgb_y_u는 random_state만 묶음 값) (2) 강화 기준선 SBB = VERIFY_R5 sens_t1.py와 같은 학습(음성 표집 RandomState·XGB random_state만 묶음 값).
 python t1_base.py <b>  → cache/t1/base_b{b}.parquet (6열 float32, 평가 패널 전체 행), cache/t1/sbb_b{b}.npy (keep 행, float32)"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from t1common import *
import gc, time
import numpy as np, pandas as pd
import xgboost as xgb
import final_run_r3 as F, r3

b = int(sys.argv[1]); S = seeds(b); FC = fcp(2019); M = F.MODES['final']; t0 = time.time()
fb, fs_ = CT1/f'base_b{b}.parquet', CT1/f'sbb_b{b}.npy'
off = pd.read_parquet(FC/'baseline_scores.parquet')
chk = {}
if not fb.exists():
    BASE = list(r3.BASE)
    btr = pd.concat([pd.read_parquet(FC/f'base_app_c{t}.parquet', columns=BASE + ['firm', 'k', 'cut']) for t in M['train_cuts']], ignore_index=True)
    bte = pd.read_parquet(FC/f'base_app_c{M["eval_cut"]}.parquet', columns=BASE + ['firm', 'k'])
    z = np.load(CT1/'btr_labels.npz'); assert len(z['y']) == len(btr)
    btr['y'] = z['y']; btr['y_pub'] = z['y_pub']; btr['y_u'] = np.maximum(btr.y.values, btr.y_pub.values)
    S6 = {k: off[k].values for k in off.columns}
    for lab in ('y', 'y_u'):
        xb = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, tree_method='hist', nthread=2, random_state=S['base_xgb'])
        xb.fit(btr[BASE].values.astype(np.float32), btr[lab].values); S6[f'xgb_{lab}'] = xb.predict_proba(bte[BASE].values.astype(np.float32))[:, 1]
        print('xgb', lab, round(time.time() - t0), G.rss_now(), flush=True)
    df = pd.DataFrame({kk: np.asarray(S6[kk]).astype(np.float32) for kk in off.columns})
    if b == 0:
        chk['xgb_y_maxdiff_vs_official'] = float(np.abs(df.xgb_y.values - off.xgb_y.values).max())
        chk['xgb_y_u_maxdiff_vs_official'] = float(np.abs(df.xgb_y_u.values - off.xgb_y_u.values).max())
    df.to_parquet(fb, index=False); del btr, bte; gc.collect()
if not fs_.exists():
    L = pd.read_parquet(CT1/'final_labels.parquet'); keep = ~L.prior_any.values
    TRC = ['cut', 'k', 'newc', 'y_reg', 'y_pub', 'lsize', 'lin', 'lout', 'inshare', 'dens', 'kpop', 'n_last1', 'gap_last', 'now1', 'now3', 'now_grow', 'k_grow', 'k_newent', 'vis_frac']
    SBBF = ['lsize', 'lin', 'lout', 'inshare', 'dens', 'kpop', 'k', 'newc', 'n_last1', 'gap_last', 'now1', 'now3', 'now_grow', 'k_grow', 'k_newent', 'vis_frac']
    te = pd.read_parquet(FC/'prep_te.parquet', columns=['lsize', 'lin', 'lout', 'inshare', 'dens', 'kpop', 'n_last1', 'gap_last', 'now1', 'now3', 'now_grow', 'k_grow', 'k_newent', 'vis_frac']).loc[keep].reset_index(drop=True)
    o = pd.read_parquet(R3/'runs'/'FROZEN_final'/'o2019.parquet', columns=['k', 'newc', 'prior_any']); assert (o.prior_any.values == L.prior_any.values).all()
    T = o[keep].reset_index(drop=True)
    for x in te.columns: T[x] = te[x].values
    tr = pd.read_parquet(FC/'prep_tr.parquet', columns=TRC); tr['y'] = np.maximum(tr.y_reg.values, tr.y_pub.values)
    assert tr.cut.max() + 4 <= 2019
    rs = np.random.RandomState(S['sbb_neg']); kp = (tr.y.values == 1) | (rs.rand(len(tr)) < 0.25); w = np.where(tr.y.values == 1, 1.0, 4.0)[kp]
    m = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, tree_method='hist', nthread=2, random_state=S['sbb_xgb'])
    m.fit(tr.loc[kp, SBBF].values.astype(np.float32), tr.y.values[kp], sample_weight=w)
    sbb = m.predict_proba(T[SBBF].values.astype(np.float32))[:, 1].astype(np.float32)
    if b == 0:
        ref = pd.read_parquet(T1/'VERIFY_R5'/'sens_t1_scores.parquet', columns=['SBB']).SBB.values
        chk['sbb_maxdiff_vs_VR5'] = float(np.abs(sbb - ref).max())
    np.save(fs_, sbb)
    print('sbb', round(time.time() - t0), G.rss_now(), flush=True)
if chk:
    json.dump(chk, open(CT1/f'base_b{b}_check.json', 'w', encoding='utf-8'), indent=1)
print('base done', b, chk, round(time.time() - t0), 'peak', G.rss_peak(), flush=True)
