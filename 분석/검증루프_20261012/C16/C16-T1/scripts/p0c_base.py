# -*- coding: utf-8 -*-
"""P0-c: 고정 기준선 6변형(동결 base_scores_mode('final')와 같은 학습) + SBB(VERIFY_R5 sens_t1 b0 = C15 t1_base b0 규칙) 재적합·저장.
학습 레이블은 이 폴더에서 동결 r3.reg_lab·pub_lab(가드 2019, cap 2019-12-31)으로 다시 계산해 C15 btr_labels와 대조.
산출: models/base_{key}.pkl|json, models/sbb.json, p0/base6.parquet(평가 패널 전체 행), p0/sbb_keep.npy, p0/p0c.json"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from c16t1 import *
import gc, pickle, time
import xgboost as xgb
import final_run_r3 as F, r3, r2
from r2 import fr
t0 = time.time(); M = F.MODES['final']; chk = {}
if (P0D/'p0c.json').exists(): print('exists'); sys.exit(0)
BASE = list(r3.BASE)
btr = pd.concat([pd.read_parquet(FC/f'base_app_c{t}.parquet') for t in M['train_cuts']], ignore_index=True)
bte = pd.read_parquet(FC/'base_app_c2019.parquet')
r2.light_load()
with F.guard(M['guard_train']):
    btr['y'] = r3.reg_lab(btr, M['cap']); btr['y_pub'] = r3.pub_lab(btr, M['cap'])
btr['y_u'] = np.maximum(btr.y.values, btr.y_pub.values)
z = np.load(C15/'cache'/'t1'/'btr_labels.npz'); chk['btr_labels_eq_C15'] = bool(np.array_equal(np.asarray(z['y']), btr.y.values) and np.array_equal(np.asarray(z['y_pub']), btr.y_pub.values))
off = pd.read_parquet(FC/'baseline_scores.parquet'); S = {}
for lab in ('y', 'y_u'):
    t2 = btr.assign(y=btr[lab].values)
    Ls = fr.logit_fit(t2, BASE); pickle.dump(Ls, open(MD/f'base_logit_stack_{lab}.pkl', 'wb')); S[f'logit_stack_{lab}'] = fr.logit_pred(Ls, bte)
    L1 = fr.logit_fit(t2[t2.cut == max(M['train_cuts'])], BASE); pickle.dump(L1, open(MD/f'base_logit_single_{lab}.pkl', 'wb')); S[f'logit_single_{lab}'] = fr.logit_pred(L1, bte)
    xb = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, tree_method='hist', nthread=2, random_state=0)
    xb.fit(t2[BASE].values.astype(np.float32), t2.y.values); xb.get_booster().save_model(str(MD/f'base_xgb_{lab}.json'))
    S[f'xgb_{lab}'] = xb.predict_proba(bte[BASE].values.astype(np.float32))[:, 1]
df = pd.DataFrame({k: np.asarray(S[k]).astype(np.float32) for k in off.columns})
chk['base6_maxdiff_vs_official'] = {k: float(np.abs(df[k].values - off[k].values).max()) for k in off.columns}
df.to_parquet(P0D/'base6.parquet', index=False); del btr; gc.collect()
# SBB
TRC = ['cut', 'k', 'newc', 'y_reg', 'y_pub', 'lsize', 'lin', 'lout', 'inshare', 'dens', 'kpop', 'n_last1', 'gap_last', 'now1', 'now3', 'now_grow', 'k_grow', 'k_newent', 'vis_frac']
SBBF = ['lsize', 'lin', 'lout', 'inshare', 'dens', 'kpop', 'k', 'newc', 'n_last1', 'gap_last', 'now1', 'now3', 'now_grow', 'k_grow', 'k_newent', 'vis_frac']
te = pd.read_parquet(FC/'prep_te.parquet', columns=['k', 'newc', 'prior_any'] + SBBF[:6] + SBBF[8:])
keep = ~te.prior_any.values
tr = pd.read_parquet(FC/'prep_tr.parquet', columns=TRC); tr['y'] = np.maximum(tr.y_reg.values, tr.y_pub.values)
rs = np.random.RandomState(2019); kp = (tr.y.values == 1) | (rs.rand(len(tr)) < 0.25); w = np.where(tr.y.values == 1, 1.0, 4.0)[kp]
m = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, tree_method='hist', nthread=2, random_state=0)
m.fit(tr.loc[kp, SBBF].values.astype(np.float32), tr.y.values[kp], sample_weight=w); m.get_booster().save_model(str(MD/'sbb.json'))
sbb_all = m.predict_proba(te[SBBF].values.astype(np.float32))[:, 1].astype(np.float32)
np.save(P0D/'sbb_all.npy', sbb_all)
ref = pd.read_parquet(T1/'VERIFY_R5'/'sens_t1_scores.parquet', columns=['SBB']).SBB.values
chk['sbb_keep_maxdiff_vs_VR5'] = float(np.abs(sbb_all[keep] - ref).max())
c15 = C15/'cache'/'t1'/'sbb_b0.npy'
if c15.exists(): chk['sbb_keep_maxdiff_vs_C15b0'] = float(np.abs(sbb_all[keep] - np.load(c15)).max())
chk.update(secs=round(time.time() - t0), peak=G.rss_peak())
jdump(chk, P0D/'p0c.json'); print(chk, flush=True)
