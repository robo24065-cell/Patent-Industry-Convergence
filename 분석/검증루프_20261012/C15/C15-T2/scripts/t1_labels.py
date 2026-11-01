# -*- coding: utf-8 -*-
"""T1 레이블(1회): (1) 최종 평가 레이블 = 동결 score_final()과 같은 계산(C10-T1 s07 labels()와 동일) → cache/t1/final_labels.parquet
(2) 고정 기준선 학습 레이블 = 동결 base_scores_mode('final')와 같은 계산(btr y·y_pub, guard 2019, cap 2019-12-31) → cache/t1/btr_labels.npz
대조: C10-T1 final_labels.parquet, VERIFY_R5 labels_c2019_official.parquet."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from t1common import *
import numpy as np, pandas as pd
import final_run_r3 as F, r3, r2

CT1.mkdir(parents=True, exist_ok=True)
FC = fcp(2019); M = F.MODES['final']
out = {}
if not (CT1/'final_labels.parquet').exists():
    te = pd.read_parquet(FC/'prep_te.parquet', columns=['firm', 'k', 'cut', 'newc', 'prior_any'])
    r2.light_load()
    with F.guard(2023):
        te['y_reg'] = r3.reg_lab(te, None); te['y_pub'] = r3.pub_lab(te, r3.END)
    te['y_u'] = np.maximum(te.y_reg.values, te.y_pub.values)
    te[['firm', 'k', 'newc', 'prior_any', 'y_reg', 'y_u']].to_parquet(CT1/'final_labels.parquet', index=False)
L = pd.read_parquet(CT1/'final_labels.parquet')
keep = ~L.prior_any.values
out['final'] = dict(rows=int(keep.sum()), pos=int(L.y_u.values[keep].sum()), pos_reg=int(L.y_reg.values[keep].sum()), new_rows=int(L.newc.values[keep].sum()))
C10 = HERE.parent.parent/'C10'/'C10-T1'/'cache'/'final_labels.parquet'
if C10.exists():
    c10 = pd.read_parquet(C10); out['eq_C10_final_labels'] = bool((c10.y_u.values == L.y_u.values).all() and (c10.firm.values == L.firm.values).all())
V = T1/'VERIFY_R5'/'labels_c2019_official.parquet'
v = pd.read_parquet(V); out['eq_VR5_labels_official'] = bool((v.y_u.values == L.y_u.values).all() and (v.firm.values == L.firm.values).all() and (v.k.values == L.k.values).all())
out['VR5_mismatch_n'] = int((v.y_u.values != L.y_u.values).sum())
if not (CT1/'btr_labels.npz').exists():
    cf = dict(eval_cut=M['eval_cut'], train_cuts=M['train_cuts'], cap=M['cap']); r2.light_load()
    btr = pd.concat([pd.read_parquet(FC/f'base_app_c{t}.parquet', columns=['firm', 'k', 'cut']) for t in cf['train_cuts']], ignore_index=True)
    with F.guard(M['guard_train']):
        y = r3.reg_lab(btr, cf['cap']); yp = r3.pub_lab(btr, cf['cap'])
    np.savez(CT1/'btr_labels.npz', y=np.asarray(y), y_pub=np.asarray(yp))
z = np.load(CT1/'btr_labels.npz'); out['btr'] = dict(n=int(len(z['y'])), pos_y=int(np.asarray(z['y']).sum()), pos_yu=int(np.maximum(z['y'], z['y_pub']).sum()))
json.dump(out, open(CT1/'labels_check.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(out, 'peak', G.rss_peak())
