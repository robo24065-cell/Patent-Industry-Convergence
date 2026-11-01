# -*- coding: utf-8 -*-
"""P0-a: 법인 단계 모형(seed 0) 재적합·저장 + 학습/평가 파일(C15 t1_bprep b0와 같은 계산)을 이 폴더에 만든다.
입력(읽기 전용): C15-T2 cache/t1/A/o2019 (1단계 덤프), 트랙 prep_tr·prep_te.
산출: cache/models/firm_seed0.json, cache/p0/s1_tr.parquet, s1_te.parquet, s1_w.npy, s1_o.parquet, p0a.json"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from c16t1 import *
import gc, pickle, time
import pyarrow.parquet as pq, pyarrow as pa, xgboost as xgb
import final_run_r3 as F
t0 = time.time(); ec = json.loads(json.dumps(F.EC)); cols = list(ec['cols'])
if (P0D/'s1_w.npy').exists(): print('exists'); sys.exit(0)
meta = json.load(open(A15/'DONE.json', encoding='utf-8'))
ft = pickle.load(open(A15/'ftab.pkl', 'rb')); Ftr, Fte, xc = ft['Ftr'], ft['Fte'], ft['xc']
p = dict(n_estimators=500, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.7, min_child_weight=10, tree_method='hist', nthread=2, random_state=0)
fold = (pd.util.hash_array(Ftr.index.get_level_values('firm').values.astype(object)) % 2).astype(int)
otr = np.zeros(len(Ftr))
for fo in (0, 1):
    m = xgb.XGBClassifier(**p).fit(Ftr[xc].values[fold != fo].astype(np.float32), Ftr.y.values[fold != fo])
    otr[fold == fo] = m.predict_proba(Ftr[xc].values[fold == fo].astype(np.float32))[:, 1]
m = xgb.XGBClassifier(**p).fit(Ftr[xc].values.astype(np.float32), Ftr.y.values)
m.get_booster().save_model(str(MD/'firm_seed0.json')); json.dump(dict(xc=xc, params=p), open(MD/'firm_seed0_meta.json', 'w', encoding='utf-8'), ensure_ascii=False)
ote = m.predict_proba(Fte[xc].values.astype(np.float32))[:, 1]
mtr = pd.Series(otr, index=Ftr.index); mte = pd.Series(ote, index=Fte.index); del Ftr, Fte, ft
chk = {}
tr = pd.read_parquet(A15/'trpre.parquet'); assert len(tr) == meta['n_tr'] and (tr._row.values == np.arange(len(tr))).all()
tr['f_any'] = mtr.reindex(pd.MultiIndex.from_arrays([tr.cut.values, tr.firm.values])).values
chk['f_any_tr_maxdiff_vs_C15A'] = float(np.nanmax(np.abs(tr.f_any.values - np.load(A15/'f_any0_tr.npy'))))
y = np.load(A15/'y_pre.npy'); w = np.load(A15/'w_pre.npy'); tr['y'] = y
q = ec['neg_q']; rs = np.random.RandomState(1234 + 2019); kp = (y == 1) | (rs.rand(len(tr)) < q)
w = np.where(y == 1, w, w / q)[kp]; tr = tr.loc[kp].reset_index(drop=True)
rest = [x for x in cols if x not in tr.columns]
if rest:
    R = pq.read_table(FC/'prep_tr.parquet', columns=rest).take(pa.array(tr._row.values)).to_pandas()
    for x in rest: tr[x] = R[x].values
    del R
tr[cols + ['y']].to_parquet(P0D/'s1_tr.parquet', index=False); chk['train_rows'] = int(len(tr)); chk['train_pos'] = int(tr.y.sum()); del tr; gc.collect()
te = pd.read_parquet(A15/'te_ext.parquet'); o = pd.read_parquet(A15/'s1_o.parquet')
rest_te = [x for x in cols if x not in te.columns and x != 'f_any']
R = pd.read_parquet(FC/'prep_te.parquet', columns=rest_te + ['firm', 'k'])
assert (R.firm.values == o.firm.values).all() and (R.k.values == o.k.values).all()
for x in rest_te: te[x] = R[x].values
del R
te['f_any'] = mte.reindex(pd.MultiIndex.from_arrays([o.cut.values, o.firm.values])).values
chk['f_any_te_maxdiff_vs_C15A'] = float(np.nanmax(np.abs(te.f_any.values - np.load(A15/'f_any0_te.npy'))))
te[cols].to_parquet(P0D/'s1_te.parquet', index=False); o.to_parquet(P0D/'s1_o.parquet', index=False); np.save(P0D/'s1_w.npy', w)
fz = pd.read_parquet(RD/'o2019_s1_o.parquet'); chk['s1_o_eq_frozen'] = bool((fz.firm.values == o.firm.values).all() and (fz.k.values == o.k.values).all())
chk['w_eq_frozen'] = bool(np.array_equal(np.load(RD/'o2019_s1_w.npy'), w))
chk.update(secs=round(time.time() - t0), peak=G.rss_peak())
jdump(chk, P0D/'p0a.json'); print(chk, flush=True)
