# -*- coding: utf-8 -*-
"""T1 묶음 준비: 1단계 덤프(t1_stageA)에서 묶음 b의 법인 단계 시드·음성 표집 시드로 학습·평가 파일을 만든다.
 python t1_bprep.py <c> <b>
동결 train_origin() 1단계 끝부분과 같은 순서: f_any(법인 단계 XGB, seed=firm) → 음성 25% 표집(RandomState(neg_base + c)) → 나머지 열(prep_tr, _row take)
→ RD/o{c}_s1_o.parquet, o{c}_s1_te.parquet(te[cols]), o{c}_s1_tr.parquet(tr[cols+['y']]), o{c}_s1_w.npy. 이후 2·3단계는 t1_btrain.py(동결 train_origin)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from t1common import *
import gc, pickle, time, shutil
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow as pa
import xgboost as xgb
import final_run_r3 as F

c = int(sys.argv[1]); b = int(sys.argv[2]); S = seeds(b); FCp = fcp(c); A = adir(c); RD = rd(b, c)
import time as _t
while (rd(b, c)/'LANE2').exists() and len(sys.argv) < 4: _t.sleep(15)   # 두 번째 차선이 같은 묶음을 처리 중이면 끝날 때까지 대기(중복 실행 방지)
s_tr, s_te, s_o, s_w = RD/f'o{c}_s1_tr.parquet', RD/f'o{c}_s1_te.parquet', RD/f'o{c}_s1_o.parquet', RD/f'o{c}_s1_w.npy'
if s_w.exists() or (RD/f'o{c}.parquet').exists():
    print('prep exists', c, b); sys.exit(0)
assert (A/'DONE.json').exists(), '1단계 덤프 없음'
t0 = time.time()
ec = json.loads(json.dumps(F.EC)); cols = list(ec['cols'])
meta = json.load(open(A/'DONE.json', encoding='utf-8'))
# ---- 법인 단계(s2lib.firm_stage와 같은 적합, seed만 묶음 값) ----
ft = pickle.load(open(A/'ftab.pkl', 'rb')); Ftr, Fte, xc = ft['Ftr'], ft['Fte'], ft['xc']
p = dict(n_estimators=500, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.7, min_child_weight=10, tree_method='hist', nthread=2, random_state=S['firm'])
fold = (pd.util.hash_array(Ftr.index.get_level_values('firm').values.astype(object)) % 2).astype(int)
otr = np.zeros(len(Ftr))
for fo in (0, 1):
    m = xgb.XGBClassifier(**p).fit(Ftr[xc].values[fold != fo].astype(np.float32), Ftr.y.values[fold != fo])
    otr[fold == fo] = m.predict_proba(Ftr[xc].values[fold == fo].astype(np.float32))[:, 1]
m = xgb.XGBClassifier(**p).fit(Ftr[xc].values.astype(np.float32), Ftr.y.values)
ote = m.predict_proba(Fte[xc].values.astype(np.float32))[:, 1]
mtr = pd.Series(otr, index=Ftr.index); mte = pd.Series(ote, index=Fte.index); del Ftr, Fte, ft, m
print('firm stage', b, S['firm'], round(time.time() - t0), G.rss_now(), flush=True)
# ---- 학습 패널(표집 전) ----
tr = pd.read_parquet(A/'trpre.parquet')
assert len(tr) == meta['n_tr'] and (tr._row.values == np.arange(len(tr))).all()
tr['f_any'] = mtr.reindex(pd.MultiIndex.from_arrays([tr.cut.values, tr.firm.values])).values
fchk = {}
if b == 0: fchk['f_any_tr_maxdiff_vs_stageA'] = float(np.nanmax(np.abs(tr.f_any.values - np.load(A/'f_any0_tr.npy'))))
y = np.load(A/'y_pre.npy'); w = np.load(A/'w_pre.npy'); tr['y'] = y
q = ec['neg_q']
rs = np.random.RandomState(S['neg_base'] + c); kp = (y == 1) | (rs.rand(len(tr)) < q)
w = np.where(y == 1, w, w / q)[kp]; tr = tr.loc[kp].reset_index(drop=True)
rest = [x for x in cols if x not in tr.columns]
if rest:
    R = pq.read_table(FCp/PTR, columns=rest).take(pa.array(tr._row.values)).to_pandas()
    for x in rest: tr[x] = R[x].values
    del R
print(f'[C15 b{b} o{c}] train rows {len(tr)} pos {int(tr.y.sum())} | rss {G.rss_now()}', flush=True)
tr[cols + ['y']].to_parquet(s_tr, index=False); ntr, npos = len(tr), int(tr.y.sum()); del tr; gc.collect()
# ---- 평가 패널: 1단계 외부 피처 열 + 나머지 cols(prep_te, 같은 행 순서) + f_any ----
te = pd.read_parquet(A/'te_ext.parquet'); o = pd.read_parquet(A/'s1_o.parquet')
rest_te = [x for x in cols if x not in te.columns and x != 'f_any']
if rest_te:
    R = pd.read_parquet(FCp/'prep_te.parquet', columns=rest_te + ['firm', 'k'])
    assert (R.firm.values == o.firm.values).all() and (R.k.values == o.k.values).all()
    for x in rest_te: te[x] = R[x].values
    del R
te['f_any'] = mte.reindex(pd.MultiIndex.from_arrays([o.cut.values, o.firm.values])).values
if b == 0: fchk['f_any_te_maxdiff_vs_stageA'] = float(np.nanmax(np.abs(te.f_any.values - np.load(A/'f_any0_te.npy'))))
te[cols].to_parquet(s_te, index=False); del te
shutil.copyfile(A/'s1_o.parquet', s_o)
np.save(s_w, w)
json.dump(dict(c=c, b=b, seeds=S, train_rows=int(ntr), train_pos=npos, rest_tr=rest, rest_te=rest_te, secs=round(time.time() - t0), peak_gb=G.rss_peak(), **fchk),
          open(RD/f'o{c}_prep.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('prep saved', c, b, fchk, round(time.time() - t0), 'peak', G.rss_peak(), flush=True)
