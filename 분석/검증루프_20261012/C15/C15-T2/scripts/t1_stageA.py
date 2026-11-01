# -*- coding: utf-8 -*-
"""T1 1단계(외부 피처) 공용 덤프 — 동결 final_run_r3.train_origin() 1단계와 같은 계산을 하되 음성 표집 직전 상태를 저장한다.
 python t1_stageA.py <원점 c: 2019 또는 2008..2012>
바뀐 것(값 동일성은 P0로 확인):
 (1) 평가 패널 te는 학습 패널과 같은 slim 열 + 키만 읽어 외부 피처를 붙이고, 나머지 cols는 이 단계에서 붙이지 않는다(묶음 단계에서 같은 행 순서로 prep_te에서 붙임) — 메모리 절감.
 (2) 법인 단계(s2lib.firm_stage)는 같은 코드로 계산하되 법인 표(Ftr·Fte)를 저장해 묶음별 시드로 다시 적합할 수 있게 한다.
 (3) 음성 표집·나머지 열 읽기·학습 파일 쓰기는 묶음 단계(t1_bprep.py)로 넘긴다.
산출: cache/t1/A/o{c}/ trpre.parquet, y_pre.npy, w_pre.npy, f_any0_tr.npy, te_ext.parquet, f_any0_te.npy, s1_o.parquet, ftab.pkl, DONE.json"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from t1common import *
import gc, pickle, time
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow as pa
import final_run_r3 as F
import r2, r2ext, r3ext, r3ext2, r3ext3
from r2 import fr

c = int(sys.argv[1]); FCp = fcp(c); A = adir(c)
if (A/'DONE.json').exists():
    print('done already', c); sys.exit(0)
t0 = time.time()
ec = json.loads(json.dumps(F.EC)); ec['ext'] = list(ec['ext']); cols = list(ec['cols'])
for kind in ('design', 'um'):
    if (FCp/f'P_{kind}.npy').exists(): r3ext.PDIR[c] = FCp
r2.light_load()
from s2lib import FIRM_LEVEL
allc = pq.ParquetFile(FCp/PTR).schema.names
slimc = [x for x in dict.fromkeys(['firm', 'k', 'cut', 'newc', 'y_reg', 'y_pub', 'w_ipw', 'prior_pub', 't_lsum'] + [x for x in cols if x in FIRM_LEVEL]
                                  + ['t_max', 'ot_max', 't_top3', 'pa_k', 'ep_k', 'ti_max'] + ec.get('need_cols', [])) if x in allc]
tr = pd.read_parquet(FCp/PTR, columns=slimc); tr['_row'] = np.arange(len(tr), dtype=np.int64)
tec = pq.ParquetFile(FCp/'prep_te.parquet').schema.names
te_cols_slim = [x for x in dict.fromkeys(slimc + ['y_u', 'prior_any', 'y_reg', 'y_pub', 'newc', 'firm', 'k', 'cut'] + list(FIRM_LEVEL)) if x in tec]
te = pd.read_parquet(FCp/'prep_te.parquet', columns=te_cols_slim)
print('read', len(tr), len(te), G.rss_now(), flush=True)
assert not ec.get('train_filter_prior') and ec.get('train_cuts_n', 4) == 4
EXT = dict(r2ext.EXT); EXT.update(r3ext.EXT); EXT.update(r3ext2.EXT); EXT.update(r3ext3.EXT)
for g in ec.get('ext', []):
    tr, te = EXT[g](tr, te, c, False); gc.collect()
    print('ext', g, round(time.time() - t0), G.rss_now(), G.rss_peak(), flush=True)
tr['y'] = tr.y_reg.values; te['y'] = 0
if 'k_prior' in cols or 'k_prior_new' in cols: tr, te = fr.k_prior(tr, te)
assert 'f_any' in cols and ec.get('f_any_label', 'reg') == 'reg'
base_cols = [x for x in cols if x != 'f_any']
rest0 = [x for x in base_cols if x not in tr.columns and x in FIRM_LEVEL]
if rest0:
    R = pq.read_table(FCp/PTR, columns=rest0).take(pa.array(tr._row.values)).to_pandas()
    for x in rest0: tr[x] = R[x].values
    del R
rest0_te = [x for x in base_cols if x not in te.columns and x in FIRM_LEVEL]
assert not rest0_te, rest0_te   # te slim 이 법인 단계 피처를 모두 가져야 함


# ---- s2lib.firm_stage 와 같은 코드(법인 표 저장만 추가) ----
def firm_stage_capture(tr, te, cols, seed=0):
    import xgboost as xgb
    fcols = [x for x in cols if x in FIRM_LEVEL]
    def firm_tab(d):
        g = d.groupby(['cut', 'firm'], sort=False)
        X = g[fcols].first(); X['n_risk'] = g.size(); X['y'] = g.y.max()
        for x in ('t_max', 'ot_max', 't_top3', 'pa_k', 'ep_k', 'ti_max'):
            if x in d: X[x + '_fmax'] = g[x].max()
        return X
    Ftr = firm_tab(tr); Fte = firm_tab(te); xc = [x for x in Ftr.columns if x != 'y']
    pickle.dump(dict(Ftr=Ftr, Fte=Fte, xc=xc), open(A/'ftab.pkl', 'wb'), protocol=4)
    p = dict(n_estimators=500, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.7, min_child_weight=10, tree_method='hist', nthread=2, random_state=seed)
    fold = (pd.util.hash_array(Ftr.index.get_level_values('firm').values.astype(object)) % 2).astype(int)
    otr = np.zeros(len(Ftr))
    for fo in (0, 1):
        m = xgb.XGBClassifier(**p).fit(Ftr[xc].values[fold != fo].astype(np.float32), Ftr.y.values[fold != fo])
        otr[fold == fo] = m.predict_proba(Ftr[xc].values[fold == fo].astype(np.float32))[:, 1]
    m = xgb.XGBClassifier(**p).fit(Ftr[xc].values.astype(np.float32), Ftr.y.values)
    ote = m.predict_proba(Fte[xc].values.astype(np.float32))[:, 1]
    mtr = pd.Series(otr, index=Ftr.index); mte = pd.Series(ote, index=Fte.index)
    tr['f_any'] = mtr.reindex(pd.MultiIndex.from_arrays([tr.cut.values, tr.firm.values])).values
    te['f_any'] = mte.reindex(pd.MultiIndex.from_arrays([te.cut.values, te.firm.values])).values
    return tr, te


tr, te = firm_stage_capture(tr, te, base_cols)
print('firm_stage', round(time.time() - t0), G.rss_now(), G.rss_peak(), flush=True)
# ---- 표집 전 가중치·레이블(공식 순서 그대로) ----
w = tr.w_ipw.values.astype(np.float64) if ec.get('ipw', 'reg') == 'reg' else np.ones(len(tr))
w = np.where(tr.y_reg.values == 1, w, 1.0); y = tr.y_reg.values.astype(np.int8)
pw = ec.get('publab_w', 1.0)
if pw > 0:
    add = (tr.y_pub.values == 1) & (y == 0); y = np.maximum(y, tr.y_pub.values).astype(np.int8); w = np.where(add, pw, w)
assert ec.get('pos_mult', 1.0) == 1.0 and not ec.get('cut_weight_step', 0.0) and ec.get('neg_q')
np.save(A/'y_pre.npy', y); np.save(A/'w_pre.npy', w)
np.save(A/'f_any0_tr.npy', tr.f_any.values); np.save(A/'f_any0_te.npy', te.f_any.values)
keep_tr = [x for x in cols if x in tr.columns and x != 'f_any'] + ['_row', 'cut', 'firm']
keep_tr = list(dict.fromkeys(keep_tr))
sch = None; wr = None
for i in range(0, len(tr), 400000):
    t = pa.Table.from_pandas(tr.iloc[i:i + 400000][keep_tr], preserve_index=False)
    if wr is None: wr = pq.ParquetWriter(str(A/'trpre.parquet'), t.schema)
    wr.write_table(t); del t
wr.close()
n_tr = len(tr); del tr; gc.collect()
te[['firm', 'k', 'cut', 'newc', 'y_reg', 'y_u', 'prior_any']].to_parquet(A/'s1_o.parquet', index=False)
keep_te = [x for x in cols if x in te.columns and x != 'f_any']
te[keep_te].to_parquet(A/'te_ext.parquet', index=False)
json.dump(dict(c=c, n_tr=int(n_tr), n_te=int(len(te)), trpre_cols=keep_tr, te_ext_cols=keep_te, secs=round(time.time() - t0), peak_gb=G.rss_peak(),
               slimc=slimc, te_cols_slim=te_cols_slim), open(A/'DONE.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('saved A', c, round(time.time() - t0), 'peak', G.rss_peak(), flush=True)
