# -*- coding: utf-8 -*-
"""verify_1 V1: 최종시험(t0=2021) 묶음 제거 재적합, 공식 경로(VERIFY_R5 r3core/r3lib, R3 cache_pub_final 읽기).
사용: python -B v1_final.py <seed_offset> <exp,exp,...>   exp ∈ full,no_pi,no_ros,no_ext
쓰기: verify_1\comp_final_s{k}\(성분 캐시), verify_1\v1_final_s{k}.json"""
import os, sys, json, copy, time, types
sys.dont_write_bytecode = True
OUT = os.path.dirname(os.path.abspath(__file__))
R5 = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\T3_기술셀부상예측\VERIFY_R5"
T3 = os.path.dirname(R5)
K = int(sys.argv[1]); EXPS = sys.argv[2].split(',')
os.environ.update(T3R3_VAR='pub', T3R3_CACHE=os.path.join(T3, 'R3', 'cache_pub_final'), T3R3_MAXYEAR='2024', T3R3_LABEL_MAX='2024', T3R3_FINAL_UNLOCK='1',
                  T3R3_TRAIN_MAX_T0='2018', T3R3_EXT_SUFFIX='_final', T3R3_EXTB_SUFFIX='_final')
sys.path.insert(0, R5)
import numpy as np, pandas as pd
import r3lib as L, r3core as C
import xgboost as _xgb
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(encoding='utf-8')
SPEC = json.load(open(os.path.join(T3, 'R3', 'final_spec_r3.json'), encoding='utf-8'))
def prep(D):
    cf = pd.read_parquet(os.path.join(R5, 'extB_cell_pub_final.parquet')); kf = pd.read_parquet(os.path.join(R5, 'extB_ksic_pub_final.parquet'))
    n0 = len(D); D = D.merge(cf, on=['cell', 't0'], how='left').merge(kf, on=['ksic', 't0'], how='left'); assert len(D) == n0
    return D
_gw = C.get_wide
def gw(Dx, mt):
    w = _gw(Dx, mt)
    if 'ros_sci' not in w: w = prep(w); C._WIDE[mt] = w
    return w
C.get_wide = gw
C.COMP_DIR = os.path.join(OUT, f'comp_final_s{K}'); os.makedirs(C.COMP_DIR, exist_ok=True)
def _train(p, d, n, *a, **k):
    p = dict(p); p['seed'] = p.get('seed', 0) + K; return _xgb.train(p, d, n, *a, **k)
C.xgb = types.SimpleNamespace(train=_train, DMatrix=_xgb.DMatrix)
G = {'full': [], 'no_pi': ['pi_prod_g2', 'pi_prod_g1', 'pi_ship_g2'], 'no_ros': ['ros_sci', 'ros_ppp']}
G['no_ext'] = G['no_pi'] + G['no_ros']
t = time.time(); D = prep(C.get_D(t0s=list(range(C.FIRST_T0, 2019)) + [2021]))
te = D[D.t0 == 2021]; p0 = C.baseline_pred(D, 2021); b = roc_auc_score(te.yb.values, p0)
fn = os.path.join(OUT, f'v1_final_s{K}.json'); res = json.load(open(fn, encoding='utf-8')) if os.path.exists(fn) else {}
for e in EXPS:
    cfg = copy.deepcopy(SPEC['cfg'])
    for c in cfg['components']: c['features'] = [f for f in c['features'] if f not in G[e]]
    p = C.score(cfg, D, 2021, f'v1_{e}'); a = roc_auc_score(te.yb.values, p)
    top = np.argsort(-p)[:504]
    res[e] = dict(auc=float(a), auc_base=float(b), skill=float((a - b) / (1 - b)), n_feat=[len(c['features']) for c in cfg['components']],
                  n_cells=int(len(te)), n_pos=int(te.yb.sum()), hits_top504=int(te.yb.values[top].sum()), secs=round(time.time() - t))
    print(K, e, res[e], flush=True)
    json.dump(res, open(fn, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
