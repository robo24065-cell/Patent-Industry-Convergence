# -*- coding: utf-8 -*-
"""verify_1 V2: 개발 원점(2010-2018) 동결 spec(E40-auditD) 위 KOSIS 3피처 제거. R3 r3core 경로 + exps3b.prep과 같은 외부B 결합.
사용: python -B v1_dev.py <exp> <origins 쉼표> [seed_offset]   exp ∈ full,no_pi,no_ros
쓰기: verify_1\\comp_dev_s{k}\\, verify_1\\v1_dev_s{k}.json"""
import os, sys, json, copy, time, types
sys.dont_write_bytecode = True
OUT = os.path.dirname(os.path.abspath(__file__))
R3 = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\T3_기술셀부상예측\R3"
E = sys.argv[1]; ORIG = [int(x) for x in sys.argv[2].split(',')]; K = int(sys.argv[3]) if len(sys.argv) > 3 else 0
os.environ['T3R3_VAR'] = 'pub'
for k in ['T3R3_CACHE', 'T3R3_MAXYEAR', 'T3R3_LABEL_MAX', 'T3R3_FINAL_UNLOCK', 'T3R3_TRAIN_MAX_T0', 'T3R3_EXT_SUFFIX', 'T3R3_EXTB_SUFFIX']: os.environ.pop(k, None)
sys.path.insert(0, R3)
import numpy as np, pandas as pd
import r3core as C
import xgboost as _xgb
sys.stdout.reconfigure(encoding='utf-8')
SPEC = json.load(open(os.path.join(R3, 'final_spec_r3.json'), encoding='utf-8'))
def prep(D):
    if 'ros_sci' in D: return D
    cf = pd.read_parquet(os.path.join(R3, 'extB_cell_pub.parquet')); kf = pd.read_parquet(os.path.join(R3, 'extB_ksic_pub.parquet'))
    n0 = len(D); D = D.merge(cf, on=['cell', 't0'], how='left').merge(kf, on=['ksic', 't0'], how='left'); assert len(D) == n0
    return D
_gw = C.get_wide
def gw(Dx, mt):
    w = _gw(Dx, mt)
    if 'ros_sci' not in w: w = prep(w); C._WIDE[mt] = w
    return w
C.get_wide = gw
C.COMP_DIR = os.path.join(OUT, f'comp_dev_s{K}'); os.makedirs(C.COMP_DIR, exist_ok=True)
def _train(p, d, n, *a, **k):
    p = dict(p); p['seed'] = p.get('seed', 0) + K; return _xgb.train(p, d, n, *a, **k)
C.xgb = types.SimpleNamespace(train=_train, DMatrix=_xgb.DMatrix)
G = {'full': [], 'no_pi': ['pi_prod_g2', 'pi_prod_g1', 'pi_ship_g2'], 'no_ros': ['ros_sci', 'ros_ppp']}
cfg = copy.deepcopy(SPEC['cfg'])
for c in cfg['components']: c['features'] = [f for f in c['features'] if f not in G[E]]
t = time.time(); D = prep(C.get_D())
assert C.L.LABEL_MAX_YEAR == 2021 and D.t0.max() == 2018
fn = os.path.join(OUT, f'v1_dev_s{K}.json'); res = json.load(open(fn, encoding='utf-8')) if os.path.exists(fn) else {}
for v in ORIG:
    r = C.evaluate(cfg, D, origins=[v])[v]
    res.setdefault(E, {})[str(v)] = dict(auc=r['auc'], base_auc=r['base_auc'], skill=r['skill'], secs=round(time.time() - t))
    print(K, E, v, round(r['skill'], 6), round(time.time() - t), flush=True)
    json.dump(res, open(fn, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
