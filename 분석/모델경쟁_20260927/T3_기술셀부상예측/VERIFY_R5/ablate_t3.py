# T3 레드팀 — 준-정합(스냅샷) 피처 묶음의 최종시험 기여 상한 추정 + 난수 시드 민감도. 공식 모형은 바꾸지 않는다(영향 크기 추정용 재적합).
# 동결 패널 캐시는 읽기만, 성분 캐시·결과는 VERIFY_R5/ 아래.
# 사용: python ablate_t3.py pub full|no_pi|no_ros|no_name|no_ipcco|seed1|seed2
#       python ablate_t3.py app full|no_pi|no_name|no_ipcco|seed1|seed2
import os, sys, json, copy, time, types
HERE = os.path.dirname(os.path.abspath(__file__)); T3 = os.path.dirname(HERE)
VAR, EXP = sys.argv[1], sys.argv[2]
if VAR == 'pub':
    os.environ.update(T3R3_VAR='pub', T3R3_CACHE=os.path.join(T3, 'R3', 'cache_pub_final'), T3R3_MAXYEAR='2024', T3R3_LABEL_MAX='2024', T3R3_FINAL_UNLOCK='1',
                      T3R3_TRAIN_MAX_T0='2018', T3R3_EXT_SUFFIX='_final', T3R3_EXTB_SUFFIX='_final')
    sys.path.insert(0, HERE)
    import numpy as np, pandas as pd
    import r3lib as L, r3core as C
    SPEC = json.load(open(os.path.join(HERE, 'final_spec_r3.json'), encoding='utf-8')); TEST = [2021]; TMAX = 2018
    def prep(D):
        cf = pd.read_parquet(os.path.join(HERE, 'extB_cell_pub_final.parquet')); kf = pd.read_parquet(os.path.join(HERE, 'extB_ksic_pub_final.parquet'))
        n0 = len(D); D = D.merge(cf, on=['cell', 't0'], how='left').merge(kf, on=['ksic', 't0'], how='left'); assert len(D) == n0
        D['oa_mg_rel'] = D.oa_mg_g - D.oa_all_g if 'oa_all_g' in D else np.nan
        D['ix_sci_fx'] = D.ros_sci_rel * D.fx_mom; D['ix_sci_kr'] = D.ros_sci_rel * D.kr_mom; D['ix_oa_gap'] = D.oa_mg_g - D.kr_mom
        return D
    _gw = C.get_wide
    def gw(Dx, mt):
        w = _gw(Dx, mt)
        if 'ros_sci' not in w: w = prep(w); C._WIDE[mt] = w
        return w
    C.get_wide = gw
    C.COMP_DIR = os.path.join(HERE, 'comp_ablate_pub'); os.makedirs(C.COMP_DIR, exist_ok=True)
    getD = lambda: prep(C.get_D(t0s=list(range(C.FIRST_T0, TMAX + 1)) + TEST))
else:
    os.environ.update(T3R2_CACHE=os.path.join(T3, 'R2', 'cache_2023'), T3R2_MAXYEAR='2023', T3R2_LABEL_MAX='2023', T3R2_FINAL_UNLOCK='1', T3R2_NO_HIER='1',
                      T3R2_TRAIN_MAX_T0='2016', T3R2_EXT_SUFFIX='_final', T3R2_COMP_DIR=os.path.join(HERE, 'comp_ablate_app'))
    sys.path.insert(0, HERE)
    import numpy as np, pandas as pd
    import r2core as C
    SPEC = json.load(open(os.path.join(HERE, 'final_spec_r2.json'), encoding='utf-8')); TEST = [2019, 2020]; TMAX = 2016
    getD = lambda: C.get_D(t0s=list(range(C.FIRST_T0, TMAX + 1)) + TEST)
import xgboost as _xgb
from sklearn.metrics import roc_auc_score
GROUPS = {
    'no_pi': ['pi_prod_g2', 'pi_prod_g1', 'pi_ship_g2'],
    'no_ros': ['ros_sci', 'ros_ppp'],
    'no_name': ['app_mom', 'app_size', 'big_sh', 'top1_sh', 'kr_newapp_sh', 'kr_newk_sh', 'all_newapp_sh', 'kr_napp', 'kr_newapp_n', 'kr_hhi', 'all_newapp_n'],
    'no_ipcco': ['kr_newpair_sh', 'all_newpair_sh', 'kr_multi_sh', 'nb_kmom', 'nb_tmom', 'nb_w', 'nb_deg'],
}
cfg = copy.deepcopy(SPEC['cfg']); tag = f'abl_{EXP}'
if EXP in GROUPS:
    for c in cfg['components']: c['features'] = [f for f in c['features'] if f not in GROUPS[EXP]]
elif EXP.startswith('seed'):
    off = int(EXP[4:])
    def _train(p, d, n, *a, **k):
        p = dict(p); p['seed'] = p.get('seed', 0) + off; return _xgb.train(p, d, n, *a, **k)
    C.xgb = types.SimpleNamespace(train=_train, DMatrix=_xgb.DMatrix)
t = time.time(); D = getD(); res = {}
for v in TEST:
    te = D[D.t0 == v]; p = C.score(cfg, D, v, tag); p0 = C.baseline_pred(D, v)
    a, b = roc_auc_score(te.yb.values, p), roc_auc_score(te.yb.values, p0); res[str(v)] = dict(auc=a, auc_base=b, skill=(a - b) / (1 - b))
out = dict(variant=VAR, exp=EXP, removed=GROUPS.get(EXP, []), n_feat=[len(c['features']) for c in cfg['components']], by_origin=res,
           mean_skill=float(np.mean([r['skill'] for r in res.values()])), mean_auc=float(np.mean([r['auc'] for r in res.values()])), secs=round(time.time() - t))
fn = os.path.join(HERE, f'ablate_{VAR}.json'); allr = json.load(open(fn, encoding='utf-8')) if os.path.exists(fn) else {}
allr[EXP] = out; json.dump(allr, open(fn, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(EXP, json.dumps(out, ensure_ascii=False))
