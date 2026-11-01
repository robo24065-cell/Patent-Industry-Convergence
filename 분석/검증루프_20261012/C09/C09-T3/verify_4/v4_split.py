"""verify_4 [B]: 최종 원점 t0=2021, AF5만 제거(S4a5)·AFK만 제거(S4ak) 재학습 (run_scen.py final 경로 사본, 출력은 verify_4).
사용: python v4_split.py S0|S4a5|S4ak|S4"""
import os, sys, json, time, copy
scen = sys.argv[1]
MY = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, MY)
sys.path.append(os.path.join(os.path.dirname(MY), 'scripts'))
from common import R3
SPEC = json.load(open(os.path.join(R3, 'final_spec_r3.json'), encoding='utf-8'))
os.environ.update(T3R3_VAR='pub', T3R3_CACHE='cache_pub_final', T3R3_MAXYEAR='2024', T3R3_LABEL_MAX='2024', T3R3_FINAL_UNLOCK='1',
                  T3R3_TRAIN_MAX_T0=str(SPEC['final']['train_max_t0']), T3R3_EXT_SUFFIX='_final', T3R3_EXTB_SUFFIX='_final')
import numpy as np, pandas as pd
import c09lib as L, c09core as C
assert os.path.dirname(C.COMP_DIR) == MY, C.COMP_DIR
from scipy.stats import rankdata
SUF = '_final'; CFG = copy.deepcopy(SPEC['cfg'])
for c in CFG['components']:
    for k in ('params',):
        if k in c and 'nthread' in c[k]: assert c[k]['nthread'] <= 2
AF5 = ['all_newapp_sh', 'all_newapp_n']
AFK = ['kr_newapp_sh', 'kr_newk_sh', 'kr_napp', 'kr_newapp_n', 'kr_hhi', 'app_mom', 'app_size', 'big_sh', 'top1_sh']
if scen.startswith('N_'):   # 사후 보조(사전 등록 외): gain 이 AF5 와 비슷한 비출원인 피처 2개를 XGB·음이항에서만 제거한 귀무 비교
    drop = set(scen[2:].split('+'))
    for c in CFG['components']:
        if c['type'] != 'logit': assert drop <= set(c['features']), drop; c['features'] = [f for f in c['features'] if f not in drop]
else:
    drop = {'S0': set(), 'S4a5': set(AF5), 'S4ak': set(AFK), 'S4': set(AF5 + AFK)}[scen]
    for c in CFG['components']: c['features'] = [f for f in c['features'] if f not in drop]
def prep(D):
    cf = pd.read_parquet(os.path.join(R3, f'extB_cell_pub{SUF}.parquet')); kf = pd.read_parquet(os.path.join(R3, f'extB_ksic_pub{SUF}.parquet'))
    n0 = len(D); D = D.merge(cf, on=['cell', 't0'], how='left').merge(kf, on=['ksic', 't0'], how='left'); assert len(D) == n0
    D['oa_mg_rel'] = D.oa_mg_g - D.oa_all_g if 'oa_all_g' in D else np.nan
    D['ix_sci_fx'] = D.ros_sci_rel * D.fx_mom; D['ix_sci_kr'] = D.ros_sci_rel * D.kr_mom; D['ix_oa_gap'] = D.oa_mg_g - D.kr_mom
    return D
_gw = C.get_wide
def gw(Dx, mt):
    w = _gw(Dx, mt)
    if 'ros_sci' not in w: w = prep(w)
    C._WIDE[mt] = w; return w
C.get_wide = gw
t = time.time(); ft = SPEC['final']['test_t0']; tmax = SPEC['final']['train_max_t0']
D = prep(C.get_D(t0s=list(range(C.FIRST_T0, tmax + 1)) + [ft]))
te = D[D.t0 == ft]; TAG = f'V4_{scen}_final'
comps = [np.asarray(C.comp_pred(c, D, ft, CFG, TAG), dtype=float) for c in CFG['components']]
p = np.average([rankdata(x) / len(te) for x in comps], axis=0, weights=CFG.get('weights', [1] * len(comps)))
p0 = C.baseline_pred(D, ft); m = C.metrics(te.y.values, te.yb.values, p); b = C.metrics(te.y.values, te.yb.values, p0)
sk = (m['auc'] - b['auc']) / (1 - b['auc'])
pd.DataFrame(dict(cell=te.cell.values, kr_W=te.kr_W.values, p=p, yb=te.yb.values, p_logit=comps[0], p_xgbc=comps[1], p_nb=comps[2])).to_parquet(os.path.join(MY, f'v4_preds_{scen}.parquet'))
json.dump(dict(scen=scen, auc=m['auc'], base_auc=b['auc'], skill=sk, n_feats={c['type']: len(c['features']) for c in CFG['components']}, secs=round(time.time() - t)),
          open(os.path.join(MY, f'v4_res_{scen}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(scen, 'AUC', round(m['auc'], 5), 'skill', round(sk, 5), {c['type']: len(c['features']) for c in CFG['components']}, round(time.time() - t), 's')
