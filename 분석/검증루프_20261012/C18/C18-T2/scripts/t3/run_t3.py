"""C18-T2 T3 변형 재학습·채점 (C09-T3 run_scen.py 사본: 시나리오 피처 덮어쓰기 제거, 변형 캐시 선택만 추가).
사용: python run_t3.py --var P0|V1|V1E|V2|V12 --mode dev|final [--origins 2010,2011]
 P0 = 트랙 공식 캐시(R3/cache_pub(_final), 읽기 전용). 변형 = C18-T2/cache/t3_<V>_<dev|fin>/ (s3t_base → build_panel → build_feat2 사본으로 생성).
동결 spec(final_spec_r3.json)의 성분·하이퍼파라미터·가중치·학습 규칙·적격·레이블·기준선 그대로. 외부 피처(extB·ext_ksic)는 트랙 공식 파일.
출력: C18-T2/cache/t3res/res_<V>_<mode>[_tag].json, preds_<V>_<mode>[_tag].parquet
"""
import os, sys, json, argparse, time, copy
ap = argparse.ArgumentParser()
ap.add_argument('--var', required=True); ap.add_argument('--mode', required=True, choices=['dev', 'final'])
ap.add_argument('--origins', default=None)
args = ap.parse_args()
MY = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, MY)
W = os.environ['C18_W']
R3 = os.path.abspath(os.path.join(W, '..', '..', '..', '모델경쟁_20260927', 'T3_기술셀부상예측', 'R3'))
OUTD = os.path.join(W, 'cache', 't3res'); os.makedirs(OUTD, exist_ok=True)
SPEC = json.load(open(os.path.join(R3, 'final_spec_r3.json'), encoding='utf-8'))
env = dict(T3R3_VAR='pub', PYTHONIOENCODING='utf-8')
if args.mode == 'dev':
    env.update(T3R3_CACHE='cache_pub', T3R3_MAXYEAR='2021', T3R3_LABEL_MAX='2021', T3R3_EXT_SUFFIX='', T3R3_EXTB_SUFFIX='')
    os.environ.pop('T3R3_FINAL_UNLOCK', None); os.environ.pop('T3R3_TRAIN_MAX_T0', None)
else:
    env.update(T3R3_CACHE='cache_pub_final', T3R3_MAXYEAR='2024', T3R3_LABEL_MAX='2024', T3R3_FINAL_UNLOCK='1',
               T3R3_TRAIN_MAX_T0=str(SPEC['final']['train_max_t0']), T3R3_EXT_SUFFIX='_final', T3R3_EXTB_SUFFIX='_final')
if args.var != 'P0':
    env['C18_T3CACHE'] = os.path.join(W, 'cache', f"t3_{args.var}_{'dev' if args.mode == 'dev' else 'fin'}")
    assert os.path.exists(os.path.join(env['C18_T3CACHE'], 'panel_ccy.parquet')) and os.path.exists(os.path.join(env['C18_T3CACHE'], 'feat2.parquet'))
else:
    os.environ.pop('C18_T3CACHE', None)
os.environ.update(env)
import numpy as np, pandas as pd
import c18lib as L
import c18core as C
from scipy.stats import rankdata
if args.var == 'P0':
    assert L.CACHE.endswith(env['T3R3_CACHE'])
else:
    assert os.path.abspath(L.CACHE) == os.path.abspath(env['C18_T3CACHE'])
SUF = env['T3R3_EXT_SUFFIX']
CFG = copy.deepcopy(SPEC['cfg'])


def prep(D):   # final_run_r3.prep 그대로 (입력은 트랙 R3 의 extB 파일, 읽기 전용)
    cf = pd.read_parquet(os.path.join(R3, f'extB_cell_pub{SUF}.parquet')); kf = pd.read_parquet(os.path.join(R3, f'extB_ksic_pub{SUF}.parquet'))
    n0 = len(D); D = D.merge(cf, on=['cell', 't0'], how='left').merge(kf, on=['ksic', 't0'], how='left'); assert len(D) == n0
    D['oa_mg_rel'] = D.oa_mg_g - D.oa_all_g if 'oa_all_g' in D else np.nan
    D['ix_sci_fx'] = D.ros_sci_rel * D.fx_mom; D['ix_sci_kr'] = D.ros_sci_rel * D.kr_mom; D['ix_oa_gap'] = D.oa_mg_g - D.kr_mom
    return D


_gw = C.get_wide
def gw(Dx, mt):
    w = _gw(Dx, mt)
    if 'ros_sci' not in w: w = prep(w)
    C._WIDE[mt] = w
    return w
C.get_wide = gw

t = time.time()
if args.mode == 'dev':
    D = prep(C.get_D()); origins = SPEC['dev']['origins']
else:
    ft = SPEC['final']['test_t0']; tmax = SPEC['final']['train_max_t0']
    D = prep(C.get_D(t0s=list(range(C.FIRST_T0, tmax + 1)) + [ft])); origins = [ft]
    assert D[D.t0 < ft].t0.max() == tmax
if args.origins: origins = [int(x) for x in args.origins.split(',')]
need = sorted({f for c in CFG['components'] for f in c['features']}); miss = [f for f in need if f not in D]; assert not miss, miss
TAG = f'C18_{args.var}_{args.mode}'
res, preds = {}, []
for v in origins:
    te = D[D.t0 == v]; tr = C.train_rows(D, v, CFG)
    assert tr.t0.max() + 3 <= v and tr.t0.max() + 3 < L.F_YEAR
    comps = [np.asarray(C.comp_pred(c, D, v, CFG, TAG), dtype=float) for c in CFG['components']]
    R = [rankdata(p) / len(te) for p in comps]; p = np.average(R, axis=0, weights=CFG.get('weights', [1] * len(comps)))
    p0 = C.baseline_pred(D, v)
    m = C.metrics(te.y.values, te.yb.values, p); b = C.metrics(te.y.values, te.yb.values, p0)
    res[v] = dict(model=m, baseline=b, skill=(m['auc'] - b['auc']) / (1 - b['auc']), n_train=int(len(tr)), n_test=int(len(te)), n_train_elig=int(len(tr)))
    print(f"[{args.var}/{args.mode}] t0={v}: AUC {m['auc']:.5f} 기준선 {b['auc']:.5f} skill {res[v]['skill']:.5f} P@10 {m['p_at10']:.3f} n_te {len(te)} ({time.time()-t:.0f}s)", flush=True)
    preds.append(pd.DataFrame(dict(cell=te.cell.values, ksic=te.ksic.values, mg=te.mg.values, t0=v, kr_W=te.kr_W.values, tot_W=te.tot_W.values,
                                   p=p, p_base=p0, y=te.y.values, yb=te.yb.values, p_logit=comps[0], p_xgbc=comps[1], p_nb=comps[2],
                                   US_lvl=te.US_lvl.values, EP_lvl=te.EP_lvl.values)))
sk = [r['skill'] for r in res.values()]
outj = dict(var=args.var, mode=args.mode, origins=origins, mean_skill=float(np.mean(sk)), mean_auc=float(np.mean([r['model']['auc'] for r in res.values()])),
            by_origin={str(k): v for k, v in res.items()}, n_rows_D=int(len(D)), created=time.strftime('%Y-%m-%d %H:%M:%S'), secs=round(time.time() - t))
sfx = '' if not args.origins else '_' + args.origins.replace(',', '-')
json.dump(outj, open(os.path.join(OUTD, f'res_{args.var}_{args.mode}{sfx}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
pd.concat(preds, ignore_index=True).to_parquet(os.path.join(OUTD, f'preds_{args.var}_{args.mode}{sfx}.parquet'))
print(f"[{args.var}/{args.mode}] 평균 skill {outj['mean_skill']:.5f} AUC {outj['mean_auc']:.5f} 총 {time.time()-t:.0f}s")
