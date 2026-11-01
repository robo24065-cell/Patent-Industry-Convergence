"""C16-T3 시나리오 재학습·채점 (C09-T3 run_scen.py 사본: 시나리오 정의만 교체).
사용: python -I run_scen16.py --scen S0|S1|S2a|S2b --mode dev|final [--origins 2010,2011]
동결 spec(final_spec_r3.json)의 성분·하이퍼파라미터·가중치·학습 규칙 그대로. 시나리오는 all_newapp_sh·all_newapp_n 값만 바꾼다(튜닝·선택 없음).
출력: cache/res_{scen}_{mode}[_tag].json, cache/preds_{scen}_{mode}[_tag].parquet (트랙 폴더에는 쓰지 않음)
"""
import os, sys, json, argparse, time, copy
ap = argparse.ArgumentParser()
ap.add_argument('--scen', required=True); ap.add_argument('--mode', required=True, choices=['dev', 'final'])
ap.add_argument('--origins', default=None)
args = ap.parse_args()
MY = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, MY)
from common import R3, CACHE
SPEC = json.load(open(os.path.join(R3, 'final_spec_r3.json'), encoding='utf-8'))
env = dict(T3R3_VAR='pub', PYTHONIOENCODING='utf-8')
if args.mode == 'dev':
    env.update(T3R3_CACHE='cache_pub', T3R3_MAXYEAR='2021', T3R3_LABEL_MAX='2021', T3R3_EXT_SUFFIX='', T3R3_EXTB_SUFFIX='')
    os.environ.pop('T3R3_FINAL_UNLOCK', None); os.environ.pop('T3R3_TRAIN_MAX_T0', None)
else:   # 이미 채점된 공식 최종 원점의 진단 재계산(공식 성적 불변, C09-T3와 같은 경로)
    env.update(T3R3_CACHE='cache_pub_final', T3R3_MAXYEAR='2024', T3R3_LABEL_MAX='2024', T3R3_FINAL_UNLOCK='1',
               T3R3_TRAIN_MAX_T0=str(SPEC['final']['train_max_t0']), T3R3_EXT_SUFFIX='_final', T3R3_EXTB_SUFFIX='_final')
os.environ.update(env)
import numpy as np, pandas as pd
import c09lib as L
import c09core as C
from scipy.stats import rankdata
assert L.CACHE.endswith(env['T3R3_CACHE'])
SUF = env['T3R3_EXT_SUFFIX']
CFG = copy.deepcopy(SPEC['cfg'])
scen = args.scen
SF = pd.read_parquet(os.path.join(CACHE, f"t3feats_{'dev' if args.mode == 'dev' else 'fin'}.parquet"))
COLMAP = {s: {'all_newapp_sh': f'all_newapp_sh_{s}', 'all_newapp_n': f'all_newapp_n_{s}'} for s in ['S1', 'S2a', 'S2b']}
assert scen in ('S0',) + tuple(COLMAP)


def override(D):
    if scen in COLMAP:
        cm = COLMAP[scen]; s = SF[['cell', 't0'] + list(cm.values())]
        n0 = len(D); D = D.merge(s, on=['cell', 't0'], how='left'); assert len(D) == n0
        assert D[list(cm.values())].notna().all().all()
        for a, b in cm.items():
            D[a] = D[b].astype(D[a].dtype); D = D.drop(columns=[b])
    D['_scen'] = scen
    return D


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
    if '_scen' not in w: w = override(w)
    C._WIDE[mt] = w
    return w
C.get_wide = gw

t = time.time()
if args.mode == 'dev':
    D = override(prep(C.get_D())); origins = SPEC['dev']['origins']
else:
    ft = SPEC['final']['test_t0']; tmax = SPEC['final']['train_max_t0']
    D = override(prep(C.get_D(t0s=list(range(C.FIRST_T0, tmax + 1)) + [ft]))); origins = [ft]
    assert D[D.t0 < ft].t0.max() == tmax
if args.origins: origins = [int(x) for x in args.origins.split(',')]
need = sorted({f for c in CFG['components'] for f in c['features']}); miss = [f for f in need if f not in D]; assert not miss, miss
TAG = f'C16_{scen}_{args.mode}'
res, preds = {}, []
for v in origins:
    te = D[D.t0 == v]; tr = C.train_rows(D, v, CFG)
    assert tr.t0.max() + 3 <= v and tr.t0.max() + 3 < L.F_YEAR
    comps = [np.asarray(C.comp_pred(c, D, v, CFG, TAG), dtype=float) for c in CFG['components']]
    R = [rankdata(p) / len(te) for p in comps]; p = np.average(R, axis=0, weights=CFG.get('weights', [1] * len(comps)))
    p0 = C.baseline_pred(D, v)
    m = C.metrics(te.y.values, te.yb.values, p); b = C.metrics(te.y.values, te.yb.values, p0)
    res[v] = dict(model=m, baseline=b, skill=(m['auc'] - b['auc']) / (1 - b['auc']), n_train=int(len(tr)))
    print(f"[{scen}/{args.mode}] t0={v}: AUC {m['auc']:.4f} 기준선 {b['auc']:.4f} skill {res[v]['skill']:.4f} ({time.time()-t:.0f}s)", flush=True)
    preds.append(pd.DataFrame(dict(cell=te.cell.values, ksic=te.ksic.values, mg=te.mg.values, t0=v, p=p, p_base=p0, y=te.y.values, yb=te.yb.values,
                                   all_newapp_sh=te.all_newapp_sh.values, all_newapp_n=te.all_newapp_n.values)))
sk = [r['skill'] for r in res.values()]
out = dict(scen=scen, mode=args.mode, origins=origins, mean_skill=float(np.mean(sk)), by_origin={str(k): v for k, v in res.items()},
           created=time.strftime('%Y-%m-%d %H:%M:%S'), secs=round(time.time() - t))
sfx = '' if not args.origins else '_' + args.origins.replace(',', '-')
json.dump(out, open(os.path.join(CACHE, f'res_{scen}_{args.mode}{sfx}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
pd.concat(preds, ignore_index=True).to_parquet(os.path.join(CACHE, f'preds_{scen}_{args.mode}{sfx}.parquet'))
print(f"[{scen}/{args.mode}] 평균 skill {out['mean_skill']:.4f} 총 {time.time()-t:.0f}s")
