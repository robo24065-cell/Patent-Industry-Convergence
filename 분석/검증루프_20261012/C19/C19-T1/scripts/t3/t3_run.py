# -*- coding: utf-8 -*-
r"""C19-T1 T3 동결 스펙(E40-auditD, final_spec_r3.json) 재학습·재채점 — C18-T2 scripts\t3\t3_run.py 사본.
바꾼 것: 쓰기 폴더(C19-T1 cache), VJ4a/VJ4b(P0 셀 캐시 + 스펙 안 JP_* 피처를 0 / 학습 행 평균으로 고정; 확대 학습 표본에도 같은 값).
외부 피처(KOSIS ext_ksic_t0_pub[_final], extB_cell/ksic_pub[_final])는 공식 파일 그대로.
사용: python t3_run.py <VAR> <final|dev> [origins 쉼표]"""
import os, sys, json, time
VAR, MODE = sys.argv[1], sys.argv[2]
ORIG = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else None
sys.dont_write_bytecode = True
W = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C19\C19-T1'
R3 = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\T3_기술셀부상예측\R3'
mm = 'fin' if MODE == 'final' else 'dev'
CVAR = 'P0' if VAR in ('VJ4a', 'VJ4b') else VAR
CACHE = os.path.join(W, 'cache', f't3_{CVAR}_{mm}')
assert os.path.exists(os.path.join(CACHE, 'panel_ccy.parquet')) and os.path.exists(os.path.join(CACHE, 'feat2.parquet')), CACHE
for k in ['T3R3_FINAL_UNLOCK', 'T3R3_TRAIN_MAX_T0', 'T3R3_HIER', 'T3R3_DATA_FIRST']: os.environ.pop(k, None)
if MODE == 'final':
    os.environ.update(T3R3_VAR='pub', T3R3_CACHE=CACHE, T3R3_MAXYEAR='2024', T3R3_LABEL_MAX='2024', T3R3_FINAL_UNLOCK='1',
                      T3R3_TRAIN_MAX_T0='2018', T3R3_EXT_SUFFIX='_final', T3R3_EXTB_SUFFIX='_final')
else:
    os.environ.update(T3R3_VAR='pub', T3R3_CACHE=CACHE, T3R3_MAXYEAR='2021', T3R3_LABEL_MAX='2021', T3R3_EXT_SUFFIX='', T3R3_EXTB_SUFFIX='')
sys.path.insert(0, R3)
import numpy as np, pandas as pd
import r3lib as L
import r3core as C
assert os.path.abspath(L.CACHE) == os.path.abspath(CACHE), (L.CACHE, CACHE)
SPEC = json.load(open(os.path.join(R3, 'final_spec_r3.json'), encoding='utf-8')); CFG = SPEC['cfg']
TAG = 'final' if MODE == 'final' else 'dev'; SUF = os.environ['T3R3_EXT_SUFFIX']
C.COMP_DIR = os.path.join(W, 'cache', 'comp_t3', f'{VAR}_{mm}'); os.makedirs(C.COMP_DIR, exist_ok=True)
OUTD = os.path.join(W, 'cache', 't3res'); os.makedirs(OUTD, exist_ok=True)
need = sorted({f for c in CFG['components'] for f in c['features']})
JPF = [f for f in need if f.startswith('JP_')]
FIX = {}


def prep(D):
    cf = pd.read_parquet(os.path.join(R3, f'extB_cell_pub{SUF}.parquet')); kf = pd.read_parquet(os.path.join(R3, f'extB_ksic_pub{SUF}.parquet'))
    n0 = len(D); D = D.merge(cf, on=['cell', 't0'], how='left').merge(kf, on=['ksic', 't0'], how='left'); assert len(D) == n0
    D['oa_mg_rel'] = D.oa_mg_g - D.oa_all_g if 'oa_all_g' in D else np.nan
    D['ix_sci_fx'] = D.ros_sci_rel * D.fx_mom; D['ix_sci_kr'] = D.ros_sci_rel * D.kr_mom; D['ix_oa_gap'] = D.oa_mg_g - D.kr_mom
    return D


def fixjp(D):
    if VAR not in ('VJ4a', 'VJ4b'): return D
    D = D.copy()
    for f in JPF: D[f] = FIX[f]
    return D


_gw = C.get_wide


def gw(Dx, mt):
    w = _gw(Dx, mt)
    if 'ros_sci' not in w: w = fixjp(prep(w)); C._WIDE[mt] = w
    return w


C.get_wide = gw
t = time.time()
if MODE == 'final':
    ft, tmax = SPEC['final']['test_t0'], SPEC['final']['train_max_t0']
    D = prep(C.get_D(t0s=list(range(C.FIRST_T0, tmax + 1)) + [ft])); origins = [ft]
    assert D[D.t0 < ft].t0.max() == tmax
else:
    D = prep(C.get_D()); origins = ORIG or SPEC['dev']['origins']
    assert set(origins) <= set(range(2010, 2019))
miss = [f for f in need if f not in D]; assert not miss, miss
if VAR in ('VJ4a', 'VJ4b'):
    # 학습 평균: 최종은 학습 t0 ≤2018 행 평균(원점 공통), 개발은 원점마다 다르지만 상수 열은 트리 분할에 쓰이지 않으므로 전체 학습 구간(t0≤2015) 평균 하나로 둔다
    trm = D[D.t0 <= (2018 if MODE == 'final' else 2015)]
    for f in JPF: FIX[f] = 0.0 if VAR == 'VJ4a' else float(np.nanmean(trm[f].values))
    D = fixjp(D)
    print('JP 고정', FIX, flush=True)
res, preds = {}, []
for v in origins:
    te = D[D.t0 == v]; tr = C.train_rows(D, v, CFG)
    assert tr.t0.max() + 3 <= v and tr.t0.max() + 3 < L.F_YEAR
    p = C.score(CFG, D, v, TAG); p0 = C.baseline_pred(D, v)
    m = C.metrics(te.y.values, te.yb.values, p); b = C.metrics(te.y.values, te.yb.values, p0)
    res[str(v)] = dict(model=m, baseline=b, skill=(m['auc'] - b['auc']) / (1 - b['auc']), n_train=int(len(tr)))
    print(f"[{VAR}/{MODE}] t0={v}: AUC {m['auc']:.5f} 기준선 {b['auc']:.5f} skill {res[str(v)]['skill']:.5f} P@10 {m['p_at10']:.3f} n_te {len(te)} evt {m['n_evt']} ({time.time()-t:.0f}s)", flush=True)
    preds.append(pd.DataFrame(dict(cell=te.cell.values, ksic=te.ksic.values, mg=te.mg.values, t0=v, kr_W=te.kr_W.values, tot_W=te.tot_W.values,
                                   p=p, p_base=p0, y=te.y.values, yb=te.yb.values)))
tag = f'{VAR}_{MODE}' + ('' if ORIG is None else '_' + '-'.join(map(str, origins)))
out = dict(var=VAR, mode=MODE, origins=origins, mean_skill=float(np.mean([r['skill'] for r in res.values()])),
           mean_auc=float(np.mean([r['model']['auc'] for r in res.values()])), by_origin=res, n_rows_D=int(len(D)),
           cache=CACHE, jp_fix=FIX, created=time.strftime('%Y-%m-%d %H:%M:%S'), secs=round(time.time() - t))
json.dump(out, open(os.path.join(OUTD, f'res_{tag}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
pd.concat(preds, ignore_index=True).to_parquet(os.path.join(OUTD, f'preds_{tag}.parquet'))
print(f"[{VAR}/{MODE}] 평균 skill {out['mean_skill']:.5f} AUC {out['mean_auc']:.5f} 총 {out['secs']}s", flush=True)
