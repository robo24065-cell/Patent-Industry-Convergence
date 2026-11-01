# -*- coding: utf-8 -*-
r"""C18-T2 T3 동결 스펙(E40-auditD, final_spec_r3.json) 재학습·재채점 — R3/final_run_r3.py 의 evaluate·prep 경로 사본.
바꾼 것: 셀 캐시 경로(T3R3_CACHE=이 폴더 cache/t3_<VAR>_<mode>), 성분 예측 캐시(C.COMP_DIR=이 폴더), 결과 쓰기(이 폴더 cache/t3res).
외부 피처(KOSIS ext_ksic_t0_pub[_final], extB_cell/ksic_pub[_final])는 공식 파일을 그대로 읽는다(재빌드 없음).
사용: python t3_run.py <VAR> <final|dev> [origins 쉼표] — final 은 t0=2021(학습 t0 ≤2018), dev 는 2010–2018 중 지정."""
import os, sys, json, time
VAR, MODE = sys.argv[1], sys.argv[2]
ORIG = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else None
sys.dont_write_bytecode = True
W = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C18\C18-T2'
R3 = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\T3_기술셀부상예측\R3'
mm = 'fin' if MODE == 'final' else 'dev'
CACHE = os.path.join(W, 'cache', f't3_{"P0" if VAR == "V2r" else VAR}_{mm}')  # V2r: P0 셀 캐시 + RoS own-only 연결 제거판 extB_cell
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


def prep(D):
    cf = pd.read_parquet(os.path.join(W, 'cache', 'extB_V2r', f'extB_cell_pub{SUF}.parquet') if VAR == 'V2r' else os.path.join(R3, f'extB_cell_pub{SUF}.parquet')); kf = pd.read_parquet(os.path.join(R3, f'extB_ksic_pub{SUF}.parquet'))
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
need = sorted({f for c in CFG['components'] for f in c['features']})
t = time.time()
if MODE == 'final':
    ft, tmax = SPEC['final']['test_t0'], SPEC['final']['train_max_t0']
    D = prep(C.get_D(t0s=list(range(C.FIRST_T0, tmax + 1)) + [ft])); origins = [ft]
    assert D[D.t0 < ft].t0.max() == tmax
else:
    D = prep(C.get_D()); origins = ORIG or SPEC['dev']['origins']
    assert set(origins) <= set(range(2010, 2019))
miss = [f for f in need if f not in D]; assert not miss, miss
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
           cache=CACHE, created=time.strftime('%Y-%m-%d %H:%M:%S'), secs=round(time.time() - t))
json.dump(out, open(os.path.join(OUTD, f'res_{tag}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
pd.concat(preds, ignore_index=True).to_parquet(os.path.join(OUTD, f'preds_{tag}.parquet'))
print(f"[{VAR}/{MODE}] 평균 skill {out['mean_skill']:.5f} AUC {out['mean_auc']:.5f} 총 {out['secs']}s", flush=True)
