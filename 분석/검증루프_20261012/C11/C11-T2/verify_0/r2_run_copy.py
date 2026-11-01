# -*- coding: utf-8 -*-
"""R9 T3 TXT 팔 실행기(R8 NONCIRC_R8/nc_run.py 사본; 경로와 txt 과제만 바꿈)
R8 원문: T3 순환 없는 필수성 검증 실행기 — 동결 E40-auditD 설정(성분·하이퍼파라미터·가중치·학습 규칙·기준선) 그대로, 개발 원점 2010–2018.
  --task mg    --ctx {none,prov,conc,concm} [--lite]
        셀 = IPC 메인그룹(라벨 무관, cache_mg_pub). 결과 = mg KR 공개 t0+1..t0+3 성장 상위 10%.
        피처 = E40 피처 중 KSIC 연결 6개(ksic_kr_mom, ksic_tot_mom, kr_share_rel_ksic, pi_*3)를 뺀 것 + 산업 맥락 CTX(체계별, s03_ctx.py)
  --task pcode --feat {pcode,conc}   셀 = 제공 KSIC→연계표 코드 × mg (P27). feat=conc 이면 피처를 연계표 셀(같은 키)에서 계산
  --task conc  --feat {conc,pcode}   셀 = 연계표 argmax 코드 × mg (C27, R7 cache_conc_pub). feat=pcode 이면 피처를 P27 셀(같은 키)에서 계산
  [--seeds k]  XGB 성분 시드 k개 평균(강건성; 동결 설정은 1)
동결 모듈(R3/r3lib.py, r3core.py, build_ext_r3b.py)은 import 만 한다. 결과는 TXT_R9/results 에만 쓴다. R9: --task txt = TXT 셀(TXT KSIC × mg) 과제, --ctx txt = TXT 산업 맥락(그 외 ctx 파일은 NONCIRC_R8 에서 읽기만)."""
import os, sys, json, time, argparse, copy
# C11-T2 사본: 모형·학습 코드는 R9 r9_run.py 그대로. 바꾼 것은 경로(캐시·ctx·RoS·결과·comp)와 --arm 인자뿐.
BASE = "C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"
T3 = BASE + "/분석/모델경쟁_20260927/T3_기술셀부상예측"; WD = BASE + "/분석/검증루프_20261012/C11/C11-T2"
R3 = os.path.join(T3, 'R3'); E7 = os.path.join(T3, 'ESSENTIAL_R7'); R8D = os.path.join(T3, 'NONCIRC_R8')
ap = argparse.ArgumentParser(); ap.add_argument('--task', required=True, choices=['mg', 'txt'])
ap.add_argument('--ctx', default='none'); ap.add_argument('--feat', default=None); ap.add_argument('--lite', action='store_true'); ap.add_argument('--seeds', type=int, default=1)
ap.add_argument('--arm', required=True); ap.add_argument('--outdir', default=WD + '/res/runs'); ap.add_argument('--rep', default='')
a = ap.parse_args()
HERE = WD + '/cache'
for k in ['T3R3_FINAL_UNLOCK', 'T3R3_TRAIN_MAX_T0', 'T3R3_HIER', 'T3R3_DATA_FIRST']: os.environ.pop(k, None)
os.environ.update(T3R3_VAR='pub', T3R3_CACHE='cache_pub', T3R3_MAXYEAR='2021', T3R3_LABEL_MAX='2021', T3R3_EXT_SUFFIX='', T3R3_EXTB_SUFFIX='')
sys.path.insert(0, R3); sys.dont_write_bytecode = True
import numpy as np, pandas as pd
import r3lib as L
import r3core as C
import build_ext_r3b as XB

SPEC = json.load(open(os.path.join(R3, 'final_spec_r3.json'), encoding='utf-8'))
CFG0 = SPEC['cfg']; ORIGINS = SPEC['dev']['origins']
ALLF = sorted({f for c in CFG0['components'] for f in c['features']})
PI = ['pi_prod_g2', 'pi_prod_g1', 'pi_ship_g2']; ROS = ['ros_sci', 'ros_ppp']
KSIC6 = ['ksic_kr_mom', 'ksic_tot_mom', 'kr_share_rel_ksic'] + PI
CTX = ['c_ind_kmom', 'c_ind_tmom', 'c_ind_fmom', 'c_ind_krsh', 'c_krsh_rel', 'c_pi_prod_g2', 'c_pi_prod_g1', 'c_pi_ship_g2', 'c_n_ind', 'c_entropy',
       'c_cell_kmax', 'c_cell_kwavg', 'c_cell_fmax', 'c_newind_sh', 'c_mix_shift', 'c_cov']
CTX_LITE = ['c_ind_kmom', 'c_ind_tmom', 'c_krsh_rel', 'c_pi_prod_g2', 'c_pi_prod_g1', 'c_pi_ship_g2']
T0S = list(range(C.FIRST_T0, max(ORIGINS) + 1))
CACHES = {'mg': os.path.join(HERE, 'mg_pub'), 'txt': os.path.join(HERE, f'txt_{a.arm}')}
ROSF = {'mg': os.path.join(R8D, 'ros_mg_issueyear_pubrule.parquet'), 'txt': os.path.join(HERE, f'ros_txt_{a.arm}.parquet')}
PIF = os.path.join(R3, 'ext_ksic_t0_pub.parquet')   # TXT 셀 = 5자리 KSIC → KSIC별 KOSIS 생산지수(동결, 제공 팔과 같은 파일)
CTXF = lambda k: os.path.join(HERE, f'ctx_txt_{a.arm}.parquet') if k == 'txt' else os.path.join(R8D, f'ctx_{k}.parquet')
if a.task == 'mg':
    tag = f"mg_ctx-{a.ctx}" + ('_lite' if a.lite else '')
else:
    tag = 'txt50_task'
if a.seeds > 1: tag += f'_s{a.seeds}'
tag += f'__{a.arm}' + (f'__rep{a.rep}' if a.rep else '')
OUTD = a.outdir; os.makedirs(OUTD, exist_ok=True)
C.COMP_DIR = os.path.join(WD, 'verify_0', 'r2_comp', tag); os.makedirs(C.COMP_DIR, exist_ok=True)


def ros_alt(cells, ksic_c, mg_c, path):
    XB.ROS_PUBRULE = path
    return XB.ros_feats(T0S, cells, ksic_c, mg_c)[['cell', 't0'] + ROS]


def load_aug(cache, extra_cells):
    """r3lib.load 와 같은 배열 + 패널에 없는 셀을 0건으로 추가 (ESSENTIAL_R7/t3_ess.py 와 같은 코드)."""
    df = pd.read_parquet(os.path.join(cache, 'panel_ccy.parquet')); assert df.year.max() <= L.Y1
    df['cell'] = df.ksic + '|' + df.mg
    cells = np.array(sorted(set(df.cell.unique()) | set(extra_cells))); cid = {c: i for i, c in enumerate(cells)}
    df['ci'] = df.cell.map(cid).values; T = L.Y1 - L.Y0 + 1
    cols = ['n', 'n_app', 'n_prio_for', 'n_newapp_apps', 'n_newapp', 'n_newk_apps', 'hhi', 'n_multi', 'n_newpair_apps', 'n_newpair', 'n_newword_apps', 'n_newword']
    A = {}
    for col in cols:
        arr = np.zeros((len(cells), len(L.CCS), T), dtype=np.float32); sub = df[['ci', 'cc', 'year', col]].dropna()
        arr[sub.ci.values, sub.cc.map({c: i for i, c in enumerate(L.CCS)}).values, sub.year.values - L.Y0] = sub[col].values; A[col] = arr
    return dict(cells=cells, ksic=np.array([c.split('|')[0] for c in cells]), mg=np.array([c.split('|')[1] for c in cells]), A=A)


def alt_frame(cache, need_D, need_W, pi_file, ros_path):
    sv = (L.CACHE, L.LOCAL); L.CACHE = L.LOCAL = cache; L._cache.clear()
    P = load_aug(cache, set(need_D.alt_cell) | set(need_W.alt_cell)); L._cache['P'] = P
    needset = pd.concat([need_D, need_W])[['alt_cell', 't0']].drop_duplicates()
    F = []
    for tt in T0S:
        f = L.feats(P, tt, 0); keep = set(needset.loc[needset.t0 == tt, 'alt_cell']); F.append(f[f.cell.isin(keep)])
    F = pd.concat(F, ignore_index=True)
    outs = []
    for need in (need_D, need_W):
        k = need[['alt_cell', 't0']].drop_duplicates().rename(columns={'alt_cell': 'cell'})
        X = F.merge(k, on=['cell', 't0']).reset_index(drop=True)
        X = C.enrich(X); X = C.extras(X)
        E = pd.read_parquet(pi_file)[['ksic', 't0'] + PI]; X = X.drop(columns=[c for c in PI if c in X]).merge(E, on=['ksic', 't0'], how='left')
        X = X.merge(ros_alt(P['cells'], P['ksic'], P['mg'], ros_path), on=['cell', 't0'], how='left')
        outs.append(X)
    L.CACHE, L.LOCAL = sv; L._cache.clear()
    return outs


def swap(T, X, repl):
    X = X[['cell', 't0', 'tot_W'] + [c for c in repl if c != 'tot_W']].rename(columns={'cell': 'alt_cell', 'tot_W': 'alt_totW'})
    n0 = len(T); T2 = T.drop(columns=[c for c in repl + ['alt_totW'] if c in T]).merge(X, on=['alt_cell', 't0'], how='left'); assert len(T2) == n0
    T2.index = T.index
    return T2, dict(row_found=float(T2.alt_totW.notna().mean()), alt_totW_pos=float((T2.alt_totW > 0).mean()),
                    alt_vs_target_totW_median_ratio=float(np.nanmedian((T2.alt_totW + 1) / (T.tot_W.values + 1))))


t = time.time(); info = {}
cache = CACHES[a.task]; L.CACHE = L.LOCAL = cache; L._cache.clear()
C.EXT_FILES = [] if a.task == 'mg' else [PIF]; C.CELL_EXT_FILE = os.path.join(HERE, '_none_.parquet')
D = C.get_D(T0S)
P = L.load(); R = ros_alt(P['cells'], P['ksic'], P['mg'], ROSF[a.task])
D = D.merge(R, on=['cell', 't0'], how='left'); W = C.get_wide(D, 10).merge(R, on=['cell', 't0'], how='left')
cfg = copy.deepcopy(CFG0)
if a.task == 'mg':
    assert (D.ksic == 'ALL').all()
    add = []
    if a.ctx != 'none':
        X = pd.read_parquet(CTXF(a.ctx)); add = CTX_LITE if a.lite else CTX
        X = X[['mg', 't0'] + add]
        n0 = len(D); D = D.merge(X, on=['mg', 't0'], how='left'); assert len(D) == n0
        n1 = len(W); W = W.merge(X, on=['mg', 't0'], how='left'); assert len(W) == n1
        info['ctx_nonmissing_eval'] = {c: float(D.loc[D.t0.isin(ORIGINS), c].notna().mean()) for c in add}
    for c in cfg['components']:
        c['features'] = [f for f in c['features'] if f not in KSIC6] + add
if a.seeds > 1:
    for c in cfg['components']:
        if c['type'] in ('xgbc', 'nb'): c['seeds'] = a.seeds
C._WIDE[10] = W
FEATS = sorted({f for c in cfg['components'] for f in c['features']})
miss = [f for f in FEATS if f not in D]; assert not miss, miss
# 기준선 = 그 과제의 kr_mom 단독 로지스틱(피처 교체 전 값, 교체 시험에서도 고정)
Dbase = D
P0 = {v: C.baseline_pred(Dbase, v) for v in ORIGINS}
res, preds = {}, []
for v in ORIGINS:
    te = D[D.t0 == v]; tr = C.train_rows(D, v, cfg); assert tr.t0.max() + 3 <= v and tr.t0.max() + 3 < L.F_YEAR
    p = C.score(cfg, D, v, 'r9'); p0 = P0[v]; teb = Dbase[Dbase.t0 == v]
    assert (te.cell.values == teb.cell.values).all()
    m = C.metrics(teb.y.values, teb.yb.values, p); b = C.metrics(teb.y.values, teb.yb.values, p0)
    res[str(v)] = dict(model=m, baseline=b, skill=(m['auc'] - b['auc']) / (1 - b['auc']))
    print(f"[{tag}] t0={v} AUC {m['auc']:.4f} (기준선 {b['auc']:.4f}) skill {res[str(v)]['skill']:.4f} P@10 {m['p_at10']:.3f} n={m['n']} evt={m['n_evt']} | {time.time()-t:.0f}s", flush=True)
    preds.append(pd.DataFrame(dict(cell=teb.cell.values, mg=teb.mg.values, ksic=teb.ksic.values, t0=v, y=teb.y.values, yb=teb.yb.values,
                                   kr_W=teb.kr_W.values, kr_fut=teb.kr_fut.values, tot_W=teb.tot_W.values, p=p, p_base=p0)))
sk = [r['skill'] for r in res.values()]
summ = dict(mean_skill=float(np.mean(sk)), mean_auc=float(np.mean([r['model']['auc'] for r in res.values()])),
            mean_base_auc=float(np.mean([r['baseline']['auc'] for r in res.values()])),
            mean_p10=float(np.mean([r['model']['p_at10'] for r in res.values()])), mean_prauc=float(np.mean([r['model']['prauc'] for r in res.values()])),
            mean_n=float(np.mean([r['model']['n'] for r in res.values()])), mean_evt=float(np.mean([r['model']['n_evt'] for r in res.values()])))
out = dict(task=a.task, ctx=a.ctx, feat=a.feat, lite=a.lite, seeds=a.seeds, tag=tag, model='E40-auditD 동결 설정', origins=ORIGINS, summary=summ,
           by_origin=res, info={k: v for k, v in info.items() if k != 'replaced'}, n_features=len(FEATS), features=FEATS,
           created=time.strftime('%Y-%m-%d %H:%M:%S'), secs=round(time.time() - t))
json.dump(out, open(os.path.join(OUTD, f'{tag}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
pd.concat(preds, ignore_index=True).to_parquet(os.path.join(OUTD, f'{tag}_preds.parquet'))
print(json.dumps(dict(summary=summ, info=out['info']), ensure_ascii=False))
