# -*- coding: utf-8 -*-
"""R7 T3 필수성 검증 실행기 — 동결 챔피언 설정(성분·하이퍼파라미터·가중치·학습 규칙) 그대로, 피처(또는 셀 정의)만 바꿔 개발 원점에서 재학습.
  --var pub : R3 공개연도 공식 챔피언 E40-auditD (R3/final_spec_r3.json), 개발 원점 2010–2018
  --var app : R2 출원연도 챔피언 E57-audit (R2/final_spec_r2.json), 개발 원점 2010–2016
  --test    :
    full          동결 설정 재현(기준점)
    drop_<묶음>   제공 데이터 유래 피처 묶음 제거(아래 BUNDLES)
    kr_rebuild    5개국 제거: KR 레코드만으로 셀 피처 전부 재계산(해외 유래 값은 상수가 됨, RoS 제외). 평가 셀·레이블·기준선은 그대로
    conc_feat     대체(결과 고정): 평가 셀·레이블·기준선은 제공 셀 그대로, 피처는 전부 '연계표 셀'(연계표 코드×메인그룹)에서 계산해
                  제공 셀 (k, mg) → 연계표 셀 (code(k), mg) 로 붙임. 음이항 성분의 과거 KR 건수(kr_W)도 연계표 셀 값
    conc_task     대체(결과 재정의): 셀·적격·레이블·기준선까지 전부 연계표 셀 기준으로 다시 만든 과제
    concm_feat / concm_task  위와 같되 연계표 '관대한 다중 배정'(s01b, 점유율 ≥0.10 코드 전부) 캐시 사용
동결 모듈(R3/r3lib.py, r3core.py, build_ext_r3b.py)은 import 만 하고, 성분 캐시·결과는 ESSENTIAL_R7 에만 쓴다.
"""
import os, sys, json, time, argparse, copy
HERE = os.path.dirname(os.path.abspath(__file__)); T3 = os.path.dirname(HERE); R3 = os.path.join(T3, 'R3')
ap = argparse.ArgumentParser(); ap.add_argument('--var', required=True, choices=['pub', 'app', 'pubF']); ap.add_argument('--test', required=True)
a = ap.parse_args()
V = 'app' if a.var == 'app' else 'pub'; FIN = a.var == 'pubF'   # pubF = 사후 최종 분석(공개연도, t0=2021, 학습 t0 ≤2018) — 공식 성적 아님
for k in ['T3R3_FINAL_UNLOCK', 'T3R3_TRAIN_MAX_T0', 'T3R3_HIER', 'T3R3_DATA_FIRST']: os.environ.pop(k, None)
if FIN:
    os.environ.update(T3R3_VAR='pub', T3R3_CACHE='cache_pub_final', T3R3_MAXYEAR='2024', T3R3_LABEL_MAX='2024', T3R3_FINAL_UNLOCK='1', T3R3_TRAIN_MAX_T0='2018',
                      T3R3_EXT_SUFFIX='_final', T3R3_EXTB_SUFFIX='_final')
else:
    os.environ.update(T3R3_VAR=V, T3R3_CACHE='cache_pub', T3R3_MAXYEAR='2021', T3R3_LABEL_MAX='2021', T3R3_EXT_SUFFIX='', T3R3_EXTB_SUFFIX='')
sys.path.insert(0, R3); sys.dont_write_bytecode = True
import numpy as np, pandas as pd
import r3lib as L
import r3core as C
import build_ext_r3b as XB

SPEC = json.load(open(os.path.join(R3, 'final_spec_r3.json') if V == 'pub' else os.path.join(T3, 'R2', 'final_spec_r2.json'), encoding='utf-8'))
CFG0 = SPEC['cfg']; ORIGINS = [SPEC['final']['test_t0']] if FIN else SPEC['dev']['origins']
REF_SKILL = json.load(open(os.path.join(R3, 'final_result_r3.json'), encoding='utf-8'))['summary']['mean_skill'] if FIN else SPEC['dev']['mean_skill']
ALLF = sorted({f for c in CFG0['components'] for f in c['features']})
PI = ['pi_prod_g2', 'pi_prod_g1', 'pi_ship_g2']; ROS = ['ros_sci', 'ros_ppp']
CTRY = [f'{c}_{s}' for c in ['US', 'CN', 'JP', 'EP'] for s in ['mom', 'mom_l1', 'mom_l2', 'lvl', 'g1', 'slope5']]
BUNDLES = {
    'foreign_mom': CTRY + ['fx_mom', 'fx_mom_l1', 'gap_fx_kr', 'mg_fx_mom', 'sub_fmom'],
    'foreign_all': CTRY + ['fx_mom', 'fx_mom_l1', 'gap_fx_kr', 'mg_fx_mom', 'sub_fmom', 'tot_mom', 'tot_lvl', 'kr_share', 'kr_share_chg',
                           'all_newapp_sh', 'all_newpair_sh', 'all_newapp_n', 'mg_tot_mom', 'ksic_tot_mom', 'nb_tmom', 'sub_krsh', 'tot_vs_long',
                           'kr_share_rel_ksic', 'kr_share_rel_sub', 'kr_share_rel_mg'] + ROS,
    'ksic_group': ['ksic_kr_mom', 'ksic_tot_mom', 'kr_share_rel_ksic', 'kr_newk_sh'],
    'applicant': ['app_mom', 'app_size', 'big_sh', 'top1_sh', 'kr_newapp_sh', 'all_newapp_sh', 'kr_napp', 'kr_newapp_n', 'kr_hhi', 'all_newapp_n', 'kr_newk_sh'],
    'title': ['kr_newword_sh'],
    'ipc_cooc': ['kr_multi_sh', 'kr_newpair_sh', 'all_newpair_sh', 'nb_kmom', 'nb_tmom', 'nb_w', 'nb_deg'],
    'prio': ['kr_prio_for_sh', 'kr_forin_mom', 'kr_dom_mom'],
    'provided_all': [f for f in ALLF if f not in PI],       # 외부 KOSIS 3피처만 남김(셀·레이블은 여전히 제공 데이터)
    'ros': ROS,
}
OUTD = os.path.join(HERE, 'results'); os.makedirs(OUTD, exist_ok=True)
C.COMP_DIR = os.path.join(HERE, f'comp_{a.var}'); os.makedirs(C.COMP_DIR, exist_ok=True)
KM = pd.read_csv(os.path.join(HERE, 'conc_ksic_map.csv'), dtype=str).set_index('ksic').code.to_dict()
ORIG_CACHE = L.CACHE
SUFX = '_final' if FIN else ''
if V == 'app':
    C.EXT_FILES = [os.path.join(T3, 'R2', 'ext_ksic_t0.parquet'), os.path.join(R3, 'ext_ksic_t0_app.parquet')]
    C.CELL_EXT_FILE = os.path.join(R3, 'ext_cell_t0_app.parquet')
else:
    C.EXT_FILES = [os.path.join(R3, f'ext_ksic_t0_pub{SUFX}.parquet')]; C.CELL_EXT_FILE = os.path.join(R3, f'ext_cell_t0_pub{SUFX}.parquet')
T0S = (list(range(C.FIRST_T0, 2019)) + [2021]) if FIN else list(range(C.FIRST_T0, max(ORIGINS) + 1))


def cfg_drop(cfg, drop):
    c2 = copy.deepcopy(cfg)
    for c in c2['components']:
        c['features'] = [f for f in c['features'] if f not in set(drop)]
        assert c['features'], '성분 피처가 비면 안 됨'
    return c2


def prep_orig(D):
    """final_run_r3.prep 과 같은 병합(공개연도 변형만 RoS 사용)."""
    if V != 'pub': return D
    cf = pd.read_parquet(os.path.join(R3, f'extB_cell_pub{SUFX}.parquet'))[['cell', 't0'] + ROS]
    n0 = len(D); D = D.merge(cf, on=['cell', 't0'], how='left'); assert len(D) == n0
    return D


def ros_alt(cells, ksic_c, mg_c, path):
    XB.ROS_PUBRULE = path
    return XB.ros_feats(T0S, cells, ksic_c, mg_c)[['cell', 't0'] + ROS]


def load_aug(cache, extra_cells):
    """r3lib.load 와 같은 배열 구성 + 패널에 없는 셀(extra_cells)을 0건으로 추가."""
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
    """대체 캐시에서 필요한 셀×원점 피처를 계산. need_*: DataFrame(alt_cell, t0). enrich 의 원점 단면 중앙값은 원본처럼 D·W 따로."""
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
        if pi_file is not None:
            E = pd.read_parquet(pi_file)[['ksic', 't0'] + PI]; X = X.drop(columns=[c for c in PI if c in X]).merge(E, on=['ksic', 't0'], how='left')
        if ros_path is not None:
            X = X.merge(ros_alt(P['cells'], P['ksic'], P['mg'], ros_path), on=['cell', 't0'], how='left')
        outs.append(X)
    L.CACHE, L.LOCAL = sv; L._cache.clear()
    return outs


def swap(T, X, repl):
    """T(목표 행)의 repl 피처를 X(대체 프레임, key=alt_cell,t0)의 값으로 교체."""
    X = X[['cell', 't0', 'tot_W'] + [c for c in repl if c != 'tot_W']].rename(columns={'cell': 'alt_cell', 'tot_W': 'alt_totW'})
    n0 = len(T); T2 = T.drop(columns=[c for c in repl + ['alt_totW'] if c in T]).merge(X, on=['alt_cell', 't0'], how='left'); assert len(T2) == n0
    T2.index = T.index
    return T2, dict(row_found=float(T2.alt_totW.notna().mean()), alt_totW_pos=float((T2.alt_totW > 0).mean()),
                    alt_vs_target_totW_median_ratio=float(np.nanmedian((T2.alt_totW + 1) / (T.tot_W.values + 1))))


t = time.time(); info = {}
cfg = CFG0; test = a.test
CK = 'concm' if test.startswith('concm') else 'conc'
if test in ('conc_task', 'concm_task'):
    cache = os.path.join(HERE, f'cache_{CK}_{a.var}'); L.CACHE = L.LOCAL = cache; L._cache.clear()
    C.EXT_FILES = [os.path.join(HERE, f'ext_pi_conc_{a.var}.parquet')]; C.CELL_EXT_FILE = os.path.join(HERE, '_none_.parquet')
    D = C.get_D(T0S)
    if V == 'pub':
        P = L.load(); R = ros_alt(P['cells'], P['ksic'], P['mg'], os.path.join(HERE, f'ros_{CK}_issueyear_pubrule.parquet'))
        D = D.merge(R, on=['cell', 't0'], how='left'); W = C.get_wide(D, 10).merge(R, on=['cell', 't0'], how='left'); C._WIDE[10] = W
else:
    D = prep_orig(C.get_D(T0S))
    if V == 'pub': C._WIDE[10] = prep_orig(C.get_wide(D, 10))
    else: C.get_wide(D, 10)
miss = [f for f in ALLF if f not in D]; assert not miss, miss
# 기준선은 언제나 '그 과제'의 kr_mom 단독 로지스틱 — 피처 교체 전에 계산(대체·제거 시험에서 기준선 고정)
P0 = {v: C.baseline_pred(D, v) for v in ORIGINS}
if test.startswith('drop_'):
    b = test[5:]; cfg = cfg_drop(CFG0, BUNDLES[b]); info['dropped'] = [f for f in BUNDLES[b] if f in ALLF]
elif test in ('kr_rebuild', 'conc_feat', 'concm_feat'):
    W = C._WIDE[10]
    if test == 'kr_rebuild':
        mp = lambda df: df.cell.values; cache = os.path.join(HERE, f'cache_kr_{a.var}'); pi_file = None; ros_path = None
        cfg = cfg_drop(CFG0, ROS); repl = [f for f in ALLF if f not in PI + ROS]
    else:
        mp = lambda df: (df.ksic.map(KM) + '|' + df.mg).values; cache = os.path.join(HERE, f'cache_{CK}_{a.var}')
        pi_file = os.path.join(HERE, f'ext_pi_conc_{a.var}.parquet'); ros_path = os.path.join(HERE, f'ros_{CK}_issueyear_pubrule.parquet') if V == 'pub' else None
        repl = ALLF + ['kr_W']
    D['alt_cell'] = mp(D); W = W.copy(); W['alt_cell'] = mp(W)
    XD, XW = alt_frame(cache, D[['alt_cell', 't0']], W[['alt_cell', 't0']], pi_file, ros_path)
    repl = [f for f in repl if f in XD.columns]
    D, covD = swap(D, XD, repl); W, covW = swap(W, XW, repl); C._WIDE[10] = W
    info.update(replaced=repl, coverage_D=covD, coverage_W=covW, n_alt_cells_D=int(D.alt_cell.nunique()), n_target_cells_D=int(D.cell.nunique()))
elif test not in ('full', 'conc_task', 'concm_task'):
    raise SystemExit('알 수 없는 test')

res, preds = {}, []
for v in ORIGINS:
    te = D[D.t0 == v]; tr = C.train_rows(D, v, cfg); assert tr.t0.max() + 3 <= v and tr.t0.max() + 3 < L.F_YEAR
    p = C.score(cfg, D, v, 'r7'); p0 = P0[v]
    m = C.metrics(te.y.values, te.yb.values, p); b = C.metrics(te.y.values, te.yb.values, p0)
    res[str(v)] = dict(model=m, baseline=b, skill=(m['auc'] - b['auc']) / (1 - b['auc']))
    print(f"[{a.var}/{test}] t0={v} AUC {m['auc']:.4f} (기준선 {b['auc']:.4f}) skill {res[str(v)]['skill']:.4f} P@10 {m['p_at10']:.3f} n={m['n']} evt={m['n_evt']} | {time.time()-t:.0f}s", flush=True)
    preds.append(pd.DataFrame(dict(cell=te.cell.values, mg=te.mg.values, ksic=te.ksic.values, t0=v, y=te.y.values, yb=te.yb.values, p=p, p_base=p0)))
sk = [r['skill'] for r in res.values()]
summ = dict(mean_skill=float(np.mean(sk)), mean_auc=float(np.mean([r['model']['auc'] for r in res.values()])),
            mean_base_auc=float(np.mean([r['baseline']['auc'] for r in res.values()])),
            mean_p10=float(np.mean([r['model']['p_at10'] for r in res.values()])), mean_prauc=float(np.mean([r['model']['prauc'] for r in res.values()])),
            mean_n=float(np.mean([r['model']['n'] for r in res.values()])), mean_evt=float(np.mean([r['model']['n_evt'] for r in res.values()])))
out = dict(var=a.var, test=test, model=SPEC.get('model_name', SPEC.get('champion_exp')), origins=ORIGINS, summary=summ, by_origin=res, info=info,
           frozen_dev_mean_skill=REF_SKILL, created=time.strftime('%Y-%m-%d %H:%M:%S'), secs=round(time.time() - t))
if test == 'full': out['reproduced'] = bool(abs(summ['mean_skill'] - REF_SKILL) < 5e-4)
json.dump(out, open(os.path.join(OUTD, f'{a.var}_{test}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
pd.concat(preds, ignore_index=True).to_parquet(os.path.join(OUTD, f'{a.var}_{test}_preds.parquet'))
print(json.dumps(dict(summary=summ, info={k: v for k, v in info.items() if k != 'replaced'}, reproduced=out.get('reproduced')), ensure_ascii=False))
