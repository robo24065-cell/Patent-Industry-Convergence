"""C15-T2 T3 시드 묶음 (C11-T3 run_ros.py 사본 S0 경로 + 시드만 바꿈).
사용: python -B t3_bundle.py <b> <final|dev>
 b=0 공식(xgbc·nb seed 0, SB3 seed 0), b≥1 σ=100+b (PREREG §1). 성분 코드·스펙·데이터는 동결 그대로.
final: 모형·M0 예측, 공식 cluster_boot(IPC 메인그룹 B=1000 default_rng(0)), SB3(sens_t3 규칙, seed σ) + ci_sb_t3 규칙(default_rng(11)) → cache/t3/b{b}_final.json, preds_b{b}_final.parquet
dev: 개발 9원점(2010–18) skill → cache/t3/b{b}_dev.json
트랙 폴더에는 쓰지 않는다(guard15)."""
import os, sys, json, time, copy, hashlib
sys.dont_write_bytecode = True
MY = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, MY); sys.path.insert(0, os.path.dirname(MY))
import guard15 as G
b = int(sys.argv[1]); mode = sys.argv[2]; assert mode in ('final', 'dev')
SEED = 0 if b == 0 else 100 + b
WD = os.path.dirname(os.path.dirname(MY)); OUT = os.path.join(WD, 'cache', 't3'); os.makedirs(OUT, exist_ok=True)
R3 = os.path.abspath(os.path.join(WD, '..', '..', '..', '모델경쟁_20260927', 'T3_기술셀부상예측', 'R3')); assert os.path.isdir(R3), R3
VR5 = os.path.join(os.path.dirname(R3), 'VERIFY_R5')
SPEC = json.load(open(os.path.join(R3, 'final_spec_r3.json'), encoding='utf-8'))
env = dict(T3R3_VAR='pub', PYTHONIOENCODING='utf-8')
if mode == 'dev':
    env.update(T3R3_CACHE='cache_pub', T3R3_MAXYEAR='2021', T3R3_LABEL_MAX='2021', T3R3_EXT_SUFFIX='', T3R3_EXTB_SUFFIX='')
    os.environ.pop('T3R3_FINAL_UNLOCK', None); os.environ.pop('T3R3_TRAIN_MAX_T0', None)
else:
    env.update(T3R3_CACHE='cache_pub_final', T3R3_MAXYEAR='2024', T3R3_LABEL_MAX='2024', T3R3_FINAL_UNLOCK='1',
               T3R3_TRAIN_MAX_T0=str(SPEC['final']['train_max_t0']), T3R3_EXT_SUFFIX='_final', T3R3_EXTB_SUFFIX='_final')
os.environ.update(env)
import numpy as np, pandas as pd
import xgboost as xgb
import c09lib as L
import c09core as C
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
assert L.CACHE.endswith(env['T3R3_CACHE'])
SUF = env['T3R3_EXT_SUFFIX']
CFG = copy.deepcopy(SPEC['cfg'])
assert all('seeds' not in c for c in CFG['components'])   # 공식: 성분당 시드 1개(seed 0)


# ---- 성분 함수: c09core.comp_xgbc / comp_nb 와 같은 코드, seed 만 SEED + s ----
def comp_xgbc_s(c, tr, te):
    ps = []
    for s in range(c.get('seeds', 1)):
        p = dict(c['params']); p['seed'] = SEED + s
        d = xgb.DMatrix(tr[c['features']].values, label=tr.yb.values, weight=C._w(tr, c.get('half')))
        m = xgb.train(p, d, c['rounds']); ps.append(m.predict(xgb.DMatrix(te[c['features']].values)))
    return np.mean([rankdata(p) for p in ps], 0) if len(ps) > 1 else ps[0]


def comp_nb_s(c, tr, te):
    outs = []
    for s in range(c.get('seeds', 1)):
        p = dict(c['params']); p['seed'] = SEED + s
        d = xgb.DMatrix(tr[c['features']].values, label=tr.kr_fut.values, weight=C._w(tr, c.get('half')))
        m = xgb.train(p, d, c['rounds'])
        a = C.fit_alpha(tr.kr_fut.values, m.predict(d)); mu = m.predict(xgb.DMatrix(te[c['features']].values))
        trq = tr[tr['_elig'] == 1] if '_elig' in tr else tr
        tau = float(np.median([np.quantile(g.y.values, 1 - C.TOP) for _, g in trq.groupby('t0')]))
        need = np.ceil((te.kr_W.values + 1) * np.exp(tau) - 1 - 1e-9); r = 1 / a
        outs.append(C.nbinom.sf(need - 1, r, r / (r + mu)))
    return np.mean([rankdata(o) for o in outs], 0) if len(outs) > 1 else outs[0]


C.COMP['xgbc'] = comp_xgbc_s; C.COMP['nb'] = comp_nb_s


def prep(D):   # final_run_r3.prep 그대로 (트랙 R3 extB 파일, 읽기 전용)
    cf = pd.read_parquet(os.path.join(R3, f'extB_cell_pub{SUF}.parquet')); kf = pd.read_parquet(os.path.join(R3, f'extB_ksic_pub{SUF}.parquet'))
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


def cluster_boot(blocks, n_boot, seed=0):   # T3 R3 final_run_r3.cluster_boot 그대로
    rng = np.random.default_rng(seed); ums = np.unique(np.concatenate([bb[0] for bb in blocks]))
    gi = [pd.Series(np.arange(len(bb[0]))).groupby(bb[0]).apply(lambda s: s.values).to_dict() for bb in blocks]
    out = []
    for _ in range(n_boot):
        s = rng.choice(ums, len(ums)); am, ab = [], []
        for bb, g in zip(blocks, gi):
            ii = np.concatenate([g[u] for u in s if u in g]); y = bb[1][ii]
            if y.min() == y.max(): continue
            am.append(roc_auc_score(y, bb[2][ii])); ab.append(roc_auc_score(y, bb[3][ii]))
        a, a0 = np.mean(am), np.mean(ab)
        out.append((a, a0, (a - a0) / (1 - a0), a - a0, np.mean([(x - y0) / (1 - y0) for x, y0 in zip(am, ab)])))
    o = np.array(out); q = lambda j: [float(v) for v in np.percentile(o[:, j], [2.5, 97.5])]
    return dict(auc=q(0), auc_baseline=q(1), skill=q(2), auc_diff=q(3), skill_mean_of_origins=q(4), n_boot=n_boot, n_clusters=int(len(ums)))


t = time.time()
TAG = f'C15_b{b}_{mode}'
if mode == 'dev':
    D = prep(C.get_D()); origins = SPEC['dev']['origins']
else:
    ft = SPEC['final']['test_t0']; tmax = SPEC['final']['train_max_t0']
    D = prep(C.get_D(t0s=list(range(C.FIRST_T0, tmax + 1)) + [ft])); origins = [ft]
    assert D[D.t0 < ft].t0.max() == tmax
need = sorted({f for c in CFG['components'] for f in c['features']}); miss = [f for f in need if f not in D]; assert not miss, miss
res, preds, blocks = {}, [], []
for v in origins:
    te = D[D.t0 == v]; tr = C.train_rows(D, v, CFG)
    assert tr.t0.max() + 3 <= v and tr.t0.max() + 3 < L.F_YEAR
    comps = [np.asarray(C.comp_pred(c, D, v, CFG, TAG), dtype=float) for c in CFG['components']]
    R = [rankdata(p) / len(te) for p in comps]; p = np.average(R, axis=0, weights=CFG.get('weights', [1] * len(comps)))
    p0 = C.baseline_pred(D, v)
    m = C.metrics(te.y.values, te.yb.values, p); bm = C.metrics(te.y.values, te.yb.values, p0)
    od = np.argsort(-p)
    res[v] = dict(model=m, baseline=bm, skill=(m['auc'] - bm['auc']) / (1 - bm['auc']), n_train=int(len(tr)),
                  top100_hits=int(te.yb.values[od[:100]].sum()), top10pct_hits=int(te.yb.values[od[:max(1, int(round(0.10 * len(te))))]].sum()))
    print(f"[b{b}/{mode}] t0={v}: AUC {m['auc']:.6f} M0 {bm['auc']:.6f} skill {res[v]['skill']:.6f} ({time.time()-t:.0f}s, rss {G.rss_now()})", flush=True)
    blocks.append((te.mg.values, te.yb.values, p, p0))
    preds.append(pd.DataFrame(dict(cell=te.cell.values, ksic=te.ksic.values, mg=te.mg.values, t0=v, p=p, p_base=p0, y=te.y.values, yb=te.yb.values,
                                   p_logit=comps[0], p_xgbc=comps[1], p_nb=comps[2])))
out = dict(bundle=b, seed=SEED, mode=mode, origins=origins, mean_skill=float(np.mean([r['skill'] for r in res.values()])),
           mean_auc=float(np.mean([r['model']['auc'] for r in res.values()])), by_origin={str(k): v for k, v in res.items()})
P = pd.concat(preds, ignore_index=True)
if mode == 'final':
    out['ci95_M0'] = cluster_boot(blocks, 1000, seed=0)
    print('M0 CI', out['ci95_M0']['skill'], round(time.time() - t), flush=True)
    # ---- SB3 (VERIFY_R5 sens_t3.py 규칙: 원천 feat 캐시, 학습 t0 2005–18, tot_W ≥ 50, label_block, XGB seed σ) ----
    CACHE = os.path.join(R3, 'cache_pub_final'); TEST = [2021]; TRAIN = list(range(2005, 2019))
    def hkey(cells): return np.array([int(hashlib.md5(c.encode()).hexdigest()[:8], 16) for c in cells], dtype=np.int64)
    def label_block(g):
        k = max(1, int(round(0.10 * len(g)))); o = np.lexsort((hkey(g.cell.values), -(g.kr_fut - g.kr_W).values, -g.y.values))
        yb = np.zeros(len(g), int); yb[o[:k]] = 1; return yb
    Fd = {tt: pd.read_parquet(os.path.join(CACHE, f'feat_{tt}.parquet')) for tt in TRAIN + TEST}
    TR = []
    for tt in TRAIN:
        g = Fd[tt][Fd[tt].tot_W >= 50].reset_index(drop=True); g['yb'] = label_block(g); TR.append(g)
    TR = pd.concat(TR, ignore_index=True)
    TE = pd.concat([Fd[tt][Fd[tt].tot_W >= 50] for tt in TEST], ignore_index=True)
    TE = TE.merge(P[['cell', 't0', 'p', 'p_base', 'yb']], on=['cell', 't0'], how='inner'); assert len(TE) == len(P)
    fs = ['kr_mom', 'kr_mom_l1', 'kr_mom_l2', 'kr_mom5', 'kr_g1', 'kr_acc', 'kr_lvl', 'kr_last', 'kr_slope5', 'kr_zero_yrs', 'kr_share', 'kr_share_chg']
    bst = xgb.train(dict(objective='binary:logistic', max_depth=2, eta=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, nthread=2, verbosity=0, seed=SEED),
                    xgb.DMatrix(TR[fs].values, label=TR.yb.values), 450)
    TE['SB3'] = bst.predict(xgb.DMatrix(TE[fs].values))
    a_m, a_s = roc_auc_score(TE.yb, TE.p), roc_auc_score(TE.yb, TE.SB3)
    out['SB3'] = dict(auc_sb=float(a_s), auc_model=float(a_m), skill_vs_sb=float((a_m - a_s) / (1 - a_s)), seed=SEED)
    # ci_sb_t3.py 규칙: t0 블록, IPC 메인그룹, default_rng(11), 1000회
    d = TE
    blk = [(g.mg.values, g.yb.values, g.p.values, {cc: g[cc].values for cc in ['p_base', 'SB3']}) for _, g in d.groupby('t0')]
    ums = np.unique(d.mg.values); gi = [pd.Series(np.arange(len(bb[0]))).groupby(bb[0]).apply(lambda s: s.values).to_dict() for bb in blk]
    rng = np.random.default_rng(11); RR = {cc: [] for cc in ['p_base', 'SB3']}
    for _ in range(1000):
        smp = rng.choice(ums, len(ums)); acc = {cc: [] for cc in RR}
        for bb, g in zip(blk, gi):
            ii = np.concatenate([g[u] for u in smp if u in g]); y = bb[1][ii]
            if y.min() == y.max(): continue
            am = roc_auc_score(y, bb[2][ii])
            for cc in RR: a0 = roc_auc_score(y, bb[3][cc][ii]); acc[cc].append((am - a0) / (1 - a0))
        for cc in RR: RR[cc].append(np.mean(acc[cc]))
    out['SB3']['ci95'] = [float(np.percentile(RR['SB3'], 2.5)), float(np.percentile(RR['SB3'], 97.5))]
    out['SB3']['ci95_p_base_sbrule'] = [float(np.percentile(RR['p_base'], 2.5)), float(np.percentile(RR['p_base'], 97.5))]
    P = P.merge(TE[['cell', 't0', 'SB3']], on=['cell', 't0'], how='left')
    print('SB3', out['SB3'], round(time.time() - t), flush=True)
out['secs'] = round(time.time() - t); out['peak_gb'] = G.rss_peak(); out['created'] = time.strftime('%Y-%m-%d %H:%M:%S')
json.dump(out, open(os.path.join(OUT, f'b{b}_{mode}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
P.to_parquet(os.path.join(OUT, f'preds_b{b}_{mode}.parquet'))
print(f"[b{b}/{mode}] 평균 skill {out['mean_skill']:.6f} AUC {out['mean_auc']:.6f} 총 {out['secs']}s peak {out['peak_gb']}", flush=True)
