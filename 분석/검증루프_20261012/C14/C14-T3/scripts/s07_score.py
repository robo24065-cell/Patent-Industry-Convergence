# -*- coding: utf-8 -*-
"""s07: 민감도 채점(진단 전용, 공식 결과 파일 미변경). C10-T1 s07_score.py와 같은 계산.
 python s07_score.py final              runs/<var>/o2019.parquet 채점 → cache/final_metrics.json
 python s07_score.py boot <part> <np>   쌍 부트스트랩 chunk ci % np == part 만(공식 재표본 RandomState(3000+ci), 기준선 AUC = 공식 r3boot) → cache/boot_<ci>.npz
 python s07_score.py bootsum            → cache/boot_summary.json
 python s07_score.py dev                개발 원점 채점(FROZEN_dev 대비) → cache/dev_metrics.csv
 최종창 레이블: C10-T1 cache/final_labels.parquet(읽기만; 동결 score_final과 같은 r3.reg_lab/pub_lab, C10에서 공식 AUC 재현 차 0)."""
import sys, os, json, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard
HERE = guard.HERE
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
T1 = B/'분석/모델경쟁_20260927/T1_산업대기실'; R3 = T1/'R3'; LOOP = B/'분석/검증루프_20261012'
sys.path.insert(0, str(R3)); os.chdir(str(T1))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
T0 = time.time()
FC = R3/'cache_final'/'o2019'
LAB = LOOP/'C10/C10-T1/cache/final_labels.parquet'
VARS = ['P0', 'S1', 'S1v', 'S2', 'S3_1', 'S3_2', 'S3_3', 'S3_4', 'S3_5', 'S4', 'AN_A', 'AN_B', 'AN_TI']


def load_final(var):
    from final_run_r3 import blend
    p = (R3/'runs'/'FROZEN_final'/'o2019.parquet') if var == 'S0' else (HERE/'runs'/var/'o2019.parquet')
    if not p.exists(): return None
    o = pd.read_parquet(p); L = pd.read_parquet(LAB)
    assert (o.firm.values == L.firm.values).all() and (o.k.values == L.k.values).all()
    keep = ~L.prior_any.values
    return blend(o[keep].reset_index(drop=True)), L[keep].reset_index(drop=True), o[keep].reset_index(drop=True)


def final():
    from r3 import mets, skill
    off = json.load(open(R3/'final_result_r3.json', encoding='utf-8'))
    bs = pd.read_parquet(FC/'baseline_scores.parquet')
    L = pd.read_parquet(LAB); keep = ~L.prior_any.values; bs = bs.loc[keep].reset_index(drop=True)
    res = {}
    for var in ['S0'] + VARS:
        r = load_final(var)
        if r is None: continue
        s, t, o = r; y = t.y_u.values; nc = t.newc.values.astype(bool)
        m = mets(y, s, nc); bm = {k: mets(y, bs[k].values, nc) for k in bs.columns}
        st = {q: max(v[q] for v in bm.values()) for q in ('auc_all', 'auc_new')}
        od = np.argsort(-s)
        res[var] = dict(auc_all=m['auc_all'], auc_new=m['auc_new'], base_auc_all=st['auc_all'], base_auc_new=st['auc_new'],
                        skill=skill(m['auc_all'], st['auc_all']), skill_new=skill(m['auc_new'], st['auc_new']),
                        top1000=int(y[od[:1000]].sum()), capture_top1pct=float(y[od[:len(y) // 100]].sum() / y.sum()), n=int(len(y)), pos=int(y.sum()))
        print(var, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in res[var].items()}, flush=True)
    res['official_json'] = dict(auc_all=off['model']['auc_all'], skill=off['skill']['auc_all'], auc_new=off['model']['auc_new'], top1000=off['topk']['model']['1000']['hits'])
    res['S0_vs_official_json'] = dict(d_auc=res['S0']['auc_all'] - off['model']['auc_all'], d_skill=res['S0']['skill'] - off['skill']['auc_all'])
    if 'P0' in res:
        a = pd.read_parquet(HERE/'runs/P0/o2019.parquet'); b = pd.read_parquet(R3/'runs/FROZEN_final/o2019.parquet')
        res['P0_repro'] = dict(d_auc=res['P0']['auc_all'] - res['S0']['auc_all'], d_skill=res['P0']['skill'] - res['S0']['skill'],
                               max_abs_px=float(np.abs(a.px.values - b.px.values).max()), max_abs_pl=float(np.abs(a.pl.values - b.pl.values).max()))
    for var in VARS:
        if var in res:
            for q in ('skill', 'auc_all', 'skill_new', 'top1000'):
                res[var]['d_' + q] = res[var][q] - res['S0'][q]
    if 'AN_A' in res and 'AN_B' in res:
        res['AN_BminusA'] = {q: res['AN_B'][q] - res['AN_A'][q] for q in ('skill', 'auc_all', 'skill_new', 'top1000')}
    json.dump(res, open(HERE/'cache/final_metrics.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)


def boot(part, npart, R=200, chunk=10, budget=500):
    S = {}; t = None
    for var in ['S0'] + VARS:
        r = load_final(var)
        if r is not None: S[var] = r[0]; t = r[1]
    keys = list(S); y = t.y_u.values; firm = t.firm.values
    fr_, inv = np.unique(firm, return_inverse=True)
    order = np.argsort(inv, kind='stable'); st = np.searchsorted(inv[order], np.arange(len(fr_))); en = np.searchsorted(inv[order], np.arange(len(fr_)), side='right')
    for ci in range(int(np.ceil(R / chunk))):
        if ci % npart != part: continue
        fp = HERE/'cache'/f'boot_{ci}.npz'
        if fp.exists() and set(np.load(fp, allow_pickle=True)['keys']) >= set(keys): continue
        if time.time() - T0 > budget: print('budget stop at chunk', ci); sys.exit(3)
        off = np.load(FC/f'r3boot_R200_{ci}.npy')
        rng = np.random.RandomState(3000 + ci); arr = np.zeros((min(chunk, R - ci * chunk), len(keys)))
        for r in range(arr.shape[0]):
            idx = np.concatenate([order[st[i]:en[i]] for i in rng.randint(0, len(fr_), len(fr_))]); yy = y[idx]
            for j, k in enumerate(keys): arr[r, j] = roc_auc_score(yy, S[k][idx])
        np.savez(fp, keys=np.array(keys), auc=arr, off_model=off[:arr.shape[0], 0, 0], bmax=off[:arr.shape[0], 4:, 0].max(1))
        print('chunk', ci, round(time.time() - T0), flush=True)


def bootsum(R=200, chunk=10):
    A = []; OM = []; BM = []; keys = None
    for ci in range(int(np.ceil(R / chunk))):
        z = np.load(HERE/'cache'/f'boot_{ci}.npz', allow_pickle=True); kk = list(z['keys'])
        keys = kk if keys is None else [k for k in keys if k in kk]
        A.append((z['auc'], kk)); OM.append(z['off_model']); BM.append(z['bmax'])
    A = np.concatenate([a[:, [kk.index(k) for k in keys]] for a, kk in A]); OM = np.concatenate(OM); BM = np.concatenate(BM)
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.median(v)), float(np.percentile(v, 97.5))]
    out = dict(R=int(len(A)), keys=keys); j0 = keys.index('S0')
    out['S0_vs_official_boot_max_abs'] = float(np.abs(A[:, j0] - OM).max())
    sk = (A - BM[:, None]) / (1 - BM[:, None]); out['S0_skill_ci'] = q(sk[:, j0])
    for j, k in enumerate(keys):
        if k == 'S0': continue
        out[k] = dict(d_skill=q(sk[:, j] - sk[:, j0]), d_auc=q(A[:, j] - A[:, j0]), skill=q(sk[:, j]), share_below_zero=float(((sk[:, j] - sk[:, j0]) < 0).mean()))
    if 'AN_A' in keys and 'AN_B' in keys:
        a, b = keys.index('AN_A'), keys.index('AN_B'); out['AN_BminusA'] = dict(d_skill=q(sk[:, b] - sk[:, a]), share_below_zero=float(((sk[:, b] - sk[:, a]) < 0).mean()))
    s3 = [keys.index(k) for k in keys if k.startswith('S3_')]
    if s3: out['S3_mean'] = dict(d_skill=q(sk[:, s3].mean(1) - sk[:, j0]))
    json.dump(out, open(HERE/'cache/boot_summary.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print(json.dumps(out, ensure_ascii=False, indent=1))


def dev():
    import r3
    from r3 import mets, skill, baselines
    from final_run_r3 import blend
    rows = []
    dirs = [('FULL', R3/'runs'/'FROZEN_dev')] + [(v, HERE/'runs'/v) for v in VARS]
    for var, d in dirs:
        for c in r3.ORIGINS:
            f = d/f'o{c}.parquet'
            if not f.exists(): continue
            o = pd.read_parquet(f); o = o[~o.prior_any.values]; s = blend(o)
            m = mets(o.y_u.values, s, o.newc.values); b = baselines(c)['strongest']
            rows.append(dict(variant=var, origin=c, auc=m['auc_all'], auc_new=m['auc_new'], base_auc=b['auc_all'], skill=skill(m['auc_all'], b['auc_all']),
                             skill_new=skill(m['auc_new'], b['auc_new']), top1000=int(o.y_u.values[np.argsort(-s)[:1000]].sum())))
    R = pd.DataFrame(rows)
    full = R[R.variant == 'FULL'].set_index('origin')
    R['d_skill'] = [r.skill - full.skill.get(r.origin, np.nan) for r in R.itertuples()]
    R['d_auc'] = [r.auc - full.auc.get(r.origin, np.nan) for r in R.itertuples()]
    if 'P0' in set(R.variant):
        for r in R[R.variant == 'P0'].itertuples():
            a = pd.read_parquet(HERE/'runs/P0'/f'o{r.origin}.parquet'); b = pd.read_parquet(R3/'runs/FROZEN_dev'/f'o{r.origin}.parquet')
            print('P0 dev repro', r.origin, float(np.abs(a.px.values - b.px.values).max()), float(np.abs(a.pl.values - b.pl.values).max()))
    R.to_csv(HERE/'cache/dev_metrics.csv', index=False, encoding='utf-8-sig'); print(R.round(5).to_string(index=False))


if __name__ == '__main__':
    a = sys.argv[1]
    if a == 'boot': boot(int(sys.argv[2]), int(sys.argv[3]))
    else: {'final': final, 'bootsum': bootsum, 'dev': dev}[a]()
