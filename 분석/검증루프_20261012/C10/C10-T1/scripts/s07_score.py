# -*- coding: utf-8 -*-
"""s07: 민감도 채점(진단 전용). score_final()과 같은 계산을 이 폴더에서 다시 한다(공식 결과 파일은 건드리지 않음).
 python s07_score.py labels          최종창 평가 레이블(2020–23) 계산 → cache/final_labels.parquet (1회)
 python s07_score.py final           runs/<var>/o2019.parquet 채점 + FROZEN_final 재채점 → cache/final_metrics.json
 python s07_score.py boot            쌍 부트스트랩(공식과 같은 법인 재표본·같은 난수, 기준선 AUC는 공식 r3boot 파일) → cache/boot_<ci>.npy, cache/boot_summary.json
 python s07_score.py dev             개발 원점 채점(FROZEN_dev 대비) → cache/dev_metrics.csv"""
import sys, os, json, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard                                   # 쓰기 가드
HERE = guard.HERE
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
T1 = B/'분석/모델경쟁_20260927/T1_산업대기실'; R3 = T1/'R3'
sys.path.insert(0, str(R3)); os.chdir(str(T1))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
T0 = time.time()
FC = R3/'cache_final'/'o2019'
VARS = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5']


def labels():
    import final_run_r3 as F, r3, r2
    te = pd.read_parquet(FC/'prep_te.parquet', columns=['firm', 'k', 'cut', 'newc', 'prior_any'])
    r2.light_load()
    with F.guard(2023):
        te['y_reg'] = r3.reg_lab(te, None); te['y_pub'] = r3.pub_lab(te, r3.END)
    te['y_u'] = np.maximum(te.y_reg.values, te.y_pub.values)
    te[['firm', 'k', 'newc', 'prior_any', 'y_reg', 'y_u']].to_parquet(HERE/'cache/final_labels.parquet', index=False)
    print('labels', int(te.y_u[~te.prior_any.values].sum()))


def load_final(var):
    from final_run_r3 import blend
    p = (R3/'runs'/'FROZEN_final'/'o2019.parquet') if var == 'OFFICIAL' else (HERE/'runs'/var/'o2019.parquet')
    if not p.exists(): return None
    o = pd.read_parquet(p); L = pd.read_parquet(HERE/'cache/final_labels.parquet')
    assert (o.firm.values == L.firm.values).all() and (o.k.values == L.k.values).all()
    keep = ~L.prior_any.values
    return blend(o[keep].reset_index(drop=True)), L[keep].reset_index(drop=True), o[keep].reset_index(drop=True)


def final():
    from r3 import mets, skill
    off = json.load(open(R3/'final_result_r3.json', encoding='utf-8'))
    bs = pd.read_parquet(FC/'baseline_scores.parquet')
    L = pd.read_parquet(HERE/'cache/final_labels.parquet'); keep = ~L.prior_any.values; bs = bs.loc[keep].reset_index(drop=True)
    res = {}
    for var in ['OFFICIAL'] + VARS:
        r = load_final(var)
        if r is None: continue
        s, t, o = r; y = t.y_u.values; nc = t.newc.values.astype(bool)
        m = mets(y, s, nc); bm = {k: mets(y, bs[k].values, nc) for k in bs.columns}
        st = {q: max(v[q] for v in bm.values()) for q in ('auc_all', 'auc_new')}
        od = np.argsort(-s)
        res[var] = dict(auc_all=m['auc_all'], auc_new=m['auc_new'], pr_all=m['pr_all'], base_auc_all=st['auc_all'], base_auc_new=st['auc_new'],
                        skill=skill(m['auc_all'], st['auc_all']), skill_new=skill(m['auc_new'], st['auc_new']),
                        top1000=int(y[od[:1000]].sum()), top100=int(y[od[:100]].sum()), capture_top1pct=float(y[od[:len(y) // 100]].sum() / y.sum()),
                        n=int(len(y)), pos=int(y.sum()))
        print(var, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in res[var].items()}, flush=True)
    res['official_json'] = dict(auc_all=off['model']['auc_all'], skill=off['skill']['auc_all'], auc_new=off['model']['auc_new'], top1000=off['topk']['model']['1000']['hits'])
    if 'S0' in res:
        res['S0_repro'] = dict(d_auc=res['S0']['auc_all'] - off['model']['auc_all'], d_skill=res['S0']['skill'] - off['skill']['auc_all'],
                               d_auc_new=res['S0']['auc_new'] - off['model']['auc_new'])
        a = pd.read_parquet(HERE/'runs/S0/o2019.parquet'); b = pd.read_parquet(R3/'runs/FROZEN_final/o2019.parquet')
        res['S0_repro']['max_abs_px'] = float(np.abs(a.px.values - b.px.values).max()); res['S0_repro']['max_abs_pl'] = float(np.abs(a.pl.values - b.pl.values).max())
    for var in VARS[1:]:
        if var in res and 'S0' in res:
            res[var]['d_skill'] = res[var]['skill'] - res['S0']['skill']; res[var]['d_auc'] = res[var]['auc_all'] - res['S0']['auc_all']
            res[var]['d_skill_new'] = res[var]['skill_new'] - res['S0']['skill_new']; res[var]['d_top1000'] = res[var]['top1000'] - res['S0']['top1000']
    json.dump(res, open(HERE/'cache/final_metrics.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)


def boot(R=200, chunk=10, budget=500):
    S = {}; t = None
    for var in VARS:
        r = load_final(var)
        if r is not None: S[var] = r[0]; t = r[1]
    keys = list(S); y = t.y_u.values; firm = t.firm.values
    fr_, inv = np.unique(firm, return_inverse=True)
    order = np.argsort(inv, kind='stable'); st = np.searchsorted(inv[order], np.arange(len(fr_))); en = np.searchsorted(inv[order], np.arange(len(fr_)), side='right')
    for ci in range(int(np.ceil(R / chunk))):
        fp = HERE/'cache'/f'boot_{ci}.npz'
        if fp.exists() and set(np.load(fp, allow_pickle=True)['keys']) >= set(keys): continue
        if time.time() - T0 > budget: print('budget stop at chunk', ci); sys.exit(3)
        off = np.load(FC/f'r3boot_R200_{ci}.npy')            # (10, 10 keys, 2): 모델 4 + 기준선 6
        rng = np.random.RandomState(3000 + ci); arr = np.zeros((min(chunk, R - ci * chunk), len(keys)))
        for r in range(arr.shape[0]):
            idx = np.concatenate([order[st[i]:en[i]] for i in rng.randint(0, len(fr_), len(fr_))]); yy = y[idx]
            for j, k in enumerate(keys): arr[r, j] = roc_auc_score(yy, S[k][idx])
        np.savez(fp, keys=np.array(keys), auc=arr, off_model=off[:arr.shape[0], 0, 0], bmax=off[:arr.shape[0], 4:, 0].max(1))
        print('chunk', ci, round(time.time() - T0), flush=True)
    A = []; OM = []; BM = []
    for ci in range(int(np.ceil(R / chunk))):
        z = np.load(HERE/'cache'/f'boot_{ci}.npz', allow_pickle=True); kk = list(z['keys'])
        A.append(z['auc'][:, [kk.index(k) for k in keys]]); OM.append(z['off_model']); BM.append(z['bmax'])
    A = np.concatenate(A); OM = np.concatenate(OM); BM = np.concatenate(BM)
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.median(v)), float(np.percentile(v, 97.5))]
    out = dict(R=int(len(A)), keys=keys)
    j0 = keys.index('S0'); out['S0_vs_official_boot_max_abs'] = float(np.abs(A[:, j0] - OM).max())
    sk = (A - BM[:, None]) / (1 - BM[:, None])
    out['S0_skill_ci'] = q(sk[:, j0])
    for j, k in enumerate(keys):
        if k == 'S0': continue
        out[k] = dict(d_skill=q(sk[:, j] - sk[:, j0]), d_auc=q(A[:, j] - A[:, j0]), skill=q(sk[:, j]), share_below_zero=float(((sk[:, j] - sk[:, j0]) < 0).mean()))
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
    R.to_csv(HERE/'cache/dev_metrics.csv', index=False, encoding='utf-8-sig'); print(R.round(5).to_string(index=False))


if __name__ == '__main__':
    {'labels': labels, 'final': final, 'boot': boot, 'dev': dev}[sys.argv[1]]()
