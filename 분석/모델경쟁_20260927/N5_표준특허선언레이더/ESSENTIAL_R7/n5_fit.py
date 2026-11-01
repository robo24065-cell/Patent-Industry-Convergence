# -*- coding: utf-8 -*-
"""R7 N5 필수성 검증 학습·채점 — 동결 E21 설정(XGB 고정 파라미터, 시드 0·1·2 평균, 학습 t0 ≤ v−2, t0 ≥2012) 그대로 피처/모집단만 바꿔 개발 원점 2014–2020.
  --test full | drop_<묶음> | conc_task | concm_task | conc_feat | concm_feat"""
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import n5_common as X
N = X.N
ap = argparse.ArgumentParser(); ap.add_argument('--test', required=True); a = ap.parse_args()
SPEC = json.load(open(os.path.join(X.N5, 'final_spec_r4.json'), encoding='utf-8'))
FE = SPEC['champion_features']; F0 = SPEC['baseline_M0']; SEEDS = tuple(SPEC['seeds']); ORIG = SPEC['dev']['eval_origins']
FAM5 = N.FEATS['M1']; ROS = N.FEATS['ros']
BUNDLES = {'fam5': FAM5, 'foreign': FAM5 + ROS, 'title': N.FEATS['title'], 'sg': N.FEATS['sg'], 'pct': N.FEATS['pct'],
           'provided_extra': FAM5 + ROS + N.FEATS['title'] + N.FEATS['sg'] + N.FEATS['pct'] + ['app_rec_rate'], 'ros': ROS}
t = time.time(); test = a.test
P = lambda name: pd.read_parquet(os.path.join(X.CACHE, f'panel_{name}.parquet'))
Do = P('orig'); feats = FE; Dm = Do; Db = Do; info = {}
if test.startswith('drop_'):
    feats = [f for f in FE if f not in BUNDLES[test[5:]]]; info['dropped'] = BUNDLES[test[5:]]
elif test in ('conc_task', 'concm_task'):
    Dm = Db = P(test.split('_')[0])
elif test in ('conc_feat', 'concm_feat'):
    H = P('hyb_' + test.split('_')[0])
    key = ['appl_no', 't0']
    Dm = Do[key + ['y', 'app']].merge(H.drop(columns=['y', 'app']), on=key, how='left', validate='1:1')
    assert Dm[FE].notna().any(axis=1).all(); assert (Dm.y.values == Do.y.values).all()
elif test != 'full':
    raise SystemExit('알 수 없는 test')
res, preds = {}, []
for v in ORIG:
    trm, tem = Dm[Dm.t0 <= v - 2], Dm[Dm.t0 == v]; trb, teb = Db[Db.t0 <= v - 2], Db[Db.t0 == v]
    pm = N.fit_pred(trm[feats].astype(float), trm.y, tem[feats].astype(float), seeds=SEEDS)
    pb = N.fit_pred(trb[F0].astype(float), trb.y, teb[F0].astype(float), seeds=SEEDS)
    y = tem.y.values; assert (y == teb.y.values).all()
    am, ab = N.auc(y, pm), N.auc(y, pb); k1 = max(1, int(0.01 * len(y)))
    r = dict(auc=float(am), auc_M0=float(ab), skill=float(N.skill(am, ab)), n=int(len(y)), pos=int(y.sum()),
             top100=int(y[np.argsort(-pm)[:100]].sum()), top100_M0=int(y[np.argsort(-pb)[:100]].sum()), top1pct_prec=float(y[np.argsort(-pm)[:k1]].mean()))
    if test in ('conc_feat', 'concm_feat'):   # 참고: 연계표 통계로 다시 만든 M0 대비
        pbh = N.fit_pred(trm[F0].astype(float), trm.y, tem[F0].astype(float), seeds=SEEDS); r['auc_M0_alt'] = float(N.auc(y, pbh)); r['skill_vs_M0_alt'] = float(N.skill(am, r['auc_M0_alt']))
    res[str(v)] = r
    print(f"[N5/{test}] t0={v} AUC {am:.4f} (M0 {ab:.4f}) skill {r['skill']:.4f} n={r['n']} pos={r['pos']} top100 {r['top100']}/{r['top100_M0']} | {time.time()-t:.0f}s", flush=True)
    preds.append(pd.DataFrame(dict(appl_no=tem.appl_no.values, app=tem.app.values, t0=v, y=y, p=pm, p_base=pb)))
summ = {k: float(np.mean([r[k] for r in res.values()])) for k in ['auc', 'auc_M0', 'skill', 'top100', 'top100_M0', 'top1pct_prec', 'n', 'pos']}
if test in ('conc_feat', 'concm_feat'): summ['skill_vs_M0_alt'] = float(np.mean([r['skill_vs_M0_alt'] for r in res.values()]))
summ['wins_vs_M0'] = int(sum(r['skill'] > 0 for r in res.values()))
out = dict(test=test, model=SPEC['champion_exp'], features=feats, origins=ORIG, summary=summ, by_origin=res, info=info,
           frozen_dev_mean_skill=SPEC['dev']['mean_skill_vs_M0'], created=time.strftime('%Y-%m-%d %H:%M:%S'), secs=round(time.time() - t))
if test == 'full':
    out['reproduced'] = bool(all(abs(res[str(v)]['auc'] - SPEC['dev']['auc_champ'][str(v)]) < 1e-4 and abs(res[str(v)]['auc_M0'] - SPEC['dev']['auc_M0'][str(v)]) < 1e-4 for v in ORIG))
json.dump(out, open(os.path.join(HERE, 'results', f'{test}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
pd.concat(preds, ignore_index=True).to_parquet(os.path.join(HERE, 'results', f'{test}_preds.parquet'))
print(json.dumps(dict(summary=summ, reproduced=out.get('reproduced')), ensure_ascii=False))
