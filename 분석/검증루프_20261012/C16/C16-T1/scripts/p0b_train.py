# -*- coding: utf-8 -*-
"""P0-b: 동결 스펙 재적합·모형 저장. python p0b_train.py logit | xgb0 | xgb1
 logit: C15 t1_btrain 2단계와 같은 수치(tr[cols≠ind2].fillna(0) → StandardScaler → LogisticRegression(max_iter=3000)) → models/logit.pkl, p0/pl.npy
 xgbS : final_run_r3.train_origin 3단계와 같은 파라미터(SPEC1 params + ec params, tree_method hist, nthread 2, random_state S, sample_weight) → models/xgb_seedS.json, p0/pxS.npy"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from c16t1 import *
import gc, pickle, time
import final_run_r3 as F
from r2 import fr, SPEC1
t0 = time.time(); ec = json.loads(json.dumps(F.EC)); cols = list(ec['cols']); job = sys.argv[1]
if job == 'logit':
    if (P0D/'pl.npy').exists(): print('exists'); sys.exit(0)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    tr = pd.read_parquet(P0D/'s1_tr.parquet'); lcols = [x for x in cols if x != 'ind2']
    X = tr[lcols].fillna(0).values; yv = tr.y.values; del tr; gc.collect()
    sc = StandardScaler().fit(X); Xs = sc.transform(X); del X; gc.collect()
    lr = LogisticRegression(max_iter=3000).fit(Xs, yv); del Xs; gc.collect()
    L = dict(sc=sc, lr=lr, cols=lcols); pickle.dump(L, open(MD/'logit.pkl', 'wb'))
    te = pd.read_parquet(P0D/'s1_te.parquet')
    pl = np.concatenate([fr.logit_pred(L, te.iloc[i:i + 200000]) for i in range(0, len(te), 200000)]).astype(np.float32)
    np.save(P0D/'pl.npy', pl); d = float(np.abs(pl - np.load(RD/'o2019_pl.npy')).max())
    print('logit maxdiff vs FROZEN', d, round(time.time() - t0), G.rss_peak(), flush=True)
else:
    import xgboost as xgb
    sd = int(job[3:]); fp = P0D/f'px{sd}.npy'
    if fp.exists(): print('exists'); sys.exit(0)
    params = dict(SPEC1['params']); params.update(ec.get('params', {}))
    tr = pd.read_parquet(P0D/'s1_tr.parquet'); w = np.load(P0D/'s1_w.npy')
    X = tr[cols].values.astype(np.float32); yv = tr.y.values; del tr
    p = dict(tree_method='hist', nthread=2, random_state=sd); p.update(params)
    m = xgb.XGBClassifier(**p); m.fit(X, yv, sample_weight=w); del X; gc.collect()
    m.get_booster().save_model(str(MD/f'xgb_seed{sd}.json')); json.dump(dict(params=p, cols=cols), open(MD/f'xgb_seed{sd}_meta.json', 'w', encoding='utf-8'), ensure_ascii=False)
    te = pd.read_parquet(P0D/'s1_te.parquet')
    px = np.concatenate([m.predict_proba(te[cols].iloc[i:i + 200000].values.astype(np.float32))[:, 1] for i in range(0, len(te), 200000)]).astype(np.float32)
    np.save(fp, px); d = float(np.abs(px - np.load(RD/f'o2019_px{sd}.npy')).max())
    # 저장한 json을 다시 읽어 예측이 같은지(저장 무손실) 확인
    bst = xgb.Booster(); bst.load_model(str(MD/f'xgb_seed{sd}.json'))
    px2 = bst.predict(xgb.DMatrix(te[cols].iloc[:200000].values.astype(np.float32))).astype(np.float32)
    print(f'xgb seed {sd} maxdiff vs FROZEN', d, 'reload maxdiff', float(np.abs(px2 - px[:200000]).max()), round(time.time() - t0), G.rss_peak(), flush=True)
