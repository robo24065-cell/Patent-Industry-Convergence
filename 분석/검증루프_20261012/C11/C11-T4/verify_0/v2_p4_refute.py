# verify_0 V2: P4(b) 반박 시도 — 다른 학습기(HGB)·다른 음성 추출(시드 777·10%)·IPC 겹침 피처 제거·역방향(공개 세계 결과) 설계.
# 사용: python -I v2_p4_refute.py v   (v ∈ 2022, 2023, 2024)
import sys, json, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.ensemble import HistGradientBoostingClassifier

C = 'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/검증루프_20261012/C11/C11-T4/cache'
OUT = 'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/검증루프_20261012/C11/C11-T4/verify_0'
v = int(sys.argv[1])
YS = [y for y in range(2017, 2025) if y < v]
t = time.time()
KF = ['nk3', 'dens', 'maxphi', 'meanq', 'shq10', 'ck_log3', 'ck_gr', 'ck_nfirm', 'rfirm', 'rksic']
BASE = ['f_log3', 'f_log1', 'f_gr', 'is_cn']


def fs(w, drop_ipc=False, own=False):
    k = [f for f in KF if not (drop_ipc and f in ('meanq', 'shq10'))]
    return BASE + [f'{w}_{f}' for f in k] + ([f'{w}_own_sh3', f'{w}_own_n1'] if own else [])


designs = {  # name: (risk/y world, feature list)
    'a_p': ('p', fs('p')), 'a_fi': ('p', fs('fi', own=True)), 'a_hs': ('p', fs('hs', own=True)),
    'b_p': ('p', fs('p', True)), 'b_fi': ('p', fs('fi', True, own=True)), 'b_hs': ('p', fs('hs', True, own=True)),
    'c_fi_p': ('fi', fs('p', own=True)), 'c_fi_fi': ('fi', fs('fi', own=True)),
    'c_hs_p': ('hs', fs('p', own=True)), 'c_hs_hs': ('hs', fs('hs', own=True)),
}
cols = sorted(set(sum([d[1] for d in designs.values()], [])) | {'cc', 'aid', 'k', 'risk_p', 'y_p', 'risk_fi', 'y_fi', 'risk_hs', 'y_hs'})
data = {Y: pd.read_parquet(f'{C}/u_A_{Y}.parquet', columns=cols) for Y in YS + [v]}
preds = {}
with threadpool_limits(2):
    for name, (w, feats) in designs.items():
        rk, yk = f'risk_{w}', f'y_{w}'
        tr = pd.concat([data[Y][data[Y][rk]] for Y in YS], ignore_index=True)
        te = data[v][data[v][rk]]
        y = tr[yk].values.astype(int)
        rng = np.random.default_rng(777)
        keep = (y == 1) | (rng.random(len(tr)) < 0.10)
        sw = np.where(y == 1, 1.0, 10.0)[keep]
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=50, random_state=777)
        m.fit(tr.loc[keep, feats].values.astype(np.float32), y[keep], sample_weight=sw)
        s = m.predict_proba(te[feats].values.astype(np.float32))[:, 1]
        fid = te.aid.values + np.where(te.cc.values == 'EP', 10_000_000, 0)
        preds[name] = pd.DataFrame({'fid': fid, 'k': te.k.values, 'y': te[yk].values.astype(np.int8), 's': s})
        print(v, name, len(te), int(te[yk].sum()), round(time.time() - t, 1), flush=True)
for name, P in preds.items():
    P.to_parquet(f'{OUT}/v2_pred_{name}_{v}.parquet', index=False)
