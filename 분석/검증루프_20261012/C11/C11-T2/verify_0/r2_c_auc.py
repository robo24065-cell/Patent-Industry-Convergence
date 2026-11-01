# -*- coding: utf-8 -*-
# 2차 반박 검증 C: 내 재실행 예측(r2_runs) vs 작업자 reg 예측, 저장 R9 예측 vs r8 예측 대조.
# ΔAUC(제공 − TXT)·부트스트랩 CI를 내 AUC 함수(정렬 기반 Mann-Whitney, 평균 순위)로 재계산. 같은 추출 규칙(원점 안 mg 복원추출 B=1000, default_rng(20261005))과 다른 seed·B.
import json, numpy as np, pandas as pd
B_ = 'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전'
T3 = B_ + '/분석/모델경쟁_20260927/T3_기술셀부상예측'
W = B_ + '/분석/검증루프_20261012/C11/C11-T2'; V = W + '/verify_0'
ORIG = list(range(2010, 2019))


def my_auc(y, p):
    # 평균 순위(동률) Mann-Whitney U / (n1 n0)
    r = pd.Series(p).rank(method='average').values
    y = np.asarray(y).astype(bool); n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def load(f):
    return pd.read_parquet(f).sort_values(['t0', 'cell']).reset_index(drop=True)


prov = load(T3 + '/NONCIRC_R8/results/mg_ctx-prov_preds.parquet')
F = {'reg_worker': W + '/res/runs/mg_ctx-txt__reg_preds.parquet', 'reg_mine': V + '/r2_runs/mg_ctx-txt__reg__repr2_preds.parquet',
     'r8_worker': W + '/res/runs/mg_ctx-txt__r8_preds.parquet', 'stored_R9': T3 + '/TXT_R9/results/mg_ctx-txt_preds.parquet',
     'pure_worker': W + '/res/runs/mg_ctx-txt__pure_preds.parquet', 'pub_worker': W + '/res/runs/mg_ctx-txt__pub_preds.parquet'}
P = {k: load(f) for k, f in F.items()}
R = {}
R['maxdp_reg_mine_vs_worker'] = float(np.abs(P['reg_mine'].p.values - P['reg_worker'].p.values).max())
R['maxdp_r8_vs_stored'] = float(np.abs(P['r8_worker'].p.values - P['stored_R9'].p.values).max())
R['maxdp_reg_vs_stored'] = float(np.abs(P['reg_worker'].p.values - P['stored_R9'].p.values).max())
for k, d in P.items():
    assert (d.cell.values == prov.cell.values).all() and (d.yb.values == prov.yb.values).all() and (d.t0.values == prov.t0.values).all(), k


def paired(dA, dB, seed, B):
    rng = np.random.default_rng(seed); dd, bs = [], np.zeros((len(ORIG), B))
    for i, v in enumerate(ORIG):
        m = dA.t0.values == v; y = dA.yb.values[m]; pa = dA.p.values[m]; pb = dB.p.values[m]
        dd.append(my_auc(y, pa) - my_auc(y, pb))
        I = rng.integers(0, len(y), (B, len(y)))
        for j in range(B):
            ii = I[j]; yy = y[ii]
            bs[i, j] = np.nan if yy.sum() in (0, len(yy)) else my_auc(yy, pa[ii]) - my_auc(yy, pb[ii])
    dd = np.array(dd); bm = np.nanmean(bs, 0)
    return dict(d=round(float(dd.mean()), 6), lo=round(float(np.nanpercentile(bm, 2.5)), 6), hi=round(float(np.nanpercentile(bm, 97.5)), 6),
                wins=int((dd > 0).sum()), by_origin=[round(float(x), 4) for x in dd])


for k in ['reg_mine', 'reg_worker', 'stored_R9', 'pure_worker', 'pub_worker']:
    d = P[k]
    R[k] = dict(txt_auc=round(float(np.mean([my_auc(d.yb.values[d.t0.values == v], d.p.values[d.t0.values == v]) for v in ORIG])), 6),
                prov_auc=round(float(np.mean([my_auc(prov.yb.values[prov.t0.values == v], prov.p.values[prov.t0.values == v]) for v in ORIG])), 6),
                same_seed=paired(prov, d, 20261005, 1000))
    if k in ('reg_mine', 'pure_worker'):
        R[k]['seed777_B2000'] = paired(prov, d, 777, 2000)
    print(k, R[k], flush=True)
json.dump(R, open(V + '/r2_c.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in R.items() if k.startswith('maxdp')}))
