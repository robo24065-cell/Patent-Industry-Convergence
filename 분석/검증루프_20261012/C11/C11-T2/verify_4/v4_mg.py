# -*- coding: utf-8 -*-
# verify_4 V4: T3 메인그룹 TXT − 맥락없음 ΔAUC (sklearn AUC). 사용: python v4_mg.py <arm> <seed> <burn_first:0|1>
import sys, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
B0 = "C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석"
T3 = B0 + "/모델경쟁_20260927/T3_기술셀부상예측"; WD = B0 + "/검증루프_20261012/C11/C11-T2"
arm, seed, burn = sys.argv[1], int(sys.argv[2]), sys.argv[3] == '1'
txtf = T3 + '/TXT_R9/results/mg_ctx-txt_preds.parquet' if arm == 'stored' else f'{WD}/res/runs/mg_ctx-txt__{arm}_preds.parquet'
TX = pd.read_parquet(txtf); NO = pd.read_parquet(T3 + '/NONCIRC_R8/results/mg_ctx-none_preds.parquet')
YRS = range(2010, 2019); rng = np.random.default_rng(seed); B = 1000
per = {}
for t in YRS:
    a = TX[TX.t0 == t].set_index('cell').sort_index(); b = NO[NO.t0 == t].set_index('cell').sort_index()
    assert a.index.equals(b.index) and (a.yb.values == b.yb.values).all()
    per[t] = (a.yb.to_numpy(), a.p.to_numpy(), b.p.to_numpy())
if burn:  # R9 s05 쌍 순서: prov_vs_txt가 먼저 같은 rng를 소비
    for t in YRS: rng.integers(0, len(per[t][0]), (B, len(per[t][0])))
d, bs = [], np.full((9, B), np.nan)
for i, t in enumerate(YRS):
    y, pa, pb = per[t]; d.append(roc_auc_score(y, pa) - roc_auc_score(y, pb))
    I = rng.integers(0, len(y), (B, len(y)))
    for j in range(B):
        yy = y[I[j]]
        if 0 < yy.sum() < len(yy): bs[i, j] = roc_auc_score(yy, pa[I[j]]) - roc_auc_score(yy, pb[I[j]])
d = np.array(d); bm = np.nanmean(bs, 0)
r = dict(arm=arm, seed=seed, burn=burn, auc_txt=float(np.mean([roc_auc_score(per[t][0], per[t][1]) for t in YRS])), d=float(d.mean()),
         lo=float(np.nanpercentile(bm, 2.5)), hi=float(np.nanpercentile(bm, 97.5)), txt_wins=int((d > 0).sum()))
print(json.dumps(r))
with open(WD + '/verify_4/v4_mg.jsonl', 'a', encoding='utf-8') as f: f.write(json.dumps(r) + '\n')
