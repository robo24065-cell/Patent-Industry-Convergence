# -*- coding: utf-8 -*-
"""P0 SBB 차(약 2e−8) 원인 확인: VERIFY_R5는 모형 점수를 0.9·rankdata(px)/n + 0.1·rankdata(pl)/n 로 다시 만들었고(sens_t1.py 32행),
공식 score_final은 final_run_r3.blend(= 0.9·pandas rank(pct) + 0.1·…)를 쓴다. 같은 SBB 점수(b0, VR5와 차 0)로 VR5 식 점수를 쓰면 VR5 값이 비트 단위로 나오는지 본다.
 → cache/t1/p0_sbb_check.json"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard15 as G
import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
W = G.HERE; C1 = W/'cache'/'t1'
V = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/모델경쟁_20260927/T1_산업대기실/VERIFY_R5")
S = pd.read_parquet(V/'sens_t1_scores.parquet'); sc = pd.read_parquet(C1/'scores_b0.parquet')
y = sc.y.values; sbb = np.load(C1/'sbb_b0.npy')
out = dict(n_order_diff=int((np.argsort(S.s.values, kind='stable') != np.argsort(sc.s.values, kind='stable')).sum()),
           max_abs_score_diff=float(np.abs(S.s.values - sc.s.values).max()))
sk = lambda a, b: (a - b) / (1 - b)
a_v, a_o, a_s = roc_auc_score(y, S.s.values), roc_auc_score(y, sc.s.values), roc_auc_score(y, sbb)
out.update(auc_model_vr5_formula=a_v, auc_model_official_blend=a_o, skill_sbb_vr5_formula=sk(a_v, a_s), skill_sbb_official_blend=sk(a_o, a_s))
fr_, inv = np.unique(sc.firm.values, return_inverse=True); order = np.argsort(inv, kind='stable')
st = np.searchsorted(inv[order], np.arange(len(fr_))); en = np.searchsorted(inv[order], np.arange(len(fr_)), side='right')
rng = np.random.RandomState(4242); A = []
for r in range(200):
    idx = np.concatenate([order[st[i]:en[i]] for i in rng.randint(0, len(fr_), len(fr_))]); yy = y[idx]
    A.append([roc_auc_score(yy, S.s.values[idx]), roc_auc_score(yy, sbb[idx])])
A = np.array(A); v = (A[:, 0] - A[:, 1]) / (1 - A[:, 1])
out['ci_sbb_vr5_formula'] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
ref = json.load(open(V/'ci_sb_t1.json', encoding='utf-8'))['skill_vs_SBB']
out['d_vs_ci_sb_t1'] = [out['ci_sbb_vr5_formula'][0] - ref[0], out['ci_sbb_vr5_formula'][1] - ref[1]]
json.dump(out, open(C1/'p0_sbb_check.json', 'w', encoding='utf-8'), indent=1); print(out)
