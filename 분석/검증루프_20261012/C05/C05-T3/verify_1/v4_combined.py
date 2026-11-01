# -*- coding: utf-8 -*-
"""verify_1 v4: 권리목록 확인 22곳 + 미조회 후보(v3 확률) 결합 Δskill, 그리고 '조회된 후보의 경험적 전환율' 적용 시나리오."""
import pandas as pd, numpy as np, sys, io, json, runpy
from sklearn.metrics import roc_auc_score
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
B = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
MC = B + r"\분석\모델경쟁_20260927"
V = B + r"\분석\검증루프_20261012\C05\C05-T3\verify_1"
pr = pd.read_parquet(MC + r"\N6_정부RnD기업_첫특허전환\final_preds_r5.parquet"); pr['bz'] = pr.bz.astype(str).str.zfill(10)
def sk(y):
    A = roc_auc_score(y, pr.p_champ); Bv = roc_auc_score(y, pr.p_m0); return (A - Bv) / (1 - Bv)
base = sk(pr.y)
# 22곳 재도출 없이 v1 산출 재사용 불가 → v1 스크립트의 newpos는 파일에 없으므로 n6 s6 규칙으로 저장된 후보 중 교집합 대신 시뮬레이션은 개수 기준
cand = pd.read_csv(V + r"\v1_y0_name_candidates.csv", dtype=str).bz
v1 = json.load(open(V + r"\v1_out.json", encoding='utf-8'))
v3 = json.load(open(V + r"\v3_out.json", encoding='utf-8'))
o = {'base': base}
# 경험적 전환율 시나리오: 조회된 후보 40곳 중 15곳 전환(37.5%) → 미조회 후보 357곳에 같은 비율
rate = v1['newpos_in_cand'] / v1['y0_cand_queried_ok']
nq = v1['y0_cand_not_queried']
rng = np.random.default_rng(2)
y0c = [b for b in cand if b in set(pr.bz[pr.y == 0])]
res = []
for i in range(200):
    s = set(rng.choice(y0c, int(round(nq * rate)), replace=False))
    res.append(sk((pr.y | pr.bz.isin(s)).astype(int)) - base)
o['empirical_rate'] = rate; o['flips'] = int(round(nq * rate))
o['empirical_dskill_mean_p5_p95'] = [float(np.mean(res)), float(np.percentile(res, 5)), float(np.percentile(res, 95))]
json.dump(o, open(V + r"\v4_out.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(o, ensure_ascii=False, indent=1))
