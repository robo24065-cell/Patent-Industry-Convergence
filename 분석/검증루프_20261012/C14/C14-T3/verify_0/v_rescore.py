# -*- coding: utf-8 -*-
"""verify_0 (ii): 최종 예측 독립 재채점 — 동결 blend·mets 대신 0.9·pct(px)+0.1·pct(pl)(R3 blend 정의를 문서로만 참조)와 순위합 AUC.
 최강 기준선 AUC는 공식 final_result_r3.json 값(0.810780)을 쓴다. 레이블은 C10-T1 final_labels(공식 재현 확인됨).
 python v_rescore.py S1 S1v AN_A AN_B ...  → verify_0/v_rescore_out.json"""
import sys, json, pathlib
import numpy as np, pandas as pd
from scipy.stats import rankdata
HERE = pathlib.Path(__file__).resolve().parent; C = HERE.parent
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
R3 = B/'분석/모델경쟁_20260927/T1_산업대기실/R3'
L = pd.read_parquet(B/'분석/검증루프_20261012/C10/C10-T1/cache/final_labels.parquet')
off = json.load(open(R3/'final_result_r3.json', encoding='utf-8'))
base = None
for k, v in off.items():
    pass
def auc(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def score(p):
    o = pd.read_parquet(p); assert (o.firm.values == L.firm.values).all()
    m = ~L.prior_any.values; o = o[m]; y = L.y_u.values[m].astype(int)
    s = 0.9 * rankdata(o.px.values) / len(o) + 0.1 * rankdata(o.pl.values) / len(o)
    return auc(y, s)
b_auc = off['model']['auc_all'] - off['skill']['auc_all'] * (1 - off['model']['auc_all']) / (1 - off['skill']['auc_all']) if False else None
# 최강 기준선 AUC: skill = (A - b)/(1 - b) → b = (A - s)/(1 - s)
A0, s0 = off['model']['auc_all'], off['skill']['auc_all']; b = (A0 - s0) / (1 - s0)
OUT = {'baseline_auc_from_official': b}
a_off = score(R3/'runs/FROZEN_final/o2019.parquet'); OUT['S0'] = dict(auc=a_off, skill=(a_off - b) / (1 - b), d_auc_vs_json=a_off - A0)
for v in sys.argv[1:]:
    p = C/'runs'/v/'o2019.parquet'
    if not p.exists(): OUT[v] = None; continue
    a = score(p); OUT[v] = dict(auc=a, skill=(a - b) / (1 - b), d_skill=(a - b) / (1 - b) - OUT['S0']['skill'])
json.dump(OUT, open(HERE/'v_rescore_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(OUT, ensure_ascii=False, indent=1))
