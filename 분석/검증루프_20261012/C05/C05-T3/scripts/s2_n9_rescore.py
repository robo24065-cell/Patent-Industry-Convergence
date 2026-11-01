# -*- coding: utf-8 -*-
"""C05-T3 s2: 재구성 레이블(yA 등)로 N9 동결 예측 재채점 -> n9_rescore.json"""
import pandas as pd, numpy as np, json, os, sys, io, math
from sklearn.metrics import roc_auc_score
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
B = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
N9 = os.path.join(B, r"모델경쟁_20260927\N9_정부RnD과제_특허성과예측")
OUT = os.path.join(B, r"검증루프_20261012\C05\C05-T3")
pr = pd.read_parquet(os.path.join(N9, "final_preds_r6.parquet"))
lab = pd.read_parquet(os.path.join(OUT, "cache", "n9_lab.parquet"))
js = json.load(open(os.path.join(N9, "final_result_r6.json"), encoding="utf-8"))
d = pr.merge(lab[['pjt_no', 'yA', 'yB', 'yA_pat', 'yA_pyear', 'yA_ryear', 'yA_noreg', 'perform_grp']], on='pjt_no', how='left', validate='1:1')
assert d.yA.notna().all()
MODELS = ['p_champ', 'p_M0_L2', 'p_M0_orig', 'p_strong_L2', 'p_strong_orig', 'p_pilot_L2', 'p_pilot_orig']


def topk(y, p, frac=None, k=None, rule='ceil'):
    n = len(y)
    if k is None:
        k = math.ceil(frac * n) if rule == 'ceil' else (int(round(frac * n)) if rule == 'round' else int(frac * n))
    o = np.argsort(-p, kind='stable')[:k]
    return float(y[o].mean()), int(y[o].sum()), k


def score(ycol):
    y = d[ycol].astype(int).values
    r = {'n': int(len(y)), 'pos': int(y.sum()), 'base_rate': float(y.mean())}
    for m in MODELS:
        p = d[m].values
        r[f'auc_{m}'] = float(roc_auc_score(y, p))
        for fr in (0.01, 0.05, 0.10):
            r[f'top{int(fr*100)}_{m}'] = topk(y, p, fr)[0]
        r[f'hit100_{m}'] = topk(y, p, k=100)[1]
        r[f'top1_floor_{m}'] = topk(y, p, 0.01, rule='floor')[0]
    a = r['auc_p_champ']
    for ref in ['p_M0_L2', 'p_M0_orig', 'p_strong_L2', 'p_strong_orig', 'p_pilot_L2', 'p_pilot_orig']:
        ar = r[f'auc_{ref}']
        r[f'skill_vs_{ref}'] = (a - ar) / (1 - ar)
    yy = y; o = np.argsort(-d.p_champ.values, kind='stable')[:math.ceil(0.10 * len(y))]
    r['recall_top10'] = float(yy[o].sum() / yy.sum())
    return r


res = {v: score(v) for v in ['y', 'yA', 'yB', 'yA_pat', 'yA_pyear', 'yA_ryear', 'yA_noreg']}
# compare with JSON
cmp = {}
R = res['yA']
pairs = {'auc': R['auc_p_champ'], 'auc_M0L2': R['auc_p_M0_L2'], 'skill': R['skill_vs_p_M0_L2'], 'top1': R['top1_p_champ'],
         'top1_M0L2': R['top1_p_M0_L2'], 'top5': R['top5_p_champ'], 'top10': R['top10_p_champ'], 'recall_top10': R['recall_top10'],
         'hit100': R['hit100_p_champ'], 'hit100_M0L2': R['hit100_p_M0_L2'], 'auc_M0_orig': R['auc_p_M0_orig'],
         'skill_vs_M0_orig': R['skill_vs_p_M0_orig'], 'top1_M0_orig': R['top1_p_M0_orig'], 'auc_strong_L2': R['auc_p_strong_L2'],
         'skill_vs_strong_L2': R['skill_vs_p_strong_L2'], 'top1_strong_orig': R['top1_p_strong_orig'], 'auc_pilot_orig': R['auc_p_pilot_orig'],
         'skill_vs_pilot_orig': R['skill_vs_p_pilot_orig'], 'top1_pilot_orig': R['top1_p_pilot_orig'], 'top1_pilot_L2': R['top1_p_pilot_L2'],
         'skill_vs_pilot_L2': R['skill_vs_p_pilot_L2'], 'auc_strong_orig': R['auc_p_strong_orig'], 'skill_vs_strong_orig': R['skill_vs_p_strong_orig'],
         'n': R['n'], 'pos': R['pos'], 'base_rate': R['base_rate']}
for k, v in pairs.items():
    j = js.get(k)
    cmp[k] = {'rebuild': v, 'json': j, 'abs_diff': (abs(v - j) if j is not None else None)}
res['compare_yA_vs_json'] = cmp
res['max_abs_diff'] = max(x['abs_diff'] for x in cmp.values() if x['abs_diff'] is not None)
res['reproduced'] = bool(res['max_abs_diff'] <= 1e-4)
# subgroups (공식 정의 확인용: 기업/대학/출연연·국공립연)
sg = {}
g = d.perform_grp.astype(str)
grp_map = {'기업': g.isin(['대기업', '중소기업', '중견기업']), '대학': g.eq('대학'), '출연연·국공립연': g.isin(['출연연', '국공립연'])}
for nm, msk in grp_map.items():
    s = d[msk]
    if s.yA.nunique() < 2:
        continue
    a = roc_auc_score(s.yA, s.p_champ); b = roc_auc_score(s.yA, s.p_M0_L2)
    sg[nm] = {'n': int(len(s)), 'pos': int(s.yA.sum()), 'auc': a, 'auc_M0L2': b, 'skill': (a - b) / (1 - b),
              'json': js['subgroups'].get(nm)}
res['subgroups_yA'] = sg
res['perform_grp_values'] = g.value_counts().to_dict()
json.dump(res, open(os.path.join(OUT, 'n9_rescore.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('reproduced', res['reproduced'], 'max diff', res['max_abs_diff'])
for v in res:
    if v in ('compare_yA_vs_json', 'subgroups_yA', 'perform_grp_values', 'max_abs_diff', 'reproduced'):
        continue
    r = res[v]
    print(v, r['pos'], round(r['auc_p_champ'], 6), round(r['skill_vs_p_M0_L2'], 6), round(r['top1_p_champ'], 6),
          {m: r[f'hit100_{m}'] for m in MODELS})
print(json.dumps(sg, ensure_ascii=False, indent=0)[:2000])
print(res['perform_grp_values'])
