# -*- coding: utf-8 -*-
"""C05-T3 s9: N9 최종창 t+1(2023 출원) 과소 보고 가능성 민감도 (판정 외 참고, 공식 성적 불변).
(1) lag0만(출원=t) 레이블로 동결 예측 채점 — t+1 절단에 둔감한 정의
(2) t=2022의 'lag1에만 의존하는 양성' 비율을 t=2020·2021 수준으로 올렸을 때 부족분(최대 467건)을 음성에서 추가:
    무작위 / p_champ 비례 / p_M0_L2 비례 / 관측 lag1-only 양성의 champion 순위분포 모사, 각 200회
"""
import pandas as pd, numpy as np, json, os, sys, io
from sklearn.metrics import roc_auc_score
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
B = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
N9 = os.path.join(B, r"모델경쟁_20260927\N9_정부RnD과제_특허성과예측")
OUT = os.path.join(B, r"검증루프_20261012\C05\C05-T3"); CA = os.path.join(OUT, "cache")
pr = pd.read_parquet(os.path.join(N9, "final_preds_r6.parquet"))
la = pd.read_parquet(os.path.join(CA, 'n9_links_A.parquet'))
la['appl_year'] = la.appl_year.astype(int)
s0 = set(la[la.appl_year == 2022].ProjectID); s1 = set(la[la.appl_year == 2023].ProjectID)
y = pr.y.values
y0 = pr.pjt_no.isin(s0).astype(int).values
only1 = pr.pjt_no.isin(s1 - s0).values & (y == 1)
o = {}
def sc(yv):
    a = roc_auc_score(yv, pr.p_champ); b = roc_auc_score(yv, pr.p_M0_L2)
    k = int(np.ceil(0.01 * len(yv))); top = np.argsort(-pr.p_champ.values, kind='stable')[:k]
    return {'pos': int(yv.sum()), 'auc': a, 'auc_M0L2': b, 'skill': (a - b) / (1 - b), 'top1': float(yv[top].mean())}
o['official'] = sc(y); o['lag0_only_label'] = sc(y0); o['lag1_only_pos'] = int(only1.sum())
# 부족분 (s3: t=2021 lag1-only 1341 / lag0 3743 = 0.358)
s3 = json.load(open(os.path.join(CA, 's3_out.json'), encoding='utf-8'))
lag = {r['t']: r for r in s3['lag']}
ratio_ref = np.mean([(lag[t]['n_pos_window'] - lag[t]['n0']) / lag[t]['n0'] for t in (2020, 2021)])
ratio_dev = np.mean([(lag[t]['n_pos_window'] - lag[t]['n0']) / lag[t]['n0'] for t in range(2014, 2020)])
r22 = (lag[2022]['n_pos_window'] - lag[2022]['n0']) / lag[2022]['n0']
o['lag1only_over_lag0'] = {'t2022': r22, 'mean_2020_21': ratio_ref, 'mean_2014_19': ratio_dev}
n_add = max(0, int(round((ratio_ref - r22) * lag[2022]['n0'])))
o['n_add_upper'] = n_add
neg = np.where(y == 0)[0]
rng = np.random.default_rng(0)
rk = pd.Series(pr.p_champ.values).rank(pct=True).values
obs_rank = rk[only1]
def sim(kind, reps=200):
    out = []
    for _ in range(reps):
        if kind == 'random':
            pick = rng.choice(neg, n_add, replace=False)
        elif kind == 'p_champ':
            w = pr.p_champ.values[neg]; pick = rng.choice(neg, n_add, replace=False, p=w / w.sum())
        elif kind == 'p_M0_L2':
            w = pr.p_M0_L2.values[neg]; pick = rng.choice(neg, n_add, replace=False, p=w / w.sum())
        else:  # 관측 lag1-only 양성의 champion 순위 분포 모사
            tgt = rng.choice(obs_rank, n_add, replace=True)
            nr = rk[neg]; order = np.argsort(nr); srt = nr[order]
            idx = np.clip(np.searchsorted(srt, tgt), 0, len(srt) - 1)
            pick = np.unique(neg[order[idx]])
        yy = y.copy(); yy[pick] = 1
        out.append(sc(yy))
    df = pd.DataFrame(out)
    return {c: [float(df[c].quantile(0.025)), float(df[c].median()), float(df[c].quantile(0.975))] for c in ['skill', 'auc', 'top1']}
for kd in ['random', 'p_champ', 'p_M0_L2', 'mimic_lag1only_rank']:
    o['sim_' + kd] = sim(kd)
json.dump(o, open(os.path.join(CA, 's9_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(o, ensure_ascii=False, indent=1, default=str))
