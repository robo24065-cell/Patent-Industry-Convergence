# C03-T1 s8 (사후 진단, 판정 아님): 등록 경로 레이블(y_reg)만으로 본 개발·최종 skill — 레이블 구성(공개 전용 양성 비중) 차이가 개발→최종 +0.126에 얽혀 있는지
import sys, json, pathlib
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전")
R3 = ROOT/'분석/모델경쟁_20260927/T1_산업대기실/R3'; WC = ROOT/'분석/검증루프_20261012/C03/C03-T1/cache'
BCOLS = ['logit_stack_y', 'logit_single_y', 'xgb_y', 'logit_stack_y_u', 'logit_single_y_u', 'xgb_y_u']
sk = lambda m, b: (m - b) / (1 - b); rank = lambda s: pd.Series(s).rank(pct=True).values
out = {}
ev = pd.read_parquet(WC/'eval_o2019.parquet'); yr = ev.yA_reg.values
m = roc_auc_score(yr, ev.s.values); b = max(roc_auc_score(yr, ev['b_' + c].values) for c in BCOLS)
out['final'] = dict(auc_reg=m, strongest_reg=b, skill_reg=sk(m, b), pos_reg=int(yr.sum()))
for c in (2008, 2009, 2010, 2011, 2012):
    o = pd.read_parquet(R3/f'runs/FROZEN_dev/o{c}.parquet'); bs = pd.read_parquet(R3/f'cache/o{c}/baseline_scores.parquet')
    kp = ~o.prior_any.values; o = o[kp]; bs = bs.loc[kp]; s = 0.9 * rank(o.px.values) + 0.1 * rank(o.pl.values); y = o.y_reg.values
    m = roc_auc_score(y, s); b = max(roc_auc_score(y, bs[cc].values) for cc in BCOLS); out[c] = dict(auc_reg=m, strongest_reg=b, skill_reg=sk(m, b), pos_reg=int(y.sum()))
out['dev_mean_skill_reg'] = float(np.mean([out[c]['skill_reg'] for c in (2008, 2009, 2010, 2011, 2012)])); out['final_minus_dev_reg'] = out['final']['skill_reg'] - out['dev_mean_skill_reg']
json.dump(out, open(WC/'s8_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float); print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
bp = pd.read_parquet(WC/'blank_pairs_2009_2024.parquet', columns=['rgst_date', 'ay']); print('blank rgst_date min', bp.rgst_date.min(), 'n<=20191231', int((bp.rgst_date <= '20191231').sum()), 'n<20231101', int((bp.rgst_date < '20231101').sum()))
