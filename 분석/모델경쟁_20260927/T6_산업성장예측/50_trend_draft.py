# 트렌드 산출물 초안(검증구간 기준): 산업별 실제 vs 예측(2012-2019), 2019 원점 예측의 피처 기여도(XGB SHAP), 특허 피처 기여 비중(v32).
import pandas as pd, numpy as np, warnings, xgboost as xg
warnings.filterwarnings('ignore')
from t6lib import *
from models import train_rows, dm_target
from versions import COMPACT, SECT, PATC
D = r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/공모전데이터"
nm = pd.read_csv(f'{D}/작업데이터/kosis_광업제조업_50개산업.csv', dtype={'세세분류코드': str})
nm = nm[(nm.기준 == '10차')].drop_duplicates('세세분류코드')[['세세분류코드', '산업명']].rename(columns={'세세분류코드': 'ksic', '산업명': 'name'})
F = load_panel()
p = pd.read_parquet('preds/v31.parquet').rename(columns={'pred': 'pred_v31'})
p = p.merge(nm, on='ksic', how='left')
p['pred_baseline'] = 0.0
p.to_csv('trend_draft_validation.csv', index=False, encoding='utf-8-sig')
P = dict(max_depth=2, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, min_child_weight=30, reg_lambda=10.0,
         objective='reg:absoluteerror', nthread=2, verbosity=0, base_score=0.0)
def contribs(feats, T=2019):
    tr = SECT(F[(F.year < T) & (F.year >= 2009) & F.y.notna()])
    tr = tr[~tr.year.isin((2010, 2011))]
    tr, y = train_rows(tr, ycap=1.0); y = dm_target(tr, y)
    te = F[(F.year == T) & (F.in50 == 1) & F.y.notna()]
    Xtr = tr[feats].astype(float).groupby(tr.year.values).rank(pct=True)
    Xte = te[feats].astype(float).groupby(te.year.values).rank(pct=True)
    C = []
    for s in range(5):
        m = xg.XGBRegressor(n_estimators=300, **P, random_state=s).fit(Xtr, y)
        C.append(m.get_booster().predict(xg.DMatrix(Xte), pred_contribs=True))
    C = np.mean(C, axis=0)
    df = pd.DataFrame(C[:, :-1], columns=feats, index=te.ksic.values)
    return df
c31 = contribs(COMPACT)
c31.to_csv('contrib_2019_v31.csv', encoding='utf-8-sig')
imp = c31.abs().mean().sort_values(ascending=False)
print('v31 2019 원점 평균|기여|:'); print(imp.round(4).to_string())
pf = PATC + ['pg3_tot_rel_L2', 'kr_lapse_rate_rel', 'kr_transfer_rate_rel']
c32 = contribs(COMPACT + pf)
c32.to_csv('contrib_2019_v32_patent.csv', encoding='utf-8-sig')
share = c32[pf].abs().sum(axis=1).sum() / c32.abs().sum(axis=1).sum()
print(f'v32(특허 포함) 2019 원점: 특허 피처 |기여| 비중 = {share:.3f}')
print(c32[pf].abs().mean().sort_values(ascending=False).round(4).to_string())
# 전체 원점(2013-2019) 특허 기여 비중
sh = []
for T in range(2013, 2020):
    c = contribs(COMPACT + pf, T)
    sh.append((T, round(float(c[pf].abs().sum(axis=1).sum() / c.abs().sum(axis=1).sum()), 3)))
print('원점별 특허 기여 비중', sh)
