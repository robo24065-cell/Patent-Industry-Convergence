# 진단용 단변량 스크린: 50개가 아닌 438개 산업에서(단위 분리) 롤링 LAD(연도중앙값 제거 타깃, 무절편) → MAE 개선 여부.
# 선택 편향을 줄이려 평가 단위를 50개와 분리. 결과: diag/screen_univariate.csv
import pandas as pd, numpy as np, os, warnings
import statsmodels.api as sm
warnings.filterwarnings('ignore')
from t6lib import *
from versions import KINT, IDX, PAT1, PAT2
os.makedirs('diag', exist_ok=True)
F = load_panel()
F = F[F.year != 2010]
F['ydm'] = F.y - F.groupby('year').y.transform('median')
def run(feat, units):
    errs, errs0 = [], []
    for T in VAL_YEARS:
        tr = F[(F.year < T) & F.y.notna()]
        if feat.startswith(('pg', 'kr_', 'cn_', 'foreign', 'us_fp', 'share50', 'l_ntot')):
            tr = tr[tr.in50 == 1]
        te = F[(F.year == T) & F.y.notna() & (F.in50 == (1 if units == 50 else 0))]
        x = tr[feat].astype(float); lo, hi = x.quantile(.01), x.quantile(.99)
        med = x.median(); sd = x.clip(lo, hi).std() or 1
        xt = ((x.clip(lo, hi).fillna(med)) - med) / sd
        xe = ((te[feat].astype(float).clip(lo, hi).fillna(med)) - med) / sd
        try:
            b = sm.QuantReg(tr.ydm.clip(-1, 1).values, xt.values[:, None]).fit(q=.5).params[0]
        except Exception:
            b = 0
        p = b * xe.values
        errs.append(np.abs(te.y.values - p)); errs0.append(np.abs(te.y.values))
    e, e0 = np.concatenate(errs), np.concatenate(errs0)
    return 1 - e.mean() / e0.mean()
rows = []
for f in KINT + IDX:
    rows.append(dict(feat=f, skill_non50=run(f, 0), skill_50=run(f, 50)))
for f in PAT1 + PAT2:
    rows.append(dict(feat=f, skill_non50=np.nan, skill_50=run(f, 50)))
R = pd.DataFrame(rows).sort_values('skill_non50', ascending=False)
R.to_csv('diag/screen_univariate.csv', index=False, encoding='utf-8-sig')
print(R.round(4).to_string())
