# v0: 고정 기준선 선택 {0 성장, 전년 성장률, 전년 전 산업 평균(50개 / 488개)} — 검증 MAE 최소 규칙을 고정
import pandas as pd, numpy as np, json
from t6lib import *
F = load_panel(with_patent=False)
te = F[(F.year.isin(VAL_YEARS)) & (F.in50 == 1) & F.y.notna()].copy()
print('검증 표본(산업-연도):', len(te), '산업 수:', te.ksic.nunique())
rules = {'zero': np.zeros(len(te)), 'last_growth': te.g1.fillna(0).values,
         'cm50_last': te.cm_501.values, 'cmall_last': te.cm_all1.values, 'cmedall_last': te.cmed_all1.values}
res = {}
for k, p in rules.items():
    d = te[['ksic', 'year', 'y']].copy(); d['pred'] = p
    res[k] = metrics(d.y, d.pred)
    print(k, {a: round(b, 4) for a, b in res[k].items()})
    print('  연도별 MAE', by_year(d).round(4).to_dict())
spec = ['zero', 'last_growth', 'cm50_last', 'cmall_last']  # 명세된 3규칙(전 산업=50개/488개 두 해석)
best = min(spec, key=lambda k: res[k]['MAE'])
print('선택 기준선:', best)
d = te[['ksic', 'year', 'y']].copy(); d['pred'] = rules[best]
json.dump({'baseline_rule': best, 'MAE': res[best]['MAE'], 'all': res}, open('baseline.json', 'w'), ensure_ascii=False, indent=1)
log_version('v0', f'고정 기준선: {best}', '명세 규칙 중 검증 MAE 최소를 고정', d, res[best]['MAE'], 'Y',
            note='후보 MAE: ' + ', '.join(f"{k}={res[k]['MAE']:.4f}" for k in rules))
d.to_parquet('preds/baseline.parquet', index=False)
