# 트렌드 산출물 초안(검증 기준시점 C=2014-01-01 모형): KSIC별 예측 포기위험 추이 2012–2018, 실현율(검증기 이하만)
import numpy as np, pandas as pd, json
from scipy.stats import spearmanr, pearsonr
from run import labels
import t4lib as T
d = pd.read_parquet('final_models/ens_20140101_20120101_20180701.parquet')
REAL_MAX = '2015-07-01'   # 실현 레이블은 L+3y <= 2018-07 인 기준일까지만(🔒 2019+ 결과 미조회)
ys = []
for L in sorted(d.L.unique()):
    s = d[d.L == L]
    ys.append(labels(L, s.pid.values, None) if L <= REAL_MAX else np.full(len(s), np.nan))
d['y'] = np.concatenate(ys)
nm = pd.read_csv(r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\공모전데이터\작업데이터\kosis_광업제조업_50개산업.csv', dtype={'세세분류코드': str})
nm = nm.drop_duplicates('세세분류코드').set_index('세세분류코드').산업명
g = d.groupby(['ksic', 'L']).agg(n_alive=('pid', 'size'), pred_risk=('p', 'mean'), exp_lapses=('p', 'sum'), real_rate=('y', 'mean')).reset_index()
g['industry'] = g.ksic.map(nm)
g['year'] = g.L.str[:4].astype(int) + (g.L.str[5:7] == '07') * 0.5
g.to_csv('trend/ksic_risk_trend_2012_2018.csv', index=False, encoding='utf-8-sig')
# 전체 추이
tot = d.groupby('L').agg(n=('pid', 'size'), pred=('p', 'mean'), real=('y', 'mean')).reset_index()
print(tot.round(4).to_string())
tot.to_csv('trend/total_risk_trend.csv', index=False, encoding='utf-8-sig')
# 산업 단위 타당도: 검증기(2014-01..2015-07) KSIC 평균 예측 vs 실현 포기율 (n=50/기준일)
val = {}
for L in ['2012-01-01', '2013-01-01', '2014-01-01', '2014-07-01', '2015-01-01', '2015-07-01']:
    s = g[g.L == L]
    val[L] = dict(spearman=round(spearmanr(s.pred_risk, s.real_rate)[0], 3), pearson=round(pearsonr(s.pred_risk, s.real_rate)[0], 3),
                  mae=round(float(np.mean(np.abs(s.pred_risk - s.real_rate))), 4))
print(json.dumps(val, indent=1))
# 추세 변화(예측위험 2014-01 -> 2018-07) 순위 & 최신(2018-07) 위험 상위 — 초안
p14 = g[g.L == '2014-01-01'].set_index('ksic').pred_risk; p18 = g[g.L == '2018-07-01'].set_index('ksic')
top = pd.DataFrame({'industry': p18.industry, 'n_alive_2018H2': p18.n_alive, 'pred_risk_2018H2': p18.pred_risk.round(4),
                    'pred_risk_2014H1': p14.round(4), 'delta_2014_2018': (p18.pred_risk - p14).round(4),
                    'exp_lapses_3y': p18.exp_lapses.round(0)}).sort_values('pred_risk_2018H2', ascending=False)
top.to_csv('trend/top_risk_draft_L2018H2.csv', encoding='utf-8-sig')
print(top.head(10).to_string()); print(top.sort_values('delta_2014_2018', ascending=False).head(10).to_string())
json.dump(val, open('trend/industry_level_validity.json', 'w'), indent=1)
# 그림
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'; plt.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].plot(tot.L.str[:7], tot.pred, 'o-', label='예측 평균위험(C=2014 모형)')
ax[0].plot(tot.L.str[:7], tot.real, 's--', label='실현 3년 불납률(≤2015-07)')
ax[0].set_title('KR 등록특허 3년 내 불납소멸 위험 추이'); ax[0].legend(); ax[0].tick_params(axis='x', rotation=60)
for k in top.head(5).index:
    s = g[g.ksic == k]; ax[1].plot(s.L.str[:7], s.pred_risk, '-', label=f'{k} {str(nm.get(k, ""))[:12]}')
ax[1].set_title('2018H2 예측위험 상위 5개 산업 추이'); ax[1].legend(fontsize=8); ax[1].tick_params(axis='x', rotation=60)
plt.tight_layout(); plt.savefig('trend/trend_draft.png', dpi=120)
