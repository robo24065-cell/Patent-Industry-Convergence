# 감사 6: holdout·future 모드 코드 경로 모의 실행(🔒 홀드아웃 자료 미사용).
# <=2019 10차 자료로 가짜 11차(2016-2019, 한 코드 수준 1.1배, 한 코드 재코드)를 만들고 ov=2016으로 연결.
# holdout 모의: 목표 2017-2019 / future 모의: 조사 <=2018로 2019(lag1)·2020(lag2) 예측. 결과는 audit/mock 에만 저장.
import sys, os, json
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, H); os.chdir(H)
import pandas as pd, numpy as np
import final_run as fr
k = pd.read_csv(f'{fr.D}/작업데이터/kosis_광업제조업_2008_2024.csv', dtype={'세세분류코드': str})
k = k[(k.연도 <= 2019) & (k.기준 == '10차')]
k50 = sorted(pd.read_csv(f'{fr.D}/작업데이터/kosis_광업제조업_50개산업.csv', dtype={'세세분류코드': str}).세세분류코드.unique())
f11 = k[k.연도 >= 2016].copy(); f11['기준'] = '11차'
f11.loc[f11.세세분류코드 == '26211', '값'] *= 1.1
f11.loc[f11.세세분류코드 == '29120', '세세분류코드'] = '2919Z'
kk = pd.concat([k[k.연도 <= 2016], f11])
spec = fr.load_spec()
out = os.path.join(H, 'audit/mock')
fr.mode_holdout(spec, survey_maxy=2019, years=[2017, 2018, 2019], k_all=kk, k50_codes=k50, ov=2016, out_dir=out)
r = json.load(open(os.path.join(out, 'holdout_result.json'), encoding='utf-8'))
print('mock holdout by_year', {a: (b['n'], round(b['skill'], 4), b['n_train']) for a, b in r['by_year'].items()})
print('label years', r['label_years_available_before_first_origin'], 'patent', round(r['patent_ablation']['skill_with_patent'], 4))
# 같은 연도(2017-2019)를 실제 검증 방식(dryrun 패널)으로 평가한 값과 비교 → 연결 로직이 결과를 바꾸지 않아야(수준 1.1배·재코드 영향만)
d = pd.read_csv('dryrun_preds.csv', dtype={'ksic': str}); d = d[d.year >= 2017]
m = pd.read_csv(os.path.join(out, 'holdout_preds.csv'), dtype={'ksic': str}).merge(d, on=['ksic', 'year'], suffixes=('_mock', '_dry'))
print('mock vs dryrun (2017-2019) n', len(m), 'y 동일', np.allclose(m.y_mock, m.y_dry), 'pred 상관', round(np.corrcoef(m.pred_mock, m.pred_dry)[0, 1], 4), 'max|pred차|', round((m.pred_mock - m.pred_dry).abs().max(), 4))
fr.mode_future(spec, survey_maxy=2018, targets=[(2019, 1), (2020, 2)], k_all=kk, k50_codes=k50, ov=2016, out_dir=out)
f = pd.read_csv(os.path.join(out, 'future_forecast.csv'), dtype={'ksic': str})
print(f.groupby('target_year').agg(n=('ksic', 'size'), nn=('pred_rel_growth', lambda s: s.notna().sum()), sd=('pred_rel_growth', 'std'), ntr=('n_train', 'first')).to_string())
print('patent share (2019)', f[f.target_year == 2019].patent_contrib_share.describe().round(3).to_dict())
