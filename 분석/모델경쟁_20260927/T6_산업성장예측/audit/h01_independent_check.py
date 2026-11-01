# -*- coding: utf-8 -*-
"""3단계 독립 대조: (1) y를 원자료에서 직접 재계산, (2) MAE·skill·CI 독립 재계산(다른 시드),
(3) 절단 검사: 원점 T마다 조사자료를 T-1까지만 로드한 패널로 다시 예측 → holdout 예측과 동일해야 함(미래 정보 미사용 증명).
결과는 audit/h01_result.json. 모형·설정은 final_spec.json 그대로 사용(변경 없음)."""
import os, sys, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import final_run as fr

spec = fr.load_spec()
ms = fr.model_spec(spec)
hp = pd.read_csv(os.path.join(HERE, 'holdout_preds.csv'), dtype={'ksic': str})
out = {}

# (1) y 재계산: 11차 출하액(T04) ln(t/t-1)
k = pd.read_csv(f'{fr.D}/작업데이터/kosis_광업제조업_2008_2024.csv', dtype={'세세분류코드': str})
s = k[(k.기준 == '11차') & (k.항목코드 == 'T04')].copy()
s['값'] = pd.to_numeric(s['값'], errors='coerce')
p = s.pivot_table(index='세세분류코드', columns='연도', values='값', aggfunc='first')
rows = []
for T in fr.HOLDOUT_YEARS:
    for c in hp[hp.year == T].ksic:
        a, b = p.loc[c, T], p.loc[c, T - 1]
        rows.append((c, T, np.log(a / b)))
yy = pd.DataFrame(rows, columns=['ksic', 'year', 'y_indep'])
m = hp.merge(yy, on=['ksic', 'year'])
out['y_recompute_maxabs_diff'] = float((m.y - m.y_indep).abs().max())
k50 = pd.read_csv(f'{fr.D}/작업데이터/kosis_광업제조업_50개산업.csv', dtype={'세세분류코드': str})
out['n_rows'] = int(len(hp)); out['n_industries'] = int(hp.ksic.nunique())
out['n50_codes'] = int(k50.세세분류코드.nunique())
out['rows_per_year'] = {int(t): int(n) for t, n in hp.groupby('year').size().items()}

# (2) 지표 독립 재계산
e_m, e_b = (hp.y - hp.pred).abs(), (hp.y - 0.0).abs()
out['MAE_model'] = float(e_m.mean()); out['MAE_zero'] = float(e_b.mean())
out['skill'] = 1 - out['MAE_model'] / out['MAE_zero']
out['baseline_pred_all_zero'] = bool((hp.pred_baseline == 0).all())
out['RMSE_model'] = float(np.sqrt(((hp.y - hp.pred) ** 2).mean()))
out['DIR_model'] = float((np.sign(hp.pred) == np.sign(hp.y)).mean())
rng = np.random.default_rng(12345)
g = hp.assign(am=e_m, ab=e_b).groupby('ksic')[['am', 'ab']].sum()
A, Bv = g.am.values, g.ab.values
sk = [1 - A[i].sum() / Bv[i].sum() for i in (rng.integers(0, len(A), len(A)) for _ in range(10000))]
out['skill_ci_industry_indep'] = [float(np.percentile(sk, 2.5)), float(np.percentile(sk, 97.5))]
out['p_skill_le_0_industry'] = float(np.mean(np.array(sk) <= 0))
# 산업×연도 2원 군집 없이 행 단위(참고)
e = (e_b - e_m).values
# 50개 중 모형이 더 나은 산업 수
gi = hp.assign(am=e_m, ab=e_b).groupby('ksic')[['am', 'ab']].mean()
out['industries_model_better'] = int((gi.am < gi.ab).sum())
# 상위 10% 예측 산업의 실제 순위(상대 성장 식별력)
hits, tot = 0, 0
for T, gg in hp.groupby('year'):
    q_p = gg.pred.rank(ascending=False) <= 5
    q_y = gg.y.rank(ascending=False) <= 10
    hits += int((q_p & q_y).sum()); tot += int(q_p.sum())
out['top5_pred_in_actual_top10'] = f'{hits}/{tot}'
from scipy.stats import spearmanr
out['spearman_by_year'] = {int(T): float(spearmanr(gg.pred, gg.y).correlation) for T, gg in hp.groupby('year')}
dm = hp.y - hp.groupby('year').y.transform('median')
out['MAE_model_vs_demeaned_y'] = float((dm - hp.pred).abs().mean())
out['MAE_zero_vs_demeaned_y'] = float(dm.abs().mean())

# (3) 절단 검사
diffs = {}
for T in fr.HOLDOUT_YEARS:
    F = fr.build_panel(T - 1, T, extra=ms.get('extra', True))
    assert F[F.y.notna()].year.max() <= T - 1
    pr = fr.rolling(F, [T], ms, require_y=False)
    q = hp[hp.year == T][['ksic', 'pred']].merge(pr[['ksic', 'pred']].rename(columns={'pred': 'p_trunc'}), on='ksic')
    diffs[int(T)] = dict(n_matched=int(len(q)), max_abs_diff=float((q.pred - q.p_trunc).abs().max()),
                         n_train_trunc=int(pr.n_train.iloc[0]))
out['truncation_test'] = diffs
json.dump(out, open(os.path.join(HERE, 'audit', 'h01_result.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
