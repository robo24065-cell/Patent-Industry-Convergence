# v20: 학습 레이블 등록절단 보정(IPW) — 학습 양성의 관측확률 π = P(등록 ≤ 2009-12-31 | 출원일, 외국법인 여부)
#      π는 2000~2002 출원 코호트의 등록지연 분포(2009년 말 기준 관측)로 추정 → 양성 가중 1/π (상한 5)
from data8 import *
t0 = time.time(); tr, te = load19(); base = json.load(open(T1/'baseline_v0.json'))['fixed']
D = load(); ins = D['ins']; kidx = D['kidx']; tp = D['fm'].set_index('applicant_cd').tp
ins = ins.assign(fr=ins.applicant_cd.map(tp).eq('외국법인'))
coh = ins[(ins.ay <= 2002) & (ins.rgst_date <= '20091231')].drop_duplicates('rgst_no')
lag = (pd.to_datetime(coh.rgst_date) - pd.to_datetime(coh.appl_date)).dt.days.values
cdf = {}
for g in (False, True):
    l = np.sort(lag[coh.fr.values == g]); cdf[g] = l
    print('foreign' if g else 'domestic', 'median lag days', int(np.median(l)), 'P(lag<=2y)', round((l <= 730).mean(), 3), 'P(lag<=4y)', round((l <= 1461).mean(), 3))
def pi(days, g):
    l = cdf[g]; return np.searchsorted(l, days, side='right') / len(l)
# 학습 패널 양성마다 첫 진입 특허 출원일
w = np.ones(len(tr))
capd = pd.Timestamp('2009-12-31')
for cut in (2002, 2003, 2004):
    fut = ins[(ins.ay > cut) & (ins.ay <= cut + 5) & (ins.rgst_date <= '20091231')]
    fst = fut.groupby(['applicant_cd', 'ksic']).appl_date.min().reset_index(); fst['k'] = fst.ksic.map(kidx)
    idx = np.nonzero((tr.cut.values == cut) & (tr.y.values == 1))[0]
    x = tr.iloc[idx][['firm', 'k']].merge(fst, left_on=['firm', 'k'], right_on=['applicant_cd', 'k'], how='left')
    days = (capd - pd.to_datetime(x.appl_date)).dt.days.values
    fr = x.firm.map(tp).eq('외국법인').values
    p = np.array([pi(dd, g) for dd, g in zip(days, fr)])
    w[idx] = np.minimum(1 / np.maximum(p, 1e-3), 5.0)
print('positive weights mean', w[tr.y.values == 1].mean().round(3), 'foreign pos mean', w[(tr.y.values == 1) & (tr.foreign.values == 1)].mean().round(3))
np.save(CACHE/'v20_w.npy', w)
import sys
V = sys.argv[1] if len(sys.argv) > 1 else 'v20'
COLS = C19 + (['foreign'] if V == 'v21' else [])
s, sx = run_std(tr, te, COLS, w=w); met = metrics(te, s); mx = metrics(te, sx)
print(V, 'xgb only', fmt(mx)); print(V, fmt(met), round(time.time() - t0))
te[['firm', 'k', 'y', 'newc']].assign(s=s, sx=sx).to_parquet(CACHE/f'{V}_scores.parquet', index=False)
if V == 'v21': lb_add('v21', 'v20 + 외국법인 플래그 재투입(IPW 보정 후)', 'IPW로 등록절단 편향을 없애면 외국법인 플래그가 더는 역학습되지 않고 도움이 된다', met, base, '?', f"xgb단독 auc_all={mx['auc_all']:.4f} auc_new={mx['auc_new']:.4f}")
else: lb_add('v20', 'v19 + 학습 레이블 등록절단 IPW(양성 가중 1/π, π=출원일·외국여부별 등록지연 CDF, 2000~02 코호트)', '학습 레이블(등록≤2009)은 늦게 등록되는 진입(특히 외국 PCT)을 놓친다: 역확률 가중이 평가 레이블(완전 관측)과 정렬시킨다', met, base, '?', f"xgb단독 auc_all={mx['auc_all']:.4f} auc_new={mx['auc_new']:.4f}")
