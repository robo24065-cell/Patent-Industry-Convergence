# 특허 피처: (ksic, 목표연도 T). 출원 기반 피처는 출원연도 T-L (L=1: T-1년 출원, L=2: T-2년 출원 = 공개지연 보수적) 이하만 사용.
# 등록원부 사건(소멸·이전·신규등록)은 사건 발생 즉시 공시 → T-1년 사건까지 사용.
import pandas as pd, numpy as np
p = pd.read_parquet('cache/pat_ksic_cc_year.parquet')
p = p[p.year <= 2018]  # 1단계: 피처 연도 상한
cnt = p.pivot_table(index=['ksic', 'year'], columns='cc', values='n_app', aggfunc='sum').fillna(0)
cnt.columns = ['n_' + c.lower() for c in cnt.columns]
cnt['n_tot'] = cnt.sum(axis=1)
fp = p[p.cc == 'KR'].set_index(['ksic', 'year'])[['n_foreignprio', 'n_applicant', 'n_app']]
fpu = p[p.cc == 'US'].set_index(['ksic', 'year'])[['n_foreignprio', 'n_app']]
cnt = cnt.join(fp.rename(columns={'n_foreignprio': 'kr_fp', 'n_applicant': 'kr_appl', 'n_app': '_kr'}))
cnt = cnt.join(fpu.rename(columns={'n_foreignprio': 'us_fp', 'n_app': '_us'}))
cnt = cnt.reset_index().sort_values(['ksic', 'year'])
c = cnt
g = c.groupby('ksic')
lg = lambda s: np.log1p(s)
base = pd.DataFrame({'ksic': c.ksic, 'year': c.year})
for cc in ['tot', 'kr', 'cn', 'us', 'jp', 'ep']:
    x = lg(c['n_' + cc])
    base[f'pg_{cc}'] = x - x.groupby(c.ksic).shift(1)
    base[f'pg3_{cc}'] = (x - x.groupby(c.ksic).shift(3)) / 3
base['cn_share'] = c.n_cn / c.n_tot
base['kr_share'] = c.n_kr / c.n_tot
base['foreign_share'] = (c.n_us + c.n_jp + c.n_ep) / c.n_tot
for s in ['cn_share', 'kr_share', 'foreign_share']:
    base[s + '_d'] = base[s] - base.groupby(c.ksic)[s].shift(1)
for s_ in ['cn_share', 'kr_share', 'foreign_share']:
    base[s_ + '_d3'] = base[s_] - base.groupby(c.ksic)[s_].shift(3)
# KR 대비 중국 출원 상대 모멘텀(경쟁 압력), 3년
base['cn_vs_kr3'] = base['pg3_cn'] - base['pg3_kr']
base['kr_fp_share'] = c.kr_fp / c._kr
base['us_fp_share'] = c.us_fp / c._us
base['kr_fp_share_d'] = base.kr_fp_share - base.groupby(c.ksic).kr_fp_share.shift(1)
base['pg_kr_appl'] = lg(c.kr_appl) - lg(c.kr_appl).groupby(c.ksic).shift(1)
base['l_ntot'] = lg(c.n_tot)
# 50개 산업 전체 대비 비중 변화(공통 연도 충격 제거)
tot_y = c.groupby('year').n_tot.transform('sum')
base['share50'] = np.log(c.n_tot / tot_y)
base['share50_d'] = base.share50 - base.groupby(c.ksic).share50.shift(1)
# 상대 성장: 연도별 단면 평균 제거
for v in ['pg_tot', 'pg3_tot', 'pg_kr', 'pg_cn', 'pg_us', 'pg3_kr', 'pg3_cn', 'cn_share_d3', 'cn_vs_kr3']:
    base[v + '_rel'] = base[v] - base.groupby('year')[v].transform('mean')

# 등록원부
r = pd.read_parquet('cache/kr_registry_ksic_year.parquet')
r = r[r.year <= 2018].sort_values(['ksic', 'year'])
r['ksic'] = r.ksic.astype(str)
full = pd.MultiIndex.from_product([r.ksic.unique(), range(2000, 2019)], names=['ksic', 'year']).to_frame(index=False)
r = full.merge(r, on=['ksic', 'year'], how='left').fillna(0)
gr = r.groupby('ksic')
r['stock'] = gr.kr_new_rgst.cumsum() - gr.kr_lapse.cumsum()  # 2000년 이후 등록분 존속 스톡(근사)
r['stock_prev'] = r.groupby('ksic').stock.shift(1)
reg = pd.DataFrame({'ksic': r.ksic, 'year': r.year})
reg['kr_lapse_rate'] = r.kr_lapse / r.stock_prev.where(r.stock_prev > 50)
reg['kr_transfer_rate'] = r.kr_transfer / r.stock_prev.where(r.stock_prev > 50)
reg['kr_rgst_g'] = np.log1p(r.kr_new_rgst) - np.log1p(r.kr_new_rgst).groupby(r.ksic).shift(1)
reg['kr_lapse_rate_d'] = reg.kr_lapse_rate - reg.groupby('ksic').kr_lapse_rate.shift(1)
reg['kr_transfer_rate_d'] = reg.kr_transfer_rate - reg.groupby('ksic').kr_transfer_rate.shift(1)
for v in ['kr_lapse_rate', 'kr_transfer_rate']:
    reg[v + '_rel'] = reg[v] - reg.groupby('year')[v].transform('mean')

out = []
for L in [1, 2]:
    b = base.copy()
    b['year'] = b.year + L  # 출원연도 T-L → 목표연도 T
    b = b.rename(columns={col: f'{col}_L{L}' for col in b.columns if col not in ('ksic', 'year')})
    out.append(b.set_index(['ksic', 'year']))
rg = reg.copy()
rg['year'] = rg.year + 1
rg = rg.set_index(['ksic', 'year'])
P = pd.concat(out + [rg], axis=1).reset_index()
P = P[P.year.between(2009, 2019)]
P.to_parquet('cache/patent_feats.parquet', index=False)
print(P.shape, P.ksic.nunique())
print(P[P.year.between(2012, 2019)].describe().T[['count', 'mean', 'std', 'min', 'max']].round(3).to_string())
