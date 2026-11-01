# -*- coding: utf-8 -*-
"""의미 검정 2: (A) ETSI SEP 선언 → 기술셀 성장(T3형) (B) SEP 선언 → KR 등록특허 등록료불납 소멸(T4형) (C) 수출단가 ← 특허 성장. 모두 ≤2016(결과 ≤2018 for B는 T4 개발 규칙 L≤2019 준수)."""
import sys; sys.path.insert(0, r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared\ext')
import pandas as pd, numpy as np, duckdb, sqlite3, statsmodels.formula.api as smf
from scipy.stats import spearmanr
from trade_lib import *
M = EXT.replace(r'\ext', r'\meta\*.parquet')
out = []
# ---------- (A) SEP → 기술셀 성장
lk = pd.read_parquet(EXT + r'\trade_etsi_sep_link.parquet')
con = duckdb.connect(); con.sql("SET memory_limit='3GB'; SET threads=2")
cell = con.sql(f"""select ksic, split_part(main_ipc,'/',1) as grp, appl_year::INT y, count(distinct cc||appl_no) n
  from read_parquet('{M}') where appl_year between 1998 and 2016 and main_ipc is not null group by 1,2,3""").df()
lk['grp'] = lk.main_ipc.str.split('/').str[0]; lk['dy'] = np.maximum(lk.first_decl.dt.year, lk.first_refl.dt.year.where(lk.first_decl.dt.year >= 2011, 0))  # 2011 이관 이전분은 서명일, 이후는 반영일
CP = cell.pivot_table(index=['ksic','grp'], columns='y', values='n', aggfunc='sum').fillna(0)
rows = []
for t in range(2006, 2014):
    s = lk[(lk.dy <= t) & (lk.appl_year <= t)].drop_duplicates(['cc','appl_no','ksic']).groupby(['ksic','grp']).size()
    stock = CP.reindex(columns=range(1998, t+1)).sum(axis=1)
    r = pd.DataFrame({'sep': s.reindex(CP.index).fillna(0), 'stock': stock,
        'fut': np.log1p(CP.reindex(columns=range(t+1, t+4)).sum(axis=1)) - np.log1p(CP.reindex(columns=range(t-2, t+1)).sum(axis=1)),
        'past': np.log1p(CP.reindex(columns=range(t-2, t+1)).sum(axis=1)) - np.log1p(CP.reindex(columns=range(t-5, t-2)).sum(axis=1))})
    r = r[r.stock >= 30]; r['t'] = t; r['sep_any'] = (r.sep > 0).astype(int); r['sep_share'] = r.sep / r.stock; r['lsize'] = np.log(r.stock)
    rows.append(r.reset_index())
A = pd.concat(rows)
for scope, d in [('전체셀', A), ('SEP 보유 KSIC 6개', A[A.ksic.isin(['26429','26422','26529','27212','27211','26519'])])]:
    m = smf.ols('fut ~ sep_any + np.log1p(sep) + past + lsize + C(t) + C(ksic)', d).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d.ksic + d.grp)[0]})
    rs = [spearmanr(smf.ols('np.log1p(sep) ~ past + lsize', g).fit().resid, smf.ols('fut ~ past + lsize', g).fit().resid).statistic for _, g in d.groupby('t')]
    out.append(dict(test=f'(A) SEP선언 누적(≤t) → 셀 특허성장 t+1..t+3 [{scope}]', n=len(d), n_cells=d.groupby(['ksic','grp']).ngroups,
        n_cells_sep=d[d.sep > 0].groupby(['ksic','grp']).ngroups,
        coef_sep_any=round(m.params['sep_any'], 4), p_sep_any=round(m.pvalues['sep_any'], 4),
        coef_logsep=round(m.params['np.log1p(sep)'], 4), p_logsep=round(m.pvalues['np.log1p(sep)'], 4),
        partial_rho_mean=round(np.mean(rs), 3), rho_pos_share=round(np.mean(np.array(rs) > 0), 2)))
    print(out[-1])
# ---------- (B) SEP → 등록료불납 소멸 (KR), 기준일 L=2016-01-01, 결과 2016–2018
R = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite'
c = sqlite3.connect(f'file:{R}?mode=ro', uri=True)
p = pd.read_sql("select rgst_no, ksic, applNo, rgstDate from patent", c)
x = pd.read_sql("select rgst_no, min(rgstCsDate) d, rgstCsReason why from right where rgstCsName='소멸등록' group by rgst_no", c)
p = p.merge(x, on='rgst_no', how='left')
L0, L1 = '20160101', '20181231'
p = p[(p.rgstDate < L0) & ~(p.d.notna() & (p.d < L0))].copy()
p['lapse'] = ((p.d >= L0) & (p.d <= L1) & (p.why == '등록료불납')).astype(int)
sepkr = lk[(lk.cc == 'KR') & (lk.first_decl < '2016-01-01') & (lk.first_refl < '2016-01-01')].appl_no.unique()  # 서명일·DB반영일 모두 L 이전
p['sep'] = p.applNo.isin(sepkr).astype(int)
app = con.sql(f"select distinct appl_no, applicant from read_parquet('{M}') where cc='KR'").df().drop_duplicates('appl_no')
p = p.merge(app, left_on='applNo', right_on='appl_no', how='left'); p['ry'] = p.rgstDate.str[:4]
ks6 = ['26429','26422','26529','27212','27211','26519']
q = p[p.ksic.isin(ks6)]
print('B 표본', len(q), 'SEP', q.sep.sum(), '소멸률 SEP', round(q[q.sep == 1].lapse.mean(), 4), '비SEP', round(q[q.sep == 0].lapse.mean(), 4))
m1 = smf.ols('lapse ~ sep + C(ry) + C(ksic)', q).fit(cov_type='HC1')
# 같은 출원인×등록연도 층 안에서 비교 (SEP 보유 출원인만)
q2 = q[q.applicant.isin(q[q.sep == 1].applicant.unique())].copy(); q2['st'] = q2.applicant.astype(str) + q2.ry
q2 = q2[q2.groupby('st').sep.transform('nunique') == 2]
q2['lapse_dm'] = q2.lapse - q2.groupby('st').lapse.transform('mean'); q2['sep_dm'] = q2.sep - q2.groupby('st').sep.transform('mean')
m2 = smf.ols('lapse_dm ~ sep_dm - 1', q2).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(q2.applicant)[0]})
out.append(dict(test='(B) SEP선언(≤2015) → KR 등록특허 등록료불납 소멸 2016–18 [SEP KSIC 6개]', n=len(q), n_sep=int(q.sep.sum()),
    lapse_sep=round(q[q.sep == 1].lapse.mean(), 4), lapse_nonsep=round(q[q.sep == 0].lapse.mean(), 4),
    lpm_coef_ksic_ry_FE=round(m1.params['sep'], 4), p1=round(m1.pvalues['sep'], 4),
    within_applicant_ry_coef=round(m2.params['sep_dm'], 4), p2=round(m2.pvalues['sep_dm'], 4), n_within=len(q2)))
print(out[-1])
# ---------- (C) 수출단가
u = pd.read_csv(EXT + r'\trade_kr_export_hs6_uv_2000_2021.csv', dtype={'hs6': str}); u['hs6'] = u.hs6.str.zfill(6)
u = u[(u.year <= 2016) & (u.net_wgt_kg > 0) & (u.export_usd > 0)].drop_duplicates(['year','hs6'])
Wv = u.pivot_table(index='hs6', columns='year', values='export_usd')
bal = Wv.index[Wv.reindex(columns=range(2000, 2017)).notna().all(axis=1)]
mp = pd.read_csv(EXT.replace(r'모델경쟁_20260927\_shared\ext', r'fusion\ksic_hs6_map.csv'), dtype=str, encoding='utf-8-sig'); mp['hs6'] = mp.hs6.str.zfill(6)
j = mp.merge(pd.DataFrame({'hs6': bal}), on='hs6').merge(u, on='hs6')
g = j.groupby(['k5','year'])[['export_usd','net_wgt_kg']].sum()
UV = (g.export_usd / g.net_wgt_kg).unstack()
print('단가 연계: HS6 균형', len(bal), 'KSIC', UV.index.nunique())
for ccs, nm in [(None, '5개국'), (['KR'], 'KR')]:
    o, d = lead_test(UV, patents(ccs), range(2005, 2014), f'(C) KR 수출단가(USD/kg) vs 특허({nm})')
    o['n_hs6_bal'] = len(bal); out.append(o); print(o)
pd.DataFrame(out).to_csv(EXT + r'\trade_meaning_tests2.csv', index=False, encoding='utf-8-sig')
