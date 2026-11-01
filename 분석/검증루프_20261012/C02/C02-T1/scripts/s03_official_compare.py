# (a) 공식 대조: 수집본 vs e-나라지표 1330(조사·분석 집행액) + 보도·KIST 보고서 과제 수
import pandas as pd, numpy as np, os, json
EXT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared\ext"
OUT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C02\C02-T1"
off = pd.read_csv(os.path.join(OUT, 'official_enara_1330_long.csv'))
off = off[off.kind.isin(['amount', 'value'])].copy()
off['value'] = pd.to_numeric(off.value, errors='coerce'); off['year'] = off.year.astype(int)
tot = off[(off.stts_cd == 133002) & (off.item == '계')].set_index('year').value  # 억원
dfn = off[(off.stts_cd == 133004) & (off.item == '국방')].set_index('year').value
bud = off[(off.stts_cd == 133001) & (off.item == 'R&D예산')].set_index('year').value  # 조원
perf = off[off.stts_cd == 133002].pivot(index='year', columns='item', values='value')
# 공식 과제 수 (출처: KIST 융합연구 성과분석 2022·2023 요약표 = 조사·분석 보고서 인용, 뉴시스 2020-06-23, 뉴스토마토, 인사이트 2026)
off_n = {2018: 63697, 2019: 70327, 2020: 73501, 2021: 74745, 2022: 76052, 2023: 71804, 2024: 60696}
off_n_src = {2018: 'KIST2022', 2019: 'KIST2022;newsis', 2020: 'KIST2022;newstomato', 2021: 'KIST2022', 2022: 'KIST2022', 2023: 'KIST2023', 2024: 'insight(조사분석 2024 보도)'}
d = pd.read_parquet(os.path.join(EXT, 'ntis_project_raw.parquet'), columns=['year', 'secret_flag', 'gov_funds', 'total_funds', 'perform_agent'])
g = d.groupby('year').agg(n=('year', 'size'), n_secret=('secret_flag', 'sum'))
g['gov_nonsecret'] = d[~d.secret_flag].groupby('year').gov_funds.sum() / 1e8
g['tot_secret'] = d[d.secret_flag].groupby('year').total_funds.sum() / 1e8
g['official_exec'] = tot
g['official_defense'] = dfn
g['official_budget_jo'] = bud
g['official_n'] = pd.Series(off_n); g['official_n_src'] = pd.Series(off_n_src)
g['ratio_n'] = g.n / g.official_n
g['ratio_gov'] = g.gov_nonsecret / g.official_exec
g['ratio_gov_plus_secret'] = (g.gov_nonsecret + g.tot_secret) / g.official_exec
g['ratio_gov_vs_off_minus_defense'] = g.gov_nonsecret / (g.official_exec - g.official_defense)
g['secret_share_n'] = g.n_secret / g.n
g['secret_share_funds_vs_off'] = g.tot_secret / g.official_exec
g['defense_share_off'] = g.official_defense / g.official_exec
g = g.reset_index()
g.to_csv(os.path.join(OUT, '_tab_a_totals.csv'), index=False, encoding='utf-8-sig')
pd.set_option('display.width', 250)
print(g.round(4).to_string())
# 수행주체 비중 비교 (금액 기준). 수집본: 비보안 정부연구비. 공식: 133002 (보안 포함)
mp = {'국공립연구소': '국공립연구소', '출연연구소': '출연연구소', '대학': '대학', '대기업': '대기업', '중견기업': '중견기업',
      '중소기업': '중소기업', '정부부처': '정부부처', '기타': '기타'}
c = d[~d.secret_flag].groupby(['year', 'perform_agent']).gov_funds.sum().unstack().fillna(0) / 1e8
rows = []
for y in range(2007, 2025):
    if y not in perf.index: continue
    o = perf.loc[y]; ot = o['계']
    ct = c.loc[y].sum(); st = g.set_index('year').loc[y, 'tot_secret']
    for k in mp:
        ov = o.get(k, np.nan); cv = c.loc[y].get(k, 0.0)
        rows.append(dict(year=y, performer=k, official_eok=ov, collected_eok=cv,
                         off_share=ov / ot, col_share_nonsecret=cv / ct,
                         col_share_with_secret_bucket=cv / (ct + st),
                         diff_pp_nonsecret=100 * (cv / ct - ov / ot),
                         ratio_col_off=cv / ov if ov else np.nan,
                         missing_eok=ov - cv))
P = pd.DataFrame(rows)
P.to_csv(os.path.join(OUT, '_tab_a_performer.csv'), index=False, encoding='utf-8-sig')
print(P[P.year.isin([2010, 2015, 2019, 2022, 2024])].round(3).to_string())
mx = P.loc[P.diff_pp_nonsecret.abs().idxmax()]
print('max |diff pp|', mx.to_dict())
summ = P.groupby('performer').agg(diff_pp_mean=('diff_pp_nonsecret', 'mean'), diff_pp_max=('diff_pp_nonsecret', lambda s: s.abs().max()),
                                  ratio_mean=('ratio_col_off', 'mean'), missing_eok_mean=('missing_eok', 'mean'))
print(summ.round(3).to_string())
# 누락액(공식−수집)의 주체별 배분 = 보안과제 금액의 추정 주체 구성
P['missing_pos'] = P.missing_eok.clip(lower=0)
mshare = P[P.year >= 2013].groupby('performer').missing_eok.sum() / P[P.year >= 2013].missing_eok.sum()
print('missing (official-collected) composition 2013-2024'); print(mshare.round(3).to_string())
json.dump({'missing_composition_2013_2024': mshare.round(4).to_dict()}, open(os.path.join(OUT, '_tab_a_missing_comp.json'), 'w'), ensure_ascii=False)
