# (b) 지역 분포(공식 133003 대조), 부처 계열 단절, perform_grp 오표기 연도별 규모
import pandas as pd, numpy as np, os, json
EXT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared\ext"
OUT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C02\C02-T1"
d = pd.read_parquet(os.path.join(EXT, 'ntis_project_raw.parquet'),
                    columns=['year', 'region_cd', 'ministry', 'secret_flag', 'gov_funds', 'perform_agent_cd', 'perform_grp', 'corp_rgst_no'])
off = pd.read_csv(os.path.join(OUT, 'official_enara_1330_long.csv'))
off = off[(off.stts_cd == 133003) & (off.kind == 'amount')].copy()
off['value'] = pd.to_numeric(off.value, errors='coerce')
op = off.pivot(index='year', columns='item', values='value')
ns = d[~d.secret_flag].copy()
rc = ns.region_cd.astype(str).str.strip().str.zfill(2)
ns['rg'] = np.select([rc.isin(['01', '04', '08']), rc.eq('06'), rc.isin(['99', '19', 'NA', '<NA>', 'nan', '00'])], ['수도권', '대전', '미상/해외'], '지방')
r = ns.groupby(['year', 'rg']).gov_funds.sum().unstack().fillna(0)
rows = []
for y in r.index:
    known = r.loc[y, ['수도권', '대전', '지방']].sum()
    for k, ok in [('수도권', '수도권'), ('대전', '대전광역시'), ('지방', '지방')]:
        o = op.loc[y] if y in op.index else None
        osh = o[ok] / o['계'] if o is not None and pd.notna(o.get('계')) else np.nan
        rows.append(dict(year=y, region=k, col_share=r.loc[y, k] / known, off_share=osh,
                         diff_pp=100 * (r.loc[y, k] / known - osh) if pd.notna(osh) else np.nan,
                         unknown_share=r.loc[y, '미상/해외'] / r.loc[y].sum() if '미상/해외' in r.columns else 0))
R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, '_tab_b_region.csv'), index=False, encoding='utf-8-sig')
print(R[R.year.isin([2008, 2012, 2015, 2019, 2022, 2024])].round(4).to_string())
print('region max |diff|', R.diff_pp.abs().max())
print('zero-pad 비일관 region_cd 행 수', int((d.region_cd.astype(str).str.len() == 1).sum()))
# 부처 계열
fam = {'과학기술부': 'SCI', '교육과학기술부': 'SCI+EDU', '미래창조과학부': 'SCI+ICT', '과학기술정보통신부': 'SCI+ICT',
       '교육인적자원부': 'EDU', '교육부': 'EDU', '산업자원부': 'IND', '지식경제부': 'IND+ICT', '산업통상자원부': 'IND', '정보통신부': 'ICT',
       '방송통신위원회': 'ICT', '중소기업청': 'SME', '중소벤처기업부': 'SME', '건설교통부': 'LAND', '국토해양부': 'LAND+SEA', '국토교통부': 'LAND',
       '해양수산부': 'SEA', '농림부': 'AGRI', '농림수산식품부': 'AGRI+SEA', '농림축산식품부': 'AGRI', '보건복지부': 'HEALTH', '보건복지가족부': 'HEALTH',
       '국무조정실': 'PM', '국무총리실': 'PM'}
ns['fam'] = ns.ministry.map(fam).fillna('OTHER')
big = {'SCI', 'SCI+EDU', 'SCI+ICT', 'EDU', 'IND', 'IND+ICT', 'ICT', 'SME'}
ns['super'] = np.where(ns.fam.isin(big), 'SCI_EDU_IND_ICT_SME', np.where(ns.fam.isin({'LAND', 'LAND+SEA', 'SEA', 'AGRI', 'AGRI+SEA'}), 'LAND_SEA_AGRI', ns.fam))
s = ns.groupby(['year', 'super']).gov_funds.sum().unstack().fillna(0) / 1e12
chg = s.pct_change()
print((s.round(2)).to_string())
jumps = []
for c in s.columns:
    for y in s.index[1:]:
        if s.loc[y - 1, c] > 0.1 and abs(chg.loc[y, c]) > 0.30:
            jumps.append(dict(super_family=c, year=int(y), prev=float(round(s.loc[y - 1, c], 3)), cur=float(round(s.loc[y, c], 3)), pct=float(round(chg.loc[y, c], 3))))
print('jumps >30%', jumps)
nmin = d.groupby('year').ministry.nunique()
# perform_grp 오표기 규모
pg = d.groupby('year').apply(lambda g: pd.Series({
    'n_code06_labeled_중견기업(실제 정부부처)': int((g.perform_agent_cd == '06').sum()),
    'n_code08_labeled_기타(실제 중견기업)': int((g.perform_agent_cd == '08').sum()),
    'gov_code08_eok': g.loc[g.perform_agent_cd == '08', 'gov_funds'].sum() / 1e8,
    'bz_code08': g.loc[g.perform_agent_cd == '08', 'corp_rgst_no'].nunique(),
    'bz_code06': g.loc[g.perform_agent_cd == '06', 'corp_rgst_no'].nunique()}))
print(pg.to_string())
pg.to_csv(os.path.join(OUT, '_tab_b_pgrp_mislabel.csv'), encoding='utf-8-sig')
json.dump({'jumps': jumps, 'n_ministry_by_year': {int(k): int(v) for k, v in nmin.items()}}, open(os.path.join(OUT, '_tab_b_ministry.json'), 'w'), ensure_ascii=False, default=int)
