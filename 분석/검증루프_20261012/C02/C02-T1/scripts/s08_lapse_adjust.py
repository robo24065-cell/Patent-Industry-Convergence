# (e)-P9 정부R&D 기원 특허 5년 내 포기율 14.2% 대 21.0% 재계산과 출원인 유형·등록연도·KSIC 표준화, 보안(국방) 기원 누락 민감도
# 원 산출: 모델경쟁_20260927\_shared_report\build_ntis_tables.py 표 3 (같은 로직을 그대로 재현). registry_full.sqlite mode=ro.
import os, re, sys, json, sqlite3, pathlib
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
BASE = pathlib.Path(r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전")
EXTD = BASE / '분석' / '모델경쟁_20260927' / '_shared' / 'ext'
OUT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C02\C02-T1"
con = sqlite3.connect(f"file:{(BASE / '수집파이프라인' / 'registry_full.sqlite').as_posix()}?mode=ro", uri=True)
pt = pd.read_sql("select rgst_no, rgstDate, ksic from patent where substr(rgst_no,1,2)='10'", con)
lap = pd.read_sql("select rgst_no, min(rgstCsDate) d from right where rgstCsName='소멸등록' and rgstCsReason like '%불납%' group by rgst_no", con)
trf = pd.read_sql("select rgst_no, min(ownerRgstCsDate) d from owner where ownerRgstCsReason like '%이전%' and ownerRgstCsDate is not null group by rgst_no", con)
app = pd.read_sql("select rgst_no, seq, applicantCd, applicantName from applicant where substr(rgst_no,1,2)='10'", con)
con.close()
pt['rd'] = pd.to_datetime(pt.rgstDate.astype(str).str[:8], format='%Y%m%d', errors='coerce'); pt = pt[pt.rd.notna()]
pt = pt[(pt.rd.dt.year >= 2007) & (pt.rd.dt.year <= 2016)].drop_duplicates('rgst_no')
for nm, t in (('lap', lap), ('trf', trf)):
    t['d'] = pd.to_datetime(t.d.astype(str).str[:8], format='%Y%m%d', errors='coerce')
    pt = pt.merge(t.rename(columns={'d': nm}), on='rgst_no', how='left')
    pt[f'{nm}5'] = (pt[nm].notna() & (pt[nm] <= pt.rd + pd.DateOffset(years=5))).astype(float)
fl = pd.read_parquet(EXTD / 'ntis_patent_flag.parquet', columns=['rgst_no', 'gov_rnd', 'has_univ', 'has_gri', 'has_firm']).dropna(subset=['rgst_no'])
fl['rgst_no'] = fl.rgst_no.astype(str); fl = fl.groupby('rgst_no')[['gov_rnd', 'has_univ', 'has_gri', 'has_firm']].max().reset_index()
pt['rgst_no'] = pt.rgst_no.astype(str); pt = pt.merge(fl, on='rgst_no', how='left')
for c in ('gov_rnd', 'has_univ', 'has_gri', 'has_firm'):
    pt[c] = pt[c].fillna(False).astype(bool)
res = {}
res['recompute'] = dict(n=len(pt), lapse5_all=pt.lap5.mean(), lapse5_gov=pt[pt.gov_rnd].lap5.mean(), lapse5_nongov=pt[~pt.gov_rnd].lap5.mean(),
                        n_gov=int(pt.gov_rnd.sum()), n_nongov=int((~pt.gov_rnd).sum()),
                        transfer5_gov=pt[pt.gov_rnd].trf5.mean(), transfer5_nongov=pt[~pt.gov_rnd].trf5.mean())
print(res['recompute'])
# 출원인 유형(대표=seq 최소 출원인)
app['seq'] = pd.to_numeric(app.seq, errors='coerce')
app = app.sort_values(['rgst_no', 'seq'])
nm_all = app.groupby('rgst_no').applicantName.agg(lambda s: '|'.join(s.fillna('').astype(str)))
first = app.drop_duplicates('rgst_no').set_index('rgst_no')
pt['aname'] = pt.rgst_no.map(first.applicantName).fillna(''); pt['acd'] = pt.rgst_no.map(first.applicantCd).fillna('').astype(str)
pt['all_names'] = pt.rgst_no.map(nm_all).fillna('')
CORP = re.compile(r'주식회사|㈜|\(주\)|유한회사|CO\.|LTD|INC|CORP|GMBH|AG$|S\.A\.|컴퍼니|코포레이션', re.I)
UNIV = re.compile(r'대학|산학협력|학교법인|UNIVERSITY|과학기술원|포항공과|KAIST', re.I)
GRI = re.compile(r'연구원|연구소|연구재단|연구회|진흥원|INSTITUTE|공사$|공단$|한국전력|센터', re.I)
GOV = re.compile(r'대한민국|국방|방위사업|청장|장관|특별시|광역시|도지사|시장|군수|국립', re.I)
def atype(n, c):
    if UNIV.search(n) and not CORP.search(n): return 'univ'
    if GOV.search(n) and not CORP.search(n): return 'gov'
    if GRI.search(n) and not CORP.search(n): return 'gri'
    if CORP.search(n) or c[:1] in ('1', '2'): return 'corp'
    if c[:1] in ('4', '5') or (len(n) <= 4 and re.fullmatch(r'[가-힣]+', n or 'x')): return 'indiv'
    return 'other'
pt['atype'] = [atype(n, c) for n, c in zip(pt.aname, pt.acd)]
pt['k2'] = pt.ksic.astype(str).str[:2]; pt['ry'] = pt.rd.dt.year
tab = pt.groupby(['atype', 'gov_rnd']).agg(n=('lap5', 'size'), lapse5=('lap5', 'mean')).unstack()
print(tab.round(4).to_string())
res['by_applicant_type'] = {f'{a}|{"gov" if g else "nongov"}': dict(n=int(tab.loc[a, ('n', g)]) if not pd.isna(tab.loc[a, ('n', g)]) else 0,
                            lapse5=float(tab.loc[a, ('lapse5', g)]) if not pd.isna(tab.loc[a, ('lapse5', g)]) else None) for a in tab.index for g in (False, True)}
res['gov_composition'] = pt[pt.gov_rnd].atype.value_counts(normalize=True).round(4).to_dict()
res['nongov_composition'] = pt[~pt.gov_rnd].atype.value_counts(normalize=True).round(4).to_dict()
# 직접 표준화: 층 = 출원인유형 × 등록연도 × KSIC(5자리; 비기원 층 n<20이면 KSIC 2자리로 수축)
def std_rate(keys):
    g = pt.groupby(keys + ['gov_rnd']).lap5.agg(['size', 'mean']).unstack('gov_rnd')
    g.columns = [f'{a}_{"g" if b else "n"}' for a, b in g.columns]
    g = g.fillna({'size_g': 0, 'size_n': 0})
    ok = (g.size_g > 0) & (g.size_n >= 20)
    cov = g.loc[ok, 'size_g'].sum() / g.size_g.sum()
    w = g.loc[ok, 'size_g'] / g.loc[ok, 'size_g'].sum()
    gov = (w * g.loc[ok, 'mean_g']).sum(); non = (w * g.loc[ok, 'mean_n']).sum()   # 정부기원 구성으로 표준화
    w2 = g.loc[ok & (g.size_g >= 20), 'size_n']; w2 = w2 / w2.sum()
    gov2 = (w2 * g.loc[w2.index, 'mean_g']).sum(); non2 = (w2 * g.loc[w2.index, 'mean_n']).sum()  # 비기원 구성으로 표준화
    return dict(strata=keys, coverage_gov=float(cov), gov_std_govmix=float(gov), nongov_std_govmix=float(non), gap_pp_govmix=float(100 * (non - gov)),
                gov_std_nongovmix=float(gov2), nongov_std_nongovmix=float(non2), gap_pp_nongovmix=float(100 * (non2 - gov2)))
res['standardized'] = [std_rate(['atype']), std_rate(['atype', 'ry']), std_rate(['atype', 'ry', 'k2']), std_rate(['atype', 'ry', 'ksic'])]
for s in res['standardized']:
    print(s)
# 보안(국방) 기원 누락 민감도: 비기원 중 국방 계열 출원인(국방과학연구소·대한민국(방위사업청·국방부)·방산 상위 기업명) 특허를 정부기원으로 옮겨 재계산
DEF = re.compile(r'국방과학연구소|방위사업청|국방부|대한민국\(?국방|LIG넥스원|엘아이지넥스원|한화에어로스페이스|한화시스템|한국항공우주산업|현대로템|풍산|한화디펜스|삼성탈레스', re.I)
pt['defense'] = pt.all_names.str.contains(DEF)
d_non = pt[~pt.gov_rnd & pt.defense]
res['defense_sensitivity'] = dict(n_nongov_defense=int(len(d_non)), share_of_nongov=float(len(d_non) / (~pt.gov_rnd).sum()), lapse5_nongov_defense=float(d_non.lap5.mean()) if len(d_non) else None,
                                  n_gov_defense=int((pt.gov_rnd & pt.defense).sum()))
g2 = pt.gov_rnd | pt.defense
res['defense_sensitivity'].update(lapse5_gov_plus_def=float(pt[g2].lap5.mean()), lapse5_nongov_minus_def=float(pt[~g2].lap5.mean()))
print(res['defense_sensitivity'])
json.dump(res, open(os.path.join(OUT, '_tab_e_lapse.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
