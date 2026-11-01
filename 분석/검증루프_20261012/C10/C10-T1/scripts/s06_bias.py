# -*- coding: utf-8 -*-
"""s06: (b) 분포·결측·중복, (c) 연결 선택 편향(OK 대 RC:001), (d3) 폐업 법인 목록 존재율 → cache/s06_out.json, bias_smd.csv, dist_*.csv"""
import pathlib, json, time
import numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent.parent
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
T1 = B/'분석/모델경쟁_20260927/T1_산업대기실'; CACHE = T1/'cache'; R3 = T1/'R3'; EXTD = B/'분석/모델경쟁_20260927/_shared/ext'
FC = R3/'cache_final'/'o2019'
t0 = time.time(); OUT = {}
def smd(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    s = np.sqrt((np.nanvar(a) + np.nanvar(b)) / 2); return float((np.nanmean(a) - np.nanmean(b)) / s) if s > 0 else 0.0
aq = pd.read_parquet(HERE/'cache/aq.parquet'); aq['pre'] = aq.applicant_cd.str[0]
fm = pd.read_parquet(CACHE/'firm_meta.parquet'); FM = set(fm.applicant_cd)
OUT['aq_status_by_prefix'] = pd.crosstab(aq.pre, aq.status).to_dict()
OUT['firm_meta_n'] = int(len(FM)); OUT['firm_meta_in_RC001'] = int(aq[(aq.status == 'RC:001') & aq.applicant_cd.isin(FM)].shape[0])
OUT['firm_meta_in_OK'] = int(aq[(aq.status == 'OK') & aq.applicant_cd.isin(FM)].shape[0])
# ---------- (c) 범위안 특허(등록원부 원출원인 코드)로 두 집단 특성
app = pd.read_parquet(CACHE/'ins_applicant.parquet', columns=['rgst_no', 'applicantCd']).drop_duplicates(); app['rgst_no'] = app.rgst_no.astype(str)
pat = pd.read_parquet(CACHE/'ins_pat.parquet', columns=['rgst_no', 'ay', 'ksic']).drop_duplicates('rgst_no'); pat['rgst_no'] = pat.rgst_no.astype(str)
x = app.merge(pat, on='rgst_no')
corp = aq[aq.pre.isin(['1', '2', '3', '5'])][['applicant_cd', 'status', 'pre']]
x = x.merge(corp, left_on='applicantCd', right_on='applicant_cd')
pre = x[x.ay <= 2019]; post = x[(x.ay >= 2020) & (x.ay <= 2023)]
g = pre.groupby('applicant_cd').agg(n=('rgst_no', 'nunique'), first=('ay', 'min'), last=('ay', 'max'))
held = pre.groupby('applicant_cd').ksic.agg(set)
newk = post.groupby('applicant_cd').ksic.agg(set)
g['entry'] = [int(len(newk.get(a, set()) - held.get(a, set())) > 0) for a in g.index]
g['any_post'] = [int(a in newk.index) for a in g.index]
g = g.join(corp.set_index('applicant_cd'))
pop = g[g.n >= 3].copy(); pop['ln'] = np.log1p(pop.n)
rows = []
for nm, sub in (('전체 법인코드(1·2·3·5)', pop), ('국내 법인(1)', pop[pop.pre == '1']), ('외국 법인(5)', pop[pop.pre == '5'])):
    a = sub[sub.status == 'OK']; b = sub[sub.status == 'RC:001']
    for v in ('first', 'ln', 'last', 'entry', 'any_post'):
        rows.append(dict(group=nm, var=v, n_ok=len(a), n_rc=len(b), mean_ok=float(a[v].mean()), mean_rc=float(b[v].mean()) if len(b) else np.nan, smd=smd(a[v], b[v]) if len(b) > 1 else np.nan))
S = pd.DataFrame(rows); S.to_csv(HERE/'bias_smd.csv', index=False, encoding='utf-8-sig')
OUT['smd_ok_vs_rc'] = S.round(4).to_dict('records')
OUT['rc_corp_share_of_pop'] = float((pop.status == 'RC:001').mean())
OUT['rc_corp_pos_share'] = float(pop[pop.status == 'RC:001'].entry.sum() / max(pop.entry.sum(), 1))
# ---------- T1 2019 모집단 안: 디자인 보유 대 미보유(참고), 폐업 법인 존재율
te = pd.read_parquet(FC/'prep_te.parquet', columns=['firm', 'k', 'lsize', 'lin', 'newc', 'prior_any'])
L = pd.read_parquet(HERE/'cache/final_labels.parquet', columns=['y_u'])
te['y_u'] = L.y_u.values; te = te[~te.prior_any.values]
ff = te.groupby('firm').agg(lsize=('lsize', 'first'), lin=('lin', 'first'), newc=('newc', 'first'), npos=('y_u', 'sum'), npairs=('y_u', 'size'))
ff['entry_any'] = (ff.npos > 0).astype(int)
fl = pd.read_parquet(HERE/'cache/dsum_flags.parquet', columns=['applicant_cd', 'rgst_no', 'rgst_date', 'kind', 'pert', 'self', 'st', 'expire_date'])
for kind in ('design', 'um'):
    s = pd.read_parquet(FC/f'P_{kind}_index.parquet', columns=['applicant_cd', 'rgst_date'])
    has = set(s[s.rgst_date.astype(str) <= '20191231'].applicant_cd)
    ff[f'has_{kind}'] = ff.index.isin(has).astype(int)
OUT['t1_2019_firms'] = int(len(ff)); OUT['t1_2019_has_design'] = float(ff.has_design.mean()); OUT['t1_2019_has_um'] = float(ff.has_um.mean())
OUT['design_holder_vs_not_smd'] = {v: smd(ff[ff.has_design == 1][v], ff[ff.has_design == 0][v]) for v in ('lsize', 'lin', 'newc', 'entry_any')}
nts = pd.read_parquet(EXTD/'r3nts_applicant_status.parquet', columns=['applicantCd', 'end_dt', 'b_stt_cd', 'registered', 'missing'])
nts = nts.drop_duplicates('applicantCd').set_index('applicantCd')
ed = pd.to_datetime(nts.end_dt, errors='coerce')
ff['closed_le2019'] = ff.index.map(lambda a: a in ed.index and pd.notna(ed.get(a)) and ed.get(a) <= pd.Timestamp('2019-12-31')).astype(int)
ff['closed_after2019'] = ff.index.map(lambda a: a in ed.index and pd.notna(ed.get(a)) and ed.get(a) > pd.Timestamp('2019-12-31')).astype(int)
ff['nts_linked'] = ff.index.isin(nts.index).astype(int)
grp = {}
for nm, m in (('폐업≤2019', ff.closed_le2019 == 1), ('폐업 2020+', ff.closed_after2019 == 1), ('국세청 연결·비폐업', (ff.nts_linked == 1) & (ff.closed_le2019 == 0) & (ff.closed_after2019 == 0)), ('국세청 미연결', ff.nts_linked == 0)):
    grp[nm] = dict(firms=int(m.sum()), has_design=float(ff[m].has_design.mean()) if m.any() else None, has_um=float(ff[m].has_um.mean()) if m.any() else None,
                   lsize=float(ff[m].lsize.mean()) if m.any() else None)
OUT['closed_presence'] = grp
# 폐업 법인의 디자인 권리 상태 분포
cl = set(ff[ff.closed_le2019 == 1].index)
d_cl = fl[fl.applicant_cd.isin(cl) & (fl.kind == 'design')]
OUT['closed_design_status'] = d_cl.st.value_counts().head(6).to_dict()
# ---------- (b) 분포·결측·중복
for kind in ('design', 'um'):
    s = pd.read_parquet(R3/'cache'/f'src_{kind}.parquet', columns=['applicant_cd', 'rgst_no', 'appl_date', 'rgst_date', 'ay'])
    raw = fl[fl.kind == kind]
    OUT[f'{kind}_raw_rows'] = int(len(raw)); OUT[f'{kind}_raw_dup_rows'] = int(raw.duplicated(['applicant_cd', 'rgst_no']).sum())
    OUT[f'{kind}_raw_rgst_date_bad'] = float((raw.rgst_date.fillna('').astype(str).str.len() != 8).mean())
    OUT[f'{kind}_src_rows'] = int(len(s)); OUT[f'{kind}_src_appl_date_missing'] = float((s.ay == 0).mean())
    OUT[f'{kind}_src_multi_holder_rgst_share'] = float((s.groupby('rgst_no').applicant_cd.transform('size') > 1).mean())
    OUT[f'{kind}_src_rows_not_rightholder'] = float((raw.drop_duplicates(['applicant_cd', 'rgst_no']).merge(s[['applicant_cd', 'rgst_no']], on=['applicant_cd', 'rgst_no']).pert != '권리자란').mean())
    s['ry'] = s.rgst_date.astype(str).str[:4].astype(int)
    OUT[f'{kind}_src_by_rgst_year'] = {int(k): int(v) for k, v in s.ry.value_counts().sort_index().items()}
    OUT[f'{kind}_src_rgst_lag_median_days'] = float((pd.to_datetime(s.rgst_date, format='%Y%m%d', errors='coerce') - pd.to_datetime(s.appl_date, format='%Y%m%d', errors='coerce')).dt.days.median())
    OUT[f'{kind}_status_dist'] = raw.st.value_counts().head(8).to_dict()
    OUT[f'{kind}_expired_share'] = float((~raw.st.fillna('').str.contains('등록유지')).mean())
# 규모·지역: 국민연금 2019-12
P = pd.read_parquet(EXTD/'firm_nps_firm_panel.parquet', columns=['applicantCd', 'snap', 'n_emp', 'indcd']); P = P[P.snap == '2019-12'].drop_duplicates('applicantCd').set_index('applicantCd')
ff['emp'] = ff.index.map(P.n_emp); ff['size_bin'] = pd.cut(ff.emp.fillna(-1), [-2, -0.5, 9, 49, 299, 1e9], labels=['미연결', '<10', '10–49', '50–299', '300+'])
OUT['size_bin_presence'] = ff.groupby('size_bin', observed=False).agg(firms=('has_design', 'size'), has_design=('has_design', 'mean'), has_um=('has_um', 'mean'), entry_any=('entry_any', 'mean')).round(4).reset_index().astype(str).to_dict('records')
ff['type'] = [{'1': '국내법인', '2': '국내기관(2)', '3': '국내(3)', '5': '외국법인'}.get(a[0], a[0]) for a in ff.index]
OUT['type_presence'] = ff.groupby('type').agg(firms=('has_design', 'size'), has_design=('has_design', 'mean'), has_um=('has_um', 'mean'), entry_any=('entry_any', 'mean')).round(4).reset_index().astype(str).to_dict('records')
json.dump(OUT, open(HERE/'cache/s06_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print('done', round(time.time() - t0))
