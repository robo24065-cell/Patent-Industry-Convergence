# -*- coding: utf-8 -*-
"""s04: 데이터 검증 (a)–(d) 원천 집계 → cache/s04_out.json, official_compare_patent.csv
 python -I 가 아닌 일반 실행(동결 코드 import 없음). 외부 통계 파일은 읽기만(C13-T1 연보 원표, C10-T1 KOSIS json)."""
import sys, pathlib, json, re, sqlite3, time, glob
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard
HERE = guard.HERE
import numpy as np, pandas as pd, openpyxl
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
T1 = B/'분석/모델경쟁_20260927/T1_산업대기실'; CACHE = T1/'cache'; R3 = T1/'R3'; EXTD = B/'분석/모델경쟁_20260927/_shared/ext'
LOOP = B/'분석/검증루프_20261012'
YB = LOOP/'C13/C13-T1/download/moip_unzip'; KO = LOOP/'C10/C10-T1/download/kosis'
t0 = time.time(); OUT = {}
def smd(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); s = np.sqrt((np.nanvar(a) + np.nanvar(b)) / 2)
    return float((np.nanmean(a) - np.nanmean(b)) / s) if s > 0 else 0.0
# ---------------- 원천 sqlite 직접 집계(부록 DATA-BIZRIGHTS 재계산) ----------------
con = sqlite3.connect((B/'수집파이프라인/business_rights.sqlite').as_uri() + '?mode=ro', uri=True)
OUT['rl_total_rows'] = con.execute('select count(*) from right_list').fetchone()[0]
OUT['rl_rows_by_prefix'] = dict(con.execute('select substr(rgst_no,1,2), count(*) from right_list group by 1').fetchall())
OUT['aq_status'] = dict(con.execute('select status, count(*) from applicant_query group by 1').fetchall())
OUT['aq_ok_fetched_eq_total'] = con.execute("select sum(fetched=total_count), count(*), sum(truncated) from applicant_query where status='OK'").fetchone()
pat = pd.read_sql("select applicant_cd, rgst_no, rgst_date, appl_date, title, json_extract(raw,'$.pertDiv') pert from right_list where substr(rgst_no,1,2)='10'", con)
pat['rgst_no'] = pat.rgst_no.astype(str)
kr = pd.read_parquet(CACHE/'ins_pat.parquet', columns=['rgst_no', 'ay', 'rgst_date', 'ksic']); kr['rgst_no'] = kr.rgst_no.astype(str)
INS = set(kr.rgst_no)
u = pat.rgst_no.drop_duplicates(); OUT['rl_patent_rows'] = int(len(pat)); OUT['rl_patent_unique'] = int(len(u))
OUT['rl_patent_unique_out_of_scope'] = int((~u.isin(INS)).sum()); OUT['rl_patent_unique_in_scope'] = int(u.isin(INS).sum())
OUT['oos_share_of_unique_patents'] = OUT['rl_patent_unique_out_of_scope'] / OUT['rl_patent_unique']
pr = pat[pat.pert == '권리자란']; ur = pr.rgst_no.drop_duplicates()
OUT['rl_patent_rightholder_unique'] = int(len(ur)); OUT['rl_patent_rightholder_unique_oos'] = int((~ur.isin(INS)).sum())
OUT['oos_rows_all_pert'] = int((~pat.rgst_no.isin(INS)).sum()); OUT['oos_rows_rightholder'] = int((~pr.rgst_no.isin(INS)).sum())
OUT['oos_share_of_rl_total_rows'] = OUT['rl_patent_unique_out_of_scope'] / OUT['rl_total_rows']
OUT['oos_unique_share_of_patent_rows'] = OUT['rl_patent_unique_out_of_scope'] / OUT['rl_patent_rows']
OUT['oos_rows_share_of_patent_rows'] = OUT['oos_rows_all_pert'] / OUT['rl_patent_rows']
OUT['kr_registered_rows'] = int(len(kr)); OUT['kr_prefix'] = kr.rgst_no.str[:2].value_counts().to_dict()
print('sqlite done', round(time.time() - t0), flush=True)
# ---------------- (a) 공식 총량 ----------------
def yb_series(path, title_key):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = [w for w in wb.worksheets if title_key in w.title.replace(' ', '')][0]
    out = {}
    for r in ws.iter_rows(values_only=True):
        if r and isinstance(r[0], (int, float)) and 1940 < r[0] < 2100 and isinstance(r[1], (int, float)) and int(r[0]) not in out:
            out[int(r[0])] = r[1:]
    return out
f25 = glob.glob(str(YB/'f_437_1'/'*1. 총괄*'))[0]; f19 = glob.glob(str(YB/'f_312_2'/'*1-총괄*'))[0]
s25 = yb_series(f25, '1.5.'); s19 = yb_series(f19, '1.5.')
yb = pd.DataFrame({'year': list(s25), 'yb25_patent': [v[0] for v in s25.values()], 'yb25_um': [v[1] for v in s25.values()]})
yb['yb19_patent'] = yb.year.map({k: v[0] for k, v in s19.items()})
rev = yb.dropna(subset=['yb19_patent']); OUT['yb_revision_2019vs2025_max_abs'] = float((rev.yb25_patent - rev.yb19_patent).abs().max()); OUT['yb_revision_years'] = [int(rev.year.min()), int(rev.year.max())]
inf = yb_series(f25, '1.7.'); OUT['yb_in_force_patent'] = {k: v[0] for k, v in inf.items()}
def J(f): x = pd.DataFrame(json.load(open(KO/f, encoding='utf-8'))); x['DT'] = pd.to_numeric(x.DT, errors='coerce'); return x
a06 = J('data_TX_13801_A006_2005_2025.json'); a40 = J('data_TX_13801_A040_1_2010_2025.json'); a45 = J('data_TX_13801_A045_1990_2016.json')
k06 = a06[a06.C1 == '15138AG1AA'].groupby('PRD_DE').DT.sum()
k40 = a40[(a40.C1 == 'A002') & (a40.C2 == 'B003') & (a40.C3 == 'C001')].groupby('PRD_DE').DT.sum()
k40d = a40[(a40.C1 == 'A002') & (a40.C2 == 'B003') & (a40.C3 == 'C002')].groupby('PRD_DE').DT.sum()
k40t = a40[(a40.C1 == 'A002') & (a40.C2 == 'B001') & (a40.C3 == 'C001')].groupby('PRD_DE').DT.sum()
x45 = a45[a45.C1 == '15138AG1AA']; k45 = {}
for y in ('2004', '2005', '2006'): k45[y] = float(x45[(x45.PRD_DE == y) & x45.C2.isin(['15138AI4ABAB', '15138AI4ADAB'])].DT.sum())
k45['2007'] = float(x45[(x45.PRD_DE == '2007') & (x45.C2 == '15138AI4AC00')].DT.sum())
yb['kosis_total'] = yb.year.astype(str).map(k06); yb['kosis_corp'] = yb.year.astype(str).map(lambda s: k40.get(s, k45.get(s, np.nan)))
yb['kosis_corp_dom'] = yb.year.astype(str).map(k40d); yb['kosis_a040_total'] = yb.year.astype(str).map(k40t)
OUT['kosis_vs_yb_total_max_abs'] = float((yb.kosis_total - yb.yb25_patent).abs().max()); OUT['kosis_a040_total_vs_yb_max_abs'] = float((yb.kosis_a040_total - yb.yb25_patent).abs().max())
# right_list 측 범위밖(권리자란), 등록연도별 고유 등록번호
rl = pd.read_parquet(CACHE/'rl_pat.parquet', columns=['applicant_cd', 'rgst_no', 'rgst_date', 'appl_date', 'pert', 'tp', 'st']); rl['rgst_no'] = rl.rgst_no.astype(str)
rr = rl[rl.pert == '권리자란'].copy(); rr['ry'] = pd.to_numeric(rr.rgst_date.astype(str).str[:4], errors='coerce'); rr['ins'] = rr.rgst_no.isin(INS)
oo = rr[~rr.ins]
fl = pd.read_parquet(HERE/'cache/oos_flags.parquet')
feat = pd.read_parquet(CACHE/'oos.parquet', columns=['applicant_cd', 'rgst_no', 'ay', 'appl_date', 'rgst_date', 'title', 'tp', 'self'])
feat['rgst_no'] = feat.rgst_no.astype(str); feat = feat[feat.self & (feat.ay >= 2000)].reset_index(drop=True)
assert (feat.rgst_no.values == fl.rgst_no.values).all()
feat['ry'] = feat.rgst_date.astype(str).str[:4].astype(int)
corp = oo.tp.isin(['국내법인', '외국법인', '국가기관', '각급시험연구기관']) | oo.tp.astype(str).str.contains('법인')
yb['rl_oos_unique'] = yb.year.map(oo.groupby('ry').rgst_no.nunique()).fillna(0)
yb['rl_oos_corp_unique'] = yb.year.map(oo[corp].groupby('ry').rgst_no.nunique()).fillna(0)
yb['rl_oos_dom_corp_unique'] = yb.year.map(oo[oo.tp == '국내법인'].groupby('ry').rgst_no.nunique()).fillna(0)
yb['t1_oos_feature_unique'] = yb.year.map(feat.groupby('ry').rgst_no.nunique()).fillna(0)
yb['rl_ins_unique'] = yb.year.map(rr[rr.ins].groupby('ry').rgst_no.nunique()).fillna(0)
krr = kr[kr.rgst_no.str[:2] == '10'].copy(); krr['ry'] = krr.rgst_date.astype(str).str[:4].astype(int)
yb['kr_provided_patent'] = yb.year.map(krr.groupby('ry').rgst_no.nunique()).fillna(0)
yb['r_oos_total'] = yb.rl_oos_unique / yb.yb25_patent; yb['r_oos_corp'] = yb.rl_oos_corp_unique / yb.kosis_corp
yb['r_oos_domcorp'] = yb.rl_oos_dom_corp_unique / yb.kosis_corp_dom; yb['r_feature_total'] = yb.t1_oos_feature_unique / yb.yb25_patent
yb['r_provided_total'] = yb.kr_provided_patent / yb.yb25_patent
yb['r_oos_over_provided'] = yb.rl_oos_unique / yb.kr_provided_patent
yb['r_rl_ins_over_provided'] = yb.rl_ins_unique / yb.kr_provided_patent
yb = yb[(yb.year >= 1995) & (yb.year <= 2025)].sort_values('year')
yb.to_csv(HERE/'official_compare_patent.csv', index=False, encoding='utf-8-sig')
for col in ('r_oos_total', 'r_oos_corp', 'r_feature_total', 'r_oos_over_provided'):
    s = yb[(yb.year >= 2000) & (yb.year <= 2019)].set_index('year')[col].dropna()
    if len(s) == 0: continue
    med = float(s.median()); OUT[f'{col}_median_2000_2019'] = med
    OUT[f'{col}_years_below_0.7med'] = [int(y) for y, v in s.items() if v < 0.7 * med]
    OUT[f'{col}_slope_per_year_2002_2019'] = float(np.polyfit(s.loc[2002:2019].index.values, s.loc[2002:2019].values, 1)[0]) if len(s.loc[2002:2019]) > 3 else None
    OUT[f'{col}_by_year'] = {int(y): round(float(v), 4) for y, v in s.items()}
# 존속 상태
oo2 = oo.copy(); oo2['alive'] = oo2.st.fillna('').str.contains('등록유지')
OUT['oos_status_top'] = oo2.st.value_counts().head(8).to_dict(); OUT['oos_alive_share'] = float(oo2.alive.mean())
oo2['yg'] = pd.cut(oo2.ry, [0, 1999, 2004, 2009, 2014, 2019, 2030], labels=['~1999', '2000–04', '2005–09', '2010–14', '2015–19', '2020+'])
OUT['oos_alive_share_by_rgst_group'] = oo2.groupby('yg', observed=False).alive.mean().round(4).astype(str).to_dict()
OUT['official_in_force_over_cum_reg_2025'] = None
print('(a) done', round(time.time() - t0), flush=True)
# ---------------- (b) 분포·결측·중복 (피처에 쓰이는 범위밖 행) ----------------
OUT['feat_rows'] = int(len(feat)); OUT['feat_unique_rgst'] = int(feat.rgst_no.nunique())
OUT['feat_dup_firm_rgst'] = int(feat.duplicated(['applicant_cd', 'rgst_no']).sum())
OUT['feat_multi_firm_rgst_row_share'] = float((feat.groupby('rgst_no').applicant_cd.transform('size') > 1).mean())
OUT['feat_missing'] = dict(appl_date=float(feat.appl_date.isna().mean() + (feat.appl_date.astype(str).str.len() != 8).mean()),
                          rgst_date=float((feat.rgst_date.astype(str).str.len() != 8).mean()), title=float(feat.title.fillna('').str.strip().eq('').mean()))
OUT['raw_pert_dist_oos'] = rl[~rl.rgst_no.isin(INS)].pert.value_counts().to_dict()
OUT['raw_rightholder_dup_rows_oos'] = int(oo.duplicated(['applicant_cd', 'rgst_no']).sum())
OUT['feat_by_rgst_year'] = {int(k): int(v) for k, v in feat.ry.value_counts().sort_index().items()}
OUT['feat_tp_row'] = feat.tp.value_counts().head(10).to_dict()
P = pd.read_parquet(EXTD/'firm_nps_firm_panel.parquet', columns=['applicantCd', 'snap', 'n_emp']); P = P[P.snap == '2019-12'].drop_duplicates('applicantCd').set_index('applicantCd').n_emp
def sbin(e):
    if pd.isna(e): return '미연결'
    return '<10' if e <= 9 else ('10–49' if e <= 49 else ('50–299' if e <= 299 else '300+'))
feat['size'] = feat.applicant_cd.map(P).map(sbin)
OUT['feat_by_size_rows'] = feat['size'].value_counts().to_dict()
ins = pd.read_parquet(CACHE/'ins_firm.parquet', columns=['applicant_cd', 'rgst_no', 'ksic', 'ay'])
mk = ins.groupby('applicant_cd').ksic.agg(lambda s: s.value_counts().index[0])
feat['mainksic'] = feat.applicant_cd.map(mk).fillna('범위안 없음')
vc = feat.mainksic.value_counts(); OUT['feat_by_main_ksic_top10'] = vc.head(10).to_dict(); OUT['feat_main_ksic_none_share'] = float((feat.mainksic == '범위안 없음').mean())
OUT['feat_main_ksic_top1_share'] = float(vc.iloc[0] / vc.sum())
# 범위밖 행이 있는 법인 수, 법인당 행
g = feat.groupby('applicant_cd').size(); OUT['feat_firms'] = int(len(g)); OUT['feat_rows_per_firm_q'] = g.quantile([.5, .9, .99]).to_dict()
OUT['feat_top10_firm_row_share'] = float(g.sort_values(ascending=False).head(10).sum() / g.sum())
print('(b) done', round(time.time() - t0), flush=True)
# ---------------- (c) 연결 편향: T1 2019 모집단 ot_ 보유 대 비보유 ----------------
te = pd.read_parquet(R3/'cache_final/o2019/prep_te.parquet', columns=['firm', 'k', 'lsize', 'lin', 'lout', 'newc', 'prior_any', 'ot_max'])
L = pd.read_parquet(LOOP/'C10/C10-T1/cache/final_labels.parquet', columns=['firm', 'k', 'y_u'])
assert (L.firm.values == te.firm.values).all()
te['y_u'] = L.y_u.values; te = te[~te.prior_any.values]
ff = te.groupby('firm').agg(lsize=('lsize', 'first'), lin=('lin', 'first'), lout=('lout', 'first'), newc=('newc', 'first'), npos=('y_u', 'sum'))
ff['has_ot'] = (ff.lout > 0).astype(int); ff['entry_any'] = (ff.npos > 0).astype(int)
ff['lemp'] = np.log1p(ff.index.map(P).astype(float)); ff['emp_linked'] = ff.index.isin(P.index).astype(int)
fm = pd.read_parquet(CACHE/'firm_meta.parquet', columns=['applicant_cd', 'tp']).set_index('applicant_cd').tp
ff['foreign'] = (ff.index.map(fm) == '외국법인').astype(int)
fa = ins[ins.ay >= 2000].groupby('applicant_cd').ay.min(); ff['first_ins'] = ff.index.map(fa).astype(float)
OUT['c_firms'] = int(len(ff)); OUT['c_has_ot_share'] = float(ff.has_ot.mean())
OUT['c_smd'] = {v: smd(ff[ff.has_ot == 1][v], ff[ff.has_ot == 0][v]) for v in ('lsize', 'lin', 'newc', 'lemp', 'emp_linked', 'foreign', 'first_ins', 'entry_any')}
OUT['c_means'] = {v: [float(ff[ff.has_ot == 1][v].mean()), float(ff[ff.has_ot == 0][v].mean())] for v in OUT['c_smd']}
OUT['c_pairs_share_has_ot'] = float(te.lout.gt(0).mean()); OUT['c_pos_share_has_ot'] = float(te[te.y_u == 1].lout.gt(0).mean())
# 아날로그 누락률과 ot_ 보유 여부(법인 수준 누락률의 분포)
A = pd.read_parquet(HERE/'cache/analog_A.parquet', columns=['applicant_cd', 'ry', 'surv'])
fd = A[A.ry <= 2019].groupby('applicant_cd').surv.agg(lambda s: 1 - s.mean())
ff['firm_dropout'] = ff.index.map(fd).astype(float)
OUT['c_firm_dropout_mean_has_ot'] = [float(ff[ff.has_ot == 1].firm_dropout.mean()), float(ff[ff.has_ot == 0].firm_dropout.mean())]
OUT['c_firm_dropout_by_entry'] = [float(ff[ff.entry_any == 1].firm_dropout.mean()), float(ff[ff.entry_any == 0].firm_dropout.mean())]
OUT['c_firm_dropout_smd_entry'] = smd(ff[ff.entry_any == 1].firm_dropout, ff[ff.entry_any == 0].firm_dropout)
print('(c) done', round(time.time() - t0), flush=True)
# ---------------- (d) 시점: 피처 통계 모음 ----------------
st = {}
for f in sorted((HERE/'cache').glob('feat_stats_o*_c*.json')):
    s = json.load(open(f, encoding='utf-8')); st[f.stem.replace('feat_stats_', '')] = {k: s.get(k) for k in ('op_rows', 'op_rgst_after_cut', 'op_ay_after_cut', 'op_not_in_holder', 'P0_equal', 'S2_removed_share', 'S3_1_removed_share', 'S4_added_share', 'AN_A_rows', 'AN_B_rows')}
OUT['d_feat_stats'] = st
fl2 = fl.merge(feat[['rgst_no', 'ry']].reset_index(drop=True), left_index=True, right_index=True, suffixes=('', '_f'))
fl2['alive'] = fl2.st.fillna('').str.contains('등록유지')
OUT['d_feature_rows_le2019_lapsed_share'] = float((~fl2[fl2.ry <= 2019].alive).mean())
OUT['d_not_in_holder_le2019'] = float((~fl2[fl2.ry <= 2019].in_holder).mean())
json.dump(OUT, open(HERE/'cache/s04_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: v for k, v in OUT.items() if not isinstance(v, dict) or len(v) < 12}, ensure_ascii=False, indent=1, default=str)[:9000]); print('done', round(time.time() - t0))
