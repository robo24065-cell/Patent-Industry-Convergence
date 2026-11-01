# -*- coding: utf-8 -*-
"""s01: 범위안 아날로그 A(등록원부 원출원인)·B(2026 right_list 권리자란+자사필터 생존) 표, 누락률 표(analog_dropout.csv), S3/S4 셀 누락률,
 + 범위밖 행(lib.load()의 oos 순서) 플래그(권리자 표기에 별칭 있음 = in_holder) → cache/oos_flags.parquet
 원천: T1 cache(rl_pat = business_rights.right_list 특허 행 추출본, ins_applicant = registry_full.applicant, ins_pat = 제공 KR 등록), 읽기만."""
import sys, pathlib, re, json, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard
HERE = guard.HERE
import numpy as np, pandas as pd
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
T1 = B/'분석/모델경쟁_20260927/T1_산업대기실'; CACHE = T1/'cache'; R3 = T1/'R3'; EXTD = B/'분석/모델경쟁_20260927/_shared/ext'
t0 = time.time(); OUT = {}
STRIP = re.compile(r"주식회사|\(주\)|㈜|유한회사|유한책임회사|재단법인|사단법인|학교법인|가부시키가이샤|가부시끼가이샤|가부시기가이샤|가부시키카이샤|코포레이션|인코포레이티드|리미티드|컴퍼니|[\s\.\,\-\(\)\'\"·&]")
def norm(s): return STRIP.sub('', str(s)).lower()
firms = pd.read_parquet(CACHE/'firms.parquet'); ok = firms[firms.status == 'OK']; OKS = set(ok.applicant_cd)
app = pd.read_parquet(CACHE/'ins_applicant.parquet', columns=['rgst_no', 'applicantCd', 'applicantName']); app['rgst_no'] = app.rgst_no.astype(str)
al = pd.concat([ok[['applicant_cd', 'applicant_name']].rename(columns={'applicant_name': 'nm'}),
                app[app.applicantCd.isin(OKS)][['applicantCd', 'applicantName']].rename(columns={'applicantCd': 'applicant_cd', 'applicantName': 'nm'})])
al['n'] = al.nm.map(norm); al = al[al.n.str.len() >= 2].drop_duplicates(['applicant_cd', 'n'])
alias = al.groupby('applicant_cd').n.apply(list).to_dict()
def has_alias(cd, txt):      # prep2.py is_self 와 같은 규칙
    a = norm(txt)
    for n in alias.get(cd, []):
        if len(n) >= 3 and n in a: return True
        if len(n) < 3 and any(norm(p) == n for p in str(txt).split(',')): return True
    return False
rl = pd.read_parquet(CACHE/'rl_pat.parquet', columns=['applicant_cd', 'rgst_no', 'rgst_date', 'appl_date', 'pert', 'tp', 'ainfo', 'hinfo', 'st'])
rl['rgst_no'] = rl.rgst_no.astype(str)
OUT['rl_pat_rows'] = int(len(rl)); OUT['rl_pat_pert'] = rl.pert.value_counts(dropna=False).to_dict()
rr = rl[rl.pert == '권리자란'].copy()
OUT['rl_rightholder_dup_rows'] = int(rr.duplicated(['applicant_cd', 'rgst_no']).sum())
rr = rr.drop_duplicates(['applicant_cd', 'rgst_no']).reset_index(drop=True)
rr['self'] = [has_alias(c, a) for c, a in zip(rr.applicant_cd.values, rr.ainfo.values)]
rr['in_holder'] = [has_alias(c, h) for c, h in zip(rr.applicant_cd.values, rr.hinfo.values)]
print('flags', round(time.time() - t0), flush=True)
kr = pd.read_parquet(CACHE/'ins_pat.parquet', columns=['rgst_no', 'ksic', 'ay', 'rgst_date']).drop_duplicates('rgst_no'); kr['rgst_no'] = kr.rgst_no.astype(str)
INS = set(kr.rgst_no)
rr['ins'] = rr.rgst_no.isin(INS)
# ---------- 범위밖 행 플래그(lib.load() 순서와 같게) ----------
oos = pd.read_parquet(CACHE/'oos.parquet', columns=['applicant_cd', 'rgst_no', 'ay', 'rgst_date', 'self'])
oos['rgst_no'] = oos.rgst_no.astype(str)
m = oos.merge(rr[['applicant_cd', 'rgst_no', 'self', 'in_holder', 'hinfo', 'st']].rename(columns={'self': 'self2'}), on=['applicant_cd', 'rgst_no'], how='left', validate='m:1')
assert len(m) == len(oos)
OUT['oos_rows'] = int(len(oos)); OUT['oos_unmatched_rl'] = int(m.self2.isna().sum())
OUT['oos_self_agree'] = float((m.self.values == m.self2.fillna(False).values).mean())
keep = m.self.values & (m.ay.values >= 2000)
mf = m[keep].reset_index(drop=True); mf['row'] = np.arange(len(mf))
mf[['row', 'applicant_cd', 'rgst_no', 'in_holder', 'st']].to_parquet(HERE/'cache/oos_flags.parquet', index=False)
OUT['oos_feature_rows'] = int(len(mf)); OUT['oos_feature_in_holder_share'] = float(mf.in_holder.mean())
print('oos flags', len(mf), round(time.time() - t0), flush=True)
# ---------- 아날로그 A·B ----------
A = app[app.applicantCd.isin(OKS)][['rgst_no', 'applicantCd']].rename(columns={'applicantCd': 'applicant_cd'}).drop_duplicates()
A = A.merge(kr, on='rgst_no', how='inner'); OUT['A_rows_um_prefix20_dropped'] = int((A.rgst_no.str[:2] != '10').sum()); A = A[(A.ay >= 2000) & (A.rgst_no.str[:2] == '10')]   # 변경 기록 1: 실용신안(20) 제외
Bs = rr[rr.ins & rr.self & rr.applicant_cd.isin(OKS)][['applicant_cd', 'rgst_no']].copy()
Bs = Bs.merge(kr, on='rgst_no', how='inner'); Bs = Bs[(Bs.ay >= 2000) & (Bs.rgst_no.str[:2] == '10')]
Bkey = set(zip(Bs.applicant_cd, Bs.rgst_no)); Akey = set(zip(A.applicant_cd, A.rgst_no))
A['surv'] = [(c, r) in Bkey for c, r in zip(A.applicant_cd.values, A.rgst_no.values)]
Bs['inA'] = [(c, r) in Akey for c, r in zip(Bs.applicant_cd.values, Bs.rgst_no.values)]
# 등록원부 최종 권리자가 F 인가(보유 유지) — 누락 기제 분해용
import sqlite3
con = sqlite3.connect((B/'수집파이프라인/registry_full.sqlite').as_uri() + '?mode=ro', uri=True)
ow = pd.read_sql('select rgst_no, seq, ownerName, ownerRgstCsDate from owner', con); ow['rgst_no'] = ow.rgst_no.astype(str)
mx = ow.groupby('rgst_no').seq.max()
fin = ow[ow.seq == ow.rgst_no.map(mx)].groupby('rgst_no').ownerName.apply(lambda s: ','.join(map(str, s)))
chg = ow[ow.seq > 0].groupby('rgst_no').ownerRgstCsDate.min()
A['has_owner'] = A.rgst_no.isin(mx.index)
A['retained'] = [has_alias(c, fin.get(r, '')) for c, r in zip(A.applicant_cd.values, A.rgst_no.values)]
A['first_chg'] = A.rgst_no.map(chg)
print('A', len(A), 'B', len(Bs), round(time.time() - t0), flush=True)
# 속성
A['ry'] = A.rgst_date.astype(str).str[:4].astype(int)
P = pd.read_parquet(EXTD/'firm_nps_firm_panel.parquet', columns=['applicantCd', 'snap', 'n_emp']); P = P[P.snap == '2019-12'].drop_duplicates('applicantCd').set_index('applicantCd').n_emp
def sbin(e):
    if pd.isna(e): return '미연결'
    return '<10' if e <= 9 else ('10–49' if e <= 49 else ('50–299' if e <= 299 else '300+'))
fm = pd.read_parquet(CACHE/'firm_meta.parquet', columns=['applicant_cd', 'tp']).set_index('applicant_cd').tp
A['size'] = A.applicant_cd.map(P).map(sbin); A['tp'] = A.applicant_cd.map(fm).fillna('미상')
pop19 = set(pd.read_parquet(R3/'cache_final/o2019/prep_te.parquet', columns=['firm']).firm.unique())
A['pop19'] = A.applicant_cd.isin(pop19)
def ygrp(y): return '2000–04' if y <= 2004 else ('2005–09' if y <= 2009 else ('2010–14' if y <= 2014 else ('2015–19' if y <= 2019 else '2020+')))
A['yg'] = A.ry.map(ygrp)
A.drop(columns=['first_chg']).to_parquet(HERE/'cache/analog_A.parquet', index=False)
Bs.to_parquet(HERE/'cache/analog_B.parquet', index=False)
# ---------- 표 ----------
rows = []
def add(dim, key, s):
    rows.append(dict(dim=dim, key=str(key), n_A=int(len(s)), dropout=float(1 - s.surv.mean()) if len(s) else np.nan,
                     transferred_share=float((s.has_owner & ~s.retained).mean()) if len(s) else np.nan,
                     dropout_if_retained=float(1 - s[s.has_owner & s.retained].surv.mean()) if (s.has_owner & s.retained).any() else np.nan,
                     dropout_if_transferred=float(1 - s[s.has_owner & ~s.retained].surv.mean()) if (s.has_owner & ~s.retained).any() else np.nan))
A19 = A[A.ry <= 2019]
add('전체(등록 2000–2019)', 'all', A19); add('전체(등록 2000–2025)', 'all', A)
for y, s in A.groupby('ry'): add('등록연도', y, s)
for y, s in A19.groupby('yg'): add('등록연도군', y, s)
for z, s in A19.groupby('size'): add('규모(국민연금 2019-12)', z, s)
for z, s in A19.groupby('tp'): add('법인유형', z, s)
for z, s in A19.groupby('pop19'): add('T1 2019 평가모집단', z, s)
for z, s in A19.groupby('ksic'): add('KSIC', z, s)
for (y, z), s in A19.groupby(['yg', 'size']): add('S3셀(등록연도군×규모)', f'{y}|{z}', s)
D = pd.DataFrame(rows); D.to_csv(HERE/'analog_dropout.csv', index=False, encoding='utf-8-sig')
cells = {r.key: r.dropout for r in D[D.dim == 'S3셀(등록연도군×규모)'].itertuples()}
json.dump(cells, open(HERE/'cache/s3_rates.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
OUT['A_rows'] = int(len(A)); OUT['A_rows_le2019'] = int(len(A19)); OUT['B_rows'] = int(len(Bs))
OUT['dropout_le2019'] = float(1 - A19.surv.mean()); OUT['B_contamination'] = float(1 - Bs.inA.mean())
OUT['B_contamination_le2019'] = float(1 - Bs[Bs.rgst_date.astype(str) <= '20191231'].inA.mean())
OUT['transferred_share_le2019'] = float((A19.has_owner & ~A19.retained).mean())
OUT['has_owner_share'] = float(A.has_owner.mean())
OUT['dropout_retained'] = float(1 - A19[A19.has_owner & A19.retained].surv.mean())
OUT['dropout_transferred'] = float(1 - A19[A19.has_owner & ~A19.retained].surv.mean())
# 넘긴 시점이 원점(2019) 이후인 비율(넘긴 행 중)
tA = A19[A19.has_owner & ~A19.retained]
OUT['transferred_after2019_share'] = float((A.loc[tA.index, 'rgst_no'].map(chg).fillna('0') > '20191231').mean())
# in_holder 정의 점검(범위안): 자사·생존 행 중 권리자 표기에 별칭 없음 비율, 그 중 등록원부상 이전 있음 비율
ri = rr[rr.ins & rr.self & rr.applicant_cd.isin(OKS)].copy()
ri['has_chg'] = ri.rgst_no.isin(chg.index)
OUT['ins_self_rows'] = int(len(ri)); OUT['ins_self_not_in_holder_share'] = float((~ri.in_holder).mean())
OUT['ins_self_not_in_holder_with_transfer'] = float(ri[~ri.in_holder].has_chg.mean()); OUT['ins_self_in_holder_with_transfer'] = float(ri[ri.in_holder].has_chg.mean())
json.dump(OUT, open(HERE/'cache/s01_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(OUT, ensure_ascii=False, indent=1, default=str)); print('done', round(time.time() - t0))
