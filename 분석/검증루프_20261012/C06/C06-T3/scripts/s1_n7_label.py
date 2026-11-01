# -*- coding: utf-8 -*-
"""C06-T3 s1: N7 모집단·레이블 독립 재구성 (정의 문서만 보고 작성, 트랙 코드 import 없음).
t0=2022. 단위 = 국내법인 출원인코드('1'로 시작).
조건: 등록원부 applicant 코드로 rgstDate<=20221231 제공범위 KR 등록특허>=1, CORP_APPLICANT 사업자번호, 국세청 registered,
      폐업일 없음 또는 >2022-12-31, 사업자번호가 NTIS 과제연도 2010–2022 corp_rgst_no에 없음.
y = 첫 등장 과제연도 2023 또는 2024, y1 = 2023.
"""
import sqlite3, os, sys, io, json, re
import pandas as pd, numpy as np, pyarrow.parquet as pq
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
SH = ROOT + r"\분석\모델경쟁_20260927\_shared\ext"
OUT = ROOT + r"\분석\검증루프_20261012\C06\C06-T3"
o = {}
# 1) 등록원부: 출원인코드별 첫 등록일
con = sqlite3.connect("file:" + ROOT + r"\수집파이프라인\registry_full.sqlite?mode=ro", uri=True)
ap = pd.read_sql("select a.rgst_no, a.applicantCd cd, p.rgstDate rd, p.applDate ad from applicant a join patent p on a.rgst_no=p.rgst_no", con)
con.close()
o['reg_applicant_rows'] = len(ap)
ap['cd'] = ap.cd.astype(str).str.strip()
ap['rd'] = pd.to_numeric(ap.rd, errors='coerce')
o['reg_rd_missing'] = int(ap.rd.isna().sum())
dom = ap[ap.cd.str[:1] == '1']
first_rd = dom.groupby('cd').rd.min()
n_reg_le = dom[dom.rd <= 20221231].groupby('cd').rgst_no.nunique()
o['dom_codes_any_reg'] = int(first_rd.size); o['dom_codes_reg_le2022'] = int(n_reg_le.size)
# 2) CORP_APPLICANT
rows = []
with open(ROOT + r"\공모전데이터\외부데이터\Corporate_20260720\CORP_APPLICANT.txt", encoding='utf-8') as f:
    head = next(f)
    for line in f:
        p = line.rstrip("\r\n").split("¶")
        if len(p) >= 5:
            rows.append((p[0].strip(), p[1].strip(), p[3].strip(), p[4].strip()))
ca = pd.DataFrame(rows, columns=['cd', 'aname', 'crn', 'bizraw'])
ca['bz'] = ca.bizraw.str.replace(r'\D', '', regex=True)
o['ca_rows'] = len(ca); o['ca_dup_cd'] = int(ca.cd.duplicated().sum())
o['ca_bz_len_dist'] = ca.bz.str.len().value_counts().head(6).to_dict()
ca10 = ca[ca.bz.str.len() == 10].drop_duplicates('cd')
# 3) 국세청 상태(사업자번호 단위)
st = pd.read_parquet(SH + r"\r3nts_status.parquet", columns=['bizno', 'registered', 'end_dt', 'b_stt_cd'])
st['bizno'] = st.bizno.astype(str)
o['nts_rows'] = len(st); o['nts_dup'] = int(st.bizno.duplicated().sum())
st = st.drop_duplicates('bizno')
# 4) NTIS
nt = pq.read_table(SH + r"\ntis_project_raw.parquet", columns=['year', 'corp_rgst_no', 'secret_flag', 'perform_grp']).to_pandas()
nt['bz'] = nt.corp_rgst_no.fillna('').astype(str).str.replace(r'\D', '', regex=True)
o['ntis_rows'] = len(nt)
ntb = nt[(nt.bz.str.len() == 10) & (nt.year >= 2010)]
firsty = ntb.groupby('bz').year.min()
o['ntis_secret_rows_by_year_2020_24'] = nt[nt.secret_flag & nt.year.between(2020, 2024)].year.value_counts().sort_index().to_dict()
o['ntis_secret_bz_nonnull'] = int((nt[nt.secret_flag].bz.str.len() > 0).sum())
o['ntis_rows_year'] = nt[nt.year.between(2021, 2024)].year.value_counts().sort_index().to_dict()
# 모집단
pop = pd.DataFrame({'cd': n_reg_le.index, 'n_reg': n_reg_le.values})
pop = pop.merge(ca10[['cd', 'bz', 'aname']], on='cd', how='left')
steps = {'c1_dom_reg_le2022': len(pop)}
pop = pop[pop.bz.notna()]; steps['c2_ca_bizno10'] = len(pop)
pop = pop.merge(st, left_on='bz', right_on='bizno', how='left')
pop = pop[pop.registered == True]; steps['c3_nts_registered'] = len(pop)
pop = pop[~(pop.end_dt.notna() & (pop.end_dt <= pd.Timestamp('2022-12-31')))]; steps['c4_not_closed_le2022'] = len(pop)
pop['first_ntis'] = pop.bz.map(firsty).astype('float64')
pop = pop[~(pop.first_ntis <= 2022)]; steps['c5_no_ntis_2010_2022'] = len(pop)
o['pop_steps'] = steps
pop['y'] = pop.first_ntis.isin([2023, 2024]).astype(int)
pop['y1'] = (pop.first_ntis == 2023).astype(int)
o['n'] = len(pop); o['pos'] = int(pop.y.sum()); o['pos_y1'] = int(pop.y1.sum()); o['n_bz'] = int(pop.bz.nunique())
pop[['cd', 'bz', 'aname', 'n_reg', 'first_ntis', 'y', 'y1', 'b_stt_cd', 'end_dt']].to_parquet(OUT + r"\cache\n7_pop_rebuild.parquet", index=False)
# 공식 대조
fr = pd.read_parquet(ROOT + r"\분석\모델경쟁_20260927\N7_특허기업_정부RnD첫수행\final_test_pred_r5.parquet")
fr['cd'] = fr.applicantCd.astype(str)
m = fr.merge(pop[['cd', 'bz', 'y', 'y1', 'first_ntis']], on='cd', how='outer', suffixes=('_off', '_rb'), indicator=True)
o['unit_merge'] = m._merge.value_counts().to_dict()
both = m[m._merge == 'both']
o['y_agree'] = float((both.y_off == both.y_rb).mean()); o['y1_agree'] = float((both.y1_off == both.y1_rb).mean())
o['bz_agree'] = float((both.bizno.astype(str) == both.bz).mean())
o['official_n'] = len(fr); o['official_pos'] = int(fr.y.sum())
m.to_parquet(OUT + r"\cache\n7_merge.parquet", index=False)
def _san(x):
    if isinstance(x, dict): return {str(k): _san(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [_san(v) for v in x]
    return x
o = _san(o)
json.dump(o, open(OUT + r"\cache\s1_out.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(o, ensure_ascii=False, indent=1, default=str))
