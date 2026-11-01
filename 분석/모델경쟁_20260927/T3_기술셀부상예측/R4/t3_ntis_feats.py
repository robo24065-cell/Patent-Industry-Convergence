# -*- coding: utf-8 -*-
"""T3 R4 — 셀(KSIC × IPC 메인그룹) 정부R&D 피처, 공개연도 변형(NTIS Open API 활용). 출력 R4/ntis_cell_pub.parquet (cell, t0, ...).
시점 규칙(PROTOCOL_R4 §3 + ntis_TESTS §6 T3 권고):
  gsh_pub / gsh_any : 공개연도 t0−3..t0−1(1년 뒤로 민 창) KR 문서 중 NTIS 성과 레코드 Year ≤ t0 로 연결되는 비율
                      (공공 = 대학·출연연·국공립 수행 / any = 전체). NTIS 성과가 2007년 출원부터라 분모는 출원연도 ≥ 2007 문서만,
                      KSIC 평균으로 수축(강도 10). KSIC 분모 < 30 이면 NaN(t0 ≤ 2008 은 사실상 NaN).
  f_grow            : 셀 배분 정부연구비 성장 log((A_{t0−1}+A_{t0−2})/(A_{t0−3}+A_{t0−4})),
                      A_y = Σ_분류 연구비(분류, y) × s(분류→셀), s 는 **성과 Year ≤ t0−2** 연결로만 만든 '분류의 KR 특허 성과 중 그 셀 비중'.
                      과제연도 y 연구비는 y+1 년부터 사용 → y ≤ t0−1. 배분표 재료 < 1000 건(t0 < 2009)이면 NaN, 배분 0 셀은 NaN.
  f_int             : log(A_{t0−1} / (셀 KR 공개 문서 t0−2..t0 + 1)) 집약도
셀 키는 T3 공개연도 패널과 같은 mgn 규칙(1라운드 build_panel.py 의 메인그룹 정규화)을 쓴다 — 적격 셀 일치율을 출력한다.
🔒 공개연도 ≤ 2018 문서만 읽고(t0 ≤ 2018 피처), 레이블은 계산하지 않는다.
"""
import os, pathlib, time
import numpy as np, pandas as pd, duckdb
R4 = pathlib.Path(__file__).resolve().parent; T3 = R4.parent; R3 = T3 / 'R3'
EXTD = T3.parent / '_shared' / 'ext'
T0S = list(range(2005, 2019))
GRP = {'대학': 'pub', '출연연구소': 'pub', '국공립연구소': 'pub', '대기업': 'firm', '중견기업': 'firm', '중소기업': 'firm'}
t = time.time()
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2")
con.execute(r"""create macro mgn(s) as (case
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{1,4}/') then substr(replace(s,' ',''),1,4)||' '||cast(cast(regexp_extract(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z]([0-9]{1,4})/',1) as int) as varchar)
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{3,6}$') then substr(replace(s,' ',''),1,4)||' '||cast(cast(substr(replace(s,' ',''),5,length(replace(s,' ',''))-6) as int) as varchar)
  else null end)""")
kr = con.execute(f"""select distinct ksic, mgn(ipc_raw) mg, year, appl_year, appl_no from read_parquet('{(R3/'cache_pub'/'base_rows.parquet').as_posix()}')
                     where cc='KR' and year between 2000 and 2018""").df()
kr = kr[kr.mg.notna()].copy(); kr['ksic'] = kr.ksic.astype(str); kr['cell'] = kr.ksic + '|' + kr.mg
cells_ref = pd.concat([pd.read_parquet(R3 / 'cache_pub' / f'feat_{y}.parquet', columns=['cell', 'tot_W']) for y in (2012, 2018)])
cells_ref = set(cells_ref[cells_ref.tot_W >= 50].cell)
print('KR 행', len(kr), '적격 셀 중 KR 문서 셀과 일치', round(len(cells_ref & set(kr.cell)) / len(cells_ref), 3), round(time.time() - t), flush=True)

lk = pd.read_parquet(EXTD / 'ntis_patent_link.parquet', columns=['appl_no', 'Year', 'PerformAgent', 'ProjectID'])
lk['yr'] = pd.to_numeric(lk.Year, errors='coerce'); lk['pg'] = lk.PerformAgent.astype(str).map(GRP).fillna('oth')
mn = lk.groupby(['appl_no', 'pg']).yr.min().unstack()
mn['any'] = mn.min(1)
kr = kr.merge(mn[['any', 'pub']].astype(float), left_on='appl_no', right_index=True, how='left')

pj = pd.read_parquet(EXTD / 'ntis_project_raw.parquet', columns=['pjt_no', 'year', 'cls_L_canon', 'gov_funds'])
pj['pjt_no'] = pj.pjt_no.astype(str); pj['cls'] = pj.cls_L_canon.astype(str)
pj = pj[~pj.cls.isin(['NA', '<NA>', 'nan', 'None', ''])]
pj['gov_funds'] = pd.to_numeric(pj.gov_funds, errors='coerce').fillna(0.0)
F = pj.groupby(['cls', 'year']).gov_funds.sum().unstack(fill_value=0.0)
cls_of = pj.drop_duplicates('pjt_no').set_index('pjt_no').cls
rp = pd.read_parquet(EXTD / 'ntis_rpatent_raw.parquet', columns=['Year', 'ProjectID', 'is_kr', 'IprType'])
rp = rp[(rp.is_kr == True) & (rp.IprType.astype(str) == '특허')]
rp = pd.DataFrame({'cls': rp.ProjectID.astype(str).map(cls_of).values, 'yr': pd.to_numeric(rp.Year, errors='coerce').values}).dropna()
lk['cls'] = lk.ProjectID.astype(str).map(cls_of)
cof = kr[['appl_no', 'cell']].drop_duplicates(); cof['m'] = cof.groupby('appl_no').cell.transform('size')
lkc = lk[lk.cls.notna()][['appl_no', 'yr', 'cls']].merge(cof, on='appl_no'); lkc['w'] = 1.0 / lkc.m
print('셀 연결 성과 링크행', len(lkc), round(time.time() - t), flush=True)

rows = []
for t0 in T0S:
    # --- 정부 기원 비중(공개연도 t0−3..t0−1, 출원 ≥2007, 레코드 Year ≤ t0)
    w = kr[(kr.year >= t0 - 3) & (kr.year <= t0 - 1) & (kr.appl_year >= 2007)].drop_duplicates(['cell', 'appl_no'])
    out = pd.DataFrame(index=pd.Index(sorted(kr.cell.unique()), name='cell'))
    out['ksic'] = [c.split('|')[0] for c in out.index]
    for g in ('pub', 'any'):
        w = w.assign(gv=(w[g] <= t0).astype(float))
        c_n = w.groupby('cell').size(); c_g = w.groupby('cell').gv.sum()
        k_n = w.groupby('ksic').size(); k_g = w.groupby('ksic').gv.sum()
        ks = (k_g / k_n).where(k_n >= 30)
        prior = out.ksic.map(ks).values
        n = c_n.reindex(out.index).fillna(0).values; gsum = c_g.reindex(out.index).fillna(0).values
        out[f'gsh_{g}'] = np.where(np.isnan(prior), np.nan, (gsum + 10 * np.nan_to_num(prior)) / (n + 10))
    # --- 셀 배분 연구비
    den = rp[rp.yr <= t0 - 2].groupby('cls').size()
    if den.sum() >= 1000 and all(y in F.columns for y in range(t0 - 4, t0)):
        s = lkc[lkc.yr <= t0 - 2].groupby(['cls', 'cell']).w.sum().unstack(fill_value=0.0)
        s = s.div(den.reindex(s.index), axis=0).fillna(0.0)
        Fy = F.reindex(s.index).fillna(0.0)
        A = {y: (Fy[y].values[:, None] * s.values).sum(0) for y in range(t0 - 4, t0)}
        A = {y: pd.Series(v, index=s.columns).reindex(out.index).fillna(0.0).values for y, v in A.items()}
        num = A[t0 - 1] + A[t0 - 2]; dnm = A[t0 - 3] + A[t0 - 4]
        ok = (num > 0) & (dnm > 0)
        out['f_grow'] = np.where(ok, np.log(np.where(ok, num, 1) / np.where(ok, dnm, 1)), np.nan)
        kw = kr[(kr.year >= t0 - 2) & (kr.year <= t0)].groupby('cell').appl_no.nunique().reindex(out.index).fillna(0).values
        out['f_int'] = np.where(A[t0 - 1] > 0, np.log(np.where(A[t0 - 1] > 0, A[t0 - 1], 1) / 1e6 / (kw + 1)), np.nan)
    else:
        out['f_grow'] = np.nan; out['f_int'] = np.nan
    out['t0'] = t0
    rows.append(out.reset_index())
    e = out[out.index.isin(cells_ref)]
    print(t0, '적격셀 비결측 gsh_pub', round(float(e.gsh_pub.notna().mean()), 3), 'f_grow', round(float(e.f_grow.notna().mean()), 3),
          '평균 gsh_any', round(float(e.gsh_any.mean()), 4) if e.gsh_any.notna().any() else None, flush=True)
R = pd.concat(rows, ignore_index=True).drop(columns='ksic')
R.to_parquet(R4 / 'ntis_cell_pub.parquet', index=False)
print('저장', len(R), round(time.time() - t), flush=True)
