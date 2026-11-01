# -*- coding: utf-8 -*-
"""C18-T1 verify v2: 원인 판정 반박 시도(대안 가설 직접 검정) + C17-T2 413건 정의 차이 분해.
 R-a 수집기 결함: 공란 문서가 수집 대상(pending)에 한 번이라도 들어갔는가, registry status 분포, 공백 달 RF/KR.
 R-b 실제 감소: 합집합/공식 비(RLshare)·일련번호 추정 대비.
 R-c 제공 스냅샷 시점(등록 전에 찍힌 사본) 가설: 같은 등록일 안에 번호 있는 문서와 공란 문서가 섞이는가, 공란 문서 등록일이 보유 문서보다 늦은가.
 R-d (사후 탐색·판정 외) 권리 이전 연관: 원출원인명이 2026 권리자 정보에 남아 있는 비율 M 대 H.
읽기 전용. 쓰기: verify/v2_out.json"""
import sys, re, pathlib, json, sqlite3, time
import numpy as np, pandas as pd
sys.dont_write_bytecode = True
t0 = time.time()
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
H = B/'분석/검증루프_20261012/C18/C18-T1'; C = H/'cache'
ro = lambda p: sqlite3.connect(pathlib.Path(p).as_uri() + '?mode=ro', uri=True)
END = '20250822'; o = {}
T = pd.read_csv(H/'monthly_capture.csv', dtype={'ym': str}).set_index('ym'); G = set(T.index[T.gap])
KP = pd.read_parquet(B/'공모전데이터/작업데이터/kr_patents.parquet', columns=['출원번호', '등록번호', '등록일자', '출원인명'])
KP.columns = ['appl_no', 'rgst_no', 'rd', 'nm']
for c in KP: KP[c] = KP[c].astype(str).str.strip()
blank = KP[~KP.rgst_no.str.match(r'^\d{13}$')]
o['blank_rd_nonempty'] = int(blank.rd.str.match(r'^\d{8}$').sum())
gc = ro(B/'수집파이프라인/registry.sqlite')
pend = pd.read_sql('select rgst_no, appl_no from pending', gc)
o['registry_status'] = gc.execute('select status, count(*) from registry group by 1').fetchall()
o['registry_tables'] = [r[0] for r in gc.execute("select name from sqlite_master where type='table'")]
gc.close()
o['R_a_blank_appl_in_pending'] = int(blank.appl_no.isin(set(pend.appl_no.astype(str))).sum())
o['R_a_gap_RF_over_KR'] = T.loc[sorted(G), 'RF_over_KR'].round(6).to_dict()
# 옛 스냅샷의 pending에도 공란 문서가 있었나
for f in sorted((B/'공모전데이터/작업데이터').glob('registry_snapshot_2026081*.sqlite')):
    c = ro(f); p = set(r[0] for r in c.execute('select appl_no from pending')); c.close()
    o[f'R_a_blank_in_pending_{f.stem[-8:]}'] = int(blank.appl_no.isin(p).sum())
# R-b
o['R_b_RLshare_gap'] = [float(T.loc[sorted(G), 'RLshare_m'].min()), float(T.loc[sorted(G), 'RLshare_m'].max())]
o['R_b_RLshare_2019_2023_10'] = [float(T.loc['201901':'202310', 'RLshare_m'].min()), float(T.loc['201901':'202310', 'RLshare_m'].max())]
o['R_b_union_over_O_by_year'] = {y: float(T.loc[[i for i in T.index if i.startswith(y)], 'union'].sum() / T.loc[[i for i in T.index if i.startswith(y)], 'O_pat'].sum()) for y in ['2022', '2023', '2024']}
o['R_b_union_over_O_2025_jan_jul'] = float(T.loc['202501':'202507', 'union'].sum() / T.loc['202501':'202507', 'O_pat'].sum())
# R-c: 같은 등록일 혼재
M = pd.read_parquet(C/'s4_missing_docs.parquet')
rc = ro(B/'수집파이프라인/registry_full.sqlite')
RF = pd.read_sql("select rgstDate rd from patent where substr(rgst_no,1,2)='10' and rgstDate>='20231101'", rc); rc.close()
dm = M.groupby('rd_rl').size(); dh = RF.groupby('rd').size()
days = sorted(set(dm.index) | set(dh.index))
both = [d for d in days if d in dm.index and d in dh.index]
o['R_c_days_with_missing'] = int(len(dm)); o['R_c_days_with_missing_and_held'] = int(len(both))
o['R_c_missing_on_days_without_held'] = int(sum(dm[d] for d in dm.index if d not in dh.index))
o['R_c_last_reg_day_held'] = RF.rd.max(); o['R_c_last_reg_day_missing'] = M.rd_rl.max()
gm = M[M.inG]; o['R_c_median_rd_missing_G'] = sorted(gm.rd_rl)[len(gm)//2] if len(gm) else None
# R-d (사후): 원출원인명이 권리목록 권리자 정보(hinfo)·출원인 정보(ainfo)에 남아 있나
STRIP = re.compile(r"주식회사|\(주\)|㈜|유한회사|유한책임회사|재단법인|사단법인|학교법인|가부시키가이샤|가부시끼가이샤|코포레이션|인코포레이티드|리미티드|컴퍼니|[\s\.\,\-\(\)\'\"·&]")
norm = lambda x: STRIP.sub('', str(x)).lower()
rl = pd.read_parquet(B/'분석/모델경쟁_20260927/T1_산업대기실/cache/rl_pat.parquet', columns=['rgst_no', 'rgst_date', 'appl_no', 'ainfo', 'hinfo'])
rl = rl[rl.rgst_date.str[:6].isin(G) & (rl.rgst_date <= END)].drop_duplicates('appl_no')
nm = dict(zip(KP.appl_no, KP.nm))
rl['grp'] = np.where(rl.appl_no.isin(set(M.appl_no)), 'M', 'H')
rl = rl[rl.appl_no.isin(set(KP.appl_no))]
def keep(row_nm, info):
    n = norm(str(row_nm).split('|')[0]); return len(n) >= 2 and n in norm(info)
rl['orig_in_holder'] = [keep(nm.get(a, ''), h) for a, h in zip(rl.appl_no, rl.hinfo)]
rl['orig_in_applicant'] = [keep(nm.get(a, ''), h) for a, h in zip(rl.appl_no, rl.ainfo)]
o['R_d_post_hoc'] = rl.groupby('grp')[['orig_in_holder', 'orig_in_applicant']].mean().round(4).to_dict('index')
o['R_d_n'] = rl.grp.value_counts().to_dict()
# C17-T2 413 분해: C16 alias_docs P-미등록 문서의 권리목록 적중(모든 종류 등록번호) 중 등록일 ≤ END
try:
    Dd = pd.read_parquet(B/'분석/검증루프_20261012/C16/C16-T1/cache/alias_docs.parquet'); Dd['appl_no'] = Dd.appl_no.astype(str)
    PU = set(Dd[(Dd.grp == 'P') & (~Dd.is_reg)].appl_no)
    bc = ro(B/'수집파이프라인/business_rights.sqlite')
    q = ','.join("'" + a + "'" for a in PU)
    h = pd.read_sql(f"select rgst_no, rgst_date, appl_no from right_list where appl_no in ({q})", bc); bc.close()
    hh = h.sort_values('rgst_date').drop_duplicates('appl_no')
    e = hh[hh.rgst_date <= END]
    o['C17_hits_all_types'] = int(len(hh)); o['C17_early_all_types'] = int(len(e))
    o['C17_early_by_prefix'] = e.rgst_no.str[:2].value_counts().to_dict()
    o['C17_hits_by_prefix'] = hh.rgst_no.str[:2].value_counts().to_dict()
except Exception as ex:
    o['C17_err'] = repr(ex)
o['secs'] = round(time.time() - t0)
json.dump(o, open(H/'verify/v2_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(o, ensure_ascii=False, default=str)[:3500]); print('done', o['secs'])
