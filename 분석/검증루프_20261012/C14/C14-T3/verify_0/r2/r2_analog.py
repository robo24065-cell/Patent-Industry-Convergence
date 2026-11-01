# -*- coding: utf-8 -*-
"""verify_0/r2: 반박 재계산 2 — 별칭 규칙을 쓰지 않는 생존 정의(F 조회 목록에 (F, 등록번호) 권리자란 행 존재)와
 등록원부 최종권리자(finalOwnerYn) 기반 이전 판정(원 출원인 명칭 정확 일치, 다른 정규화), 진입 상관 재계산. 원천 sqlite mode=ro, 읽기만."""
import sqlite3, pathlib, json, time
import numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
t0 = time.time(); O = {}
br = sqlite3.connect((B/'수집파이프라인/business_rights.sqlite').as_uri()+'?mode=ro', uri=True)
rg = sqlite3.connect((B/'수집파이프라인/registry_full.sqlite').as_uri()+'?mode=ro', uri=True)
aq = pd.read_sql("select applicant_cd, applicant_name, status from applicant_query", br)
OK = set(aq[aq.status == 'OK'].applicant_cd)
kr = pd.read_parquet(B/'공모전데이터/작업데이터/kr_registered.parquet')
cols = list(kr.columns); print([ (i,c) for i,c in enumerate(cols)])
kr = kr.rename(columns={'등록번호':'rn','등록일자':'rd','출원연도':'ay'})
kr['rn'] = kr.rn.astype(str); kr = kr[kr.rn.str[:2] == '10'].drop_duplicates('rn')
ksic_col = [c for c in cols if '산업' in c and ('코드' in c or 'KSIC' in c.upper())]
O['ksic_col'] = ksic_col
kr['ksic'] = kr[ksic_col[0]].astype(str) if ksic_col else 'NA'
ap = pd.read_sql("select rgst_no, applicantCd, applicantName from applicant where rgst_no like '10%'", rg)
A = ap[ap.applicantCd.isin(OK)][['rgst_no','applicantCd']].drop_duplicates()
A = A.merge(kr[['rn','rd','ay','ksic']], left_on='rgst_no', right_on='rn')
A = A[pd.to_numeric(A.ay, errors='coerce') >= 2000].copy()
A['ry'] = A.rd.astype(str).str[:4].astype(int)
print('A', len(A), round(time.time()-t0), flush=True)
rl = pd.read_sql("select applicant_cd, rgst_no, json_extract(raw,'$.pertDiv') pert from right_list where rgst_no like '10%'", br)
O['rl_rows_10'] = int(len(rl))
k_any = set(zip(rl.applicant_cd, rl.rgst_no))
r2 = rl[rl.pert == '권리자란']; k_rh = set(zip(r2.applicant_cd, r2.rgst_no))
other_rh = r2.groupby('rgst_no').applicant_cd.apply(set).to_dict()
del rl
A['s_rh'] = [(c, r) in k_rh for c, r in zip(A.applicantCd.values, A.rgst_no.values)]
A['s_any'] = [(c, r) in k_any for c, r in zip(A.applicantCd.values, A.rgst_no.values)]
A['other_holder'] = [len(other_rh.get(r, set()) - {c}) > 0 for c, r in zip(A.applicantCd.values, A.rgst_no.values)]
a = A[A.ry <= 2019].copy()
O['n_le2019'] = int(len(a))
O['dropout_rh_noalias'] = float(1 - a.s_rh.mean()); O['dropout_any_pert_noalias'] = float(1 - a.s_any.mean())
yg = pd.cut(a.ry, [1999, 2004, 2009, 2014, 2019], labels=['2000–04','2005–09','2010–14','2015–19'])
O['by_group_rh_noalias'] = {str(k): round(float(1-v), 5) for k, v in a.groupby(yg, observed=False).s_rh.mean().items()}
ks = a.groupby('ksic').agg(n=('s_rh','size'), d=('s_rh', lambda s: 1-s.mean()))
O['ksic_n'] = int(len(ks)); O['ksic_range_all'] = [float(ks.d.min()), float(ks.d.max())]
O['ksic_range_n500'] = [float(ks[ks.n>=500].d.min()), float(ks[ks.n>=500].d.max())]
# 이전 판정(독립): 등록원부 owner 최종권리자(finalOwnerYn=='Y', 없으면 최대 seq) 명칭 집합 vs 이 특허의 F 원출원 명칭(정확 일치, 공백·(주)·주식회사만 제거)
def nm(s): return str(s).replace('주식회사','').replace('(주)','').replace('㈜','').replace(' ','').lower()
ow = pd.read_sql("select rgst_no, seq, finalOwnerYn, ownerName from owner where rgst_no like '10%'", rg)
O['owner_finalYn_vals'] = ow.finalOwnerYn.value_counts(dropna=False).head(5).to_dict()
fy = ow[ow.finalOwnerYn == 'Y']
mx = ow.groupby('rgst_no').seq.transform('max')
fb = ow[(ow.seq == mx) & ~ow.rgst_no.isin(set(fy.rgst_no))]
fin = pd.concat([fy, fb]).groupby('rgst_no').ownerName.apply(lambda s: set(nm(x) for x in s)).to_dict()
nchg = ow.groupby('rgst_no').seq.nunique().to_dict()
apn = ap.assign(n=ap.applicantName.map(nm)).groupby(['rgst_no','applicantCd']).n.apply(set).to_dict()
qn = {c: nm(n) for c, n in zip(aq.applicant_cd, aq.applicant_name)}
def retained(c, r):
    f = fin.get(r)
    if f is None: return np.nan
    names = set(apn.get((r, c), set())) | {qn.get(c, '')}
    names = {x for x in names if len(x) >= 2}
    return any(any(x in y or y in x for y in f if len(y) >= 2) for x in names)
a['ret'] = [retained(c, r) for c, r in zip(a.applicantCd.values, a.rgst_no.values)]
O['owner_missing'] = float(a.ret.isna().mean())
b = a.dropna(subset=['ret']); b['ret'] = b.ret.astype(bool)
O['transferred_share'] = float((~b.ret).mean())
O['dropout_if_retained'] = float(1 - b[b.ret].s_rh.mean()); O['dropout_if_transferred'] = float(1 - b[~b.ret].s_rh.mean())
# 다른 판정: 등록원부 소유 seq 1개(변동 없음) vs 2개 이상
a['nseq'] = a.rgst_no.map(nchg)
O['dropout_no_owner_change'] = float(1 - a[a.nseq == 1].s_rh.mean()); O['dropout_owner_change'] = float(1 - a[a.nseq > 1].s_rh.mean())
O['share_no_owner_change'] = float((a.nseq == 1).mean())
# 누락 행 중 다른 조회법인 권리자란에 나타나는 비율(넘어간 곳이 확인되는 비율)
O['dropped_found_in_other_firm_rh'] = float(a[~a.s_rh].other_holder.mean()); O['kept_found_in_other_firm_rh'] = float(a[a.s_rh].other_holder.mean())
print('transfer', round(time.time()-t0), flush=True)
# 진입 상관
T1 = B/'분석/모델경쟁_20260927/T1_산업대기실'
te = pd.read_parquet(T1/'R3/cache_final/o2019/prep_te.parquet', columns=['firm','k','prior_any'])
L = pd.read_parquet(B/'분석/검증루프_20261012/C10/C10-T1/cache/final_labels.parquet', columns=['firm','k','y_u','prior_any'])
assert (L.firm.values == te.firm.values).all() and (L.k.values == te.k.values).all()
te['y'] = L.y_u.values; te = te[~te.prior_any.values]
ent = te.groupby('firm').y.max()
fd = a.groupby('applicantCd').s_rh.agg(lambda s: 1 - s.mean())
ff = pd.DataFrame({'ent': ent}); ff['fd'] = ff.index.map(fd).astype(float)
O['entry_firms'] = int(len(ff)); O['entry_firms_with_fd'] = int(ff.fd.notna().sum()); O['entry_share'] = float(ff.ent.mean())
e1 = ff[ff.ent == 1].fd.dropna(); e0 = ff[ff.ent == 0].fd.dropna()
sd = np.sqrt((e1.var(ddof=0) + e0.var(ddof=0)) / 2)
O['fd_entry'] = [float(e1.mean()), float(e0.mean())]; O['smd'] = float((e1.mean() - e0.mean()) / sd)
O['fd_entry_median'] = [float(e1.median()), float(e0.median())]
# 패턴 수준(특허 가중): 진입 법인 특허 누락률
a['ent'] = a.applicantCd.map(ent)
O['patent_weighted_dropout_by_entry'] = {str(k): float(1 - v) for k, v in a.dropna(subset=['ent']).groupby('ent').s_rh.mean().items()}
# 등록연도 구성 통제: 연도군별 진입/비진입 법인 특허 누락률
a['yg'] = yg
O['by_yg_entry'] = {f'{g}|{int(e)}': round(float(1 - v), 4) for (g, e), v in a.dropna(subset=['ent']).groupby(['yg','ent'], observed=True).s_rh.mean().items()}
O['sec'] = round(time.time() - t0)
json.dump(O, open(HERE/'r2_analog_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(O, ensure_ascii=False, indent=1, default=str))
