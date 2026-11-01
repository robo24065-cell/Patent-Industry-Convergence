# C03-T1 반박검증 v1: 공란 등록 분포(치우침)와 위험집합 밖 분해를 원천에서 다시 계산
import sys, json, re, pathlib, time
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전")
T1 = ROOT/'분석/모델경쟁_20260927/T1_산업대기실'; TC = T1/'cache'; R3 = T1/'R3'
W = ROOT/'분석/검증루프_20261012/C03/C03-T1'; V = W/'verify_1'
DATA = ROOT/'공모전데이터/작업데이터'; EXTD = ROOT/'분석/모델경쟁_20260927/_shared/ext'
END = '20250822'; out = {}
# 1) 공란 등록 독립 재구성
k = pd.read_parquet(DATA/'kr_patents.parquet', columns=['출원번호','출원연도','등록번호','세세분류코드'])
k.columns = ['appl_no','ay','rgst_no','ksic']; k['ay'] = k.ay.astype(int); k['ksic'] = k.ksic.astype(str)
k['blank'] = k.rgst_no.fillna('').astype(str).str.strip().str.len() <= 3
rl = pd.read_parquet(TC/'rl_pat.parquet', columns=['applicant_cd','rgst_no','rgst_date','appl_no','pert','ainfo'])
rlE = rl[rl.rgst_date <= END]
bw = k[k.blank & k.ay.between(2020,2023) & (k.appl_no.str[:2]=='10') & k.appl_no.isin(set(rlE.appl_no))]
out['blank_patents_by_ay'] = bw.groupby('ay').size().to_dict(); out['blank_patents_total'] = len(bw)
out['kr_patents_dup_applno'] = int(k.appl_no.duplicated().sum())
# 공란 등록 중 권리목록 등록번호가 '10'으로 시작하는가
rr = rlE[rlE.appl_no.isin(set(bw.appl_no))]
out['blank_rl_rgst_prefix10_share'] = float(rr.rgst_no.astype(str).str[:2].eq('10').mean())
# 2) s2 산출 blank_reg_decomp 대조
d = pd.read_csv(W/'blank_reg_decomp.csv', dtype={'applicant_cd':str,'appl_no':str,'ksic':str})
out['decomp_rows'] = len(d); out['decomp_patents'] = int(d.appl_no.nunique())
out['decomp_patents_not_in_bw'] = int(len(set(d.appl_no) - set(bw.appl_no))); out['bw_not_in_decomp'] = int(len(set(bw.appl_no)-set(d.appl_no)))
out['decomp_self_pairs'] = int(d['self'].sum())
A = d[d['self']].copy()
# 3) 비교군: ins_firm 출원 2020-23
ins = pd.read_parquet(TC/'ins_firm.parquet', columns=['rgst_no','applicant_cd','ksic','ay','rgst_date'])
fset = set(pd.read_parquet(W/'cache/alias_src.parquet', columns=['applicant_cd']).applicant_cd)
B = ins[ins.ay.between(2020,2023)].copy()
out['B_rows'] = len(B); out['B_dup_rgst_firm'] = int(B.duplicated(['rgst_no','applicant_cd']).sum())
out['B_firms_not_in_fset'] = int((~B.applicant_cd.isin(fset)).sum())
out['B_rgst_after_END'] = int((B.rgst_date.astype(str) > END).sum()); out['B_rgst_max'] = str(B.rgst_date.max())
out['A_firms_not_in_ins_firm_ever'] = int((~A.applicant_cd.isin(set(ins.applicant_cd))).sum())
fm = pd.read_parquet(TC/'firm_meta.parquet', columns=['applicant_cd','tp']).set_index('applicant_cd').tp
fsz = ins[ins.ay <= 2019].groupby('applicant_cd').size()
def at(x):
    x = x.copy(); x['natl'] = x.applicant_cd.map(fm).fillna('결측').map(lambda v: v if v in ('국내법인','외국법인') else '기타')
    s = x.applicant_cd.map(fsz).fillna(0).values; x['size'] = pd.cut(s, [-1,2,20,200,1e9], labels=['≤2건','3–20','21–200','200+']).astype(str); return x
A = at(A); B = at(B); Bc = B[B.rgst_date.astype(str) <= END]
def share(Aa, Bb, dim):
    ca, cb = Aa[dim].value_counts(), Bb[dim].value_counts(); tot = len(Aa)/(len(Aa)+len(Bb))
    return {str(v): [int(ca.get(v,0)), int(cb.get(v,0)), round(ca.get(v,0)/(ca.get(v,0)+cb.get(v,0)),4), round(ca.get(v,0)/(ca.get(v,0)+cb.get(v,0))/tot,3)] for v in sorted(set(ca.index)|set(cb.index), key=str)}, tot
for nm, Bb in (('B_all', B), ('B_capEND', Bc)):
    r = {}
    for dim in ('ay','natl','size'): r[dim], tot = share(A, Bb, dim)
    ks, tot = share(A, Bb, 'ksic'); r['ksic_top'] = dict(sorted(ks.items(), key=lambda kv: -kv[1][2])[:6]); r['overall'] = tot
    out[nm] = r
# 4) 등록월 통제: 같은 등록월(2023-11 이후) 비교 — 공란이 '외국'에 고유한지, 등록시점 탓인지
rmon = rlE.drop_duplicates('appl_no').set_index('appl_no').rgst_date.astype(str).str[:6]
A['rm'] = A.appl_no.map(rmon); Bc = Bc.copy(); Bc['rm'] = Bc.rgst_date.astype(str).str[:6]
out['A_rm_min'] = str(A.rm.min()); 
blank_months = sorted(A.rm.value_counts()[lambda s: s >= 50].index)
out['blank_months'] = blank_months
Bm = Bc[Bc.rm.isin(blank_months)]; Am = A[A.rm.isin(blank_months)]
out['same_month_compare'] = {dim: share(Am, Bm, dim)[0] for dim in ('natl','size')}
out['same_month_overall'] = len(Am)/(len(Am)+len(Bm))
# 5) 위험집합 밖 분해
te = pd.read_parquet(R3/'cache_final/o2019/prep_te.parquet', columns=['firm','k'])
panel_firms = set(te.firm)
K = sorted(pd.read_parquet(TC/'ins_pat.parquet', columns=['ksic']).ksic.astype(str).unique())
o = A[A.cat=='3a_위험집합밖'].copy()
out['out_pairs_self'] = len(o)
held_ay = set(zip(ins[ins.ay<=2019].applicant_cd, ins[ins.ay<=2019].ksic.astype(str)))
held_rd = set(zip(ins[ins.rgst_date.astype(str)<='20191231'].applicant_cd, ins[ins.rgst_date.astype(str)<='20191231'].ksic.astype(str)))
o['held_ay'] = [(a,b) in held_ay for a,b in zip(o.applicant_cd, o.ksic)]
o['held_rd'] = [(a,b) in held_rd for a,b in zip(o.applicant_cd, o.ksic)]
o['inpanel'] = o.applicant_cd.isin(panel_firms)
out['out_held_ay'] = int(o.held_ay.sum()); out['out_held_rd'] = int(o.held_rd.sum()); out['out_firm_not_in_panel'] = int((~o.inpanel).sum())
out['out_inpanel_not_held_ay'] = int((o.inpanel & ~o.held_ay).sum())
out['out_notpanel_and_held'] = int((~o.inpanel & o.held_ay).sum())
# 서브그룹별 위험집합 밖 비율
A['out'] = A.cat.isin(['3a_위험집합밖','3b_prior_any제외'])
grp = {}
grp['natl'] = A.groupby('natl').out.agg(['size','mean']).round(4).to_dict('index')
grp['size'] = A.groupby('size').out.agg(['size','mean']).round(4).to_dict('index')
A['k4'] = A.ksic.isin(['20491','26212','26211','26111'])
grp['ksic4'] = A.groupby('k4').out.agg(['size','mean']).round(4).to_dict('index')
grp['overall'] = float(A.out.mean())
out['out_share_by_group'] = grp
# 새 양성 실패 사유
npz = A[A.cat=='6_새양성']; out['newpos_pairs_rows'] = len(npz); out['newpos_pubmatch'] = npz.pubmatch.value_counts().to_dict()
out['newpos_unique_pairs'] = int(npz[['applicant_cd','k']].drop_duplicates().shape[0])
# 위험집합 안(4,5,6) 공개매칭
ev = A[A.cat.str[0].isin(['4','5','6'])]; out['in_eval_pubmatch'] = ev.pubmatch.value_counts().to_dict(); out['in_eval_n'] = len(ev)
json.dump(out, open(V/'v1_out.json','w',encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
