# -*- coding: utf-8 -*-
"""Q0(레이블·점수 보지 않음, PREREG 고정 전 계산량·정의 계획용): 신규진입자 법인의 원점 가시 증거 분류, RC:001 법인 규모, N3 표집틀 크기.
 → cache/q0_pop.json, cache/newc_evidence.parquet(법인별 증거 플래그)"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard17 as G
from c17common import *
import time
t0 = time.time(); out = {}
te = pd.read_parquet(FC/'prep_te.parquet', columns=['firm', 'k', 'newc', 'prior_any'])
kept = ~te.prior_any.values
nf = sorted(te.firm[te.newc.values].unique()); NF = set(nf)
out['newc_firms_all'] = len(nf); out['newc_pairs_scored'] = int((te.newc.values & kept).sum()); out['pairs_scored'] = int(kept.sum())
out['newc_firms_scored'] = int(te.firm[te.newc.values & kept].nunique())
# 등록원부 원출원인 코드(범위안 등록) — applicant_query 수집 대상의 정의
app = pd.read_parquet(CACHE/'ins_applicant.parquet', columns=['rgst_no', 'applicantCd']).drop_duplicates(); app['rgst_no'] = app.rgst_no.astype(str)
pat = pd.read_parquet(CACHE/'ins_pat.parquet', columns=['rgst_no', 'ay', 'appl_date', 'rgst_date', 'ksic']).drop_duplicates('rgst_no'); pat['rgst_no'] = pat.rgst_no.astype(str)
x = app.merge(pat, on='rgst_no', how='left'); x['rg'] = s8(x.rgst_date.values)
xn = x[x.applicantCd.isin(NF)]
g = xn.groupby('applicantCd').agg(first_ay=('ay', 'min'), first_rgst=('rg', 'min'), n_ins_all=('rgst_no', 'nunique'))
out['newc_registry_first_ay_dist'] = {int(k): int(v) for k, v in g.first_ay.value_counts().sort_index().items()}
out['newc_registry_first_rgst_year_dist'] = {int(k): int(v) for k, v in (g.first_rgst // 10000).value_counts().sort_index().items()}
out['newc_registry_rgst_le_origin_firms'] = int((g.first_rgst <= CAPV).sum())
out['newc_registry_ay_lt2000_firms'] = int((g.first_ay < 2000).sum())
out['newc_firms_without_registry_row'] = int(len(NF - set(g.index)))
# 제공 데이터(kr_patents 이름매칭 app_pub) 원점 가시 출원
a = pd.read_parquet(CACHE/'app_pub.parquet'); a['v'] = np.minimum(s8(a.pub_date.values), s8(a.rgst_date.values))
an = a[a.applicant_cd.isin(NF)]
E_pub = set(an[(an.ay <= CUT) & (an.v <= CAPV)].applicant_cd)
E_pub_unreg = set(an[(an.ay <= CUT) & (an.v <= CAPV) & ~an.reg.astype(bool)].applicant_cd)
E_pub_appl = set(an[(an.ay <= CUT)].applicant_cd)              # 원점 전 출원(가시 여부 무관)
# 권리목록(2026 스냅샷) 원점 전 등록 기록
rl = pd.read_parquet(CACHE/'rl_pat.parquet', columns=['applicant_cd', 'rgst_no', 'rgst_date', 'pert'])
rl = rl[rl.applicant_cd.isin(NF) & (rl.pert == '권리자란')]; rl['rg'] = s8(rl.rgst_date.values); insset = set(pat.rgst_no)
rl['ins'] = rl.rgst_no.astype(str).isin(insset)
E_rl_ins = set(rl[rl.ins & (rl.rg <= CAPV)].applicant_cd)
E_rl_any = set(rl[rl.rg <= CAPV].applicant_cd)
oos = pd.read_parquet(CACHE/'oos.parquet', columns=['applicant_cd', 'rgst_no', 'ay', 'rgst_date', 'self'])
oo = oos[oos.self & (oos.ay >= 2000) & (oos.ay <= CUT) & oos.applicant_cd.isin(NF)].copy(); oo['rg'] = s8(oo.rgst_date.values)
E_oos_vis = set(oo[oo.rg <= CAPV].applicant_cd)
cnt_vis = oo[oo.rg <= CAPV].drop_duplicates(['applicant_cd', 'rgst_no']).groupby('applicant_cd').size()
E_oos_vis3 = set(cnt_vis[cnt_vis >= 3].index)
E_reg = set(g.index[g.first_rgst <= CAPV])
F = pd.DataFrame(index=pd.Index(nf, name='firm'))
for nm, S in (('E_pub', E_pub), ('E_pub_unreg', E_pub_unreg), ('E_pub_appl', E_pub_appl), ('E_reg', E_reg), ('E_rl_ins', E_rl_ins), ('E_rl_any', E_rl_any), ('E_oos_vis', E_oos_vis), ('E_oos_vis3', E_oos_vis3)):
    F[nm] = F.index.isin(S)
F['first_ins_ay'] = g.first_ay.reindex(F.index).values; F['first_ins_rgst'] = g.first_rgst.reindex(F.index).values
F['N1_primary'] = F.E_pub | F.E_reg | F.E_rl_ins
F['N1_narrow'] = F.E_pub
F['N1_wide'] = F.N1_primary | F.E_rl_any | F.E_oos_vis
F.reset_index().to_parquet(CC/'newc_evidence.parquet', index=False)
out['newc_evidence_firms'] = {c: int(F[c].sum()) for c in F.columns if c.startswith(('E_', 'N1_'))}
pn = te[te.newc.values & kept]; pf = pn.firm.map(F.N1_primary)
out['newc_pairs_scored_N1_primary_keep'] = int(pf.sum()); out['newc_pairs_scored_N1_narrow_keep'] = int(pn.firm.map(F.N1_narrow).sum()); out['newc_pairs_scored_N1_wide_keep'] = int(pn.firm.map(F.N1_wide).sum())
# RC:001 법인 코드(1·2·3·5)로 ≥3건(범위안 등록, 출원 2000–2019) — 범위밖은 권리목록이 없어 셀 수 없음(하한)
aq = pd.read_parquet(CACHE/'firms.parquet', columns=['applicant_cd', 'applicant_name', 'status'])
rc = aq[(aq.status == 'RC:001') & aq.applicant_cd.str[0].isin(list('1235'))]; RC = set(rc.applicant_cd)
out['rc_corp_codes'] = len(RC); out['aq_status'] = aq.status.value_counts().to_dict()
xr = x[x.applicantCd.isin(RC) & (x.ay >= 2000)]
xp = xr[xr.ay <= CUT].drop_duplicates(['applicantCd', 'rgst_no']); c3 = xp.groupby('applicantCd').size(); RC3 = set(c3[c3 >= 3].index)
held = xp[xp.applicantCd.isin(RC3)].drop_duplicates(['applicantCd', 'ksic']).groupby('applicantCd').size()
out['rc_firms_ge3'] = len(RC3); out['rc_pairs_risk'] = int(len(RC3) * 50 - held.sum())
out['rc_firms_ge1_le2019'] = int(c3.size); out['rc_prefix_ge3'] = pd.Series([a[0] for a in RC3]).value_counts().to_dict()
pd.DataFrame(dict(firm=sorted(RC3))).to_parquet(CC/'rc_firms.parquet', index=False)
# N3 표집틀: CA(DART corp_applicant) 법인 코드 중 applicant_query에 없는 코드(=범위안 등록 원출원인 이력 없음) + 원점 존재 증거
ca = pd.read_parquet(CA); ca.columns = ['applicant_cd', 'name', 'jurir_raw', 'bizr_raw', 'nm', 'jurir', 'bizr']
ca = ca[ca.applicant_cd.astype(str).str[0].isin(list('1235'))]
AQ = set(aq.applicant_cd); fr0 = ca[~ca.applicant_cd.isin(AQ)].drop_duplicates('applicant_cd')
P = pd.read_parquet(EXTD/'firm_nps_firm_panel.parquet', columns=['applicantCd', 'snap', 'n_emp'])
nps19 = set(P[(P.snap == '2019-12') & (P.n_emp.fillna(0) > 0)].applicantCd)
out['ca_corp_codes'] = int(ca.applicant_cd.nunique()); out['ca_not_in_aq'] = int(len(fr0)); out['ca_not_in_aq_nps2019'] = int(fr0.applicant_cd.isin(nps19).sum())
out['nps_snaps'] = sorted(P.snap.unique().tolist())[:3] + ['…'] + sorted(P.snap.unique().tolist())[-3:]
out['secs'] = round(time.time() - t0); out['mem_peak'] = G.rss_peak()
jdump(out, CC/'q0_pop.json'); print(json.dumps(out, ensure_ascii=False, indent=1))
