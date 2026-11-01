# -*- coding: utf-8 -*-
"""N1 증거(PREREG §1)를 공식 신규진입자 ∪ S1 가시 신규진입자(C16 pop.pkl: Fvis − vis_any) 전원에 대해 같은 규칙으로 계산.
 → cache/evid_all.parquet(firm, is_newc_off, is_newc_s1, E_*, N1_*), cache/n1_prep.json. q0/q0b 결과와 공식 신규진입자 부분이 같은지 대조."""
import sys, pathlib, re, pickle; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard17 as G
from c17common import *
te = pd.read_parquet(FC/'prep_te.parquet', columns=['firm', 'newc'])
NO = set(te.firm[te.newc.values]); PP = pickle.load(open(C16C/'build'/'pop.pkl', 'rb'))
NS = set(PP['Fvis']) - set(PP['vis_any_firms']); NF = NO | NS
STRIP = re.compile(r"주식회사|\(주\)|㈜|유한회사|유한책임회사|재단법인|사단법인|학교법인|가부시키가이샤|가부시끼가이샤|가부시기가이샤|가부시키카이샤|코포레이션|인코포레이티드|리미티드|컴퍼니|[\s\.\,\-\(\)\'\"·&]")
norm = lambda s: STRIP.sub('', str(s)).lower()
app = pd.read_parquet(CACHE/'ins_applicant.parquet', columns=['rgst_no', 'applicantCd', 'applicantName']); app['rgst_no'] = app.rgst_no.astype(str)
pat = pd.read_parquet(CACHE/'ins_pat.parquet', columns=['rgst_no', 'ay', 'rgst_date']).drop_duplicates('rgst_no'); pat['rgst_no'] = pat.rgst_no.astype(str)
x = app[app.applicantCd.isin(NF)].merge(pat, on='rgst_no', how='left'); x['rg'] = s8(x.rgst_date.values)
E_reg = set(x[x.rg <= CAPV].applicantCd)
a = pd.read_parquet(CACHE/'app_pub.parquet'); a = a[a.applicant_cd.isin(NF)]; a['v'] = np.minimum(s8(a.pub_date.values), s8(a.rgst_date.values))
E_pub = set(a[(a.ay <= CUT) & (a.v <= CAPV)].applicant_cd)
rl = pd.read_parquet(CACHE/'rl_pat.parquet', columns=['applicant_cd', 'rgst_no', 'rgst_date', 'pert'])
rl = rl[rl.applicant_cd.isin(NF) & (rl.pert == '권리자란')].copy(); rl['rgst_no'] = rl.rgst_no.astype(str); rl['rg'] = s8(rl.rgst_date.values)
insset = set(pat.rgst_no); rl['ins'] = rl.rgst_no.isin(insset)
E_rl_any = set(rl[rl.rg <= CAPV].applicant_cd)
rli = rl[rl.ins & (rl.rg <= CAPV)].drop_duplicates(['applicant_cd', 'rgst_no']); E_rl_ins = set(rli.applicant_cd)
oos = pd.read_parquet(CACHE/'oos.parquet', columns=['applicant_cd', 'rgst_no', 'ay', 'rgst_date', 'self'])
oo = oos[oos.self & (oos.ay >= 2000) & (oos.ay <= CUT) & oos.applicant_cd.isin(NF)]; E_oos_vis = set(oo[s8(oo.rgst_date.values) <= CAPV].applicant_cd)
firms = pd.read_parquet(CACHE/'firms.parquet', columns=['applicant_cd', 'applicant_name'])
al = pd.concat([firms[firms.applicant_cd.isin(NF)].rename(columns={'applicant_cd': 'cd', 'applicant_name': 'nm'}), app[app.applicantCd.isin(NF)][['applicantCd', 'applicantName']].rename(columns={'applicantCd': 'cd', 'applicantName': 'nm'})])
al['n'] = al.nm.map(norm); al = al[al.n.str.len() >= 2].drop_duplicates(); alias = al.groupby('cd').n.agg(set).to_dict()
con = ro(RGF); rn = sorted(rli.rgst_no.unique()); parts = []
for i in range(0, len(rn), 900):
    ch = rn[i:i + 900]; parts.append(pd.read_sql(f"select rgst_no, ownerName, ownerRgstCsDate from owner where rgst_no in ({','.join('?' * len(ch))})", con, params=ch))
ow = pd.concat(parts, ignore_index=True); ow['n'] = ow.ownerName.map(norm); ow['d'] = s8(ow.ownerRgstCsDate.values)
m = rli.merge(ow, on='rgst_no', how='left')
m['name_ok'] = [isinstance(n, str) and (n in alias.get(c, set()) or any(len(q) >= 3 and q in n for q in alias.get(c, set()))) for c, n in zip(m.applicant_cd, m.n)]
E_own = set(m[m.name_ok & (m.d <= CAPV)].applicant_cd)
F = pd.DataFrame(dict(firm=sorted(NF))); F['is_newc_off'] = F.firm.isin(NO); F['is_newc_s1'] = F.firm.isin(NS)
for nm, S in (('E_pub', E_pub), ('E_reg', E_reg), ('E_own', E_own), ('E_rl_ins', E_rl_ins), ('E_rl_any', E_rl_any), ('E_oos_vis', E_oos_vis)): F[nm] = F.firm.isin(S)
F['N1_primary'] = F.E_pub | F.E_reg | F.E_own; F['N1_narrow'] = F.E_pub; F['N1_rlins'] = F.E_pub | F.E_reg | F.E_rl_ins; F['N1_wide'] = F.N1_rlins | F.E_rl_any | F.E_oos_vis
F.to_parquet(CC/'evid_all.parquet', index=False)
Q = pd.read_parquet(CC/'newc_evidence.parquet').set_index('firm'); Fo = F[F.is_newc_off].set_index('firm').reindex(Q.index)
out = dict(newc_off=len(NO), newc_s1=len(NS), overlap=len(NO & NS), s1_only=len(NS - NO), off_only=len(NO - NS),
           eq_q0={c: bool((Fo[c].values == Q[c].values).all()) for c in ['E_pub', 'E_reg', 'E_own', 'E_rl_ins', 'N1_primary', 'N1_narrow', 'N1_rlins', 'N1_wide']},
           off={c: int(F[F.is_newc_off][c].sum()) for c in F.columns if c.startswith(('E_', 'N1_'))},
           s1={c: int(F[F.is_newc_s1][c].sum()) for c in F.columns if c.startswith(('E_', 'N1_'))}, mem_peak=G.rss_peak())
jdump(out, CC/'n1_prep.json'); print(json.dumps(out, ensure_ascii=False, indent=1))
