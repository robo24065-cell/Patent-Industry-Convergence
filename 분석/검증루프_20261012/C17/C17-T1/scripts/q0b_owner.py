# -*- coding: utf-8 -*-
"""Q0b(레이블·점수 보지 않음): E_rl_ins(2026 권리목록의 범위안 보유, 등록 ≤ 원점)를 등록원부 owner 이전등록일로 날짜 확인.
 owner 행 이름(정규화)이 법인 별칭(조회명 + 등록원부 같은 코드 이름)과 같고 ownerRgstCsDate ≤ 2019-12-31이면 E_own.
 → cache/newc_evidence.parquet에 E_own 열 추가(덮어쓰기, 같은 폴더), cache/q0b_owner.json"""
import sys, pathlib, re; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard17 as G
from c17common import *
F = pd.read_parquet(CC/'newc_evidence.parquet'); NF = set(F.firm)
STRIP = re.compile(r"주식회사|\(주\)|㈜|유한회사|유한책임회사|재단법인|사단법인|학교법인|가부시키가이샤|가부시끼가이샤|가부시기가이샤|가부시키카이샤|코포레이션|인코포레이티드|리미티드|컴퍼니|[\s\.\,\-\(\)\'\"·&]")
norm = lambda s: STRIP.sub('', str(s)).lower()                      # prep2.norm과 같은 식
firms = pd.read_parquet(CACHE/'firms.parquet', columns=['applicant_cd', 'applicant_name']); app = pd.read_parquet(CACHE/'ins_applicant.parquet', columns=['applicantCd', 'applicantName'])
al = pd.concat([firms[firms.applicant_cd.isin(NF)].rename(columns={'applicant_cd': 'cd', 'applicant_name': 'nm'}), app[app.applicantCd.isin(NF)].rename(columns={'applicantCd': 'cd', 'applicantName': 'nm'})])
al['n'] = al.nm.map(norm); al = al[al.n.str.len() >= 2].drop_duplicates(); alias = al.groupby('cd').n.agg(set).to_dict()
pat = set(pd.read_parquet(CACHE/'ins_pat.parquet', columns=['rgst_no']).rgst_no.astype(str))
rl = pd.read_parquet(CACHE/'rl_pat.parquet', columns=['applicant_cd', 'rgst_no', 'rgst_date', 'pert'])
rl = rl[rl.applicant_cd.isin(NF) & (rl.pert == '권리자란')]; rl['rgst_no'] = rl.rgst_no.astype(str)
rl = rl[rl.rgst_no.isin(pat) & (s8(rl.rgst_date.values) <= CAPV)].drop_duplicates(['applicant_cd', 'rgst_no'])
con = ro(RGF); rn = sorted(rl.rgst_no.unique()); rows = []
for i in range(0, len(rn), 900):
    ch = rn[i:i + 900]; q = f"select rgst_no, ownerName, ownerRgstCsDate, ownerRgstCsReason, finalOwnerYn from owner where rgst_no in ({','.join('?' * len(ch))})"
    rows.append(pd.read_sql(q, con, params=ch))
ow = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=['rgst_no', 'ownerName', 'ownerRgstCsDate', 'ownerRgstCsReason', 'finalOwnerYn'])
ow['n'] = ow.ownerName.map(norm); ow['d'] = s8(ow.ownerRgstCsDate.values)
m = rl.merge(ow, on='rgst_no', how='left')
m['name_ok'] = [isinstance(n, str) and (n in alias.get(c, set()) or any(len(a) >= 3 and a in n for a in alias.get(c, set()))) for c, n in zip(m.applicant_cd, m.n)]
own_ok = m[m.name_ok]
E_own = set(own_ok[own_ok.d <= CAPV].applicant_cd)
first_own = own_ok.groupby('applicant_cd').d.min()
F['E_own'] = F.firm.isin(E_own)
F['N1_primary'] = F.E_pub | F.E_reg | F.E_own
F['N1_rlins'] = F.E_pub | F.E_reg | F.E_rl_ins
F['N1_narrow'] = F.E_pub
F['N1_wide'] = F.N1_rlins | F.E_rl_any | F.E_oos_vis
F.to_parquet(CC/'newc_evidence.parquet', index=False)
te = pd.read_parquet(FC/'prep_te.parquet', columns=['firm', 'newc', 'prior_any']); pn = te[te.newc.values & ~te.prior_any.values]
out = dict(rl_ins_patents=int(len(rl)), rl_ins_firms=int(rl.applicant_cd.nunique()), owner_rows=int(len(ow)), rl_rows_name_matched=int(m.name_ok.sum()),
           rl_firms_name_matched=int(own_ok.applicant_cd.nunique()), E_own_firms=len(E_own),
           first_own_year_dist={int(k): int(v) for k, v in (first_own[first_own < 99999999] // 10000).value_counts().sort_index().items()},
           firms={c: int(F[c].sum()) for c in ['E_pub', 'E_reg', 'E_own', 'E_rl_ins', 'N1_primary', 'N1_rlins', 'N1_narrow', 'N1_wide']},
           pairs_keep={c: int(pn.firm.map(F.set_index('firm')[c]).sum()) for c in ['N1_primary', 'N1_rlins', 'N1_narrow', 'N1_wide']},
           owner_reason_top=ow.ownerRgstCsReason.value_counts().head(6).to_dict(), mem_peak=G.rss_peak())
jdump(out, CC/'q0b_owner.json'); print(json.dumps(out, ensure_ascii=False, indent=1))
