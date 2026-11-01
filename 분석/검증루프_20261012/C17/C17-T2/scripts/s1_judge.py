# -*- coding: utf-8 -*-
"""C17-T2 s1: PREREG §1–3 코드 근거 위계로 근거 문서·쌍을 판정한다(P·R 전체).
 입력: C16 alias_docs(근거 문서, s0에서 재구성 일치 확인), registry_full(applicant·owner·patent), right_list 적중(cache/rl_hits.parquet, e0에서 전수 스캔),
       business_rights.applicant_query(코드 이름), T1 firms·ins_applicant, CA(jurir·bizr)
 산출: cache/doc_judg.parquet, cache/pair_judg.parquet, cache/s1_out.json"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard17 as G
from c17common import *
t0 = time.time(); out = {}
D = pd.read_parquet(C16/'cache/alias_docs.parquet')
D['firm'] = D.firm.astype(str); D['appl_no'] = D.appl_no.astype(str); D['rgst_no'] = D.rgst_no.fillna('').astype(str)
D['nn'] = D.name_raw.fillna('').map(norm)
# ---- 등록원부 ----
con = ro(REG)
pat = pd.read_sql('select rgst_no, applNo from patent', con); pat['applNo'] = pat.applNo.astype(str)
ra = pd.read_sql('select rgst_no, applicantCd, applicantName from applicant', con); ra['applicantCd'] = ra.applicantCd.astype(str)
ow = pd.read_sql('select rgst_no, seq, finalOwnerYn, ownerName from owner', con)
con.close()
print('registry loaded', round(time.time() - t0), G.rss_now(), flush=True)
a2r = dict(zip(pat.applNo, pat.rgst_no))
D['reg_key'] = np.where(D.rgst_no.str.len() > 5, D.rgst_no, D.appl_no.map(a2r).fillna(''))
codes_by_rg = ra.groupby('rgst_no').applicantCd.apply(lambda s: set(s)).to_dict()
fin = ow[ow.finalOwnerYn == 'Y'].copy(); fin['n'] = fin.ownerName.map(norm); fin_by_rg = fin.groupby('rgst_no').n.apply(set).to_dict()
# 법인 이름집합: firms·ins_applicant·등록원부 applicant 이름 + owner seq0(그 코드가 원출원인인 등록 문서의 첫 권리자명)
firms = pd.read_parquet(TC/'firms.parquet'); firms['applicant_cd'] = firms.applicant_cd.astype(str)
ia = pd.read_parquet(TC/'ins_applicant.parquet'); ia['applicantCd'] = ia.applicantCd.astype(str)
con = ro(BIZ); aq = pd.read_sql('select applicant_cd, applicant_name, status from applicant_query', con); con.close(); aq['applicant_cd'] = aq.applicant_cd.astype(str)
nm_rows = pd.concat([firms[['applicant_cd', 'applicant_name']].set_axis(['cd', 'nm'], axis=1), ia[['applicantCd', 'applicantName']].set_axis(['cd', 'nm'], axis=1),
                     ra[['applicantCd', 'applicantName']].set_axis(['cd', 'nm'], axis=1), aq[['applicant_cd', 'applicant_name']].set_axis(['cd', 'nm'], axis=1)])
need_cd = set(D.firm)
s0 = ow[ow.seq == 0][['rgst_no', 'ownerName']].merge(ra[ra.applicantCd.isin(need_cd)][['rgst_no', 'applicantCd']], on='rgst_no')
# 단독출원 문서의 seq0만 이름 이력으로 쓴다(공동출원이면 seq0이 다른 출원인일 수 있음)
nco = ra.groupby('rgst_no').applicantCd.nunique(); s0 = s0[s0.rgst_no.map(nco) == 1]
nm_rows = pd.concat([nm_rows, s0[['applicantCd', 'ownerName']].set_axis(['cd', 'nm'], axis=1).assign(src='seq0')])
nm_rows['n'] = nm_rows.nm.map(norm); nm_rows = nm_rows[nm_rows.n.str.len() >= 2]
names_of = nm_rows.groupby('cd').n.apply(set).to_dict()
codes_of_name = nm_rows.drop_duplicates(['cd', 'n']).groupby('n').cd.apply(set).to_dict()
out['name_universe'] = dict(codes=len(names_of), names=len(codes_of_name), seq0_rows=int(len(s0)))
del ow, nm_rows
print('names', round(time.time() - t0), G.rss_now(), flush=True)
# ---- CA 같은 실체 ----
ca = pd.read_parquet(CA, columns=['출원인코드', 'jurir', 'bizr']); ca['출원인코드'] = ca['출원인코드'].astype(str)
ca_j = {c: digits(j) for c, j in zip(ca['출원인코드'], ca.jurir)}
ca_b = {c: re.sub(r'\D', '', str(b or '')) for c, b in zip(ca['출원인코드'], ca.bizr)}
# ---- 권리목록 적중 ----
RL = pd.read_parquet(CC/'rl_hits.parquet'); RL['holder_cd'] = RL.holder_cd.astype(str); RL['corp'] = RL.corp_rgstno.map(digits)
corp_of = dict(ca_j)
for c, j in zip(RL.holder_cd, RL.corp):
    if j and not corp_of.get(c): corp_of[c] = j


def same_entity(c1, c2):
    if c1 == c2: return True
    j1, j2 = corp_of.get(c1, ''), corp_of.get(c2, ''); b1, b2 = ca_b.get(c1, ''), ca_b.get(c2, '')
    return (len(j1) == 13 and j1 == j2) or (len(b1) >= 10 and b1 == b2)


rl_by = {k: g for k, g in RL.groupby('appl_no')}


def in_info(n, infos):
    """이름 n(정규화)이 원출원인명 목록 문자열들에 있는가. 외국 이름 안 쉼표('…, 인크.') 때문에 쉼표 분할 대신
    4자 이상은 정규화 문자열 포함, 3자 이하는 쉼표 분할 조각과 완전 일치(변경 기록 1: 구현 오류 수정)."""
    if not n: return False
    for x in infos:
        if len(n) >= 4 and n in norm(x): return True
        if len(n) < 4 and any(norm(p) == n for p in str(x).split(',')): return True
    return False


def judge(firm, nn, reg_key, appl_no):
    fnames = names_of.get(firm, set())
    if reg_key and reg_key in codes_by_rg:
        S = codes_by_rg[reg_key]
        if firm in S: return 'A_정확(코드)', 1, 'correct', ''
        if any(same_entity(firm, c) for c in S): return 'A_정확(CA)', 1, 'correct', ''
        g = rl_by.get(appl_no)
        hold = g is not None and firm in set(g.holder_cd)
        tr = hold or (nn and nn in fin_by_rg.get(reg_key, set()))
        if tr: return 'A_오매칭(권리이전형)', 1, 'wrong', '이전'
        hom = bool(nn) and any(nn in names_of.get(c, set()) for c in S)
        return ('A_오매칭(동명 코드)' if hom else 'A_오매칭(기타·승계)'), 1, 'wrong', ('동명' if hom else '기타')
    g = rl_by.get(appl_no)
    if g is None: return 'N_코드판정불가', 9, 'undet', ''
    H = set(g.holder_cd)
    infos = list(g.ainfo.dropna().astype(str))
    if any(same_entity(firm, h) for h in H):
        if in_info(nn, infos) or any(in_info(f, infos) for f in fnames if len(f) >= 4): return 'B1_정확', 1, 'correct', ''
        return 'B1_권리이전형', 1, 'wrong', '이전'
    if nn:
        for h in H:
            if nn in names_of.get(h, set()): return 'B2_오매칭(동명 코드)', 1, 'wrong', '동명'
        if in_info(nn, infos): return 'B2_2순위정확(이전)', 2, 'correct', ''
    return 'B_판정불가', 9, 'undet', ''


R = [judge(f, n, r, a) for f, n, r, a in zip(D.firm, D.nn, D.reg_key, D.appl_no)]
D['v'] = [x[0] for x in R]; D['tier'] = [x[1] for x in R]; D['st'] = [x[2] for x in R]; D['wtype'] = [x[3] for x in R]
D['in_rl'] = D.appl_no.isin(set(RL.appl_no)); D['in_reg'] = D.reg_key.isin(set(codes_by_rg))

kpn = pd.read_parquet(KP, columns=['출원번호', '출원인명']); kpn['출원번호'] = kpn['출원번호'].astype(str); kpn = kpn[kpn['출원번호'].isin(set(D.appl_no))]
nco_kp = dict(zip(kpn['출원번호'], kpn['출원인명'].fillna('').str.count(r'\|') + 1)); D['n_coapp'] = D.appl_no.map(nco_kp).fillna(1).astype(int)
D['homonym_codes'] = [len(codes_of_name.get(n, set())) for n in D.nn]
D.drop(columns=['reg_key']).to_parquet(CC/'doc_judg.parquet', index=False)
out['doc_verdicts'] = D.groupby(['grp', 'is_reg']).v.value_counts().reset_index().astype(str).to_dict('records')
# ---- 쌍 판정 ----
nd = ra.groupby('applicantCd').rgst_no.nunique().to_dict()


def pair_agg(g):
    st = g.st.values; tier = g.tier.values
    corr = st == 'correct'
    src = 'A' if g.v.str.startswith('A_').any() else ('B' if (g.v.str.startswith('B') & (g.st != 'undet')).any() else '')
    base = dict(n_docs=len(g), n_det=int((st != 'undet').sum()), F=bool((g.addr_rule == '불일치').all()), ay_min=int(g.ay.min()),
                coapp=bool((g.n_coapp > 1).any()), hom=int(g.homonym_codes.max()), det_src=src, any_rl=bool(g.in_rl.any()))
    if corr.any():
        return pd.Series(dict(pv='correct', ptier=int(tier[corr].min()), ptype='', **base))
    if (st == 'wrong').all():
        wt = set(g.wtype)
        tp = '권리이전형' if wt == {'이전'} else ('동명형' if '동명' in wt else '기타·승계형')
        return pd.Series(dict(pv='wrong', ptier=1, ptype=tp, **base))
    return pd.Series(dict(pv='undet', ptier=9, ptype=('부분오매칭' if (st == 'wrong').any() else ''), **base))


PJ = D.groupby(['grp', 'firm', 'k']).apply(pair_agg).reset_index()
PJ['nd'] = PJ.firm.map(nd).fillna(0).astype(int); PJ['size'] = size_bin(PJ.nd.values); PJ['foreign'] = is_foreign(PJ.firm.values)
c16p = pd.read_parquet(C16/'cache/alias_pairs_P.parquet'); c16p['firm'] = c16p.firm.astype(str)
PJ = PJ.merge(c16p[['firm', 'k', 'all_wrong']].rename(columns={'all_wrong': 'c16_flag'}).assign(grp='P'), on=['grp', 'firm', 'k'], how='left')
PJ['c16_flag'] = PJ.c16_flag.fillna(False).astype(bool)
PJ.to_parquet(CC/'pair_judg.parquet', index=False)
for g in ('P', 'R'):
    x = PJ[PJ.grp == g]
    out[f'pairs_{g}'] = dict(n=len(x), verdict=x.pv.value_counts().to_dict(), tier=x.ptier.value_counts().to_dict(), wrong_type=x[x.pv == 'wrong'].ptype.value_counts().to_dict(),
                             F=int(x.F.sum()), by_F=x.groupby('F').pv.value_counts().reset_index().astype(str).to_dict('records'),
                             cell=x.groupby(['F', 'foreign', 'size']).pv.value_counts().unstack(fill_value=0).reset_index().astype(str).to_dict('records'))
out['P_F_vs_c16flag'] = pd.crosstab(PJ[PJ.grp == 'P'].F, PJ[PJ.grp == 'P'].c16_flag).astype(int).to_dict()
out['secs'] = round(time.time() - t0); out['peak'] = G.rss_peak()
jdump(out, CC/'s1_out.json'); print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
