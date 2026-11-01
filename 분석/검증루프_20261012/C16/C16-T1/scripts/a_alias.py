# -*- coding: utf-8 -*-
"""이름·별칭 매칭 감사(PREREG §6). 레이블·점수는 cache/labels.parquet, 공식 점수.
 산출: alias_audit.csv(층화 표본 320), cache/alias_docs.parquet(P·R 전체 근거 문서 판정), cache/alias_summary.json"""
import sys, pathlib, sqlite3, re; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard16 as G
from c16common import *
import time
t0 = time.time()
STRIP = re.compile(r"주식회사|\(주\)|㈜|유한회사|유한책임회사|재단법인|사단법인|학교법인|가부시키가이샤|가부시끼가이샤|가부시기가이샤|가부시키카이샤|코포레이션|인코포레이티드|리미티드|컴퍼니|[\s\.\,\-\(\)\'\"·&]")
norm = lambda s: STRIP.sub('', str(s)).lower()            # prep_app.py와 같은 정규화
SIDO = [('서울', '서울'), ('부산', '부산'), ('대구', '대구'), ('인천', '인천'), ('광주', '광주전남'), ('전라남', '광주전남'), ('전남', '광주전남'), ('대전', '대전'), ('울산', '울산'),
        ('세종', '세종'), ('경기', '경기'), ('강원', '강원'), ('충청북', '충북'), ('충북', '충북'), ('충청남', '충남'), ('충남', '충남'), ('전라북', '전북'), ('전북', '전북'),
        ('경상북', '경북'), ('경북', '경북'), ('경상남', '경남'), ('경남', '경남'), ('제주', '제주')]
CTRY = {'일본국': '일본', '일본': '일본', '미국': '미국', '미합중국': '미국', '중화민국': '대만', '대만': '대만', '중국': '중국', '중화인민공화국': '중국', '독일연방공화국': '독일', '독일': '독일',
        '대영제국': '영국', '영국': '영국', '대한민국': None}


def region(addr):
    t = str(addr or '').replace(',', ' ').split()
    if t and t[0] == '대한민국': t = t[1:]
    if not t or t[0].startswith('...') or t[0] == 'None': return None
    w = t[0]
    for pre, r in SIDO:
        if w.startswith(pre): return r
    if re.match(r'^[\*\d]', w): return None
    return CTRY.get(w, w)


# ---- 레이블·쌍 ----
L = pd.read_parquet(CC/'labels.parquet'); L = L[L.src == 'te'].reset_index(drop=True)
kept = ~L.prior_any.values
Pm = kept & (L.y_pub.values == 1) & (L.y_reg.values == 0); Rm = L.prior_any.values
P = L[Pm][['firm', 'k']].assign(grp='P'); R = L[Rm][['firm', 'k']].assign(grp='R')
summ = dict(eval_pos=int(L.y_u[kept].sum()), P_pairs=int(Pm.sum()), R_pairs=int(Rm.sum()), pos_both_paths=int((kept & (L.y_pub.values == 1) & (L.y_reg.values == 1)).sum()), pos_reg_only=int((kept & (L.y_pub.values == 0) & (L.y_reg.values == 1)).sum()))
# ---- 근거 문서 ----
a = load_app()
aP = a[(a.ay > 2019) & (a.ay <= 2023) & (a.v <= END)].merge(P, left_on=['applicant_cd', 'k'], right_on=['firm', 'k'])
aR = a[a.ay <= 2019].merge(R, left_on=['applicant_cd', 'k'], right_on=['firm', 'k'])
docs = pd.concat([aP, aR], ignore_index=True)[['grp', 'firm', 'k', 'appl_no', 'ay', 'pub_date', 'rgst_date', 'reg']]
summ['P_pairs_with_docs'] = int(docs[docs.grp == 'P'].groupby(['firm', 'k']).ngroups); summ['R_pairs_with_docs'] = int(docs[docs.grp == 'R'].groupby(['firm', 'k']).ngroups)
kp = pd.read_parquet(B/'공모전데이터/작업데이터/kr_patents.parquet', columns=['출원번호', '등록번호', '출원인명', '출원인주소'])
kp = kp[kp['출원번호'].astype(str).isin(set(docs.appl_no.astype(str)))].drop_duplicates('출원번호')
kp.index = kp['출원번호'].astype(str)
docs['kp_rgst_no'] = docs.appl_no.astype(str).map(kp['등록번호']).fillna('').astype(str)
docs['names_raw'] = docs.appl_no.astype(str).map(kp['출원인명']).fillna(''); docs['addrs_raw'] = docs.appl_no.astype(str).map(kp['출원인주소']).fillna('')
# ---- 별칭 사전(prep_app.py와 같은 규칙) ----
firms = pd.read_parquet(CACHE/'firms.parquet'); firms = firms[firms.status == 'OK']
app = pd.read_parquet(CACHE/'ins_applicant.parquet')
al = pd.concat([firms[['applicant_cd', 'applicant_name']].rename(columns={'applicant_name': 'nm'}),
                app[app.applicantCd.isin(firms.applicant_cd)][['applicantCd', 'applicantName']].rename(columns={'applicantCd': 'applicant_cd', 'applicantName': 'nm'})])
al['n'] = al.nm.map(norm); al = al[al.n.str.len() >= 2].drop_duplicates(['applicant_cd', 'n'])
amb = al.groupby('n').applicant_cd.nunique(); al = al[al.n.map(amb) == 1]; n2c = dict(zip(al.n, al.applicant_cd))
# ---- 등록원부(코드·주소·이름 이력) ----
con = sqlite3.connect((B/'수집파이프라인/registry_full.sqlite').as_uri() + '?mode=ro', uri=True)
ra = pd.read_sql('select rgst_no, applicantCd, applicantName, applicantAddr from applicant', con); con.close()
ra['reg_r'] = ra.applicantAddr.map(region); ra['n'] = ra.applicantName.map(norm)
codes_by_rg = ra.groupby('rgst_no').applicantCd.apply(lambda s: set(s.dropna().astype(str)))
fset = set(docs.firm)
rf = ra[ra.applicantCd.isin(fset)]
firm_regions = rf.dropna(subset=['reg_r']).groupby('applicantCd').reg_r.apply(set)
firm_names = rf.groupby('applicantCd').n.apply(set)
# ---- CA 법인번호·사업자번호 ----
ca = pd.read_parquet(B/'공모전데이터/외부데이터/DART/corp_applicant.parquet', columns=['출원인코드', '출원인명', 'jurir', 'bizr'])
ca['출원인코드'] = ca['출원인코드'].astype(str); ca['jurir'] = ca.jurir.fillna('').astype(str).str.strip(); ca['bizr'] = ca.bizr.fillna('').astype(str).str.strip()
ca_j = dict(zip(ca['출원인코드'], ca.jurir)); ca_b = dict(zip(ca['출원인코드'], ca.bizr))
ca['n'] = ca['출원인명'].map(norm)
hom = ca[ca.jurir.str.len() > 5].groupby('n').jurir.nunique()


def same_entity(c1, c2):
    j1, j2 = ca_j.get(c1, ''), ca_j.get(c2, ''); b1, b2 = ca_b.get(c1, ''), ca_b.get(c2, '')
    return (len(j1) > 5 and j1 == j2) or (len(b1) > 5 and b1 == b2)


rows = []
for r in docs.itertuples(index=False):
    names = str(r.names_raw).split('|'); addrs = str(r.addrs_raw).split('|')
    posi = [i for i, nm in enumerate(names) if n2c.get(norm(nm)) == r.firm]
    i0 = posi[0] if posi else None
    nm_raw = names[i0] if i0 is not None else ''; nmn = norm(nm_raw)
    if i0 is not None and len(addrs) == len(names): sa = {region(addrs[i0])} - {None}
    else: sa = {region(x) for x in addrs} - {None}
    sf = firm_regions.get(r.firm, set())
    e2 = 'none' if (not sa or not sf) else ('match' if sa & sf else 'mismatch')
    e4 = nmn in firm_names.get(r.firm, set())
    e3_hom = bool(hom.get(nmn, 0) >= 2)
    is_reg = len(r.kp_rgst_no) > 5
    e1 = codes_by_rg.get(r.kp_rgst_no) if is_reg else None
    if e1 is not None:
        if r.firm in e1: v = '정확(코드 일치)'
        elif any(same_entity(r.firm, c) for c in e1): v = '정확(CA 같은 실체)'
        else: v = '오매칭(코드 불일치)'
    else:
        if e2 == 'match': v = '정확(주소 시도 일치)'
        elif e2 == 'mismatch': v = '오매칭 의심(주소 시도 불일치)'
        elif e4: v = '정확(상호 이력 일치)'
        elif e3_hom: v = '오매칭 의심(CA 동명)'
        else: v = '판정 불가'
    # 등록 문서에 주소 규칙도 적용(보정용)
    v_addr = ('정확' if e2 == 'match' else ('불일치' if e2 == 'mismatch' else ('이력' if e4 else ('동명' if e3_hom else '불가'))))
    rows.append(dict(grp=r.grp, firm=r.firm, k=r.k, appl_no=r.appl_no, ay=r.ay, is_reg=is_reg, rgst_no=r.kp_rgst_no, name_raw=nm_raw, name_norm=nmn, name_len=len(nmn),
                     app_regions='/'.join(sorted(sa)), firm_regions='/'.join(sorted(sf))[:120], e1_codes='/'.join(sorted(e1))[:80] if e1 is not None else '',
                     e1_has_firm=(r.firm in e1) if e1 is not None else None, e2=e2, e3_homonym=e3_hom, e4_name_history=e4, verdict=v, addr_rule=v_addr,
                     firm_jurir=ca_j.get(r.firm, ''), name_found=i0 is not None))
DOC = pd.DataFrame(rows)
DOC['wrong'] = DOC.verdict.str.startswith('오매칭'); DOC['undet'] = DOC.verdict.eq('판정 불가')
DOC.to_parquet(CC/'alias_docs.parquet', index=False)
# ---- 층화 표본(시드 20261017, 쌍마다 근거 문서 1건) ----
rs = np.random.RandomState(20261017)
DOC['_u'] = rs.rand(len(DOC))
one = DOC.sort_values(['grp', 'firm', 'k', '_u']).groupby(['grp', 'firm', 'k'], sort=False).head(1).reset_index(drop=True)
one['stratum'] = np.where(one.grp == 'R', 'R', np.where(one.is_reg, 'P-등록', 'P-미등록'))
NS = {'P-등록': 80, 'P-미등록': 140, 'R': 100}
samp = []
for st_, n in NS.items():
    g = one[one.stratum == st_]; samp.append(g.sample(n=min(n, len(g)), random_state=20261017).assign(stratum_size=len(g)))
S = pd.concat(samp, ignore_index=True)


def wilson(x, n, z=1.959963984540054):
    if n == 0: return [None, None, None]
    p = x / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [p, max(0.0, c - h), min(1.0, c + h)]


st = {}
for g, d in list(S.groupby('stratum')) + [('표본 전체', S), ('P 전체(표본)', S[S.stratum != 'R'])]:
    n_det = int((~d.undet).sum()); x = int(d.wrong.sum())
    st[g] = dict(n=int(len(d)), n_determinable=n_det, wrong=x, undet=int(d.undet.sum()), rate_wilson=wilson(x, n_det),
                 worst_rate_wilson=wilson(x + int(d.undet.sum()), int(len(d))), verdicts=d.verdict.value_counts().to_dict())
summ['sample'] = st
# 주소 규칙 보정(등록 문서: E1 정답 대비)
regd = DOC[DOC.is_reg & DOC.e1_has_firm.notna()]
corr = regd[~regd.wrong]; wr = regd[regd.wrong]
summ['addr_rule_calibration_on_registered'] = dict(n_correct_by_code=int(len(corr)), addr_mismatch_among_correct=int((corr.e2 == 'mismatch').sum()),
    false_mismatch_rate=float((corr.e2 == 'mismatch').sum() / max((corr.e2 != 'none').sum(), 1)), n_wrong_by_code=int(len(wr)),
    addr_mismatch_among_wrong=int((wr.e2 == 'mismatch').sum()), detect_rate=float((wr.e2 == 'mismatch').sum() / max((wr.e2 != 'none').sum(), 1)))
# 모집단(전체 근거 문서) 비율과 쌍 단위 판정
pop = {}
for g in ('P', 'R'):
    d = DOC[DOC.grp == g]
    pr = d.groupby(['firm', 'k']).agg(n_docs=('wrong', 'size'), all_wrong=('wrong', 'all'), any_correct=('verdict', lambda v: v.str.startswith('정확').any()), all_undet_or_wrong=('verdict', lambda v: (~v.str.startswith('정확')).all()))
    pop[g] = dict(docs=int(len(d)), docs_wrong=int(d.wrong.sum()), docs_undet=int(d.undet.sum()), doc_wrong_rate_det=float(d.wrong.sum() / max((~d.undet).sum(), 1)),
                  pairs=int(len(pr)), pairs_all_docs_wrong=int(pr.all_wrong.sum()), pairs_no_correct_doc=int(pr.all_undet_or_wrong.sum()),
                  by_reg={str(k): dict(docs=int(len(x)), wrong=int(x.wrong.sum()), undet=int(x.undet.sum())) for k, x in d.groupby('is_reg')})
    pr.reset_index().to_parquet(CC/f'alias_pairs_{g}.parquet', index=False)
summ['population'] = pop
# 동명 중 짧은 이름(C03-T1 비교용)
summ['sample_short_name_le3_wrong'] = dict(n=int((S.name_len <= 3).sum()), wrong=int(S.wrong[S.name_len <= 3].sum()))
cols = ['stratum', 'stratum_size', 'grp', 'firm', 'k', 'appl_no', 'ay', 'is_reg', 'rgst_no', 'name_raw', 'name_len', 'app_regions', 'firm_regions', 'e1_codes', 'e1_has_firm', 'e2', 'e3_homonym', 'e4_name_history', 'firm_jurir', 'verdict', 'addr_rule']
S[cols].to_csv(HERE/'alias_audit.csv', index=False, encoding='utf-8-sig')
summ['secs'] = round(time.time() - t0); summ['peak'] = G.rss_peak()
jdump(summ, CC/'alias_summary.json'); print(json.dumps(summ, ensure_ascii=False, indent=1, default=str))
