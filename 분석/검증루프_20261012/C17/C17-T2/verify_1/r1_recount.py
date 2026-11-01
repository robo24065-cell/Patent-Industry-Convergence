# -*- coding: utf-8 -*-
"""C17-T2 verify_1 r1: 반박 검증 - right_list 독립 전수 스캔, P-미등록 적중 수, 등록일 분포, 간이 독립 판정, kappa 재계산.
 원천은 읽기 전용(sqlite mode=ro). 쓰기는 verify_1 안만."""
import sys, pathlib, json, re, sqlite3, time
import numpy as np, pandas as pd
sys.dont_write_bytecode = True
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
t0 = time.time()
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
H = B/'분석/검증루프_20261012/C17/C17-T2'; OUT = H/'verify_1'
assert OUT.exists()
C16 = B/'분석/검증루프_20261012/C16/C16-T1'
ro = lambda p: sqlite3.connect(pathlib.Path(p).as_uri() + '?mode=ro', uri=True)
o = {}
D = pd.read_parquet(C16/'cache/alias_docs.parquet'); D['appl_no'] = D.appl_no.astype(str); D['firm'] = D.firm.astype(str)
o['docs'] = len(D); o['docs_dup_rows'] = int(D.duplicated(['grp', 'firm', 'k', 'appl_no']).sum())
o['grp_isreg'] = D.groupby(['grp', 'is_reg']).appl_no.agg(['size', 'nunique']).reset_index().astype(str).to_dict('records')
o['appl_no_len'] = D.appl_no.str.len().value_counts().to_dict()
need = set(D.appl_no); need_d = {re.sub(r'\D', '', a) for a in need}
con = ro(B/'수집파이프라인/business_rights.sqlite')
rows = []; n = 0; fmt = {}
cur = con.execute("select applicant_cd, appl_no, rgst_date, corp_rgstno, json_extract(raw,'$.applicantInfo') from right_list")
while True:
    ch = cur.fetchmany(250000)
    if not ch: break
    n += len(ch)
    for r in ch:
        a = r[1] or ''
        if a in need: rows.append(r)
        elif re.sub(r'\D', '', a) in need_d: fmt[a] = fmt.get(a, 0) + 1   # 형식만 다른 적중(놓친 적중) 검사
o['rl_rows'] = n; o['format_only_hits'] = len(fmt)
qcodes = set(x[0] for x in con.execute('select applicant_cd from applicant_query'))
con.close()
RL = pd.DataFrame(rows, columns=['h', 'appl_no', 'rd', 'corp', 'ainfo']); RL['h'] = RL.h.astype(str)
o['rl_hit_rows'] = len(RL); o['rl_hit_appl'] = int(RL.appl_no.nunique())
o['firm_in_query_codes_P'] = float(D[D.grp == 'P'].firm.isin(qcodes).mean())
o['firm_in_query_codes_pairsP'] = float(D[D.grp == 'P'].drop_duplicates(['firm', 'k']).firm.isin(qcodes).mean())
PU = D[(D.grp == 'P') & (~D.is_reg)]
o['P_unreg_docs_rows'] = len(PU); o['P_unreg_docs_unique_appl'] = int(PU.appl_no.nunique())
inrl = PU.appl_no.isin(set(RL.appl_no))
o['P_unreg_in_rl_rows'] = int(inrl.sum()); o['P_unreg_in_rl_unique'] = int(PU[inrl].appl_no.nunique())
o['P_unreg_in_rl_pct'] = float(inrl.mean())
rd = RL[RL.appl_no.isin(set(PU.appl_no))].groupby('appl_no').rd.min().astype(str)
o['rgst_date_min'] = rd.min(); o['rgst_date_max'] = rd.max()
o['rgst_ym_hist'] = rd.str[:6].value_counts().sort_index().to_dict()
o['rgst_in_202508_202608'] = int(((rd >= '20250801') & (rd <= '20260831')).sum())
o['rgst_before_20250822'] = int((rd <= '20250822').sum())
# 간이 독립 판정(P-미등록 문서): 보유자 = 법인(코드 또는 법인번호) & 원출원인명에 kr 출원인명 포함
STRIP = re.compile(r"주식회사|\(주\)|㈜|유한회사|유한책임회사|재단법인|사단법인|학교법인|코포레이션|인코포레이티드|인코퍼레이티드|리미티드|컴퍼니|[^0-9A-Za-z가-힣]")
nz = lambda s: STRIP.sub('', str(s or '')).lower()
dg = lambda s: re.sub(r'\D', '', str(s or ''))
corp = {}
for c, j in zip(RL.h, RL.corp):
    j = dg(j)
    if len(j) == 13: corp.setdefault(c, j)
fj = {f: dg(j) for f, j in zip(D.firm, D.firm_jurir)}
g = {k: v for k, v in RL.groupby('appl_no')}
def jv(firm, nm, a):
    x = g.get(a)
    if x is None: return 'undet'
    Hs = set(x.h); fjj = fj.get(firm, '') if len(fj.get(firm, '')) == 13 else corp.get(firm, '')
    hold = firm in Hs or (fjj and any(corp.get(h) == fjj for h in Hs))
    nn = nz(nm); infos = [nz(i) for i in x.ainfo.dropna()]
    nmok = bool(nn) and any(nn in i for i in infos)
    if hold and nmok: return 'correct'
    if hold: return 'hold_name_diff'
    if nmok: return 'name_only'
    return 'other'
PU = PU.assign(jv=[jv(f, n, a) for f, n, a in zip(PU.firm, PU.name_raw, PU.appl_no)])
o['P_unreg_simple_judge'] = PU.jv.value_counts().to_dict()
# 쌍 단위: 내 간이 판정 + 등록 문서는 s1 결과 그대로(등록원부 판정은 별도 확인 안 함)
dj = pd.read_parquet(H/'cache/doc_judg.parquet'); dj['appl_no'] = dj.appl_no.astype(str); dj['firm'] = dj.firm.astype(str)
P = D[D.grp == 'P'].merge(dj[['grp', 'firm', 'k', 'appl_no', 'st', 'v']], on=['grp', 'firm', 'k', 'appl_no'], how='left')
P['mine'] = np.where(P.is_reg, P.st, P.appl_no.map(dict(zip(PU.appl_no, PU.jv))).fillna('undet'))
P['mine'] = P.mine.replace({'hold_name_diff': 'wrong', 'name_only': 'correct2', 'other': 'undet'})
def agg(v):
    v = set(v)
    if 'correct' in v or 'correct2' in v: return 'correct'
    if v == {'wrong'}: return 'wrong'
    return 'undet'
pm = P.groupby(['firm', 'k']).mine.agg(agg)
o['pairs_P'] = int(len(pm)); o['pairs_mine'] = pm.value_counts().to_dict()
pj = pd.read_parquet(H/'cache/pair_judg.parquet'); pj = pj[pj.grp == 'P']; pj['firm'] = pj.firm.astype(str)
cmp_ = pj.set_index(['firm', 'k']).pv.to_frame().join(pm.rename('mine'))
o['pairs_main_vs_mine'] = pd.crosstab(cmp_.pv, cmp_.mine).to_dict()
o['pairs_det_main'] = int((pj.pv != 'undet').sum())
o['det_via_rl_only'] = int(((pj.pv != 'undet') & (pj.det_src == 'B')).sum())
o['det_via_reg'] = int(((pj.pv != 'undet') & (pj.det_src == 'A')).sum())
# kappa 재계산
vp = pd.read_csv(H/'verify/v_indep_pairs.csv', dtype=str)
x = vp.pv.values; y = np.where(vp.indep.str.startswith('undet'), 'undet', vp.indep.values)
def kap(x, y):
    cats = sorted(set(x) | set(y)); cm = pd.crosstab(pd.Categorical(x, cats), pd.Categorical(y, cats), dropna=False).values
    po = np.trace(cm) / cm.sum(); pe = (cm.sum(1) * cm.sum(0)).sum() / cm.sum() ** 2
    return float(po), float(pe), float((po - pe) / (1 - pe))
o['kappa3'] = kap(x, y)
b = (x != 'undet') & (y != 'undet'); o['kappa_det_only_n'] = int(b.sum()); o['kappa_det_only'] = kap(x[b], y[b])
o['v_indep_n'] = len(vp); S = pd.read_csv(H/'alias_audit_v2.csv', dtype=str)
o['audit_sample_pairs'] = int(S.drop_duplicates(['firm', 'k']).shape[0]); o['audit_rows'] = len(S)
o['v_indep_sample_pv'] = vp.pv.value_counts().to_dict(); o['audit_pv'] = S.drop_duplicates(['firm', 'k']).pv.value_counts().to_dict() if 'pv' in S else None
o['secs'] = round(time.time() - t0)
json.dump(o, open(OUT/'r1_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(o, ensure_ascii=False, indent=1, default=str))
