# -*- coding: utf-8 -*-
"""verify_1 v1: C05-T3 N6 공란 등록 발견 반박 시도 — 독립 재계산(트랙 규칙 재작성) + 권리목록 조회 범위 편향 측정."""
import duckdb, pandas as pd, numpy as np, os, sys, io, re, json, sqlite3, time
from sklearn.metrics import roc_auc_score
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
t0 = time.time()
B = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
MC = B + r"\분석\모델경쟁_20260927"; SH = MC + r"\_shared"
V = B + r"\분석\검증루프_20261012\C05\C05-T3\verify_1"
P = (SH + r"\ext\ntis_project_raw.parquet").replace(os.sep, "/")
M = (SH + r"\meta\*.parquet").replace(os.sep, "/")
o = {}
SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
def nrm(s):
    s = SUF.sub("", str(s or "")).upper(); return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", s)
pr = pd.read_parquet(MC + r"\N6_정부RnD기업_첫특허전환\final_preds_r5.parquet"); pr['bz'] = pr.bz.astype(str).str.zfill(10)
nt = duckdb.sql(f"""select year::int y, regexp_replace(corp_rgst_no,'[^0-9]','','g') bz, research_agency nm
   from read_parquet('{P}') where year::int between 2008 and 2021 and perform_grp in ('중소기업','중견기업','대기업')""").df()
nt = nt[nt.bz.str.len() == 10]; pop = set(nt[nt.y.between(2019, 2021)].bz)
ns = pd.read_parquet(SH + r"\ext\r3nts_applicant_status.parquet", columns=["applicantCd", "bizno", "aname"])
b2a = ns.groupby("bizno").applicantCd.apply(list).to_dict()
ns["nn"] = ns.aname.map(nrm); g = ns.groupby("nn").applicantCd.agg(["first", "size"]); uq = g[g["size"] == 1]["first"].to_dict()
mvc = nt[nt.y >= 2010].groupby("bz").nm.agg(lambda s: s.value_counts().index[0]).to_dict()
rows = []; how = {}
for bz in pop:
    cds = b2a.get(bz)
    if cds:
        rows += [(bz, x) for x in cds]; how[bz] = 'bizno'
    else:
        x = uq.get(nrm(mvc.get(bz)))
        if x:
            rows.append((bz, x)); how[bz] = 'name'
fl = pd.DataFrame(rows, columns=['bz', 'cd'])
af = pd.read_parquet(MC + r"\_ideas_R3\cache\app_firm.parquet", columns=["applicantCd", "appl_no", "appl_date", "known_date", "src"]).rename(columns={'applicantCd': 'cd'})
o['af_rows'] = len(af); o['af_src'] = af.src.value_counts().to_dict()
o['af_appl_after_2023_before_filter'] = int((af.appl_date > 20231231).sum())
af = af[af.appl_date <= 20231231]
fa = fl.merge(af, on='cd')
first = fa.groupby('bz').known_date.min()
risk = pop - set(first[first <= 20211231].index)
pos = set(fa[fa.appl_date >= 20220101].bz)
o['rebuild_vs_official'] = {'n_risk': len(risk), 'units_equal': set(pr.bz) == risk, 'pos_rebuilt': len(pos & risk),
                            'y_equal': bool((pr.bz.isin(pos).astype(int) == pr.y).all())}
kr = duckdb.sql(f"select appl_no, max(rgst_no) rg, max(appl_date) ad, max(pub_date) pdt, max(rgst_date) rd from read_parquet('{M}') where cc='KR' group by 1").df()
for c in ['ad', 'pdt', 'rd']:
    kr[c] = pd.to_numeric(kr[c], errors='coerce')
blank = kr[kr.rg.isna() | (kr.rg.astype(str).str.strip() == '')]
o['kr_total'] = len(kr); o['kr_blank'] = len(blank)
con = sqlite3.connect('file:' + B + r"\수집파이프라인\business_rights.sqlite" + '?mode=ro', uri=True)
rl = pd.read_sql("select applicant_cd cd, rgst_no, rgst_date, appl_no, appl_date from right_list", con)
aq = pd.read_sql("select applicant_cd cd, status, truncated, n_in_fusion, total_count from applicant_query", con)
con.close()
o['rl_rows'] = len(rl)
rl['appl_no'] = rl.appl_no.astype(str).str.replace(r'\D', '', regex=True)
rl['rgst_no'] = rl.rgst_no.astype(str)
o['rl_applno_len'] = rl.appl_no.str.len().value_counts().head(5).to_dict()
nb = kr[kr.rg.notna()]
m1 = rl.merge(nb[['appl_no', 'rg']], on='appl_no')
o['nonblank_applno_match_rows'] = len(m1)
o['nonblank_rgst_agree'] = float((m1.rgst_no.str.replace(r'\D', '', regex=True) == m1.rg.astype(str)).mean())
rlp = rl[rl.rgst_no.str[:2] == '10']
o['rl_patent_rows'] = len(rlp)
rlb = rl[rl.appl_no.isin(set(blank.appl_no))]
o['blank_docs_in_rl_any_prefix'] = int(rlb.appl_no.nunique())
o['blank_docs_rl_prefix'] = rlb.drop_duplicates('appl_no').rgst_no.str[:2].value_counts().to_dict()
rlpb = rlp[rlp.appl_no.isin(set(blank.appl_no))].drop_duplicates(['cd', 'appl_no']).merge(blank, on='appl_no')
rlpb['rdt'] = pd.to_numeric(rlpb.rgst_date, errors='coerce')
o['blank_docs_patent'] = int(rlpb.appl_no.nunique()); o['blank_min_rgst'] = float(rlpb.rdt.min())
bf = fl.merge(rlpb, on='cd')
o['blank_docs_linked_pop'] = int(bf.appl_no.nunique()); o['pop_firms_blank'] = int(bf.bz.nunique())
cap = bf.merge(fa[['bz', 'appl_no']].drop_duplicates(), on=['bz', 'appl_no'], how='left', indicator=True)
o['pairs_captured'] = int((cap._merge == 'both').sum()); o['pairs_missed'] = int((cap._merge == 'left_only').sum())
miss = cap[cap._merge == 'left_only']
newpos = set(miss[(miss.ad >= 20220101) & (miss.ad <= 20231231) & miss.bz.isin(risk)].bz) - pos
drop = set(miss[(miss.pdt <= 20211231) & miss.bz.isin(risk)].bz)
o['newpos'] = len(newpos); o['drop'] = len(drop); o['drop_overlap_newpos'] = len(drop & newpos)
def sk(d, ycol):
    A = roc_auc_score(d[ycol], d.p_champ); Bv = roc_auc_score(d[ycol], d.p_m0); return [A, Bv, (A - Bv) / (1 - Bv)]
d = pr.copy(); d['y2'] = (d.y | d.bz.isin(newpos)).astype(int)
r0 = sk(d, 'y'); r1 = sk(d, 'y2'); d3 = d[~d.bz.isin(drop)]; r2 = sk(d3, 'y2')
o['rescore'] = {'off': r0, 'add': r1, 'add_drop': r2, 'dsk_add': r1[2] - r0[2], 'dsk_add_drop': r2[2] - r0[2], 'pos2': int(d3.y2.sum()), 'n2': len(d3)}
kp = pd.read_parquet(B + r"\공모전데이터\작업데이터\kr_patents.parquet", columns=["출원번호", "출원일", "출원인명"])
kp.columns = ['appl_no', 'ad', 'names']
ca_names = ns.set_index('applicantCd').aname.to_dict()
np_rows = miss[miss.bz.isin(newpos) & (miss.ad >= 20220101) & (miss.ad <= 20231231)].merge(kp[['appl_no', 'names']], on='appl_no', how='left')
def namehit(r):
    n = nrm(ca_names.get(r.cd, '')); ns_ = [nrm(x) for x in str(r.names or '').split('|')]
    return n in ns_
np_rows['name_hit'] = np_rows.apply(namehit, axis=1)
fh = np_rows.groupby('bz').name_hit.max()
o['newpos_docs'] = len(np_rows); o['newpos_firms_with_name_in_doc'] = int(fh.sum())
o['newpos_examples'] = np_rows.head(10)[['bz', 'cd', 'appl_no', 'ad', 'names', 'name_hit']].assign(cdname=lambda x: x.cd.map(ca_names)).astype(str).values.tolist()
o['newpos_how'] = pd.Series([how[b] for b in newpos]).value_counts().to_dict()
aq_ok = set(aq[aq.status == 'OK'].cd); aq_all = set(aq.cd)
fr = fl[fl.bz.isin(risk)].copy(); fr['q_ok'] = fr.cd.isin(aq_ok); fr['q_any'] = fr.cd.isin(aq_all)
fq = fr.groupby('bz')[['q_ok', 'q_any']].max()
yy = pr.set_index('bz').y
cov = {}
for yv in (0, 1):
    ids = yy[yy == yv].index
    sub = fq.reindex(ids)
    cov[f'y{yv}'] = {'n': len(ids), 'has_code': int(sub.q_ok.notna().sum()), 'queried_any': int(sub.q_any.fillna(False).astype(bool).sum()),
                     'queried_ok': int(sub.q_ok.fillna(False).astype(bool).sum())}
o['risk_coverage_by_y'] = cov
o['aq_truncated_ok'] = int(((aq.status == 'OK') & (aq.truncated == 1)).sum())
kpw = kp.copy()
kpw['ad'] = pd.to_numeric(kpw.ad.astype(str).str.replace(r'\D', '', regex=True).str[:8], errors='coerce')
kpw = kpw[(kpw.ad >= 20220101) & (kpw.ad <= 20231231)]
o['window_apps'] = int(kpw.appl_no.nunique())
kpw['nm'] = kpw.names.fillna('').str.split('|'); kpw = kpw.explode('nm'); kpw['n'] = kpw.nm.map(nrm)
wn = set(kpw.n)
rows = []
with open(B + r"\공모전데이터\외부데이터\Corporate_20260720\CORP_APPLICANT.txt", encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("¶")
        if len(p) >= 2:
            rows.append((p[0].strip(), p[1].strip()))
ca = pd.DataFrame(rows, columns=['cd', 'aname']); ca['n'] = ca.aname.map(nrm)
cnt = ca.groupby('n').cd.nunique().to_dict(); cdn = dict(zip(ca.cd, ca.n))
y0 = set(yy[yy == 0].index)
f0 = fl[fl.bz.isin(y0)].copy(); f0['n'] = f0.cd.map(cdn)
f0['in_window'] = f0.n.isin(wn) & (f0.n.fillna('').str.len() >= 2); f0['uniq'] = f0.n.map(cnt) == 1
cand = f0[f0.in_window]
o['y0_firms_codename_in_window_names'] = int(cand.bz.nunique())
o['y0_cand_by_uniq'] = {str(k): int(v) for k, v in cand.groupby('uniq').bz.nunique().items()}
o['y0_cand_ncodes_sharing_name'] = cand.n.map(cnt).describe().to_dict()
cq = cand.merge(fq.reset_index(), on='bz', how='left')
o['y0_cand_queried_ok'] = int(cq[cq.q_ok == True].bz.nunique())
o['y0_cand_not_queried'] = int(cq[cq.q_ok != True].bz.nunique())
cand_set = set(cand.bz)
o['newpos_in_cand'] = len(newpos & cand_set)
nq = set(cq[cq.q_ok != True].bz) - newpos
sc = {}
rng = np.random.default_rng(0)
for frac in (1.0, 0.5, 0.25, 0.1):
    v = []
    for i in range(50 if frac < 1 else 1):
        s = set(rng.choice(sorted(nq), int(round(len(nq) * frac)), replace=False)) if frac < 1 else nq
        dd = pr.copy(); dd['y3'] = (dd.y | dd.bz.isin(newpos | s)).astype(int); v.append(sk(dd, 'y3')[2] - r0[2])
    sc[str(frac)] = [float(np.mean(v)), float(np.min(v)), float(np.max(v))]
o['scenario_unqueried_cand_flip_dskill'] = sc
o['sec'] = round(time.time() - t0)
pd.Series(sorted(cand_set)).to_csv(V + r"\v1_y0_name_candidates.csv", index=False, header=['bz'])
json.dump(o, open(V + r"\v1_out.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(o, ensure_ascii=False, indent=1, default=str))
