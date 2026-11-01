# C03-T1 s2: 등록번호 공란 등록(권리목록으로 확인) 복원 — 원출원인 법인 추정(R_self·R_any), 보정(등록원부 대비 정밀도·재현율),
#            최종 진입창(출원 2020–23) 건 단위 분해(P3), 개발 원점용 공란 쌍(출원 2009–2016)도 저장.
import os, sys, re, json, time, sqlite3, pathlib
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
t0 = time.time()
ROOT = pathlib.Path(r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전")
T1 = ROOT/'분석/모델경쟁_20260927/T1_산업대기실'; R3 = T1/'R3'; TC = T1/'cache'
W = ROOT/'분석/검증루프_20261012/C03/C03-T1'; WC = W/'cache'
DATA = ROOT/'공모전데이터/작업데이터'; EXTD = ROOT/'분석/모델경쟁_20260927/_shared/ext'
END = '20250822'
def ro(p): return sqlite3.connect(pathlib.Path(p).as_uri() + '?mode=ro', uri=True)
STRIP = re.compile(r"주식회사|\(주\)|㈜|유한회사|유한책임회사|재단법인|사단법인|학교법인|가부시키가이샤|가부시끼가이샤|가부시기가이샤|가부시키카이샤|코포레이션|인코포레이티드|리미티드|컴퍼니|[\s\.\,\-\(\)\'\"·&]")
norm = lambda x: STRIP.sub('', str(x)).lower()
K = sorted(pd.read_parquet(TC/'ins_pat.parquet', columns=['ksic']).ksic.astype(str).unique()); kidx = {k: i for i, k in enumerate(K)}

# ---- 제공 KR 전체(출원 2009–2024) ----
k = pd.read_parquet(DATA/'kr_patents.parquet', columns=['출원번호', '출원연도', '등록번호', '공개일자', '출원인명', '세세분류코드'])
k.columns = ['appl_no', 'ay', 'rgst_no', 'pub_date', 'names', 'ksic']; k['ay'] = k.ay.astype(int); k['ksic'] = k.ksic.astype(str)
k = k[(k.ay >= 2009) & (k.ay <= 2024)].copy(); k['blank'] = k.rgst_no.fillna('').astype(str).str.len() <= 3
# ---- 권리목록(T1 cache rl_pat = business_rights.right_list 중 등록번호 '10') ----
rl = pd.read_parquet(TC/'rl_pat.parquet', columns=['applicant_cd', 'rgst_no', 'rgst_date', 'appl_no', 'pert', 'tp', 'ainfo', 'hinfo'])
rl = rl[rl.rgst_date <= END]
# s10 재현: 공란 등록 출원연도별(권리목록 아무 행)
bl = k[k.blank & k.appl_no.isin(set(rl.appl_no)) & (k.appl_no.str[:2] == '10')]
s10 = bl.groupby('ay').size().to_dict()
print('s10 재현', {y: s10.get(y, 0) for y in range(2016, 2025)}, 'sum 2020-23', sum(s10.get(y, 0) for y in range(2020, 2024)), round(time.time() - t0), flush=True)
# ---- 별칭(원천판, s1에서 저장) ----
al = pd.read_parquet(WC/'alias_src.parquet')
alias = al.groupby('applicant_cd').n.apply(list).to_dict()
n2c = dict(zip(al[~al.ambiguous].n, al[~al.ambiguous].applicant_cd)); amb_set = set(al[al.ambiguous].n)
fset = set(al.applicant_cd)
def is_self(cd, ainfo):
    a = norm(ainfo)
    for n in alias.get(cd, []):
        if len(n) >= 3 and n in a: return True
        if len(n) < 3 and any(norm(p) == n for p in str(ainfo).split(',')): return True
    return False
rl = rl[rl.applicant_cd.isin(fset) & (rl.pert == '권리자란')].copy()
# ---- 보정: 등록번호 있는 범위안 등록특허(출원 2020–23)에서 권리목록 규칙 vs 등록원부 원출원인 코드 ----
kr = pd.read_parquet(DATA/'kr_registered.parquet', columns=['출원번호', '출원연도', '등록번호']); kr.columns = ['appl_no', 'ay', 'rgst_no']; kr['ay'] = kr.ay.astype(int); kr['rgst_no'] = kr.rgst_no.astype(str)
app = pd.read_sql('select rgst_no, applicantCd from applicant', ro(ROOT/'수집파이프라인/registry_full.sqlite'))
truth = app[app.applicantCd.isin(fset)].merge(kr, on='rgst_no')[['appl_no', 'applicantCd', 'ay']].drop_duplicates()
del app
cal = {}
for lo, hi in ((2020, 2023), (2009, 2016)):
    tw = truth[(truth.ay >= lo) & (truth.ay <= hi)]; aw = set(kr[(kr.ay >= lo) & (kr.ay <= hi)].appl_no)
    r = rl[rl.appl_no.isin(aw)].copy(); r['self'] = [is_self(c_, a_) for c_, a_ in zip(r.applicant_cd.values, r.ainfo.values)]
    T = set(zip(tw.appl_no, tw.applicantCd)); Sself = set(zip(r[r.self].appl_no, r[r.self].applicant_cd)); Sany = set(zip(r.appl_no, r.applicant_cd))
    cal[f'{lo}-{hi}'] = dict(truth_pairs=len(T), R_self_pairs=len(Sself), R_any_pairs=len(Sany),
                             R_self_precision=len(T & Sself) / max(len(Sself), 1), R_self_recall=len(T & Sself) / max(len(T), 1),
                             R_any_precision=len(T & Sany) / max(len(Sany), 1), R_any_recall=len(T & Sany) / max(len(T), 1),
                             truth_patents_with_rl_row=len({a for a, _ in T} & set(r.appl_no)) / max(len({a for a, _ in T}), 1))
print('보정', json.dumps(cal, ensure_ascii=False), round(time.time() - t0), flush=True)
del truth
# ---- 공란 등록 × 원출원인 법인 ----
bl = k[k.blank & (k.appl_no.str[:2] == '10')][['appl_no', 'ay', 'ksic', 'pub_date', 'names']]
r = rl.merge(bl, on='appl_no')
r['self'] = [is_self(c_, a_) for c_, a_ in zip(r.applicant_cd.values, r.ainfo.values)]
r = r.drop_duplicates(['appl_no', 'applicant_cd'])
r['k'] = r.ksic.map(kidx)
# 이름 매칭(공개 경로)과 같은 규칙으로 이 출원의 출원인명이 그 법인에 매칭되는지, 안 되면 사유
def why(names, cd):
    nn = [norm(x) for x in str(names).split('|')]
    if any(n2c.get(n) == cd for n in nn): return 'matched'
    own = set(alias.get(cd, []))
    if any(n in own and n in amb_set for n in nn): return 'ambiguous_alias_removed'
    if any(n in n2c for n in nn): return 'matched_other_firm_only'
    if any(n in own for n in nn): return 'alias_short_or_filtered'
    return 'no_alias_match'
r['pubmatch'] = [why(n_, c_) for n_, c_ in zip(r.names.values, r.applicant_cd.values)]
r = r.drop(columns=['names'])
r.to_parquet(WC/'blank_pairs_2009_2024.parquet', index=False)
print('blank pairs', len(r), 'patents', r.appl_no.nunique(), 'self pairs', int(r.self.sum()), round(time.time() - t0), flush=True)

# ---- 최종 진입창 분해 ----
ev = pd.read_parquet(WC/'eval_o2019.parquet', columns=['firm', 'k', 'newc', 'yA_reg', 'yA_pub'])
te_all = pd.read_parquet(R3/'cache_final/o2019/prep_te.parquet', columns=['firm', 'k', 'prior_any'])
evk = {(f_, int(k_)): (int(a_), int(b_)) for f_, k_, a_, b_ in zip(ev.firm.values, ev.k.values, ev.yA_reg.values, ev.yA_pub.values)}; prior_set = set(zip(te_all.firm[te_all.prior_any], te_all.k[te_all.prior_any].astype(int)))
fw = r[(r.ay >= 2020) & (r.ay <= 2023)].copy()
blank_win = k[k.blank & (k.ay >= 2020) & (k.ay <= 2023) & (k.appl_no.str[:2] == '10') & k.appl_no.isin(set(pd.read_parquet(TC/'rl_pat.parquet', columns=['appl_no', 'rgst_date']).query('rgst_date <= @END').appl_no))]
def cls(row):
    if pd.isna(row.k): return '1_KSIC50밖'
    key = (row.applicant_cd, int(row.k))
    if key in prior_set: return '3b_prior_any제외'
    if key not in evk: return '3a_위험집합밖'
    e_ = evk[key]
    if e_[0] == 1: return '4_이미y_reg'
    if e_[1] == 1: return '5a_같은출원공개매칭' if row.pubmatch == 'matched' else '5b_다른출원으로y_pub'
    return '6_새양성'
fw['cat'] = [cls(x) for x in fw.itertuples()]
fw['k'] = fw.k.astype('Int64')
fw.to_csv(W/'blank_reg_decomp.csv', index=False, encoding='utf-8-sig')
pri = ['6_새양성', '5a_같은출원공개매칭', '5b_다른출원으로y_pub', '4_이미y_reg', '3b_prior_any제외', '3a_위험집합밖', '1_KSIC50밖']
out = dict(s10_reproduced={str(y): int(s10.get(y, 0)) for y in range(2016, 2025)}, blank_window_patents_any_rl=int(len(blank_win)), calibration=cal)
for lab, sub in (('R_self', fw[fw.self]), ('R_any', fw)):
    pc = sub.groupby('appl_no').cat.apply(lambda x: min(x, key=pri.index))
    n_pat = len(blank_win); out[lab] = dict(
        patents_with_firm=int(sub.appl_no.nunique()), patents_no_firm=int(n_pat - sub.appl_no.nunique()),
        patent_level={c_: int((pc == c_).sum()) for c_ in pri}, pair_level={c_: int((sub.cat == c_).sum()) for c_ in pri},
        new_pos_pairs_unique=int(sub[sub.cat == '6_새양성'][['applicant_cd', 'k']].drop_duplicates().shape[0]),
        new_pos_pairs_newc=int(sub[sub.cat == '6_새양성'][['applicant_cd', 'k']].drop_duplicates().merge(ev[ev.newc][['firm', 'k']].astype({'k': 'int64'}), left_on=['applicant_cd', 'k'], right_on=['firm', 'k']).shape[0]),
        pubmatch=sub.pubmatch.value_counts().to_dict(),
        pubmatch_in_eval=sub[sub.cat.str[0].isin(['4', '5', '6'])].pubmatch.value_counts().to_dict())
# ---- 분포: 공란 vs 등록번호 있는 등록(원출원인 T1 법인, 출원 2020–23) ----
ins = pd.read_parquet(TC/'ins_firm.parquet', columns=['rgst_no', 'applicant_cd', 'ksic', 'ay']); ins = ins[(ins.ay >= 2020) & (ins.ay <= 2023)]
fm = pd.read_parquet(TC/'firm_meta.parquet', columns=['applicant_cd', 'tp']).set_index('applicant_cd').tp
P = pd.read_parquet(EXTD/'firm_nps_firm_panel.parquet', columns=['applicantCd', 'snap']); nps19 = set(P[P.snap == '2019-12'].applicantCd.astype(str)); del P
fsz = pd.read_parquet(TC/'ins_firm.parquet', columns=['applicant_cd', 'ay']).query('ay <= 2019').groupby('applicant_cd').size()
def attrs(d, cd, ay, ks):
    x = pd.DataFrame({'ay': d[ay].values, 'ksic': d[ks].astype(str).values, 'cd': d[cd].values})
    x['natl'] = x.cd.map(fm).fillna('결측').map(lambda v: v if v in ('국내법인', '외국법인') else '기타')
    x['nps2019'] = np.where(x.cd.isin(nps19), 'NPS연결', '미연결')
    sz = x.cd.map(fsz).fillna(0).values; x['size'] = pd.cut(sz, [-1, 2, 20, 200, 1e9], labels=['≤2건', '3–20', '21–200', '200+']).astype(str)
    return x
A = attrs(fw[fw.self], 'applicant_cd', 'ay', 'ksic'); Bn = attrs(ins, 'applicant_cd', 'ay', 'ksic')
dist = []; tot_share = len(A) / (len(A) + len(Bn))
for dim in ('ay', 'natl', 'nps2019', 'size', 'ksic'):
    ca, cb = A[dim].value_counts(), Bn[dim].value_counts()
    for v in sorted(set(ca.index) | set(cb.index), key=str):
        na, nb = int(ca.get(v, 0)), int(cb.get(v, 0)); sh = na / max(na + nb, 1)
        dist.append(dict(dim=dim, value=str(v), blank_pairs=na, reg_pairs=nb, blank_share=sh, ratio_to_overall=sh / tot_share, concentrated=bool(sh / tot_share >= 1.5 and na >= 30)))
dd = pd.DataFrame(dist); dd.to_csv(WC/'blank_dist.csv', index=False, encoding='utf-8-sig')
out['overall_blank_share_firm_pairs'] = tot_share
out['concentrated'] = dd[dd.concentrated][['dim', 'value', 'blank_pairs', 'blank_share', 'ratio_to_overall']].to_dict('records')
out['by_ay'] = dd[dd.dim == 'ay'].to_dict('records'); out['by_natl'] = dd[dd.dim == 'natl'].to_dict('records')
out['by_nps2019'] = dd[dd.dim == 'nps2019'].to_dict('records'); out['by_size'] = dd[dd.dim == 'size'].to_dict('records')
ks = dd[dd.dim == 'ksic'].sort_values('blank_share'); out['ksic_share_range'] = [float(ks.blank_share.min()), float(ks.blank_share.median()), float(ks.blank_share.max())]
# 새 양성 분포
npz = fw[fw.self & (fw.cat == '6_새양성')]
out['new_pos_by_ay'] = npz.groupby('ay').size().to_dict(); out['new_pos_by_natl'] = attrs(npz, 'applicant_cd', 'ay', 'ksic').natl.value_counts().to_dict()
out['new_pos_top_ksic'] = npz.ksic.value_counts().head(8).to_dict()
json.dump(out, open(WC/'s2_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps({x: out[x] for x in ('blank_window_patents_any_rl', 'R_self', 'R_any', 'overall_blank_share_firm_pairs', 'concentrated', 'ksic_share_range')}, ensure_ascii=False, indent=1, default=str))
print('done', round(time.time() - t0))
