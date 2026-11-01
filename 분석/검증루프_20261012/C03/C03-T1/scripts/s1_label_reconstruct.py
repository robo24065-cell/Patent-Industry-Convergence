# C03-T1 s1: T1 최종(원점 2019, 진입 2020–23) 공식 평가 레이블 독립 재구성 + 동결 예측 재채점 (P1)
# 읽기 전용: 원자료·트랙 폴더·sqlite(mode=ro). 쓰기: C03\C03-T1\cache, label_reconstruct.json
import os, sys, re, json, time, sqlite3, pathlib
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
sys.stdout.reconfigure(encoding='utf-8')
t0 = time.time()
ROOT = pathlib.Path(r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전")
T1 = ROOT/'분석/모델경쟁_20260927/T1_산업대기실'; R3 = T1/'R3'; TC = T1/'cache'
W = ROOT/'분석/검증루프_20261012/C03/C03-T1'; WC = W/'cache'; WC.mkdir(exist_ok=True)
DATA = ROOT/'공모전데이터/작업데이터'
END = '20250822'; C = 2019; LO, HI = 2020, 2023

def ro(p): return sqlite3.connect(pathlib.Path(p).as_uri() + '?mode=ro', uri=True)

# ---------- 평가 집합과 동결 점수 ----------
o = pd.read_parquet(R3/'runs/FROZEN_final/o2019.parquet')
te = pd.read_parquet(R3/'cache_final/o2019/prep_te.parquet', columns=['firm', 'k', 'cut', 'newc', 'prior_any'])
assert (te.firm.values == o.firm.values).all() and (te.k.values == o.k.values).all()
bs = pd.read_parquet(R3/'cache_final/o2019/baseline_scores.parquet')
keep = ~te.prior_any.values
t = te[keep].reset_index(drop=True); oo = o[keep].reset_index(drop=True); bs = bs.loc[keep].reset_index(drop=True)
rank = lambda s: pd.Series(s).rank(pct=True).values
s = 0.9 * rank(oo.px.values) + 0.1 * rank(oo.pl.values)
print('eval rows', len(t), 'removed prior', int((~keep).sum()), round(time.time() - t0), flush=True)

K = sorted(pd.read_parquet(TC/'ins_pat.parquet', columns=['ksic']).ksic.astype(str).unique()); kidx = {k: i for i, k in enumerate(K)}

def pair_label(d, e):
    e = e[['applicant_cd', 'k']].drop_duplicates().assign(z=1)
    x = d[['firm', 'k']].merge(e, left_on=['firm', 'k'], right_on=['applicant_cd', 'k'], how='left')
    return x.z.fillna(0).values.astype(np.int8)

# ---------- (A) 캐시 기반(ins_firm·app_pub — T1 코드와 같은 입력) ----------
ins = pd.read_parquet(TC/'ins_firm.parquet'); ins = ins[ins.ay >= 2000]
e = ins[(ins.ay > C) & (ins.ay <= C + 4)].copy(); e['k'] = e.ksic.astype(str).map(kidx)
yA_reg = pair_label(t, e.dropna(subset=['k']).astype({'k': int}))
a = pd.read_parquet(TC/'app_pub.parquet'); a['k'] = a.ksic.astype(str).map(kidx); a = a.dropna(subset=['k']); a['k'] = a.k.astype(int)
a['pub_date'] = a.pub_date.fillna('').astype(str); a['rgst_date'] = a.rgst_date.fillna('').astype(str); a = a[a.ay >= 2000]
vis = ((a.pub_date.str.len() == 8) & (a.pub_date <= END)) | ((a.rgst_date.str.len() == 8) & (a.rgst_date <= END))
yA_pub = pair_label(t, a[vis & (a.ay > C) & (a.ay <= C + 4)])
print('cache labels', yA_reg.sum(), yA_pub.sum(), round(time.time() - t0), flush=True)
del ins, a, e

# ---------- (B) 원천 독립 재구성 ----------
kr = pd.read_parquet(DATA/'kr_registered.parquet', columns=['출원번호', '출원연도', '등록번호', '등록일자', '세세분류코드'])
kr.columns = ['appl_no', 'ay', 'rgst_no', 'rgst_date', 'ksic']; kr['rgst_no'] = kr.rgst_no.astype(str); kr['ksic'] = kr.ksic.astype(str); kr['ay'] = kr.ay.astype(int)
K_src = sorted(kr.ksic.unique()); assert K_src == K, 'KSIC 목록 불일치'
firms = pd.read_sql('select applicant_cd, applicant_name, status from applicant_query', ro(ROOT/'수집파이프라인/business_rights.sqlite'))
firms = firms[firms.status == 'OK']; fset = set(firms.applicant_cd)
app = pd.read_sql('select rgst_no, applicantCd, applicantName from applicant', ro(ROOT/'수집파이프라인/registry_full.sqlite'))
insB = app[app.applicantCd.isin(fset)][['rgst_no', 'applicantCd']].rename(columns={'applicantCd': 'applicant_cd'}).merge(kr, on='rgst_no').drop_duplicates(['rgst_no', 'applicant_cd'])
insB = insB[insB.ay >= 2000]; insB['k'] = insB.ksic.map(kidx)
eB = insB[(insB.ay > C) & (insB.ay <= C + 4)]
yB_reg = pair_label(t, eB)
reg_cov = dict(kr_window=int(((kr.ay > C) & (kr.ay <= C + 4)).sum()), kr_window_in_registry=int(kr[(kr.ay > C) & (kr.ay <= C + 4)].rgst_no.isin(set(app.rgst_no)).sum()))
print('src reg', yB_reg.sum(), reg_cov, round(time.time() - t0), flush=True)
del insB
STRIP = re.compile(r"주식회사|\(주\)|㈜|유한회사|유한책임회사|재단법인|사단법인|학교법인|가부시키가이샤|가부시끼가이샤|가부시기가이샤|가부시키카이샤|코포레이션|인코포레이티드|리미티드|컴퍼니|[\s\.\,\-\(\)\'\"·&]")
norm = lambda x: STRIP.sub('', str(x)).lower()
al = pd.concat([firms[['applicant_cd', 'applicant_name']].rename(columns={'applicant_name': 'nm'}),
                app[app.applicantCd.isin(fset)][['applicantCd', 'applicantName']].rename(columns={'applicantCd': 'applicant_cd', 'applicantName': 'nm'})])
del app
al['n'] = al.nm.map(norm); al = al[al.n.str.len() >= 2].drop_duplicates(['applicant_cd', 'n'])
amb = al.groupby('n').applicant_cd.nunique(); amb_set = set(amb[amb > 1].index); al1 = al[al.n.map(amb) == 1]
n2c = dict(zip(al1.n, al1.applicant_cd))
pd.DataFrame({'n': al.n, 'applicant_cd': al.applicant_cd, 'ambiguous': al.n.isin(amb_set)}).to_parquet(WC/'alias_src.parquet', index=False)
k = pd.read_parquet(DATA/'kr_patents.parquet', columns=['출원번호', '출원연도', '등록번호', '등록일자', '공개일자', '출원인명', '세세분류코드'])
k.columns = ['appl_no', 'ay', 'rgst_no', 'rgst_date', 'pub_date', 'names', 'ksic']; k['ksic'] = k.ksic.astype(str); k['ay'] = k.ay.astype(int)
k = k[(k.ay > C) & (k.ay <= C + 4)].copy()
k['nm'] = k.names.fillna('').str.split('|'); k = k.explode('nm'); k['nn'] = k.nm.map(norm); k['applicant_cd'] = k.nn.map(n2c)
m = k.dropna(subset=['applicant_cd']).drop_duplicates(['appl_no', 'applicant_cd']).copy()
m['k'] = m.ksic.map(kidx); m = m.dropna(subset=['k']); m['k'] = m.k.astype(int)
pdt = m.pub_date.fillna('').astype(str); rdt = m.rgst_date.fillna('').astype(str)
visB = ((pdt.str.len() == 8) & (pdt <= END)) | ((rdt.str.len() == 8) & (rdt <= END))
yB_pub = pair_label(t, m[visB])
m[['appl_no', 'applicant_cd', 'ay', 'ksic', 'k', 'pub_date', 'rgst_date']].to_parquet(WC/'app_pub_src_2020_23.parquet', index=False)
k[['appl_no', 'nm', 'nn']].to_parquet(WC/'kr_names_2020_23.parquet', index=False)
print('src pub', yB_pub.sum(), round(time.time() - t0), flush=True)

# ---------- 채점 ----------
def mets(y, sc, nc):
    return dict(auc_all=float(roc_auc_score(y, sc)), pr_all=float(average_precision_score(y, sc)),
                auc_new=float(roc_auc_score(y[nc], sc[nc])), pr_new=float(average_precision_score(y[nc], sc[nc])))
skill = lambda m_, b_: (m_ - b_) / (1 - b_)
nc = t.newc.values.astype(bool)
def score(yr, yp):
    y = np.maximum(yr, yp)
    mm = mets(y, s, nc); mm['auc_all_reglabel'] = float(roc_auc_score(yr, s))
    bm = {kk: mets(y, bs[kk].values, nc) for kk in bs.columns}
    st = {q: max(v[q] for v in bm.values()) for q in ('auc_all', 'auc_new', 'pr_all', 'pr_new')}
    return dict(n=dict(pos=int(y.sum()), pos_reg=int(yr.sum()), pos_pub=int(yp.sum()), new_pos=int(y[nc].sum()), rows=int(len(y)), new_rows=int(nc.sum())),
                model=mm, baseline_strongest=st, strongest_arg=max(bm, key=lambda q: bm[q]['auc_all']),
                skill=dict(auc_all=skill(mm['auc_all'], st['auc_all']), auc_new=skill(mm['auc_new'], st['auc_new'])))
off = json.load(open(R3/'final_result_r3.json', encoding='utf-8'))
res = dict(definition='공식 평가 레이블 y_u = max(y_reg[등록원부 원출원인 코드, 출원 2020–23, cap 없음], y_pub[kr_patents 출원인명→법인 별칭, 출원 2020–23, 공개·등록일 ≤ 2025-08-22]); 채점 prior_any=False',
           frozen_preds='R3/runs/FROZEN_final/o2019.parquet (px·pl), blend 0.9*rank(px)+0.1*rank(pl) — final_run 재실행 없음',
           registry_coverage_window=reg_cov,
           cache=score(yA_reg, yA_pub), source=score(yB_reg, yB_pub),
           official=dict(n=off['n'], model=off['model'], baseline_strongest=off['baseline_strongest'], skill=off['skill']),
           label_agreement=dict(reg_cache_vs_src_diff=int((yA_reg != yB_reg).sum()), pub_cache_vs_src_diff=int((yA_pub != yB_pub).sum()),
                                reg_only_cache=int(((yA_reg == 1) & (yB_reg == 0)).sum()), reg_only_src=int(((yA_reg == 0) & (yB_reg == 1)).sum()),
                                pub_only_cache=int(((yA_pub == 1) & (yB_pub == 0)).sum()), pub_only_src=int(((yA_pub == 0) & (yB_pub == 1)).sum())))
for nm_ in ('cache', 'source'):
    r = res[nm_]; d = {}
    for q in ('pos', 'pos_reg', 'new_pos', 'rows', 'new_rows'): d[q] = r['n'][q] - off['n'][q]
    for q in ('auc_all', 'auc_new', 'pr_all', 'pr_new', 'auc_all_reglabel'): d[q] = r['model'][q] - off['model'][q]
    for q in ('auc_all', 'auc_new'): d['strongest_' + q] = r['baseline_strongest'][q] - off['baseline_strongest'][q]; d['skill_' + q] = r['skill'][q] - off['skill'][q]
    r['diff_vs_official'] = d
    r['P1'] = '재현' if all(d[q] == 0 for q in ('pos', 'pos_reg', 'new_pos')) and all(abs(d[q]) <= 1e-3 for q in d if q not in ('pos', 'pos_reg', 'new_pos', 'rows', 'new_rows')) else '불일치'
json.dump(res, open(W/'label_reconstruct.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
ev = pd.DataFrame(dict(firm=t.firm.values, k=t.k.values.astype(np.int16), newc=nc, s=s.astype(np.float64), yA_reg=yA_reg, yA_pub=yA_pub, yB_reg=yB_reg, yB_pub=yB_pub))
for c_ in bs.columns: ev['b_' + c_] = bs[c_].values
ev.to_parquet(WC/'eval_o2019.parquet', index=False)
print(json.dumps({x: (res[x]['n'], res[x]['P1'], res[x]['diff_vs_official']) for x in ('cache', 'source')}, ensure_ascii=False, indent=0)); print(res['label_agreement'])
print('done', round(time.time() - t0))
