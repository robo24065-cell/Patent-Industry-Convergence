# -*- coding: utf-8 -*-
"""C18-T1 s5: T1 P0(공식 비트 재현) + 공백 달 누락 등록 보충 y_reg 재구성(S1 R_self, S2 R_any, S1b 모든 달) 점추정.
동결 예측(R3/runs/FROZEN_final/o2019.parquet)과 동결 기준선만 쓴다. 재학습 없음. 읽기 전용. 쓰기: C18-T1/cache."""
import sys, re, json, time, sqlite3, pathlib
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.dont_write_bytecode = True
t0 = time.time()
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
H = B/'분석/검증루프_20261012/C18/C18-T1'; C = H/'cache'
T1 = B/'분석/모델경쟁_20260927/T1_산업대기실'; R3 = T1/'R3'; TC = T1/'cache'
ro = lambda p: sqlite3.connect(pathlib.Path(p).as_uri() + '?mode=ro', uri=True)
END = '20250822'
BCOLS = ['logit_stack_y', 'logit_single_y', 'xgb_y', 'logit_stack_y_u', 'logit_single_y_u', 'xgb_y_u']
rank = lambda s: pd.Series(s).rank(pct=True).values
skill = lambda m, b: (m - b) / (1 - b)
o = {}
# ---------- P0: 동결 예측·동결 레이블 ----------
fz = pd.read_parquet(R3/'runs/FROZEN_final/o2019.parquet')
bs = pd.read_parquet(R3/'cache_final/o2019/baseline_scores.parquet')
keep = ~fz.prior_any.values
fz = fz[keep].reset_index(drop=True); bs = bs.loc[keep].reset_index(drop=True)
s = 0.9 * rank(fz.px.values) + 0.1 * rank(fz.pl.values)
nc = fz.newc.values.astype(bool)
o['frozen_y_u_sum'] = int(fz.y_u.sum())   # 동결 파일의 y_u는 봉인(0) → 원천에서 공식 레이블 재구성
# ---------- 공식 레이블 원천 재구성(final_run_r3 score_final 정의; C03-T1 P1과 같은 규칙, 독립 구현) ----------
K0 = sorted(pd.read_parquet(TC/'ins_pat.parquet', columns=['ksic']).ksic.astype(str).unique()); kidx0 = {k: i for i, k in enumerate(K0)}
STRIP0 = re.compile(r"주식회사|\(주\)|㈜|유한회사|유한책임회사|재단법인|사단법인|학교법인|가부시키가이샤|가부시끼가이샤|가부시기가이샤|가부시키카이샤|코포레이션|인코포레이티드|리미티드|컴퍼니|[\s\.\,\-\(\)\'\"·&]")
norm0 = lambda x: STRIP0.sub('', str(x)).lower()
def pair_label(e):
    ks = set(zip(e.applicant_cd, e.k.astype(int)))
    return np.fromiter(((f_, int(k_)) in ks for f_, k_ in zip(fz.firm.values, fz.k.values)), bool, len(fz)).astype(np.int8)
kr = pd.read_parquet(B/'공모전데이터/작업데이터/kr_registered.parquet', columns=['출원번호', '출원연도', '등록번호', '세세분류코드'])
kr.columns = ['appl_no', 'ay', 'rgst_no', 'ksic']; kr['rgst_no'] = kr.rgst_no.astype(str); kr['ksic'] = kr.ksic.astype(str); kr['ay'] = kr.ay.astype(int)
fq = pd.read_sql('select applicant_cd, applicant_name, status from applicant_query', ro(B/'수집파이프라인/business_rights.sqlite'))
fq = fq[fq.status == 'OK']; fset0 = set(fq.applicant_cd)
_rc = ro(B/'수집파이프라인/registry_full.sqlite'); ap0 = pd.read_sql('select rgst_no, applicantCd, applicantName from applicant', _rc); _rc.close()
ins = ap0[ap0.applicantCd.isin(fset0)][['rgst_no', 'applicantCd']].rename(columns={'applicantCd': 'applicant_cd'}).merge(kr, on='rgst_no').drop_duplicates(['rgst_no', 'applicant_cd'])
ins = ins[(ins.ay >= 2020) & (ins.ay <= 2023)].copy(); ins['k'] = ins.ksic.map(kidx0); ins = ins.dropna(subset=['k'])
yr0 = pair_label(ins)
al0 = pd.concat([fq[['applicant_cd', 'applicant_name']].rename(columns={'applicant_name': 'nm'}),
                 ap0[ap0.applicantCd.isin(fset0)][['applicantCd', 'applicantName']].rename(columns={'applicantCd': 'applicant_cd', 'applicantName': 'nm'})])
del ap0, ins
al0['n'] = al0.nm.map(norm0); al0 = al0[al0.n.str.len() >= 2].drop_duplicates(['applicant_cd', 'n'])
amb0 = al0.groupby('n').applicant_cd.nunique(); al1 = al0[al0.n.map(amb0) == 1]; n2c0 = dict(zip(al1.n, al1.applicant_cd))
kk = pd.read_parquet(B/'공모전데이터/작업데이터/kr_patents.parquet', columns=['출원번호', '출원연도', '등록일자', '공개일자', '출원인명', '세세분류코드'])
kk.columns = ['appl_no', 'ay', 'rgst_date', 'pub_date', 'names', 'ksic']; kk['ksic'] = kk.ksic.astype(str); kk['ay'] = kk.ay.astype(int)
kk = kk[(kk.ay >= 2020) & (kk.ay <= 2023)].copy()
kk['nm'] = kk.names.fillna('').str.split('|'); kk = kk.explode('nm'); kk['applicant_cd'] = kk.nm.map(norm0).map(n2c0)
mm = kk.dropna(subset=['applicant_cd']).drop_duplicates(['appl_no', 'applicant_cd']).copy(); del kk
mm['k'] = mm.ksic.map(kidx0); mm = mm.dropna(subset=['k'])
pdt = mm.pub_date.fillna('').astype(str); rdt = mm.rgst_date.fillna('').astype(str)
yp0 = pair_label(mm[((pdt.str.len() == 8) & (pdt <= END)) | ((rdt.str.len() == 8) & (rdt <= END))]); del mm
y0 = np.maximum(yr0, yp0).astype(np.int8)
o['label_reconstruct'] = dict(pos=int(y0.sum()), pos_reg=int(yr0.sum()), pos_pub=int(yp0.sum()), pos_new=int(y0[nc].sum()), rows=int(len(y0)))
print('labels', o['label_reconstruct'], round(time.time() - t0), flush=True)
def score(y):
    m = roc_auc_score(y, s); mn = roc_auc_score(y[nc], s[nc])
    bm = {c: roc_auc_score(y, bs[c].values) for c in BCOLS}; bn = {c: roc_auc_score(y[nc], bs[c].values[nc]) for c in BCOLS}
    arg = max(bm, key=bm.get); st, stn = bm[arg], max(bn.values())
    oo = np.argsort(-s, kind='stable'); ob = np.argsort(-bs[arg].values, kind='stable')
    return dict(pos=int(y.sum()), new_pos=int(y[nc].sum()), auc_all=float(m), auc_new=float(mn), strongest=float(st), strongest_arg=arg, strongest_new=float(stn),
                skill_all=float(skill(m, st)), skill_new=float(skill(mn, stn)),
                top100=int(y[oo[:100]].sum()), top1000=int(y[oo[:1000]].sum()), top10000=int(y[oo[:10000]].sum()), base_top1000=int(y[ob[:1000]].sum()))
P0 = score(y0)
off = json.load(open(R3/'final_result_r3.json', encoding='utf-8'))
P0['official'] = dict(auc_all=off['model']['auc_all'], skill_all=off['skill']['auc_all'], skill_new=off['skill']['auc_new'], pos=off['n']['pos'], top1000=off['topk']['model']['1000']['hits'])
P0['pass'] = bool(abs(P0['auc_all'] - off['model']['auc_all']) <= 1e-7 and abs(P0['skill_all'] - off['skill']['auc_all']) <= 1e-7
                  and P0['top1000'] == off['topk']['model']['1000']['hits'] and P0['pos'] == off['n']['pos'])
o['P0'] = P0; print('P0', P0, round(time.time() - t0), flush=True)
# ---------- 누락 등록(원천 재도출): kr_patents 등록번호 공란 + 권리목록 등록(≤END) + registry_full 미보유 ----------
Tm = pd.read_csv(H/'monthly_capture.csv', dtype={'ym': str}).set_index('ym'); G = set(Tm.index[Tm.gap])
K = sorted(pd.read_parquet(TC/'ins_pat.parquet', columns=['ksic']).ksic.astype(str).unique()); kidx = {k: i for i, k in enumerate(K)}
kp = pd.read_parquet(B/'공모전데이터/작업데이터/kr_patents.parquet', columns=['출원번호', '출원연도', '등록번호', '세세분류코드', '출원인명'])
kp.columns = ['appl_no', 'ay', 'rgst_no', 'ksic', 'names']; kp['ay'] = pd.to_numeric(kp.ay, errors='coerce'); kp['ksic'] = kp.ksic.astype(str)
kp = kp[kp.ay.between(2020, 2023) & ~kp.rgst_no.astype(str).str.strip().str.match(r'^\d{13}$')]
rl = pd.read_parquet(TC/'rl_pat.parquet', columns=['applicant_cd', 'rgst_no', 'rgst_date', 'appl_no', 'pert', 'ainfo'])
rl = rl[(rl.rgst_date <= END) & rl.appl_no.isin(set(kp.appl_no))]
rc = ro(B/'수집파이프라인/registry_full.sqlite')
rfset = set(r[0] for r in rc.execute("select rgst_no from patent"))
app = pd.read_sql('select rgst_no, applicantCd, applicantName from applicant', rc); rc.close()
o['rl_rows_blank_window'] = int(len(rl)); o['rl_rows_in_rf'] = int(rl.rgst_no.isin(rfset).sum())
docs = rl.sort_values('rgst_date').drop_duplicates('appl_no')[['appl_no', 'rgst_no', 'rgst_date']]
o['missing_docs_T1_window'] = int(len(docs)); o['missing_docs_T1_window_inG'] = int(docs.rgst_date.str[:6].isin(G).sum())
# ---------- 원출원인 법인 추정(별칭) ----------
firms = pd.read_sql('select applicant_cd, applicant_name, status from applicant_query', ro(B/'수집파이프라인/business_rights.sqlite'))
firms = firms[firms.status == 'OK']; fset = set(firms.applicant_cd)
STRIP = re.compile(r"주식회사|\(주\)|㈜|유한회사|유한책임회사|재단법인|사단법인|학교법인|가부시키가이샤|가부시끼가이샤|가부시기가이샤|가부시키카이샤|코포레이션|인코포레이티드|리미티드|컴퍼니|[\s\.\,\-\(\)\'\"·&]")
norm = lambda x: STRIP.sub('', str(x)).lower()
al = pd.concat([firms[['applicant_cd', 'applicant_name']].rename(columns={'applicant_name': 'nm'}),
                app[app.applicantCd.isin(fset)][['applicantCd', 'applicantName']].rename(columns={'applicantCd': 'applicant_cd', 'applicantName': 'nm'})])
del app
al['n'] = al.nm.map(norm); al = al[al.n.str.len() >= 2].drop_duplicates(['applicant_cd', 'n'])
alias = al.groupby('applicant_cd').n.apply(list).to_dict()
def is_self(cd, ainfo):
    a = norm(ainfo)
    for n in alias.get(cd, []):
        if len(n) >= 3 and n in a: return True
        if len(n) < 3 and any(norm(p) == n for p in str(ainfo).split(',')): return True
    return False
r = rl[rl.applicant_cd.isin(fset) & (rl.pert == '권리자란')].merge(kp[['appl_no', 'ay', 'ksic']], on='appl_no')
r['self'] = [is_self(c_, a_) for c_, a_ in zip(r.applicant_cd.values, r.ainfo.values)]
r = r.drop_duplicates(['appl_no', 'applicant_cd']); r['k'] = r.ksic.map(kidx); r = r.dropna(subset=['k']); r['k'] = r.k.astype(int)
r['inG'] = r.rgst_date.str[:6].isin(G)
o['pairs_any'] = int(len(r)); o['pairs_self'] = int(r.self.sum()); o['pairs_self_inG'] = int((r.self & r.inG).sum())
key_firm = fz.firm.values; key_k = fz.k.values.astype(int)
def add_from(sub):
    ks = set(zip(sub.applicant_cd, sub.k))
    return np.fromiter(((f_, k_) in ks for f_, k_ in zip(key_firm, key_k)), bool, len(key_firm))
SC = {'S1_R_self_G': r[r.self & r.inG], 'S2_R_any_G': r[r.inG], 'S1b_R_self_all': r[r.self], 'S2b_R_any_all': r}
res = {}; labs = {}
for nm, sub in SC.items():
    a = add_from(sub); y = np.maximum(y0, a.astype(np.int8)); labs[nm] = y
    sc_ = score(y); sc_['added_rows'] = int(a.sum()); sc_['new_pos'] = int(((y == 1) & (y0 == 0)).sum()); sc_['new_pos_newc'] = int(((y == 1) & (y0 == 0) & nc).sum())
    sc_['pos_change_rate'] = (sc_['pos'] - P0['pos']) / P0['pos']
    for q in ('auc_all', 'auc_new', 'skill_all', 'skill_new', 'strongest'): sc_['d_' + q] = sc_[q] - P0[q]
    for q in ('top100', 'top1000', 'top10000', 'base_top1000'): sc_['d_' + q] = sc_[q] - P0[q]
    res[nm] = sc_; print(nm, {k: sc_[k] for k in ('pos', 'new_pos', 'd_auc_all', 'd_skill_all', 'top1000')}, round(time.time() - t0), flush=True)
o['scenarios'] = res
# kr_registered 보충: kr_registered 특허 등록번호 집합 = registry_full 집합(s1: 대칭차 0) → 추가 0
o['kr_registered_supplement_adds'] = 0
np.savez_compressed(C/'s5_labels.npz', y0=y0, **{k: v for k, v in labs.items()})
pd.DataFrame(dict(firm=fz.firm.values, newc=nc, s=s, **{'b_' + c: bs[c].values for c in BCOLS})).to_parquet(C/'s5_eval.parquet', index=False)
o['secs'] = round(time.time() - t0)
json.dump(o, open(C/'s5_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print('done', o['secs'])
