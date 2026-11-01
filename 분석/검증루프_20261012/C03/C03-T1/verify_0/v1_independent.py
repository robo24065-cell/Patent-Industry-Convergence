# verify_0 v1: C03-T1 발견 반박용 독립 재계산 (원천 직접, 기존 cache 미사용)
# 읽기 전용. 쓰기: verify_0\ 만.
import sys, re, json, time, sqlite3, pathlib
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(encoding='utf-8')
t0 = time.time()
ROOT = pathlib.Path(r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전")
R3 = ROOT/'분석/모델경쟁_20260927/T1_산업대기실/R3'
V = ROOT/'분석/검증루프_20261012/C03/C03-T1/verify_0'
DATA = ROOT/'공모전데이터/작업데이터'
END = '20250822'
def ro(p): return sqlite3.connect(pathlib.Path(p).as_uri() + '?mode=ro', uri=True)

o = pd.read_parquet(R3/'runs/FROZEN_final/o2019.parquet')
o = o[~o.prior_any.values].reset_index(drop=True)
bs = pd.read_parquet(R3/'cache_final/o2019/baseline_scores.parquet')
te = pd.read_parquet(R3/'runs/FROZEN_final/o2019.parquet', columns=['prior_any'])
bs = bs.loc[~te.prior_any.values].reset_index(drop=True)
rk = lambda x: pd.Series(x).rank(pct=True).values
s = 0.9*rk(o.px.values) + 0.1*rk(o.pl.values)
nc = o.newc.values.astype(bool)
kr = pd.read_parquet(DATA/'kr_registered.parquet', columns=['세세분류코드'])
K = sorted(kr['세세분류코드'].astype(str).unique()); kidx = {k: i for i, k in enumerate(K)}; del kr
# 공식 레이블: T1 공식 입력 캐시(ins_firm·app_pub)로 r3.reg_lab / r2.pub_lab 정의 그대로
TC = ROOT/'분석/모델경쟁_20260927/T1_산업대기실/cache'
ins = pd.read_parquet(TC/'ins_firm.parquet', columns=['applicant_cd', 'ksic', 'ay'])
e = ins[(ins.ay > 2019) & (ins.ay <= 2023)][['applicant_cd', 'ksic']].drop_duplicates(); e['k'] = e.ksic.astype(str).map(kidx)
e = e.dropna(subset=['k']); E = set(zip(e.applicant_cd, e.k.astype(int))); del ins
pairs0 = list(zip(o.firm.values, o.k.values.astype(int)))
yreg = np.fromiter((p in E for p in pairs0), bool, len(o)).astype(np.int8)
a = pd.read_parquet(TC/'app_pub.parquet'); a['k'] = a.ksic.astype(str).map(kidx); a = a.dropna(subset=['k']); a['k'] = a.k.astype(int)
pdt = a.pub_date.fillna('').astype(str); rdt = a.rgst_date.fillna('').astype(str)
a = a[(((pdt.str.len() == 8) & (pdt <= END)) | ((rdt.str.len() == 8) & (rdt <= END))) & (a.ay > 2019) & (a.ay <= 2023)]
P = set(zip(a.applicant_cd, a.k)); del a
ypub = np.fromiter((p in P for p in pairs0), bool, len(o)).astype(np.int8)
y0 = np.maximum(yreg, ypub)
out = dict(rows=len(o), pos=int(y0.sum()), pos_reg=int(yreg.sum()), pos_pub=int(ypub.sum()), new_pos=int(y0[nc].sum()))
print(out, flush=True)

kp = pd.read_parquet(DATA/'kr_patents.parquet', columns=['출원번호', '출원연도', '등록번호', '출원인명', '세세분류코드'])
kp.columns = ['appl_no', 'ay', 'rgst_no', 'names', 'ksic']; kp['ay'] = kp.ay.astype(int)
kp = kp[(kp.ay >= 2020) & (kp.ay <= 2023)]
kp['blank'] = kp.rgst_no.fillna('').astype(str).str.len() <= 3
blank = kp[kp.blank].copy(); blank['k'] = blank.ksic.astype(str).map(kidx)
out['window_blank_all'] = len(blank)

c = ro(ROOT/'수집파이프라인/business_rights.sqlite')
rl = pd.read_sql("select applicant_cd, appl_no, rgst_no, rgst_date, json_extract(raw,'$.pertDiv') pert, json_extract(raw,'$.applicantInfo') ainfo, json_extract(raw,'$.indvCorpTpcd') tp "
                 "from right_list where rgst_no like '10%' and substr(appl_no,3,4) >= '2020'", c)
q = pd.read_sql('select applicant_cd, applicant_name, status from applicant_query', c)
rl = rl[rl.rgst_date <= END]
bl_rl = rl[rl.appl_no.isin(set(blank.appl_no))]
out['blank_confirmed_any_rl'] = int(bl_rl.appl_no.nunique())
out['blank_confirmed_by_ay'] = blank[blank.appl_no.isin(set(bl_rl.appl_no))].groupby('ay').size().to_dict()
ok = set(q[q.status == 'OK'].applicant_cd)
# 평가 법인 중 조회 OK 밖
ef = set(o.firm.unique()); out['eval_firms'] = len(ef); out['eval_firms_not_OK'] = len(ef - ok)
out['eval_firms_status'] = q[q.applicant_cd.isin(ef)].status.value_counts().to_dict()

# 별칭: applicant_query 이름 + 등록원부 출원인명
app = pd.read_sql('select rgst_no, applicantCd, applicantName from applicant', ro(ROOT/'수집파이프라인/registry_full.sqlite'))
STRIP = re.compile(r"주식회사|\(주\)|㈜|유한회사|유한책임회사|재단법인|사단법인|학교법인|가부시키가이샤|가부시끼가이샤|가부시기가이샤|가부시키카이샤|코포레이션|인코포레이티드|리미티드|컴퍼니|[\s\.\,\-\(\)\'\"·&]")
norm = lambda x: STRIP.sub('', str(x)).lower()
al = pd.concat([q[q.applicant_cd.isin(ok)][['applicant_cd', 'applicant_name']].rename(columns={'applicant_name': 'nm'}),
                app[app.applicantCd.isin(ok)][['applicantCd', 'applicantName']].rename(columns={'applicantCd': 'applicant_cd', 'applicantName': 'nm'})])
al['n'] = al.nm.map(norm); al = al[al.n.str.len() >= 2].drop_duplicates(['applicant_cd', 'n'])
alias = al.groupby('applicant_cd').n.apply(list).to_dict()
# RC:001 법인이 등록원부에 출원 2020-23 등록특허를 갖고 있나 (조회 실패 vs 권리 없음)
krr = pd.read_parquet(DATA/'kr_registered.parquet', columns=['등록번호', '출원연도']); krr.columns = ['rgst_no', 'ay']; krr['rgst_no'] = krr.rgst_no.astype(str)
rc = set(q[q.status != 'OK'].applicant_cd)
appw = app.merge(krr[(krr.ay.astype(int) >= 2020) & (krr.ay.astype(int) <= 2023)], on='rgst_no')
out['RC_firms_with_reg_2020_23'] = int(appw[appw.applicantCd.isin(rc)].applicantCd.nunique())
out['RC_firms_in_eval_with_reg_2020_23'] = int(appw[appw.applicantCd.isin(rc & ef)].applicantCd.nunique())
del app, appw

def is_self(cd, ainfo):
    a = norm(ainfo)
    for n in alias.get(cd, []):
        if len(n) >= 3 and n in a: return True
        if len(n) < 3 and any(norm(p) == n for p in str(ainfo).split(',')): return True
    return False
b = bl_rl[bl_rl.applicant_cd.isin(ok)].merge(blank[['appl_no', 'ay', 'k']], on='appl_no')
b['self'] = [is_self(x, y) for x, y in zip(b.applicant_cd, b.ainfo)]
b = b.dropna(subset=['k']); b['k'] = b.k.astype(int)
variants = {
    'R_self(권리자란)': b[(b.pert == '권리자란') & b.self],
    'R_any(권리자란)': b[b.pert == '권리자란'],
    'R_anyall(모든 pert)': b,
}
key0 = pd.Series(np.arange(len(o)), index=pd.MultiIndex.from_arrays([o.firm.values, o.k.values.astype(int)]))
BC = list(bs.columns)
def score(y):
    m_all = roc_auc_score(y, s); m_new = roc_auc_score(y[nc], s[nc])
    ba = {cc: roc_auc_score(y, bs[cc].values) for cc in BC}; bn = {cc: roc_auc_score(y[nc], bs[cc].values[nc]) for cc in BC}
    sa, sn = max(ba.values()), max(bn.values())
    return dict(pos=int(y.sum()), new_pos=int(y[nc].sum()), auc_all=m_all, auc_new=m_new, strong_all=sa, strong_arg=max(ba, key=ba.get), strong_new=sn,
                skill_all=(m_all-sa)/(1-sa), skill_new=(m_new-sn)/(1-sn),
                top1000=int(y[np.argsort(-s, kind='stable')[:1000]].sum()), top100=int(y[np.argsort(-s, kind='stable')[:100]].sum()))
res = {'official': score(y0)}
for nm_, sub in variants.items():
    pairs = sub[['applicant_cd', 'k']].drop_duplicates()
    idx = key0.reindex(pd.MultiIndex.from_arrays([pairs.applicant_cd.values, pairs.k.values])).dropna().astype(int).values
    y = y0.copy(); y[idx] = 1
    r = score(y); r['patents'] = int(sub.appl_no.nunique()); r['new_pairs'] = int((y != y0).sum()); r['new_pairs_newc'] = int(((y != y0) & nc).sum())
    r['pos_change'] = r['new_pairs'] / int(y0.sum())
    for q_ in ('auc_all', 'auc_new', 'skill_all', 'skill_new'): r['d_' + q_] = r[q_] - res['official'][q_]
    res[nm_] = r
    print(nm_, {k_: r[k_] for k_ in ('patents', 'new_pairs', 'new_pairs_newc', 'pos_change', 'd_auc_all', 'd_skill_all', 'd_auc_new', 'd_skill_new', 'top100', 'top1000')}, flush=True)
out['scores'] = res
out['bl_rl_patents_OK_firm'] = int(b.appl_no.nunique())
json.dump(out, open(V/'v1_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k_: v for k_, v in out.items() if k_ != 'scores'}, ensure_ascii=False, default=str))
print(json.dumps(res['official'], default=float)); print('done', round(time.time()-t0))
