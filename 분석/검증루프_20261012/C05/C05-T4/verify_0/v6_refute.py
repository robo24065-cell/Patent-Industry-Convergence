# v6 반박 검증(별도 세션): 잠금 레이블을 원자료에서 독립 재계산, 지표·CI 재계산, 2025-H1 수록 완전성, 상위1000 동점 점검.
import os, re, json, sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
WD = os.path.join(B, '분석', '검증루프_20261012', 'C05', 'C05-T4')
OUT = os.path.join(WD, 'verify_0', 'v6_refute_out.json')
SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
def norm(s):
    s = SUF.sub("", str(s or "")).upper()
    return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", s)
out = {}
k = pd.read_parquet(os.path.join(B, '공모전데이터', '작업데이터', 'kr_patents.parquet'), columns=['출원번호', '출원연도', '공개일자', '등록일자', '출원인명'])
k.columns = ['appl_no', 'ay', 'pub', 'rg', 'aname']
for c in ['pub', 'rg']:
    k[c] = k[c].fillna('').astype(str).str.strip()
    k.loc[k[c].str.len() != 8, c] = None
out['n_docs'] = len(k); out['dup_appl_no'] = int(k.appl_no.duplicated().sum())
out['pub_max'] = k.pub.dropna().max(); out['rg_max'] = k.rg.dropna().max()
# 반기별 공개 문서 수(수록 완전성)
h = k.pub.dropna(); hh = h.str[:4] + np.where(h.str[4:6].astype(int) <= 6, 'H1', 'H2')
out['pub_by_half'] = {kk: int(v) for kk, v in hh.value_counts().sort_index().items() if kk >= '2019'}
mm = h[h >= '20240101'].str[:6].value_counts().sort_index()
out['pub_by_month_2024_'] = {kk: int(v) for kk, v in mm.items()}
k['vis'] = np.where(k.rg.notna() & (k.pub.isna() | (k.rg.fillna('9') < k.pub.fillna('9'))), k.rg, k.pub)
k['nname'] = k.aname.map(norm)
# CA
rows = []
with open(os.path.join(B, '공모전데이터', '외부데이터', 'Corporate_20260720', 'CORP_APPLICANT.txt'), encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\r\n').split('¶')
        if len(p) >= 5:
            rows.append((p[0].strip(), p[1].strip(), re.sub(r'\D', '', p[4])))
ca = pd.DataFrame(rows, columns=['cd', 'aname', 'biz'])
ca['nname'] = ca.aname.map(norm); ca['hb'] = ca.biz.str.len() == 10
ca['b6'] = np.where(ca.hb, ca.biz.str[:6], None)
ca['tc'] = ca.hb & ca.biz.str[3:5].isin([str(x) for x in range(81, 89)])
ca = ca[ca.nname.str.len() >= 2]
g = ca.groupby('nname').agg(n=('cd', 'size'), nb=('hb', 'sum'), nb6=('b6', 'nunique'), b6=('b6', 'first'), tc=('tc', 'max'))
ok = g[(g.nb == g.n) & (g.nb6 == 1) & g.tc].b6
out['ca_rows'] = len(rows); out['ca_unique_names_ok'] = int(len(ok))
k['b6'] = k.nname.map(ok)
REF, END = '20241231', '20250630'
lk = k[k.b6.notna()]
fv = lk[lk.vis.notna()].groupby('nname').vis.min()
w = lk[lk.pub.notna() & (lk.pub > REF) & (lk.pub <= END)]
w = w[w.nname.map(fv) > REF]
keys = set(w.b6 + '|' + w.nname)
fz = pd.read_parquet(os.path.join(WD, 'frozen_scores.parquet'))
fz['y'] = fz.key.isin(keys).astype(int)
out['indep_pos'] = int(fz.y.sum()); out['indep_window_keys_all'] = len(keys)
lab = pd.read_parquet(os.path.join(WD, 'cache', 'lock_labels.parquet'))
mg = fz[['key', 'y']].merge(lab, on='key', suffixes=('', '_s6'))
out['label_disagree'] = int((mg.y != mg.y_s6).sum())
# 위험집합 정합: 양성 키 중 기준일 이전 보인 문서가 있는 키(있으면 안 됨) / 위험집합 키 중 기준일 이전 보인 연결 문서 보유
fz_n = fz.key.str.split('|', n=1).str[1]
pre = set(fv[fv <= REF].index)
out['riskset_keys_with_prior_linked_doc'] = int(fz_n.isin(pre).sum())
# 지표
y = fz.y.values; res = {}
for m in ['M0', 'M1', 'M1_pub_TXT', 'M1_pub_IPC', 'champion']:
    s = fz[f'score_{m}'].values
    o = np.argsort(-s, kind='mergesort')
    thr = s[o[999]]; n_gt = int((s > thr).sum()); n_eq = int((s == thr).sum())
    pos_gt = int(y[s > thr].sum()); pos_eq = int(y[s == thr].sum())
    exp_top = (pos_gt + (1000 - n_gt) * pos_eq / n_eq) / 1000
    res[m] = dict(auc=round(roc_auc_score(y, s), 5), ap=round(average_precision_score(y, s), 6),
                  top1000_stable=float(y[o[:1000]].mean()), top1000_tie_expected=round(exp_top, 5), tie_at_boundary=n_eq,
                  top500=float(y[o[:500]].mean()), top2000=float(y[o[:2000]].mean()), top1000_hits=int(y[o[:1000]].sum()))
out['metrics'] = res
# 부트스트랩(시드 7, 1000회, 행=기업키) — AUC·skill CI
rng = np.random.default_rng(7); n = len(fz)
S = {m: fz[f'score_{m}'].values for m in ['M0', 'M1']}
from scipy.stats import rankdata
R = {m: rankdata(S[m]) for m in S}
pi = np.where(y == 1)[0]; ni = np.where(y == 0)[0]
def wauc(s, wt):
    # 가중 Mann-Whitney: 정렬 후 누적
    o = np.argsort(s, kind='mergesort'); ss = s[o]; yy = y[o]; ww = wt[o]
    _, st = np.unique(ss, return_index=True); grp = np.repeat(np.arange(len(st)), np.diff(np.append(st, len(ss))))
    g0 = np.bincount(grp, np.where(yy == 0, ww, 0)); g1 = np.bincount(grp, np.where(yy == 1, ww, 0))
    return float(np.sum(g1 * (np.cumsum(g0) - g0 + 0.5 * g0)) / (g1.sum() * g0.sum()))
A = {'M0': [], 'M1': []}; T = []
oc = np.argsort(-S['M1'], kind='mergesort')
for b in range(1000):
    wt = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
    for m in A: A[m].append(wauc(S[m], wt))
    wd = wt[oc]; cn = np.cumsum(wd); i = min(np.searchsorted(cn, 1000), n - 1); T.append(float(np.cumsum(wd * y[oc])[i] / cn[i]))
q = lambda x: [round(float(np.quantile(x, .025)), 5), round(float(np.quantile(x, .975)), 5)]
a0, a1 = np.array(A['M0']), np.array(A['M1'])
out['boot_seed7'] = dict(M1_auc_ci=q(a1), M0_auc_ci=q(a0), skill_ci=q((a1 - a0) / (1 - a0)), M1_top1000_ci=q(T))
out['base_rate'] = float(y.mean()); out['ratio_top1000_to_base'] = res['champion']['top1000_stable'] / float(y.mean())
json.dump(out, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
