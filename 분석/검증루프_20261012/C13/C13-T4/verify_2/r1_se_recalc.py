# verify_2 r1: s5_pred_2023.parquet에서 AUC·SE를 독립 코드로 다시 계산(작업 lib13 미사용).
# 방법: (A) 작업과 같은 곱 Poisson 가중(다른 시드) (B) 기업 단위 iid (C) KSIC 군집만(중첩 구조의 정확한 상위 군집)
# (D) 해석적 DeLong 상관 ΔAUC SE(비군집) (E) 곱 가중의 과대 분해: 2*var_firm + var_ksic 검산
import json, sys, numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
W = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C13\C13-T4'
d = pd.read_parquet(W + r'\cache\s5_pred_2023.parquet')
u3 = pd.read_parquet(W + r'\cache\u_prov_2023.parquet')
res = {'n': len(d), 'pos': int(d.y.sum()), 'n_u_prov_2023': len(u3), 'pos_u_prov_2023': int(u3.y.sum()),
       'dup_u': int(d.u.duplicated().sum()), 'modal_na': int(d.modal_prov.isna().sum())}
y = d.y.values.astype(int)
auc = {k: roc_auc_score(y, d['p_'+k]) for k in ['base','prov','free','hsc']}
res['auc'] = auc
pub = max(['free','hsc'], key=lambda k: auc[k]); res['pub'] = pub
dA = auc['prov'] - auc[pub]; res['dAUC'] = dA
k = d.modal_prov.fillna('NA').values
uk, kinv = np.unique(k, return_inverse=True)
res['n_ksic_clusters'] = len(uk)
cs = pd.Series(kinv).value_counts()
res['ksic_cluster_size_top5_share'] = float(cs.nlargest(5).sum()/len(d))
pos_by_k = pd.Series(y).groupby(kinv).sum()
res['ksic_clusters_with_pos'] = int((pos_by_k>0).sum())
p1 = d['p_prov'].values; p2 = d['p_'+pub].values
rng = np.random.default_rng(4242)
B = 1000
dd = {'two_prod': [], 'firm': [], 'ksic': []}
def wd(w):
    return roc_auc_score(y, p1, sample_weight=w) - roc_auc_score(y, p2, sample_weight=w)
for b in range(B):
    wf = rng.poisson(1.0, len(y)).astype(float)
    wkc = rng.poisson(1.0, len(uk)).astype(float)
    wk = wkc[kinv]
    for nm, w in (('two_prod', wf*wk), ('firm', wf), ('ksic', wk)):
        if w[y==1].sum() > 0 and w[y==0].sum() > 0:
            dd[nm].append(wd(w))
se = {nm: float(np.std(v, ddof=1)) for nm, v in dd.items()}
res['SE'] = se
res['ci'] = {nm: [float(np.percentile(v,2.5)), float(np.percentile(v,97.5))] for nm, v in dd.items()}
res['check_prod_minus_2firm_vs_ksic'] = {'sqrt(two^2-2firm^2)': float(np.sqrt(max(se['two_prod']**2 - 2*se['firm']**2, 0))), 'ksic': se['ksic']}
# DeLong (비군집)
def delong(y, a, b):
    pos = y==1; neg = ~pos
    def comps(s):
        sp, sn = s[pos], s[neg]
        allr = pd.Series(np.r_[sp, sn]).rank().values
        rp = pd.Series(sp).rank().values; rn = pd.Series(sn).rank().values
        m, n = len(sp), len(sn)
        v10 = (allr[:m] - rp)/n          # P(s_pos > s_neg) per positive
        v01 = 1 - (allr[m:] - rn)/m      # per negative
        return v10, v01
    a10, a01 = comps(a); b10, b01 = comps(b)
    m, n = pos.sum(), neg.sum()
    S10 = np.cov(np.vstack([a10, b10])); S01 = np.cov(np.vstack([a01, b01]))
    var = (S10[0,0]+S10[1,1]-2*S10[0,1])/m + (S01[0,0]+S01[1,1]-2*S01[0,1])/n
    return float(np.sqrt(var)), float(a10.mean()-b10.mean())
res['delong_se'], res['delong_dA'] = delong(y, p1, p2)
LOCK = 1981.5073339898618
sc = np.sqrt(res['pos']/LOCK); res['scale'] = float(sc)
def pw(se_, t): return float(norm.cdf(t/(se_*sc) - 1.96))
res['power_t0.002'] = {nm: pw(v, 0.002) for nm, v in list(se.items()) + [('delong', res['delong_se'])]}
res['power_t0.0043'] = {nm: pw(v, 0.0043) for nm, v in list(se.items()) + [('delong', res['delong_se'])]}
res['SE_lock_needed_for_0.80_at_0.002'] = 0.002/(1.96+norm.ppf(0.8))
res['dskill0.03_to_dAUC'] = 0.03*(1-auc['base'])
json.dump(res, open(W + r'\verify_2\r1_out.json','w',encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
