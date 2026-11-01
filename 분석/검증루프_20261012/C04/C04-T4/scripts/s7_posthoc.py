# s7 (사후, 판정 밖): C04-A 단변량 진단과 C04-B 문서 겹침(N9 유형) 비율. 개발 원점만.
import sys, json
sys.path.insert(0, r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C04\C04-T4\scripts')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from libA import *
def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float); m = ~np.isnan(s); y, s = y[m], s[m]
    r = pd.Series(s).rank().values; n1 = y.sum(); n0 = len(y) - n1
    return round(float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)), 4)
u = pd.read_parquet(C + r'\A_units_dev.parquet')
res = {}
# KSIC 최빈 산업의 다른 원점 양성률(leave-one-origin-out)
lo = []
for s0 in DEV_A:
    tr = u[u.s0 != s0].groupby('ksic_mode').yA.mean()
    lo.append(u[u.s0 == s0].ksic_mode.map(tr))
u['ksic_rate_loo'] = pd.concat(lo).reindex(u.index)
res['auc_yA_ksic_mode_rate_loo'] = auc(u.yA, u.ksic_rate_loo)
res['auc_yA_log_nvis'] = auc(u.yA, np.log1p(u.n_vis))
res['auc_yA_first_ay(최근 진입)'] = auc(u.yA, u.first_ay)
res['auc_yA_E0(규모)'] = auc(u.yA, u.E0)
ind = []
for s0 in DEV_A:
    tr = u[u.s0 != s0].groupby('ind2').yA.mean(); ind.append(u[u.s0 == s0].ind2.map(tr))
u['ind2_rate_loo'] = pd.concat(ind).reindex(u.index)
res['auc_yA_nps_ind2_rate_loo'] = auc(u.yA, u.ind2_rate_loo)
res['auc_yC_ksic_mode_rate_loo'] = auc(u[u.eligC == 1].yC, u[u.eligC == 1].ksic_rate_loo)
print(res, flush=True)
# C04-B 양성 문서 중 정부 R&D 성과 표지(N9 유형 문서 겹침)
fl = pd.read_parquet(EXT + r'\ntis_patent_flag.parquet', columns=['appl_no', 'gov_rnd', 'min_project_year'])
docs = load_docs()
pos = pd.read_parquet(C + r'\B_dev_pos.parquet')
ca = load_ca(); nm2b6, _ = name_map(ca); docs['b6'] = docs.nname.map(nm2b6)
P = []
for s0 in DEV_B:
    ref = REF[s0]; end = add_months(ref, 6)
    w = docs[docs.pub.notna() & (docs.pub > ref) & (docs.pub <= end) & docs.b6.notna()].copy(); w['key'] = w.b6 + '|' + w.nname
    P.append(w[w.key.isin(set(pos[pos.s0 == s0].key))][['appl_no', 'key']].assign(s0=s0))
P = pd.concat(P).merge(fl, on='appl_no', how='left')
res['B_pos_docs'] = int(len(P)); res['B_pos_docs_gov_rnd_share'] = round(float(P.gov_rnd.fillna(False).astype(bool).mean()), 4)
res['B_pos_firms_any_gov_rnd_doc_share'] = round(float(P.groupby(['s0', 'key']).gov_rnd.apply(lambda s: s.fillna(False).astype(bool).any()).mean()), 4)
print(res)
json.dump(res, open(C + r'\s7_posthoc.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
