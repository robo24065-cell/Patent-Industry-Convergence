# s7_C: C03-C 개발 양성·위험집합 분포 점검(편향·기계적 사건 비중)
import sys, json
sys.path.insert(0, r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C03\C03-T4\scripts')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from libC import *
d = load(); out = {}
R_all, P_all = [], []
for t0 in DEV:
    st, pos, r = build(t0, d)
    ws, we = f'{t0+1}0101', f'{t0+1}0630'; bs, be = f'{t0-2}0101', f'{t0}1231'
    w = d[(d.p >= ws) & (d.p <= we)]; b = d[(d.p >= bs) & (d.p <= be)]
    nw = w.groupby('cd').appl_no.nunique(); nb = b.groupby('cd').appl_no.nunique()
    kw = w.groupby(['cd', 'ksic']).appl_no.nunique()
    r = r.reset_index(); r['t0'] = t0
    r['nw'] = r.cd.map(nw); r['nb'] = r.cd.map(nb)
    r['kw'] = [kw.get((c, k), 0) for c, k in zip(r.cd, r.k)]
    R_all.append(r)
R = pd.concat(R_all, ignore_index=True)
Y = R[R.y]
out['pos_total'] = len(Y)
out['pos_window_k_docs_dist'] = Y.kw.value_counts().sort_index().head(8).to_dict()
out['pos_share_window_k_docs_le1'] = round(float((Y.kw <= 1).mean()), 4)
out['pos_share_window_n_eq3'] = round(float((Y.nw == 3).mean()), 4)
out['risk_window_n_median'] = float(R.drop_duplicates(['t0', 'cd']).nw.median())
out['risk_base_n_median'] = float(R.drop_duplicates(['t0', 'cd']).nb.median())
out['pos_base_share_median'] = round(float(Y.sb.median()), 4)
out['risk_base_share_median'] = round(float(R.sb.median()), 4)
fc = Y.cd.value_counts(); out['pos_top10_firm_share'] = round(float(fc.head(10).sum() / len(Y)), 4)
out['pos_firms_unique'] = int(Y.cd.nunique())
kc = Y.k.value_counts(normalize=True); rk = R.k.value_counts(normalize=True)
out['pos_top5_ksic'] = kc.head(5).round(4).to_dict(); out['risk_share_of_those'] = rk.reindex(kc.head(5).index).round(4).to_dict()
out['base_rate_by_t0'] = R.groupby('t0').y.mean().round(4).to_dict()
# 위험집합 상위 기업(문서 많은 기업) 비중
big = R.drop_duplicates(['t0', 'cd']).nb
out['firm_base_docs_p10_p50_p90'] = [float(big.quantile(q)) for q in (0.1, 0.5, 0.9)]
# 기계적 사건 대리: 기준 몫이 작고(≤0.1) 창 문서 3건 중 1건 → 0.333 상승
out['pos_share_mechanical_proxy(nw==3&kw==1)'] = round(float(((Y.nw == 3) & (Y.kw == 1)).mean()), 4)
json.dump(out, open(C + r'\s7_C_bias.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(out, ensure_ascii=False, default=str))
