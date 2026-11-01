# 양성 셀 순증가 상위 셀 구성(집계가 소수 셀에 좌우되는지). applicant_concentration.csv(s2 산출, v1에서 독립 재현값과 일치 확인) 사용.
import os, sys, json
import pandas as pd, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
W = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C04\C04-T3"
ac = pd.read_csv(os.path.join(W, 'applicant_concentration.csv'), dtype={'ksic': str})
p = ac[ac.yb == 1].copy(); p['bD'] = p.big_fut - p.big_past
p = p.sort_values('D', ascending=False)
p['cumD'] = p.D.cumsum() / p.D.sum()
print(p[['cell', 'kr_W', 'kr_fut', 'D', 'bD', 'top_app_k1', 'top_delta', 'p']].head(15).to_string())
r = {}
for k in [5, 10, 20]:
    t = p.head(k); rest = p.iloc[k:]
    r[f'top{k}_cells_shareD'] = float(t.D.sum() / p.D.sum()); r[f'big_share_excl_top{k}'] = float(rest.bD.sum() / rest.D.sum())
r['big_share_S0S1_pos'] = float(p[p.kr_W <= 1].bD.sum() / p[p.kr_W <= 1].D.sum()); r['big_share_S2_pos'] = float(p[p.kr_W >= 2].bD.sum() / p[p.kr_W >= 2].D.sum())
r['big_share_top504hits'] = None
o = np.argsort(-ac.p.values); t5 = ac.iloc[o[:504]]; h = t5[t5.yb == 1]
r['big_share_increase_in_top504_hits'] = float((h.big_fut - h.big_past).sum() / h.D.sum()); r['top504_hits_cell_median_big_share'] = float(((h.big_fut - h.big_past) / h.D).median())
r['ksic_of_top10'] = p.head(10).ksic.value_counts().to_dict()
json.dump(r, open(os.path.join(W, 'verify_2', 'v2_results.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(r, ensure_ascii=False, indent=1))
