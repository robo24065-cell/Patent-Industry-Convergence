# -*- coding: utf-8 -*-
# verify_4 V1: TXT panel_ccy에서 T3 셀 부상 라벨(yb)을 독립 재구성해 txt50 예측의 yb와 대조. 출력: v4_labels_<arm>.parquet (ksic,t0,cell,yb)
import sys, hashlib, json
import numpy as np, pandas as pd
B0 = "C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석"
T3 = B0 + "/모델경쟁_20260927/T3_기술셀부상예측"; WD = B0 + "/검증루프_20261012/C11/C11-T2"; OUT = WD + "/verify_4"
arm = sys.argv[1]
panel = {'stored': T3 + '/TXT_R9/cache_txt_pub/panel_ccy.parquet'}.get(arm, f'{WD}/cache/txt_{arm}/panel_ccy.parquet')
preds = {'stored': T3 + '/TXT_R9/results/txt50_task_preds.parquet'}.get(arm, f'{WD}/res/runs/txt50_task__{arm}_preds.parquet')
P = pd.read_parquet(panel, columns=['ksic', 'mg', 'cc', 'year', 'n'])
P['cell'] = P.ksic + '|' + P.mg
out = []
for t0 in range(2010, 2019):
    w = P[(P.year >= t0 - 2) & (P.year <= t0)]
    tot = w.groupby('cell').n.sum()
    krw = w[w.cc == 'KR'].groupby('cell').n.sum()
    f = P[(P.year >= t0 + 1) & (P.year <= t0 + 3) & (P.cc == 'KR')].groupby('cell').n.sum()
    d = pd.DataFrame({'tot': tot}); d = d[d.tot >= 50]
    d['krw'] = krw.reindex(d.index).fillna(0); d['fut'] = f.reindex(d.index).fillna(0)
    d['y'] = np.log((d.fut + 1) / (d.krw + 1)); d['inc'] = d.fut - d.krw
    d['key'] = [int(hashlib.md5(c.encode()).hexdigest()[:8], 16) for c in d.index]
    d = d.sort_values(['y', 'inc', 'key'], ascending=[False, False, True])
    k = max(1, int(round(0.10 * len(d))))
    d['yb'] = 0; d.iloc[:k, d.columns.get_loc('yb')] = 1
    d['t0'] = t0; d['ksic'] = [c.split('|')[0] for c in d.index]
    out.append(d.reset_index().rename(columns={'index': 'cell'})[['cell', 'ksic', 't0', 'yb', 'tot', 'krw', 'fut']])
L = pd.concat(out, ignore_index=True)
R = pd.read_parquet(preds, columns=['cell', 't0', 'yb', 'tot_W', 'kr_W', 'kr_fut'])
m = L.merge(R, on=['cell', 't0'], how='outer', indicator=True)
both = m[m._merge == 'both']
res = dict(arm=arm, n_mine=len(L), n_preds=len(R), only_mine=int((m._merge == 'left_only').sum()), only_preds=int((m._merge == 'right_only').sum()),
           yb_mismatch=int((both.yb_x != both.yb_y).sum()), tot_mismatch=int((both.tot != both.tot_W).sum()),
           fut_mismatch=int((both.fut != both.kr_fut).sum()), n_pos_mine=int(L.yb.sum()), n_pos_preds=int(R.yb.sum()))
print(json.dumps(res, ensure_ascii=False))
L.to_parquet(f'{OUT}/v4_labels_{arm}.parquet', index=False)
json.dump(res, open(f'{OUT}/v4_labels_{arm}.json', 'w', encoding='utf-8'), ensure_ascii=False)
