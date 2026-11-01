# 사후 민감도(판정 밖): 그룹명 한·영 표기(SK=에스케이 등)를 같은 그룹으로 보면 양성이 얼마나 줄어드나 + 상위 양수인 목록
import sys, json, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C07\C07-T4\scripts')
from lib7 import *
CANON = {'SK': '에스케이', 'LG': '엘지', 'GS': '지에스', 'KT': '케이티', 'CJ': '씨제이', 'LS': '엘에스', 'OCI': '오씨아이', 'KCC': '케이씨씨'}
ev = pd.read_parquet(C + r'\events_acc.parquet')
own = pd.read_parquet(C + r'\own.parquet', columns=['rgst_no', 'seq', 'ownerName', 'd']).sort_values(['rgst_no', 'd', 'seq'])
first = own.groupby(['rgst_no', 'd']).ownerName.first().reset_index()
first['dd'] = pd.to_datetime(first.d, format='%Y%m%d'); first['prev_raw'] = first.groupby('rgst_no').ownerName.shift(1)
R = {}; rows = []
for Y in DEV:
    b = pd.read_parquet(C + rf'\base_{Y}.parquet', columns=['rgst_no', 'ev_date', 'buyer', 'buyer_raw']); l = pd.read_parquet(C + rf'\lab_prov_{Y}.parquet', columns=['rgst_no', 'y'])
    p = b.merge(l, on='rgst_no'); p = p[p.y == 1].merge(first[['rgst_no', 'dd', 'prev_raw']], left_on=['rgst_no', 'ev_date'], right_on=['rgst_no', 'dd'], how='left')
    p['t0'] = Y; rows.append(p)
P = pd.concat(rows)
cg = lambda r: (lambda g: CANON.get(g, g))(group_of(r))
P['gb'] = P.buyer_raw.map(cg); P['gs'] = P.prev_raw.map(cg)
P['same_canon'] = P.gb.notna() & (P.gb == P.gs)
R['pos'] = len(P); R['same_group_canonical'] = int(P.same_canon.sum())
R['by_year_after'] = {int(k): int(v) for k, v in P[~P.same_canon].groupby('t0').size().items()}
R['top_buyers'] = {k: int(v) for k, v in P.buyer.value_counts().head(10).items()}
R['top_pairs'] = P.groupby(['prev_raw', 'buyer_raw']).size().sort_values(ascending=False).head(10)
R['top_pairs'] = {f'{a} → {b}': int(v) for (a, b), v in R['top_pairs'].items()}
json.dump(R, open(C + r'\s5_group_sens.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
