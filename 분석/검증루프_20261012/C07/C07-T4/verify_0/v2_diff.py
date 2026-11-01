# 작업자(lab_prov) vs 독립(v1_pos) 양성 차이 진단
import sys, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C07\C07-T4\scripts')
from lib7 import *
Y = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
b = pd.read_parquet(C + rf'\base_{Y}.parquet'); l = pd.read_parquet(C + rf'\lab_prov_{Y}.parquet')
w = set(l.rgst_no[l.y == 1]); v = pd.read_parquet(W + rf'\verify_0\v1_pos_{Y}.parquet'); vi = set(v.rgst_no)
print('worker', len(w), 'indep', len(vi), 'both', len(w & vi), 'w-only', len(w - vi), 'v-only', len(vi - w))
d = b.merge(l, on='rgst_no').merge(v.rename(columns={'n': 'vn'}), on='rgst_no', how='left')
x = d[d.rgst_no.isin(vi - w)]
print(x[['rgst_no', 'buyer', 'vn', 'buyer_type', 'buyer_nonempty', 'ksic_x', 'y_A', 'y_AB']].head(15).to_string())
print('v-only: buyer==vn', (x.buyer == x.vn).mean(), 'type', x.buyer_type.value_counts().to_dict(), 'nonempty', x.buyer_nonempty.value_counts().to_dict())
x2 = d[d.rgst_no.isin(w - vi)]; print(x2[['rgst_no', 'buyer', 'buyer_type', 'ksic_x']].head(10).to_string())
