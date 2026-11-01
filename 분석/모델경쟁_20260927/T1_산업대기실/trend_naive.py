# 대기실 지수 vs 산업 단위 순진 기준선(직전 5년 신규진입 법인 수, 기준시점까지 등록된 것만) — 산업 간 순위상관
import pandas as pd, numpy as np
from lib import *
from scipy.stats import spearmanr
t = pd.read_csv(T1/'trend_waiting_room_index_2005_2009.csv', encoding='utf-8-sig', dtype={'ksic': str})
ins = load()['ins']; res = []
for c, g in t.groupby('cut'):
    k = ins[ins.rgst_date <= f'{c}1231'].sort_values('appl_date').drop_duplicates(['applicant_cd', 'ksic'])
    prev = k[(k.ay > c - 5) & (k.ay <= c)].ksic.value_counts()
    g = g.assign(naive_prev_entries=g.ksic.map(prev).fillna(0).values)
    r_n = spearmanr(g.naive_prev_entries, g.realized).correlation; r_i = g.rho_all.iloc[0]
    # 지수의 증분: 순진기준선 순위를 통제한 편상관(순위 잔차)
    from numpy.linalg import lstsq
    a = g.naive_prev_entries.rank().values; b = g['index'].rank().values; y = g.realized.rank().values
    X = np.c_[np.ones(len(a)), a]; rb = b - X @ lstsq(X, b, rcond=None)[0]; ry = y - X @ lstsq(X, y, rcond=None)[0]
    pc = np.corrcoef(rb, ry)[0, 1]
    res.append(dict(cut=c, rho_index=round(r_i, 3), rho_naive_prev5=round(r_n, 3), partial_rho_index_given_naive=round(pc, 3)))
    t.loc[g.index, 'naive_prev_entries'] = g.naive_prev_entries.values
print(pd.DataFrame(res).to_string(index=False))
t.to_csv(T1/'trend_waiting_room_index_2005_2009.csv', index=False, encoding='utf-8-sig')
pd.DataFrame(res).to_csv(T1/'trend_index_validation.csv', index=False, encoding='utf-8-sig')
