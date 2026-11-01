# -*- coding: utf-8 -*-
# P0(c) 보충: 위험모형 학습 양성 (pid,d) 고유 단위가 cd<=C 소멸 사건과 모두 짝지어지는지
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, c08lib as C
D = C.load(); lap = C.lapse_events(D)
fh = pd.read_parquet(os.path.join(C.FUT, "frame_haz.parquet"), columns=["pid", "L", "y", "d"])
h1 = fh[fh.y == 1]
u = pd.DataFrame({'pid': h1.pid.values, 'd': pd.to_datetime(h1.d).values.astype('datetime64[D]').astype('int64')}).drop_duplicates()
key = pd.DataFrame({'pid': lap.pid.values, 'd': lap.E.values.astype('datetime64[D]').astype('int64'), 'cd': lap.cd.values})
m = u.merge(key, on=['pid', 'd'], how='left')
g = m.groupby(['pid', 'd']).cd.min()
Cc = pd.Timestamp('2025-08-22')
o = dict(rows_pos=len(h1), uniq_pos=len(u), uniq_matched=int(g.notna().sum()), uniq_unmatched=int(g.isna().sum()),
         matched_min_cd_gt_C=int((g > Cc).sum()), matched_cd_max=str(g.max()),
         frame_haz_rows=len(fh), haz_L_values=sorted(map(str, fh.L.unique()))[-5:], y_mean=float(fh.y.mean()))
# 짝 없는 양성: lmax 정의 차이인지(같은 pid의 다른 E와 매칭) 확인
un = g[g.isna()].reset_index()[['pid', 'd']]
if len(un):
    lp = key.groupby('pid').agg(n=('d', 'size'), cdmax=('cd', 'max'))
    j = un.merge(lp, left_on='pid', right_index=True, how='left')
    o['unmatched_pid_has_any_lapse'] = int(j.n.notna().sum()); o['unmatched_pid_lapse_cd_max'] = str(j.cdmax.max())
    o['unmatched_pid_any_lapse_cd_gt_C'] = int((j.cdmax > Cc).sum())
print(json.dumps(o, ensure_ascii=False, indent=1)); C.jdump(o, os.path.join(C.CACHE, 's02b_hazpos.json'))
