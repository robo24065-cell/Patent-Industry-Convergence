# s2: C06-B 「특허기업의 신규 사업장 추가 개설」 단위(as-of 연결)·결과(개발 S1 ≤ 2024-12)·잠금 위험집합 크기, P1·P3 집계.
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from lib6 import *
T0 = time.time()
DEV = ['201512', '201612', '201712', '201812', '201912', '202012', '202112', '202212', '202312']
LOCK = '202412'
LOCKED = {'202512', '202608'}
REF = {'201512': '20151130', '201612': '20161130', '201712': '20171130', '201812': '20181130', '201912': '20191130',
       '202012': '20201231', '202112': '20211231', '202212': '20221231', '202312': '20231231', '202412': '20241231'}
NEXT = {s: str(int(s[:4]) + 1) + '12' for s in DEV}


def load_nps(s):
    assert s not in LOCKED
    d = pd.read_parquet(EXT + rf'\firm_nps_wkpl_{s}.parquet', columns=['name', 'biz6', 'stat', 'adpt_dt', 'n_emp'])
    d = d[(d.stat.astype(str) == '1') & d.biz6.astype(str).str[3:5].isin(CORP_TC)].copy()
    d['key'] = d.biz6.astype(str) + '|' + d.name.map(norm)
    d['ad'] = d.adpt_dt.fillna('').astype(str).str.replace('-', '').str[:8]
    return d


ca = load_ca(); nm2b6, _ = name_map(ca)
docs = load_docs().drop_duplicates('appl_no')
docs = docs[docs.vis.notna() & docs.nname.isin(nm2b6.keys())]
first_vis = docs.groupby('nname').vis.min()
OUT, LOG = [], {}
cache = {}
for s0 in DEV + [LOCK]:
    ref0 = REF[s0]
    d0 = cache.pop(s0, None)
    if d0 is None:
        d0 = load_nps(s0)
    f0 = d0.groupby('key').agg(E0=('n_emp', 'sum'), nw0=('ad', 'size'))
    names = first_vis[first_vis <= ref0].index
    keys = set(pd.Series(names).map(nm2b6).astype(str) + '|' + pd.Series(names))
    u = f0[f0.index.isin(keys) & (f0.E0 >= 5)].reset_index()
    u['s0'] = s0
    rec = dict(s0=s0, units=len(u), nps_corp_keys_E0ge5=int((f0.E0 >= 5).sum()))
    if s0 != LOCK:
        s1 = NEXT[s0]; ref1 = REF[s1]
        assert s1 <= '202412'
        d1 = load_nps(s1); cache[s1] = d1
        d1u = d1[d1.key.isin(set(u.key))]
        f1 = d1u.groupby('key').agg(nw1=('ad', 'size'))
        newk = set(d1u[(d1u.ad > ref0) & (d1u.ad <= ref1)].key)
        u['nw1'] = u.key.map(f1.nw1).fillna(0).astype(int)
        u['y'] = (u.key.isin(newk) & (u.nw1 > u.nw0)).astype(int)
        u['y_newad_only'] = u.key.isin(newk).astype(int)  # 진단: 사업장 수 조건 없이
        rec.update(s1=s1, pos=int(u.y.sum()), rate=round(float(u.y.mean()), 5), pos_newad_only=int(u.y_newad_only.sum()),
                   multi_wkpl_share_S0=round(float((u.nw0 > 1).mean()), 4), months=round((pd.Timestamp(ref1) - pd.Timestamp(ref0)).days / 30.44, 1))
    else:
        u['y'] = -1
    OUT.append(u); LOG[s0] = rec
    print(rec, round(time.time() - T0), flush=True)
units = pd.concat(OUT, ignore_index=True)
units.to_parquet(os.path.join(C, 'B_units.parquet'), index=False)
dev = units[units.y >= 0]
g = dev.groupby('s0').agg(n=('y', 'size'), pos=('y', 'sum'), rate=('y', 'mean'))
uniq = int(dev[dev.y == 1].key.nunique())
lock_n = int((units.s0 == LOCK).sum()); last2 = float(g.rate.iloc[-2:].mean())
P1 = dict(dev_by_origin=g.reset_index().to_dict('records'), dev_sum_events=int(g.pos.sum()), dev_unique_events=uniq,
          lock_risk_set=lock_n, last2_rate=last2, lock_expected=round(lock_n * last2, 1))
P1['pass_a'] = uniq >= 1000; P1['pass_b'] = lock_n * last2 >= 1000; P1['pass'] = bool(P1['pass_a'] and P1['pass_b'])
P1['rate_max_min_ratio'] = float(g.rate.max() / g.rate.min()) if g.rate.min() > 0 else None
LOG['P1'] = P1
LOG['P3'] = dict(n_dev_origins=len(g), min_pos=int(g.pos.min()), pass_=bool(len(g) >= 4 and g.pos.min() >= 100), warn_lt200=int((g.pos < 200).sum()))
json.dump(LOG, open(os.path.join(C, 's2_B_log.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: LOG[k] for k in ('P1', 'P3')}, ensure_ascii=False, indent=1, default=str))
print('done', round(time.time() - T0))
