# -*- coding: utf-8 -*-
r"""C19-T2 s1: 네 프레임(frame_cls·frame_haz·ev·ev_haz)의 fam_n 변형 벡터 계산(행 순서 그대로) → cache/famn/{var}__{frame}.npy
 var: P0(원문 재계산), F1(국가+접미 제거+앞0 제거), F1x(F1 + US 공개 11↔10자리, JP 공개 연호↔서기), F2(상수), F3s1·F3s2(위약)
 fam_n 정의(feat2.fam_dated): yr <= L연도-1 인 토큰 수. 읽기만: T4 cache/patent.parquet·s2_fam_tokens.parquet·s2_fam_pid.parquet, R2/runs_final/F22/*.parquet
 출력: cache/famn/*.npy, cache/s1_famn.json, cache/s1_tokens_keys.parquet(pid·yr·cc·tok·k1·kx, 후속 점검용)"""
import os, sys, re
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib19 import *
import numpy as np, pandas as pd

OD = J(C, 'famn'); os.makedirs(OD, exist_ok=True)
pt = pd.read_parquet(J(T4, 'cache', 'patent.parquet'), columns=['rgst_no', 'rgstDate', 'applNo'])
pt = pt[pt.rgstDate.notna() & (pt.rgstDate.astype(str).str.strip() != '') & (pt.rgst_no.str[:2] == '10')].reset_index(drop=True)
pt['pid'] = np.arange(len(pt)); NP = len(pt)
tk = pd.read_parquet(J(T4, 'cache', 's2_fam_tokens.parquet'))
m = pd.Series(pt.pid.values, index=pt.applNo.values); m = m[~m.index.duplicated()]
tk['pid'] = tk.applNo.map(m); tk = tk[tk.pid.notna()].reset_index(drop=True); tk['pid'] = tk.pid.astype('int64')
fp = pd.read_parquet(J(T4, 'cache', 's2_fam_pid.parquet'))
R = {'tokens_all': None, 'tokens_mapped': int(len(tk)), 's2_fam_pid_rows': int(len(fp)),
     'pid_yr_cc_equal_s2_fam_pid': bool(len(fp) == len(tk) and (fp.pid.values == tk.pid.values).all() and (fp.yr.values == tk.yr.values).all()
                                       and (fp.cc.astype(str).values == tk.cc.values).all())}
out('mapped', R)
# ---------------- 정규화 키
cc = tk.tok.str[:2]; rest = tk.tok.str[2:]
rest1 = rest.str.replace(r'[A-Z][0-9]?$', '', regex=True)           # 종류 코드 접미(A1·B2·A) 제거 — 연도 추정 토큰은 숫자뿐이라 실제 변화 0 예상
R['suffix_stripped_tokens'] = int((rest1 != rest).sum())
k1 = cc + rest1.str.lstrip('0')
L_ = rest1.str.len()
kx = k1.copy()
mu = (cc == 'US') & (L_ == 11) & (rest1.str[4] == '0')
kx[mu] = 'US' + rest1[mu].str[:4] + rest1[mu].str[5:]
mj = (cc == 'JP') & (L_ == 8) & (~rest1.str.startswith('0'))
yy = rest1[mj].str[:2].astype(int)
kx[mj] = 'JP' + (1988 + yy).astype(str) + rest1[mj].str[2:]
tk['k1'] = k1.values; tk['kx'] = kx.values
for k in ['k1', 'kx']:
    g = tk.groupby(['pid', k]).yr.agg(['nunique', 'size'])
    R[f'{k}_groups_yr_differ'] = int((g['nunique'] > 1).sum())
    R[f'{k}_dup_tokens'] = int((g['size'] - 1).sum())
    R[f'{k}_max_group_size'] = int(g['size'].max())
R['us_pub11_tokens_converted'] = int(mu.sum()); R['jp_era_tokens_converted'] = int(mj.sum())
tk[['pid', 'yr', 'cc', 'tok', 'k1', 'kx']].to_parquet(J(C, 's1_tokens_keys.parquet'), index=False)
out('keys', {k: v for k, v in R.items() if 'k1' in k or 'kx' in k})

# ---------------- 변형별 (pid, yr, 가중치) 토큰표
pid = tk.pid.values; yr = tk.yr.values.astype(int)
tabs = {'P0': (pid, yr, np.ones(len(tk)))}
for v, k in [('F1', 'k1'), ('F1x', 'kx')]:
    d = tk.sort_values('yr', kind='stable').drop_duplicates(['pid', k])      # 같은 키의 가장 이른 yr
    tabs[v] = (d.pid.values, d.yr.values.astype(int), np.ones(len(d)))
dup1 = tk.duplicated(['pid', 'k1']).values
pyr = pd.Series(dup1).groupby(yr).mean()
R['F1_dup_share_by_yr_all'] = {int(a): round(float(b), 5) for a, b in pyr.items()}
pt_yr = pyr.reindex(yr).values
for s in (1, 2):
    rng = np.random.default_rng(s)
    flag = rng.random(len(tk)) < pt_yr
    dirn = rng.choice(np.array([-1, 1]), size=NP)
    w = 1.0 + flag * dirn[pid]
    tabs[f'F3s{s}'] = (pid, yr, w)
    R[f'F3s{s}_flag_share'] = float(flag.mean()); R[f'F3s{s}_removed'] = int((w == 0).sum()); R[f'F3s{s}_doubled'] = int((w == 2).sum())
R['F1_flag_share_actual'] = float(dup1.mean())


def famn(tab, Lyears, pids):
    p_, y_, w_ = tab
    outv = np.zeros(len(pids), dtype='float64')
    for c in np.unique(Lyears):
        sel = y_ <= c - 1
        cnt = np.bincount(p_[sel], weights=w_[sel], minlength=NP)
        ix = np.where(Lyears == c)[0]
        outv[ix] = cnt[pids[ix]]
    return outv


import pyarrow.parquet as pq
F2c = None
for fr in FRAMES:
    d = pq.read_table(J(F22, fr + '.parquet'), columns=['pid', 'L', 'fam_n']).to_pandas()
    Ly = pd.to_datetime(d.L.astype(str)).dt.year.values
    pids = d.pid.values.astype('int64')
    stored = d.fam_n.values.astype('float64')
    res = {}
    for v, tab in tabs.items():
        res[v] = famn(tab, Ly, pids)
    if fr == 'frame_cls':
        F2c = float(stored.mean())
    res['F2'] = np.full(len(d), F2c)
    rr = {'rows': int(len(d)), 'L_years': sorted(int(x) for x in np.unique(Ly)), 'repro_max_abs_P0_vs_stored': float(np.abs(res['P0'] - stored).max()),
          'fam_pos': int((stored > 0).sum())}
    for v in ['F1', 'F1x', 'F3s1', 'F3s2']:
        ch = res[v] != res['P0']
        rr[f'{v}_changed'] = int(ch.sum()); rr[f'{v}_changed_share'] = float(ch.mean())
        rr[f'{v}_changed_share_famPos'] = float(ch[stored > 0].mean()) if (stored > 0).any() else None
        rr[f'{v}_mean'] = float(res[v].mean())
    rr['P0_mean'] = float(res['P0'].mean())
    for v, a in res.items():
        np.save(J(OD, f'{v}__{fr}.npy'), a.astype('float32'))
    R[fr] = rr
    out(fr, rr)
R['F2_constant'] = F2c
jdump(R, J(C, 's1_famn.json'))
out('done')
