# -*- coding: utf-8 -*-
"""s5b: build_ext_r3b.ros_feats 와 같은 규칙으로 ros_sci·ros_ppp 재계산 (V0 재현 확인, V1, V2, V3=RoS 연결 없음(결측 처리 진단)).
입력: cache/ros_cell_{V}.parquet, 트랙 R3 extB_cell_pub{,_final}.parquet(셀 목록·저장값 대조, 읽기만)
출력: cache/scen_ros_{dev,fin}.parquet (cell, t0, ros_sci_V*, ros_ppp_V*), res/s5b_feats.json"""
import json
from common import *
import numpy as np, pandas as pd
R = {}
def ros_feats(ci, t0s, cells, k=5.0, zero_cell=False):
    # build_ext_r3b.ros_feats 그대로(입력 표만 교체)
    ci = ci[ci.issue_year >= min(t0s) - 9].copy()
    ci['cell'] = ci.ksic + '|' + ci.mg
    ksic_c = pd.Series(cells).str[:5].values
    out = []
    for t0 in t0s:
        base = ci[(ci.first_rec_year <= t0 - 2) & (ci.issue_year <= t0) & (ci.first_pub_year <= t0)]
        w10 = base[base.issue_year >= t0 - 9]
        A = w10.groupby('cell')[['n_us_pat', 'n_us_pat_sci', 'n_paper_cit']].sum().reindex(cells).fillna(0.0)
        ks = w10.groupby('ksic')[['n_us_pat', 'n_us_pat_sci', 'n_paper_cit']].sum()
        ks_sh = (ks.n_us_pat_sci / ks.n_us_pat.clip(lower=1)).reindex(ksic_c).fillna(0.2).values
        ks_pp = (ks.n_paper_cit / ks.n_us_pat.clip(lower=1)).reindex(ksic_c).fillna(1.0).values
        if zero_cell: A = A * 0.0   # V3: 셀 연결 정보 제거(모든 셀 = 결측 처리값 = KSIC 축소값)
        n = A.n_us_pat.values
        sci = (A.n_us_pat_sci.values + k * ks_sh) / (n + k)
        ppp = np.log1p((A.n_paper_cit.values + k * ks_pp) / (n + k))
        out.append(pd.DataFrame(dict(cell=cells, t0=t0, ros_sci=sci, ros_ppp=ppp, ros_n=n)))
    return pd.concat(out, ignore_index=True)
CI = {V: pd.read_parquet(f"{W}/cache/ros_cell_{V}.parquet") for V in ('V0', 'V1', 'V2')}
CI['V3'] = None   # 진단: 셀 연결 정보 제거(모든 셀이 미연결 셀의 결측 처리값 = KSIC 평균)
for mode, suf in (('dev', ''), ('fin', '_final')):
    X = pd.read_parquet(f"{T3R3}/extB_cell_pub{suf}.parquet")
    t0s = sorted(X.t0.unique()); cells_all = sorted(X.cell.unique())
    out = X[['cell', 't0', 'ros_sci', 'ros_ppp', 'ros_lnus']].rename(columns={'ros_sci': 'ros_sci_off', 'ros_ppp': 'ros_ppp_off'})
    for V, ci in CI.items():
        f = ros_feats(CI['V0'] if V == 'V3' else ci, t0s, cells_all, zero_cell=(V == 'V3'))
        out = out.merge(f.rename(columns={'ros_sci': f'ros_sci_{V}', 'ros_ppp': f'ros_ppp_{V}', 'ros_n': f'ros_n_{V}'}), on=['cell', 't0'], how='left')
    assert out[[f'ros_sci_{V}' for V in CI]].notna().all().all()
    d1 = float(np.abs(out.ros_sci_V0 - out.ros_sci_off).max()); d2 = float(np.abs(out.ros_ppp_V0 - out.ros_ppp_off).max())
    R[f'{mode}_V0_maxdiff'] = [d1, d2]; R[f'{mode}_rows'] = [len(out), int(min(t0s)), int(max(t0s))]
    for V in ('V1', 'V2', 'V3'):
        R[f'{mode}_{V}_spearman_by_t0'] = {int(t): [round(float(g[f'ros_sci_{V}'].rank().corr(g.ros_sci_V0.rank())), 4), round(float(g[f'ros_ppp_{V}'].rank().corr(g.ros_ppp_V0.rank())), 4)]
                                         for t, g in out.groupby('t0') if t >= 2010}
    R[f'{mode}_linked_share'] = {V: float((out[f'ros_n_{V}'] > 0).mean()) for V in ('V0', 'V1', 'V2')}
    out.to_parquet(f"{W}/cache/scen_ros_{mode}.parquet")
    log(mode, R[f'{mode}_V0_maxdiff'], R[f'{mode}_rows'], R[f'{mode}_linked_share'])
json.dump(R, open(W + "/res/s5b_feats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: v for k, v in R.items() if 'spearman' in k}, ensure_ascii=False)[:3000])
