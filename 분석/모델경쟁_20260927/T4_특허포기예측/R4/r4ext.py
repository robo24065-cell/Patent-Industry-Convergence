# -*- coding: utf-8 -*-
"""T4 4라운드 새 피처 gv_ — NTIS 국가R&D 성과 특허 연결(NTIS Open API 활용, `_shared/ext/ntis_patent_link.parquet`, 읽기만).
시점 규칙(PROTOCOL_R4 §3): 기준일 L 에서 **성과 레코드 Year ≤ L연도−1** 인 연결만 '알 수 있던' 정부 기원으로 본다.
NTIS 성과가 2007년부터라 **등록연도 < 2007 특허는 NaN**('정부 R&D 아님'이 아니라 관측 불가).
  gv_univ / gv_gri / gv_big / gv_sme / gv_oth : 수행주체(대학 / 출연연·국공립연 / 대기업·중견 / 중소 / 기타·정부부처)별 정부 기원 플래그
  gv_self   : 기업(대·중견·중소)이 직접 수행한 과제의 특허
  gv_lnproj : log(1 + 연결 과제 수)   (T4-N2)
  gv_any    : 어느 주체든 정부 기원
"""
import os
import numpy as np, pandas as pd
import r4lib
R = r4lib.R
EXTD = os.path.join(os.path.dirname(r4lib.T4), '_shared', 'ext')
GRP = {'대학': 'univ', '출연연구소': 'gri', '국공립연구소': 'gri', '대기업': 'big', '중견기업': 'big', '중소기업': 'sme', '기타': 'oth', '정부부처': 'oth'}
GRPS = ['univ', 'gri', 'big', 'sme', 'oth']
YRS = np.arange(2007, 2027)
_C = {}


def _tab():
    if _C:
        return _C
    pt = pd.read_parquet(os.path.join(r4lib.T4, 'cache', 'patent.parquet'), columns=['rgst_no', 'ksic', 'rgstDate'])
    rd = pd.to_datetime(pt.rgstDate.astype(str).str.strip(), format='%Y%m%d', errors='coerce')
    keep = rd.notna().values & (pt.rgst_no.astype(str).str[:2] == '10').values
    pt = pt[keep].reset_index(drop=True); ry = rd[keep].dt.year.values
    n = len(pt)
    lk = pd.read_parquet(os.path.join(EXTD, 'ntis_patent_link.parquet'), columns=['rgst_no', 'Year', 'PerformAgent', 'ProjectID'])
    lk = lk[lk.rgst_no.notna() & (lk.rgst_no.astype(str).str.len() > 5)].copy()
    lk['grp'] = lk.PerformAgent.astype(str).map(GRP).fillna('oth')
    lk['yr'] = pd.to_numeric(lk.Year, errors='coerce')
    lk = lk[lk.yr.notna()]
    pid = pd.Series(np.arange(n), index=pt.rgst_no.astype(str).values)
    pid = pid[~pid.index.duplicated()]
    lk = lk[lk.rgst_no.astype(str).isin(pid.index)]
    lk['pid'] = pid[lk.rgst_no.astype(str).values].values
    first = {}
    for g in GRPS:
        a = np.full(n, np.inf); s = lk[lk.grp == g].groupby('pid').yr.min()
        a[s.index.values] = s.values; first[g] = a
    pj = lk.groupby(['pid', 'ProjectID']).yr.min().reset_index()          # 과제별 첫 레코드 연도
    cnt = np.zeros((n, len(YRS)), np.int16)
    yi = np.clip(pj.yr.values.astype(int) - YRS[0], 0, len(YRS) - 1)
    np.add.at(cnt, (pj.pid.values, yi), 1)
    _C.update(first=first, cum=np.cumsum(cnt, 1), ry=ry, ksic=pt.ksic.astype(str).values)
    return _C


def add(d, cols):
    T = _tab()
    pid = d.pid.values.astype(int)
    chk = (T['ksic'][pid] == d.ksic.astype(str).values).mean()
    assert chk > 0.999, f'pid→KSIC 불일치 {chk}'
    Ly = pd.to_datetime(d.L.astype(str).str[:10]).dt.year.values
    lim = Ly - 1                                                      # 레코드 Year ≤ L연도−1
    na = T['ry'][pid] < 2007                                          # 연결 불가 구간 → NaN
    flag = {g: (T['first'][g][pid] <= lim) for g in GRPS}
    out = {}
    for g in GRPS:
        out[f'gv_{g}'] = flag[g].astype('float32')
    out['gv_self'] = (flag['big'] | flag['sme']).astype('float32')
    out['gv_any'] = np.any(np.column_stack([flag[g] for g in GRPS]), 1).astype('float32')
    j = np.clip(lim - YRS[0], -1, len(YRS) - 1)
    c = np.where(j >= 0, T['cum'][pid, np.maximum(j, 0)], 0)
    out['gv_lnproj'] = np.log1p(c).astype('float32')
    for k in cols:
        v = out[k].copy(); v[na] = np.nan
        d[k] = v
    return d
