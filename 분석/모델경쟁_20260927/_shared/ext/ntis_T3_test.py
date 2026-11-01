# -*- coding: utf-8 -*-
"""T3 의미 검정: 셀(KSIC×IPC 메인그룹)의 정부R&D 성과 특허 비중·과제 연구비 성장이 이후 KR 셀 성장을 선행하는가.
- 평가행·레이블: T3 R2 개발 캐시 `R2/cache_dev/feat_{t0}.parquet`(출원연도 ≤2019로 만든 것, 읽기만). t0=2010..2016, 적격 tot_W≥50,
  부상 yb = t0별 성장률 상위 10%(T3 t3lib.breakout 규칙 그대로). 🔒 t0 ≥2017 은 쓰지 않는다(assert).
- 시점 규칙:
  * 비중: t0-2..t0 출원 KR 특허 중 NTIS 성과 레코드 Year ≤ t0 로 연결되는 비율(스냅샷이라 준-정합).
  * 연구비: 연도 y 과제 연구비는 y+1 년부터 사용(avail) → t0 에는 y ≤ t0-1. 분류(공통 대분류 20)→셀 배분표는 성과 Year ≤ t0-2 연결만.
  * KSIC 수준(B 임무 산출): ntis_project_ksic_year 의 a_link_pit / b2_dict_hard 비중, y ≤ t0-1.
출력: ntis_T_work/ntis_T3_*.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, duckdb
from ntis_T_common import *
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

T0S = list(range(2010, 2017))
assert max(T0S) + 3 <= 2019
CD = os.path.join(T3, 'R2', 'cache_dev')

con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2")
con.execute(r"""create macro mgn(s) as (case
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{1,4}/') then substr(replace(s,' ',''),1,4)||' '||cast(cast(regexp_extract(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z]([0-9]{1,4})/',1) as int) as varchar)
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{3,6}$') then substr(replace(s,' ',''),1,4)||' '||cast(cast(substr(replace(s,' ',''),5,length(replace(s,' ',''))-6) as int) as varchar)
  else null end)""")
kr = con.execute(f"""select distinct ksic, mgn(ipc_raw) mg, year, appl_no from read_parquet('{CD}/base_rows.parquet')
                     where cc='KR' and year between 2002 and 2016""").df()
kr = kr[kr.mg.notna()].copy()
kr['cell'] = kr.ksic.astype(str) + '|' + kr.mg
print('KR 셀 행', len(kr), flush=True)

# ---- 특허별 NTIS 최소 레코드 연도(전체·공공(대학+출연연)·기업)
lk = pd.read_parquet(os.path.join(EXT, 'ntis_patent_link.parquet'), columns=['appl_no', 'Year', 'PerformAgent', 'ProjectID'])
lk['yr'] = pd.to_numeric(lk.Year, errors='coerce'); lk['grp'] = lk.PerformAgent.astype(str).map(GRP_MAP).fillna('oth')
lk['pg'] = np.where(lk.grp.isin(['univ', 'gri']), 'pub', np.where(lk.grp.isin(['big', 'sme']), 'firm', 'oth'))
mn = lk.groupby(['appl_no', 'pg']).yr.min().unstack(); mn['any'] = mn.min(1)
kr = kr.merge(mn[['any', 'pub', 'firm']].astype(float), left_on='appl_no', right_index=True, how='left')

# ---- 과제 연구비(공통 대분류) 와 분류→셀 배분표 재료
pj = pd.read_parquet(os.path.join(EXT, 'ntis_project_raw.parquet'), columns=['pjt_no', 'year', 'cls_L_canon', 'gov_funds'])
pj['pjt_no'] = pj.pjt_no.astype(str); pj['cls'] = pj.cls_L_canon.astype(str)
pj = pj[pj.cls.notna() & (pj.cls != 'NA') & (pj.cls != '<NA>')]
F = pj.groupby(['cls', 'year']).gov_funds.sum().unstack(fill_value=0.0)           # 분류 × 연도
cls_of = pj.drop_duplicates('pjt_no').set_index('pjt_no').cls
rp = pd.read_parquet(os.path.join(EXT, 'ntis_rpatent_raw.parquet'), columns=['ResultID', 'Year', 'ProjectID', 'is_kr', 'IprType'])
rp = rp[(rp.is_kr == True) & (rp.IprType.astype(str) == '특허')].copy()
rp['cls'] = rp.ProjectID.astype(str).map(cls_of); rp['yr'] = pd.to_numeric(rp.Year, errors='coerce')
rp = rp[rp.cls.notna()]
lk['cls'] = lk.ProjectID.astype(str).map(cls_of)
# 링크행 → 셀 (특허가 여러 셀이면 1/m)
cells_of = kr[['appl_no', 'cell']].drop_duplicates()
cells_of['m'] = cells_of.groupby('appl_no').cell.transform('size')
lkc = lk[lk.cls.notna()][['appl_no', 'yr', 'cls']].merge(cells_of, on='appl_no')
lkc['w'] = 1.0 / lkc.m
print('분류 있는 KR 특허 성과', len(rp), '셀 연결 링크행', len(lkc), flush=True)

# ---- KSIC 수준 B 신호
kq = pd.read_parquet(os.path.join(EXT, 'ntis_project_ksic_year.parquet'))
kq['ksic'] = kq.ksic.astype(str)


def ksic_grow(method, t0):
    s = kq[kq.method == method].pivot_table(index='ksic', columns='year', values='gov_share_of_year')
    need = [t0 - 1, t0 - 2, t0 - 3, t0 - 4]
    if not all(y in s.columns for y in need):
        return pd.Series(dtype=float)
    a = s[t0 - 1] + s[t0 - 2]; b = s[t0 - 3] + s[t0 - 4]
    eps = 1e-6
    return np.log((a + eps) / (b + eps)).where(b.notna() & a.notna())


def breakout(y, top=0.10, tie=None):
    k = max(1, int(round(top * len(y))))
    order = np.lexsort((-tie, -y))
    yb = np.zeros(len(y), dtype=int); yb[order[:k]] = 1
    return yb


frames = []
for t0 in T0S:
    d = pd.read_parquet(os.path.join(CD, f'feat_{t0}.parquet'))
    d = d[d.tot_W >= 50].reset_index(drop=True)
    d['yb'] = breakout(d.y.values, tie=(d.kr_fut - d.kr_W).values)
    w = kr[(kr.year >= t0 - 2) & (kr.year <= t0)]
    g = w.groupby('cell').agg(n=('appl_no', 'size'), gov=('any', lambda v: int((v <= t0).sum())),
                              pub=('pub', lambda v: int((v <= t0).sum())), firm=('firm', lambda v: int((v <= t0).sum())))
    w0 = kr[(kr.year >= t0 - 5) & (kr.year <= t0 - 3)]
    g0 = w0.groupby('cell').agg(n0=('appl_no', 'size'), gov0=('any', lambda v: int((v <= t0).sum())))
    d = d.merge(g, left_on='cell', right_index=True, how='left').merge(g0, left_on='cell', right_index=True, how='left')
    for c in ('n', 'gov', 'pub', 'firm', 'n0', 'gov0'):
        d[c] = d[c].fillna(0)
    # KSIC 사전(경험적 베이즈 수축, 강도 10)
    ks = kr[(kr.year >= t0 - 2) & (kr.year <= t0)].assign(gv=lambda x: (x['any'] <= t0)).groupby('ksic').gv.mean()
    p = d.ksic.astype(str).map(ks.rename(index=str)).fillna(ks.mean()).values
    d['gsh'] = (d.gov + 10 * p) / (d.n + 10)
    d['gsh_pub'] = (d.pub + 10 * p * 0.6) / (d.n + 10)
    d['gsh_firm'] = (d.firm + 10 * p * 0.3) / (d.n + 10)
    d['gsh_rel'] = np.log(d.gsh / p)                                  # KSIC 대비 상대 비중
    p0 = d.gov0.sum() / max(d.n0.sum(), 1)
    d['gsh_chg'] = d.gsh - (d.gov0 + 10 * p0) / (d.n0 + 10) if t0 - 5 >= 2007 else np.nan
    # 분류→셀 배분표(성과 Year ≤ t0-2) × 연구비(y ≤ t0-1)
    den = rp[rp.yr <= t0 - 2].groupby('cls').size()
    num = lkc[lkc.yr <= t0 - 2].groupby(['cell', 'cls']).w.sum().unstack(fill_value=0.0)
    num = num.reindex(columns=den.index, fill_value=0.0)
    S = num / den                                                       # 셀 × 분류: 분류 c 성과 중 그 셀 비중
    Fy = lambda y: F[y].reindex(den.index).fillna(0.0) if y in F.columns else pd.Series(0.0, index=den.index)
    A = {y: S.values @ Fy(y).values for y in range(t0 - 4, t0)}
    A = pd.DataFrame(A, index=S.index)
    d = d.merge(A, left_on='cell', right_index=True, how='left')
    for y in range(t0 - 4, t0):
        d[y] = d[y].fillna(0.0)
    d['f_link'] = d.cell.isin(S.index[num.sum(1) > 0]).astype(float)
    d['f_grow'] = np.log((d[t0 - 1] + d[t0 - 2] + 1e7) / (d[t0 - 3] + d[t0 - 4] + 1e7))
    d['f_int'] = np.log((d[t0 - 1] + 1e7) / 1e7) - np.log1p(d.kr_W)
    d = d.drop(columns=[t0 - 4, t0 - 3, t0 - 2, t0 - 1])
    for m, nm in (('a_link_pit', 'k_a_grow'), ('b2_dict_hard', 'k_b2_grow'), ('b1_tfidf_hard', 'k_b1_grow')):
        kg = ksic_grow(m, t0)
        d[nm] = d.ksic.astype(str).map(kg) if len(kg) else np.nan
    frames.append(d)
    print(t0, len(d), 'gsh 평균', round(float(d.gsh.mean()), 4), 'f_link', round(float(d.f_link.mean()), 3), peak_mem_gb(), flush=True)

D = pd.concat(frames, ignore_index=True)
FEATS = ['gsh', 'gsh_rel', 'gsh_pub', 'gsh_firm', 'gsh_chg', 'f_grow', 'f_int', 'k_a_grow', 'k_b2_grow', 'k_b1_grow']
CTRL = ['kr_mom', 'kr_lvl', 'tot_mom', 'fx_mom', 'kr_share', 'kr_g1', 'ksic_kr_mom']
D[['cell', 'ksic', 'mg', 't0', 'y', 'yb', 'kr_W', 'n', 'gov', 'pub', 'firm'] + FEATS].to_parquet(out('ntis_T3_cellpanel.parquet'), index=False)
for c in CTRL:
    D[c] = D[c].fillna(0.0)

# ---- (1) 원점별: 편 Spearman(통제 잔차) · 로지스틱 계수(순위정규화 피처, 통제 포함) · KSIC 고정효과 LPM
res = []
for f in FEATS:
    for t0, g in D.groupby('t0'):
        g = g[g[f].notna()]
        if len(g) < 200 or g[f].nunique() < 5:
            continue
        x = rank_norm(g[f]).values
        C = np.column_stack([np.ones(len(g))] + [rank_norm(g[c]).values for c in CTRL])
        rx = x - C @ np.linalg.lstsq(C, x, rcond=None)[0]
        ry0 = rank_norm(g.y).values; ry = ry0 - C @ np.linalg.lstsq(C, ry0, rcond=None)[0]
        rho_p = float(np.corrcoef(rx, ry)[0, 1]); rho = float(spearmanr(g[f], g.y)[0])
        Xl = np.column_stack([x] + [zscore(g[c]).values for c in CTRL])
        lr = LogisticRegression(C=1e4, max_iter=1000).fit(Xl, g.yb)
        gg = g.assign(xr=x)
        fe = fe_ols(gg.assign(**{c: zscore(gg[c]) for c in CTRL}), 'yb', ['xr'] + CTRL, 'ksic', 'mg').loc['xr']
        res.append(dict(feat=f, t0=t0, n=len(g), rho=round(rho, 4), rho_partial=round(rho_p, 4),
                        logit_coef=round(float(lr.coef_[0][0]), 4), fe_ksic_coef=round(float(fe.coef), 4), fe_ksic_t=round(float(fe.t), 2)))
R = pd.DataFrame(res); R.to_csv(out('ntis_T3_by_origin.csv'), index=False, encoding='utf-8-sig')

# 귀무분포: 원점마다 피처를 KSIC 안에서 섞어 편상관 100회 → 평균 편상관의 귀무 SD
rng = np.random.default_rng(0)
nul = []
for f in ['gsh', 'gsh_rel', 'gsh_pub', 'gsh_firm', 'gsh_chg', 'f_grow', 'f_int']:
    vals = []
    for it in range(100):
        rr = []
        for t0, g in D.groupby('t0'):
            g = g[g[f].notna()]
            if len(g) < 200 or g[f].nunique() < 5:
                continue
            xs = g.groupby('ksic')[f].transform(lambda s: s.sample(frac=1, random_state=int(rng.integers(1e9))).values)
            x = rank_norm(xs).values
            C = np.column_stack([np.ones(len(g))] + [rank_norm(g[c]).values for c in CTRL])
            rx = x - C @ np.linalg.lstsq(C, x, rcond=None)[0]
            ry0 = rank_norm(g.y).values; ry = ry0 - C @ np.linalg.lstsq(C, ry0, rcond=None)[0]
            rr.append(np.corrcoef(rx, ry)[0, 1])
        vals.append(np.mean(rr))
    nul.append(dict(feat=f, null_mean=round(float(np.mean(vals)), 4), null_sd=round(float(np.std(vals)), 4)))
NUL = pd.DataFrame(nul)

# ---- (2) 표본 밖 증분 AUC: 평가 원점 v, 학습 t0 ∈ [2010, v-3] (T3 규칙과 같은 3년 간격), 기준 = 통제 로지스틱
oos = []
for f in FEATS:
    for v in range(2013, 2017):
        tr = D[(D.t0 <= v - 3) & D[f].notna()]; te = D[(D.t0 == v) & D[f].notna()]
        if len(tr) < 500 or len(te) < 200 or tr[f].nunique() < 5:
            continue
        def X(g, withf):
            cols = [zscore(g[c]).values for c in CTRL]
            if withf:
                cols.append(g.groupby('t0')[f].transform(lambda s: rank_norm(s)).values)
            return np.column_stack(cols)
        m0 = LogisticRegression(C=1.0, max_iter=1000).fit(X(tr, False), tr.yb)
        m1 = LogisticRegression(C=1.0, max_iter=1000).fit(X(tr, True), tr.yb)
        a0 = roc_auc_score(te.yb, m0.predict_proba(X(te, False))[:, 1]); a1 = roc_auc_score(te.yb, m1.predict_proba(X(te, True))[:, 1])
        oos.append(dict(feat=f, v=v, n_te=len(te), auc0=round(a0, 4), auc1=round(a1, 4), d_auc=round(a1 - a0, 4), coef=round(float(m1.coef_[0][-1]), 4)))
O = pd.DataFrame(oos); O.to_csv(out('ntis_T3_oos.csv'), index=False, encoding='utf-8-sig')

S = R.groupby('feat').agg(n_orig=('t0', 'size'), t0s=('t0', lambda v: f'{v.min()}-{v.max()}'), rho=('rho', 'mean'), rho_partial=('rho_partial', 'mean'),
                          pos_partial=('rho_partial', lambda v: int((v > 0).sum())), logit=('logit_coef', 'mean'),
                          pos_logit=('logit_coef', lambda v: int((v > 0).sum())), fe_ksic=('fe_ksic_coef', 'mean'),
                          pos_fe=('fe_ksic_coef', lambda v: int((v > 0).sum())), fe_t_min=('fe_ksic_t', 'min'), fe_t_max=('fe_ksic_t', 'max')).reset_index()
S = S.merge(NUL, on='feat', how='left')
if len(O):
    S = S.merge(O.groupby('feat').agg(oos_d_auc=('d_auc', 'mean'), oos_win=('d_auc', lambda v: f'{int((v > 0).sum())}/{len(v)}')).reset_index(), on='feat', how='left')
S.to_csv(out('ntis_T3_summary.csv'), index=False, encoding='utf-8-sig')
pd.set_option('display.width', 250)
print(S.to_string()); print(O.to_string())
