# -*- coding: utf-8 -*-
"""C14-T3 공용: 동결 T1 코드 import(읽기만) + 범위밖 의존 피처 재계산 + 섭동/아날로그 행 구성.
재계산 줄의 출처: T1_산업대기실/final_run.py cut_features() 145–203행(동결), feats.agg_feats·partner_feats(동결 함수 그대로 호출), lib.build_panel(동결 그대로 호출)."""
import sys, os, json, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard                                           # 쓰기 가드(이 폴더 밖 쓰기 금지)
HERE = guard.HERE
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
T1 = B/'분석/모델경쟁_20260927/T1_산업대기실'; R3 = T1/'R3'; R2 = T1/'R2'; CACHE = T1/'cache'; EXTD = B/'분석/모델경쟁_20260927/_shared/ext'
sys.path.insert(0, str(R3)); os.chdir(str(T1))
import numpy as np, pandas as pd
import final_run_r3 as F                               # 동결 실행기(환경변수 T1_STRICT=1, T1_VIS=rgst 설정은 final_run import 시)
import r3, lib
from r3 import fr
from feats import agg_feats, partner_feats

ORIGINS = [2019, 2008, 2009, 2010, 2011, 2012]
def cfg(c):
    if c == 2019: return dict(eval_cut=2019, train_cuts=(2012, 2013, 2014, 2015), cap='20191231')
    return dict(eval_cut=c, train_cuts=tuple(range(c - 7, c - 3)), cap=f'{c}1231')
def fcp(c): return R3/'cache_final'/'o2019' if c == 2019 else R3/'cache'/f'o{c}'      # train_origin 의 FCp(prep 위치)
def psrc(c): return R3/'cache_final'/'o2019' if c == 2019 else R2/'cache'/f'o{c}'     # 분류기 확률 위치(build_r3.py: src=fc_r2)
REPL_OP = ['lsize', 'lout', 'inshare', 't_mean', 't_lsum', 't_max', 't_hi', 't_n20', 't_top3', 't_rank', 't_ent', 't_rw', 't_rec3', 't_mom', 't_lrec_n',
           't_grow', 't_nbr', 'ot_mean', 'ot_max', 'ot_top3', 'ot_hi', 'ot_rw', 'ot_nbr', 'oin50', 'p_n', 'p_k', 'p_share', 'tsib3', 'tsib2',
           'gap_last', 'n_last1', 'r_n1', 'r_n2', 'r_in2', 't_rrw', 't_rrec2', 't_rrec2_max', 'now1', 'now3', 'now_grow', 'now_in3', 'vis_frac']
OT6 = ['ot_mean', 'ot_max', 'ot_top3', 'ot_hi', 'ot_rw', 'ot_nbr']
OT7 = OT6 + ['oin50']
TI4 = ['ti_mean', 'ti_max', 'ti_n20', 'ti_lsum']


def panel(t, c):
    """동결 build_panel 그대로(h=0: 결과창 없음 → 레이블 미계산). cap: 평가 패널 None, 학습 패널 cfg cap."""
    cf = cfg(c); cap = None if t == c else cf['cap']
    old = lib.HOLDOUT_GUARD_MAX_ENTRY_YEAR; lib.HOLDOUT_GUARD_MAX_ENTRY_YEAR = max(old, t)
    try: d, ctx = lib.build_panel(t, 0, cap=cap)
    finally: lib.HOLDOUT_GUARD_MAX_ENTRY_YEAR = old
    d['cut'] = t
    return d, ctx


def op_feats(d0, ctx, P, P51, PO, cdf):
    """final_run.cut_features 의 범위밖(op) 의존 열만 다시 계산(같은 식·같은 dtype). ctx['op']만 바뀐 ctx를 넣는다."""
    cut = ctx['cut']; D = lib.load()
    d = d0[['fidx', 'k', 'firm']].copy()
    f, k = d.fidx.values, d.k.values; op = ctx['op']; fi = ctx['fi']; nF = len(ctx['firms']); rows = fi[op.applicant_cd].values
    # lib.build_panel 102–108행: nout·lsize·lout·inshare (nin=M.sum(1) 엄격 가시 범위안)
    nout = np.bincount(fi[op.applicant_cd].values, minlength=nF).astype(np.float32); nin = ctx['nin']
    d['lsize'] = np.log1p(nin[f] + nout[f]); d['lout'] = np.log1p(nout[f]); d['inshare'] = nin[f] / np.maximum(nin[f] + nout[f], 1)
    # cut_features 144행: agg_feats(d, c, P)
    d = agg_feats(d, ctx, P)
    # 147–151행 open-set 51
    o = agg_feats(d[['fidx', 'k']].copy(), ctx, P51)
    for cc in ['t_mean', 't_max', 't_top3', 't_hi', 't_rw', 't_nbr']: d['o' + cc] = o[cc].values
    s50 = np.bincount(rows, weights=1 - PO[op.row.values], minlength=nF) / np.maximum(np.bincount(rows, minlength=nF), 1)
    d['oin50'] = s50[f]
    # 153–154행 파트너
    x = partner_feats(d[['fidx', 'k']].copy(), ctx)
    for cc in ['p_n', 'p_k', 'p_share']: d[cc] = x[cc].values
    # 165–170행 형제(tsib만 op 의존: TM=t_mean)
    K = D['K']; M = ctx['M']; TM = np.zeros_like(M); TM[f, k] = d.t_mean.values
    for L, nm in [(3, 'sib3'), (2, 'sib2')]:
        G = np.array([[1.0 if (K[i][:L] == K[j][:L] and i != j) else 0.0 for j in range(50)] for i in range(50)], np.float32)
        d['t' + nm] = (TM @ G)[f, k]
    # 177–181행 활동 최근성
    ip = ctx['ip'].drop_duplicates(['applicant_cd', 'rgst_no']); r = fi[ip.applicant_cd].values
    allp = pd.concat([op[['applicant_cd', 'ay', 'rgst_no']], ctx['ip'][['applicant_cd', 'ay', 'rgst_no']]]).drop_duplicates(['applicant_cd', 'rgst_no'])
    last = allp.groupby('applicant_cd').ay.max(); n1 = allp[allp.ay == cut].groupby('applicant_cd').size()
    d['gap_last'] = cut - d.firm.map(last).values; d['n_last1'] = np.log1p(d.firm.map(n1).fillna(0).values)
    # 183–193행 등록일 기준 최근성
    ry = pd.to_numeric(op.rgst_date.str[:4]).values; Po = P[op.row.values]
    ryi = pd.to_numeric(ip.rgst_date.str[:4]).values
    r_n1 = np.bincount(rows[ry >= cut], minlength=nF) + np.bincount(r[ryi >= cut], minlength=nF)
    r_n2 = np.bincount(rows[ry >= cut - 1], minlength=nF) + np.bincount(r[ryi >= cut - 1], minlength=nF)
    in2 = np.bincount(r[ryi >= cut - 1], minlength=nF)
    w = np.power(0.5, (cut - ry) / 2.0).astype(np.float32)
    TW = np.zeros((nF, 50), np.float32); np.add.at(TW, rows, Po * w[:, None]); W = np.bincount(rows, weights=w, minlength=nF)
    m2 = ry >= cut - 1
    T2 = np.zeros((nF, 50), np.float32); np.add.at(T2, rows[m2], Po[m2]); N2r = np.bincount(rows[m2], minlength=nF)
    MX2 = np.zeros((nF, 50), np.float32); np.maximum.at(MX2, rows[m2], Po[m2])
    d['r_n1'] = np.log1p(r_n1[f]); d['r_n2'] = np.log1p(r_n2[f]); d['r_in2'] = np.log1p(in2[f]); d['t_rrw'] = (TW / np.maximum(W, 1e-9)[:, None])[f, k]
    d['t_rrec2'] = (T2 / np.maximum(N2r, 1)[:, None])[f, k]; d['t_rrec2_max'] = MX2[f, k]
    # 195–206행 나우캐스팅
    tp = D['fm'].set_index('applicant_cd').tp; ce = pd.Timestamp(f'{cut}-12-31')
    al = pd.concat([op[['applicant_cd', 'rgst_no', 'appl_date', 'ay']].assign(ins=0), ctx['ip'][['applicant_cd', 'rgst_no', 'appl_date', 'ay']].assign(ins=1)]).drop_duplicates(['applicant_cd', 'rgst_no'])
    days = (ce - pd.to_datetime(al.appl_date.astype(str), format='%Y%m%d', errors='coerce')).dt.days.fillna(3650).values
    frn = al.applicant_cd.map(tp).eq('외국법인').values
    pi = np.where(frn, np.searchsorted(cdf[True], days, side='right') / len(cdf[True]), np.searchsorted(cdf[False], days, side='right') / len(cdf[False]))
    al['w'] = 1 / np.maximum(pi, 0.05); al['pi'] = pi
    g_n1 = al[al.ay == cut].groupby('applicant_cd').w.sum(); g_n3 = al[al.ay > cut - 3].groupby('applicant_cd').w.sum()
    g_p3 = al[(al.ay > cut - 6) & (al.ay <= cut - 3)].groupby('applicant_cd').w.sum(); g_i3 = al[(al.ay > cut - 3) & (al.ins == 1)].groupby('applicant_cd').w.sum()
    fm_ = d.firm
    d['now1'] = np.log1p(fm_.map(g_n1).fillna(0).values); d['now3'] = np.log1p(fm_.map(g_n3).fillna(0).values)
    d['now_in3'] = np.log1p(fm_.map(g_i3).fillna(0).values); d['vis_frac'] = fm_.map(al.groupby('applicant_cd').pi.mean()).fillna(0).values
    d['now_grow'] = d.now3 - np.log1p(fm_.map(g_p3).fillna(0).values)
    return d


def ti_feats(d0, ctx, rows_df, Pins, ins_idx):
    """cut_features 172–176행 범위안 제목 2차 산업(ti_*), 행 집합을 바꿔 넣을 수 있게."""
    f, k = d0.fidx.values, d0.k.values; fi = ctx['fi']; nF = len(ctx['firms'])
    pos = pd.Series(np.arange(len(ins_idx)), index=ins_idx.values)
    ip = rows_df.drop_duplicates(['applicant_cd', 'rgst_no']); r = fi[ip.applicant_cd].values; S = Pins[pos[ip.rgst_no].values].astype(np.float32)
    T = np.zeros((nF, 50), np.float32); np.add.at(T, r, S); nn = np.bincount(r, minlength=nF)
    MX = np.zeros((nF, 50), np.float32); np.maximum.at(MX, r, S); N2 = np.zeros((nF, 50), np.float32); np.add.at(N2, r, (S > 0.2).astype(np.float32))
    return pd.DataFrame({'ti_mean': (T / np.maximum(nn, 1)[:, None])[f, k], 'ti_max': MX[f, k], 'ti_n20': np.log1p(N2[f, k]), 'ti_lsum': np.log1p(T[f, k])})


def an_feats(d0, ctx, rows_df, Pins, ins_idx):
    """ot_* 와 같은 agg_feats 정의를 범위안 행(rows_df: applicant_cd, rgst_no, ay)과 범위안 교차적합 확률 Pins로."""
    pos = pd.Series(np.arange(len(ins_idx)), index=ins_idx.values)
    x = rows_df.drop_duplicates(['applicant_cd', 'rgst_no']).copy(); x['row'] = pos[x.rgst_no.values].values
    c2 = dict(ctx); c2['op'] = x[['applicant_cd', 'row', 'ay']]
    o = agg_feats(d0[['fidx', 'k']].copy(), c2, Pins, pre='an')
    return pd.DataFrame({'ot_' + s: o['an_' + s].values for s in ('mean', 'max', 'top3', 'hi', 'rw', 'nbr')})


def u01(keys, salt):
    """(법인, 등록번호, salt) 결정적 균등난수 — 같은 권리는 모든 원점·패널에서 같은 값."""
    h = pd.util.hash_pandas_object(pd.Series(keys + '|' + str(salt)), index=False).values
    return (h >> np.uint64(11)).astype(np.float64) / float(1 << 53)


_SIZE = None
def size_bin(firms):
    global _SIZE
    if _SIZE is None:
        P = pd.read_parquet(EXTD/'firm_nps_firm_panel.parquet', columns=['applicantCd', 'snap', 'n_emp']); _SIZE = P[P.snap == '2019-12'].drop_duplicates('applicantCd').set_index('applicantCd').n_emp
    e = pd.Series(firms).map(_SIZE).values
    out = np.where(pd.isna(e), '미연결', np.where(e <= 9, '<10', np.where(e <= 49, '10–49', np.where(e <= 299, '50–299', '300+'))))
    return out
def ygrp(ry):
    return np.where(ry <= 2004, '2000–04', np.where(ry <= 2009, '2005–09', np.where(ry <= 2014, '2010–14', np.where(ry <= 2019, '2015–19', '2020+'))))
def cell_rate(op):
    rates = json.load(open(HERE/'cache/s3_rates.json', encoding='utf-8'))
    ry = pd.to_numeric(op.rgst_date.str[:4]).values
    key = pd.Series(ygrp(ry)).str.cat(pd.Series(size_bin(op.applicant_cd.values)), sep='|').values
    d = pd.Series(key).map(rates).values.astype(np.float64)
    assert not np.isnan(d).any(), '셀 누락률 없음'
    return d


def peak_gb():
    try: return r3.peak_gb()
    except Exception: return float('nan')
