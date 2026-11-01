# -*- coding: utf-8 -*-
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
import sys, time, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from run import *

BASE_NUM = ['age', 'claims', 'rgst_year']
BASE_CAT = ['atype', 'ksic']
PAYSTRUCT = ['paid_ahead', 'paid_year', 'next_annl', 'n_due_win', 'to_expiry', 'exp_in_win', 'tier_jump', 'tier_in_win', 'age_total', 'init_span']
PAYBEH = ['n_renew', 'n_late', 'mean_early', 'min_early', 'max_span', 'last_early', 'last_span', 'last_amt_per_yr', 'yrs_since_pay', 'late_share', 'fee_rel', 'init_fee_rel']
HIST = ['n_recov', 'n_partdel', 'n_techeval', 'n_trial', 'n_prev_lapse', 'n_transfer', 'yrs_since_tr', 'own_corp', 'own_foreign', 'n_owner_now', 'n_lic']
HIST_CAT = ['tr_kind']
PORT = ['port_n', 'port_alive', 'port_new3', 'port_ksic_n', 'a_lapse_rate', 'a_lapse_rate2', 'a_decisions', 'port_share_ksic']
BIB = ['n_appl', 'n_appl_types', 'n_inv', 'n_inv_f', 'n_cofpr', 'n_agent', 'pct', 'n_ipc', 'n_cpc', 'title_len', 'contested', 'grant_lag', 'exam_lag']
BIB_CAT = ['cofpr_natl', 'ipc_sec', 'dspst']
TREND = ['ksic_lapse_1y', 'ksic_lapse_trend', 'ipc_lapse_2y']


def logit_fit(tr, ev, num, cat, Creg=1.0):
    X = lambda d: d[num + cat].assign(claims=np.log1p(d.claims.fillna(d.claims.median())) if 'claims' in num else 0)
    ct = ColumnTransformer([('n', StandardScaler(), num), ('c', OneHotEncoder(handle_unknown='ignore', min_frequency=20), cat)])
    m = make_pipeline(ct, LogisticRegression(C=Creg, max_iter=2000))
    m.fit(X(tr).fillna(0), tr.y)
    return m.predict_proba(X(ev).fillna(0))[:, 1]


def xgb_fit(tr, ev, num, cat, params=None, rounds=600, return_model=False, weight=None, bm_tr=None, bm_ev=None):
    cols = num + cat
    def M(d):
        X = d[cols].copy()
        for c in cat: X[c] = pd.Categorical(X[c], categories=sorted(pd.concat([tr[c], ev[c]]).astype(str).unique()))
        return X
    Xtr, Xev = M(tr.assign(**{c: tr[c].astype(str) for c in cat})), M(ev.assign(**{c: ev[c].astype(str) for c in cat}))
    P = dict(objective='binary:logistic', eval_metric='auc', tree_method='hist', nthread=2, max_depth=6, eta=0.05,
             subsample=0.8, colsample_bytree=0.8, min_child_weight=20, max_cat_to_onehot=8, reg_lambda=1.0)
    if params: P.update(params)
    dtr = xgb.DMatrix(Xtr, tr.y.values, enable_categorical=True, weight=weight, base_margin=bm_tr)
    dev = xgb.DMatrix(Xev, enable_categorical=True, base_margin=bm_ev)
    b = xgb.train(P, dtr, rounds)
    p = b.predict(dev)
    return (p, b) if return_model else p


def run_std(version, change, hyp, fit, kept_rule=None, note='', train_first='2005-01-01', trainLs=None):
    t = time.time()
    trL = trainLs or train_landmarks(C_VAL, train_first)
    tr = get_frame(trL, C_VAL); ev = get_frame(EVAL_L, None)
    p = fit(tr, ev)
    m = metrics(ev.y.values, p)
    b = base_auc()
    save_pred(version, ev, p)
    pl = per_L(ev, p)
    print(version, m, pl, f'train n={len(tr)} pos={tr.y.mean():.4f} Ls={trL[0]}..{trL[-1]} ({len(trL)}) eval n={len(ev)}', round(time.time()-t,1), flush=True)
    return m, b, pl, tr, ev, p

# ---------------------------------------------------------------- versions
def v0():
    m, b, pl, *_ = run_std('v0', '고정 기준선: 로지스틱(경과연수+출원인유형+KSIC+log청구항+등록연도)', '기준선',
                           lambda tr, ev: logit_fit(tr, ev, BASE_NUM, BASE_CAT))
    log_lb('v0', '고정 기준선 로지스틱(경과연수+출원인유형+KSIC+청구항+등록연도)', '기준선', m, None, 'Y', f'L별AUC {pl}; 학습 L 2005-01~2009-07 반기 10개')


def v1():
    m, b, pl, *_ = run_std('v1', 'XGBoost(기준선 공변량 동일)', '비선형·상호작용만으로 개선되는가',
                           lambda tr, ev: xgb_fit(tr, ev, BASE_NUM, BASE_CAT, rounds=300))
    log_lb('v1', 'XGBoost, 기준선 공변량 동일', '비선형·상호작용', m, b, 'Y' if m['auc'] > b else 'N', f'L별AUC {pl}')


def _xv(version, change, hyp, num, cat, note='', params=None, rounds=600, **kw):
    m, b, pl, tr, ev, p = run_std(version, change, hyp, lambda tr, ev: xgb_fit(tr, ev, num, cat, params=params, rounds=rounds), **kw)
    return m, b, pl, tr, ev, p


def v2():
    m, b, pl, *_ = _xv('v2', '+납부구조(다음 기한까지 년, 창 안 납부기한 수, 다음 연차, 요금계단, 만료근접)', '창 안에 납부 결정이 몇 번·어느 요금계단에서 오는지가 위험을 좌우',
                       BASE_NUM + PAYSTRUCT, BASE_CAT)
    log_lb('v2', 'XGB +납부구조(paid_ahead,n_due_win,next_annl,tier,만료근접)', '납부결정 횟수·요금계단', m, b, 'Y', f'L별AUC {pl}')


def v3():
    m, b, pl, *_ = _xv('v3', '+L 이전 납부행동(지연납부·조기납부일수·일괄납부·감면비율)', '과거 납부 지연·감면 여부가 포기 성향을 드러냄',
                       BASE_NUM + PAYSTRUCT + PAYBEH, BASE_CAT)
    log_lb('v3', 'XGB +L 이전 납부행동(지연·조기일수·일괄·감면추정)', '납부행동=의지 신호', m, b, 'Y', f'L별AUC {pl}')


def v4():
    m, b, pl, *_ = _xv('v4', '+이전·실시권·회복·심판·기술평가 이력', '거래·분쟁·회복 이력은 가치 신호',
                       BASE_NUM + PAYSTRUCT + PAYBEH + HIST, BASE_CAT + HIST_CAT)
    log_lb('v4', 'XGB +권리이력(이전·실시권·회복·심판·현재권리자 유형)', '가치·관심 신호', m, b, 'Y', f'L별AUC {pl}')


def v5():
    m, b, pl, *_ = _xv('v5', '+출원인 포트폴리오·L 이전 포기율', '출원인의 과거 포기 성향과 포트폴리오 정리 추세',
                       BASE_NUM + PAYSTRUCT + PAYBEH + HIST + PORT, BASE_CAT + HIST_CAT)
    log_lb('v5', 'XGB +출원인 포트폴리오·L이전 포기율(평활)', '출원인 성향', m, b, 'Y', f'L별AUC {pl}')


def v6():
    m, b, pl, *_ = _xv('v6', '+서지(발명자·해외우선권·PCT·IPC·심사경과)', '해외출원·발명규모는 가치 대리변수',
                       BASE_NUM + PAYSTRUCT + PAYBEH + HIST + PORT + BIB, BASE_CAT + HIST_CAT + BIB_CAT)
    log_lb('v6', 'XGB +서지(발명자수·해외우선권·PCT·IPC수·재심사경과)', '가치 대리', m, b, 'Y', f'L별AUC {pl}')


def v7():
    m, b, pl, *_ = _xv('v7', '+같은 KSIC/IPC 최근 포기추세', '산업·기술 퇴조 추세가 개별 포기를 끌어올림',
                       BASE_NUM + PAYSTRUCT + PAYBEH + HIST + PORT + BIB + TREND, BASE_CAT + HIST_CAT + BIB_CAT)
    log_lb('v7', 'XGB +KSIC 1년 포기율·추세, IPC 2년 포기율', '산업 퇴조 추세', m, b, 'Y', f'L별AUC {pl}')

import hazard as HZ
DEC = ['lead', 'j', 'k', 'age_d', 'tier_d', 'exp_d']


def hazard_run(version, num, cat, trainLs, params=None, rounds=600, fitter=None, ret=False):
    t = time.time()
    tr = pd.concat([HZ.train_decisions(L, C_VAL) for L in trainLs], ignore_index=True)
    ev = get_frame(EVAL_L, None)
    decs = []
    for L in EVAL_L:
        e = ev[ev.L == L].reset_index()
        d = HZ.eval_decisions(e)
        d['row'] = e['index'].values[d.row.values]
        decs.append(d)
    dec = pd.concat(decs, ignore_index=True)
    if fitter is None:
        h = xgb_fit(tr, dec, num, cat, params=params, rounds=rounds)
    else:
        h = fitter(tr, dec)
    p = HZ.combine(dec, h, len(ev))
    m = metrics(ev.y.values, p); pl = per_L(ev, p)
    save_pred(version, ev, p)
    print(version, m, pl, f'train dec n={len(tr)} pos={tr.y.mean():.4f} Ls={trainLs[0]}..{trainLs[-1]} eval n={len(ev)} dec={len(dec)}', round(time.time()-t, 1), flush=True)
    if ret: return m, pl, tr, dec, ev, p, h
    return m, pl


ANNUAL = [f'{y}-01-01' for y in range(2005, 2013)]
F6N = BASE_NUM + PAYSTRUCT + PAYBEH + HIST + PORT + BIB
F6C = BASE_CAT + HIST_CAT + BIB_CAT


def v8():
    m, pl = hazard_run('v8', F6N + DEC, F6C, ANNUAL)
    log_lb('v8', '랜드마크 이산시간 위험모형(기한별 불납 hazard, 학습 L 2005~2012 연1회, 결정 d+450일<=C) + v6 피처', '고연차 결정(8~11년차)을 학습에 포함해 경과연수 외삽 문제 해소',
           m, base_auc(), 'Y', f'L별AUC {pl}')

NOCAL = [c for c in F6N if c != 'rgst_year']


def v9():
    m, pl = hazard_run('v9', NOCAL + DEC, F6C, ANNUAL)
    log_lb('v9', 'v8에서 등록연도(코호트 달력변수) 제거', '트리가 등록연도로 먼저 갈라 고연차×최근코호트 조합을 저위험 잎으로 보내는 외삽 오류 제거',
           m, base_auc(), 'Y', f'L별AUC {pl}')


def v10():
    m, b, pl, *_ = _xv('v10', 'v6(3년 랜드마크 분류)에서 등록연도 제거', '달력변수 외삽 제거 효과를 분류형에서도 확인', NOCAL, F6C)
    log_lb('v10', 'v6 랜드마크 분류형에서 등록연도 제거', '달력변수 외삽 제거', m, b, 'Y', f'L별AUC {pl}')

DRIFT = ['last_amt_per_yr', 'port_n', 'port_alive', 'port_new3', 'a_decisions', 'age_total', 'to_expiry', 'exp_d']


def add_ratio(d):
    d = d.copy()
    d['port_alive_r'] = d.port_alive / d.port_n.replace(0, np.nan)
    d['port_new3_r'] = d.port_new3 / d.port_n.replace(0, np.nan)
    d['log_port'] = np.log1p(d.port_n)
    d['exp_near'] = (d.exp_d if 'exp_d' in d else d.to_expiry) < 3
    d['exp_near'] = d.exp_near.astype(int)
    return d


RATIO = ['port_alive_r', 'port_new3_r', 'exp_near']
F11N = [c for c in NOCAL if c not in DRIFT] + RATIO


def v11():
    fit = lambda tr, ev: xgb_fit(add_ratio(tr), add_ratio(ev), F11N + [c for c in DEC if c not in DRIFT], F6C)
    m, pl = hazard_run('v11', None, None, ANNUAL, fitter=fit)
    log_lb('v11', 'v9에서 달력 드리프트 피처 제거(절대 요금액·포트폴리오 절대규모·만료잔여) -> 비율(생존비율·최근3년 신규비율·만료3년내)로 대체', '시간에 따라 수준이 이동하는 피처가 고연차 hazard를 과소추정하게 만든다',
           m, base_auc(), 'Y', f'L별AUC {pl}')


def v12():
    def fit(tr, ev):
        w = 1.0 + (pd.to_datetime(tr.L).dt.year.values - 2005) / 2.0
        return xgb_fit(add_ratio(tr), add_ratio(ev), F11N + [c for c in DEC if c not in DRIFT], F6C, weight=w)
    m, pl = hazard_run('v12', None, None, ANNUAL, fitter=fit)
    log_lb('v12', 'v11 + 최근 랜드마크 가중(가중치 1+(연도-2005)/2)', '최근 결정일수록 검증기의 위험구조와 닮음', m, base_auc(), 'Y', f'L별AUC {pl}')

_BIZ = {}


def biz_table():
    if 'b' in _BIZ: return _BIZ['b']
    q = pd.read_parquet(os.path.join(T.CACHE, 'biz_query.parquet'))
    ok = set(q[q.status == 'OK'].applicant_cd)
    r = pd.read_parquet(os.path.join(T.CACHE, 'biz_counts.parquet'))
    r = r[r.applicant_cd.isin(ok)].copy()
    r['y'] = pd.to_numeric(r.y, errors='coerce')
    r['typ'] = np.select([r.t == '10', r.t == '20', r.t == '30', r.t.str[0] == '4'], ['pat', 'um', 'des', 'tm'], 'oth')
    b = r.groupby(['applicant_cd', 'typ', 'y']).n.sum().unstack('typ', fill_value=0)
    _BIZ['b'] = (b, ok)
    return _BIZ['b']


def add_biz(d):
    """business_rights(법인 전체 포트폴리오) — 기준일 L의 전년도까지 등록된 권리만 집계(스냅샷 누수 차단)"""
    b, ok = biz_table()
    d = d.copy()
    if 'acd' not in d: d['acd'] = T.load()['pt'].acd.values[d.pid.values]
    Y = pd.to_datetime(d.L).dt.year.values
    out = {}
    for Yv in np.unique(Y):
        bb = b.reset_index()
        def win(a, z):
            w = bb[(bb.y >= a) & (bb.y <= z)].groupby('applicant_cd')[['pat', 'um', 'des', 'tm']].sum()
            return w
        cum = win(1900, Yv - 1); r3 = win(Yv - 3, Yv - 1); p3 = win(Yv - 6, Yv - 4)
        t = pd.DataFrame({'biz_pat_cum': cum.pat, 'biz_tm_cum': cum.tm, 'biz_des_cum': cum.des,
                          'biz_pat_3y': r3.pat, 'biz_tm_3y': r3.tm, 'biz_des_3y': r3.des,
                          'biz_pat_p3': p3.pat, 'biz_tm_p3': p3.tm}).fillna(0)
        out[Yv] = t
    cols = ['biz_pat_cum', 'biz_tm_cum', 'biz_des_cum', 'biz_pat_3y', 'biz_tm_3y', 'biz_des_3y', 'biz_pat_p3', 'biz_tm_p3']
    M = np.full((len(d), len(cols)), np.nan)
    for Yv, t in out.items():
        idx = np.where(Y == Yv)[0]
        a = d.acd.values[idx]
        inok = np.array([x in ok for x in a])
        vals = t.reindex(a)[cols].fillna(0).values
        vals[~inok] = np.nan
        M[idx] = vals
    for i, c in enumerate(cols): d[c] = M[:, i]
    d['biz_pat_growth'] = (d.biz_pat_3y + 1) / (d.biz_pat_p3 + 1)
    d['biz_tm_growth'] = (d.biz_tm_3y + 1) / (d.biz_tm_p3 + 1)
    d['biz_scope_share'] = d.port_n / d.biz_pat_cum.replace(0, np.nan)
    d['biz_tm_per_pat'] = (d.biz_tm_3y + 1) / (d.biz_pat_3y + 1)
    return d


BIZ = ['biz_pat_3y', 'biz_tm_3y', 'biz_des_3y', 'biz_pat_growth', 'biz_tm_growth', 'biz_scope_share', 'biz_tm_per_pat', 'biz_tm_cum']
H12N = F11N + [c for c in DEC if c not in DRIFT]


def recency_w(tr):
    return 1.0 + (pd.to_datetime(tr.L).dt.year.values - 2005) / 2.0


def v13():
    fit = lambda tr, ev: xgb_fit(add_biz(add_ratio(tr)), add_biz(add_ratio(ev)), H12N + BIZ, F6C, weight=recency_w(tr))
    m, pl = hazard_run('v13', None, None, ANNUAL, fitter=fit)
    log_lb('v13', 'v12 + business_rights 법인 전체 포트폴리오(L 전년까지 등록된 특허·상표·디자인 최근3년/이전3년, 성장률, 범위내 비중)', '범위 밖 사업활동(상표·타분야 특허) 정체는 기업 퇴조·포트폴리오 정리의 선행신호',
           m, base_auc(), 'Y', f'L별AUC {pl}')

_RH = {}


def rec_haz_tables(L):
    """기준일 L 직전 2년에 결론 난 납부결정(납부 pd<=L / 소멸등록 cd<=L)의 연차별·출원인유형별 불납률 — L 시점 공개정보"""
    if L in _RH: return _RH[L]
    D = T.load(); pt, pay, lap = D['pt'], D['pay'], D['lap']
    Ld = np.datetime64(L); L2 = T.add_years(np.array([Ld]), -2)[0]
    p = pay[(pay.seq > 0) & (pay.pd > L2) & (pay.pd <= Ld)]
    l = lap[(lap.cd > L2) & (lap.cd <= Ld)]
    kb = lambda k: np.minimum(k, 14)
    r = pd.DataFrame({'k': np.r_[kb(p.s.values), kb(l.lmax.values.astype(int) + 1)], 'y': np.r_[np.zeros(len(p)), np.ones(len(l))],
                      'at': np.r_[pt.atype.values[p.pid.values], pt.atype.values[l.pid.values]]})
    t1 = r.groupby('k').y.agg(['sum', 'size']); g = r.y.mean()
    hk = (t1['sum'] + 50 * g) / (t1['size'] + 50)
    r['kb'] = pd.cut(r.k, [0, 4, 6, 9, 12, 99], labels=False)
    t2 = r.groupby(['at', 'kb']).y.agg(['sum', 'size'])
    hak = (t2['sum'] + 50 * g) / (t2['size'] + 50)
    _RH[L] = (hk, hak, g)
    return _RH[L]


def add_rechaz(d):
    d = d.copy()
    if 'atype' not in d: d['atype'] = T.load()['pt'].atype.values[d.pid.values]
    out = np.full((len(d), 3), np.nan)
    for L in d.L.unique():
        idx = np.where(d.L.values == L)[0]
        hk, hak, g = rec_haz_tables(L)
        k = np.minimum(d.k.values[idx], 14)
        kb = pd.cut(pd.Series(k), [0, 4, 6, 9, 12, 99], labels=False).values
        out[idx, 0] = hk.reindex(k).values
        out[idx, 1] = hak.reindex(list(zip(d.atype.values[idx], kb))).values
        out[idx, 2] = g
    d['rh_k'], d['rh_atk'], d['rh_all'] = out[:, 0], out[:, 1], out[:, 2]
    return d


RH = ['rh_k', 'rh_atk', 'rh_all']


def v14():
    fit = lambda tr, ev: xgb_fit(add_rechaz(add_ratio(tr)), add_rechaz(add_ratio(ev)), H12N + RH, F6C, weight=recency_w(tr))
    m, pl = hazard_run('v14', None, None, ANNUAL, fitter=fit)
    log_lb('v14', 'v12 + 기간효과 추적: L 직전 2년 연차별·(출원인유형×연차대)별 불납률(L 시점 공개분만)', '고연차 불납률이 달력시간에 따라 상승(검증기 k≥9 과소예측) -> 최근 관측 hazard를 피처로 주면 추세를 따라감',
           m, base_auc(), 'Y', f'L별AUC {pl}')


def k_offset(tr, ev, by_atype=True, kmax=12, prior=200):
    """학습 결정에서 연차별(×출원인유형) 경험 hazard -> logit offset. 연차 효과를 트리가 코호트 대리변수로 흡수하지 못하게 명시."""
    at_tr = tr.atype.values if 'atype' in tr else T.load()['pt'].atype.values[tr.pid.values]
    at_ev = ev.atype.values if 'atype' in ev else T.load()['pt'].atype.values[ev.pid.values]
    ktr, kev = np.minimum(tr.k.values, kmax), np.minimum(ev.k.values, kmax)
    g = tr.y.mean()
    hk = tr.assign(kk=ktr).groupby('kk').y.agg(['sum', 'size'])
    hk = (hk['sum'] + prior * g) / (hk['size'] + prior)
    def lookup(k, at):
        base = hk.reindex(k).fillna(g).values
        if not by_atype: return base
        t = tr.assign(kk=ktr, at=at_tr).groupby(['at', 'kk']).y.agg(['sum', 'size'])
        hh = (t['sum'] + prior * hk.reindex(t.index.get_level_values(1)).values) / (t['size'] + prior)
        v = hh.reindex(list(zip(at, k))).values
        return np.where(np.isnan(v), base, v)
    lg = lambda p: np.log(p / (1 - p))
    return lg(lookup(ktr, at_tr)), lg(lookup(kev, at_ev))


def v15():
    def fit(tr, ev):
        a, b = k_offset(tr, ev)
        return xgb_fit(add_ratio(tr), add_ratio(ev), H12N, F6C, weight=recency_w(tr), bm_tr=a, bm_ev=b)
    m, pl = hazard_run('v15', None, None, ANNUAL, fitter=fit)
    log_lb('v15', 'v12 + 연차(×출원인유형) 경험 hazard를 base_margin(오프셋)으로 고정, 트리는 상대위험만 학습', '고연차 hazard를 트리가 코호트 대리변수(심사기간 등)로 학습해 검증기 고연차를 과소예측 -> 연차효과를 오프셋으로 명시',
           m, base_auc(), 'Y', f'L별AUC {pl}')

import soc as SO


def v16():
    fit = lambda tr, ev: xgb_fit(SO.add_soc(add_ratio(tr)), SO.add_soc(add_ratio(ev)), H12N + SO.SOC, F6C, weight=recency_w(tr))
    m, pl = hazard_run('v16', None, None, ANNUAL, fitter=fit)
    log_lb('v16', 'v12 + 관계 피처(같은 발명자팀·출원인×출원연도 형제·출원인×KSIC·출원인×IPC의 L 이전 불납률·생존비율, 출원인 최근 6개월 포기 수)', '같은 기술라인(발명자팀·형제특허)이 먼저 버려지면 이 특허도 버려진다',
           m, base_auc(), 'Y', f'L별AUC {pl}')


LM_N = F11N


def v17():
    m, b, pl, *_ = run_std('v17', '', '', lambda tr, ev: xgb_fit(SO.add_soc(add_ratio(tr)), SO.add_soc(add_ratio(ev)), [c for c in LM_N if c != 'exp_near'] + ['to_expiry'] + SO.SOC, F6C))
    log_lb('v17', '3년 랜드마크 분류형(v10) + 드리프트 제거·비율 + 관계 피처', '분류형도 관계 피처로 개선되는지(앙상블 재료)', m, b, 'Y', f'L별AUC {pl}')

F17N = [c for c in LM_N if c != 'exp_near'] + ['to_expiry'] + SO.SOC


def lm_w(tr):
    return 1.0 + (pd.to_datetime(tr.L).dt.year.values - 2005) / 2.0


def v18():
    m, b, pl, *_ = run_std('v18', '', '', lambda tr, ev: xgb_fit(SO.add_soc(add_ratio(tr)), SO.add_soc(add_ratio(ev)), F17N, F6C, weight=lm_w(tr)))
    log_lb('v18', 'v17 + 최근 랜드마크 가중(1+(연도-2005)/2)', '최근 학습 기준일이 검증기와 더 닮음(위험모형 v12에서 +0.01)', m, b, 'Y', f'L별AUC {pl}')


def v19():
    P = dict(max_depth=8, eta=0.03, min_child_weight=50, colsample_bytree=0.6, subsample=0.8, reg_lambda=5.0)
    m, b, pl, *_ = run_std('v19', '', '', lambda tr, ev: xgb_fit(SO.add_soc(add_ratio(tr)), SO.add_soc(add_ratio(ev)), F17N, F6C, params=P, rounds=1000))
    log_lb('v19', 'v17 하이퍼파라미터 변경(depth8, eta0.03, 1000라운드, mcw50, colsample0.6, lambda5)', '더 깊고 규제 강한 트리로 상호작용 포착', m, b, 'Y', f'L별AUC {pl}')

P19 = dict(max_depth=8, eta=0.03, min_child_weight=50, colsample_bytree=0.6, subsample=0.8, reg_lambda=5.0)


def v20():
    m, b, pl, *_ = run_std('v20', '', '', lambda tr, ev: xgb_fit(SO.add_soc(add_ratio(tr)), SO.add_soc(add_ratio(ev)), F17N, F6C, params=P19, rounds=1000, weight=lm_w(tr)))
    log_lb('v20', 'v18 가중 + v19 하이퍼파라미터 결합', '두 개선이 독립적이면 합쳐짐', m, b, 'Y', f'L별AUC {pl}')

import textfeat as TX


def v21():
    def fit(tr, ev):
        t0 = time.time()
        a, b = TX.text_scores(tr, ev)
        from sklearn.metrics import roc_auc_score
        print('text-only AUC train-OOF', round(roc_auc_score(tr.y, a), 4), 'eval', round(roc_auc_score(ev.y, b), 4), round(time.time() - t0, 1), flush=True)
        pd.DataFrame({'pid': ev.pid.values, 'L': ev.L.values, 'txt': b}).to_parquet(os.path.join(HERE, 'cache', 'txt_eval_val.parquet'))
        pd.DataFrame({'pid': tr.pid.values, 'L': tr.L.values, 'txt': a}).to_parquet(os.path.join(HERE, 'cache', 'txt_train_val.parquet'))
        tr = tr.assign(txt=a); ev = ev.assign(txt=b)
        return xgb_fit(SO.add_soc(add_ratio(tr)), SO.add_soc(add_ratio(ev)), F17N + ['txt'], F6C, weight=lm_w(tr))
    m, b, pl, *_ = run_std('v21', '', '', fit)
    log_lb('v21', 'v18 + 발명명칭 텍스트 점수(TF-IDF 단어1-2+글자2-3gram, SGD 로지스틱, 학습행은 출원인 5겹 OOF)', '명칭에 드러난 기술유형(주변·개량기술)이 포기 성향과 관련', m, b, 'Y', f'L별AUC {pl}')


def v22():
    def w(tr): return np.exp((pd.to_datetime(tr.L).dt.year.values - 2009) / 1.5)
    m, b, pl, *_ = run_std('v22', '', '', lambda tr, ev: xgb_fit(SO.add_soc(add_ratio(tr)), SO.add_soc(add_ratio(ev)), F17N, F6C, weight=w(tr)))
    log_lb('v22', 'v18의 선형 가중 -> 지수 가중 exp((연도-2009)/1.5) (최근 기준일 집중)', '더 강한 최근성 가중이 추가로 도움', m, b, 'Y', f'L별AUC {pl}')

_FAM = {}
FAM = ['fam_n', 'fam_cc', 'fam_us', 'fam_ep', 'fam_jp', 'fam_cn', 'fam_wo']


def add_fam(d):
    if 'f' not in _FAM:
        fm = pd.read_parquet(os.path.join(T.CACHE, 'kr_family.parquet')).drop_duplicates('applNo').set_index('applNo')
        pt = T.load()['pt']
        _FAM['f'] = fm.reindex(pt.applNo.values)[FAM].values
    d = d.copy()
    M = _FAM['f'][d.pid.values]
    for i, c in enumerate(FAM): d[c] = M[:, i]
    return d


def v23():
    m, b, pl, *_ = run_std('v23', '', '', lambda tr, ev: xgb_fit(add_fam(SO.add_soc(add_ratio(tr))), add_fam(SO.add_soc(add_ratio(ev))), F17N + FAM, F6C, weight=lm_w(tr)))
    log_lb('v23', 'v18 + 제공 융합데이터 패밀리(해외 패밀리 수·국가 수·US/EP/JP/CN/WO 여부)', '해외에 패밀리를 둔 특허는 가치가 높아 덜 버려짐(특히 외국법인 세그먼트 보강)', m, b, 'Y', f'L별AUC {pl}')

BIZR = ['biz_pat_growth', 'biz_tm_growth', 'biz_scope_share', 'biz_tm_per_pat']
F23N = F17N + FAM


def v24():
    m, b, pl, *_ = run_std('v24', '', '', lambda tr, ev: xgb_fit(add_biz(add_fam(SO.add_soc(add_ratio(tr)))), add_biz(add_fam(SO.add_soc(add_ratio(ev)))), F23N + BIZR, F6C, weight=lm_w(tr)))
    log_lb('v24', 'v23 + business_rights 비율형만(특허·상표 3년 성장률, 범위내 비중, 상표/특허 비) — 절대건수 제외', 'v13 실패는 절대건수 드리프트 탓 -> 비율형이면 법인 사업활동 신호가 살아남음', m, b, 'Y', f'L별AUC {pl}')


def v25():
    fit = lambda tr, ev: xgb_fit(add_fam(SO.add_soc(add_ratio(tr))), add_fam(SO.add_soc(add_ratio(ev))), H12N + SO.SOC + FAM, F6C, weight=recency_w(tr))
    m, pl = hazard_run('v25', None, None, ANNUAL, fitter=fit)
    log_lb('v25', '위험모형(v16) + 패밀리 피처', '앙상블 재료: 위험모형도 패밀리로 개선', m, base_auc(), 'Y', f'L별AUC {pl}')


def ens_log(version, parts, weights, change, hyp, kept='Y'):
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    ps = [pd.read_parquet(os.path.join(HERE, 'preds', f'pred_{v}.parquet')) for v in parts]
    z = sum(w * lg(p.p.values) for w, p in zip(weights, ps))
    p = 1 / (1 + np.exp(-z))
    ev = ps[0].copy(); ev['p'] = p
    m = metrics(ev.y.values, p)
    pl = {L: round(roc_auc_score(ev.y[ev.L == L], p[(ev.L == L).values]), 4) for L in sorted(ev.L.unique())}
    ev.to_parquet(os.path.join(HERE, 'preds', f'pred_{version}.parquet'), index=False)
    print(version, m, pl)
    log_lb(version, change, hyp, m, base_auc(), kept, f'L별AUC {pl}')


def v26():
    ens_log('v26', ['v23', 'v25'], [0.5, 0.5], '앙상블: 3년 분류형 v23 + 이산시간 위험모형 v25 로짓 평균(가중 0.5/0.5 사전고정)',
            '두 모형족의 오차가 다름(분류형은 고연차, 위험모형은 저연차에 강함)')


def logit_full(extra_fn=None, extra_cols=()):
    from sklearn.impute import SimpleImputer
    def fit(tr, ev):
        A = add_fam(SO.add_soc(add_ratio(tr))); B = add_fam(SO.add_soc(add_ratio(ev)))
        if extra_fn: A, B = extra_fn(A), extra_fn(B)
        num = F23N + list(extra_cols)
        def prep(d):
            X = d[num].astype(float).copy()
            for c in ['claims', 'n_renew', 'port_ksic_n', 'n_inv', 'n_ipc', 'n_cpc', 'title_len', 'sib_n', 'aks_n', 'aipc_n', 'inv_n', 'a_lap6m', 'fam_n', 'n_transfer', 'n_lic']:
                if c in X: X[c] = np.log1p(X[c].clip(lower=0))
            for c in num:
                X[c + '_na'] = X[c].isna().astype(float) if X[c].isna().mean() > 0.01 else 0.0
            return X
        XA, XB = prep(A), prep(B)
        XB = XB[XA.columns]
        lo, hi = XA.quantile(0.005), XA.quantile(0.995)
        XA = XA.clip(lo, hi, axis=1); XB = XB.clip(lo, hi, axis=1)
        ct = ColumnTransformer([('n', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), list(XA.columns)),
                                ('c', OneHotEncoder(handle_unknown='ignore', min_frequency=50), F6C)])
        m = make_pipeline(ct, LogisticRegression(C=0.5, max_iter=3000))
        m.fit(pd.concat([XA, A[F6C].astype(str)], axis=1), A.y, logisticregression__sample_weight=lm_w(A))
        return m.predict_proba(pd.concat([XB, B[F6C].astype(str)], axis=1))[:, 1]
    return fit


def v27():
    m, b, pl, *_ = run_std('v27', '', '', logit_full())
    log_lb('v27', '선형 모형족: v23 피처 전체의 로지스틱(로그변환·윈저화·결측지시자·가중)', '트리 없이도 신호가 재현되는지 + 앙상블 다양성', m, b, 'Y', f'L별AUC {pl}')


def v28():
    ens_log('v28', ['v23', 'v25', 'v27'], [1/3, 1/3, 1/3], '앙상블 3개 모형족 동일가중 로짓평균: 분류 XGB(v23)+위험 XGB(v25)+로지스틱(v27)',
            '선형모형은 트리가 못하는 외삽(고연차·최근 코호트)을 선형으로 이어줘 보완')


def v29():
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import SplineTransformer
    SPL = ['age', 'paid_ahead', 'paid_year', 'to_expiry', 'a_lapse_rate', 'a_lapse_rate2', 'sib_alive', 'aks_alive', 'inv_alive', 'aks_rate', 'port_alive_r', 'port_new3_r',
           'last_early', 'mean_early', 'fee_rel', 'grant_lag']
    def fit(tr, ev):
        A = add_fam(SO.add_soc(add_ratio(tr))); B = add_fam(SO.add_soc(add_ratio(ev)))
        num = F23N
        def prep(d):
            X = d[num].astype(float).copy()
            for c in ['claims', 'n_renew', 'port_ksic_n', 'n_inv', 'n_ipc', 'n_cpc', 'title_len', 'sib_n', 'aks_n', 'aipc_n', 'inv_n', 'a_lap6m', 'fam_n', 'n_transfer', 'n_lic']:
                if c in X: X[c] = np.log1p(X[c].clip(lower=0))
            for c in num:
                X[c + '_na'] = X[c].isna().astype(float) if X[c].isna().mean() > 0.01 else 0.0
            return X
        XA, XB = prep(A), prep(B); XB = XB[XA.columns]
        lo, hi = XA.quantile(0.005), XA.quantile(0.995)
        XA = XA.clip(lo, hi, axis=1); XB = XB.clip(lo, hi, axis=1)
        rest = [c for c in XA.columns if c not in SPL]
        ct = ColumnTransformer([('s', make_pipeline(SimpleImputer(strategy='median'), SplineTransformer(n_knots=5, degree=3, extrapolation='linear'), StandardScaler()), SPL),
                                ('n', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), rest),
                                ('c', OneHotEncoder(handle_unknown='ignore', min_frequency=50), F6C)])
        m = make_pipeline(ct, LogisticRegression(C=0.5, max_iter=3000))
        m.fit(pd.concat([XA, A[F6C].astype(str)], axis=1), A.y, logisticregression__sample_weight=lm_w(A))
        return m.predict_proba(pd.concat([XB, B[F6C].astype(str)], axis=1))[:, 1]
    m, b, pl, *_ = run_std('v29', '', '', fit)
    log_lb('v29', 'v27 로지스틱 + 핵심 16개 연속변수 3차 스플라인(5매듭, 경계 밖 선형외삽)', '비선형을 넣되 외삽은 선형으로 유지해 트리보다 안정적인 외삽', m, b, 'Y', f'L별AUC {pl}')


def v30():
    m, b, pl, *_ = run_std('v30', '', '', lambda tr, ev: xgb_fit(SO.add_ak(add_fam(SO.add_soc(add_ratio(tr))), 'next_annl'), SO.add_ak(add_fam(SO.add_soc(add_ratio(ev))), 'next_annl'), F23N + ['ak_rate', 'ak_n'], F6C, weight=lm_w(tr)))
    log_lb('v30', 'v23 + 출원인×연차대(다음 납부연차 기준) L 이전 불납률', '기업마다 "몇 년차에 정리한다"는 관행(연차별 정책)이 있다', m, b, 'Y', f'L별AUC {pl}')


def v31():
    fit = lambda tr, ev: xgb_fit(SO.add_ak(add_fam(SO.add_soc(add_ratio(tr))), 'k'), SO.add_ak(add_fam(SO.add_soc(add_ratio(ev))), 'k'), H12N + SO.SOC + FAM + ['ak_rate', 'ak_n'], F6C, weight=recency_w(tr))
    m, pl = hazard_run('v31', None, None, ANNUAL, fitter=fit)
    log_lb('v31', '위험모형 v25 + 출원인×연차대(결정 연차 기준) 불납률', '연차별 정책 신호는 결정 단위 위험모형에 더 직접적', m, base_auc(), 'Y', f'L별AUC {pl}')


def v32():
    m, b, pl, *_ = run_std('v32', '', '', logit_full(lambda d: SO.add_ak(d, 'next_annl'), ['ak_rate', 'ak_n']))
    log_lb('v32', '로지스틱 v27 + 출원인×연차대 불납률', '선형 모형족도 연차정책 신호로 개선', m, b, 'Y', f'L별AUC {pl}')


def v33():
    ens_log('v33', ['v30', 'v25', 'v32'], [1/3, 1/3, 1/3], '앙상블 동일가중: 분류 XGB v30 + 위험 XGB v25 + 로지스틱 v32', '각 모형족의 최선 구성으로 교체')


def v34():
    ens_log('v34', ['v30', 'v25', 'v27'], [1/3, 1/3, 1/3], '앙상블 동일가중: 분류 XGB v30 + 위험 XGB v25 + 로지스틱 v27', 'v33에서 로지스틱을 더 나은 v27로 되돌림')


def label_trunc(L, pids, C, lag=450):
    """창을 (L, min(L+3y, C-lag)] 로 자른 학습 레이블(C 시점 공개정보만). 반환 y, 창 길이(년)"""
    fn = os.path.join(T.CACHE, f'labT_{L.replace("-", "")}_{C.replace("-", "")}.npy')
    D = T.load(); pt, lap = D['pt'], D['lap']
    Ld, Cd = np.datetime64(L), np.datetime64(C)
    end = min(T.add_years(np.array([Ld]), 3)[0], Cd - np.timedelta64(lag, 'D'))
    h = T.yrs(np.array([end]), np.array([Ld]))[0]
    if os.path.exists(fn):
        y = np.load(fn)
        if len(y) == len(pids): return y, h
    e = lap[(lap.E > Ld) & (lap.E <= end) & (lap.cd <= Cd)].pid.unique()
    pthC, _ = T.paid_through(C, asof=C)
    ml = D['ev']['malso']; mz = ml[(ml.cd > Ld) & (ml.cd <= end)].pid.unique()
    y = np.full(len(pids), np.nan)
    zero = (pthC[pids] > end) | (pt.exp.values[pids] <= end) | np.isin(pids, mz)
    y[zero] = 0; y[np.isin(pids, e)] = 1
    np.save(fn, y)
    return y, h


def v35():
    t = time.time()
    trL = train_landmarks(C_VAL) + ['2010-01-01', '2010-07-01', '2011-01-01', '2011-07-01', '2012-01-01']
    fs = []
    for L in trL:
        f = T.features(L)
        y, h = label_trunc(L, f.pid.values, C_VAL)
        f = f.assign(y=y, h_win=h, acd=T.load()['pt'].acd.values[f.pid.values])
        fs.append(f[np.isfinite(f.y)])
    tr = pd.concat(fs, ignore_index=True)
    ev = get_frame(EVAL_L, None).assign(h_win=3.0)
    print('trunc label check: pos by h', tr.groupby('h_win').y.agg(['size', 'mean']).round(3).to_dict())
    A = add_fam(SO.add_soc(add_ratio(tr))); B = add_fam(SO.add_soc(add_ratio(ev)))
    p = xgb_fit(A, B, F23N + ['h_win'], F6C, weight=lm_w(tr))
    m = metrics(ev.y.values, p); pl = per_L(ev, p); save_pred('v35', ev, p)
    print('v35', m, pl, len(tr), round(time.time() - t, 1))
    log_lb('v35', 'v23 + 최근 기준일(2010-01~2012-01) 부분관측 레이블 추가(창을 C-450일에서 자르고 창길이 h를 피처로; 예측 시 h=3)', '최근 기준일의 고연차 특허 정보를 분류형에도 투입', m, base_auc(), 'Y', f'L별AUC {pl}')


def hgb_fit(tr, ev, num, cat, weight=None):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import OrdinalEncoder
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, encoded_missing_value=-1, max_categories=60)
    ctr = oe.fit_transform(tr[cat].astype(str)); cev = oe.transform(ev[cat].astype(str))
    ctr[ctr < 0] = np.nan; cev[cev < 0] = np.nan
    Xtr = np.c_[tr[num].astype(float).values, ctr]; Xev = np.c_[ev[num].astype(float).values, cev]
    m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=500, max_leaf_nodes=48, min_samples_leaf=200, l2_regularization=1.0,
                                       categorical_features=list(range(len(num), len(num) + len(cat))), early_stopping=False, random_state=0)
    m.fit(Xtr, tr.y.values, sample_weight=weight)
    return m.predict_proba(Xev)[:, 1]


def v36():
    import os as _o
    _o.environ['OMP_NUM_THREADS'] = '2'
    m, b, pl, *_ = run_std('v36', '', '', lambda tr, ev: hgb_fit(SO.add_ak(add_fam(SO.add_soc(add_ratio(tr))), 'next_annl'), SO.add_ak(add_fam(SO.add_soc(add_ratio(ev))), 'next_annl'), F23N + ['ak_rate', 'ak_n'], F6C, weight=lm_w(tr)))
    log_lb('v36', '다른 부스팅 구현(sklearn HistGradientBoosting, 잎 48, 500회) — v30 피처', '구현·분할 방식이 다른 트리 모형으로 앙상블 다양성 확보', m, b, 'Y', f'L별AUC {pl}')


def v37():
    ens_log('v37', ['v30', 'v25', 'v27', 'v36'], [0.25] * 4, '앙상블 4개 동일가중 로짓평균: XGB 분류 v30 + XGB 위험 v25 + 로지스틱 v27 + HGB 분류 v36', '구현이 다른 부스팅을 추가해 분산 감소')


if __name__ == '__main__':
    for v in sys.argv[1:]:
        globals()[v]()
