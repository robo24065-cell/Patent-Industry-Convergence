# 2단계(감사 후) 공용 라이브러리: 감사 수정(F1~F3)을 반영한 2단 계층모형.
#   p = owner_hat(L, 권리자) × (p_pat / 권리자 내 평균 p_pat)^beta
# 사용: import v2lib; ap = v2lib.fit_apply(cfg, train_years, apply_years)
import json, numpy as np, pandas as pd
import xgboost as xgb
import run
from common import *
from evalkit import *

# ---- 감사 수정 피처 목록 ----
# 특허 모형: v12(F12)에서 registry-등록분 기반 출원활동(own_app3·own_app_prev·own_app_mom·own_years)을 kr_patents 전체출원 기반(a_*)으로 1:1 교체
PF_FIX = [f for f in run.F12 if f not in ('own_app3', 'own_app_prev', 'own_app_mom', 'own_years')] + ['a_app3', 'a_app_prev', 'a_mom', 'a_years']
OWN_MEAN = ['age', 'lclaims', 'inv_is_app', 'prior_MAIN', 'cur_is_orig', 'paid_thru', 'prepaid', 'for_prio', 'pct', 'a_tech_exit', 'a_ipc_app3',
            'own_ipc_share', 'lic_any', 'n_inv', 'exam_yrs', 'ksic_rate3', 'multi', 'amt_per_yr', 'rgst_year']
OWN_LEVEL = ['own_port', 'own_lapse3', 'own_lapse1', 'own_lapse_rate', 'own_lapse_trend', 'own_corp3', 'own_in3', 'own_in1',
             'a_app3', 'a_app_prev', 'a_mom', 'a_years', 'a_dormant', 'own_n_ksic', 'cur_type', 'cur_n']
ATTR2 = ['natl_grp', 'chaebol', 'ip_entity']
CFGS = [(5, 0.7), (4, 0.5), (6, 0.5), (5, 0.5), (4, 0.7), (6, 0.7)]

BASE = dict(pat_feats=PF_FIX, pat_xgb={}, own_mean=OWN_MEAN, own_level=OWN_LEVEL, own_attr=ATTR2, bag=6, ntree=500, lr=0.03, mcw=5,
            sub=0.8, lam=5.0, wpow=0.5, beta=0.5, recency=None, extra=[], pat_recency=None, extra_level=[], extra_mean=[], extra_pat=[])

EXTRA_FN = {}   # 이름 → f(d) (d에 열 추가). 버전별 신규 피처 빌더를 등록


def load(years, cfg=BASE):
    years = list(years)
    d = run.load(years)
    d = d.merge(pd.concat([pd.read_parquet(f'cache/act_L{y}.parquet') for y in years]), on=['rgst_no', 'L'], how='left')
    d = d.merge(pd.read_parquet('cache/owner_attr2.parquet').rename(columns={'nm': 'cur'}), on='cur', how='left')
    for name in cfg.get('extra', []):
        d = EXTRA_FN[name](d)
    return d


def owner_table(d, cfg, with_y=True):
    g = d.groupby(['L', 'cur'])
    o = g[cfg['own_mean'] + cfg['extra_mean']].mean().add_prefix('m_')
    o = o.join(g[cfg['own_level'] + cfg['own_attr'] + cfg['extra_level']].first())
    o['ksic_mode'] = g.ksic_c.agg(lambda s: s.value_counts().index[0])
    o['n'] = g.size()
    if with_y: o['rate'] = g.ytr.mean()
    return o.reset_index()


def owner_feats(cfg):
    return ['m_' + c for c in cfg['own_mean'] + cfg['extra_mean']] + cfg['own_level'] + ['ksic_mode'] + cfg['own_attr'] + cfg['extra_level']


def rec_w(L, gamma, Lref):
    return np.ones(len(L)) if gamma is None else np.power(float(gamma), (Lref - np.asarray(L)).astype(float))


def fit_apply(cfg, train_years, apply_years, tr=None, ap=None):
    cfg = {**BASE, **cfg}
    if tr is None: tr = load(train_years, cfg)
    if ap is None: ap = load(apply_years, cfg)
    tr = tr.copy(); ap = ap.copy(); tr['ytr'] = tr.y_MAIN; ap['ytr'] = 0
    Lref = max(train_years)
    # (1) 특허 단위 점수
    PFe = cfg['pat_feats'] + cfg['extra_pat']
    prm = dict(run.XGB_DEF); prm.update(cfg['pat_xgb'])
    m = xgb.XGBClassifier(**prm, nthread=2, tree_method='hist', eval_metric='auc', random_state=0)
    m.fit(tr[PFe], tr.ytr, sample_weight=rec_w(tr.L, cfg['pat_recency'], Lref))
    ap['p_pat'] = m.predict_proba(ap[PFe])[:, 1]
    # (2) 권리자 단위 매각률
    ot = owner_table(tr, cfg); oa = owner_table(ap, cfg, with_y=False)
    F = owner_feats(cfg)
    w = ot.n.values ** cfg['wpow'] * rec_w(ot.L, cfg['recency'], Lref)
    preds = []
    for i in range(cfg['bag']):
        depth, col = CFGS[i % len(CFGS)]
        mo = xgb.XGBRegressor(objective='reg:logistic', n_estimators=cfg['ntree'], max_depth=depth, learning_rate=cfg['lr'], subsample=cfg['sub'],
                              colsample_bytree=col, min_child_weight=cfg['mcw'], reg_lambda=cfg['lam'], nthread=2, tree_method='hist', random_state=i)
        mo.fit(ot[F], ot.rate, sample_weight=w); preds.append(mo.predict(oa[F]))
    oa['owner_hat'] = np.mean(preds, axis=0)
    if cfg.get('hurdle'):   # 허들: P(권리자가 3년 안에 1건 이상 매각) × E[매각비율 | 매각]
        any_ = (ot.rate > 0).astype(int); pa, pr = [], []
        sel = ot.rate > 0
        for i in range(cfg['bag']):
            depth, col = CFGS[i % len(CFGS)]
            mc = xgb.XGBClassifier(n_estimators=cfg['ntree'], max_depth=depth, learning_rate=cfg['lr'], subsample=cfg['sub'], colsample_bytree=col,
                                   min_child_weight=cfg['mcw'], reg_lambda=cfg['lam'], nthread=2, tree_method='hist', random_state=i)
            mc.fit(ot[F], any_, sample_weight=w); pa.append(mc.predict_proba(oa[F])[:, 1])
            mr = xgb.XGBRegressor(objective='reg:logistic', n_estimators=cfg['ntree'], max_depth=depth, learning_rate=cfg['lr'], subsample=cfg['sub'],
                                  colsample_bytree=col, min_child_weight=cfg['mcw'], reg_lambda=cfg['lam'], nthread=2, tree_method='hist', random_state=i)
            mr.fit(ot.loc[sel, F], ot.rate[sel], sample_weight=w[sel.values]); pr.append(mr.predict(oa[F]))
        oa['owner_hat'] = np.mean(pa, axis=0) * np.mean(pr, axis=0)
    ap = ap.merge(oa[['L', 'cur', 'owner_hat']], on=['L', 'cur'], how='left')
    ap['rel'] = ap.p_pat / ap.groupby(['L', 'cur']).p_pat.transform('mean')
    ap['p'] = ap.owner_hat * ap.rel ** cfg['beta']
    return ap


def evaluate(ver, ap, change=None, hyp=None, kept='?', ref='v28', write=True):
    base = json.load(open('cache/v0_metrics.json'))['y_MAIN']
    mt = metrics(ap.y_MAIN, ap.p); ci = cluster_boot(ap.y_MAIN.values, ap.p.values, ap.cur.values)
    per = {L: round(metrics(ap.y_MAIN[ap.L == L], ap.p[ap.L == L])['auc'], 4) for L in sorted(ap.L.unique())}
    oh = metrics(ap.y_MAIN, ap.owner_hat)['auc']
    print(ver, 'AUC', round(mt['auc'], 4), 'PR', round(mt['prauc'], 4), 'p1', round(mt['p1'], 4), 'CI', ci[0].round(3), 'perL', per,
          'owner_hat only', round(oh, 4), 'skill', round(skill(mt['auc'], base['auc']), 4), flush=True)
    dstr = ''
    if write:
        ap[['rgst_no', 'L', 'p']].to_parquet(f'cache/pred_{ver}.parquet', index=False)
        json.dump(dict(auc=mt['auc'], prauc=mt['prauc'], p1=mt['p1'], ci_auc=list(ci[0]), ci_pr=list(ci[1])), open(f'cache/metrics_{ver}.json', 'w'))
        if ref is not None and ref != ver:
            dd = paired(ver, ref); dstr = f' | 쌍대부트 d(vs {ref}) {dd[0]:+.4f} [{dd[1]:.4f},{dd[2]:.4f}] P(d>0)={dd[3]:.2f}'
            print('  ', dstr, flush=True)
        if change is not None:
            note = (f"PR-AUC {mt['prauc']:.4f} (기준선 {base['prauc']:.4f}), p@1% {mt['p1']:.4f}; AUC 95%CI(양도인 군집) {ci[0][0]:.3f}-{ci[0][1]:.3f}"
                    f" | L별 {per} | 권리자점수 단독 {oh:.4f}" + dstr)
            log(ver, change, hyp, mt['auc'], base['auc'], kept, note)
    return mt, ci


def paired(v, ref, B=200):
    from sklearn.metrics import roc_auc_score
    va = load_panel(VAL_L, ['rgst_no', 'L', 'cur', 'y_MAIN'])
    for x in (v, ref):
        va = va.merge(pd.read_parquet(f'cache/pred_{x}.parquet').rename(columns={'p': 'p_' + x}), on=['rgst_no', 'L'])
    rng = np.random.default_rng(1)
    codes, uniq = pd.factorize(va.cur.fillna('NA')); order = np.argsort(codes, kind='stable'); cs = codes[order]
    st = np.searchsorted(cs, np.arange(len(uniq))); en = np.searchsorted(cs, np.arange(len(uniq)), side='right')
    y = va.y_MAIN.values; a = va['p_' + v].values; b = va['p_' + ref].values; D = []
    for _ in range(B):
        idx = np.concatenate([order[st[c]:en[c]] for c in rng.integers(0, len(uniq), len(uniq))])
        D.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    D = np.array(D)
    return D.mean(), np.percentile(D, 2.5), np.percentile(D, 97.5), np.mean(D > 0)


# ---- 2단계 개선 버전용 피처 빌더 ----
FAM = ['fam_US', 'fam_EP', 'fam_JP', 'fam_CN', 'fam_WO', 'n_fam_cc', 'n_fam_docs']


def _fam(d):
    """제공 5개국 메타 family: 출원 후 2.5년(30개월 국내진입 기한) 미경과 특허는 결측 처리."""
    f = pd.read_parquet('cache/fam.parquet')
    d = d.merge(f, on='rgst_no', how='left')
    d.loc[d.age_appl < 2.5, FAM] = np.nan
    return d


MKT_OWN = ['sale_prop5', 'sale_ever', 'yrs_since_sale']
MKT_MKT = ['ksic_buyers3', 'ksic_sellers3', 'ipc4_buyers3', 'ipc4_sellers3', 'ksic_b_for3', 'ksic_b_ip3', 'ksic_b_ind3']


def _mkt(d):
    ys = sorted(d.L.unique())
    return d.merge(pd.concat([pd.read_parquet(f'cache/mkt_L{y}.parquet') for y in ys]), on=['rgst_no', 'L'], how='left')


EXTRA_FN['fam'] = _fam
EXTRA_FN['mkt'] = _mkt

IND = ['ind_corp_inv5', 'ind_corp_inv5_ksic']


def _ind(d):
    ys = sorted(d.L.unique())
    return d.merge(pd.concat([pd.read_parquet(f'cache/ind_L{y}.parquet') for y in ys]), on=['rgst_no', 'L'], how='left')


COST = ['own_fee_sum', 'fee_per_app', 'own_share_old', 'own_share_new']


def _cost(d):
    """포트폴리오 유지비 압박: 권리자 연간 유지비 합(연당 납부액 합), 최근 출원 대비 유지비, 오래된(9년+)·신규(3년-) 특허 비중. 모두 L 이전 납부·등록 정보."""
    g = d.groupby(['L', 'cur'])
    d['own_fee_sum'] = np.log1p(g.amt_per_yr.transform('sum'))
    d['fee_per_app'] = d.own_fee_sum - np.log1p(d.a_app3)
    d['own_share_old'] = (d.age >= 9).groupby([d.L, d.cur]).transform('mean')
    d['own_share_new'] = (d.age < 3).groupby([d.L, d.cur]).transform('mean')
    return d


EXTRA_FN['ind'] = _ind
EXTRA_FN['cost'] = _cost
