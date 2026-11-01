# 실험 러너: python exp.py <version>
import sys, time, pandas as pd, numpy as np
from lib import *
from feats import agg_feats, industry_mom, partner_feats
CAP_TR = '20091231'
BASE = ['lsize', 'lin', 'lout', 'inshare', 'dens', 'kpop', 'dm2', 'dm3', 'dm4', 'has_dart', 'ind2']
AGG = ['t_mean', 't_hi', 't_lsum', 't_max', 't_n20', 't_top3', 't_rank', 't_ent']
MOM = ['t_rw', 't_rec3', 't_mom', 't_lrec_n', 't_grow']
NBR = ['t_nbr']
KMOM = ['k_grow', 'k_newent']
PART = ['p_n', 'p_k', 'p_share']

def panel(cut, cap, clf_tag, P, extra=()):
    tag = f'panel_c{cut}_{cap}_{clf_tag}_{"-".join(extra)}.parquet'
    p = PC/tag
    d, c = build_panel(cut, 5, cap=cap)
    if p.exists():
        return pd.read_parquet(p), c
    d = add_dart(d, c)
    d = agg_feats(d, c, P)
    d = industry_mom(d, c)
    if 'part' in extra: d = partner_feats(d, c)
    d['cut'] = cut
    d.to_parquet(p, index=False)
    return d, c

def get(train_cuts=(2004,), clf_tag='c2009_base', extra=(), clf_kw=None):
    P = train_clf(CAP_TR, clf_tag, **(clf_kw or {}))
    trs = [panel(c, CAP_TR, clf_tag, P, extra)[0] for c in train_cuts]
    te, cte = panel(2009, None, clf_tag, P, extra)
    return pd.concat(trs, ignore_index=True), te, cte

def logit(tr, te, cols, C=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    cols = [c for c in cols if c != 'ind2']
    X = tr[cols].fillna(0).values; Xt = te[cols].fillna(0).values
    sc = StandardScaler().fit(X); lr = LogisticRegression(max_iter=3000, C=C).fit(sc.transform(X), tr.y)
    return lr.predict_proba(sc.transform(Xt))[:, 1]

def rank(s):
    return pd.Series(s).rank(pct=True).values

if __name__ == '__main__':
    v = sys.argv[1]; t0 = time.time()
    base = json.load(open(T1/'baseline_v0.json'))['fixed']
    res = {}
    if v == 'v2':   # 피처 집계 변형 추가
        tr, te, _ = get()
        cols = BASE + AGG; s = predict(fit_xgb(tr, cols), te, cols); met = metrics(te, s)
        lb_add(v, '+노출 집계 변형(최대·상위3평균·로그합·p>0.2 건수·법인 내 순위·엔트로피)', '평균확률은 대형 포트폴리오에서 희석된다: 최대/상위k/순위가 신호를 보존', met, base, '?', '')
    elif v == 'v3':  # + 시간 모멘텀
        tr, te, _ = get()
        cols = BASE + AGG + MOM; s = predict(fit_xgb(tr, cols), te, cols); met = metrics(te, s)
        lb_add(v, '+시간 모멘텀(반감기3년 가중평균·최근3년 평균·최근-과거 차이·최근 출원량·성장률)', '최근에 산업 쪽으로 이동 중인 기업이 곧 진입한다', met, base, '?', '')
    elif v == 'v4':  # + 관련산업 확산 + 산업 모멘텀
        tr, te, _ = get()
        cols = BASE + AGG + MOM + NBR + KMOM; s = predict(fit_xgb(tr, cols), te, cols); met = metrics(te, s)
        lb_add(v, '+관련산업 확산(노출×산업근접도) +산업 모멘텀(범위안 출원 성장·최근 신규진입 수)', '인접 산업 노출과 끌어당기는 산업의 추세가 진입 확률을 높인다', met, base, '?', '')
    elif v == 'v5':  # 학습 원점 스태킹
        tr, te, _ = get(train_cuts=(2002, 2003, 2004))
        cols = BASE + AGG + MOM + NBR + KMOM; s = predict(fit_xgb(tr, cols), te, cols); met = metrics(te, s)
        lb_add(v, '학습 원점 스태킹(2002·2003·2004, 레이블 등록≤2009-12-31)', '학습 표본 부족(6천사)이 병목: 원점을 쌓으면 분산이 줄어든다', met, base, '?', f'train rows={len(tr)} pos={tr.y.sum()}')
    print(v, fmt(met), 'time', round(time.time() - t0, 1))
    te[['firm', 'k', 'y', 'newc']].assign(s=s).to_parquet(CACHE/f'{v}_scores.parquet', index=False)
