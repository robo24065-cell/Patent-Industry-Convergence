# T1 산업 대기실 — 동결 최종 모형 실행기 (2단계 동결, 2026-09-28)
#   python final_run.py --mode dryrun    검증 구간(평가 c=2009, 학습 원점 2001–04, 레이블·분류기 cap 2009-12-31)으로 처음부터 재구축·학습·평가
#   python final_run.py --mode holdout   🔒 SPLITS 규칙: 평가 c=2014(피처 ≤2014 출원·≤2014-12-31 등록 가시) → 진입 2015–2019.
#                                         학습 원점 2006–09, 학습 레이블·분류기·IPW·지연CDF는 2014-12-31 이전 관측분만 → holdout_result.json
#   python final_run.py --mode future    평가 c=2024(피처 ≤2024-12-31 가시) → 2025–2029 잠재 진입, 학습 원점 2016–19(cap 2024-12-31) → future_forecast.csv
# 한 번 실행이 9분을 넘지 않도록 단계별로 final_cache/<mode>/ 에 캐시하고, 시간 예산을 넘으면 종료코드 3으로 끝난다 → 같은 명령을 다시 실행하면 이어서 진행.
# 사전 조건(모드 무관 원천 캐시, 이미 존재): cache/ins_pat, ins_firm, oos, firms, firm_meta, dart_map, app_pub (prep1.py, prep2.py, prep_app.py)
import os, sys, json, time, argparse, pathlib, pickle
os.environ['T1_STRICT'] = '1'; os.environ['T1_VIS'] = 'rgst'
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import lib
from lib import load, build_panel, add_dart, metrics, CACHE, T1, B, SNAP_CUT
from feats import agg_feats, industry_mom, partner_feats
from sklearn.metrics import roc_auc_score, average_precision_score

T_START = time.time(); BUDGET = 530
SPEC = json.load(open(HERE/'final_spec.json', encoding='utf-8'))
MODES = {
    'dryrun':  dict(eval_cut=2009, train_cuts=(2001, 2002, 2003, 2004), cap='20091231', guard=2014),
    'holdout': dict(eval_cut=2014, train_cuts=(2006, 2007, 2008, 2009), cap='20141231', guard=2019),
    'future':  dict(eval_cut=2024, train_cuts=(2016, 2017, 2018, 2019), cap='20241231', guard=2029),
}
BASE = ['lsize', 'lin', 'lout', 'inshare', 'dens', 'kpop', 'dm2', 'dm3', 'dm4', 'has_dart', 'ind2']


def elapsed(): return time.time() - T_START
def need_time(sec, what):
    if elapsed() + sec > BUDGET:
        print(f'[시간 예산] {what} 전에 중단 — 같은 명령을 다시 실행하면 캐시에서 이어집니다 (경과 {elapsed():.0f}s)', flush=True); sys.exit(3)


# ---------------- 분류기 ----------------
def clf_base(cfg, FC):
    """제목→KSIC 50-class: cap까지 등록된 범위안 특허로 학습 → 범위밖 자사출원 전체 채점"""
    p = FC/'P_base.npy'
    if p.exists(): return np.load(p).astype(np.float32)
    need_time(300, '50-class 분류기')
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier
    D = load(); oos = D['oos']
    kr = pd.read_parquet(CACHE/'ins_pat.parquet', columns=['title', 'ksic', 'rgst_date'])
    tr = kr[kr.rgst_date <= cfg['cap']].dropna(subset=['title'])
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=5, max_features=400000, sublinear_tf=True, dtype=np.float32)
    clf = SGDClassifier(loss='log_loss', alpha=2e-6, max_iter=12, tol=None, random_state=0).fit(vec.fit_transform(tr.title), tr.ksic)
    assert list(clf.classes_) == D['K']
    P = np.zeros((len(oos), 50), np.float16)
    for s in range(0, len(oos), 200000):
        P[s:s+200000] = clf.predict_proba(vec.transform(oos.title.iloc[s:s+200000].fillna(''))).astype(np.float16)
    np.save(p, P); print('clf_base saved', round(elapsed()), flush=True)
    return P.astype(np.float32)


def clf_os51(cfg, FC):
    """open-set 51-class(범위밖 자사출원=OTHER), 법인 해시 2-fold 교차적합. fold별 캐시."""
    pP, pO = FC/'P_os51.npy', FC/'PO_os51.npy'
    if pP.exists(): return np.load(pP).astype(np.float32), np.load(pO).astype(np.float32)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier
    D = load(); oos = D['oos']
    kr = pd.read_parquet(CACHE/'ins_pat.parquet', columns=['title', 'ksic', 'rgst_date'])
    tr_in = kr[kr.rgst_date <= cfg['cap']].dropna(subset=['title'])
    fold = (pd.util.hash_array(oos.applicant_cd.values.astype(object)) % 2).astype(int)
    for fo in (0, 1):
        fp = FC/f'os51_fold{fo}.npz'
        if fp.exists(): continue
        need_time(330, f'open-set 분류기 fold {fo}')
        oth = oos[(fold != fo) & (oos.rgst_date <= cfg['cap'])]
        titles = pd.concat([tr_in.title, oth.title.fillna('')]); ys = np.concatenate([tr_in.ksic.values, np.array(['OTHER'] * len(oth))])
        vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=5, max_features=400000, sublinear_tf=True, dtype=np.float32)
        clf = SGDClassifier(loss='log_loss', alpha=2e-6, max_iter=12, tol=None, random_state=0).fit(vec.fit_transform(titles), ys, sample_weight=np.ones(len(ys)))
        cls = list(clf.classes_); oi = cls.index('OTHER'); ki = [cls.index(k) for k in D['K']]
        idx = np.nonzero(fold == fo)[0]; Pf = np.zeros((len(idx), 50), np.float16); POf = np.zeros(len(idx), np.float16)
        for s in range(0, len(idx), 200000):
            pr = clf.predict_proba(vec.transform(oos.title.iloc[idx[s:s+200000]].fillna('')))
            Pf[s:s+200000] = pr[:, ki].astype(np.float16); POf[s:s+200000] = pr[:, oi].astype(np.float16)
        np.savez(fp, P=Pf, PO=POf); print('os51 fold', fo, 'saved', round(elapsed()), flush=True)
    P = np.zeros((len(oos), 50), np.float16); PO = np.zeros(len(oos), np.float16)
    for fo in (0, 1):
        z = np.load(FC/f'os51_fold{fo}.npz'); idx = np.nonzero(fold == fo)[0]; P[idx] = z['P']; PO[idx] = z['PO']
    np.save(pP, P); np.save(pO, PO)
    return P.astype(np.float32), PO.astype(np.float32)


def clf_ins(cfg, FC):
    """범위안 제목의 2차 산업 확률(특허 해시 2-fold 교차적합, cap까지 등록된 범위안으로 학습)"""
    p = FC/'P_ins_cf.npy'
    if p.exists(): return np.load(p), pd.read_parquet(FC/'P_ins_cf_index.parquet').rgst_no
    need_time(300, '범위안 2차 산업 분류기')
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier
    kr = pd.read_parquet(CACHE/'ins_pat.parquet', columns=['rgst_no', 'title', 'ksic', 'rgst_date']); kr['title'] = kr.title.fillna('')
    fold = (pd.util.hash_array(kr.rgst_no.values.astype(object)) % 2).astype(int)
    P = np.zeros((len(kr), 50), np.float16); K = load()['K']
    for fo in (0, 1):
        tr = kr[(fold != fo) & (kr.rgst_date <= cfg['cap'])]
        vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=5, max_features=400000, sublinear_tf=True, dtype=np.float32)
        clf = SGDClassifier(loss='log_loss', alpha=2e-6, max_iter=12, tol=None, random_state=0).fit(vec.fit_transform(tr.title), tr.ksic)
        assert list(clf.classes_) == K
        idx = np.nonzero(fold == fo)[0]; P[idx] = clf.predict_proba(vec.transform(kr.title.iloc[idx])).astype(np.float16)
    np.save(p, P); kr[['rgst_no']].to_parquet(FC/'P_ins_cf_index.parquet', index=False); print('clf_ins saved', round(elapsed()), flush=True)
    return P, kr.rgst_no


def ep_match(cfg, FC):
    """EP 출원인 영문명 ↔ 법인 영문명(registry applicantEngName) 정규화 일치, 출원연도 ≤ 평가 기준시점"""
    p = FC/'ep_firm.parquet'
    if p.exists(): return pd.read_parquet(p)
    need_time(200, 'EP 매칭')
    import duckdb, re
    M = (B/'분석/모델경쟁_20260927/_shared/meta').as_posix() + '/*.parquet'
    ep = duckdb.sql(f"select appl_no, applicant, appl_year, pub_date, ksic, prio_cc from read_parquet('{M}') where cc='EP' and appl_year<={cfg['eval_cut']}").df()
    SUF = re.compile(r"\b(CO|COMPANY|CORP|CORPORATION|INC|INCORPORATED|LTD|LIMITED|LLC|PLC|GMBH|AG|SA|SAS|SPA|BV|NV|SE|AB|OY|KK|KABUSHIKI KAISHA|KABUSHIKIKAISHA|CO LTD|THE|& CO|KG|S A|S P A|B V|N V)\b")
    def norm(s):
        s = str(s).upper(); s = re.sub(r"[\.\,\(\)\-&'/]", ' ', s); s = SUF.sub(' ', s); return re.sub(r'\s+', '', s)
    ep = ep.dropna(subset=['applicant']); ep['nm'] = ep.applicant.str.split('|'); ep = ep.explode('nm'); ep['n'] = ep.nm.map(norm)
    app = pd.read_sql('select applicantCd, applicantEngName from applicant', lib.RG())
    firms = set(pd.read_parquet(CACHE/'firms.parquet').query("status=='OK'").applicant_cd)
    app = app[app.applicantCd.isin(firms)].dropna(subset=['applicantEngName'])
    app['n'] = app.applicantEngName.map(norm); app = app[app.n.str.len() >= 4].drop_duplicates(['applicantCd', 'n'])
    amb = app.groupby('n').applicantCd.nunique(); app = app[app.n.map(amb) == 1]
    m = ep.merge(app[['n', 'applicantCd']], on='n').drop_duplicates(['appl_no', 'ksic', 'applicantCd'])
    m = m.rename(columns={'applicantCd': 'applicant_cd'})[['applicant_cd', 'appl_no', 'appl_year', 'pub_date', 'ksic', 'prio_cc']]
    m.to_parquet(p, index=False); print('ep saved', len(m), round(elapsed()), flush=True)
    return m


# ---------------- 기준시점 패널 피처 ----------------
def lag_cdf(cfg):
    """등록지연 CDF(국내/외국법인): 출원연도 ≤ 평가기준시점−7 코호트, cap까지 등록된 것(관측 가능분만)"""
    D = load(); ins = D['ins']; tp = D['fm'].set_index('applicant_cd').tp
    coh = ins[(ins.ay <= cfg['eval_cut'] - 7) & (ins.rgst_date <= cfg['cap'])].drop_duplicates('rgst_no')
    fr = coh.applicant_cd.map(tp).eq('외국법인').values
    lag = (pd.to_datetime(coh.rgst_date) - pd.to_datetime(coh.appl_date)).dt.days.values
    return {g: np.sort(lag[fr == g]) for g in (False, True)}


def cut_features(cut, cap, cfg, FC, P, P51, PO, Pins, ins_idx, ep, cdf):
    p = FC/f'feat_c{cut}.parquet'
    if p.exists(): return pd.read_parquet(p)
    need_time(150, f'패널 c={cut}')
    t0 = time.time(); D = load(); K = D['K']; kidx = D['kidx']
    d, c = build_panel(cut, 5, cap=cap)
    d = add_dart(d, c); d = agg_feats(d, c, P); d = industry_mom(d, c); d['cut'] = cut
    f, k = d.fidx.values, d.k.values; op = c['op']; fi = c['fi']; nF = len(c['firms']); rows = fi[op.applicant_cd].values
    # open-set 51
    o = agg_feats(d[['fidx', 'k']].copy(), c, P51)
    for cc in ['t_mean', 't_max', 't_top3', 't_hi', 't_rw', 't_nbr']: d['o' + cc] = o[cc].values
    s50 = np.bincount(rows, weights=1 - PO[op.row.values], minlength=nF) / np.maximum(np.bincount(rows, minlength=nF), 1)
    d['oin50'] = s50[f]
    # 공동출원 파트너
    x = partner_feats(d[['fidx', 'k']].copy(), c)
    for cc in ['p_n', 'p_k', 'p_share']: d[cc] = x[cc].values
    # 해외 선행 EP (기준시점 연말까지 공개분)
    e = ep.copy(); e['k'] = e.ksic.astype(str).map(kidx); e = e.dropna(subset=['k']); e['k'] = e.k.astype(int)
    e = e[(e.pub_date.astype(str) <= f'{cut}1231') & (e.appl_year >= 2000) & e.pub_date.notna()].drop_duplicates(['applicant_cd', 'appl_no', 'k'])
    def pairc(sub):
        s = sub.groupby(['applicant_cd', 'k']).size()
        return s.reindex(pd.MultiIndex.from_arrays([d.firm.values, k])).fillna(0).values
    ck = pairc(e); cr = pairc(e[e.appl_year > cut - 3]); ckr = pairc(e[e.prio_cc.fillna('').str.contains('KR')])
    n = d.firm.map(e.drop_duplicates(['applicant_cd', 'appl_no']).groupby('applicant_cd').size()).fillna(0).values
    d['ep_k'] = np.log1p(ck); d['ep_n'] = np.log1p(n); d['ep_share'] = ck / np.maximum(n, 1); d['ep_k_rec'] = np.log1p(cr); d['ep_krprio'] = np.log1p(ckr)
    # KSIC 계층 형제
    M = c['M']; TM = np.zeros_like(M); TM[f, k] = d.t_mean.values
    for L, nm in [(4, 'sib4'), (3, 'sib3'), (2, 'sib2')]:
        G = np.array([[1.0 if (K[i][:L] == K[j][:L] and i != j) else 0.0 for j in range(50)] for i in range(50)], np.float32)
        d[nm] = np.log1p(M @ G)[f, k]
        if L in (3, 2): d['t' + nm] = (TM @ G)[f, k]
    # 범위안 제목 2차 산업
    pos = pd.Series(np.arange(len(ins_idx)), index=ins_idx.values)
    ip = c['ip'].drop_duplicates(['applicant_cd', 'rgst_no']); r = fi[ip.applicant_cd].values; S = Pins[pos[ip.rgst_no].values].astype(np.float32)
    T = np.zeros((nF, 50), np.float32); np.add.at(T, r, S); nn = np.bincount(r, minlength=nF)
    MX = np.zeros((nF, 50), np.float32); np.maximum.at(MX, r, S); N2 = np.zeros((nF, 50), np.float32); np.add.at(N2, r, (S > 0.2).astype(np.float32))
    d['ti_mean'] = (T / np.maximum(nn, 1)[:, None])[f, k]; d['ti_max'] = MX[f, k]; d['ti_n20'] = np.log1p(N2[f, k]); d['ti_lsum'] = np.log1p(T[f, k])
    # 활동 최근성
    allp = pd.concat([op[['applicant_cd', 'ay', 'rgst_no']], c['ip'][['applicant_cd', 'ay', 'rgst_no']]]).drop_duplicates(['applicant_cd', 'rgst_no'])
    last = allp.groupby('applicant_cd').ay.max(); n1 = allp[allp.ay == cut].groupby('applicant_cd').size()
    d['gap_last'] = cut - d.firm.map(last).values; d['n_last1'] = np.log1p(d.firm.map(n1).fillna(0).values)
    # 등록일 기준 최근성
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
    # 나우캐스팅
    tp = D['fm'].set_index('applicant_cd').tp; ce = pd.Timestamp(f'{cut}-12-31')
    al = pd.concat([op[['applicant_cd', 'rgst_no', 'appl_date', 'ay']].assign(ins=0), c['ip'][['applicant_cd', 'rgst_no', 'appl_date', 'ay']].assign(ins=1)]).drop_duplicates(['applicant_cd', 'rgst_no'])
    days = (ce - pd.to_datetime(al.appl_date.astype(str), format='%Y%m%d', errors='coerce')).dt.days.fillna(3650).values
    fr = al.applicant_cd.map(tp).eq('외국법인').values
    pi = np.where(fr, np.searchsorted(cdf[True], days, side='right') / len(cdf[True]), np.searchsorted(cdf[False], days, side='right') / len(cdf[False]))
    al['w'] = 1 / np.maximum(pi, 0.05); al['pi'] = pi
    g_n1 = al[al.ay == cut].groupby('applicant_cd').w.sum(); g_n3 = al[al.ay > cut - 3].groupby('applicant_cd').w.sum()
    g_p3 = al[(al.ay > cut - 6) & (al.ay <= cut - 3)].groupby('applicant_cd').w.sum(); g_i3 = al[(al.ay > cut - 3) & (al.ins == 1)].groupby('applicant_cd').w.sum()
    fm_ = d.firm
    d['now1'] = np.log1p(fm_.map(g_n1).fillna(0).values); d['now3'] = np.log1p(fm_.map(g_n3).fillna(0).values)
    d['now_in3'] = np.log1p(fm_.map(g_i3).fillna(0).values); d['vis_frac'] = fm_.map(al.groupby('applicant_cd').pi.mean()).fillna(0).values
    d['now_grow'] = d.now3 - np.log1p(fm_.map(g_p3).fillna(0).values)
    # 공개 출원(범위안, 미등록 포함) — 기준시점 연말 현재 가시 정보만(감사 수정판)
    from s2lib import add_pa2
    d = add_pa2(d)
    d.to_parquet(p, index=False); print(f'feat c={cut} rows={len(d)} {time.time() - t0:.0f}s', flush=True)
    return d


def base_app_panel(cut, cap, FC):
    """고정 기준선(트랙 정의)용 출원 기준 패널: 규모+density+kpop+DART, 기준시점 이하 출원이면 이후 등록분도 사용(기준선에 유리한 정보 → 보수적 비교)"""
    p = FC/f'base_app_c{cut}.parquet'
    if p.exists(): return pd.read_parquet(p)
    need_time(90, f'기준선 패널 c={cut}')
    lib.STRICT = False
    try:
        d, c = build_panel(cut, 5, cap=cap); d = add_dart(d, c)
    finally:
        lib.STRICT = True
    d = d[['firm', 'k', 'y', 'newc'] + BASE]; d['cut'] = cut; d.to_parquet(p, index=False)
    return d


# ---------------- 학습 ----------------
def ipw(tr, cfg):
    D = load(); ins = D['ins']; kidx = D['kidx']; tp = D['fm'].set_index('applicant_cd').tp
    cdf = lag_cdf(cfg); cap = cfg['cap']; w = np.ones(len(tr)); capd = pd.Timestamp(cap)
    for cut in sorted(tr.cut.unique()):
        fut = ins[(ins.ay > cut) & (ins.ay <= cut + 5) & (ins.rgst_date <= cap)]
        fst = fut.groupby(['applicant_cd', 'ksic']).appl_date.min().reset_index(); fst['k'] = fst.ksic.map(kidx)
        idx = np.nonzero((tr.cut.values == cut) & (tr.y.values == 1))[0]
        x = tr.iloc[idx][['firm', 'k']].merge(fst, left_on=['firm', 'k'], right_on=['applicant_cd', 'k'], how='left')
        days = (capd - pd.to_datetime(x.appl_date)).dt.days.values; fr = x.firm.map(tp).eq('외국법인').values
        pr = np.array([np.searchsorted(cdf[g], dd, side='right') / len(cdf[g]) for dd, g in zip(days, fr)])
        w[idx] = np.minimum(1 / np.maximum(pr, 1e-3), SPEC['ipw_wmax'])
    return w


def k_prior(tr, te):
    cuts = sorted(tr.cut.unique()); glob = tr.y.mean(); glob_n = tr[tr.newc].y.mean()
    def enc(sub, g0):
        g = sub.groupby('k').y.agg(['sum', 'size']); return ((g['sum'] + 20 * g0) / (g['size'] + 20)).reindex(range(50)).fillna(g0)
    ktr = np.zeros(len(tr)); ntr = np.zeros(len(tr))
    for c in cuts:
        m = tr.cut.values == c
        ktr[m] = np.log(enc(tr[~m], glob).values[tr.k.values[m]]); ntr[m] = np.log(enc(tr[~m & tr.newc.values], glob_n).values[tr.k.values[m]])
    tr['k_prior'] = ktr; tr['k_prior_new'] = ntr
    te['k_prior'] = np.log(enc(tr, glob).values[te.k.values]); te['k_prior_new'] = np.log(enc(tr[tr.newc], glob_n).values[te.k.values])
    return tr, te


def logit_fit(tr, cols, w=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    cols = [c for c in cols if c != 'ind2']
    X = tr[cols].fillna(0).values; sc = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=3000).fit(sc.transform(X), tr.y.values)
    return dict(sc=sc, lr=lr, cols=cols)
def logit_pred(L, te): return L['lr'].predict_proba(L['sc'].transform(te[L['cols']].fillna(0).values))[:, 1]
def rank(s): return pd.Series(s).rank(pct=True).values


def full_metrics(d, s):
    m = metrics(d, s)
    vis3 = np.expm1(d.lsize.values).round() >= 3   # 보조: 기준시점 가시 이력 ≥3건 법인(모집단을 기준시점 정보만으로 정의)
    m['auc_all_vis3'] = float(roc_auc_score(d.y.values[vis3], s[vis3])); nv = vis3 & d.newc.values
    m['auc_new_vis3'] = float(roc_auc_score(d.y.values[nv], s[nv])); m['pr_new_vis3'] = float(average_precision_score(d.y.values[nv], s[nv]))
    return m


def boot(y, nc, firm, S, FC, R=200, chunk=20):
    """법인 군집 부트스트랩(법인을 복원추출, 그 법인의 모든 쌍을 함께). chunk 단위로 FC/boot_*.npy 캐시 → 시간 예산 초과 시 재실행으로 이어감"""
    fr, inv = np.unique(firm, return_inverse=True)
    order = np.argsort(inv, kind='stable'); st = np.searchsorted(inv[order], np.arange(len(fr))); en = np.searchsorted(inv[order], np.arange(len(fr)), side='right')
    keys = list(S); M = ['auc_all', 'pr_all', 'auc_new', 'pr_new']; parts = []
    for ci in range(int(np.ceil(R / chunk))):
        fp = FC/f'boot_R{R}_{ci}.npy'
        if fp.exists(): parts.append(np.load(fp)); continue
        need_time(chunk * 2.2 * len(y) / 660199 * len(keys) / 3, f'부트스트랩 chunk {ci}')
        rng = np.random.RandomState(1000 + ci); arr = np.zeros((min(chunk, R - ci * chunk), len(keys), 4))
        for r in range(arr.shape[0]):
            idx = np.concatenate([order[st[i]:en[i]] for i in rng.randint(0, len(fr), len(fr))]); yy = y[idx]; nn = nc[idx]
            for j, kk in enumerate(keys):
                ss = S[kk][idx]
                arr[r, j] = [roc_auc_score(yy, ss), average_precision_score(yy, ss), roc_auc_score(yy[nn], ss[nn]), average_precision_score(yy[nn], ss[nn])]
        np.save(fp, arr); parts.append(arr); print('boot chunk', ci, 'saved', round(elapsed()), flush=True)
    A = np.concatenate(parts)
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.median(v)), float(np.percentile(v, 97.5))]
    out = {kk: {m: q(A[:, j, i]) for i, m in enumerate(M)} for j, kk in enumerate(keys)}
    for j, kk in enumerate(keys[1:], start=1):
        out[f'diff_model_minus_{kk}'] = {m: q(A[:, 0, i] - A[:, j, i]) for i, m in enumerate(M)}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--mode', required=True, choices=list(MODES)); ap.add_argument('--boot', type=int, default=200)
    a = ap.parse_args(); mode = a.mode; cfg = MODES[mode]
    lib.HOLDOUT_GUARD_MAX_ENTRY_YEAR = cfg['guard']   # 🔒 holdout/future 모드에서만 가드가 풀린다(평가자 실행용)
    FC = T1/'final_cache'/mode; FC.mkdir(parents=True, exist_ok=True)
    print(f'[{mode}] eval_cut={cfg["eval_cut"]} train_cuts={cfg["train_cuts"]} cap={cfg["cap"]}', flush=True)
    P = clf_base(cfg, FC); P51, PO = clf_os51(cfg, FC); Pins, ins_idx = clf_ins(cfg, FC); ep = ep_match(cfg, FC); cdf = lag_cdf(cfg)
    COLS = list(SPEC['cols'])
    keep = list(dict.fromkeys(['firm', 'k', 'y', 'newc', 'cut', 'lsize'] + BASE + [c for c in COLS if c not in ('k_prior', 'k_prior_new', 'f_any')]))
    def slim(d):   # 메모리(프로세스당 4GB) 보호: 필요한 열만, float32
        d = d[keep].copy()
        for c in d.columns:
            if d[c].dtype == np.float64: d[c] = d[c].astype(np.float32)
        return d
    trs = [slim(cut_features(c, cfg['cap'], cfg, FC, P, P51, PO, Pins, ins_idx, ep, cdf)) for c in cfg['train_cuts']]
    te = slim(cut_features(cfg['eval_cut'], None, cfg, FC, P, P51, PO, Pins, ins_idx, ep, cdf))
    del P, P51, PO, Pins
    tr = pd.concat(trs, ignore_index=True); del trs
    w = ipw(tr, cfg)
    if 'k_prior' in COLS: tr, te = k_prior(tr, te)
    if 'f_any' in COLS:
        # 법인 단계: '5년 내 어느 산업이든 첫 진입' 확률(법인 해시 2-fold 교차적합). 레이블은 SPEC['f_any_label'](reg=등록 / pub=등록∪공개 보강)
        from s2lib import firm_stage, pub_label
        trf = tr if SPEC.get('f_any_label', 'reg') == 'reg' else tr.assign(y=np.maximum(tr.y.values, pub_label(tr, cap=cfg['cap'])))
        trf, te = firm_stage(trf, te, [c for c in COLS if c != 'f_any']); tr['f_any'] = trf.f_any.values
    npos_reg = int(tr.y.sum()); tr['y_reg'] = tr.y.values
    if SPEC.get('publab_weight', 0) > 0:
        # 학습 전용 보조 레이블: cap까지 공개된 범위안 출원(법인명 매칭)으로 (cut, cut+5] 진입 → 등록절단으로 음성 처리된 진입을 양성으로
        from s2lib import pub_label
        yp = pub_label(tr, cap=cfg['cap']); add = (yp == 1) & (tr.y.values == 0)
        tr['y'] = np.maximum(tr.y.values, yp).astype(np.int8); w = np.where(add, SPEC['publab_weight'], w)
        print(f'publab: 등록 양성 {npos_reg} + 공개만 양성 {int(add.sum())}', flush=True)
    import xgboost as xgb
    Xte = te[COLS].values.astype(np.float32); pxs = []
    est = 190.0 * len(tr) / 877070 * SPEC['params'].get('n_estimators', 400) / 1200
    for sd in SPEC['seeds']:
        fp = FC/f'xgb_seed{sd}.json'
        if not fp.exists():
            need_time(est, f'XGB seed {sd} (예상 {est:.0f}s)')
            p = dict(tree_method='hist', nthread=2, random_state=sd, subsample=0.8); p.update(SPEC['params'])
            m = xgb.XGBClassifier(**p); m.fit(tr[COLS].values.astype(np.float32), tr.y.values, sample_weight=w); m.save_model(fp)
            print('xgb seed', sd, 'saved', round(elapsed()), flush=True)
        m = xgb.XGBClassifier(); m.load_model(fp); pxs.append(m.predict_proba(Xte)[:, 1])
    px = np.mean(pxs, 0)
    fl = FC/'logit.pkl'
    if not fl.exists():
        need_time(120, '로지스틱'); pickle.dump(logit_fit(tr, COLS), open(fl, 'wb'))
    pl = logit_pred(pickle.load(open(fl, 'rb')), te)
    s = SPEC['blend_w_xgb'] * rank(px) + (1 - SPEC['blend_w_xgb']) * rank(pl)
    te['score'] = s; te['px'] = px
    K = load()['K']
    if mode == 'future':
        fm = pd.read_parquet(CACHE/'firms.parquet').set_index('applicant_cd').applicant_name
        nm = {}
        pk = B/'공모전데이터/외부데이터/KSIC해설/ksic50_definitions.json'
        if pk.exists():
            try:
                js = json.load(open(pk, encoding='utf-8'))
                nm = {str(k2): (v.get('name') if isinstance(v, dict) else str(v)) for k2, v in (js.items() if isinstance(js, dict) else [])}
            except Exception: nm = {}
        te['ksic'] = [K[i] for i in te.k.values]; te['firm_name'] = te.firm.map(fm)
        g = te.groupby('ksic').agg(at_risk=('px', 'size'), waiting_room_index=('px', 'sum'), index_newcomer=('px', lambda x: x[te.loc[x.index, 'newc']].sum()),
                                  mean_prob=('px', 'mean'), n_high=('score', lambda x: int((x >= 0.99).sum())))
        g['index_share'] = g.waiting_room_index / g.waiting_room_index.sum()
        top = te.sort_values('score', ascending=False).groupby('ksic').head(5).groupby('ksic').firm_name.apply(lambda x: ' | '.join(map(str, x)))
        g['top5_candidates'] = top; g['ksic_name'] = [nm.get(k2, '') for k2 in g.index]
        g['window'] = f'{cfg["eval_cut"] + 1}-{cfg["eval_cut"] + 5}'; g['info_cutoff'] = f'{cfg["eval_cut"]}-12-31 등록·공개분'
        g.sort_values('waiting_room_index', ascending=False).reset_index().to_csv(T1/'future_forecast.csv', index=False, encoding='utf-8-sig')
        cand = te.sort_values('score', ascending=False).groupby('ksic').head(50)[['ksic', 'firm', 'firm_name', 'newc', 'score', 'px']]
        cand.to_csv(T1/'future_candidates_top50.csv', index=False, encoding='utf-8-sig')
        print('saved future_forecast.csv', len(g), 'industries; candidates', len(cand)); return
    # ---- 평가(dryrun: 검증 / holdout: 🔒 1회) ----
    mm = full_metrics(te, s); mx = full_metrics(te, px)
    # 동일정보(엄격) 기준선: 고정 기준선 피처를 같은 엄격 패널에서, 로지스틱 스태킹
    trb = tr.assign(y=tr['y_reg']) if 'y_reg' in tr else tr
    Ls = logit_fit(trb, BASE); sb_strict = logit_pred(Ls, te); mbs = full_metrics(te, sb_strict)
    # 고정 기준선(트랙 정의, 출원 기준 정보 → 기준선에 유리): 로지스틱, 학습원점 스태킹과 마지막 원점 단일 중 지표별 최댓값(강화)
    btr = pd.concat([base_app_panel(c, cfg['cap'], FC) for c in cfg['train_cuts']], ignore_index=True); bte = base_app_panel(cfg['eval_cut'], None, FC)
    assert (bte.firm.values == te.firm.values).all() and (bte.k.values == te.k.values).all() and (bte.y.values == te.y.values).all()
    bte = bte.assign(lsize_vis=te.lsize.values)
    sb_stack = logit_pred(logit_fit(btr, BASE), bte); sb_single = logit_pred(logit_fit(btr[btr.cut == max(cfg['train_cuts'])], BASE), bte)
    tmp = bte.drop(columns=['lsize']).rename(columns={'lsize_vis': 'lsize'})
    m_stack = full_metrics(tmp, sb_stack); m_single = full_metrics(tmp, sb_single)
    fixed = {k2: max(m_stack[k2], m_single[k2]) for k2 in m_stack}
    sk = lambda m_, b_: (m_ - b_) / (1 - b_)
    res = dict(mode=mode, config={k2: (list(v) if isinstance(v, tuple) else v) for k2, v in cfg.items()},
               n=dict(rows=int(len(te)), firms=int(te.firm.nunique()), pos=int(te.y.sum()), new_rows=int(te.newc.sum()), new_pos=int(te[te.newc].y.sum()),
                      train_rows=int(len(tr)), train_pos=int(tr.y.sum())),
               model=mm, model_xgb_only=mx, baseline_fixed_strengthened=fixed, baseline_fixed_stack=m_stack, baseline_fixed_single=m_single, baseline_same_info_strict=mbs,
               skill=dict(auc_all_vs_fixed=sk(mm['auc_all'], fixed['auc_all']), auc_new_vs_fixed=sk(mm['auc_new'], fixed['auc_new']),
                          auc_all_vs_strict=sk(mm['auc_all'], mbs['auc_all']), auc_new_vs_strict=sk(mm['auc_new'], mbs['auc_new']),
                          pr_new_ratio_vs_fixed=mm['pr_new'] / fixed['pr_new'], pr_all_ratio_vs_fixed=mm['pr_all'] / fixed['pr_all'],
                          pr_new_ratio_vs_strict=mm['pr_new'] / mbs['pr_new']),
               notes='평가 단위=(법인×KSIC) 위험집합 쌍, 군집=법인(부트스트랩 재표집 단위). skill=(AUC_m-AUC_b)/(1-AUC_b). 고정 기준선은 출원 기준 정보(미래 등록분 포함)라 모형에 불리한 비교.')
    print(f'[{mode}] model', json.dumps({k2: round(v, 4) for k2, v in mm.items()}, ensure_ascii=False))
    print(f'[{mode}] fixed baseline', json.dumps({k2: round(v, 4) for k2, v in fixed.items()}))
    print(f'[{mode}] strict baseline', json.dumps({k2: round(v, 4) for k2, v in mbs.items()}))
    print(f'[{mode}] skill', json.dumps({k2: round(v, 4) for k2, v in res['skill'].items()}), flush=True)
    if a.boot > 0:
        res['bootstrap_ci_firm_cluster'] = boot(te.y.values, te.newc.values, te.firm.values, {'model': s, 'baseline_fixed_stack': sb_stack, 'baseline_same_info_strict': sb_strict}, FC, R=a.boot)
        res['bootstrap_R'] = a.boot
    out = T1/('holdout_result.json' if mode == 'holdout' else 'dryrun_result.json')
    json.dump(res, open(out, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    te[['firm', 'k', 'y', 'newc', 'score', 'px']].assign(sb_strict=sb_strict, sb_fixed=sb_stack).to_parquet(FC/'eval_scores.parquet', index=False)
    print('saved', out, round(elapsed()))


if __name__ == '__main__':
    main()
