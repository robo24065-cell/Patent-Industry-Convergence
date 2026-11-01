# v8 피처셋 결합 패널 로더(재사용)
from exp import *
OS = ['ot_mean', 'ot_max', 'ot_top3', 'ot_hi', 'ot_rw', 'ot_nbr', 'oin50']
C8 = BASE + AGG + MOM + NBR + KMOM + OS + PART
def comb(cut, cap):
    a = pd.read_parquet(PC/f'panel_c{cut}_{cap}_c2009_base_.parquet')
    b = pd.read_parquet(PC/f'panel_c{cut}_{cap}_c2009_os51_.parquet')
    for c in ['t_mean', 't_max', 't_top3', 't_hi', 't_rw', 't_nbr', 'in50']: a['o' + c] = b[c].values
    p = pd.read_parquet(PC/f'part_c{cut}_{cap}.parquet')
    for c in p.columns: a[c] = p[c].values
    return a
def load8(cuts=(2002, 2003, 2004)):
    tr = pd.concat([comb(c, CAP_TR) for c in cuts], ignore_index=True); te = comb(2009, 'None')
    return tr, te

EPF = ['ep_k', 'ep_n', 'ep_share', 'ep_k_rec', 'ep_krprio']
def add_ep(d):
    """해외 선행(EP): 기준시점 연말까지 **공개된** EP 출원(2000+) 중 법인 매칭분의 KSIC별 건수"""
    ep = pd.read_parquet(CACHE/'ep_firm.parquet'); K = load()['K']; kidx = {k: i for i, k in enumerate(K)}
    ep['k'] = ep.ksic.astype(str).map(kidx)
    out = []
    for cut, g in d.groupby('cut', sort=False):
        e = ep[(ep.pub_date.astype(str) <= f'{cut}1231') & (ep.appl_year >= 2000) & ep.pub_date.notna()]
        e = e.drop_duplicates(['applicant_cd', 'appl_no', 'k'])
        ck = e.groupby(['applicant_cd', 'k']).size().rename('c').reset_index()
        cr = e[e.appl_year > cut - 3].groupby(['applicant_cd', 'k']).size().rename('cr').reset_index()
        kp = e[e.prio_cc.fillna('').str.contains('KR')].groupby(['applicant_cd', 'k']).size().rename('ckr').reset_index()
        tot = e.drop_duplicates(['applicant_cd', 'appl_no']).groupby('applicant_cd').size()
        x = g[['firm', 'k']].reset_index().merge(ck, left_on=['firm', 'k'], right_on=['applicant_cd', 'k'], how='left').drop(columns='applicant_cd')
        x = x.merge(cr, left_on=['firm', 'k'], right_on=['applicant_cd', 'k'], how='left').drop(columns='applicant_cd')
        x = x.merge(kp, left_on=['firm', 'k'], right_on=['applicant_cd', 'k'], how='left').drop(columns='applicant_cd')
        x = x.fillna(0).set_index('index'); n = g.firm.map(tot).fillna(0).values
        x['ep_k'] = np.log1p(x.c); x['ep_n'] = np.log1p(n); x['ep_share'] = x.c / np.maximum(n, 1)
        x['ep_k_rec'] = np.log1p(x.cr); x['ep_krprio'] = np.log1p(x.ckr)
        out.append(x[EPF])
    o = pd.concat(out).loc[d.index]
    for c in EPF: d[c] = o[c].values
    return d

SIMF = ['sr_mean', 'sr_max', 'sr_top3', 'se_mean', 'se_max', 'se_top3', 'se_rel']
def sim_panel(cut, cap):
    p = PC/f'simf_c{cut}_{cap}.parquet'
    if p.exists(): return pd.read_parquet(p)
    d, c = build_panel(cut, 5, cap=None if cap == 'None' else cap)
    op = c['op']; fi = c['fi']; nF = len(c['firms'])
    rows_all = np.load(CACHE/f'sim_rows_c{cut}.npy'); pos = pd.Series(np.arange(len(rows_all)), index=rows_all)
    ii = pos[op.row.values].values; r = fi[op.applicant_cd].values
    out = {}
    for nm in ('rec', 'ent'):
        S = np.load(CACHE/f'sim_{nm}_c{cut}.npy')[ii].astype(np.float32)
        T = np.zeros((nF, 50), np.float32); np.add.at(T, r, S); n = np.bincount(r, minlength=nF)
        MX = np.zeros((nF, 50), np.float32); np.maximum.at(MX, r, S)
        order = np.argsort(r, kind='stable'); rs = r[order]; Ss = S[order]
        st = np.searchsorted(rs, np.arange(nF)); en = np.searchsorted(rs, np.arange(nF), side='right')
        T3 = np.zeros((nF, 50), np.float32)
        for i in np.nonzero(en > st)[0]:
            b = Ss[st[i]:en[i]]; T3[i] = b.mean(0) if len(b) <= 3 else np.sort(b, 0)[-3:].mean(0)
        out[nm] = (T / np.maximum(n, 1)[:, None], MX, T3)
    f, k = d.fidx.values, d.k.values
    x = pd.DataFrame({'sr_mean': out['rec'][0][f, k], 'sr_max': out['rec'][1][f, k], 'sr_top3': out['rec'][2][f, k],
                      'se_mean': out['ent'][0][f, k], 'se_max': out['ent'][1][f, k], 'se_top3': out['ent'][2][f, k]})
    em = out['ent'][0]; x['se_rel'] = (em - em.mean(1, keepdims=True))[f, k] / np.maximum(em.std(1), 1e-6)[f]
    x.to_parquet(p, index=False); return x
def add_sim(d, cuts):
    xs = [sim_panel(c, CAP_TR) for c in cuts] + [sim_panel(2009, 'None')]
    return xs

SIBF = ['sib4', 'sib3', 'sib2', 'tsib3', 'tsib2']
def sib_panel(cut, cap):
    """KSIC 계층 형제 산업: 법인이 같은 4/3/2자리 KSIC에 가진 범위안 특허 수, 그리고 형제 산업 텍스트 노출 합"""
    p = PC/f'sibf_c{cut}_{cap}.parquet'
    if p.exists(): return pd.read_parquet(p)
    d, c = build_panel(cut, 5, cap=None if cap == 'None' else cap)
    K = load()['K']; M = c['M']
    a = pd.read_parquet(PC/f'panel_c{cut}_{cap}_c2009_base_.parquet', columns=['t_mean'])
    # 법인×KSIC 텍스트 평균 행렬 재구성(위험집합 밖 칸은 0)
    TM = np.zeros_like(M); TM[d.fidx.values, d.k.values] = a.t_mean.values
    x = {}
    for L, nm in [(4, 'sib4'), (3, 'sib3'), (2, 'sib2')]:
        G = np.array([[1.0 if (K[i][:L] == K[j][:L] and i != j) else 0.0 for j in range(50)] for i in range(50)], np.float32)
        x[nm] = np.log1p(M @ G)[d.fidx.values, d.k.values]
        if L in (3, 2): x['t' + nm] = (TM @ G)[d.fidx.values, d.k.values]
    x = pd.DataFrame(x); x.to_parquet(p, index=False); return x

INSF = ['ti_mean', 'ti_max', 'ti_n20', 'ti_lsum']
def ins_panel(cut, cap):
    """법인의 기존 범위안 특허 제목이 가리키는 2차 산업 확률(교차적합)"""
    p = PC/f'insf_c{cut}_{cap}.parquet'
    if p.exists(): return pd.read_parquet(p)
    d, c = build_panel(cut, 5, cap=None if cap == 'None' else cap)
    P = np.load(CACHE/'P_ins_cf.npy'); idx = pd.read_parquet(CACHE/'P_ins_cf_index.parquet').rgst_no
    pos = pd.Series(np.arange(len(idx)), index=idx.values)
    ip = c['ip'].drop_duplicates(['applicant_cd', 'rgst_no'])
    r = c['fi'][ip.applicant_cd].values; S = P[pos[ip.rgst_no].values].astype(np.float32); nF = len(c['firms'])
    T = np.zeros((nF, 50), np.float32); np.add.at(T, r, S); n = np.bincount(r, minlength=nF)
    MX = np.zeros((nF, 50), np.float32); np.maximum.at(MX, r, S)
    N2 = np.zeros((nF, 50), np.float32); np.add.at(N2, r, (S > 0.2).astype(np.float32))
    f, k = d.fidx.values, d.k.values
    x = pd.DataFrame({'ti_mean': (T / np.maximum(n, 1)[:, None])[f, k], 'ti_max': MX[f, k], 'ti_n20': np.log1p(N2[f, k]), 'ti_lsum': np.log1p(T[f, k])})
    x.to_parquet(p, index=False); return x
def load13(cuts=(2002, 2003, 2004)):
    tr, te = load8(cuts); tr = add_ep(tr); te = add_ep(te)
    xtr = pd.concat([sib_panel(c, CAP_TR) for c in cuts], ignore_index=True); xte = sib_panel(2009, 'None')
    for c in SIBF: tr[c] = xtr[c].values; te[c] = xte[c].values
    return tr, te
C13 = C8 + EPF + SIBF

def load14(cuts=(2002, 2003, 2004)):
    tr, te = load13(cuts)
    xtr = pd.concat([ins_panel(c, CAP_TR) for c in cuts], ignore_index=True); xte = ins_panel(2009, 'None')
    for c in INSF: tr[c] = xtr[c].values; te[c] = xte[c].values
    return tr, te
C14 = C13 + INSF

TIMF = ['gap_last', 'gap_first', 'n_last1', 'n_last2', 'ins_rec3', 'age', 'foreign', 'gov', 'act_years']
def tim_panel(cut, cap):
    """법인 활동 시점 피처: 마지막/첫 출원 이후 경과, 최근 1·2년 출원 수, 최근3년 범위안 출원 수, 활동 연수, 설립 경과(DART), 외국법인·국가기관"""
    p = PC/f'timf_c{cut}_{cap}.parquet'
    if p.exists(): return pd.read_parquet(p)
    d, c = build_panel(cut, 5, cap=None if cap == 'None' else cap)
    d = add_dart(d, c)
    allp = pd.concat([c['op'][['applicant_cd', 'ay', 'rgst_no']], c['ip'][['applicant_cd', 'ay', 'rgst_no']]]).drop_duplicates(['applicant_cd', 'rgst_no'])
    g = allp.groupby('applicant_cd').ay
    last = g.max(); first = g.min(); yrs = g.nunique()
    n1 = allp[allp.ay == cut].groupby('applicant_cd').size(); n2 = allp[allp.ay >= cut - 1].groupby('applicant_cd').size()
    ir = c['ip'][c['ip'].ay > cut - 3].drop_duplicates(['applicant_cd', 'rgst_no']).groupby('applicant_cd').size()
    tp = load()['fm'].set_index('applicant_cd').tp
    fr = d.firm
    x = pd.DataFrame({'gap_last': cut - fr.map(last).values, 'gap_first': cut - fr.map(first).values, 'n_last1': np.log1p(fr.map(n1).fillna(0).values),
                      'n_last2': np.log1p(fr.map(n2).fillna(0).values), 'ins_rec3': np.log1p(fr.map(ir).fillna(0).values), 'age': d.age.values,
                      'foreign': (fr.map(tp) == '외국법인').astype(np.int8).values, 'gov': fr.map(tp).isin(['국가기관', '각급시험연구기관']).astype(np.int8).values,
                      'act_years': fr.map(yrs).values})
    x.to_parquet(p, index=False); return x

def load19(cuts=(2002, 2003, 2004)):
    tr, te = load14(cuts)
    xtr = pd.concat([tim_panel(c, CAP_TR) for c in cuts], ignore_index=True); xte = tim_panel(2009, 'None')
    for c in ('gap_last', 'n_last1', 'foreign'): tr[c] = xtr[c].values; te[c] = xte[c].values
    return tr, te
C19 = C14 + ['gap_last', 'n_last1']
def run_std(tr, te, cols, P=None, seeds=3, w=None, key=None, budget=480):
    """시드별 점수를 key 이름으로 캐시 — 9분 제한 안에 못 끝나면 다시 실행해 이어서 계산"""
    import xgboost as xgb
    P = P or json.load(open(T1/'v9_cv.json'))['g4']['params']
    ss = []; _t = time.time()
    for sd in range(seeds):
        fp = PC/f'seed_{key}_{sd}.npy' if key else None
        if fp is not None and fp.exists(): ss.append(np.load(fp)); continue
        if key and time.time() - _t > budget: raise SystemExit(f'[run_std] 시간 예산 초과: seed {sd}부터 다시 실행 필요')
        p = dict(n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, tree_method='hist', nthread=2, random_state=sd); p.update(P)
        m = xgb.XGBClassifier(**p); m.fit(tr[cols].values.astype(np.float32), tr.y.values, sample_weight=w); ss.append(m.predict_proba(te[cols].values.astype(np.float32))[:, 1])
        if fp is not None: np.save(fp, ss[-1])
    sx = np.mean(ss, 0); sl = logit(tr, te, cols)
    return 0.9 * rank(sx) + 0.1 * rank(sl), sx

def ipw_weights(tr, cap='20091231', cohort_max=2002, wmax=5.0):
    """학습 양성 가중 1/π, π = P(등록 ≤ cap | 첫 진입특허 출원일, 외국법인 여부) — 등록지연 CDF는 cohort_max 이하 출원 코호트(cap까지 관측)"""
    D = load(); ins = D['ins']; kidx = D['kidx']; tp = D['fm'].set_index('applicant_cd').tp
    fr_all = ins.applicant_cd.map(tp).eq('외국법인')
    coh = ins[(ins.ay <= cohort_max) & (ins.rgst_date <= cap)].assign(fr=fr_all).drop_duplicates('rgst_no')
    lag = (pd.to_datetime(coh.rgst_date) - pd.to_datetime(coh.appl_date)).dt.days.values
    cdf = {g: np.sort(lag[coh.fr.values == g]) for g in (False, True)}
    w = np.ones(len(tr)); capd = pd.Timestamp(cap)
    for cut in sorted(tr.cut.unique()):
        fut = ins[(ins.ay > cut) & (ins.ay <= cut + 5) & (ins.rgst_date <= cap)]
        fst = fut.groupby(['applicant_cd', 'ksic']).appl_date.min().reset_index(); fst['k'] = fst.ksic.map(kidx)
        idx = np.nonzero((tr.cut.values == cut) & (tr.y.values == 1))[0]
        x = tr.iloc[idx][['firm', 'k']].merge(fst, left_on=['firm', 'k'], right_on=['applicant_cd', 'k'], how='left')
        days = (capd - pd.to_datetime(x.appl_date)).dt.days.values; fr = x.firm.map(tp).eq('외국법인').values
        p = np.array([np.searchsorted(cdf[g], dd, side='right') / len(cdf[g]) for dd, g in zip(days, fr)])
        w[idx] = np.minimum(1 / np.maximum(p, 1e-3), wmax)
    return w

PAF = ['pa_k', 'pa_n', 'pa_rec2', 'pa_k_rec2', 'pa_unreg_n', 'pa_share']
def add_pa(d):
    """공개 출원(범위안, 미등록 포함) — 기준시점 연말까지 공개(또는 등록)된 법인 출원만. 모든 모드에서 공개정보."""
    a = pd.read_parquet(CACHE/'app_pub.parquet'); K = load()['K']; kidx = {k: i for i, k in enumerate(K)}
    a['k'] = a.ksic.map(kidx); a['pub_date'] = a.pub_date.fillna('').astype(str); a['rgst_date'] = a.rgst_date.fillna('').astype(str)
    a = a[a.ay >= 2000]
    out = []
    for cut, g in d.groupby('cut', sort=False):
        ce = f'{cut}1231'; c2 = f'{cut - 2}1231'
        vis = ((a.pub_date.str.len() == 8) & (a.pub_date <= ce)) | ((a.rgst_date.str.len() == 8) & (a.rgst_date <= ce))
        e = a[vis & (a.ay <= cut)].copy()
        vd = np.where((e.pub_date.str.len() == 8), e.pub_date, e.rgst_date); e['rec'] = vd > c2
        ck = e.groupby(['applicant_cd', 'k']).size().rename('ck'); ckr = e[e.rec].groupby(['applicant_cd', 'k']).size().rename('ckr')
        n = e.groupby('applicant_cd').size(); nr = e[e.rec].groupby('applicant_cd').size(); nu = e[~e.reg].groupby('applicant_cd').size()
        x = g[['firm', 'k']].reset_index()
        x = x.merge(ck.reset_index(), left_on=['firm', 'k'], right_on=['applicant_cd', 'k'], how='left').drop(columns='applicant_cd')
        x = x.merge(ckr.reset_index(), left_on=['firm', 'k'], right_on=['applicant_cd', 'k'], how='left').drop(columns='applicant_cd').fillna(0).set_index('index')
        nn = g.firm.map(n).fillna(0).values
        x['pa_k'] = np.log1p(x.ck); x['pa_k_rec2'] = np.log1p(x.ckr); x['pa_n'] = np.log1p(nn)
        x['pa_rec2'] = np.log1p(g.firm.map(nr).fillna(0).values); x['pa_unreg_n'] = np.log1p(g.firm.map(nu).fillna(0).values)
        x['pa_share'] = x.ck / np.maximum(nn, 1)
        out.append(x[PAF])
    o = pd.concat(out).loc[d.index]
    for c in PAF: d[c] = o[c].values
    return d

RRF = ['r_n1', 'r_n2', 'r_in2', 't_rrw', 't_rrec2', 't_rrec2_max']
def rr_panel(cut, cap):
    """등록일 기준 최근성(엄격 모드에서 출원연도 기반 최근성은 절단으로 왜곡됨): 최근 1·2년 등록 수, 등록연도 반감기 가중 노출, 최근2년 등록분 노출"""
    p = PC/f'rrf_c{cut}_{cap}.parquet'
    if p.exists(): return pd.read_parquet(p)
    d, c = build_panel(cut, 5, cap=None if cap == 'None' else cap)
    P = train_clf(CAP_TR, 'c2009_base'); op = c['op']; ip = c['ip'].drop_duplicates(['applicant_cd', 'rgst_no']); fi = c['fi']; nF = len(c['firms'])
    ry = pd.to_numeric(op.rgst_date.str[:4]).values; r = fi[op.applicant_cd].values; Po = P[op.row.values]
    ryi = pd.to_numeric(ip.rgst_date.str[:4]).values; ri = fi[ip.applicant_cd].values
    n1 = np.bincount(r[ry >= cut], minlength=nF) + np.bincount(ri[ryi >= cut], minlength=nF)
    n2 = np.bincount(r[ry >= cut - 1], minlength=nF) + np.bincount(ri[ryi >= cut - 1], minlength=nF)
    in2 = np.bincount(ri[ryi >= cut - 1], minlength=nF)
    w = np.power(0.5, (cut - ry) / 2.0).astype(np.float32)
    TW = np.zeros((nF, 50), np.float32); np.add.at(TW, r, Po * w[:, None]); W = np.bincount(r, weights=w, minlength=nF)
    m2 = ry >= cut - 1
    T2 = np.zeros((nF, 50), np.float32); np.add.at(T2, r[m2], Po[m2]); N2 = np.bincount(r[m2], minlength=nF)
    MX = np.zeros((nF, 50), np.float32); np.maximum.at(MX, r[m2], Po[m2])
    f, k = d.fidx.values, d.k.values
    x = pd.DataFrame({'r_n1': np.log1p(n1[f]), 'r_n2': np.log1p(n2[f]), 'r_in2': np.log1p(in2[f]), 't_rrw': (TW / np.maximum(W, 1e-9)[:, None])[f, k],
                      't_rrec2': (T2 / np.maximum(N2, 1)[:, None])[f, k], 't_rrec2_max': MX[f, k]})
    x.to_parquet(p, index=False); return x
def add_rr(tr, te, cuts):
    xtr = pd.concat([rr_panel(c, CAP_TR) for c in cuts], ignore_index=True); xte = rr_panel(2009, 'None')
    for c in RRF: tr[c] = xtr[c].values; te[c] = xte[c].values
    return tr, te

NOWF = ['now1', 'now3', 'now_grow', 'now_in3', 'vis_frac']
_cdf = {}
def lag_cdf():
    if _cdf: return _cdf
    D = load(); ins = D['ins']; tp = D['fm'].set_index('applicant_cd').tp
    coh = ins[(ins.ay <= 2002) & (ins.rgst_date <= '20091231')].drop_duplicates('rgst_no')
    fr = coh.applicant_cd.map(tp).eq('외국법인').values
    lag = (pd.to_datetime(coh.rgst_date) - pd.to_datetime(coh.appl_date)).dt.days.values
    for g in (False, True): _cdf[g] = np.sort(lag[fr == g])
    return _cdf
def now_panel(cut, cap):
    """나우캐스팅: 최근 출원연도별 가시(등록) 건수를 등록완료확률 π(경과일)로 나눠 실제 출원량 추정(엄격 모드의 절단 보정)"""
    p = PC/f'nowf_c{cut}_{cap}.parquet'
    if p.exists(): return pd.read_parquet(p)
    d, c = build_panel(cut, 5, cap=None if cap == 'None' else cap)
    cdf = lag_cdf(); tp = load()['fm'].set_index('applicant_cd').tp
    ce = pd.Timestamp(f'{cut}-12-31')
    allp = pd.concat([c['op'][['applicant_cd', 'rgst_no', 'appl_date', 'ay']].assign(ins=0), c['ip'][['applicant_cd', 'rgst_no', 'appl_date', 'ay']].assign(ins=1)]).drop_duplicates(['applicant_cd', 'rgst_no'])
    days = (ce - pd.to_datetime(allp.appl_date.astype(str), format='%Y%m%d', errors='coerce')).dt.days.fillna(3650).values
    fr = allp.applicant_cd.map(tp).eq('외국법인').values
    pi = np.where(fr, np.searchsorted(cdf[True], days, side='right') / len(cdf[True]), np.searchsorted(cdf[False], days, side='right') / len(cdf[False]))
    allp['w'] = 1 / np.maximum(pi, 0.05); allp['pi'] = pi
    g = allp.groupby('applicant_cd')
    n1 = allp[allp.ay == cut].groupby('applicant_cd').w.sum(); n3 = allp[allp.ay > cut - 3].groupby('applicant_cd').w.sum()
    p3 = allp[(allp.ay > cut - 6) & (allp.ay <= cut - 3)].groupby('applicant_cd').w.sum()
    i3 = allp[(allp.ay > cut - 3) & (allp.ins == 1)].groupby('applicant_cd').w.sum()
    vf = g.pi.mean()
    fr_ = d.firm
    x = pd.DataFrame({'now1': np.log1p(fr_.map(n1).fillna(0).values), 'now3': np.log1p(fr_.map(n3).fillna(0).values),
                      'now_in3': np.log1p(fr_.map(i3).fillna(0).values), 'vis_frac': fr_.map(vf).fillna(0).values})
    x['now_grow'] = x.now3 - np.log1p(fr_.map(p3).fillna(0).values)
    x.to_parquet(p, index=False); return x
def add_now(tr, te, cuts):
    xtr = pd.concat([now_panel(c, CAP_TR) for c in cuts], ignore_index=True); xte = now_panel(2009, 'None')
    for c in NOWF: tr[c] = xtr[c].values; te[c] = xte[c].values
    return tr, te
