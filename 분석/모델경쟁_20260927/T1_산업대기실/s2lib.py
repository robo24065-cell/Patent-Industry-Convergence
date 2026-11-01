# 2단계(감사·개선) 공용 함수 — 엄격 정보집합 전용. 🔒 홀드아웃 결과는 여기서 계산하지 않는다(build_panel 가드 유지).
import os
os.environ.setdefault('T1_STRICT', '1')
assert os.environ['T1_STRICT'] == '1'
from data8 import *

PAF2 = ['pa_k', 'pa_n', 'pa_rec2', 'pa_k_rec2', 'pa_pend_n', 'pa_share', 'pa_k_pend']


def add_pa2(d, cap=None):
    """공개 출원 피처(감사 수정판). v27~v31의 pa_unreg_n은 스냅샷 등록여부(reg=2025년 기준 '끝내 미등록')를 써서
    미래 정보 누수였다 → '기준시점 연말 현재 아직 미등록(심사 중·거절·취하 구분 불가)'로 교체.
    pa_k_pend: 그 KSIC에서 기준시점 현재 미등록 상태인 공개출원 수(가시 정보)."""
    a = pd.read_parquet(CACHE/'app_pub.parquet'); K = load()['K']; kidx = {k: i for i, k in enumerate(K)}
    a['k'] = a.ksic.map(kidx); a['pub_date'] = a.pub_date.fillna('').astype(str); a['rgst_date'] = a.rgst_date.fillna('').astype(str)
    a = a[a.ay >= 2000]
    out = []
    for cut, g in d.groupby('cut', sort=False):
        ce = f'{cut}1231'; c2 = f'{cut - 2}1231'
        regd = (a.rgst_date.str.len() == 8) & (a.rgst_date <= ce)
        vis = ((a.pub_date.str.len() == 8) & (a.pub_date <= ce)) | regd
        e = a[vis & (a.ay <= cut)].copy(); e['pend'] = ~regd[vis & (a.ay <= cut)].values
        vd = np.where((e.pub_date.str.len() == 8), e.pub_date, e.rgst_date); e['rec'] = vd > c2
        ck = e.groupby(['applicant_cd', 'k']).size().rename('ck'); ckr = e[e.rec].groupby(['applicant_cd', 'k']).size().rename('ckr')
        ckp = e[e.pend].groupby(['applicant_cd', 'k']).size().rename('ckp')
        n = e.groupby('applicant_cd').size(); nr = e[e.rec].groupby('applicant_cd').size(); nu = e[e.pend].groupby('applicant_cd').size()
        x = g[['firm', 'k']].reset_index()
        for s_ in (ck, ckr, ckp):
            x = x.merge(s_.reset_index(), left_on=['firm', 'k'], right_on=['applicant_cd', 'k'], how='left').drop(columns='applicant_cd')
        x = x.fillna(0).set_index('index')
        nn = g.firm.map(n).fillna(0).values
        x['pa_k'] = np.log1p(x.ck); x['pa_k_rec2'] = np.log1p(x.ckr); x['pa_n'] = np.log1p(nn); x['pa_k_pend'] = np.log1p(x.ckp)
        x['pa_rec2'] = np.log1p(g.firm.map(nr).fillna(0).values); x['pa_pend_n'] = np.log1p(g.firm.map(nu).fillna(0).values)
        x['pa_share'] = x.ck / np.maximum(nn, 1)
        out.append(x[PAF2])
    o = pd.concat(out).loc[d.index]
    for c in PAF2: d[c] = o[c].values
    return d


def load_s2(cuts=(2001, 2002, 2003, 2004)):
    tr, te = load19(cuts)
    tr = add_pa2(tr); te = add_pa2(te)
    tr, te = add_rr(tr, te, cuts); tr, te = add_now(tr, te, cuts)
    return tr, te


DART = ['dm2', 'dm3', 'dm4', 'has_dart', 'ind2']
C31 = C19 + PAF + RRF + NOWF
C32 = C19 + PAF2 + RRF + NOWF


def k_prior(tr, te):
    """산업 진입 성향(타깃 인코딩): 학습 레이블(등록≤학습 cap)만으로. 학습 행은 자기 원점을 뺀 나머지 원점으로(leave-one-cut-out)."""
    cuts = sorted(tr.cut.unique()); glob = tr.y.mean()
    def enc(sub):
        g = sub.groupby('k').y.agg(['sum', 'size']); return ((g['sum'] + 20 * glob) / (g['size'] + 20)).reindex(range(50)).fillna(glob)
    ktr = np.zeros(len(tr))
    for c in cuts:
        m = tr.cut.values == c; e = enc(tr[~m]); ktr[m] = np.log(e.values[tr.k.values[m]])
    e = enc(tr); kte = np.log(e.values[te.k.values])
    # 신규진입자 전용 인코딩도
    ntr = np.zeros(len(tr)); glob_n = tr[tr.newc].y.mean()
    def encn(sub):
        s = sub[sub.newc]; g = s.groupby('k').y.agg(['sum', 'size']); return ((g['sum'] + 20 * glob_n) / (g['size'] + 20)).reindex(range(50)).fillna(glob_n)
    for c in cuts:
        m = tr.cut.values == c; e = encn(tr[~m]); ntr[m] = np.log(e.values[tr.k.values[m]])
    nte = np.log(encn(tr).values[te.k.values])
    tr['k_prior'] = ktr; te['k_prior'] = kte; tr['k_prior_new'] = ntr; te['k_prior_new'] = nte
    return tr, te


def pub_label(tr, cap='20091231', h=5):
    """공개 기준 보조 레이블: (cut, cut+h] 출원 범위안 공개출원(cap까지 공개·등록된 것, 법인 이름 매칭)이 그 KSIC에 있으면 1.
    학습 전용(검증·홀드아웃 평가 레이블은 등록 기준 그대로). cap 이전에 관측 완료된 정보만 사용."""
    a = pd.read_parquet(CACHE/'app_pub.parquet'); K = load()['K']; kidx = {k: i for i, k in enumerate(K)}
    a['k'] = a.ksic.map(kidx); a['pub_date'] = a.pub_date.fillna('').astype(str); a['rgst_date'] = a.rgst_date.fillna('').astype(str)
    vis = ((a.pub_date.str.len() == 8) & (a.pub_date <= cap)) | ((a.rgst_date.str.len() == 8) & (a.rgst_date <= cap))
    a = a[vis]
    yp = np.zeros(len(tr), np.int8)
    for cut in sorted(tr.cut.unique()):
        e = a[(a.ay > cut) & (a.ay <= cut + h)].drop_duplicates(['applicant_cd', 'k'])[['applicant_cd', 'k']].assign(z=1)
        idx = np.nonzero(tr.cut.values == cut)[0]
        x = tr.iloc[idx][['firm', 'k']].merge(e, left_on=['firm', 'k'], right_on=['applicant_cd', 'k'], how='left')
        yp[idx] = x.z.fillna(0).values.astype(np.int8)
    return yp


def firm_stage(tr, te, cols, seed=0):
    """2단계 분해용 법인 수준 피처: '5년 안에 어느 산업이든 첫 진입' 확률(법인 수준 XGB, 학습은 원점별 법인 행, 학습 행은 법인 해시 2-fold 교차적합)"""
    import xgboost as xgb
    fcols = [c for c in cols if c in FIRM_LEVEL]
    def firm_tab(d):
        g = d.groupby(['cut', 'firm'], sort=False)
        X = g[fcols].first(); X['n_risk'] = g.size(); X['y'] = g.y.max()
        for c in ('t_max', 'ot_max', 't_top3', 'pa_k', 'ep_k', 'ti_max'):
            if c in d: X[c + '_fmax'] = g[c].max()
        return X
    Ftr = firm_tab(tr); Fte = firm_tab(te); xc = [c for c in Ftr.columns if c != 'y']
    p = dict(n_estimators=500, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.7, min_child_weight=10, tree_method='hist', nthread=2, random_state=seed)
    fold = (pd.util.hash_array(Ftr.index.get_level_values('firm').values.astype(object)) % 2).astype(int)
    otr = np.zeros(len(Ftr))
    for fo in (0, 1):
        m = xgb.XGBClassifier(**p).fit(Ftr[xc].values[fold != fo].astype(np.float32), Ftr.y.values[fold != fo])
        otr[fold == fo] = m.predict_proba(Ftr[xc].values[fold == fo].astype(np.float32))[:, 1]
    m = xgb.XGBClassifier(**p).fit(Ftr[xc].values.astype(np.float32), Ftr.y.values)
    ote = m.predict_proba(Fte[xc].values.astype(np.float32))[:, 1]
    mtr = pd.Series(otr, index=Ftr.index); mte = pd.Series(ote, index=Fte.index)
    tr['f_any'] = mtr.reindex(pd.MultiIndex.from_arrays([tr.cut.values, tr.firm.values])).values
    te['f_any'] = mte.reindex(pd.MultiIndex.from_arrays([te.cut.values, te.firm.values])).values
    return tr, te


FIRM_LEVEL = ['lsize', 'lin', 'lout', 'inshare', 'has_dart', 't_ent', 't_lrec_n', 't_grow', 'p_n', 'ep_n', 'gap_last', 'n_last1',
              'pa_n', 'pa_rec2', 'pa_pend_n', 'r_n1', 'r_n2', 'r_in2', 'now1', 'now3', 'now_grow', 'now_in3', 'vis_frac', 'oin50']
