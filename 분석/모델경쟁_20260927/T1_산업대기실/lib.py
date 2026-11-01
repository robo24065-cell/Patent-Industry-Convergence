# T1 산업 대기실 — 패널 구성/분류기/평가 공용 라이브러리
# 설계(검증): 평가 원점 c=2009 (피처 2000~2009 출원, 진입 2010~2014 출원)
#            학습 원점 c=2004 (피처 2000~2004, 진입 2005~2009), 학습 패널은 등록일 <= 평가원점 연말(2009-12-31)인 특허만 보이게(cap)
# 🔒 홀드아웃(c=2014 → 2015~2019)의 결과변수는 이 파일 어디에서도 계산하지 않는다. build_panel 은 h 창 끝이 2014 를 넘으면 거부한다.
import pandas as pd, numpy as np, time, json, os, datetime
from common import *

HOLDOUT_GUARD_MAX_ENTRY_YEAR = 2014   # 1단계에서 볼 수 있는 진입 연도 상한
# 엄격 모드(T1_STRICT=1): 모든 패널의 피처를 '기준시점 연말까지 등록·공개된 특허'로만 계산. 패널 캐시는 cache/strict/
STRICT = os.environ.get('T1_STRICT', '0') == '1'
VIS = os.environ.get('T1_VIS', 'rgst')   # 'rgst': 가시일=등록일 / 'pub': 가시일=min(공개일, 등록일)
PC = (CACHE / ('strict' if VIS == 'rgst' else 'public')) if STRICT else CACHE
PC.mkdir(exist_ok=True)

_cache = {}
def load():
    if 'ins' in _cache: return _cache
    ins = pd.read_parquet(CACHE/'ins_firm.parquet')
    ins = ins[ins.ay >= 2000]
    oos = pd.read_parquet(CACHE/'oos.parquet')
    oos = oos[oos.self & (oos.ay >= 2000)].drop(columns=['self']).reset_index(drop=True)
    oos['row'] = np.arange(len(oos))
    K = sorted(pd.read_parquet(CACHE/'ins_pat.parquet', columns=['ksic']).ksic.unique())
    def _vis(r, pbd):
        pbd = pd.Series(pbd).fillna('').astype(str).str.strip().values
        ok = (np.char.str_len(pbd.astype(str)) == 8)
        return np.where(ok & (pbd < r), pbd, r)
    if VIS == 'pub':
        # right_list의 pubDate는 등록공고일이라 조기공개일이 아님 → 조기공개 = 출원일 + 18개월(법정 공개시점)로 근사
        def p18(ad):
            d = pd.to_datetime(pd.Series(ad).astype(str), format='%Y%m%d', errors='coerce') + pd.DateOffset(months=18)
            return d.dt.strftime('%Y%m%d').fillna('99999999').values
        oos['vis'] = _vis(oos.rgst_date.values.astype(str), p18(oos.appl_date.values))
        kp = pd.read_parquet(KR_REG, columns=['등록번호', '공개일자']); km = dict(zip(kp['등록번호'].astype(str), kp['공개일자'].astype(str)))
        v1 = _vis(ins.rgst_date.values.astype(str), ins.rgst_no.map(km).values)
        ins = ins.assign(vis=_vis(v1, p18(ins.appl_date.values)))
    else:
        oos['vis'] = oos.rgst_date.values; ins = ins.assign(vis=ins.rgst_date.values)
    fm = pd.read_parquet(CACHE/'firm_meta.parquet')
    _cache.update(ins=ins, oos=oos, K=K, kidx={k: i for i, k in enumerate(K)}, fm=fm)
    return _cache

def dart_map():
    p = CACHE/'dart_map.parquet'
    if p.exists(): return pd.read_parquet(p)
    ca = pd.read_parquet(B/'공모전데이터/외부데이터/DART/corp_applicant.parquet')
    ca = ca.rename(columns={'출원인코드': 'applicant_cd'}); ca['j'] = ca.jurir.fillna('').astype(str).str.strip()
    dart = pd.read_sql('select jurir_no, induty_code, est_dt, corp_cls from company', DT())
    dart = dart[dart.jurir_no.notna() & (dart.jurir_no.str.len() > 5)]
    m = ca.merge(dart, left_on='j', right_on='jurir_no')[['applicant_cd','induty_code','est_dt','corp_cls']].drop_duplicates('applicant_cd')
    m.to_parquet(p, index=False)
    return m

def build_panel(cut, h=5, cap=None, min_pat=3, fcap=None):
    if STRICT and fcap is None: fcap = f'{cut}1231'
    """(법인 x KSIC) 위험집합 패널. cap: 'YYYYMMDD' 이 날짜까지 등록된 특허만 보이게(학습 패널용)."""
    assert cut + h <= HOLDOUT_GUARD_MAX_ENTRY_YEAR, '🔒 홀드아웃 진입창 접근 금지'
    D = load(); ins, oos, K, kidx = D['ins'], D['oos'], D['K'], D['kidx']
    if cap is not None:
        ins = ins[ins.rgst_date <= cap]; oos = oos[oos.rgst_date <= cap]
    ip = ins[ins.ay <= cut]; op = oos[oos.ay <= cut]
    cnt = pd.concat([ip.drop_duplicates(['applicant_cd','rgst_no']).applicant_cd, op.drop_duplicates(['applicant_cd','rgst_no']).applicant_cd]).value_counts()
    firms = np.array(sorted(cnt[cnt >= min_pat].index))
    fi = pd.Series(np.arange(len(firms)), index=firms); nF = len(firms)
    ip = ip[ip.applicant_cd.isin(fi.index)]; op = op[op.applicant_cd.isin(fi.index)]
    M = np.zeros((nF, 50), np.float32)
    np.add.at(M, (fi[ip.applicant_cd].values, ip.ksic.map(kidx).values), 1)
    # 산업 근접도(기준시점까지 모든 법인 범위안 포트폴리오의 산업 동시보유)
    allp = ins[ins.ay <= cut]
    A = pd.crosstab(allp.applicant_cd, allp.ksic).reindex(columns=K, fill_value=0).values > 0
    A = A.astype(np.float32); co = A.T @ A; d = np.diag(co).copy()
    prox = co / np.maximum(np.maximum.outer(d, d), 1); np.fill_diagonal(prox, 0)
    Bm = (M > 0).astype(np.float32)
    dens = (Bm @ prox) / np.maximum(prox.sum(0), 1e-9)
    kpop = np.log1p(A.sum(0))
    # 결과: (cut, cut+h] 출원 범위안 특허
    fut = ins[(ins.ay > cut) & (ins.ay <= cut + h) & ins.applicant_cd.isin(fi.index)]
    Y = np.zeros((nF, 50), np.int8); NF = np.zeros((nF, 50), np.int16)
    np.add.at(NF, (fi[fut.applicant_cd].values, fut.ksic.map(kidx).values), 1)
    Y[NF > 0] = 1
    Mrisk = M
    if fcap is not None:
        # 엄격 모드: 위험집합·결과·법인 모집단은 그대로, 피처는 fcap(기준시점 연말)까지 '등록되어 공개된' 특허만
        ins_f = ins[ins.vis <= fcap]; oos_f = oos[oos.vis <= fcap]
        ip = ins_f[(ins_f.ay <= cut) & ins_f.applicant_cd.isin(fi.index)]; op = oos_f[(oos_f.ay <= cut) & oos_f.applicant_cd.isin(fi.index)]
        M = np.zeros((nF, 50), np.float32); np.add.at(M, (fi[ip.applicant_cd].values, ip.ksic.map(kidx).values), 1)
        allp = ins_f[ins_f.ay <= cut]
        A = pd.crosstab(allp.applicant_cd, allp.ksic).reindex(columns=K, fill_value=0).values > 0
        A = A.astype(np.float32); co = A.T @ A; d = np.diag(co).copy()
        prox = co / np.maximum(np.maximum.outer(d, d), 1); np.fill_diagonal(prox, 0)
        dens = ((M > 0).astype(np.float32) @ prox) / np.maximum(prox.sum(0), 1e-9); kpop = np.log1p(A.sum(0))
        cap = fcap
    nout = np.bincount(fi[op.applicant_cd].values, minlength=nF).astype(np.float32)
    nin = M.sum(1)
    f, k = np.nonzero(Mrisk == 0)
    df = pd.DataFrame({'fidx': f, 'k': k, 'firm': firms[f],
        'lsize': np.log1p(nin[f] + nout[f]), 'lin': np.log1p(nin[f]), 'lout': np.log1p(nout[f]),
        'inshare': nin[f] / np.maximum(nin[f] + nout[f], 1), 'dens': dens[f, k], 'kpop': kpop[k],
        'y': Y[f, k], 'nfut': NF[f, k], 'newc': (Mrisk.sum(1)[f] == 0)})
    ctx = dict(cut=cut, h=h, cap=cap, firms=firms, fi=fi, M=M, prox=prox, op=op, ip=ip, nout=nout, nin=nin, A=A)
    return df, ctx

def add_dart(df, ctx):
    dm = dart_map(); K = load()['K']
    ind = dict(zip(dm.applicant_cd, dm.induty_code.astype(str)))
    est = dict(zip(dm.applicant_cd, dm.est_dt.astype(str)))
    firm_ind = pd.Series(ctx['firms']).map(ind).values
    kk = np.array(K)
    fi_ind = firm_ind[df.fidx.values]; kv = kk[df.k.values]
    has = pd.notna(fi_ind)
    s = np.where(has, fi_ind, '').astype(str)
    for L in (2, 3, 4):
        df[f'dm{L}'] = (has & (pd.Series(s).str[:L].values == pd.Series(kv).str[:L].values)).astype(np.int8)
    df['has_dart'] = has.astype(np.int8)
    df['ind2'] = pd.to_numeric(pd.Series(s).str[:2], errors='coerce').values
    e = pd.to_numeric(pd.Series(pd.Series(ctx['firms']).map(est).values).str[:4], errors='coerce').values
    age = ctx['cut'] - e
    age = np.where(age >= 0, age, np.nan)   # 기준시점 이후 설립은 미상 처리
    df['age'] = age[df.fidx.values]
    return df

# ---------------- 제목 → KSIC 분류기 ----------------
def train_clf(cap, tag, analyzer='char_wb', ngram=(2, 4), max_features=400000, alpha=2e-6, iters=12, class_weight=None, extra=None):
    """cap(YYYYMMDD)까지 등록된 범위안 KR 특허로 학습 → 범위밖 자사출원 특허 전체(2000+) 확률 산출·캐시"""
    pP = CACHE/f'P_{tag}.npy'
    if pP.exists(): return np.load(pP).astype(np.float32)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import normalize
    D = load(); oos = D['oos']
    kr = pd.read_parquet(CACHE/'ins_pat.parquet', columns=['title','ksic','rgst_date','ay'])
    tr = kr[(kr.rgst_date <= cap)].dropna(subset=['title'])
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=5, max_features=max_features, sublinear_tf=True, dtype=np.float32)
    X = vec.fit_transform(tr.title)
    clf = SGDClassifier(loss='log_loss', alpha=alpha, max_iter=iters, tol=None, random_state=0, class_weight=class_weight).fit(X, tr.ksic)
    assert list(clf.classes_) == D['K']
    P = np.zeros((len(oos), 50), np.float16)
    for s in range(0, len(oos), 200000):
        Xo = vec.transform(oos.title.iloc[s:s+200000].fillna(''))
        P[s:s+200000] = clf.predict_proba(Xo).astype(np.float16)
    np.save(pP, P)
    return P.astype(np.float32)

def text_feats(df, ctx, P, prefix='t', decay=None, hi=0.5):
    op = ctx['op']; fi = ctx['fi']; nF = len(ctx['firms'])
    rows = fi[op.applicant_cd].values; Po = P[op.row.values]
    w = np.ones(len(op), np.float32)
    if decay is not None:
        w = np.power(0.5, (ctx['cut'] - op.ay.values) / decay).astype(np.float32)
    T = np.zeros((nF, 50), np.float32); np.add.at(T, rows, Po * w[:, None])
    W = np.bincount(rows, weights=w, minlength=nF).astype(np.float32)
    H = np.zeros((nF, 50), np.float32); m = Po.max(1) > hi
    np.add.at(H, (rows[m], Po[m].argmax(1)), 1)
    f, k = df.fidx.values, df.k.values
    df[f'{prefix}_mean'] = (T / np.maximum(W, 1e-9)[:, None])[f, k]
    df[f'{prefix}_hi'] = np.log1p(H[f, k])
    return df, T, W

# ---------------- 평가 ----------------
from sklearn.metrics import roc_auc_score, average_precision_score
def pat_k(d, s, n):
    t = d.assign(s=s).sort_values('s', ascending=False).groupby('k').head(n)
    return t.y.mean()

def metrics(d, s):
    out = {}
    y = d.y.values; nc = d.newc.values
    out['auc_all'] = roc_auc_score(y, s); out['pr_all'] = average_precision_score(y, s)
    out['auc_new'] = roc_auc_score(y[nc], s[nc]); out['pr_new'] = average_precision_score(y[nc], s[nc])
    dn = d[nc]; sn = s[nc]
    out['p20_new'] = pat_k(dn, sn, 20); out['p100_new'] = pat_k(dn, sn, 100)
    out['p20_all'] = pat_k(d, s, 20); out['p100_all'] = pat_k(d, s, 100)
    inc = ~nc
    out['auc_inc'] = roc_auc_score(y[inc], s[inc]); out['pr_inc'] = average_precision_score(y[inc], s[inc])
    return {k: float(v) for k, v in out.items()}

def fit_xgb(tr, cols, seed=0, **kw):
    import xgboost as xgb
    p = dict(n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
             tree_method='hist', nthread=2, random_state=seed, eval_metric='logloss')
    p.update(kw)
    m = xgb.XGBClassifier(**p)
    m.fit(tr[cols].values.astype(np.float32), tr.y.values)
    return m

def predict(m, te, cols):
    return m.predict_proba(te[cols].values.astype(np.float32))[:, 1]

LB = T1/'leaderboard.csv'
def lb_add(version, change, hypothesis, met, base, kept, note):
    """주 지표 두 개(전체 AUC, 신규진입자 AUC)를 각각 한 행으로, 나머지는 note에."""
    rows = []
    today = datetime.date.today().isoformat()
    for key, name in [('auc_all', 'AUC_all'), ('auc_new', 'AUC_newcomer'), ('pr_new', 'PRAUC_newcomer'), ('pr_all', 'PRAUC_all')]:
        v = met[key]; b = base[key]
        sk = (v - b) / (1 - b) if key.startswith('auc') else v / b
        rows.append(dict(version=version, date=today, change=change, hypothesis=hypothesis, metric=name, value=round(v, 4),
                         baseline_value=round(b, 4), skill=round(sk, 4), kept=kept,
                         note=note + f" | p20_new={met['p20_new']:.3f} p100_new={met['p100_new']:.3f} p20_all={met['p20_all']:.3f} auc_inc={met['auc_inc']:.4f}" + (' (PR skill=배수)' if key.startswith('pr') else '')))
    df = pd.DataFrame(rows)
    if LB.exists():
        old = pd.read_csv(LB, encoding='utf-8-sig'); old = old[old.version != version]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(LB, index=False, encoding='utf-8-sig')

def fmt(m):
    return ' '.join(f'{k}={v:.4f}' for k, v in m.items())

def train_clf51(cap, tag, analyzer='char_wb', ngram=(2, 4), max_features=400000, alpha=2e-6, iters=12, word=False, other_w=1.0):
    """open-set 51-class(범위밖 자사출원=OTHER) 교차적합: 법인 해시 2-fold, 각 fold의 범위밖 특허는 다른 fold OTHER로 학습된 모형으로 채점.
    반환 P(50열, 재정규화하지 않은 산업 확률), Pother"""
    pP = CACHE/f'P_{tag}.npy'; pO = CACHE/f'PO_{tag}.npy'
    if pP.exists(): return np.load(pP).astype(np.float32), np.load(pO).astype(np.float32)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import FeatureUnion
    D = load(); oos = D['oos']
    kr = pd.read_parquet(CACHE/'ins_pat.parquet', columns=['title','ksic','rgst_date'])
    tr_in = kr[kr.rgst_date <= cap].dropna(subset=['title'])
    fold = (pd.util.hash_array(oos.applicant_cd.values.astype(object)) % 2).astype(int)
    P = np.zeros((len(oos), 50), np.float16); PO = np.zeros(len(oos), np.float16)
    for fo in (0, 1):
        oth = oos[(fold != fo) & (oos.rgst_date <= cap)]
        titles = pd.concat([tr_in.title, oth.title.fillna('')]); ys = np.concatenate([tr_in.ksic.values, np.array(['OTHER'] * len(oth))])
        if word:
            vec = FeatureUnion([('c', TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=5, max_features=max_features, sublinear_tf=True, dtype=np.float32)),
                                ('w', TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=3, max_features=200000, sublinear_tf=True, dtype=np.float32))])
        else:
            vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=5, max_features=max_features, sublinear_tf=True, dtype=np.float32)
        X = vec.fit_transform(titles)
        sw = np.where(ys == 'OTHER', other_w, 1.0)
        clf = SGDClassifier(loss='log_loss', alpha=alpha, max_iter=iters, tol=None, random_state=0).fit(X, ys, sample_weight=sw)
        cls = list(clf.classes_); oi = cls.index('OTHER'); ki = [cls.index(k) for k in D['K']]
        idx = np.nonzero(fold == fo)[0]
        for s in range(0, len(idx), 200000):
            ii = idx[s:s+200000]
            pr = clf.predict_proba(vec.transform(oos.title.iloc[ii].fillna('')))
            P[ii] = pr[:, ki].astype(np.float16); PO[ii] = pr[:, oi].astype(np.float16)
        print('fold', fo, 'done; OTHER train n', len(oth), flush=True)
    np.save(pP, P); np.save(pO, PO)
    return P.astype(np.float32), PO.astype(np.float32)
