# T2 산업×국가 특허 출원 추세 예측 — 동결 실행 스크립트 (2단계 감사·동결, 2026-09-27)
#
#   python final_run.py --mode dryrun   : 검증 원점 2012–2015, h=1..3 (목표연도 ≤2018) 재현
#   python final_run.py --mode holdout  : 🔒 원점 2018 → 2019·2020·2021 (3단계 평가자 전용, 1·2단계 실행 금지)
#   python final_run.py --mode future   : 전체 가용자료(출원연도 ≤2022, 공개지연 역확률 보정)로 원점 2022 → 2023–2027 트렌드 예측
#                                         (--no-nowcast 로 보정 끄기)
#
# 설계 원칙(감사 반영)
#  - 모형 입력은 매 원점 o마다 "출원연도 ≤ o" 로 잘라낸 배열만 받는다(truncate). 전역 배열 재사용 없음.
#  - holdout 모드는 원점 2018 패널(≤2018)만 집계해 예측을 먼저 파일로 저장한 뒤에야 2019–2021 실제값을 집계한다.
#  - 결과변수 = (ksic, cc, 출원연도)별 고유 출원 수(cc+appl_no). 2023~ 출원은 공개지연 절단이라 쓰지 않는다.
#  - 동결 설정은 final_spec.json 에서 읽는다(없으면 아래 DEFAULT_CFG).
import argparse, os, json, time, warnings
import numpy as np, pandas as pd

warnings.filterwarnings('ignore')
T = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(T, '..', '_shared', 'meta', '*.parquet').replace(os.sep, '/')
CACHE = os.path.join(T, 'cache2'); os.makedirs(CACHE, exist_ok=True)
CCS = ['CN', 'EP', 'JP', 'KR', 'US']
Y0 = 2000
VAL_ORIGINS = [2012, 2013, 2014, 2015]
HOLDOUT_ORIGIN, HOLDOUT_TARGETS = 2018, [2019, 2020, 2021]
FUTURE_ORIGIN, FUTURE_H = 2022, 5
YMAX_USABLE = 2022  # 2023~ 출원연도는 공개지연 절단 → 사용 금지

DEFAULT_CFG = {
    'members': {
        'ridge_lead': {'alpha': 10.0, 'tmin': 2005, 'clipq': 0.99},
        'flow_um': {'k': 8, 'phi': 0.8, 'lag': 1, 'rk': 1, 'um_split': True, 'um_min': 20},
        'topdown': {'kt': 8, 'pt': 1.0, 'ks': 8, 'ps': 0.8},
    },
    'combine': 'mean',
}


def load_cfg():
    p = os.path.join(T, 'final_spec.json')
    if os.path.exists(p):
        return json.load(open(p, encoding='utf-8'))['model']
    return DEFAULT_CFG


# ───────────────────────── 데이터 ─────────────────────────
def _con():
    import duckdb
    con = duckdb.connect(); con.sql("SET threads=2; SET memory_limit='3GB'")
    return con


UM_SQL = """case when cc='CN' and length(appl_no)>=12 and substr(appl_no,5,1)='2' then 1
       when cc='CN' and length(appl_no)=9 and substr(appl_no,3,1)='2' then 1
       when cc='KR' and substr(appl_no,1,2)='20' then 1 else 0 end"""


def build_panel(ymax, ymin=Y0):
    """출원연도 [ymin, ymax] 만 집계. 반환 (kc, prio) DataFrame. 캐시: cache2/"""
    assert ymax <= YMAX_USABLE
    f1 = os.path.join(CACHE, f'kc_{ymin}_{ymax}.parquet'); f2 = os.path.join(CACHE, f'prio_{ymin}_{ymax}.parquet')
    if not (os.path.exists(f1) and os.path.exists(f2)):
        con = _con()
        con.sql(f"""create temp table b as select cc, appl_no, appl_year::int y, ksic,
            nullif(split_part(coalesce(prio_cc,''),'|',1),'') p1, {UM_SQL} um
            from read_parquet('{META}') where appl_year between {ymin} and {ymax}""")
        # 고유 출원 수(cc+appl_no) — 같은 (ksic,cc,appl_no) 중복행 방지
        con.sql("create temp table u as select ksic, cc, appl_no, any_value(y) y, max(um) um, any_value(p1) p1 from b group by 1,2,3")
        con.sql(f"""copy (select ksic, cc, y AS "year", count(*) n, sum(um) n_um from u group by 1,2,3 order by 1,2,3)
                  to '{f1.replace(os.sep, '/')}' (format parquet)""")
        con.sql(f"""copy (select ksic, cc, p1, y AS "year", count(*) n from u where p1 is not null group by 1,2,3,4)
                  to '{f2.replace(os.sep, '/')}' (format parquet)""")
        con.close()
    kc = pd.read_parquet(f1); pr = pd.read_parquet(f2)
    assert kc.year.max() <= ymax and pr.year.max() <= ymax
    return kc, pr


def make_data(kc, pr, ymax):
    years = np.arange(Y0, ymax + 1)
    ks = sorted(kc.ksic.unique())
    idx = pd.MultiIndex.from_product([ks, CCS], names=['ksic', 'cc'])

    def arr(df, v='n'):
        w = df.pivot_table(index=['ksic', 'cc'], columns='year', values=v, aggfunc='sum')
        return w.reindex(idx).reindex(columns=years).fillna(0.0).values.astype(float)

    n = arr(kc); um = arr(kc, 'n_um')
    FY = {Y: arr(pr[(pr.p1 == Y) & (pr.cc != Y)]) for Y in CCS}
    OTHER = arr(pr[~pr.p1.isin(CCS)])
    DOM = n - sum(FY.values()) - OTHER  # 자국 우선권 + 우선권 없음
    assert DOM.min() >= -1e-9
    CCI = np.array([CCS.index(c) for c in idx.get_level_values('cc')])
    KSI = np.array([ks.index(k) for k in idx.get_level_values('ksic')])
    ROW = {(KSI[i], CCI[i]): i for i in range(len(idx))}
    SRCROWS = {Y: np.array([ROW[(KSI[i], CCS.index(Y))] for i in range(len(idx))]) for Y in CCS}
    return dict(idx=idx, years=years, n=n, um=um, FY=FY, OTHER=OTHER, DOM=DOM, CCI=CCI, KSI=KSI, SRCROWS=SRCROWS)


def truncate(D, o):
    """출원연도 ≤ o 만 남긴 사본 — 모든 모형은 이 사본만 본다"""
    c = int(o - Y0) + 1
    E = dict(D)
    for k in ['n', 'um', 'OTHER', 'DOM']:
        E[k] = D[k][:, :c].copy()
    E['FY'] = {Y: D['FY'][Y][:, :c].copy() for Y in CCS}
    E['years'] = D['years'][:c]
    assert E['years'][-1] == o
    return E


# ───────────────────────── 구성원 모형 ─────────────────────────
def _ld(M, k, phi, H, log1p=True):
    """최근 k년 평균 로그성장을 phi로 감쇠 외삽. k가 튜플이면 여러 창 기울기 평균"""
    L = np.log1p(M) if log1p else np.log(np.maximum(M, 1e-9))
    ks = k if isinstance(k, (list, tuple)) else [k]
    g = np.mean([(L[:, -1] - L[:, -1 - kk]) / kk for kk in ks], 0)
    steps = np.cumsum(phi ** np.arange(1, H + 1))
    Z = L[:, -1:] + g[:, None] * steps[None, :]
    return np.expm1(Z) if log1p else np.exp(Z)


def _ridge_feats(E, c):
    n = E['n']; CCI = E['CCI']
    L = np.log1p(n)
    F = {}
    for l in range(1, 6):
        F[f'g_l{l}'] = L[:, c - l + 1] - L[:, c - l]
    F['g3'] = (L[:, c] - L[:, c - 3]) / 3
    F['g5'] = (L[:, c] - L[:, c - 5]) / 5
    F['acc'] = F['g_l1'] - F['g3']
    F['lvl'] = L[:, c]
    Ct = np.stack([n[CCI == j].sum(0) for j in range(5)]); Lt = np.log1p(Ct)
    F['c_g1'] = (Lt[:, c] - Lt[:, c - 1])[CCI]
    F['c_g3'] = ((Lt[:, c] - Lt[:, c - 3]) / 3)[CCI]
    for j, cj in enumerate(CCS):
        F[f'is_{cj}'] = (CCI == j).astype(float)
    nn = np.maximum(n[:, c], 1)
    num = np.zeros(len(n)); lead1 = np.zeros(len(n)); lead3 = np.zeros(len(n))
    for Y in CCS:
        S = E['DOM'][E['SRCROWS'][Y]]
        sh = E['FY'][Y][:, c] / nn
        num += sh
        lead1 += sh * (np.log1p(S[:, c]) - np.log1p(S[:, c - 1]))
        lead3 += sh * (np.log1p(S[:, c]) - np.log1p(S[:, c - 3])) / 3
    F['for_sh'] = num; F['lead_g1'] = lead1; F['lead_g3'] = lead3
    F['dom_g3'] = (np.log1p(E['DOM'][:, c]) - np.log1p(E['DOM'][:, c - 3])) / 3
    return pd.DataFrame(F)


def ridge_lead(E, o, H, alpha=10.0, tmin=2005, clipq=0.99):
    """전역 Ridge(h별): 타깃 log1p(y_{t+h})-log1p(y_t), 학습 기준연도 t∈[tmin, o-h] (라벨 ≤ o)"""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    n = E['n']; L = np.log1p(n); co = n.shape[1] - 1
    out = np.zeros((n.shape[0], H))
    for h in range(1, H + 1):
        X, y, w = [], [], []
        for t in range(tmin, o - h + 1):
            c = t - Y0
            assert c + h <= co
            X.append(_ridge_feats(E, c)); y.append(L[:, c + h] - L[:, c])
            sc = np.mean(np.abs(np.diff(n[:, :c + 1], axis=1)), axis=1)
            w.append(n[:, c] / np.maximum(sc, 1))
        X = pd.concat(X, ignore_index=True); y = np.concatenate(y); w = np.concatenate(w)
        w = np.clip(w, 0, np.quantile(w, clipq))
        ss = StandardScaler().fit(X.values)
        m = Ridge(alpha=alpha).fit(ss.transform(X.values), y, sample_weight=w)
        g = m.predict(ss.transform(_ridge_feats(E, co)[X.columns].values))
        out[:, h - 1] = np.expm1(L[:, -1] + g)
    return np.maximum(out, 0)


def flow_um(E, o, H, k=8, phi=0.8, lag=1, rk=1, um_split=True, um_min=20):
    """우선권 흐름 분해: n_X = D_X + Σ_Y rate_{Y→X}·D_Y(t-lag) + O_X, D_X는 실용신안/발명 분리 외삽"""
    D = E['DOM']; O = E['OTHER']; T_ = D.shape[1]
    FD = _ld(D, k, phi, H)
    if um_split:
        U = np.minimum(E['um'], D)
        kk = max(k) if isinstance(k, (list, tuple)) else k
        has = U[:, -kk - 1:].min(1) >= um_min
        FDs = _ld(D - U, k, phi, H) + _ld(U, k, phi, H)
        FD[has] = FDs[has]
    out = FD + _ld(O, k, phi, H)
    for Y in CCS:
        Fy = E['FY'][Y]; rows = E['SRCROWS'][Y]
        Sx = np.concatenate([D[rows], FD[rows]], axis=1)  # 원천국 자국기원: 관측(≤o) + 예측
        num = Fy[:, -rk:].sum(1)
        den = Sx[:, T_ - rk - lag:T_ - lag].sum(1)
        rate = np.where(den > 0, num / np.maximum(den, 1), 0)
        for h in range(1, H + 1):
            out[:, h - 1] += rate * Sx[:, T_ - 1 + h - lag]
    return np.maximum(out, 0)


def topdown(E, o, H, kt=8, pt=1.0, ks=8, ps=0.8):
    """국가합(비감쇠 로그드리프트) × 산업점유율(감쇠 로그드리프트, 국가 내 재정규화)"""
    n = E['n']; CCI = E['CCI']
    Ct = np.stack([n[CCI == j].sum(0) for j in range(5)])
    FT = _ld(Ct, kt, pt, H, log1p=False)
    FS = _ld((n + 0.5) / Ct[CCI], ks, ps, H, log1p=False)
    for j in range(5):
        FS[CCI == j] /= FS[CCI == j].sum(0, keepdims=True)
    return FT[CCI] * FS


def ebdrift(E, o, H, k=8, phi=0.8, kc=8):
    """경험적 베이즈 부분풀링: 계열 로그기울기를 국가합 기울기 쪽으로 잡음비례 수축"""
    n = E['n']; CCI = E['CCI']
    L = np.log1p(n)
    g = (L[:, -1] - L[:, -1 - k]) / k
    Ct = np.log(np.stack([n[CCI == j].sum(0) for j in range(5)]))
    gc = ((Ct[:, -1] - Ct[:, -1 - kc]) / kc)[CCI]
    d = np.diff(L[:, -k - 1:], axis=1)
    s2 = d.var(1, ddof=1) / k  # 기울기 추정 분산
    B = np.zeros(len(n))
    for j in range(5):
        m = CCI == j
        tau2 = max(np.var(g[m] - gc[m], ddof=1) - s2[m].mean(), 1e-6)
        B[m] = tau2 / (tau2 + s2[m])
    gg = gc + B * (g - gc)
    steps = np.cumsum(phi ** np.arange(1, H + 1))
    return np.expm1(L[:, -1:] + gg[:, None] * steps[None, :])


def etslog(E, o, H, start=2005):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    n = E['n']; out = np.zeros((n.shape[0], H))
    for i in range(n.shape[0]):
        yy = np.log1p(n[i, start - Y0:])
        try:
            f = ExponentialSmoothing(yy, trend='add', damped_trend=True, initialization_method='estimated').fit().forecast(H)
        except Exception:
            f = np.repeat(yy[-1], H)
        out[i] = np.expm1(np.asarray(f))
    return np.maximum(out, 0)


MEMBER_FNS = {'ridge_lead': ridge_lead, 'flow_um': flow_um, 'topdown': topdown, 'ebdrift': ebdrift, 'etslog': etslog}


def _fn_of(name):
    return MEMBER_FNS[name.split('#')[0]]  # 'flow_um#k5' 처럼 같은 구조의 변형을 구분


def forecast(E, o, H, cfg, return_members=False):
    assert E['years'][-1] == o
    Fs = {m: np.asarray(_fn_of(m)(E, o, H, **kw), float) for m, kw in cfg['members'].items()}
    A = np.stack(list(Fs.values()))
    how = cfg.get('combine', 'mean')
    if how == 'mean':
        F = A.mean(0)
    elif how == 'geo':
        F = np.expm1(np.log1p(np.maximum(A, 0)).mean(0))
    elif how == 'median':
        F = np.median(A, 0)
    elif isinstance(how, dict):  # 가중 평균 {member: w}
        w = np.array([how[m] for m in Fs]); F = np.tensordot(w / w.sum(), A, 1)
    else:
        raise ValueError(how)
    if cfg.get('recon'):  # 국가합을 지정 구성원의 국가합에 맞춤
        ref = Fs[cfg['recon']]
        for j in range(5):
            m = E['CCI'] == j
            F[m] = F[m] * (ref[m].sum(0) / F[m].sum(0))[None, :]
    F = np.maximum(F, 0)
    return (F, Fs) if return_members else F


# ───────────────────────── 평가 ─────────────────────────
def mase_scale(hist):
    s = np.mean(np.abs(np.diff(hist, axis=1)), axis=1)
    return np.where(s <= 0, 1.0, s)


def score_rows(D, o, H, F, Yfut):
    """F, Yfut: (250,H). Yfut = 목표연도 실제값. 척도 = 2000..o 1-step naive 평균절대차"""
    E = truncate(D, o) if D['years'][-1] > o else D
    sc = mase_scale(E['n'])
    rows = []
    for h in range(1, H + 1):
        for i, (k, c) in enumerate(D['idx']):
            rows.append((k, c, o, h, o + h, Yfut[i, h - 1], F[i, h - 1], E['n'][i, -1], sc[i]))
    r = pd.DataFrame(rows, columns=['ksic', 'cc', 'origin', 'h', 'target', 'y', 'f', 'ylast', 'scale'])
    r['ae'] = (r.y - r.f).abs(); r['ase'] = r.ae / r.scale
    return r


def evaluate(D, cfg, origins=VAL_ORIGINS, H=3, fn=None):
    out = []
    for o in origins:
        E = truncate(D, o)
        F = fn(E, o, H) if fn is not None else forecast(E, o, H, cfg)
        c = o - Y0
        Yfut = D['n'][:, c + 1:c + 1 + H]
        assert Yfut.shape[1] == H
        out.append(score_rows(D, o, H, F, Yfut))
    return pd.concat(out, ignore_index=True)


def naive_fc(E, o, H):
    return np.repeat(E['n'][:, -1:], H, axis=1)


def drift_fc(E, o, H):
    n = E['n']; s = (n[:, -1] - n[:, 0]) / (n.shape[1] - 1)
    return n[:, -1:] + s[:, None] * np.arange(1, H + 1)[None, :]


def ets_fc(E, o, H):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    n = E['n']; out = np.zeros((n.shape[0], H))
    for i in range(n.shape[0]):
        try:
            out[i] = ExponentialSmoothing(n[i], trend='add', damped_trend=True, initialization_method='estimated').fit().forecast(H)
        except Exception:
            out[i] = n[i, -1]
    return np.maximum(out, 0)


def boot_ci(r, nb, B=2000, seed=0):
    """KSIC 군집 부트스트랩(50개 산업 재표집; 같은 산업의 5개국·원점·h는 함께 재표집)"""
    rng = np.random.default_rng(seed)
    a = r.groupby('ksic').ase.agg(['sum', 'count']); b = nb.groupby('ksic').ase.agg(['sum', 'count'])
    ks = a.index.values
    sk, dm = [], []
    for _ in range(B):
        s = rng.choice(ks, len(ks), replace=True)
        ma = a.loc[s, 'sum'].sum() / a.loc[s, 'count'].sum(); mb = b.loc[s, 'sum'].sum() / b.loc[s, 'count'].sum()
        sk.append(1 - ma / mb); dm.append(ma - mb)
    return np.percentile(sk, [2.5, 97.5]).tolist(), np.percentile(dm, [2.5, 97.5]).tolist()


def summary(r, nb):
    s = {'MASE': float(r.ase.mean()), 'MASE_naive': float(nb.ase.mean()),
         'skill': float(1 - r.ase.mean() / nb.ase.mean()),
         'WAPE': float(r.ae.sum() / r.y.abs().sum()), 'WAPE_naive': float(nb.ae.sum() / nb.y.abs().sum())}
    for g in ['h', 'cc', 'origin']:
        s[f'by_{g}'] = {str(k): {'MASE': float(r[r[g] == k].ase.mean()), 'MASE_naive': float(nb[nb[g] == k].ase.mean()),
                                 'skill': float(1 - r[r[g] == k].ase.mean() / nb[nb[g] == k].ase.mean())}
                        for k in sorted(r[g].unique())}
    return s


# ───────────────────────── 모드 ─────────────────────────
def mode_dryrun(cfg):
    kc, pr = build_panel(2018)
    D = make_data(kc, pr, 2018)
    t0 = time.time()
    r = evaluate(D, cfg); nb = evaluate(D, cfg, fn=naive_fc)
    s = summary(r, nb)
    ci_sk, ci_d = boot_ci(r, nb)
    s['skill_CI95_ksic_boot'] = ci_sk; s['MASE_diff_CI95'] = ci_d
    s['n_forecasts'] = len(r); s['sec'] = round(time.time() - t0, 1)
    for nm, fn in [('drift', drift_fc), ('damped_ets', ets_fc)]:
        rb = evaluate(D, cfg, fn=fn); s[f'MASE_{nm}'] = float(rb.ase.mean()); s[f'skill_{nm}'] = float(1 - rb.ase.mean() / nb.ase.mean())
    print(json.dumps({k: v for k, v in s.items() if not k.startswith('by_')}, ensure_ascii=False, indent=1))
    for g in ['h', 'cc', 'origin']:
        print(g, {k: round(v['skill'], 4) for k, v in s[f'by_{g}'].items()})
    json.dump(s, open(os.path.join(T, 'dryrun_result.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    r.to_parquet(os.path.join(T, 'dryrun_preds.parquet'))
    # 코드 경로 리허설(검증 구간 안에서만): holdout 절차를 원점 2015→2016–18 로, future 절차를 원점 2018→h=1..5(실제값 없음)로
    rh = mode_holdout(cfg, origin=2015, targets=[2016, 2017, 2018], out_json='dryrun_rehearsal_holdout2015.json',
                      out_fc='dryrun_rehearsal_holdout2015_forecast.csv', out_preds='dryrun_rehearsal_holdout2015_preds.parquet')
    v15 = s['by_origin']['2015']
    print('리허설(원점2015) MASE', round(rh['MASE'], 4), '검증 원점2015 MASE', round(v15['MASE'], 4))
    assert abs(rh['MASE'] - v15['MASE']) < 1e-9
    mode_future(cfg, nowcast=True, ymax=2018, FH=5, out_csv='dryrun_rehearsal_future2018.csv', out_fac='dryrun_rehearsal_completeness2018.csv')
    return s


def mode_holdout(cfg, origin=HOLDOUT_ORIGIN, targets=HOLDOUT_TARGETS, out_json='holdout_result.json',
                 out_fc='holdout_forecast.csv', out_preds='holdout_preds.parquet'):
    # 1) 원점 패널(출원연도 ≤ origin)만으로 학습·예측 → 먼저 저장
    HOLDOUT_ORIGIN_, HOLDOUT_TARGETS_ = origin, list(targets)
    assert HOLDOUT_TARGETS_ == [origin + h for h in range(1, len(targets) + 1)]
    kc, pr = build_panel(HOLDOUT_ORIGIN_)
    D = make_data(kc, pr, HOLDOUT_ORIGIN_)
    E = truncate(D, HOLDOUT_ORIGIN_)
    H = len(HOLDOUT_TARGETS_)
    F, Fs = forecast(E, HOLDOUT_ORIGIN_, H, cfg, return_members=True)
    comp = {'naive': naive_fc(E, HOLDOUT_ORIGIN_, H), 'drift': drift_fc(E, HOLDOUT_ORIGIN_, H), 'damped_ets': ets_fc(E, HOLDOUT_ORIGIN_, H)}
    fp = []
    for i, (k, c) in enumerate(D['idx']):
        for h in range(1, H + 1):
            fp.append(dict(ksic=k, cc=c, origin=HOLDOUT_ORIGIN_, target=HOLDOUT_ORIGIN_ + h, y_origin=E['n'][i, -1], f=F[i, h - 1],
                           **{f'f_{m}': Fs[m][i, h - 1] for m in Fs}, **{f'f_{m}': comp[m][i, h - 1] for m in comp}))
    pd.DataFrame(fp).to_csv(os.path.join(T, out_fc), index=False, encoding='utf-8-sig')
    # 2) 예측 저장 후에만 목표연도 실제값 집계 (2019–2021, 고유 출원 수)
    con = _con()
    act = con.sql(f"""select ksic, cc, y AS "year", count(*) n from
        (select ksic, cc, appl_no, any_value(appl_year::int) y from read_parquet('{META}')
         where appl_year between {HOLDOUT_TARGETS_[0]} and {HOLDOUT_TARGETS_[-1]} group by 1,2,3) group by 1,2,3""").df()
    con.close()
    w = act.pivot_table(index=['ksic', 'cc'], columns='year', values='n', aggfunc='sum')
    Yfut = w.reindex(D['idx']).reindex(columns=HOLDOUT_TARGETS_).fillna(0.0).values.astype(float)
    r = score_rows(D, HOLDOUT_ORIGIN_, H, F, Yfut)
    nb = score_rows(D, HOLDOUT_ORIGIN_, H, comp['naive'], Yfut)
    s = summary(r, nb)
    s['skill_CI95_ksic_boot'], s['MASE_diff_CI95'] = boot_ci(r, nb)
    for m in ['drift', 'damped_ets']:
        rb = score_rows(D, HOLDOUT_ORIGIN_, H, comp[m], Yfut)
        s[f'MASE_{m}'] = float(rb.ase.mean()); s[f'skill_{m}'] = float(1 - rb.ase.mean() / nb.ase.mean())
    for m in Fs:
        rb = score_rows(D, HOLDOUT_ORIGIN_, H, Fs[m], Yfut)
        s[f'MASE_member_{m}'] = float(rb.ase.mean()); s[f'skill_member_{m}'] = float(1 - rb.ase.mean() / nb.ase.mean())
    ex = r.ksic != '31312'
    s['skill_excl_31312'] = float(1 - r[ex].ase.mean() / nb[ex].ase.mean())
    s['n_forecasts'] = len(r); s['eval_unit'] = '(ksic,cc) 250계열 × h=1..3'; s['cluster_unit'] = 'KSIC(50)'
    s['actual_country_totals'] = {f'{c}_{y}': float(Yfut[D['CCI'] == j][:, i].sum()) for j, c in enumerate(CCS) for i, y in enumerate(HOLDOUT_TARGETS_)}
    s['forecast_country_totals'] = {f'{c}_{y}': float(F[D['CCI'] == j][:, i].sum()) for j, c in enumerate(CCS) for i, y in enumerate(HOLDOUT_TARGETS_)}
    s['config'] = cfg
    json.dump(s, open(os.path.join(T, out_json), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    r.to_parquet(os.path.join(T, out_preds))
    print(json.dumps({k: v for k, v in s.items() if not k.startswith('by_') and k != 'config'}, ensure_ascii=False, indent=1))
    return s


def completeness_factors(ymax, years):
    """공개지연 절단 보정계수: 각 국가의 (공개일|등록일) 지연 경험분포(출원 2012–2016)로
    출원월별 관측확률 p = F(수집마감 − 출원월)을 구하고, (ksic,cc,year)별 Σ1/p ÷ 관측건수 를 반환"""
    con = _con()
    lag = con.sql(f"""select cc, date_diff('month', strptime(appl_date,'%Y%m%d'),
            coalesce(try_strptime(nullif(pub_date,''),'%Y%m%d'), try_strptime(nullif(rgst_date,''),'%Y%m%d'))) m, count(*) n
        from read_parquet('{META}') where appl_year between 2012 and 2016 group by 1,2""").df()
    cut = con.sql(f"""select cc, max(coalesce(try_strptime(nullif(pub_date,''),'%Y%m%d'), try_strptime(nullif(rgst_date,''),'%Y%m%d'))) d
        from read_parquet('{META}') group by 1""").df()
    ym = ','.join(str(y) for y in years)
    cnt = con.sql(f"""select ksic, cc, appl_year::int y, substr(appl_date,5,2)::int mo, count(*) n from
        (select ksic, cc, appl_no, any_value(appl_year) appl_year, any_value(appl_date) appl_date from read_parquet('{META}')
         where appl_year in ({ym}) group by 1,2,3) group by 1,2,3,4""").df()
    con.close()
    lag = lag.dropna(); lag = lag[lag.m >= 0]
    fac_rows = []
    for c in CCS:
        l = lag[lag.cc == c].sort_values('m'); cdf = l.n.cumsum() / l.n.sum()
        Fm = lambda mm: float(cdf[l.m.values <= mm].iloc[-1]) if (l.m.values <= mm).any() else 0.0
        cd = pd.Timestamp(cut[cut.cc == c].d.iloc[0])
        sub = cnt[cnt.cc == c].copy()
        sub['p'] = [max(Fm((cd.year - y) * 12 + cd.month - mo), 0.5) for y, mo in zip(sub.y, sub.mo)]
        sub['w'] = sub.n / sub.p
        fac_rows.append(sub)
    f = pd.concat(fac_rows).groupby(['ksic', 'cc', 'y'])[['n', 'w']].sum()
    f['factor'] = f.w / f.n
    return f.factor.reset_index()


def mode_future(cfg, nowcast=True, ymax=FUTURE_ORIGIN, FH=FUTURE_H, out_csv='future_forecast.csv', out_fac='future_completeness_factors.csv'):
    kc, pr = build_panel(ymax)
    fac = None
    if nowcast:  # 공개지연 미관측분 역확률 보정 — 일관성을 위해 전 연도(2000..ymax)에 같은 방식 적용(과거 연도는 ≈1.00)
        fac = completeness_factors(ymax, list(range(Y0, ymax + 1)))
        fac.to_csv(os.path.join(T, out_fac), index=False, encoding='utf-8-sig')
        kc = kc.merge(fac.rename(columns={'y': 'year'}), on=['ksic', 'cc', 'year'], how='left')
        kc['factor'] = kc.factor.fillna(1.0)
        for v in ['n', 'n_um']:
            kc[v] = kc[v] * kc.factor
        pr = pr.merge(fac.rename(columns={'y': 'year'}), on=['ksic', 'cc', 'year'], how='left')
        pr['n'] = pr.n * pr.factor.fillna(1.0)
    D = make_data(kc, pr, ymax)
    E = truncate(D, ymax)
    F, Fs = forecast(E, ymax, FH, cfg, return_members=True)
    # 구간: 검증 로그오차의 국가×h 10·90% 분위(h>3은 h=3 분위를 sqrt(h/3)배 확장) — 검증 내 보정이라 낙관적
    q = None
    p = os.path.join(T, 'dryrun_preds.parquet')
    if os.path.exists(p):
        rv = pd.read_parquet(p); rv['le'] = np.log1p(rv.y) - np.log1p(rv.f)
        q = rv.groupby(['cc', 'h']).le.quantile([0.1, 0.9]).unstack()
    rows = []
    for i, (k, c) in enumerate(D['idx']):
        for h in range(1, FH + 1):
            d = dict(ksic=k, cc=c, origin=ymax, target=ymax + h, y_origin=E['n'][i, -1], f=F[i, h - 1],
                     **{f'f_{m}': Fs[m][i, h - 1] for m in Fs})
            if q is not None:
                ql, qh = q.loc[(c, min(h, 3))]; sf = np.sqrt(max(h, 3) / 3)
                d['lo80'] = np.expm1(np.log1p(F[i, h - 1]) + ql * sf); d['hi80'] = np.expm1(np.log1p(F[i, h - 1]) + qh * sf)
            rows.append(d)
    out = pd.DataFrame(rows)
    L = np.log1p(E['n'])
    past = (L[:, -1] - L[:, -6]) / 5
    fut = (np.log1p(F[:, -1]) - L[:, -1]) / FH
    acc = pd.DataFrame({'ksic': D['idx'].get_level_values(0), 'cc': D['idx'].get_level_values(1),
                        'cagr_past5': np.expm1(past), 'cagr_fut5': np.expm1(fut), 'accel_pp': (np.expm1(fut) - np.expm1(past)) * 100})
    out = out.merge(acc, on=['ksic', 'cc'], how='left')
    out.to_csv(os.path.join(T, out_csv), index=False, encoding='utf-8-sig')
    print(out.groupby(['cc', 'target']).f.sum().unstack().round(0))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['dryrun', 'holdout', 'future'])
    ap.add_argument('--no-nowcast', action='store_true')
    a = ap.parse_args()
    cfg = load_cfg()
    if a.mode == 'dryrun':
        mode_dryrun(cfg)
    elif a.mode == 'holdout':
        mode_holdout(cfg)
    else:
        mode_future(cfg, nowcast=not a.no_nowcast)
