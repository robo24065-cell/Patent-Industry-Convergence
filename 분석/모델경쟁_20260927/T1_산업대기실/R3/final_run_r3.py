# T1 산업 대기실 — 3라운드 동결 실행기 (R3/final_spec_r3.json)
#   python R3/final_run_r3.py --mode dev      개발 원점 2008–2012 + 보조 원점 c=2015 재현(학습·채점) → R3/dev_result_r3.json
#                                             + 사전등록 보조 변형의 국민연금 보정층을 c=2015에서 적합해 동결 → R3/nps_layer_r3.json
#   python R3/final_run_r3.py --mode final    🔒 PROTOCOL_R3 §3 새 최종시험: 기준시점 c=2019(피처 = 2019-12-31까지 등록·공개된 정보) → 진입 2020–2023 출원 (F'=2020)
#                                             학습 원점 2012–2015(진입창 끝 ≤2019 < F'), 학습 레이블·분류기·IPW·공개 보강은 2019-12-31 이전 관측분만
#                                             → R3/final_result_r3.json (모형·고정 기준선 6변형·최강 기준선 지표 + 법인 군집 부트스트랩 95% CI, 보조 변형)
#   python R3/final_run_r3.py --mode future   전체 가용 자료: c=2024(→2025–2028 잠재 진입), 학습 원점 2016–2019(레이블 cap=2025-08-22) → R3/future_forecast_r3.csv
# 한 번 실행은 9분 이내로 끊긴다(종료코드 3 → 같은 명령 재실행 시 캐시에서 이어짐: run_final_r3.sh). 메모리: 단계별 별도 프로세스(외부 피처·표본 추출 / 로지스틱 / XGB 시드별).
# 🔒 누수 가드
#   ① 학습 레이블은 cut+4 ≤ 2019를 assert(r3.GUARD_DEV). 평가 원점 패널은 진입창 0으로 만들어 레이블 열이 비어 있고(y=0),
#      평가 레이블(2020–23)은 채점 함수 score_final()에서만 가드를 2023으로 올려 계산한다.
#   ② 제목 분류기(범위밖 특허·디자인·실용신안)는 cap(2019-12-31) 이전 등록 범위안 명칭으로만 학습.
#   ③ 외부 피처: 디자인·실용신안 등록일 ≤ 원점 연말, 물가 ≤ 원점−1년, 논문 ≤ 원점−1년, US 발행 ≤ 원점 연말, 국세청 폐업일 ≤ 원점 연말, 국민연금 원점 연말 스냅샷.
#   ③' 기술 적합도·근접 피처(r3ext2 ipcx, r3ext3 ipcm·entfit·peer): 원점 연말까지 공개 또는 등록된 KR 출원(출원연도 ≤ 원점)만 — vis(min(공개일, 등록일)) ≤ 원점-12-31.
#   ④ 2라운드 최종시험(c=2017) 결과는 보지 않았고 튜닝에 쓰지 않았다.
import sys, json, argparse, pathlib, os, gc, contextlib
R3 = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(R3))
import r3
from r3 import *
import r2ext, r3ext, r3ext2, r3ext3
import pyarrow.parquet as pq, pyarrow as pa

SPEC = json.load(open(R3/'final_spec_r3.json', encoding='utf-8'))
EC = SPEC['config']
EXTD = B/'분석/모델경쟁_20260927/_shared/ext'
MODES = {
    'final': dict(eval_cut=2019, train_cuts=(2012, 2013, 2014, 2015), cap='20191231', guard_train=2019, guard_eval=2023,
                  FC=R3/'cache_final'/'o2019', RD=R3/'runs'/'FROZEN_final', window='2020-2023', F=2020),
    'future': dict(eval_cut=2024, train_cuts=(2016, 2017, 2018, 2019), cap=END, guard_train=2023, guard_eval=None,
                   FC=R3/'cache_future'/'o2024', RD=R3/'runs'/'FROZEN_future', window='2025-2028', F=None),
}


@contextlib.contextmanager
def guard(g):
    """r3.reg_lab·r3.pub_lab·build_panel의 결과창 가드를 g로(블록이 끝나면 원복)"""
    o1, o2 = r3.GUARD_DEV, lib.HOLDOUT_GUARD_MAX_ENTRY_YEAR
    r3.GUARD_DEV = g; lib.HOLDOUT_GUARD_MAX_ENTRY_YEAR = g
    try: yield
    finally: r3.GUARD_DEV = o1; lib.HOLDOUT_GUARD_MAX_ENTRY_YEAR = o2


# =============================== 구축(final·future) ===============================
def clf_ext_build(FC, cap):
    """R3/clf_ext.py와 같은 분류기: cap까지 등록된 범위안 특허 명칭 → 디자인·범위밖 실용신안(등록일 ≤ cap) 채점"""
    kinds = [k for k in ('design', 'um') if not (FC/f'P_{k}.npy').exists()]
    if not kinds: return
    need(300, '디자인·실용신안 분류기')
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier
    kr = pd.read_parquet(CACHE/'ins_pat.parquet', columns=['title', 'ksic', 'rgst_date'])
    tr = kr[kr.rgst_date <= cap].dropna(subset=['title']); del kr
    K = sorted(pd.read_parquet(CACHE/'ins_pat.parquet', columns=['ksic']).ksic.unique())
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=5, max_features=400000, sublinear_tf=True, dtype=np.float32)
    X = vec.fit_transform(tr.title)
    clf = SGDClassifier(loss='log_loss', alpha=2e-6, max_iter=12, tol=None, random_state=0).fit(X, tr.ksic); del X
    assert list(clf.classes_) == K
    for k in kinds:
        s = pd.read_parquet(R3/'cache'/f'src_{k}.parquet'); s = s[s.rgst_date <= cap].reset_index(drop=True)
        P = np.zeros((len(s), 50), np.float16)
        for i in range(0, len(s), 200000): P[i:i + 200000] = clf.predict_proba(vec.transform(s.title.iloc[i:i + 200000])).astype(np.float16)
        np.save(FC/f'P_{k}.npy', P); s[['applicant_cd', 'rgst_no', 'ay', 'rgst_date']].to_parquet(FC/f'P_{k}_index.parquet', index=False)
    print('clf_ext saved', FC, flush=True)


def build(mode):
    M = MODES[mode]; c = M['eval_cut']; FC = M['FC']; FC.mkdir(parents=True, exist_ok=True)
    cf = dict(eval_cut=c, train_cuts=M['train_cuts'], cap=M['cap'])
    cuts = list(cf['train_cuts']) + [c]
    done = all((FC/f'feat_c{t}.parquet').exists() and (FC/f'base_app_c{t}.parquet').exists() for t in cuts) and (FC/'P_design.npy').exists() and (FC/'P_um.npy').exists()
    if done: return
    with guard(max(M['guard_train'], c)):
        _bp = lib.build_panel
        # 🔒 평가 원점 패널은 진입창 0(레이블 열 y=0, 결과변수 미계산). 학습 원점은 진입창 4년(끝 ≤ guard_train).
        fr.build_panel = lambda cut, h=5, cap=None, **kw: _bp(cut, 0 if cut == c else H, cap=cap, **kw)
        for t in cf['train_cuts']: assert t + H <= M['guard_train'], '🔒 학습 레이블 결과창 가드'
        new_clf = not all((FC/f).exists() for f in ('P_base.npy', 'P_os51.npy', 'P_ins_cf.npy', 'ep_firm.parquet'))
        P = fr.clf_base(cf, FC); P51, PO = fr.clf_os51(cf, FC); Pins, ins_idx = fr.clf_ins(cf, FC); ep = fr.ep_match(cf, FC); cdf = fr.lag_cdf(cf)
        if new_clf: sys.exit(3)                       # 분류기 단계와 패널 단계를 다른 프로세스로(메모리)
        built = False
        for t in cuts:
            if not (FC/f'feat_c{t}.parquet').exists():
                fr.cut_features(t, None if t == c else cf['cap'], cf, FC, P, P51, PO, Pins, ins_idx, ep, cdf); built = True
                sys.exit(3)                           # [3라운드 최종 실행수정] 패널 1개마다 프로세스 분리(메모리 4GB 규칙). 값 동일
        del P, P51, PO, Pins; gc.collect()
        if built: sys.exit(3)
        for t in cuts:
            if not (FC/f'base_app_c{t}.parquet').exists():
                fr.base_app_panel(t, None if t == c else cf['cap'], FC); sys.exit(3)   # [3라운드 최종 실행수정] 기준선 패널 1개마다 프로세스 분리(메모리). 값 동일
        te = pd.read_parquet(FC/f'feat_c{c}.parquet', columns=['y']); assert int(te.y.sum()) == 0, '🔒 평가 패널에 레이블이 들어 있음'
    if not ((FC/'P_design.npy').exists() and (FC/'P_um.npy').exists()):
        clf_ext_build(FC, cf['cap']); sys.exit(3)                    # [3라운드 최종 실행수정] 분류기 단계 분리(메모리). 값 동일


def prep_mode(mode):
    M = MODES[mode]; c = M['eval_cut']; FC = M['FC']; ptr, pte = FC/'prep_tr.parquet', FC/'prep_te.parquet'
    if ptr.exists() and pte.exists(): return
    cf = dict(eval_cut=c, train_cuts=M['train_cuts'], cap=M['cap']); r2.light_load()
    if not ptr.exists():
        with guard(M['guard_train']):
            tr = pd.concat([pd.read_parquet(FC/f'feat_c{t}.parquet', columns=KEEP) for t in cf['train_cuts']], ignore_index=True)
            tr = f32(tr); tr['y_reg'] = r3.reg_lab(tr, cf['cap']); tr['y'] = tr.y_reg.values
            tr['y_pub'] = r3.pub_lab(tr, cf['cap']); tr['w_ipw'] = r3.ipw4(tr, cf, tr.y_reg.values).astype(np.float32); tr['prior_pub'] = prior_app(tr, cf['cap'])
            tr.to_parquet(ptr, index=False); del tr; gc.collect()
        sys.exit(3)                                                   # [3라운드 최종 실행수정] 학습·평가 준비 단계 분리(메모리). 값 동일
    te = f32(pd.read_parquet(FC/f'feat_c{c}.parquet', columns=KEEP))
    te['y_reg'] = np.int8(0); te['y'] = np.int8(0); te['y_pub'] = np.int8(0); te['y_u'] = np.int8(0)   # 🔒 평가 레이블은 score_final()에서만
    te['prior_any'] = prior_app(te, None)
    te.to_parquet(pte, index=False)
    print('prep saved', mode, c, round(elapsed()), flush=True)


# =============================== 학습(dev·final·future 공용, 단계별 프로세스) ===============================
def train_origin(c, FCp, PTR, RD, ec):
    """R3/r3run.py run_origin과 같은 절차·같은 난수. 단계 사이 값은 parquet/npy로 넘긴다(float32 왕복 무손실)."""
    RD.mkdir(parents=True, exist_ok=True); out = RD/f'o{c}.parquet'
    if out.exists(): return
    cols = list(ec['cols'])
    s_tr, s_te, s_o, s_w = RD/f'o{c}_s1_tr.parquet', RD/f'o{c}_s1_te.parquet', RD/f'o{c}_s1_o.parquet', RD/f'o{c}_s1_w.npy'
    for kind in ('design', 'um'):
        if (FCp/f'P_{kind}.npy').exists(): r3ext.PDIR[c] = FCp
    if not s_w.exists():
        r2.light_load()
        from s2lib import FIRM_LEVEL
        allc = pq.ParquetFile(FCp/PTR).schema.names
        slimc = [x for x in dict.fromkeys(['firm', 'k', 'cut', 'newc', 'y_reg', 'y_pub', 'w_ipw', 'prior_pub', 't_lsum'] + [x for x in cols if x in FIRM_LEVEL]
                                          + ['t_max', 'ot_max', 't_top3', 'pa_k', 'ep_k', 'ti_max'] + ec.get('need_cols', [])) if x in allc]
        tr = pd.read_parquet(FCp/PTR, columns=slimc); tr['_row'] = np.arange(len(tr), dtype=np.int64)
        tec = pq.ParquetFile(FCp/'prep_te.parquet').schema.names
        te = pd.read_parquet(FCp/'prep_te.parquet', columns=[x for x in dict.fromkeys(slimc + cols + ['y_u', 'prior_any', 'y_reg', 'y_pub', 'newc', 'firm', 'k', 'cut'] + list(FIRM_LEVEL)) if x in tec])
        if ec.get('train_filter_prior'): tr = tr[~tr.prior_pub.values].reset_index(drop=True)
        n_cuts = ec.get('train_cuts_n', 4)
        if n_cuts < 4: tr = tr[tr.cut.values >= c - 7 + (4 - n_cuts)].reset_index(drop=True)
        EXT = dict(r2ext.EXT); EXT.update(r3ext.EXT); EXT.update(r3ext2.EXT); EXT.update(r3ext3.EXT)
        for g in ec.get('ext', []): tr, te = EXT[g](tr, te, c, False)
        tr['y'] = tr.y_reg.values; te['y'] = 0
        if 'k_prior' in cols or 'k_prior_new' in cols: tr, te = fr.k_prior(tr, te)
        if 'f_any' in cols:
            from s2lib import firm_stage
            base_cols = [x for x in cols if x != 'f_any']
            rest0 = [x for x in base_cols if x not in tr.columns and x in FIRM_LEVEL]
            if rest0:
                R = pq.read_table(FCp/PTR, columns=rest0).take(pa.array(tr._row.values)).to_pandas()
                for x in rest0: tr[x] = R[x].values
            trf = tr.assign(y=np.maximum(tr.y_reg.values, tr.y_pub.values)) if ec.get('f_any_label', 'reg') == 'union' else tr
            trf, te = firm_stage(trf, te, base_cols); tr['f_any'] = trf.f_any.values; del trf
        w = tr.w_ipw.values.astype(np.float64) if ec.get('ipw', 'reg') == 'reg' else np.ones(len(tr))
        w = np.where(tr.y_reg.values == 1, w, 1.0); y = tr.y_reg.values.astype(np.int8)
        pw = ec.get('publab_w', 1.0)
        if pw > 0:
            add = (tr.y_pub.values == 1) & (y == 0); y = np.maximum(y, tr.y_pub.values).astype(np.int8); w = np.where(add, pw, w)
        tr['y'] = y
        pm = ec.get('pos_mult', 1.0)
        if pm != 1.0: w = np.where(y == 1, w * pm, w)
        cw = ec.get('cut_weight_step', 0.0)
        if cw: w = w * (1.0 + cw * (tr.cut.values - tr.cut.values.min()))
        q = ec.get('neg_q')
        if q:
            rs = np.random.RandomState(1234 + c); kp = (y == 1) | (rs.rand(len(tr)) < q)
            w = np.where(y == 1, w, w / q)[kp]; tr = tr.loc[kp].reset_index(drop=True)
        rest = [x for x in cols if x not in tr.columns]
        if rest:
            R = pq.read_table(FCp/PTR, columns=rest).take(pa.array(tr._row.values)).to_pandas()
            for x in rest: tr[x] = R[x].values
            del R
        print(f'[FROZEN o{c}] train rows {len(tr)} pos {int(tr.y.sum())} | eval rows {len(te)} | peak {peak_gb():.2f}GB', flush=True)
        te[['firm', 'k', 'cut', 'newc', 'y_reg', 'y_u', 'prior_any']].to_parquet(s_o, index=False)
        te[cols].to_parquet(s_te, index=False); tr[cols + ['y']].to_parquet(s_tr, index=False)
        np.save(s_w, w); sys.exit(3)                                                  # 외부 피처 단계 종료(메모리)
    fl = RD/f'o{c}_pl.npy'
    if not fl.exists():
        if ec.get('blend_w', 0.9) < 1.0:
            need(150, '로지스틱')
            tr = pd.read_parquet(s_tr); L = fr.logit_fit(tr[cols + ['y']], cols); del tr; gc.collect()
            te = pd.read_parquet(s_te)
            np.save(fl, np.concatenate([fr.logit_pred(L, te.iloc[i:i + 200000]) for i in range(0, len(te), 200000)]).astype(np.float32))
        else: np.save(fl, np.zeros(pq.ParquetFile(s_te).metadata.num_rows, np.float32))
        sys.exit(3)                                                                   # 로지스틱 단계 종료(메모리)
    import xgboost as xgb
    params = dict(SPEC1['params']); params.update(ec.get('params', {}))
    pxs = []
    for sd in ec['seeds']:
        fp = RD/f'o{c}_px{sd}.npy'
        if not fp.exists():
            tr = pd.read_parquet(s_tr); w = np.load(s_w)
            est = 125.0 * len(tr) / 877070 * params['n_estimators'] / 800 * (params.get('max_depth', 6) / 6)
            need(est, f'XGB seed {sd} (예상 {est:.0f}s)')
            X = tr[cols].values.astype(np.float32); yv = tr.y.values; del tr
            p = dict(tree_method='hist', nthread=2, random_state=sd); p.update(params)
            m = xgb.XGBClassifier(**p); m.fit(X, yv, sample_weight=w); del X; gc.collect()
            te = pd.read_parquet(s_te)
            np.save(fp, np.concatenate([m.predict_proba(te[cols].iloc[i:i + 200000].values.astype(np.float32))[:, 1] for i in range(0, len(te), 200000)]).astype(np.float32))
            if sd == ec['seeds'][0]:
                imp = pd.Series(m.get_booster().get_score(importance_type='gain')); imp.index = [cols[int(i[1:])] for i in imp.index]
                imp.sort_values(ascending=False).to_csv(RD/f'o{c}_gain.csv', encoding='utf-8-sig')
            print('seed', sd, round(elapsed()), f'peak {peak_gb():.2f}GB', flush=True); del te, m; gc.collect()
        pxs.append(np.load(fp))
    o = pd.read_parquet(s_o); o['px'] = np.mean(pxs, 0); o['pl'] = np.load(fl)
    o.to_parquet(out, index=False)
    for f in (s_tr, s_te): f.unlink()
    print('saved', out, round(elapsed()), f'peak {peak_gb():.2f}GB', flush=True)


def blend(o, ec=EC):
    bw = ec.get('blend_w', 0.9); return bw * fr.rank(o.px.values) + (1 - bw) * fr.rank(o.pl.values)


# =============================== 사전등록 보조 변형 ===============================
def nts_closed(cut):
    """국세청 폐업일 ≤ 원점 연말 법인(되돌릴 수 없는 사건 날짜, 2013년 이전 좌측 절단)"""
    s = pd.read_parquet(EXTD/'r3nts_applicant_status.parquet', columns=['applicantCd', 'end_dt']).dropna(subset=['end_dt'])
    return set(s[s.end_dt <= pd.Timestamp(f'{cut}-12-31')].applicantCd.astype(str))


NPS = ['nps_link', 'nps_in', 'nps_lemp', 'nps_hire', 'nps_loss', 'nps_age']
def nps_feats(firm, c):
    """국민연금 사업장 원점 연말 스냅샷(자료생성 11월) — R3/r3aux.py와 같은 6피처"""
    P = pd.read_parquet(EXTD/'firm_nps_firm_panel.parquet'); P = P[P.snap == f'{c}-12'].set_index('applicantCd')
    assert len(P) > 0, f'국민연금 {c}-12 스냅샷 없음'
    L = pd.read_parquet(EXTD/'firm_nps_applicant_link.parquet').drop_duplicates('applicantCd').set_index('applicantCd')
    f = pd.Series(firm); X = pd.DataFrame(index=f.index)
    X['nps_link'] = f.isin(L.index).astype(float); X['nps_in'] = f.isin(P.index).astype(float)
    emp = f.map(P.n_emp).fillna(0).values; X['nps_lemp'] = np.log1p(emp)
    X['nps_hire'] = np.where(emp > 0, f.map(P.n_new).fillna(0).values / np.maximum(emp, 1), 0)
    X['nps_loss'] = np.where(emp > 0, f.map(P.n_loss).fillna(0).values / np.maximum(emp, 1), 0)
    X['nps_age'] = np.where(X.nps_in > 0, c - pd.to_numeric(f.map(P.first_adpt).astype(str).str[:4], errors='coerce').fillna(c).values, 0)
    return X


def logit_of(s):
    s = np.clip(s, 1e-6, 1 - 1e-6); return np.log(s / (1 - s))


def fit_nps_layer(o):
    """보조 원점 c=2015(레이블 2016–19, 최종시험과 겹치지 않음)에서 '주 모형 점수 로짓 + NPS 6피처' 로지스틱 적합 → 계수 동결"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    o = o[~o.prior_any.values].reset_index(drop=True); s = blend(o)
    X = nps_feats(o.firm.values, 2015); X.insert(0, 'ls', logit_of(s)); cols = list(X.columns)
    sc = StandardScaler().fit(X.values); m = LogisticRegression(max_iter=2000).fit(sc.transform(X.values), o.y_u.values)
    # 기록용: 법인 해시 2-fold 교차적합 AUC(주 모형 점수만 vs +NPS)
    fold = (pd.util.hash_array(o.firm.values.astype(object)) % 2).astype(int); cf = {}
    for nm, cc in (('score_only', ['ls']), ('score+nps', cols)):
        p = np.zeros(len(o))
        for fo in (0, 1):
            t = fold != fo; s2 = StandardScaler().fit(X[cc].values[t]); mm = LogisticRegression(max_iter=2000).fit(s2.transform(X[cc].values[t]), o.y_u.values[t])
            p[~t] = mm.predict_proba(s2.transform(X[cc].values[~t]))[:, 1]
        cf[nm] = float(roc_auc_score(o.y_u.values, p))
    lay = dict(cols=cols, mean=sc.mean_.tolist(), scale=sc.scale_.tolist(), coef=m.coef_[0].tolist(), intercept=float(m.intercept_[0]),
               fit_origin=2015, fit_label='공개 기준 진입 2016–2019', n=int(len(o)), pos=int(o.y_u.sum()), crossfit_auc=cf)
    json.dump(lay, open(R3/'nps_layer_r3.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    return lay


def apply_nps_layer(o_keep, s, c):
    lay = json.load(open(R3/'nps_layer_r3.json', encoding='utf-8'))
    X = nps_feats(o_keep.firm.values, c); X.insert(0, 'ls', logit_of(s)); X = X[lay['cols']].values
    z = ((X - np.array(lay['mean'])) / np.array(lay['scale'])) @ np.array(lay['coef']) + lay['intercept']
    return 1 / (1 + np.exp(-z))


# =============================== 채점 ===============================
def base_scores_mode(mode):
    """고정 기준선 6변형(로지스틱 스택·마지막 원점·XGB × 학습 레이블 등록/공개)을 최종 캐시에서 학습·예측(학습 레이블 cap·가드 = 학습 규칙)"""
    M = MODES[mode]; FC = M['FC']; p = FC/'baseline_scores.parquet'
    if p.exists(): return pd.read_parquet(p)
    import xgboost as xgb
    need(240, '기준선')
    cf = dict(eval_cut=M['eval_cut'], train_cuts=M['train_cuts'], cap=M['cap']); r2.light_load()
    btr = pd.concat([pd.read_parquet(FC/f'base_app_c{t}.parquet') for t in cf['train_cuts']], ignore_index=True)
    bte = pd.read_parquet(FC/f'base_app_c{M["eval_cut"]}.parquet')
    te = pd.read_parquet(FC/'prep_te.parquet', columns=['firm', 'k'])
    assert (bte.firm.values == te.firm.values).all() and (bte.k.values == te.k.values).all()
    with guard(M['guard_train']):
        btr['y'] = r3.reg_lab(btr, cf['cap']); btr['y_pub'] = r3.pub_lab(btr, cf['cap'])
    btr['y_u'] = np.maximum(btr.y.values, btr.y_pub.values)
    S = {}
    for lab in ('y', 'y_u'):
        t2 = btr.assign(y=btr[lab].values)
        S[f'logit_stack_{lab}'] = fr.logit_pred(fr.logit_fit(t2, BASE), bte)
        S[f'logit_single_{lab}'] = fr.logit_pred(fr.logit_fit(t2[t2.cut == max(cf['train_cuts'])], BASE), bte)
        xb = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, tree_method='hist', nthread=2, random_state=0)
        xb.fit(t2[BASE].values.astype(np.float32), t2.y.values); S[f'xgb_{lab}'] = xb.predict_proba(bte[BASE].values.astype(np.float32))[:, 1]
    df = pd.DataFrame({kk: v.astype(np.float32) for kk, v in S.items()}); df.to_parquet(p, index=False); return df


def boot_ci(y, nc, firm, M_, Bs, FC, R=200, chunk=10):
    """법인 군집 부트스트랩: 법인을 복원추출하고 그 법인의 모든 쌍을 함께 넣는다. 재표본마다 기준선은 6변형 중 최강값. M_: 모형 점수 dict, Bs: 기준선 점수 dict"""
    fr_, inv = np.unique(firm, return_inverse=True)
    order = np.argsort(inv, kind='stable'); st = np.searchsorted(inv[order], np.arange(len(fr_))); en = np.searchsorted(inv[order], np.arange(len(fr_)), side='right')
    keys = list(M_) + list(Bs); S = dict(M_); S.update(Bs); nm = len(M_); parts = []
    for ci in range(int(np.ceil(R / chunk))):
        fp = FC/f'r3boot_R{R}_{ci}.npy'
        if fp.exists(): parts.append(np.load(fp)); continue
        need(chunk * 0.6 * len(keys) * len(y) / 1e6, f'부트스트랩 chunk {ci}')
        rng = np.random.RandomState(3000 + ci); arr = np.zeros((min(chunk, R - ci * chunk), len(keys), 2))
        for r in range(arr.shape[0]):
            idx = np.concatenate([order[st[i]:en[i]] for i in rng.randint(0, len(fr_), len(fr_))]); yy = y[idx]; nn = nc[idx]
            for j, kk in enumerate(keys):
                ss = S[kk][idx]; arr[r, j] = [roc_auc_score(yy, ss), roc_auc_score(yy[nn], ss[nn])]
        np.save(fp, arr); parts.append(arr)
    A = np.concatenate(parts); q = lambda v: [float(np.percentile(v, 2.5)), float(np.median(v)), float(np.percentile(v, 97.5))]
    out = {kk: {'auc_all': q(A[:, j, 0]), 'auc_new': q(A[:, j, 1])} for j, kk in enumerate(keys)}
    bmax = A[:, nm:, :].max(1)
    for j, kk in enumerate(M_):
        out[f'skill_vs_strongest__{kk}'] = {'auc_all': q((A[:, j, 0] - bmax[:, 0]) / (1 - bmax[:, 0])), 'auc_new': q((A[:, j, 1] - bmax[:, 1]) / (1 - bmax[:, 1]))}
        out[f'diff_minus_strongest__{kk}'] = {'auc_all': q(A[:, j, 0] - bmax[:, 0]), 'auc_new': q(A[:, j, 1] - bmax[:, 1])}
    for j, kk in enumerate(list(M_)[1:], start=1):
        out[f'diff_vs_main__{kk}'] = {'auc_all': q(A[:, j, 0] - A[:, 0, 0]), 'auc_new': q(A[:, j, 1] - A[:, 0, 1])}
    return out


def topk(y, s, ks=(100, 1000, 10000)):
    o = np.argsort(-s); return {str(k): dict(hits=int(y[o[:k]].sum()), precision=float(y[o[:k]].mean())) for k in ks}


def score_final(boot=200):
    M = MODES['final']; c = M['eval_cut']; FC = M['FC']; RD = M['RD']
    o = pd.read_parquet(RD/f'o{c}.parquet')
    te = pd.read_parquet(FC/'prep_te.parquet', columns=['firm', 'k', 'cut', 'newc', 'prior_any'])
    assert (te.firm.values == o.firm.values).all() and (te.k.values == o.k.values).all()
    r2.light_load()
    with guard(M['guard_eval']):                                   # 🔒 여기서만 평가 레이블(2020–2023 진입) 계산
        te['y_reg'] = r3.reg_lab(te, None); te['y_pub'] = r3.pub_lab(te, END)
    te['y_u'] = np.maximum(te.y_reg.values, te.y_pub.values)
    keep = ~te.prior_any.values; t = te[keep].reset_index(drop=True); oo = o[keep].reset_index(drop=True)
    s = blend(oo); y = t.y_u.values; nc = t.newc.values.astype(bool)
    bs = base_scores_mode('final').loc[keep].reset_index(drop=True)
    mm = mets(y, s, nc); mm['auc_all_reglabel'] = float(roc_auc_score(t.y_reg.values, s))
    bm = {kk: mets(y, bs[kk].values, nc) for kk in bs.columns}
    strongest = {m: max(v[m] for v in bm.values()) for m in ('auc_all', 'auc_new', 'pr_all', 'pr_new')}
    # 사전등록 보조 변형(공식 성적 아님)
    closed = nts_closed(c); cm = t.firm.astype(str).isin(closed).values
    s_nts = np.where(cm, -1.0, s)
    V = {'model_nts': s_nts}
    try:
        s_nps = apply_nps_layer(t, s, c); V['model_nps'] = s_nps; V['model_nps_nts'] = np.where(cm, -1.0, s_nps)
    except Exception as ex: print('NPS 보정층 미적용:', repr(ex), flush=True)
    vm = {kk: mets(y, v, nc) for kk, v in V.items()}
    res = dict(mode='final', spec=SPEC['exp_id'], origin=c, window=M['window'], F=M['F'], h=H,
               n=dict(rows=int(len(t)), firms=int(t.firm.nunique()), pos=int(y.sum()), new_rows=int(nc.sum()), new_pos=int(y[nc].sum()),
                      pos_reg=int(t.y_reg.sum()), removed_prior_pairs=int((~keep).sum()), nts_closed_pairs=int(cm.sum()), nts_closed_firms=int(t.firm[cm].nunique())),
               model=mm, baselines=bm, baseline_strongest=strongest,
               skill=dict(auc_all=skill(mm['auc_all'], strongest['auc_all']), auc_new=skill(mm['auc_new'], strongest['auc_new'])),
               topk=dict(model=topk(y, s), strongest_auc_baseline=topk(y, bs[max(bm, key=lambda q: bm[q]['auc_all'])].values)),
               secondary_preregistered={kk: dict(metrics=v, skill_auc_all=skill(v['auc_all'], strongest['auc_all']), skill_auc_new=skill(v['auc_new'], strongest['auc_new'])) for kk, v in vm.items()},
               note='공식 성적 = model(주 모형) skill. secondary_preregistered는 사전등록 보조 변형(국세청 폐업 정제·국민연금 보정층).')
    if boot > 0:
        Ms = {'model': s}; Ms.update(V)
        res['bootstrap_firm_cluster'] = boot_ci(y, nc, t.firm.values, Ms, {kk: bs[kk].values for kk in bs.columns}, FC, R=boot); res['bootstrap_R'] = boot
    json.dump(res, open(R3/'final_result_r3.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print(json.dumps(res['skill'])); print('saved final_result_r3.json')


def forecast_future():
    M = MODES['future']; c = M['eval_cut']; o = pd.read_parquet(M['RD']/f'o{c}.parquet')
    te = o[~o.prior_any.values].copy(); te['score'] = blend(te)
    closed = nts_closed(c); te = te[~te.firm.astype(str).isin(closed)]           # 폐업 법인 제외(원점 연말 기준)
    K = lib.load()['K'] if 'K' in lib._cache else sorted(pd.read_parquet(CACHE/'ins_pat.parquet', columns=['ksic']).ksic.unique())
    te['ksic'] = [K[i] for i in te.k.values]
    fm = pd.read_parquet(CACHE/'firms.parquet').set_index('applicant_cd').applicant_name; te['firm_name'] = te.firm.map(fm)
    try:
        js = json.load(open(B/'공모전데이터/외부데이터/KSIC해설/ksic50_definitions.json', encoding='utf-8'))
        nm = {str(x.get('code', x.get('ksic', ''))): (x.get('codenmKr') or x.get('name', '')) for x in js} if isinstance(js, list) else {str(k2): (v.get('name') if isinstance(v, dict) else str(v)) for k2, v in js.items()}
    except Exception: nm = {}
    g = te.groupby('ksic').agg(at_risk=('px', 'size'), waiting_room_index=('px', 'sum'), mean_prob=('px', 'mean'))
    g['index_newcomer'] = te[te.newc].groupby('ksic').px.sum().reindex(g.index).fillna(0)
    g['index_share'] = g.waiting_room_index / g.waiting_room_index.sum(); g['ksic_name'] = [nm.get(k2, '') for k2 in g.index]
    g['top5_candidates'] = te.sort_values('score', ascending=False).groupby('ksic').head(5).groupby('ksic').firm_name.apply(lambda x: ' | '.join(map(str, x)))
    g['window'] = M['window']; g['info_cutoff'] = f'{c}-12-31 등록·공개분'
    g.sort_values('waiting_room_index', ascending=False).reset_index().to_csv(R3/'future_forecast_r3.csv', index=False, encoding='utf-8-sig')
    print('saved future_forecast_r3.csv')


# =============================== main ===============================
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--mode', required=True, choices=['dev', 'final', 'future']); ap.add_argument('--boot', type=int, default=200)
    a = ap.parse_args(); mode = a.mode
    if mode == 'dev':
        RD = R3/'runs'/'FROZEN_dev'; ptr = 'prep_tr5.parquet' if EC.get('train_cuts_n', 4) == 5 else 'prep_tr.parquet'
        for c in ORIGINS + AUX:
            if not (RD/f'o{c}.parquet').exists():
                if ptr == 'prep_tr5.parquet': prep5(c)
                train_origin(c, fc_r3(c), ptr, RD, EC)
                sys.exit(3)
        rows = {}; ref = json.load(open(R3/'runs'/SPEC['exp_id']/'summary.json', encoding='utf-8'))['per_origin']
        for c in ORIGINS + AUX:
            o = pd.read_parquet(RD/f'o{c}.parquet'); f = o[~o.prior_any.values]; s = blend(f)
            m = mets(f.y_u.values, s, f.newc.values); m['auc_all_reglabel'] = float(roc_auc_score(f.y_reg.values, s)); b = baselines(c)['strongest']
            m['baseline_strongest_auc_all'] = b['auc_all']; m['baseline_strongest_auc_new'] = b['auc_new']
            m['skill'] = skill(m['auc_all'], b['auc_all']); m['skill_new'] = skill(m['auc_new'], b['auc_new'])
            if str(c) in ref: m['experiment_skill'] = ref[str(c)]['skill']; m['repro_diff'] = m['skill'] - ref[str(c)]['skill']
            closed = nts_closed(c); cmk = f.firm.astype(str).isin(closed).values
            m['nts_closed_pairs'] = int(cmk.sum()); m['skill_nts'] = skill(roc_auc_score(f.y_u.values, np.where(cmk, -1.0, s)), b['auc_all'])
            rows[c] = m
        lay = fit_nps_layer(pd.read_parquet(RD/'o2015.parquet'))
        res = dict(mode='dev', spec=SPEC['exp_id'], per_origin=rows, mean_skill=float(np.mean([rows[c]['skill'] for c in ORIGINS])),
                   mean_skill_new=float(np.mean([rows[c]['skill_new'] for c in ORIGINS])), mean_auc_all=float(np.mean([rows[c]['auc_all'] for c in ORIGINS])),
                   aux_c2015=dict(skill=rows[2015]['skill'], skill_nts=rows[2015]['skill_nts'], nps_layer_crossfit_auc=lay['crossfit_auc'],
                                  nps_layer_crossfit_skill={k: skill(v, rows[2015]['baseline_strongest_auc_all']) for k, v in lay['crossfit_auc'].items()}),
                   max_abs_repro_diff=float(max(abs(rows[c].get('repro_diff', 0.0)) for c in ORIGINS)))
        json.dump(res, open(R3/'dev_result_r3.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
        print(json.dumps({k: v for k, v in res.items() if k != 'per_origin'}, ensure_ascii=False)); return
    M = MODES[mode]; c = M['eval_cut']
    if mode == 'final': assert c == 2019 and all(t + H <= 2019 for t in M['train_cuts']), '🔒 R3 최종시험 규칙'
    if not (M['FC']/'prep_te.parquet').exists():
        build(mode); prep_mode(mode); sys.exit(3)                        # 구축 단계는 별도 프로세스(메모리)
    if not (M['RD']/f'o{c}.parquet').exists():
        train_origin(c, M['FC'], 'prep_tr.parquet', M['RD'], EC); sys.exit(3)
    if mode == 'final': score_final(a.boot)
    else: forecast_future()


if __name__ == '__main__':
    main()
