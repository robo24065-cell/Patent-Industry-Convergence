# -*- coding: utf-8 -*-
"""재계산판 전체 재구축(PREREG §2): C16-T1 c1_build.py(FULL 모드)와 같은 절차 + 원천 주입/법인 제외.
 python c2_build.py <mode> <feat|ext|pred|cmp>
 mode: ctrlF(주입 장치를 켜되 빈 주입 → 공식과 차 0이어야 함), n1F(N1 비가시 신규진입자 법인 제외), n2F(신규진입자 전부 제외),
       rcF(RC:001 법인 코드 2,465개 범위안·이름매칭·유형 주입), s1rcF(rcF + C16 Call 쌍 + RC S1 Call 쌍)
 주입: pd.read_parquet을 감싸 트랙 캐시 ins_firm·app_pub·firm_meta를 읽을 때 RC 행을 뒤에 붙인다(트랙 파일은 그대로)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from c17t1 import *
import gc, pickle, time
mode = sys.argv[1]; stage = sys.argv[2]
assert mode in ('ctrlF', 'n1F', 'n2F', 'rcF', 's1rcF')
BD = CC/'build'; BD.mkdir(exist_ok=True); OUT = BD/mode; OUT.mkdir(exist_ok=True)
INJ = mode in ('rcF', 's1rcF')
# ---- 주입 장치
_rp = pd.read_parquet
TGT = {(CACHE/'ins_firm.parquet').resolve(): CC/'rc_ins.parquet', (CACHE/'app_pub.parquet').resolve(): CC/'rc_app.parquet', (CACHE/'firm_meta.parquet').resolve(): CC/'rc_fm.parquet'}
NINJ = {}


def rp(path, *a, **k):
    df = _rp(path, *a, **k)
    try: rpth = pathlib.Path(path).resolve()
    except Exception: return df
    if rpth in TGT:
        add = _rp(TGT[rpth], *a, **k) if INJ else _rp(TGT[rpth], *a, **k).iloc[:0]
        NINJ[rpth.name] = NINJ.get(rpth.name, 0) + len(add)
        df = pd.concat([df, add[df.columns]], ignore_index=True)
    return df


pd.read_parquet = rp
EXCL = set()
if mode in ('n1F', 'n2F'):
    E = pd.read_parquet(CC/'evid_all.parquet'); E = E[E.is_newc_off]
    EXCL = set(E.firm) if mode == 'n2F' else set(E.firm[~E.N1_primary])
EXTRA = set()
if mode == 's1rcF':
    EXTRA = set(map(tuple, pickle.load(open(C16C/'build'/'pop.pkl', 'rb'))['Call'])) | set(map(tuple, pickle.load(open(CC/'rc_s1.pkl', 'rb'))['Call']))
import final_run_r3 as F, r3, r2, lib, r2ext, r3ext, r3ext2, r3ext3
from r2 import fr
cf = dict(eval_cut=2019, train_cuts=(2012, 2013, 2014, 2015), cap='20191231')
ec = json.loads(json.dumps(F.EC)); cols = list(ec['cols'])
t0 = time.time()


def build_panel_c17(cut, h=5, cap=None, min_pat=3, fcap=None):
    """lib.build_panel 사본(C16 build_panel_c16 FULL과 같음). 바뀐 곳 [C17] 두 군데: 법인 제외, 추가 위험쌍."""
    if lib.STRICT and fcap is None: fcap = f'{cut}1231'
    assert cut + h <= lib.HOLDOUT_GUARD_MAX_ENTRY_YEAR, '🔒 홀드아웃 진입창 접근 금지'
    D = lib.load(); ins, oos, K, kidx = D['ins'], D['oos'], D['K'], D['kidx']
    if cap is not None:
        ins = ins[ins.rgst_date <= cap]; oos = oos[oos.rgst_date <= cap]
    ip = ins[ins.ay <= cut]; op = oos[oos.ay <= cut]
    cnt = pd.concat([ip.drop_duplicates(['applicant_cd', 'rgst_no']).applicant_cd, op.drop_duplicates(['applicant_cd', 'rgst_no']).applicant_cd]).value_counts()
    firms = np.array(sorted(set(cnt[cnt >= min_pat].index) - EXCL))                     # [C17] 법인 제외
    fi = pd.Series(np.arange(len(firms)), index=firms); nF = len(firms)
    ip = ip[ip.applicant_cd.isin(fi.index)]; op = op[op.applicant_cd.isin(fi.index)]
    M = np.zeros((nF, 50), np.float32)
    np.add.at(M, (fi[ip.applicant_cd].values, ip.ksic.map(kidx).values), 1)
    allp = ins[ins.ay <= cut]
    A = pd.crosstab(allp.applicant_cd, allp.ksic).reindex(columns=K, fill_value=0).values > 0
    A = A.astype(np.float32); co = A.T @ A; d = np.diag(co).copy()
    prox = co / np.maximum(np.maximum.outer(d, d), 1); np.fill_diagonal(prox, 0)
    Bm = (M > 0).astype(np.float32)
    dens = (Bm @ prox) / np.maximum(prox.sum(0), 1e-9)
    kpop = np.log1p(A.sum(0))
    fut = ins[(ins.ay > cut) & (ins.ay <= cut + h) & ins.applicant_cd.isin(fi.index)]
    Y = np.zeros((nF, 50), np.int8); NF = np.zeros((nF, 50), np.int16)
    np.add.at(NF, (fi[fut.applicant_cd].values, fut.ksic.map(kidx).values), 1)
    Y[NF > 0] = 1
    Mrisk = M
    if fcap is not None:
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
    EX = np.zeros((nF, 50), bool)                                                           # [C17] 추가 위험쌍
    for (f_, k_) in EXTRA:
        if f_ in fi.index: EX[fi[f_], k_] = True
    f, k = np.nonzero((Mrisk == 0) | EX)
    df = pd.DataFrame({'fidx': f, 'k': k, 'firm': firms[f],
        'lsize': np.log1p(nin[f] + nout[f]), 'lin': np.log1p(nin[f]), 'lout': np.log1p(nout[f]),
        'inshare': nin[f] / np.maximum(nin[f] + nout[f], 1), 'dens': dens[f, k], 'kpop': kpop[k],
        'y': Y[f, k], 'nfut': NF[f, k], 'newc': (Mrisk.sum(1)[f] == 0)})
    ctx = dict(cut=cut, h=h, cap=cap, firms=firms, fi=fi, M=M, prox=prox, op=op, ip=ip, nout=nout, nin=nin, A=A)
    return df, ctx


if stage == 'feat':
    if (OUT/'feat_c2019.parquet').exists() and (OUT/'base_app_c2019.parquet').exists(): print('exists'); sys.exit(0)
    D = lib.load(); oos = D['oos']
    P = np.load(FC/'P_base.npy', mmap_mode='r')[oos.row.values].astype(np.float32)
    P51 = np.load(FC/'P_os51.npy', mmap_mode='r')[oos.row.values].astype(np.float32); PO = np.load(FC/'PO_os51.npy', mmap_mode='r')[oos.row.values].astype(np.float32)
    Pins, ins_idx = fr.clf_ins(cf, FC); ep = fr.ep_match(cf, FC); cdf = fr.lag_cdf(cf)
    print('loaded', len(oos), 'ins', len(D['ins']), 'inj', NINJ, G.rss_now(), flush=True)
    fr.build_panel = lambda cut, h=5, cap=None, **kw: build_panel_c17(cut, 0, cap=cap, **kw)
    with F.guard(2019):
        if not (OUT/'feat_c2019.parquet').exists(): fr.cut_features(2019, None, cf, OUT, P, P51, PO, Pins, ins_idx, ep, cdf)
        if not (OUT/'base_app_c2019.parquet').exists(): fr.base_app_panel(2019, None, OUT)
    print('feat done', mode, round(time.time() - t0), 'inj', NINJ, G.rss_peak(), flush=True)

elif stage == 'ext':
    from s2lib import FIRM_LEVEL
    if (OUT/'te_ext.parquet').exists(): print('exists'); sys.exit(0)
    te = r2.f32(pd.read_parquet(OUT/'feat_c2019.parquet', columns=r2.KEEP))
    r2.light_load()
    te['y_reg'] = np.int8(0); te['y'] = np.int8(0); te['y_pub'] = np.int8(0); te['y_u'] = np.int8(0)
    te['prior_any'] = r2.prior_app(te, None)
    for kind in ('design', 'um'):
        if (FC/f'P_{kind}.npy').exists(): r3ext.PDIR[2019] = FC
    EXT = dict(r2ext.EXT); EXT.update(r3ext.EXT); EXT.update(r3ext2.EXT); EXT.update(r3ext3.EXT)
    trd = te.iloc[:3000].copy()
    for g in ec['ext']:
        trd, te = EXT[g](trd, te, 2019, False); gc.collect()
        print('ext', g, round(time.time() - t0), G.rss_now(), flush=True)
    del trd; gc.collect()
    extra_c = ['t_max', 'ot_max', 't_top3', 'pa_k', 'ep_k', 'ti_max', 'lsize', 'lin', 'lout', 'inshare', 'dens', 'kpop', 'n_last1', 'gap_last', 'now1', 'now3', 'now_grow', 'k_grow', 'k_newent', 'vis_frac']
    keepc = [x for x in dict.fromkeys(['firm', 'k', 'cut', 'newc', 'prior_any'] + [x for x in cols if x in te.columns] + list(FIRM_LEVEL) + extra_c) if x in te.columns]
    te[keepc].to_parquet(OUT/'te_ext.parquet', index=False)
    print('te_ext saved', len(te), round(time.time() - t0), 'inj', NINJ, G.rss_peak(), flush=True)

elif stage == 'pred':
    import xgboost as xgb
    from s2lib import FIRM_LEVEL
    if (OUT/'scores.parquet').exists(): print('exists'); sys.exit(0)
    te = pd.read_parquet(OUT/'te_ext.parquet'); te['y'] = 0
    ref = pd.read_parquet(P0D/'s1_te.parquet', columns=['k', 'k_prior', 'k_prior_new']).drop_duplicates('k').set_index('k'); assert len(ref) == 50
    te['k_prior'] = te.k.map(ref.k_prior).values; te['k_prior_new'] = te.k.map(ref.k_prior_new).values
    base_cols = [x for x in cols if x != 'f_any']; fcols = [x for x in base_cols if x in FIRM_LEVEL]
    g_ = te.groupby(['cut', 'firm'], sort=False)
    Xf = g_[fcols].first(); Xf['n_risk'] = g_.size(); Xf['y'] = g_.y.max()
    for x in ('t_max', 'ot_max', 't_top3', 'pa_k', 'ep_k', 'ti_max'):
        if x in te: Xf[x + '_fmax'] = g_[x].max()
    fm = json.load(open(MD/'firm_seed0_meta.json', encoding='utf-8')); assert list(Xf.drop(columns='y').columns) == fm['xc']
    bf = xgb.Booster(); bf.load_model(str(MD/'firm_seed0.json'))
    ote = bf.predict(xgb.DMatrix(Xf[fm['xc']].values.astype(np.float32)))
    te['f_any'] = pd.Series(ote, index=Xf.index).reindex(pd.MultiIndex.from_arrays([te.cut.values, te.firm.values])).values
    out = te[['firm', 'k', 'newc', 'prior_any']].copy(); pxs = []
    for sd in (0, 1):
        b = xgb.Booster(); b.load_model(str(MD/f'xgb_seed{sd}.json')); b.set_param({'nthread': 2})
        pxs.append(np.concatenate([b.predict(xgb.DMatrix(te[cols].iloc[i:i + 200000].values.astype(np.float32))) for i in range(0, len(te), 200000)]).astype(np.float32))
    L = pickle.load(open(MD/'logit.pkl', 'rb')); pl = np.concatenate([fr.logit_pred(L, te.iloc[i:i + 200000]) for i in range(0, len(te), 200000)]).astype(np.float32)
    out['px0'] = pxs[0]; out['px1'] = pxs[1]; out['px'] = np.mean(pxs, 0).astype(np.float32); out['pl'] = pl
    out['is_extra'] = pd.MultiIndex.from_arrays([out.firm.values, out.k.values]).isin(list(EXTRA)) if EXTRA else False
    SBBF = ['lsize', 'lin', 'lout', 'inshare', 'dens', 'kpop', 'k', 'newc', 'n_last1', 'gap_last', 'now1', 'now3', 'now_grow', 'k_grow', 'k_newent', 'vis_frac']
    b = xgb.Booster(); b.load_model(str(MD/'sbb.json')); out['sbb'] = b.predict(xgb.DMatrix(te[SBBF].values.astype(np.float32))).astype(np.float32)
    del te; gc.collect()
    bte = pd.read_parquet(OUT/'base_app_c2019.parquet'); assert (bte.firm.values == out.firm.values).all() and (bte.k.values == out.k.values).all()
    BASE = list(r3.BASE)
    for lab in ('y', 'y_u'):
        for nm in ('stack', 'single'):
            Lb = pickle.load(open(MD/f'base_logit_{nm}_{lab}.pkl', 'rb')); out[f'logit_{nm}_{lab}'] = fr.logit_pred(Lb, bte).astype(np.float32)
        b = xgb.Booster(); b.load_model(str(MD/f'base_xgb_{lab}.json')); out[f'xgb_{lab}'] = b.predict(xgb.DMatrix(bte[BASE].values.astype(np.float32))).astype(np.float32)
    out.to_parquet(OUT/'scores.parquet', index=False)
    print('pred done', mode, len(out), int(np.sum(out.is_extra)), round(time.time() - t0), G.rss_peak(), flush=True)

elif stage == 'cmp':
    # 공식 행 점수 대조: ctrlF는 차 0이어야 함, 다른 모드는 변화 규모 기록
    key = lambda d: pd.MultiIndex.from_arrays([d.firm.values, d.k.values])
    sc = pd.read_parquet(OUT/'scores.parquet'); O = official_frame()
    pos = pd.Series(np.arange(len(sc)), index=key(sc)).reindex(key(O))
    found = pos.notna().values; p = pos.values[found].astype(np.int64)
    res = dict(mode=mode, n_rows=int(len(sc)), n_firms=int(sc.firm.nunique()), official_rows_found=int(found.sum()), official_rows=int(len(O)),
               new_rows=int(len(sc) - found.sum()), new_firms=int(sc.firm[~key(sc).isin(key(O))].nunique()))
    res['score_maxdiff_official_rows'] = {c: float(np.abs(sc[c].values[p].astype(float) - O[c].values[found].astype(float)).max()) for c in ['px', 'pl', 'sbb'] + BK}
    res['prior_any_eq'] = bool((sc.prior_any.values[p] == O.prior_any.values[found]).all())
    res['newc_eq'] = bool((sc.newc.values[p] == O.newc.values[found]).all())
    d = np.abs(sc.px.values[p] - O.px.values[found]); res['px_rows_changed'] = int((d > 0).sum()); res['px_rows_changed_gt1e-4'] = int((d > 1e-4).sum())
    if mode == 'ctrlF':
        res['pass'] = bool(res['official_rows_found'] == len(O) and res['new_rows'] == 0 and max(res['score_maxdiff_official_rows'].values()) <= 1e-6 and res['prior_any_eq'])
    jdump(res, BD/f'{mode}_compare.json'); print(json.dumps(res, ensure_ascii=False, indent=1))
