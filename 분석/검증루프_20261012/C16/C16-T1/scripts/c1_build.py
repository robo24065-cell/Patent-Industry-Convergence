# -*- coding: utf-8 -*-
"""C 쌍(공식 점수 없음) 피처·점수: 트랙 동결 코드(final_run.cut_features·base_app_panel·EXT 11종·법인 단계)를 C 법인으로 제한한 패널에 그대로 적용.
 python c1_build.py pop                  → cache/build/pop.pkl (v_pub 기준 Call·C·대상 법인)
 python c1_build.py <ctrl|s1> feat       → cache/build/<mode>/feat_c2019.parquet, base_app_c2019.parquet
 python c1_build.py <ctrl|s1> ext        → cache/build/<mode>/scores.parquet (px0·px1·pl·base6·sbb)
 python c1_build.py ctrl cmp             → cache/build/ctrl_compare.json (공식값과 대조, PREREG §2 필수)
 ctrl = 같은 법인·공식 위험집합(추가 쌍 없음), s1 = 같은 법인 + Call 쌍(가시 기준 위험집합)
 build_panel 사본(lib.build_panel과 같은 코드, 바뀐 곳: 법인 목록을 대상 법인으로 고정, 위험 마스크에 추가 쌍 OR)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from c16t1 import *
import gc, pickle, time
mode = sys.argv[1]; stage = sys.argv[2] if len(sys.argv) > 2 else ''
FULL = mode.endswith('F')          # ctrlF·s1F: 공식 전체 법인으로 구축(행렬곱 크기·이웃 탐색 청크를 공식과 같게 → 비트 대조용)
BD = CC/'build'; BD.mkdir(exist_ok=True)
if mode == 'pop':
    from popdef import pop
    P_ = pop('v_pub'); firms_sel = sorted(set(f for f, _ in P_['Call']))
    pickle.dump(dict(Call=P_['Call'], C=P_['C'], firms_sel=firms_sel, Fvis=sorted(P_['Fvis']), vis_any_firms=sorted(P_['vis_any_firms']), prior_vis=sorted(P_['prior_vis'])), open(BD/'pop.pkl', 'wb'))
    print('Call', len(P_['Call']), 'C', len(P_['C']), 'firms_sel', len(firms_sel), mem()); sys.exit(0)
import final_run_r3 as F, r3, r2, lib, r2ext, r3ext, r3ext2, r3ext3
from r2 import fr
PP = pickle.load(open(BD/'pop.pkl', 'rb')); firms_sel = PP['firms_sel']; OUT = BD/mode; OUT.mkdir(exist_ok=True)
EXTRA = set(map(tuple, PP['Call'])) if mode.startswith('s1') else set()
cf = dict(eval_cut=2019, train_cuts=(2012, 2013, 2014, 2015), cap='20191231')
ec = json.loads(json.dumps(F.EC)); cols = list(ec['cols'])
t0 = time.time()


def build_panel_c16(cut, h=5, cap=None, min_pat=3, fcap=None):
    """lib.build_panel 사본. 바뀐 곳 두 군데만 [C16] 표시."""
    if lib.STRICT and fcap is None: fcap = f'{cut}1231'
    assert cut + h <= lib.HOLDOUT_GUARD_MAX_ENTRY_YEAR, '🔒 홀드아웃 진입창 접근 금지'
    D = lib.load(); ins, oos, K, kidx = D['ins'], D['oos'], D['K'], D['kidx']
    if cap is not None:
        ins = ins[ins.rgst_date <= cap]; oos = oos[oos.rgst_date <= cap]
    ip = ins[ins.ay <= cut]; op = oos[oos.ay <= cut]
    cnt = pd.concat([ip.drop_duplicates(['applicant_cd', 'rgst_no']).applicant_cd, op.drop_duplicates(['applicant_cd', 'rgst_no']).applicant_cd]).value_counts()
    firms = np.array(sorted(cnt[cnt >= min_pat].index)) if FULL else np.array(sorted(firms_sel))   # [C16] 대상 법인 고정(FULL이면 공식 규칙 그대로)
    assert (cnt.reindex(firms).fillna(0).values >= min_pat).all()
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
    EX = np.zeros((nF, 50), bool)                                                          # [C16] 추가 위험쌍 마스크
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
    D = lib.load(); oos = D['oos']; sel = np.ones(len(oos), bool) if FULL else oos.applicant_cd.isin(set(firms_sel)).values
    rows_g = oos.row.values[sel]; oos2 = oos[sel].copy(); oos2['row'] = np.arange(len(oos2)); D['oos'] = oos2; del oos; gc.collect()
    P = np.load(FC/'P_base.npy', mmap_mode='r')[rows_g].astype(np.float32)
    P51 = np.load(FC/'P_os51.npy', mmap_mode='r')[rows_g].astype(np.float32); PO = np.load(FC/'PO_os51.npy', mmap_mode='r')[rows_g].astype(np.float32)
    Pins, ins_idx = fr.clf_ins(cf, FC); ep = fr.ep_match(cf, FC); cdf = fr.lag_cdf(cf)
    print('loaded', len(oos2), G.rss_now(), flush=True)
    fr.build_panel = lambda cut, h=5, cap=None, **kw: build_panel_c16(cut, 0, cap=cap, **kw)
    with F.guard(2019):
        if not (OUT/'feat_c2019.parquet').exists(): fr.cut_features(2019, None, cf, OUT, P, P51, PO, Pins, ins_idx, ep, cdf)
        if not (OUT/'base_app_c2019.parquet').exists(): fr.base_app_panel(2019, None, OUT)
    print('feat done', mode, round(time.time() - t0), G.rss_peak(), flush=True)

elif stage == 'ext':
    # 1단계: 외부 피처(EXT 11종) → te_ext.parquet 저장 후 종료(메모리)
    from s2lib import FIRM_LEVEL
    if not (OUT/'te_ext.parquet').exists():
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
        print('te_ext saved', len(te), round(time.time() - t0), G.rss_peak(), flush=True)
elif stage == 'pred':
    import xgboost as xgb
    from s2lib import FIRM_LEVEL
    if (OUT/'scores.parquet').exists(): print('exists'); sys.exit(0)
    te = pd.read_parquet(OUT/'te_ext.parquet'); te['y'] = 0
    ref = pd.read_parquet(P0D/'s1_te.parquet', columns=['k', 'k_prior', 'k_prior_new']).drop_duplicates('k').set_index('k')
    assert len(ref) == 50
    te['k_prior'] = te.k.map(ref.k_prior).values; te['k_prior_new'] = te.k.map(ref.k_prior_new).values
    base_cols = [x for x in cols if x != 'f_any']; fcols = [x for x in base_cols if x in FIRM_LEVEL]
    g_ = te.groupby(['cut', 'firm'], sort=False)
    Xf = g_[fcols].first(); Xf['n_risk'] = g_.size(); Xf['y'] = g_.y.max()
    for x in ('t_max', 'ot_max', 't_top3', 'pa_k', 'ep_k', 'ti_max'):
        if x in te: Xf[x + '_fmax'] = g_[x].max()
    fm = json.load(open(MD/'firm_seed0_meta.json', encoding='utf-8')); assert list(Xf.drop(columns='y').columns) == fm['xc'], (list(Xf.columns), fm['xc'])
    bf = xgb.Booster(); bf.load_model(str(MD/'firm_seed0.json'))
    ote = bf.predict(xgb.DMatrix(Xf[fm['xc']].values.astype(np.float32)))
    te['f_any'] = pd.Series(ote, index=Xf.index).reindex(pd.MultiIndex.from_arrays([te.cut.values, te.firm.values])).values
    out = te[['firm', 'k', 'newc', 'prior_any']].copy()
    pxs = []
    for sd in (0, 1):
        b = xgb.Booster(); b.load_model(str(MD/f'xgb_seed{sd}.json'))
        pxs.append(np.concatenate([b.predict(xgb.DMatrix(te[cols].iloc[i:i + 200000].values.astype(np.float32))) for i in range(0, len(te), 200000)]).astype(np.float32))
    L = pickle.load(open(MD/'logit.pkl', 'rb')); pl = np.concatenate([fr.logit_pred(L, te.iloc[i:i + 200000]) for i in range(0, len(te), 200000)]).astype(np.float32)
    out['px0'] = pxs[0]; out['px1'] = pxs[1]; out['px'] = np.mean(pxs, 0).astype(np.float32); out['pl'] = pl
    out['is_extra'] = pd.MultiIndex.from_arrays([out.firm.values, out.k.values]).isin(list(EXTRA)) if EXTRA else False
    SBBF = ['lsize', 'lin', 'lout', 'inshare', 'dens', 'kpop', 'k', 'newc', 'n_last1', 'gap_last', 'now1', 'now3', 'now_grow', 'k_grow', 'k_newent', 'vis_frac']
    b = xgb.Booster(); b.load_model(str(MD/'sbb.json')); out['sbb'] = b.predict(xgb.DMatrix(te[SBBF].values.astype(np.float32))).astype(np.float32)
    te[[x for x in dict.fromkeys(['firm', 'k'] + cols)]].to_parquet(OUT/'te_cols.parquet', index=False); del te; gc.collect()
    bte = pd.read_parquet(OUT/'base_app_c2019.parquet'); assert (bte.firm.values == out.firm.values).all() and (bte.k.values == out.k.values).all()
    BASE = list(r3.BASE)
    for lab in ('y', 'y_u'):
        for nm in ('stack', 'single'):
            Lb = pickle.load(open(MD/f'base_logit_{nm}_{lab}.pkl', 'rb')); out[f'logit_{nm}_{lab}'] = fr.logit_pred(Lb, bte).astype(np.float32)
        b = xgb.Booster(); b.load_model(str(MD/f'base_xgb_{lab}.json')); out[f'xgb_{lab}'] = b.predict(xgb.DMatrix(bte[BASE].values.astype(np.float32))).astype(np.float32)
    out.to_parquet(OUT/'scores.parquet', index=False)
    print('pred done', mode, len(out), int(np.sum(out.is_extra)), round(time.time() - t0), G.rss_peak(), flush=True)

elif stage == 'cmp':
    res = {}
    key = lambda d: pd.MultiIndex.from_arrays([d.firm.values, d.k.values])
    # (1) 패널 피처 vs 공식 feat_c2019
    import pyarrow.parquet as pq
    fl = None if FULL else [('firm', 'in', list(firms_sel))]
    myk = pd.read_parquet(OUT/'feat_c2019.parquet', columns=['firm', 'k']); offk = pd.read_parquet(FC/'feat_c2019.parquet', columns=['firm', 'k'], filters=fl)
    res['n_rows_my'] = len(myk); res['n_rows_off_same_firms'] = len(offk)
    pos = pd.Series(np.arange(len(offk)), index=key(offk)).reindex(key(myk))
    res['all_rows_found'] = bool(pos.notna().all()); pos = pos.values.astype(np.int64)
    dif = {}
    for c in pq.ParquetFile(OUT/'feat_c2019.parquet').schema.names:
        if c in ('fidx', 'firm', 'k'): continue
        a = pd.to_numeric(pd.read_parquet(OUT/'feat_c2019.parquet', columns=[c])[c], errors='coerce').values.astype(float)
        b = pd.to_numeric(pd.read_parquet(FC/'feat_c2019.parquet', columns=[c], filters=fl)[c], errors='coerce').values.astype(float)[pos]
        both = np.isnan(a) & np.isnan(b); dif[c] = float(np.nanmax(np.where(both, 0, np.abs(a - b)))) if len(a) else 0.0
    res['feat_maxdiff'] = max(dif.values()); res['feat_maxdiff_by_col_nonzero'] = {c: v for c, v in dif.items() if v > 0}
    # (2) 기준선 패널
    mb = pd.read_parquet(OUT/'base_app_c2019.parquet'); ob = pd.read_parquet(FC/'base_app_c2019.parquet', filters=None if FULL else [('firm', 'in', list(firms_sel))]).set_index(['firm', 'k']).reindex(key(mb))
    res['base_panel_maxdiff'] = float(max(np.nanmax(np.abs(mb[c].values.astype(float) - ob[c].values.astype(float))) for c in r3.BASE if c != 'ind2'))
    # (3) 점수 vs 공식
    sc = pd.read_parquet(OUT/'scores.parquet')
    te = pd.read_parquet(FC/'prep_te.parquet', columns=['firm', 'k', 'prior_any']); fz = pd.read_parquet(RD/'o2019.parquet')
    assert (fz.firm.values == te.firm.values).all()
    bo = pd.read_parquet(FC/'baseline_scores.parquet'); sb = np.load(P0D/'sbb_all.npy')
    T = pd.DataFrame(dict(firm=te.firm.values, k=te.k.values, prior_any=te.prior_any.values, px=fz.px.values, pl=fz.pl.values, sbb=sb))
    for c in bo.columns: T[c] = bo[c].values
    Ti = T.set_index(['firm', 'k']).reindex(key(sc))
    res['score_maxdiff'] = {c: float(np.abs(sc[c].values.astype(float) - Ti[c].values.astype(float)).max()) for c in ['px', 'pl', 'sbb'] + list(bo.columns)}
    res['prior_any_eq'] = bool((sc.prior_any.values == Ti.prior_any.values).all())
    # (4) 외부 피처·f_any vs P0 평가 파일(공식 경로)
    tck = pd.read_parquet(OUT/'te_cols.parquet', columns=['firm', 'k'])
    pos2 = pd.Series(np.arange(len(te)), index=key(te)).reindex(key(tck)).values.astype(np.int64)
    dd = {}
    for c in cols:
        if c == 'k': continue
        a = pd.read_parquet(OUT/'te_cols.parquet', columns=[c])[c].values.astype(float); b = pd.read_parquet(P0D/'s1_te.parquet', columns=[c])[c].values.astype(float)[pos2]
        dd[c] = float(np.nanmax(np.where(np.isnan(a) & np.isnan(b), 0, np.abs(a - b))))
    res['te_cols_maxdiff'] = max(dd.values()); res['te_cols_nonzero'] = {c: v for c, v in dd.items() if v > 0}
    res['pass'] = bool(res['all_rows_found'] and res['feat_maxdiff'] <= 1e-6 and res['base_panel_maxdiff'] <= 1e-6 and max(res['score_maxdiff'].values()) <= 1e-6 and res['te_cols_maxdiff'] <= 1e-6 and res['prior_any_eq'])
    jdump(res, BD/f'{mode}_compare.json'); print(json.dumps(res, ensure_ascii=False, indent=1))
