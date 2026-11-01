# T1 R3 실험 실행기 (R2 r2run.py와 같은 학습·채점 절차, 4년창·R3 원점·R3 경로)
#   python R3/r3run.py --exp R3E00 --origin 2010     한 원점 학습·예측(시드별 캐시, 종료코드 3 → 재실행)
#   python R3/r3run.py --exp R3E01 --summarize --vs R3E00
import sys, json, argparse, pathlib, os
R3 = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(R3))
from r3 import *
from r3exps import EXPS, resolve
import pyarrow.parquet as pq, pyarrow as pa


def _pm(tag):
    if os.environ.get('R3_MEMTRACE'): print(f'[memtrace] {tag} peak {peak_gb():.2f} GB', flush=True)


def run_origin(eid, c, ec=None):
    ec = ec or resolve(eid)
    RD = R3/'runs'/(os.environ.get('R3_RUNROOT') or eid); RD.mkdir(parents=True, exist_ok=True)
    out = RD/f'o{c}.parquet'
    if out.exists(): print('exists', out); return
    lib.HOLDOUT_GUARD_MAX_ENTRY_YEAR = GUARD_DEV
    prep(c)
    PTR = 'prep_tr.parquet'
    if ec.get('train_cuts_n', 4) == 5: prep5(c); PTR = 'prep_tr5.parquet'
    cols = list(ec['cols'])
    r2.light_load()
    from s2lib import FIRM_LEVEL
    FCp = fc_r3(c); allc = pq.ParquetFile(FCp/PTR).schema.names
    slimc = [x for x in dict.fromkeys(['firm', 'k', 'cut', 'newc', 'y_reg', 'y_pub', 'w_ipw', 'prior_pub', 't_lsum'] + [x for x in cols if x in FIRM_LEVEL]
                                      + ['t_max', 'ot_max', 't_top3', 'pa_k', 'ep_k', 'ti_max'] + ec.get('need_cols', [])) if x in allc]
    tr = pd.read_parquet(FCp/PTR, columns=slimc); tr['_row'] = np.arange(len(tr), dtype=np.int64)
    # 메모리(4GB) 보호: 평가표는 필요한 열만 읽는다(값 동일)
    tec = pq.ParquetFile(FCp/'prep_te.parquet').schema.names
    te = pd.read_parquet(FCp/'prep_te.parquet', columns=[x for x in dict.fromkeys(slimc + cols + ['y_u', 'prior_any', 'y_reg', 'y_pub', 'newc', 'firm', 'k', 'cut'] + [x for x in FIRM_LEVEL]) if x in tec])
    if ec.get('train_filter_prior'): tr = tr[~tr.prior_pub.values].reset_index(drop=True)
    n_cuts = ec.get('train_cuts_n', 4)
    if n_cuts < 4: tr = tr[tr.cut.values >= c - 7 + (4 - n_cuts)].reset_index(drop=True)
    _pm('load')
    import r2ext, r3ext, r3ext2, r3ext3
    EXT = dict(r2ext.EXT); EXT.update(r3ext.EXT); EXT.update(r3ext2.EXT); EXT.update(r3ext3.EXT)   # r3ext2·r3ext3 = B단계(3라운드 새 수집분)
    for g in ec.get('ext', []):
        tr, te = EXT[g](tr, te, c, False)
    _pm('ext')
    tr['y'] = tr.y_reg.values; te['y'] = 0
    if 'k_prior' in cols or 'k_prior_new' in cols: tr, te = fr.k_prior(tr, te)
    if 'f_any' in cols:
        from s2lib import firm_stage
        base_cols = [x for x in cols if x != 'f_any']
        # f_any 법인 단계는 필요한 열이 전부 있어야 한다 → 없는 열은 추출 전 전체 행으로 붙인다
        rest0 = [x for x in base_cols if x not in tr.columns and x in FIRM_LEVEL]
        if rest0:
            R = pq.read_table(FCp/PTR, columns=rest0).take(pa.array(tr._row.values)).to_pandas()
            for x in rest0: tr[x] = R[x].values
        trf = tr.assign(y=np.maximum(tr.y_reg.values, tr.y_pub.values)) if ec.get('f_any_label', 'reg') == 'union' else tr
        trf, te = firm_stage(trf, te, base_cols); tr['f_any'] = trf.f_any.values
    ipw_mode = ec.get('ipw', 'reg')
    w = tr.w_ipw.values.astype(np.float64) if ipw_mode == 'reg' else np.ones(len(tr))
    w = np.where(tr.y_reg.values == 1, w, 1.0)
    y = tr.y_reg.values.astype(np.int8)
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
    _pm('attach')
    print(f'[{eid} o{c}] train rows {len(tr)} pos {int(tr.y.sum())} | eval rows {len(te)}', flush=True)
    import xgboost as xgb
    import gc
    # 메모리 보호(값 동일): 출력용 식별열을 먼저 떼고, 학습·평가표는 모형 열만 남긴다
    o = te[['firm', 'k', 'cut', 'newc', 'y_reg', 'y_u', 'prior_any']].copy()
    # 로지스틱(혼합 10%)을 XGB보다 먼저: 순서만 바뀌고 값은 같다. 두 단계의 메모리 정점이 겹치지 않게 한다.
    fl = RD/f'o{c}_pl.npy'
    if not fl.exists():
        if ec.get('blend_w', 0.9) < 1.0:
            need(90, '로지스틱')
            tmp = RD/f'_te_tmp_o{c}.parquet'; te[cols].to_parquet(tmp, index=False); del te; gc.collect()   # 적합 중에는 평가표를 디스크로(값 동일)
            L = fr.logit_fit(tr[cols + ['y']], cols); te = pd.read_parquet(tmp); tmp.unlink()
            np.save(fl, np.concatenate([fr.logit_pred(L, te.iloc[i:i + 200000]) for i in range(0, len(te), 200000)]).astype(np.float32)); del L; gc.collect()
        else: np.save(fl, np.zeros(len(te), np.float32))
    _pm('logit')
    X = tr[cols].values.astype(np.float32)
    XteC = lambda i: te[cols].iloc[i:i + 200000].values.astype(np.float32)
    params = dict(SPEC1['params']); params.update(ec.get('params', {}))
    est = 125.0 * len(tr) / 877070 * params['n_estimators'] / 800 * (params.get('max_depth', 6) / 6)
    pxs = []
    for sd in ec['seeds']:
        fp = RD/f'o{c}_px{sd}.npy'
        if not fp.exists():
            need(est, f'XGB seed {sd} (예상 {est:.0f}s)')
            p = dict(tree_method='hist', nthread=2, random_state=sd); p.update(params)
            m = xgb.XGBClassifier(**p); m.fit(X, tr.y.values, sample_weight=w)
            np.save(fp, np.concatenate([m.predict_proba(XteC(i))[:, 1] for i in range(0, len(te), 200000)]).astype(np.float32))
            print('seed', sd, round(elapsed()), flush=True)
            if sd == ec['seeds'][0]:
                imp = pd.Series(m.get_booster().get_score(importance_type='gain')); imp.index = [cols[int(i[1:])] for i in imp.index]
                imp.sort_values(ascending=False).to_csv(RD/f'o{c}_gain.csv', encoding='utf-8-sig')
        pxs.append(np.load(fp)); _pm(f'xgb{sd}')
    o['px'] = np.mean(pxs, 0); o['pl'] = np.load(fl)
    o.to_parquet(out, index=False); print('saved', out, round(elapsed()), f'peak {peak_gb():.2f}GB', flush=True)


def score_origin(eid, c, ec=None):
    ec = ec or resolve(eid)
    o = pd.read_parquet(R3/'runs'/eid/f'o{c}.parquet'); keep = ~o.prior_any.values; f = o[keep]
    bw = ec.get('blend_w', 0.9)
    s = bw * fr.rank(f.px.values) + (1 - bw) * fr.rank(f.pl.values)
    m = mets(f.y_u.values, s, f.newc.values); m['auc_all_reglabel'] = float(roc_auc_score(f.y_reg.values, s))
    return m, s


def summarize(eid, vs=None, origins=None):
    ec = resolve(eid); origins = origins or ORIGINS; rows = {}
    for c in origins:
        m, _ = score_origin(eid, c, ec); b = baselines(c)['strongest']
        m['skill'] = skill(m['auc_all'], b['auc_all']); m['skill_new'] = skill(m['auc_new'], b['auc_new']); m['base_auc_all'] = b['auc_all']; rows[c] = m
    mean = float(np.mean([rows[c]['skill'] for c in origins]))
    res = dict(exp=eid, per_origin=rows, mean_skill=mean, mean_skill_new=float(np.mean([rows[c]['skill_new'] for c in origins])),
               mean_auc_all=float(np.mean([rows[c]['auc_all'] for c in origins])))
    aux = origins != ORIGINS
    if vs and not aux:
        rv = json.load(open(R3/'runs'/vs/'summary.json', encoding='utf-8'))
        wins = sum(rows[c]['skill'] > rv['per_origin'][str(c)]['skill'] for c in origins)
        d = mean - rv['mean_skill']; res.update(vs=vs, delta=d, wins=int(wins), win_rate=wins / len(origins),
                                                   decision='ACCEPT' if (d >= 0.003 and wins / len(origins) >= 0.6) else 'ROLLBACK')
    json.dump(res, open(R3/'runs'/eid/('summary_aux.json' if aux else 'summary.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    if not aux:
        E = EXPS[eid]; p = R3/'experiments.csv'
        row = dict(exp_id=eid, parent=E.get('base'), change=E.get('change', ''))
        for c in origins: row[f'skill_o{c}'] = round(rows[c]['skill'], 4)
        for c in origins: row[f'auc_all_o{c}'] = round(rows[c]['auc_all'], 4)
        row.update(mean_skill=round(mean, 4), mean_skill_new=round(res['mean_skill_new'], 4), mean_auc_all=round(res['mean_auc_all'], 4),
                   vs_champion=vs or '', delta_vs_champion=round(res.get('delta', 0.0), 4) if vs else '', win_rate=res.get('win_rate', ''),
                   decision=res.get('decision', E.get('decision', 'START')), note=E.get('note', ''))
        df = pd.DataFrame([row])
        if p.exists():
            old = pd.read_csv(p, encoding='utf-8-sig'); old = old[old.exp_id != eid]; df = pd.concat([old, df], ignore_index=True)
        df.to_csv(p, index=False, encoding='utf-8-sig')
    print(json.dumps({k: v for k, v in res.items() if k != 'per_origin'}, ensure_ascii=False))
    for c in origins: print(c, {k: round(v, 4) for k, v in rows[c].items()})
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--exp'); ap.add_argument('--origin', type=int); ap.add_argument('--summarize', action='store_true')
    ap.add_argument('--vs'); ap.add_argument('--baselines', action='store_true'); ap.add_argument('--prep', type=int); ap.add_argument('--aux', action='store_true')
    a = ap.parse_args()
    if a.prep: prep(a.prep); print(json.dumps(baselines(a.prep)['strongest']))
    elif a.baselines:
        for c in ORIGINS: print(c, json.dumps(baselines(c)['strongest']))
    elif a.summarize: summarize(a.exp, a.vs, AUX if a.aux else None)
    else: run_origin(a.exp, a.origin)
