# T1 R2 실험 실행기
#   python R2/r2run.py --exp E01 --origin 2010    한 원점 학습·채점(시드별 캐시, 종료코드 3 → 재실행)
#   python R2/r2run.py --exp E01 --summarize      원점별 지표·챔피언 대비 판정 → experiments.csv
import sys, json, argparse, pathlib, copy
R2 = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(R2))
from r2 import *
from r2exps import EXPS, resolve


def _pm(tag):
    if os.environ.get('R2_MEMTRACE'): print(f'[memtrace] {tag} peak {peak_gb():.2f} GB', flush=True)


def run_origin(eid, c, final=False, run_root=None, ec=None, mode=None, lowmem=False):
    ec = ec or resolve(eid)
    mode = mode or ('final' if final else 'dev'); final = mode != 'dev'
    RD = (run_root or (R2/'runs'/eid)); RD.mkdir(parents=True, exist_ok=True)
    out = RD/f'o{c}.parquet'
    if out.exists(): print('exists', out); return
    lib.HOLDOUT_GUARD_MAX_ENTRY_YEAR = GUARD[mode]
    prep(c, mode=mode)
    cols = list(ec['cols'])
    if lowmem:
        light_load()
        # 메모리 절약 경로(결과 동일): 파생·외부 피처 계산에 필요한 열만 전체 행으로 읽고, 음성 추출 후 나머지 피처 열을 추출된 행만 읽어 붙인다.
        import pyarrow.parquet as pq, pyarrow as pa
        from s2lib import FIRM_LEVEL
        FCp = fc_of(c, mode=mode); allc = pq.ParquetFile(FCp/'prep_tr.parquet').schema.names
        slimc = [x for x in dict.fromkeys(['firm', 'k', 'cut', 'newc', 'y_reg', 'y_pub', 'w_ipw', 'prior_pub', 't_lsum'] + [x for x in cols if x in FIRM_LEVEL]
                                          + ['t_max', 'ot_max', 't_top3', 'pa_k', 'ep_k', 'ti_max']) if x in allc]
        tr = pd.read_parquet(FCp/'prep_tr.parquet', columns=slimc); tr['_row'] = np.arange(len(tr), dtype=np.int64)
        te = pd.read_parquet(FCp/'prep_te.parquet').drop(columns=['y'])
    else:
        tr, te = load_prep(c, mode=mode)
        tr = tr.drop(columns=['y']); te = te.drop(columns=['y'])
    if ec.get('train_filter_prior'): tr = tr[~tr.prior_pub.values].reset_index(drop=True)
    n_cuts = ec.get('train_cuts_n', 4)
    if n_cuts < 4: tr = tr[tr.cut.values >= c - 8 + (4 - n_cuts)].reset_index(drop=True)
    import r2ext
    _pm('load')
    for g in ec.get('ext', []):
        tr, te = r2ext.EXT[g](tr, te, c, final)
    _pm('ext')
    # 파생(1라운드 순서 유지: k_prior·f_any는 등록 레이블)
    tr['y'] = tr.y_reg.values
    te['y'] = 0
    if 'k_prior' in cols or 'k_prior_new' in cols: tr, te = fr.k_prior(tr, te)
    if 'f_any' in cols:
        from s2lib import firm_stage
        base_cols = [x for x in cols if x != 'f_any']
        if ec.get('f_any_label', 'reg') == 'union': trf = tr.assign(y=np.maximum(tr.y_reg.values, tr.y_pub.values))
        else: trf = tr
        trf, te = firm_stage(trf, te, base_cols); tr['f_any'] = trf.f_any.values
    _pm('derived')
    # 학습 레이블·가중
    ipw_mode = ec.get('ipw', 'reg')
    w = tr.w_ipw.values.astype(np.float64) if ipw_mode == 'reg' else np.ones(len(tr))
    if ipw_mode == 'reg' and ec.get('ipw_wmax', 5.0) != 5.0: w = np.minimum(w, ec['ipw_wmax'])
    w = np.where(tr.y_reg.values == 1, w, 1.0)
    y = tr.y_reg.values.astype(np.int8)
    pw = ec.get('publab_w', 1.0)
    if pw > 0:
        add = (tr.y_pub.values == 1) & (y == 0); y = np.maximum(y, tr.y_pub.values).astype(np.int8); w = np.where(add, pw, w)
    tr['y'] = y
    pm = ec.get('pos_mult', 1.0)
    if pm != 1.0: w = np.where(y == 1, w * pm, w)
    cw = ec.get('cut_weight_step', 0.0)
    if cw:   # 최근 학습 원점일수록 가중(원점 순번 0,1,2,3 → 1, 1+cw, 1+2cw, 1+3cw)
        w = w * (1.0 + cw * (tr.cut.values - tr.cut.values.min()))
    q = ec.get('neg_q')
    if q:
        rs = np.random.RandomState(1234 + c); kp = (y == 1) | (rs.rand(len(tr)) < q)
        w = np.where(y == 1, w, w / q)[kp]; tr = tr.loc[kp].reset_index(drop=True)
    if lowmem:
        rest = [x for x in cols if x not in tr.columns]
        R = pq.read_table(FCp/'prep_tr.parquet', columns=rest).take(pa.array(tr._row.values)).to_pandas()
        for x in rest: tr[x] = R[x].values
        del R
    print(f'[{eid} o{c}] train rows {len(tr)} pos {int(tr.y.sum())} | eval rows {len(te)}', flush=True)
    _pm('subsample+attach')
    import xgboost as xgb
    X = tr[cols].values.astype(np.float32)
    XteC = lambda i: te[cols].iloc[i:i + 200000].values.astype(np.float32)   # 평가 행렬은 청크로만 만든다(메모리)
    params = dict(SPEC1['params']); params.update(ec.get('params', {}))
    est = 125.0 * len(tr) / 877070 * params['n_estimators'] / 800 * (params.get('max_depth', 6) / 6)
    pxs = []
    for sd in ec['seeds']:
        fp = RD/f'o{c}_px{sd}.npy'
        if not fp.exists():
            need(est, f'XGB seed {sd} (예상 {est:.0f}s)')
            p = dict(tree_method='hist', nthread=2, random_state=sd); p.update(params)
            if str(p.get('objective', '')).startswith('rank:'):
                # 순위 목적함수: 전체를 한 질의(qid=0)로 두고 쌍 비교 — 점수는 margin(단조 변환이라 AUC·순위 혼합에 영향 없음)
                m = xgb.XGBRanker(**p); m.fit(X, tr.y.values, qid=np.zeros(len(X), np.int32), sample_weight=None)
                np.save(fp, np.concatenate([m.predict(XteC(i)) for i in range(0, len(te), 200000)]).astype(np.float32))
            else:
                m = xgb.XGBClassifier(**p); m.fit(X, tr.y.values, sample_weight=w)
                np.save(fp, np.concatenate([m.predict_proba(XteC(i))[:, 1] for i in range(0, len(te), 200000)]).astype(np.float32))   # 청크 예측(값 동일)
            print('seed', sd, round(elapsed()), flush=True)
            if sd == ec['seeds'][0]:
                imp = pd.Series(m.get_booster().get_score(importance_type='gain')); imp.index = [cols[int(i[1:])] for i in imp.index]
                imp.sort_values(ascending=False).to_csv(RD/f'o{c}_gain.csv', encoding='utf-8-sig')
        pxs.append(np.load(fp))
    _pm('xgb')
    fl = RD/f'o{c}_pl.npy'
    if not fl.exists():
        if ec.get('blend_w', 0.9) < 1.0:
            need(90, '로지스틱'); L = fr.logit_fit(tr, cols); del X
            np.save(fl, np.concatenate([fr.logit_pred(L, te.iloc[i:i + 200000]) for i in range(0, len(te), 200000)]).astype(np.float32))   # 청크 예측(값 동일)
        else: np.save(fl, np.zeros(len(te), np.float32))
    o = te[['firm', 'k', 'cut', 'newc', 'y_reg', 'y_u', 'prior_any']].copy()
    o['px'] = np.mean(pxs, 0); o['pl'] = np.load(fl)
    o.to_parquet(out, index=False); print('saved', out, round(elapsed()), flush=True)


def score_origin(eid, c, ec=None, run_root=None):
    ec = ec or resolve(eid)
    o = pd.read_parquet((run_root or (R2/'runs'/eid))/f'o{c}.parquet'); keep = ~o.prior_any.values; f = o[keep]
    bw = ec.get('blend_w', 0.9)
    s = bw * fr.rank(f.px.values) + (1 - bw) * fr.rank(f.pl.values)
    m = mets(f.y_u.values, s, f.newc.values); m['auc_all_reglabel'] = float(roc_auc_score(f.y_reg.values, s))
    sfull = bw * fr.rank(o.px.values) + (1 - bw) * fr.rank(o.pl.values)
    m['auc_all_R1def'] = float(roc_auc_score(o.y_reg.values, sfull))   # 1라운드 정의(필터 없음·등록 레이블) 재현 확인용
    return m, s


def summarize(eid, vs=None):
    ec = resolve(eid); rows = {}
    for c in ORIGINS:
        m, _ = score_origin(eid, c, ec); b = baselines(c)['strongest']
        m['skill'] = skill(m['auc_all'], b['auc_all']); m['skill_new'] = skill(m['auc_new'], b['auc_new']); rows[c] = m
    mean = float(np.mean([rows[c]['skill'] for c in ORIGINS]))
    res = dict(exp=eid, per_origin=rows, mean_skill=mean, mean_skill_new=float(np.mean([rows[c]['skill_new'] for c in ORIGINS])),
               mean_auc_all=float(np.mean([rows[c]['auc_all'] for c in ORIGINS])))
    if vs:
        rv = json.load(open(R2/'runs'/vs/'summary.json', encoding='utf-8'))
        wins = sum(rows[c]['skill'] > rv['per_origin'][str(c)]['skill'] for c in ORIGINS)
        d = mean - rv['mean_skill']; res.update(vs=vs, delta=d, wins=int(wins), win_rate=wins / len(ORIGINS),
                                                   decision='ACCEPT' if (d >= 0.003 and wins / len(ORIGINS) >= 0.6) else 'ROLLBACK')
    json.dump(res, open(R2/'runs'/eid/'summary.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    # experiments.csv 기록
    E = EXPS[eid]; p = R2/'experiments.csv'
    row = dict(exp_id=eid, parent=E.get('base'), change=E.get('change', ''))
    for c in ORIGINS: row[f'skill_o{c}'] = round(rows[c]['skill'], 4)
    for c in ORIGINS: row[f'auc_all_o{c}'] = round(rows[c]['auc_all'], 4)
    row.update(mean_skill=round(mean, 4), mean_skill_new=round(res['mean_skill_new'], 4), mean_auc_all=round(res['mean_auc_all'], 4),
               vs_champion=vs or '', delta_vs_champion=round(res.get('delta', 0.0), 4) if vs else '', win_rate=res.get('win_rate', ''),
               decision=res.get('decision', E.get('decision', 'START')), note=E.get('note', ''))
    df = pd.DataFrame([row])
    if p.exists():
        old = pd.read_csv(p, encoding='utf-8-sig'); old = old[old.exp_id != eid]; df = pd.concat([old, df], ignore_index=True)
    df.to_csv(p, index=False, encoding='utf-8-sig')
    print(json.dumps({k: v for k, v in res.items() if k != 'per_origin'}, ensure_ascii=False))
    for c in ORIGINS: print(c, {k: round(v, 4) for k, v in rows[c].items()})
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--exp', required=True); ap.add_argument('--origin', type=int); ap.add_argument('--summarize', action='store_true')
    ap.add_argument('--vs'); ap.add_argument('--baselines', action='store_true')
    a = ap.parse_args()
    if a.baselines:
        for c in ORIGINS: print(c, json.dumps(baselines(c)['strongest']))
    elif a.summarize: summarize(a.exp, a.vs)
    else: run_origin(a.exp, a.origin)
