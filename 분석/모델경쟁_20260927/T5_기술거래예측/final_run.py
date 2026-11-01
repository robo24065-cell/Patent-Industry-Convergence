# T5 권리이전(기술거래) 예측 — 동결 모형 실행기 (2단계 동결: v29)
#   모형: 2단 계층모형  p = owner_hat(L, 권리자) × (p_pat / 권리자 내 평균 p_pat)^0.5
#     p_pat    : XGB 특허 단위(감사 수정 피처 PF_FIX, 400트리·깊이6)
#     owner_hat: 권리자×L 매각률 XGBRegressor(reg:logistic) 6-배깅, 가중 sqrt(보유건수)×0.85^(학습 마지막 L − L)
#   피처·하이퍼파라미터: final_spec.json / v2lib.py (BASE + recency=0.85)
#
# 사용 (작업 폴더에서):
#   python final_run.py --mode dryrun              # 검증: 학습 L2006–2011 → 적용 L2014·2015 (기존 cache/ 사용, CUT=2019-01-01)
#   python final_run.py --mode dryrun --rebuild    # 위와 같되 _work_dryrun/ 에 캐시를 처음부터 재생성(파이프라인 점검용)
#   python final_run.py --mode holdout [--stage build|fit|all]   # 🔒 3단계 평가자 전용
#       _work_holdout/ 에 CUT=2025-08-23(데이터 최종일 다음날), LMAX=2021 로 캐시 재생성 →
#       학습 L2006–2016 (레이블창 [L, L+3y) 이 2019-01-01 이전에 끝남 = SPLITS 규칙 4) → 적용 L2019·2020·2021
#       → holdout_result.json (모형·기준선 AUC/PR-AUC/p@1%, 권리자(양도인) 군집 부트스트랩 95% CI, 쌍대 ΔAUC)
#   python final_run.py --mode future [--stage build|fit|all]    # 홀드아웃 평가 뒤에만
#       _work_future/ 에 CUT=2025-08-23, LMAX=2025 로 캐시 재생성(L+3y > CUT 인 기준일은 레이블 -1) →
#       학습 L2006–2022(레이블 완결분 전체) → 적용 L2023·2024·2025 → future_forecast.csv
# 각 build 단계는 스크립트별 하위 프로세스로 돈다(한 단계당 수 분). 9분 제한 환경에서는 --stage build 와 --stage fit 을 나눠 실행.
import os, sys, json, time, shutil, argparse, subprocess, datetime
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = ['s10_extract.py', 's11_inv.py', 's20_panel.py', 's21_extra.py', 's23_unit.py', 's24_techexit.py', 's50_actfix.py', 's51_ownerattr2.py']
LIBS = ['common.py', 'evalkit.py', 'run.py', 'v2lib.py']
MODES = {
    'dryrun': dict(cut='2019-01-01', lmax=2015, train=(2006, 2011), apply=[2014, 2015]),
    'holdout': dict(cut='2025-08-23', lmax=2021, train=(2006, 2016), apply=[2019, 2020, 2021]),
    'future': dict(cut='2025-08-23', lmax=2025, train=(2006, 2022), apply=[2023, 2024, 2025]),
}
FROZEN_CFG = dict(recency=0.85)   # v29 = v2lib.BASE + 최근성 가중


def P(*a):
    print(*a, flush=True)


def prepare_workdir(mode):
    wd = os.path.join(HERE, f'_work_{mode}')
    os.makedirs(os.path.join(wd, 'cache'), exist_ok=True)
    for f in PIPE + LIBS:
        src = open(os.path.join(HERE, f), encoding='utf-8').read()
        if mode == 'future' and f == 's20_panel.py':
            # 미래 적용 기준일(L+3y > CUT)은 레이블이 미완결 → 레이블을 -1 로 두고 패널(피처)만 생성
            src = src.replace("    assert E <= HOLD_CUT\n", "    LABEL_OK = E <= HOLD_CUT\n")
            src = src.replace("    s = s.drop(columns=['last_pay', 'applDate', 'cndrtExptnDate', 'opnDate', 'orig_pref'])",
                              "    if not LABEL_OK:\n        for c in ['y_MAIN', 'y_CORP', 'y_INH', 'y_OTHER', 'y_ANY']: s[c] = np.int8(-1)\n"
                              "        s['y_assignee_type'] = np.nan; s['y_date'] = pd.NaT; s['y_assignee'] = None\n"
                              "    s = s.drop(columns=['last_pay', 'applDate', 'cndrtExptnDate', 'opnDate', 'orig_pref'])")
            assert 'LABEL_OK' in src
        open(os.path.join(wd, f), 'w', encoding='utf-8').write(src)
    return wd


def build(mode, wd):
    env = dict(os.environ, T5_CUT=MODES[mode]['cut'], T5_LMAX=str(MODES[mode]['lmax']), PYTHONIOENCODING='utf-8')
    for f in PIPE:
        t = time.time(); P(f'[build:{mode}] {f} ...')
        r = subprocess.run([sys.executable, f], cwd=wd, env=env)
        if r.returncode != 0:
            raise SystemExit(f'build step failed: {f}')
        P(f'[build:{mode}] {f} done {time.time() - t:.0f}s')
    json.dump(dict(mode=mode, cut=MODES[mode]['cut'], lmax=MODES[mode]['lmax'], built=datetime.datetime.now().isoformat()),
              open(os.path.join(wd, 'cache', 'BUILD_OK.json'), 'w'))


def use_dir(wd, mode):
    os.environ['T5_CUT'] = MODES[mode]['cut']; os.environ['T5_LMAX'] = str(MODES[mode]['lmax'])
    os.chdir(wd); sys.path.insert(0, wd)


def baseline(tr, ap):
    """고정 기준선(v0): 로지스틱(경과연수 + 출원인 유형 + KSIC + 청구항 수 + 등록연도)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import make_pipeline
    num = ['age', 'lclaims', 'rgst_year']; cat = ['orig_type', 'ksic']
    X = lambda d: d[['age', 'rgst_year', 'orig_type', 'ksic']].assign(lclaims=np.log1p(d.claims.fillna(d.claims.median())))
    m = make_pipeline(ColumnTransformer([('n', StandardScaler(), num), ('c', OneHotEncoder(handle_unknown='ignore'), cat)]),
                      LogisticRegression(max_iter=2000, C=1.0))
    m.fit(X(tr), tr.y_MAIN)
    return m.predict_proba(X(ap))[:, 1]


def boot(y, p, pb, cl, B=1000, seed=0):
    from sklearn.metrics import roc_auc_score, average_precision_score
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(pd.Series(cl).fillna('NA')); order = np.argsort(codes, kind='stable'); cs = codes[order]
    st = np.searchsorted(cs, np.arange(len(uniq))); en = np.searchsorted(cs, np.arange(len(uniq)), side='right')
    R = []
    for _ in range(B):
        idx = np.concatenate([order[st[c]:en[c]] for c in rng.integers(0, len(uniq), len(uniq))])
        yy = y[idx]
        if yy.min() == yy.max(): continue
        am, ab = roc_auc_score(yy, p[idx]), roc_auc_score(yy, pb[idx])
        R.append((am, ab, am - ab, (am - ab) / (1 - ab), average_precision_score(yy, p[idx]), average_precision_score(yy, pb[idx])))
    R = np.array(R); q = lambda j: [float(np.percentile(R[:, j], 2.5)), float(np.percentile(R[:, j], 97.5))]
    return dict(B=len(R), auc_model=q(0), auc_baseline=q(1), dauc=q(2), skill=q(3), prauc_model=q(4), prauc_baseline=q(5),
                p_dauc_gt0=float(np.mean(R[:, 2] > 0)))


def fit_eval(mode, out_json=None, B=1000):
    import v2lib
    from evalkit import metrics, skill
    a, b = MODES[mode]['train']; apply = MODES[mode]['apply']
    tr = v2lib.load(range(a, b + 1)); ap = v2lib.load(apply)
    # 🔒 규칙 4 점검: 학습 레이블 창 [L, L+3y) 이 적용 첫 기준일 이전에 끝나는지
    assert max(tr.L) + 3 <= min(apply), '학습 레이블 창이 적용 기준일을 넘는다'
    assert set(tr.y_MAIN.unique()) <= {0, 1}
    t = time.time()
    res = v2lib.fit_apply(FROZEN_CFG, range(a, b + 1), apply, tr, ap)
    res['p_base'] = baseline(tr, ap)
    P(f'fit {time.time() - t:.0f}s, train n={len(tr)} pos={int(tr.y_MAIN.sum())}, apply n={len(ap)}')
    if mode == 'holdout':   # [3단계 추가] 독립 재계산용 행 단위 예측 저장(모형 무관)
        keep = ['rgst_no', 'L', 'cur', 'ksic', 'cur_type', 'orig_type', 'y_MAIN', 'y_CORP', 'y_INH', 'y_ANY', 'y_assignee_type', 'y_date',
                'p_pat', 'owner_hat', 'p', 'p_base']
        res[[c for c in keep if c in res.columns]].to_parquet(os.path.join(HERE, 'holdout_pred.parquet'), index=False)
    y = res.y_MAIN.values
    mt, mb = metrics(y, res.p.values), metrics(y, res.p_base.values)
    per = {}
    for L in apply:
        k = (res.L == L).values
        per[str(L)] = dict(n=int(k.sum()), pos=int(y[k].sum()), auc_model=metrics(y[k], res.p.values[k])['auc'],
                           auc_baseline=metrics(y[k], res.p_base.values[k])['auc'])
    ci = boot(y, res.p.values, res.p_base.values, res.cur.values, B=B)
    out = dict(track='T5 권리이전(기술거래) 예측', frozen_version='v29', mode=mode, train_L=[a, b], apply_L=apply, cut=MODES[mode]['cut'],
               unit='(특허, 기준일 L=1월1일), L 시점 생존·권리자 이력 보유 KR 등록특허', target="y_MAIN: [L, L+3y) 안 '양도'류 권리이전 등록",
               cluster='L 시점 현재 권리자(=양도인) 군집 부트스트랩', n=int(len(res)), pos=int(y.sum()), base_rate=float(y.mean()),
               model=dict(auc=mt['auc'], prauc=mt['prauc'], p_at_1pct=mt['p1'], mean_pred=float(res.p.mean())),
               baseline=dict(auc=mb['auc'], prauc=mb['prauc'], p_at_1pct=mb['p1']),
               auc_skill=skill(mt['auc'], mb['auc']), prauc_ratio_vs_baseline=mt['prauc'] / mb['prauc'], prauc_ratio_vs_base_rate=mt['prauc'] / float(y.mean()),
               per_L=per, ci95=ci, y_ANY_auc=metrics(res.y_ANY.values, res.p.values)['auc'], y_CORP_auc=metrics(res.y_CORP.values, res.p.values)['auc'],
               run_at=datetime.datetime.now().isoformat())
    P(json.dumps({k: out[k] for k in ['n', 'pos', 'model', 'baseline', 'auc_skill', 'per_L']}, ensure_ascii=False, indent=1))
    P('CI', json.dumps(ci, ensure_ascii=False))
    if out_json:
        json.dump(out, open(out_json, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        P('saved', out_json)
    return out, res


def future(out_csv, calib):
    import v2lib
    a, b = MODES['future']['train']; apply = MODES['future']['apply']
    tr = v2lib.load(range(a, b + 1)); ap = v2lib.load(apply)
    assert set(tr.y_MAIN.unique()) <= {0, 1}, '학습 기준일에 미완결 레이블이 섞였다'
    res = v2lib.fit_apply(FROZEN_CFG, range(a, b + 1), apply, tr, ap)
    res['p_cal'] = (res.p * calib).clip(upper=1)
    TN = {1: '국내법인', 2: '대학', 3: '연구기관', 4: '국내개인', 5: '외국', 7: '국가·공공', 0: '미상'}
    res['seller_type'] = res.cur_type.map(TN).fillna('미상')
    rows = []
    for (L, k), d in res.groupby(['L', 'ksic']):
        rows.append(dict(kind='forecast', L=L, window=f'{L}-{L + 2}', ksic=k, group='전체', n_alive=len(d), pred_rate_raw=d.p.mean(),
                         pred_rate_cal=d.p_cal.mean(), expected_transfers=d.p_cal.sum()))
    for (L, t), d in res.groupby(['L', 'seller_type']):
        rows.append(dict(kind='forecast', L=L, window=f'{L}-{L + 2}', ksic='ALL', group=f'양도인유형:{t}', n_alive=len(d), pred_rate_raw=d.p.mean(),
                         pred_rate_cal=d.p_cal.mean(), expected_transfers=d.p_cal.sum()))
    for L, d in res.groupby('L'):
        rows.append(dict(kind='forecast', L=L, window=f'{L}-{L + 2}', ksic='ALL', group='전체', n_alive=len(d), pred_rate_raw=d.p.mean(),
                         pred_rate_cal=d.p_cal.mean(), expected_transfers=d.p_cal.sum()))
    for (L, k), d in tr.groupby(['L', 'ksic']):   # 관측(학습 기준일) 추이: 비교용
        rows.append(dict(kind='observed', L=L, window=f'{L}-{L + 2}', ksic=k, group='전체', n_alive=len(d), obs_rate=d.y_MAIN.mean(), obs_transfers=int(d.y_MAIN.sum())))
    f = pd.DataFrame(rows)
    fc = f[(f.kind == 'forecast') & (f.ksic != 'ALL')]
    f.loc[fc.index, 'rank_in_L'] = fc.groupby('L').pred_rate_raw.rank(ascending=False)
    f.to_csv(out_csv, index=False, encoding='utf-8-sig')
    P('saved', out_csv, len(f))


if __name__ == '__main__':
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--mode', required=True, choices=list(MODES))
    ap_.add_argument('--stage', default='all', choices=['build', 'fit', 'all'])
    ap_.add_argument('--rebuild', action='store_true', help='dryrun 에서도 _work_dryrun 에 캐시 재생성')
    ap_.add_argument('--B', type=int, default=1000)
    A = ap_.parse_args()
    spec = json.load(open(os.path.join(HERE, 'final_spec.json'), encoding='utf-8'))
    if A.mode == 'dryrun' and not A.rebuild:
        use_dir(HERE, 'dryrun')
        out, _ = fit_eval('dryrun', os.path.join(HERE, 'dryrun_result.json'), B=A.B)
        exp = spec['validation']['auc']
        P(f"dryrun AUC {out['model']['auc']:.4f} vs spec {exp:.4f} ->", 'OK' if abs(out['model']['auc'] - exp) < 1e-3 else 'MISMATCH')
        sys.exit(0)
    wd = prepare_workdir(A.mode)
    if A.stage in ('build', 'all'):
        build(A.mode, wd)
    if A.stage in ('fit', 'all'):
        assert os.path.exists(os.path.join(wd, 'cache', 'BUILD_OK.json')), '먼저 --stage build'
        use_dir(wd, A.mode)
        if A.mode == 'dryrun':
            out, _ = fit_eval('dryrun', os.path.join(HERE, 'dryrun_rebuild_result.json'), B=A.B)
            P(f"dryrun(rebuild) AUC {out['model']['auc']:.4f} vs spec {spec['validation']['auc']:.4f}")
        elif A.mode == 'holdout':
            fit_eval('holdout', os.path.join(HERE, 'holdout_result.json'), B=A.B)
        else:
            future(os.path.join(HERE, 'future_forecast.csv'), spec['calibration']['ratio_obs_over_pred_val'])
