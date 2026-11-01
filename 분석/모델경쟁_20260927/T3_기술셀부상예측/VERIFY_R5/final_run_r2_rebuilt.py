# [VERIFY_R5 사본-새 패널판] R2/final_run_r2.py 복사: 패널 캐시를 VERIFY_R5/cache_2023_rebuilt에 새로 빌드, 성분 캐시 새로, 출력명만 변경
"""T3 2라운드 동결 모형 실행기 (final_spec_r2.json, 챔피언 E57).

사용 (R2 폴더 또는 어디서든):
  python final_run_r2.py --mode dev      # 개발 원점 2010..2016 재현 → dev_result.json (캐시 cache_dev, 출원연도 ≤2019)
                                         #   --fresh 를 붙이면 성분 예측 캐시를 쓰지 않고 처음부터 다시 학습
  python final_run_r2.py --mode final    # 🔒 평가자 전용 1회. cache_2023(출원연도 ≤2023) 빌드 → 학습 t0 2005..2016(결과창 ≤2019 < F=2020)
                                         #   → t0=2019(KR 2020–22), t0=2020(KR 2021–23) 채점 + 메인그룹 군집 부트스트랩 95% CI → final_result.json
  python final_run_r2.py --mode selftest # 코드 점검: 개발 캐시로 final·future 코드 경로를 가짜 원점(평가 2015·2016, 학습 ≤2012 / 예측 2016, 학습 ≤2013)으로 실행
  python final_run_r2.py --mode future   # final 뒤에만. 학습 t0 2005..2019 → t0=2022 예측(2023–2025) → future_forecast_r2.csv
빌드 단계는 산출물이 있으면 건너뛴다(9분 제한에 걸리면 같은 명령을 다시 실행).
"""
import os, sys, json, argparse, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__)); T3 = os.path.dirname(HERE)
ap = argparse.ArgumentParser()
ap.add_argument('--mode', required=True, choices=['dev', 'final', 'future', 'selftest'])
ap.add_argument('--n_boot', type=int, default=1000)
ap.add_argument('--fresh', action='store_true')
args = ap.parse_args()
SPEC = json.load(open(os.path.join(HERE, 'final_spec_r2.json'), encoding='utf-8'))

if args.mode in ('dev', 'selftest'):
    env = dict(T3R2_CACHE='cache_dev', T3R2_MAXYEAR='2019', T3R2_LABEL_MAX='2019', T3R2_EXT_SUFFIX='')
    build_env = dict(T3_CACHE='R2/cache_dev', T3_MAXYEAR='2019', T3_T0MAX='2016'); ext_args = ['--maxyear', '2019', '--io_year', '10']
    os.environ.pop('T3R2_FINAL_UNLOCK', None)
    if args.mode == 'selftest': env.update(T3R2_TRAIN_MAX_T0='2012', T3R2_NO_HIER='1')
else:
    # 🔒 final/future 에서만 2020년 이후 자료 해제
    env = dict(T3R2_CACHE='cache_2023_rebuilt', T3R2_COMP_DIR=os.path.join(HERE,'comp_r2_rebuilt'), T3R2_MAXYEAR='2023', T3R2_LABEL_MAX='2023', T3R2_FINAL_UNLOCK='1', T3R2_NO_HIER='1')
    build_env = dict(T3_CACHE='VERIFY_R5/cache_2023_rebuilt', T3_MAXYEAR='2023', T3_T0MAX='2022')
    if args.mode == 'final':
        env.update(T3R2_TRAIN_MAX_T0=str(SPEC['final']['train_max_t0']), T3R2_EXT_SUFFIX='_final')
        ext_args = ['--maxyear', '2020', '--io_year', '15', '--out_suffix', '_final']   # 피처는 t0 ≤2020 까지만
    else:
        env.update(T3R2_TRAIN_MAX_T0=str(SPEC['future']['last_train_t0']), T3R2_EXT_SUFFIX='_future')
        ext_args = ['--maxyear', str(SPEC['future']['t0']), '--io_year', '15', '--out_suffix', '_future']
env['PYTHONIOENCODING'] = 'utf-8'
os.environ.update(env)
CACHE = os.path.join(HERE, env['T3R2_CACHE']); os.makedirs(CACHE, exist_ok=True)

def run(cmd, cwd):
    t = time.time(); print('[build]', ' '.join(cmd), flush=True)
    r = subprocess.run([sys.executable] + cmd, cwd=cwd, env={**os.environ, **build_env})
    if r.returncode != 0: raise SystemExit(f'{cmd} 실패')
    print(f'[build] 완료 {time.time()-t:.0f}s', flush=True)
for step, out in [('build_base.py', 'base_rows.parquet'), ('build_panel.py', 'panel_ccy.parquet'), ('build_feat2.py', 'feat2.parquet')]:
    if not os.path.exists(os.path.join(CACHE, out)): run([step], T3)   # 1라운드 빌더(수정 없음)를 R2 캐시로 실행
ext_file = os.path.join(HERE, f"ext_ksic_t0{env['T3R2_EXT_SUFFIX']}.parquet")
if not os.path.exists(ext_file) or args.mode in ('final', 'future'): run(['build_ext.py'] + ext_args, HERE)

sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import r2core as C
from sklearn.metrics import roc_auc_score
CFG = SPEC['cfg']; TAG = {'dev': 'dev_fresh' if args.fresh else 'dev', 'final': 'final', 'future': 'future', 'selftest': 'selftest'}[args.mode]

def cluster_boot(blocks, n_boot, seed=0):
    """blocks: [(mg, yb, p_model, p_base)] 원점별. 군집 = IPC 메인그룹(같은 메인그룹의 다른 KSIC 셀·다른 원점 함께)."""
    rng = np.random.default_rng(seed); ums = np.unique(np.concatenate([b[0] for b in blocks]))
    gi = [pd.Series(np.arange(len(b[0]))).groupby(b[0]).apply(lambda s: s.values).to_dict() for b in blocks]
    out = []
    for _ in range(n_boot):
        s = rng.choice(ums, len(ums)); am, ab = [], []
        for b, g in zip(blocks, gi):
            ii = np.concatenate([g[u] for u in s if u in g]); y = b[1][ii]
            if y.min() == y.max(): continue
            am.append(roc_auc_score(y, b[2][ii])); ab.append(roc_auc_score(y, b[3][ii]))
        a, a0 = np.mean(am), np.mean(ab)
        out.append((a, a0, (a - a0) / (1 - a0), a - a0, np.mean([(x - y0) / (1 - y0) for x, y0 in zip(am, ab)])))
    o = np.array(out); q = lambda j: [float(v) for v in np.percentile(o[:, j], [2.5, 97.5])]
    return dict(auc=q(0), auc_baseline=q(1), skill=q(2), auc_diff=q(3), skill_mean_of_origins=q(4), n_boot=n_boot,
                cluster='IPC 메인그룹', n_clusters=int(len(ums)))

def evaluate(D, origins, out_json, pred_file, boot):
    res, blocks, preds = {}, [], []
    for v in origins:
        te = D[D.t0 == v]; tr = C.train_rows(D, v, CFG)
        assert tr.t0.max() + 3 <= v and tr.t0.max() + 3 < C.L.F_YEAR, '학습 레이블은 결과창이 F 이전에 끝난 원점만'
        p = C.score(CFG, D, v, TAG); p0 = C.baseline_pred(D, v)
        m = C.metrics(te.y.values, te.yb.values, p); b = C.metrics(te.y.values, te.yb.values, p0)
        res[v] = dict(model=m, baseline=b, skill=(m['auc'] - b['auc']) / (1 - b['auc']), train_t0=[int(tr.t0.min()), int(tr.t0.max())], n_train=int(len(tr)))
        print(f"t0={v}: 모형 AUC {m['auc']:.4f} PR-AUC {m['prauc']:.3f} P@10 {m['p_at10']:.3f} | 기준선 {b['auc']:.4f} | skill {res[v]['skill']:.4f}", flush=True)
        blocks.append((te.mg.values, te.yb.values, p, p0))
        preds.append(pd.DataFrame(dict(cell=te.cell.values, ksic=te.ksic.values, mg=te.mg.values, t0=v, p=p, p_base=p0, y=te.y.values, yb=te.yb.values)))
    au = np.mean([r['model']['auc'] for r in res.values()]); a0 = np.mean([r['baseline']['auc'] for r in res.values()])
    summ = dict(auc=float(au), auc_baseline=float(a0), skill_of_mean_auc=float((au - a0) / (1 - a0)),
                mean_skill=float(np.mean([r['skill'] for r in res.values()])),
                prauc=float(np.mean([r['model']['prauc'] for r in res.values()])), prauc_baseline=float(np.mean([r['baseline']['prauc'] for r in res.values()])),
                p_at10=float(np.mean([r['model']['p_at10'] for r in res.values()])), p_at10_baseline=float(np.mean([r['baseline']['p_at10'] for r in res.values()])),
                rho=float(np.mean([r['model']['rho'] for r in res.values()])))
    out = dict(mode=args.mode, model=SPEC['model_name'], origins=origins, summary=summ, by_origin={str(k): v for k, v in res.items()},
               created=time.strftime('%Y-%m-%d %H:%M:%S'))
    if boot: out['ci95'] = cluster_boot(blocks, args.n_boot)
    json.dump(out, open(os.path.join(HERE, out_json), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    os.makedirs(os.path.join(HERE, 'preds'), exist_ok=True)
    pd.concat(preds, ignore_index=True).to_parquet(os.path.join(HERE, 'preds', pred_file))
    print(json.dumps(dict(summary=summ, ci95=out.get('ci95')), ensure_ascii=False, indent=1))
    return out

def run_future(ft, lt, prefix):
    D = C.get_D(t0s=list(range(C.FIRST_T0, lt + 1)) + [ft])
    te = D[D.t0 == ft].copy(); te['score'] = C.score(CFG, D, ft, TAG)
    te['rank_pct'] = te.score.rank(pct=True, ascending=False); te['pred_breakout_top10'] = (te.rank_pct <= 0.10).astype(int)
    te['rank_in_ksic'] = te.groupby('ksic').score.rank(ascending=False, method='first').astype(int)
    cols = ['ksic', 'mg', 'kr_W', 'tot_W', 'kr_mom', 'US_mom', 'CN_mom', 'JP_mom', 'EP_mom', 'pi_prod_g2', 'pi_util', 'kr_share_rel_ksic', 'score',
            'rank_pct', 'pred_breakout_top10', 'rank_in_ksic']
    te = te.sort_values('score', ascending=False); te[cols].to_csv(os.path.join(HERE, prefix + '.csv'), index=False, encoding='utf-8-sig')
    te.groupby('ksic').agg(n_cells=('score', 'size'), n_pred_breakout=('pred_breakout_top10', 'sum'), pi_prod_g2=('pi_prod_g2', 'first'))       .assign(pred_rate=lambda d: d.n_pred_breakout / d.n_cells).sort_values('pred_rate', ascending=False)       .to_csv(os.path.join(HERE, prefix + '_ksic.csv'), encoding='utf-8-sig')
    json.dump(dict(mode=args.mode, t0=ft, window=[ft + 1, ft + 3], train_t0=[C.FIRST_T0, lt], n_cells=int(len(te)), created=time.strftime('%Y-%m-%d %H:%M:%S'),
                   note='KR 2023년 12월 약 10% 미완·타국 2023 절단 → 예측 원점은 전 국가 완결인 2022'),
              open(os.path.join(HERE, prefix + '_meta.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('저장:', prefix + '.csv')

t = time.time()
if args.mode == 'dev':
    D = C.get_D()
    out = evaluate(D, SPEC['dev']['origins'], 'dev_result_fresh.json' if args.fresh else 'dev_result.json', 'dev_preds.parquet', boot=True)
    ref = SPEC['dev']['mean_skill']
    print(f"재현 점검: 평균 skill {out['summary']['mean_skill']:.4f} vs 동결 기록 {ref:.4f} → {'일치' if abs(out['summary']['mean_skill'] - ref) < 5e-4 else '불일치!'}")
elif args.mode == 'final':
    D = C.get_D(t0s=list(range(C.FIRST_T0, SPEC['final']['train_max_t0'] + 1)) + SPEC['final']['test_t0'])
    assert D[D.t0 < 2019].t0.max() == 2016
    evaluate(D, SPEC['final']['test_t0'], 'r2_rerun_rebuilt_final_result.json', 'R2_RERUN_REBUILT_FINAL.parquet', boot=True)
elif args.mode == 'selftest':
    D = C.get_D(t0s=list(range(C.FIRST_T0, 2013)) + [2015, 2016])
    evaluate(D, [2015, 2016], 'selftest_final.json', 'selftest_final.parquet', boot=True)
    C._D.clear(); C._WIDE.clear(); os.environ['T3R2_TRAIN_MAX_T0'] = '2013'; C.TRAIN_MAX_T0 = 2013
    run_future(2016, 2013, 'selftest_future')
else:
    run_future(SPEC['future']['t0'], SPEC['future']['last_train_t0'], 'future_forecast_r2')
print(f'총 {time.time()-t:.0f}s')
