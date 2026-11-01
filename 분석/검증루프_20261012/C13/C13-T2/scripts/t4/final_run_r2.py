# -*- coding: utf-8 -*-
"""T4 특허 포기(등록료불납) 예측 — 2라운드 동결 파이프라인. 모형·피처·하이퍼파라미터는 final_spec_r2.json 에서만 읽는다.

  python final_run_r2.py --mode dev      # 개발 원점 4개(O12·O14·O16·O18) 재현 → R2/dev_result.json (experiments.csv 챔피언 행과 대조)
  python final_run_r2.py --mode final    # 🔒 최종시험 1회: C=2022-01-01 이전에 관측이 끝난 레이블만으로 학습(결과 창 끝 <= 2020-11),
                                          #    L=2022-01-01 위험집합 예측 → 3년 결과(2022-01~2025-01)를 납부기록 레이블(label_np)로 채점
                                          #    → R2/final_result.json (모형·기준선 AUC·skill·PR-AUC·Brier, 출원인 군집 부트스트랩 95% CI)
  python final_run_r2.py --mode future   # (최종시험 뒤) 전체 가용자료로 재학습 → KSIC × 반기 기준일 포기위험 추이 R2/future_forecast_r2.csv
옵션: --par N 동시 학습 프로세스(기본 2, 각 4GB 이하·nthread 2) / --boot B (기본 1000)

최종시험 기준일을 2022-01-01 하나로 둔 이유: 등록원부 최종일이 2025-08-22 라서, 3년 창 끝 + 추가납부 유예 6개월(+30일)이
그 안에 들어오는 2022년 기준일은 2022-01-01 부근뿐이다(2022-07-01 이면 2025-07 창 끝의 불납 여부가 아직 확정되지 않음).
소멸등록 기록은 실효 뒤 약 321일에 찍히므로 최종 레이블은 소멸등록이 아니라 납부기록(label_np)으로 판정한다.
개발 원점에서 label_np 와 1라운드 레이블(label_full)의 일치율은 99.31~99.62% (R2/label_np_check_dev.json).
"""
import os, sys, json, time, argparse, subprocess
R2 = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--mode', required=True, choices=['dev', 'final', 'future'])
ap.add_argument('--par', type=int, default=2)
ap.add_argument('--boot', type=int, default=1000)
ap.add_argument('--step', default='all')
ap.add_argument('--L', default=None)
A = ap.parse_args()
os.environ['T4R2_MODE'] = A.mode
if A.mode in ('final', 'future'):
    os.environ['T4R2_SAVE'] = '1'
sys.path.insert(0, R2)
import numpy as np, pandas as pd
import r2lib as R
import r2run

SPECF = os.path.join(R2, 'final_spec_r2.json')
FS = json.load(open(SPECF, encoding='utf-8'))
SPEC = FS['spec']
TAG = f'FROZEN_{A.mode}'


def sub(*args):
    r = subprocess.run([sys.executable, '-W', 'ignore', os.path.abspath(__file__), '--mode', A.mode, '--par', str(A.par), '--boot', str(A.boot)] + list(args))
    if r.returncode: sys.exit(f'단계 실패 {args}')


def frames():
    for o, _, EL in R.ORIGINS:
        for part in ['cls', 'haz'] + [f'ev{i}' for i in range(len(EL))] + ['ev']:
            r = subprocess.run([sys.executable, '-W', 'ignore', os.path.join(R2, 'r2run.py'), 'frames', o, part], env=os.environ)
            if r.returncode: sys.exit('frames 실패')


def train():
    json.dump(dict(spec=SPEC, parent='-', change='동결 spec', exp=TAG), open(os.path.join(R2, 'specs', f'{TAG}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    jobs = R.jobs_for(SPEC)
    print(A.mode, 'jobs', len(jobs), flush=True)
    if jobs:
        r2run.ensure_extra(SPEC, jobs)
        r2run.run_jobs(TAG, jobs, A.par)


def cluster_boot(y, p, pb, acd, pid, B, seed=0):
    from r2boot import wauc_prep, wauc, wap
    cl = np.where(acd == 'NA', 'P' + pid.astype(str), acd)
    codes, uniq = pd.factorize(cl); K = len(uniq)
    pm, pbb = wauc_prep(y, p), wauc_prep(y, pb)
    rng = np.random.default_rng(seed)
    R_ = {k: [] for k in ['auc', 'auc_base', 'auc_diff', 'skill', 'prauc', 'prauc_base']}
    for _ in range(B):
        w = np.bincount(rng.integers(0, K, K), minlength=K).astype(float)[codes]
        a, ab = wauc(pm, w), wauc(pbb, w)
        R_['auc'].append(a); R_['auc_base'].append(ab); R_['auc_diff'].append(a - ab); R_['skill'].append((a - ab) / (1 - ab))
        R_['prauc'].append(wap(pm, w)); R_['prauc_base'].append(wap(pbb, w))
    return {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] for k, v in R_.items()}, K


def score_final():
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    from scipy.stats import spearmanr
    o = R.ORIGINS[0][0]
    ev = pd.read_parquet(os.path.join(R.odir(o), 'ev.parquet'), columns=['pid', 'L', 'ksic', 'acd', 'atype', 'y'])
    und = int(ev.y.isna().sum())
    p, ps = R.ensemble(o, SPEC)
    pb = R.fit_predict_sk(o, SPEC, 'base')
    m = ev.y.notna().values
    y = ev.y.values[m]; p = p[m]; pb = pb[m]
    met = lambda q: dict(auc=float(roc_auc_score(y, q)), prauc=float(average_precision_score(y, q)), brier=float(brier_score_loss(y, q)), mean_pred=float(q.mean()))
    res = dict(mode='final', F=2022, C_train_label_cutoff=R.ORIGINS[0][1], eval_L=R.ORIGINS[0][2], label='label_np(납부기록, DATA_END 2025-08-22, 유예 213일)',
               n=int(m.sum()), n_undetermined_dropped=und, base_rate=float(y.mean()), model=met(p), baseline=met(pb))
    res['skill_auc'] = (res['model']['auc'] - res['baseline']['auc']) / (1 - res['baseline']['auc'])
    res['components'] = {k: float(roc_auc_score(y, v[m])) for k, v in ps.items()}
    ci, K = cluster_boot(y, p, pb, ev.acd.values[m], ev.pid.values[m], A.boot)
    res['bootstrap_95ci'] = ci; res['bootstrap_B'] = A.boot; res['n_clusters'] = K; res['cluster'] = '1순위 출원인코드(없으면 특허)'
    top = np.argsort(-p)[:int(0.1 * len(p))]; topb = np.argsort(-pb)[:int(0.1 * len(pb))]
    res['precision_top10'] = dict(model=float(y[top].mean()), baseline=float(y[topb].mean()))
    g = pd.DataFrame({'k': ev.ksic.values[m], 'p': p, 'pb': pb, 'y': y}).groupby('k').mean()
    res['industry_spearman'] = dict(model=float(spearmanr(g.p, g.y)[0]), baseline=float(spearmanr(g.pb, g.y)[0]), n_ksic=len(g))
    res['by_atype'] = {a: dict(n=int(len(gg)), auc=float(roc_auc_score(gg.y, gg.p)), auc_base=float(roc_auc_score(gg.y, gg.pb)))
                       for a, gg in pd.DataFrame({'a': ev.atype.values[m], 'p': p, 'pb': pb, 'y': y}).groupby('a') if gg.y.nunique() > 1}
    res['spec_version'] = FS.get('version')
    json.dump(res, open(os.path.join(R2, 'final_result.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps({k: res[k] for k in ['model', 'baseline', 'skill_auc', 'bootstrap_95ci']}, ensure_ascii=False, indent=1))


def score_dev():
    res = {o: R.score_origin(o, SPEC) for o, _, _ in R.ORIGINS}
    ms = float(np.mean([r['skill'] for r in res.values()]))
    exp = pd.read_csv(os.path.join(R2, 'experiments.csv'))
    row = exp[exp.exp_id == FS['source_exp']].iloc[0]
    out = dict(mode='dev', per_origin=res, mean_skill=ms, experiments_csv_mean_skill=float(row.mean_skill), reproduced=bool(abs(ms - row.mean_skill) < 1e-3))
    json.dump(out, open(os.path.join(R2, 'dev_result.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps({o: {k: round(v, 4) for k, v in r.items() if k in ('auc', 'auc_base', 'skill')} for o, r in res.items()}, indent=1))
    print('mean skill', round(ms, 4), 'experiments.csv', row.mean_skill, 'reproduced', out['reproduced'])


FUT_L = R.semi('2012-01-01', '2025-07-01')


def predict_future_L(L):
    o = R.ORIGINS[0][0]
    fn = os.path.join(R.odir(o), f'pred_{L.replace("-", "")}.parquet')
    if os.path.exists(fn): return
    d = R.predict_saved(o, SPEC, L)
    end = R.T.add_years(np.array([np.datetime64(L)]), 3)[0]
    if end + np.timedelta64(R.GRACE_DAYS, 'D') <= R.DATA_END:
        d['y_realized'] = R.label_np(L, d.pid.values)
    else:
        d['y_realized'] = np.nan
    d.to_parquet(fn, index=False)
    print('future pred', L, len(d), 'peak', R.peak_mem_gb(), flush=True)


def future_output():
    o = R.ORIGINS[0][0]
    d = pd.concat([pd.read_parquet(os.path.join(R.odir(o), f'pred_{L.replace("-", "")}.parquet')) for L in FUT_L], ignore_index=True)
    nm = pd.read_csv(R.SPEC0['paths']['ksic_names'], dtype={'세세분류코드': str}).drop_duplicates('세세분류코드').set_index('세세분류코드')['산업명']
    agg = dict(n_alive=('pid', 'size'), pred_risk=('p', 'mean'), exp_lapses_3y=('p', 'sum'), base_risk=('p_base', 'mean'), realized_rate=('y_realized', 'mean'))
    g = d.groupby(['ksic', 'L']).agg(**agg).reset_index()
    tot = d.groupby('L').agg(**agg).reset_index(); tot['ksic'] = 'ALL'
    g = pd.concat([g, tot], ignore_index=True)
    g['industry'] = g.ksic.map(nm).fillna('전체')
    g['universe_complete'] = g.L <= '2025-07-01'
    latest, base = '2025-07-01', '2014-01-01'
    cur = g[g.L == latest].set_index('ksic'); b0 = g[g.L == base].set_index('ksic')
    g['rank_risk_latest'] = g.ksic.map(cur.pred_risk.rank(ascending=False))
    g['delta_vs_2014'] = g.ksic.map(cur.pred_risk - b0.pred_risk)
    g['rank_rise_latest'] = g.ksic.map((cur.pred_risk - b0.pred_risk).rank(ascending=False))
    g['in_sample_fit'] = g.L <= '2021-07-01'       # 재학습 레이블 기간(표본 내 적합)
    g['C_train_label_cutoff'] = R.ORIGINS[0][1]
    g.sort_values(['ksic', 'L']).to_csv(os.path.join(R2, 'future_forecast_r2.csv'), index=False, encoding='utf-8-sig')
    print(g[g.L == latest].sort_values('pred_risk', ascending=False).head(10)[['ksic', 'industry', 'n_alive', 'pred_risk', 'delta_vs_2014']].to_string())


if __name__ == '__main__':
    if A.step == 'all':
        t = time.time()
        frames(); train()
        if A.mode == 'dev': sub('--step', 'score_dev')
        elif A.mode == 'final': sub('--step', 'score_final')
        else:
            for L in FUT_L: sub('--step', 'pred_future', '--L', L)
            sub('--step', 'future_output')
        print('완료', f'{time.time() - t:.0f}s')
    elif A.step == 'score_dev': score_dev()
    elif A.step == 'score_final': score_final()
    elif A.step == 'pred_future': predict_future_L(A.L)
    elif A.step == 'future_output': future_output()
