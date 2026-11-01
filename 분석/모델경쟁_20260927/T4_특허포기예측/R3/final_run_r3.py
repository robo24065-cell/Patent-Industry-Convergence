# -*- coding: utf-8 -*-
"""T4 특허 포기(등록료불납) 예측 — 3라운드 동결 파이프라인. 모형·피처·학습 가중·하이퍼파라미터는 final_spec_r3.json 에서만 읽는다.

  python final_run_r3.py --mode dev      # 개발 원점 4개(O12·O14·O16·O18) 재현 → R3/dev_result_r3.json (experiments.csv 챔피언 행과 대조)
  python final_run_r3.py --mode future   # 전체 가용자료(등록원부 2025-08-22)로 재학습 → KSIC × 반기 기준일 포기위험 추이 R3/future_forecast_r3.csv
옵션: --par N 동시 학습 프로세스(기본 2, 각 4GB 이하·nthread 2)

3라운드 T4 에는 새 최종시험이 없다(PROTOCOL_R3 §3: 등록원부 납부기록이 2025-08 까지라 L=2023 이후 3년 결과를 관측할 수 없음).
따라서 --mode final 은 없다. 공식 성적은 2라운드 최종시험 skill 0.246 [0.133, 0.320] 을 유지한다.

구현: R3/r3lib.py 가 2라운드 r2lib 를 읽기 전용으로 import 하고 쓰기 경로를 R3 로 돌린다(모드는 환경변수 T4R3_MODE).
  dev    → R3/runs/<O>/ (A·B단계 실험과 같은 예측 파일을 해시 일치로 재사용 — 같은 spec 이면 재학습 0건)
  future → R3/runs_future/FUT/ (프레임은 R2/runs_future/FUT 에서 복사, 모형은 새 spec 으로 재학습해 R3 에 저장)
"""
import os, sys, json, time, argparse, subprocess
R3 = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--mode', required=True, choices=['dev', 'future'])
ap.add_argument('--par', type=int, default=2)
ap.add_argument('--step', default='all')
ap.add_argument('--L', default=None)
A = ap.parse_args()
os.environ['T4R3_MODE'] = A.mode
os.environ['PYTHONIOENCODING'] = 'utf-8'
if A.mode == 'future':
    os.environ['T4R2_SAVE'] = '1'
sys.path.insert(0, R3)
import numpy as np, pandas as pd
import r3lib
import r3run
R = r3lib.R

SPECF = os.path.join(R3, 'final_spec_r3.json')
FS = json.load(open(SPECF, encoding='utf-8'))
SPEC = FS['spec']
TAG = f'FROZEN_R3_{A.mode}'
assert not SPEC.get('ensemble', {}).get('pool'), '동결 spec 에 앙상블 풀(pool)이 있으면 predict_saved 가 지원하지 않는다'


def _patch_r2ext():
    """predict_saved(future) 가 r2ext.add 로 누락 열을 만들 때 3라운드 새 피처(r3ext 접두사)도 계산되게 한다."""
    import r2ext, r3ext
    if getattr(r2ext.add, '_r3', False):
        return
    orig = r2ext.add

    def add2(d, cols):
        new = [c for c in cols if c.split('_')[0] in r3ext.NEW]
        old = [c for c in cols if c not in new]
        if old:
            d = orig(d, old)
        if new:
            d = r3ext.add(d, new)
        return d
    add2._r3 = True
    r2ext.add = add2


def r2cache_snapshot():
    c = os.path.join(os.path.dirname(R3), 'R2', 'cache')
    return {f: os.path.getmtime(os.path.join(c, f)) for f in os.listdir(c)}


def sub(*args):
    r = subprocess.run([sys.executable, '-W', 'ignore', os.path.abspath(__file__), '--mode', A.mode, '--par', str(A.par)] + list(args))
    if r.returncode: sys.exit(f'단계 실패 {args}')


def train():
    json.dump(dict(spec=SPEC, parent='-', change='동결 spec(3라운드)', exp=TAG), open(os.path.join(R3, 'specs', f'{TAG}.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    for o, _, _ in R.ORIGINS:
        r3lib.seed_frames(o, FS.get('reuse_r2_x', []))
    jobs = R.jobs_for(SPEC)
    print(A.mode, 'jobs', len(jobs), flush=True)
    if jobs:
        r3run.ensure_extra(SPEC, jobs, TAG)
        r3run.run_jobs(TAG, jobs, A.par)


def score_dev():
    res = {o: R.score_origin(o, SPEC) for o, _, _ in R.ORIGINS}
    ms = float(np.mean([r['skill'] for r in res.values()]))
    exp = pd.read_csv(os.path.join(R3, 'experiments.csv'))
    row = exp[exp.exp_id == FS['source_exp']].iloc[0]
    e0 = exp[exp.exp_id == 'E00'].iloc[0]
    out = dict(mode='dev', source_exp=FS['source_exp'], per_origin=res, mean_skill=ms, mean_auc=float(np.mean([r['auc'] for r in res.values()])),
               experiments_csv_mean_skill=float(row.mean_skill), reproduced=bool(abs(ms - row.mean_skill) < 1e-3),
               start_champion_E00_mean_skill=float(e0.mean_skill), gain_vs_E00=round(ms - float(e0.mean_skill), 4),
               official_score='2라운드 최종시험 skill 0.246 [0.133, 0.320] 유지(3라운드 T4 새 최종시험 없음)')
    json.dump(out, open(os.path.join(R3, 'dev_result_r3.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps({o: {k: round(v, 4) for k, v in r.items() if k in ('auc', 'auc_base', 'skill')} for o, r in res.items()}, indent=1))
    print('mean skill', round(ms, 4), 'experiments.csv', row.mean_skill, 'reproduced', out['reproduced'])


FUT_L = R.semi('2012-01-01', '2025-07-01')


def predict_future_L(L):
    _patch_r2ext()
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
    g['spec_version'] = FS.get('version')
    g.sort_values(['ksic', 'L']).to_csv(os.path.join(R3, 'future_forecast_r3.csv'), index=False, encoding='utf-8-sig')
    # 2라운드 동결 모형과의 비교(같은 기준일·KSIC)
    r2f = os.path.join(os.path.dirname(R3), 'R2', 'future_forecast_r2.csv')
    if os.path.exists(r2f):
        r2 = pd.read_csv(r2f, dtype={'ksic': str})
        m = g.merge(r2[['ksic', 'L', 'pred_risk']], on=['ksic', 'L'], suffixes=('', '_r2'))
        lt = m[(m.L == latest) & (m.ksic != 'ALL')]
        from scipy.stats import spearmanr
        info = dict(latest=latest, spearman_ksic_risk_r3_vs_r2=float(spearmanr(lt.pred_risk, lt.pred_risk_r2)[0]),
                    mean_risk_r3=float(lt.pred_risk.mean()), mean_risk_r2=float(lt.pred_risk_r2.mean()))
        json.dump(info, open(os.path.join(R3, 'future_vs_r2.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print(info)
    print(g[g.L == latest].sort_values('pred_risk', ascending=False).head(10)[['ksic', 'industry', 'n_alive', 'pred_risk', 'delta_vs_2014']].to_string())


if __name__ == '__main__':
    if A.step == 'all':
        t = time.time()
        snap0 = r2cache_snapshot()
        train()
        if A.mode == 'dev':
            sub('--step', 'score_dev')
        else:
            for L in FUT_L: sub('--step', 'pred_future', '--L', L)
            sub('--step', 'future_output')
        snap1 = r2cache_snapshot()
        changed = sorted(f for f in snap1 if f not in snap0 or snap1[f] != snap0[f])
        print('R2/cache 변경 파일', changed if changed else '없음(읽기 전용 유지)')
        print('완료', f'{time.time() - t:.0f}s')
    elif A.step == 'score_dev': score_dev()
    elif A.step == 'pred_future': predict_future_L(A.L)
    elif A.step == 'future_output': future_output()
