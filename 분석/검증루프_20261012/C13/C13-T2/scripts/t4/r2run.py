# -*- coding: utf-8 -*-
"""T4 2라운드 챔피언/도전자 실험 드라이버.
  python r2run.py frames O16              # 원점 프레임 캐시(1회)
  python r2run.py exp E03 <mod> [--par 2] # 현재 챔피언 spec 에 r2exp.<mod> 변경 1개 적용 → 4개 원점 학습·채점 → experiments.csv
  python r2run.py job <EXP> <O> <comp> <bag|-> # (내부) 학습 작업 1개 — 각 9분 이내, 4GB 이하
  python r2run.py champ                   # 출발 챔피언(1라운드 동결 모형) 재채점 → experiments.csv 첫 줄
  python r2run.py rescore <EXP>           # 캐시된 예측으로 다시 채점만
"""
import os, sys, json, time, subprocess, copy
R2 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, R2)
import numpy as np, pandas as pd

SPECS = os.path.join(R2, 'specs'); os.makedirs(SPECS, exist_ok=True)
CHAMP = os.path.join(R2, 'champion.json')
EXPCSV = os.path.join(R2, 'experiments.csv')
MIN_GAIN, MIN_WIN = 0.003, 0.6


def load_spec(e):
    return json.load(open(os.path.join(SPECS, f'{e}.json'), encoding='utf-8'))


def run_jobs(exp, jobs, par):
    """학습 작업을 별도 프로세스로 최대 par 개 동시 실행"""
    q = list(jobs); run = []
    while q or run:
        while q and len(run) < par:
            o, comp, k = q.pop(0)
            log = open(os.path.join(R2, 'logs', f'job_{exp}_{o}_{comp}_{k}.log'), 'w', encoding='utf-8')
            p = subprocess.Popen([sys.executable, os.path.abspath(__file__), 'job', exp, o, comp, str(k) if k is not None else '-'],
                                 stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONIOENCODING='utf-8'))
            run.append((p, (o, comp, k), log, time.time()))
            print(f'[start] {o} {comp} {k} pid={p.pid}', flush=True)
        time.sleep(3)
        for r in list(run):
            p, j, log, t0 = r
            if p.poll() is not None:
                log.close(); run.remove(r)
                print(f'[done] {j} rc={p.returncode} {time.time() - t0:.0f}s', flush=True)
                if p.returncode != 0:
                    for rr in run: rr[0].kill()
                    sys.exit(f'작업 실패 {j} — 로그 logs/job_{exp}_{j[0]}_{j[1]}_{j[2]}.log')


def score(exp):
    import r2lib as R
    spec = load_spec(exp)['spec']
    res = {o: R.score_origin(o, spec) for o, _, _ in R.ORIGINS}
    return res


def log_exp(exp, parent, change, res, champ_res, note, decide=True):
    os_ = list(res.keys())
    ms = float(np.mean([res[o]['skill'] for o in os_]))
    row = dict(exp_id=exp, parent=parent, change=change)
    for o in os_:
        row[f'{o}_auc'] = round(res[o]['auc'], 4); row[f'{o}_base'] = round(res[o]['auc_base'], 4); row[f'{o}_skill'] = round(res[o]['skill'], 4)
    row['mean_auc'] = round(float(np.mean([res[o]['auc'] for o in os_])), 4)
    row['mean_skill'] = round(ms, 4)
    row['mean_ind_spearman'] = round(float(np.mean([res[o]['ind_sp'] for o in os_])), 3)
    if champ_res is None:
        row.update(diff_vs_champ=0.0, win_rate=np.nan, decision='CHAMPION(start)')
    else:
        cs = float(np.mean([champ_res[o]['skill'] for o in os_]))
        wins = np.mean([res[o]['skill'] > champ_res[o]['skill'] for o in os_])
        row['diff_vs_champ'] = round(ms - cs, 4); row['win_rate'] = round(float(wins), 2)
        row['decision'] = ('ACCEPT' if (ms - cs >= MIN_GAIN and wins >= MIN_WIN) else 'ROLLBACK') if decide else 'INFO'
    row['comps'] = json.dumps({o: res[o]['comps'] for o in os_})
    row['note'] = note
    df = pd.read_csv(EXPCSV) if os.path.exists(EXPCSV) else pd.DataFrame()
    if len(df): df = df[df.exp_id != exp]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(EXPCSV, index=False, encoding='utf-8-sig')
    json.dump(res, open(os.path.join(R2, 'runs', f'res_{exp}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in row.items() if k != 'comps'}, ensure_ascii=False, indent=0), flush=True)
    return row


def main():
    a = sys.argv[1:]
    cmd = a[0]
    if cmd == 'frames':
        if len(a) > 2:
            import r2lib as R
            R.build_frames(a[1], which=(a[2],))
            print(a[1], a[2], 'peak', R.peak_mem_gb(), flush=True)
        else:
            for part in ('cls', 'haz', 'ev0', 'ev1', 'ev'):
                r = subprocess.run([sys.executable, '-W', 'ignore', os.path.abspath(__file__), 'frames', a[1], part])
                if r.returncode: sys.exit('frames 실패')
    elif cmd == 'job':
        import r2lib as R
        exp, o, comp, k = a[1], a[2], a[3], a[4]
        spec = load_spec(exp)['spec']
        if k != '-':
            R.fit_predict_xgb_bag(o, spec, comp, int(k))
        else:
            R.fit_predict_sk(o, spec, comp)
    elif cmd == 'ext':
        import r2lib as R
        R.build_extra(a[1], a[2], a[3].split(','))
        print('peak', R.peak_mem_gb(), flush=True)
    elif cmd == 'champ':
        import r2lib as R
        spec = copy.deepcopy(R.SPEC0)
        json.dump(dict(spec=spec, parent='-', change='1라운드 동결 모형(final_spec.json v53) 재채점', exp='E00'),
                  open(os.path.join(SPECS, 'E00.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        _run_exp('E00', par=int(a[1]) if len(a) > 1 else 2)
    elif cmd == 'exp':
        exp, mod = a[1], a[2]
        par = int(a[a.index('--par') + 1]) if '--par' in a else 2
        base = a[a.index('--base') + 1] if '--base' in a else json.load(open(CHAMP))['exp']
        import r2exp
        parent = load_spec(base)
        spec, change = getattr(r2exp, mod)(copy.deepcopy(parent['spec']))
        json.dump(dict(spec=spec, parent=base, change=change, exp=exp, mod=mod), open(os.path.join(SPECS, f'{exp}.json'), 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)
        _run_exp(exp, par=par, decide='--info' not in a)
    elif cmd == 'rescore':
        _run_exp(a[1], par=2, decide='--info' not in a)


def ensure_extra(spec, jobs):
    import r2lib as R
    need = {}
    for o, comp, k in jobs:
        if comp.startswith('haz') and not os.path.exists(os.path.join(R.odir(o), R.haz_stem(spec) + '.parquet')):
            st = R.haz_stem(spec)
            r = subprocess.run([sys.executable, '-W', 'ignore', os.path.abspath(__file__), 'frames', o, 'hazs' if st == 'frame_hazs' else 'haz'])
            if r.returncode: sys.exit('frames 실패')
    for o, comp, k in jobs:
        if comp == 'base': continue
        num = spec['components'][comp]['num']
        stems = [R.haz_stem(spec), 'ev_haz'] if comp.startswith('haz') else ['frame_cls', 'ev']
        for st in stems:
            for c in R.missing_extra(o, st, num):
                need.setdefault((o, st), set()).add(c)
    for (o, st), cs in need.items():
        if not os.path.exists(os.path.join(R.odir(o), st + '.parquet')):
            r = subprocess.run([sys.executable, '-W', 'ignore', os.path.abspath(__file__), 'frames', o, 'hazs' if st == 'frame_hazs' else st])
            if r.returncode: sys.exit('frames 실패')
        r = subprocess.run([sys.executable, '-W', 'ignore', os.path.abspath(__file__), 'ext', o, st, ','.join(sorted(cs))])
        if r.returncode: sys.exit(f'ext 실패 {o} {st}')


def _run_exp(exp, par=2, decide=True):
    import r2lib as R
    S = load_spec(exp)
    spec = S['spec']
    jobs = R.jobs_for(spec)
    print(exp, S['change'], 'jobs', len(jobs), flush=True)
    t = time.time()
    ensure_extra(spec, jobs)
    if jobs:
        # 작업 프로세스들은 specs/<exp>.json 의 'spec' 을 읽는다
        run_jobs(exp, jobs, par)
    res = {o: R.score_origin(o, spec) for o, _, _ in R.ORIGINS}
    champ = json.load(open(CHAMP)) if os.path.exists(CHAMP) else None
    if exp == 'E00':
        row = log_exp(exp, '-', S['change'], res, None, f'출발 챔피언. {time.time() - t:.0f}s')
        json.dump(dict(exp='E00'), open(CHAMP, 'w'))
        return
    cres = json.load(open(os.path.join(R2, 'runs', f'res_{S["parent"]}.json'), encoding='utf-8'))
    if champ and S['parent'] != champ['exp']:
        print('주의: parent 가 현재 챔피언이 아님 — 비교 대상은 parent', flush=True)
    row = log_exp(exp, S['parent'], S['change'], res, cres, S.get('note', '') + f' {time.time() - t:.0f}s', decide=decide)
    if decide and row['decision'] == 'ACCEPT':
        json.dump(dict(exp=exp), open(CHAMP, 'w'))
        print('>>> 새 챔피언', exp, flush=True)


if __name__ == '__main__':
    main()
