# -*- coding: utf-8 -*-
"""T4 3라운드 챔피언/도전자 드라이버 (r2run.py 를 R3 경로로 옮긴 것, 규칙 동일).
  python r3run.py champ                      # R2 챔피언(E03) 을 R3 개발 원점에서 재채점 → experiments.csv 첫 줄(E00)
  python r3run.py exp X01 <mod> [--par 2] [--base E00] [--info]
  python r3run.py job <EXP> <O> <comp> <bag|->   # (내부) 학습 작업 1개
  python r3run.py ext <O> <stem> <cols>          # (내부) 추가열 계산
  python r3run.py rescore <EXP>
"""
import os, sys, json, time, subprocess, copy
R3 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, R3)
import numpy as np, pandas as pd

SPECS = os.path.join(R3, 'specs'); os.makedirs(SPECS, exist_ok=True)
LOGS = os.path.join(R3, 'logs'); os.makedirs(LOGS, exist_ok=True)
RUNS = os.path.join(R3, 'runs'); os.makedirs(RUNS, exist_ok=True)
CHAMP = os.path.join(R3, 'champion.json')
EXPCSV = os.path.join(R3, 'experiments.csv')
MIN_GAIN, MIN_WIN = 0.003, 0.6
R2SPEC = os.path.join(os.path.dirname(R3), 'R2', 'specs', 'E03.json')


def load_spec(e):
    return json.load(open(os.path.join(SPECS, f'{e}.json'), encoding='utf-8'))


def run_jobs(exp, jobs, par):
    q = list(jobs); run = []
    while q or run:
        while q and len(run) < par:
            o, comp, k = q.pop(0)
            log = open(os.path.join(LOGS, f'job_{exp}_{o}_{comp}_{k}.log'), 'w', encoding='utf-8')
            p = subprocess.Popen([sys.executable, '-W', 'ignore', os.path.abspath(__file__), 'job', exp, o, comp, str(k) if k is not None else '-'],
                                 stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONIOENCODING='utf-8', OMP_NUM_THREADS='2'))
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
        wins = np.mean([res[o]['skill'] > champ_res[o]['skill'] + 1e-6 for o in os_])   # (B단계) 재적재 부동소수 오차(~1e-9)를 우세로 세지 않음
        row['diff_vs_champ'] = round(ms - cs, 4); row['win_rate'] = round(float(wins), 2)
        row['decision'] = ('ACCEPT' if (ms - cs >= MIN_GAIN and wins >= MIN_WIN) else 'ROLLBACK') if decide else 'INFO'
    row['comps'] = json.dumps({o: res[o]['comps'] for o in os_})
    row['note'] = note
    df = pd.read_csv(EXPCSV) if os.path.exists(EXPCSV) else pd.DataFrame()
    if len(df): df = df[df.exp_id != exp]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(EXPCSV, index=False, encoding='utf-8-sig')
    json.dump(res, open(os.path.join(RUNS, f'res_{exp}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in row.items() if k != 'comps'}, ensure_ascii=False, indent=0), flush=True)
    return row


def ensure_extra(spec, jobs, exp):
    import r3lib
    R = r3lib.R
    need = {}
    for o, comp, k in jobs:
        if comp == 'base': continue
        num = spec['components'][comp]['num']
        stems = [R.haz_stem(spec), 'ev_haz'] if comp.startswith('haz') else ['frame_cls', 'ev']
        for st in stems:
            for c in R.missing_extra(o, st, num):
                need.setdefault((o, st), set()).add(c)
    for (o, st), cs in sorted(need.items()):
        grp = {}
        for c in sorted(cs): grp.setdefault(c.split('_')[0], []).append(c)
        for g, gc_ in grp.items():                       # 접두사 그룹마다 별도 프로세스(메모리 4GB 이하)
            log = os.path.join(LOGS, f'ext_{exp}_{o}_{st}_{g}.log')
            with open(log, 'w', encoding='utf-8') as fh:
                r = subprocess.run([sys.executable, '-W', 'ignore', os.path.abspath(__file__), 'ext', o, st, ','.join(gc_)],
                                   stdout=fh, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONIOENCODING='utf-8'))
            print(open(log, encoding='utf-8').read().strip()[-400:], flush=True)
            if r.returncode: sys.exit(f'ext 실패 {o} {st} — {log}')


def _run_exp(exp, par=2, decide=True):
    import r3lib
    R = r3lib.R
    S = load_spec(exp)
    spec = S['spec']
    for o, _, _ in R.ORIGINS:
        r3lib.seed_frames(o, S.get('reuse_r2_x', []))
    for comp, src in S.get('copy_bags', {}).items():      # n_bag 만 늘린 경우: 같은 seed·열순서의 기존 구성원은 동일 모형이므로 복사
        src = S['parent'] if src == 'PARENT' else src
        ss = load_spec(src)['spec']
        a = {k: v for k, v in ss['components'][comp].items() if k != 'n_bag'}
        b = {k: v for k, v in spec['components'][comp].items() if k != 'n_bag'}
        assert a == b and {k: v for k, v in ss['train'].items()} == spec['train'], 'n_bag 외 차이가 있으면 복사 불가'
        ko, kn = R.comp_key(ss, comp), R.comp_key(spec, comp)
        for o, _, _ in R.ORIGINS:
            for k in range(ss['components'][comp]['n_bag']):
                f0 = os.path.join(R.odir(o), 'pred', f'{ko}_b{k}.npy'); f1 = os.path.join(R.odir(o), 'pred', f'{kn}_b{k}.npy')
                if not os.path.exists(f1):
                    import shutil; shutil.copyfile(f0, f1)
    jobs = R.jobs_for(spec)
    print(exp, S['change'], 'jobs', len(jobs), flush=True)
    t = time.time()
    ensure_extra(spec, jobs, exp)
    if jobs:
        run_jobs(exp, jobs, par)
    res = {o: R.score_origin(o, spec) for o, _, _ in R.ORIGINS}
    if exp == 'E00':
        log_exp(exp, '-', S['change'], res, None, S.get('note', '') + f' {time.time() - t:.0f}s')
        json.dump(dict(exp='E00'), open(CHAMP, 'w'))
        return
    cres = json.load(open(os.path.join(RUNS, f'res_{S["parent"]}.json'), encoding='utf-8'))
    champ = json.load(open(CHAMP))
    if S['parent'] != champ['exp']:
        print('주의: parent 가 현재 챔피언이 아님 — 비교 대상은 parent', flush=True)
    row = log_exp(exp, S['parent'], S['change'], res, cres, S.get('note', '') + f' {time.time() - t:.0f}s', decide=decide)
    if decide and row['decision'] == 'ACCEPT':
        json.dump(dict(exp=exp), open(CHAMP, 'w'))
        print('>>> 새 챔피언', exp, flush=True)


def main():
    a = sys.argv[1:]
    cmd = a[0]
    if cmd == 'job':
        import r3lib
        R = r3lib.R
        exp, o, comp, k = a[1], a[2], a[3], a[4]
        spec = load_spec(exp)['spec']
        if k != '-':
            R.fit_predict_xgb_bag(o, spec, comp, int(k))
        else:
            R.fit_predict_sk(o, spec, comp)
    elif cmd == 'ext':
        import r3lib
        r3lib.R.build_extra(a[1], a[2], a[3].split(','))
        print('peak', r3lib.R.peak_mem_gb(), flush=True)
    elif cmd == 'champ':
        import r3lib
        R = r3lib.R
        S2 = json.load(open(R2SPEC, encoding='utf-8'))
        spec = S2['spec']
        n = 0
        for o, _, _ in R.ORIGINS:
            r3lib.seed_frames(o)
            n += r3lib.seed_preds(o, spec)
        json.dump(dict(spec=spec, parent='-', exp='E00', mod='R2:E03',
                       change='2라운드 챔피언(R2 E03: v53 + 위험모형 가중 0.4) 을 R3 개발 원점(=R2 개발 원점 O12·O14·O16·O18)에서 재채점',
                       note=f'R2 예측 파일 {n}개를 해시 일치로 복사해 재채점(재학습 없음).'),
                  open(os.path.join(SPECS, 'E00.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        _run_exp('E00', par=2)
    elif cmd == 'exp':
        exp, mod = a[1], a[2]
        par = int(a[a.index('--par') + 1]) if '--par' in a else 2
        base = a[a.index('--base') + 1] if '--base' in a else json.load(open(CHAMP))['exp']
        import r3exp
        parent = load_spec(base)
        out = getattr(r3exp, mod)(copy.deepcopy(parent['spec']))
        spec, change = out[0], out[1]
        extra = out[2] if len(out) > 2 else {}
        json.dump(dict(spec=spec, parent=base, change=change, exp=exp, mod=mod, **extra),
                  open(os.path.join(SPECS, f'{exp}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        _run_exp(exp, par=par, decide='--info' not in a)
    elif cmd == 'rescore':
        _run_exp(a[1], par=2, decide='--info' not in a)


if __name__ == '__main__':
    main()
