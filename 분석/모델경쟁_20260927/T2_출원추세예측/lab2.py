# 2단계 추가 개선 실험실: final_run.py 의 구성원/결합을 바꿔 검증(원점 2012–2015, h=1..3) 평가
# 구성원 예측을 원점별로 캐시(cache2/members_*.pkl) — 모든 원점은 truncate(D, o) 사본만 사용
import os, sys, json, pickle, itertools, csv, datetime
import numpy as np, pandas as pd
import final_run as FR

kc, pr = FR.build_panel(2018)
D = FR.make_data(kc, pr, 2018)
H = 3
NB = FR.evaluate(D, None, fn=FR.naive_fc)
NBM = NB.ase.mean()
ORIG_ALL = list(range(2008, 2016))  # 2008–2011 은 스태킹 내부 백테스트 전용(목표연도 ≤ 각 검증원점)


def member_cache(name, kw):
    key = name + '_' + '_'.join(f'{k}{v}' for k, v in sorted(kw.items())).replace(' ', '').replace(',', '-').replace('(', '').replace(')', '')
    p = os.path.join(FR.CACHE, f'mem_{key}.pkl')
    if os.path.exists(p):
        return pickle.load(open(p, 'rb'))
    out = {}
    for o in ORIG_ALL:
        E = FR.truncate(D, o)
        out[o] = np.asarray(FR._fn_of(name)(E, o, H, **kw), float)
    pickle.dump(out, open(p, 'wb'))
    return out


def rows_from(Fdict, origins=FR.VAL_ORIGINS):
    rs = []
    for o in origins:
        c = o - FR.Y0
        rs.append(FR.score_rows(D, o, H, Fdict[o], D['n'][:, c + 1:c + 1 + H]))
    return pd.concat(rs, ignore_index=True)


def rep(r, name):
    s = r.ase.mean(); sk = 1 - s / NBM
    byo = {o: 1 - r[r.origin == o].ase.mean() / NB[NB.origin == o].ase.mean() for o in FR.VAL_ORIGINS}
    byc = {c: 1 - r[r.cc == c].ase.mean() / NB[NB.cc == c].ase.mean() for c in FR.CCS}
    byh = {h: 1 - r[r.h == h].ase.mean() / NB[NB.h == h].ase.mean() for h in (1, 2, 3)}
    print(f'[{name}] MASE={s:.4f} skill={sk:.4f} WAPE={r.ae.sum()/r.y.sum():.4f} | h ' + ' '.join(f'{v:.3f}' for v in byh.values())
          + ' | o ' + ' '.join(f'{v:.3f}' for v in byo.values()) + ' | ' + ' '.join(f'{k}:{v:.3f}' for k, v in byc.items()))
    return dict(MASE=s, skill=sk, byo=byo, byc=byc, byh=byh)


def log(version, change, hyp, s, kept, note):
    p = os.path.join(FR.T, 'leaderboard.csv')
    with open(p, 'a', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerow([version, datetime.date.today().isoformat(), change, hyp, 'MASE(val 2012-15,h1-3)',
                                f"{s['MASE']:.4f}", f'{NBM:.4f}', f"{s['skill']:.4f}", kept, note])


def combine(mems, how='mean', w=None):
    out = {}
    for o in ORIG_ALL:
        A = np.stack([m[o] for m in mems])
        if how == 'mean':
            out[o] = A.mean(0) if w is None else np.tensordot(np.asarray(w) / np.sum(w), A, 1)
        elif how == 'geo':
            out[o] = np.expm1(np.log1p(np.maximum(A, 0)).mean(0))
        elif how == 'median':
            out[o] = np.median(A, 0)
    return out


BASE = {'ridge_lead': {'alpha': 10.0, 'tmin': 2005, 'clipq': 0.99},
        'flow_um': {'k': 8, 'phi': 0.8, 'lag': 1, 'rk': 1, 'um_split': True, 'um_min': 20},
        'topdown': {'kt': 8, 'pt': 1.0, 'ks': 8, 'ps': 0.8}}
