# C09-T4 s2: T4 지연납부 5개 피처(n_late·late_share·mean_early·min_early·last_early)를 원판·R1·R2·R3로 다시 계산해
#  트랙 프레임(읽기 전용) 행 순서에 맞춘 npy로 저장. 원판 재계산 값은 프레임 값과 대조(재현 점검).
#  python s2_t4_feats.py extract            : R2/cache/loaded_r2dev.pkl(읽기 전용)에서 pay·pt 추출 → cache/t4_pay.parquet
#  python s2_t4_feats.py feats <O>          : 원점 O의 frame_cls·frame_haz·ev·ev_haz 4개 프레임
import os, sys, json, time, pickle, gc
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
from c9lib import *
LATE = ['n_late', 'late_share', 'mean_early', 'min_early', 'last_early']
ROOT = {'O12': 'runs', 'O14': 'runs', 'O16': 'runs', 'O18': 'runs', 'F22': 'runs_final'}
FR = ['frame_cls', 'frame_haz', 'ev', 'ev_haz']


def fdir(o):
    return os.path.join(T4D, 'R2', ROOT[o], o)


def xdir(o):
    p = os.path.join(C, 't4x', o); os.makedirs(p, exist_ok=True); return p


def extract():
    D = pickle.load(open(os.path.join(T4D, 'R2', 'cache', 'loaded_r2dev.pkl'), 'rb'))
    pay = D['pay'][['pid', 'seq', 's', 'pd', 'due', 'early']].copy()
    pt = D['pt'][['pid', 'rgst_no', 'acd', 'atype']].copy() if 'rgst_no' in D['pt'] else D['pt'][['pid', 'acd', 'atype']].copy()
    del D; gc.collect()
    hol = load_hol()
    due = pay.due.values.astype('datetime64[D]'); pdd = pay.pd.values.astype('datetime64[D]')
    chk = (due - pdd).astype('timedelta64[D]').astype(float)
    assert np.allclose(chk, pay.early.values.astype(float)), 'early 재계산 불일치'
    pay['earlyR1'] = (adjust_due(due, 'R1') - pdd).astype('timedelta64[D]').astype('float32')
    pay['earlyR2'] = (adjust_due(due, 'R2', hol) - pdd).astype('timedelta64[D]').astype('float32')
    pay.to_parquet(os.path.join(C, 't4_pay.parquet'), index=False)
    pt.to_parquet(os.path.join(C, 't4_pt.parquet'), index=False)
    info = dict(rows=len(pay), seq_gt0=int((pay.seq > 0).sum()), pd_max=str(pay.pd.max()), changed_R1=float((np.sign(pay.early) != np.sign(pay.earlyR1)).mean()))
    print(info)


def feats_at(p, L, pids):
    """t4lib.features 의 'L 이전 납부행동' 블록과 같은 계산(+보정판). p: pay(seq>0) 전체, 정렬 pid·pd·seq"""
    q = p[(p.pd.values <= np.datetime64(L)) & np.isin(p.pid.values, pids)]
    out = {}
    for rule, ecol, lat in [('orig', 'early', lambda d: d.early < 0), ('R1', 'earlyR1', lambda d: d.earlyR1 < 0),
                            ('R2', 'earlyR2', lambda d: d.earlyR2 < 0), ('R3', 'early', lambda d: d.early < -3)]:
        r = q.assign(late=lat(q).astype(int), e=q[ecol])
        agg = r.groupby('pid').agg(n_renew=('seq', 'size'), n_late=('late', 'sum'), mean_early=('e', 'mean'), min_early=('e', 'min'))
        agg['last_early'] = r.groupby('pid').e.last()
        agg = agg.reindex(pids)
        agg['n_renew'] = agg.n_renew.fillna(0); agg['n_late'] = agg.n_late.fillna(0)
        agg['late_share'] = agg.n_late / agg.n_renew.replace(0, np.nan)
        out[rule] = agg
    return out


def feats(o):
    import pyarrow.parquet as pq
    t = time.time()
    p = pd.read_parquet(os.path.join(C, 't4_pay.parquet'))
    p = p[p.seq > 0].sort_values(['pid', 'pd', 'seq'], kind='stable').reset_index(drop=True)
    log = {}
    for fr in FR:
        d = pq.read_table(os.path.join(fdir(o), fr + '.parquet'), columns=['pid', 'L'] + LATE).to_pandas()
        d['L'] = d.L.astype(str)
        res = {r: np.full((len(d), len(LATE)), np.nan, dtype='float64') for r in ['orig', 'R1', 'R2', 'R3']}
        for L, idx in d.groupby('L').indices.items():
            pids = np.unique(d.pid.values[idx])
            F = feats_at(p, L, pids)
            for r in res:
                res[r][idx] = F[r].loc[d.pid.values[idx], LATE].to_numpy(dtype='float64')
        # 재현 점검(원판 vs 프레임)
        mm = {}
        for j, c in enumerate(LATE):
            a = d[c].to_numpy(dtype='float64'); b = res['orig'][:, j]
            same = (np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= np.maximum(1e-3, 1e-4 * np.abs(a)))
            mm[c] = dict(mismatch_rate=float(1 - same.mean()), max_abs=float(np.nanmax(np.abs(a - b))) if np.isfinite(a - b).any() else 0.0)
        chg = {r: {c: float((np.abs(np.nan_to_num(res[r][:, j], nan=-9e9) - np.nan_to_num(res['orig'][:, j], nan=-9e9)) > 1e-9).mean()) for j, c in enumerate(LATE)} for r in ['R1', 'R2', 'R3']}
        for r in ['R1', 'R2', 'R3']:
            for j, c in enumerate(LATE):
                np.save(os.path.join(xdir(o), f'{fr}__{c}__{r}.npy'), res[r][:, j].astype('float32'))
        log[fr] = dict(rows=len(d), n_L=int(d.L.nunique()), repro=mm, changed_row_share=chg)
        print(o, fr, len(d), 'repro', {c: round(v['mismatch_rate'], 6) for c, v in mm.items()}, f'{time.time() - t:.0f}s', flush=True)
    json.dump(log, open(os.path.join(xdir(o), 'feat_log.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    if sys.argv[1] == 'extract': extract()
    else: feats(sys.argv[2])
