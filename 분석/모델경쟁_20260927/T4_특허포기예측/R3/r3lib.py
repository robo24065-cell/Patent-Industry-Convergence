# -*- coding: utf-8 -*-
"""T4 3라운드 하네스 — 2라운드 r2lib 를 **읽기 전용으로 import** 하고, 모든 쓰기 경로만 R3/ 로 돌린다.

- 개발 원점은 PROTOCOL_R3 §2 = 2라운드와 같다(O12·O14·O16·O18, 기준일 L <= 2018-07-01, 결과 창 끝 <= 2021-07 < F=2022).
- 소멸 이벤트 적재는 r2lib 의 dev 모드(E < 2022-01-01) 그대로. 🔒 2022 이후 결과는 메모리에 올리지 않는다.
- 원점 프레임(ev/ev_haz/frame_*)은 R2/runs/<O>/ 의 파일을 R3/runs/<O>/ 로 **복사**해서 쓴다(R2 파일은 읽기만).
- 새 외부 피처는 r3ext.add 로 계산해 R3/runs/<O>/x/ 에 저장한다.
"""
import os, sys, shutil, json

R3 = os.path.dirname(os.path.abspath(__file__))
T4 = os.path.dirname(R3)
R2DIR = os.path.join(T4, 'R2')
for p in (R3, R2DIR, T4):
    if p not in sys.path:
        sys.path.insert(1, p)
MODE3 = os.environ.get('T4R3_MODE', 'dev')   # dev(기본) | future — (B단계) final_run_r3.py --mode future 만 future 로 둔다
assert MODE3 in ('dev', 'future'), '3라운드 T4 는 최종시험 모드가 없다(PROTOCOL_R3 §3)'
os.environ['T4R2_MODE'] = MODE3          # 🔒 3라운드 T4 는 개발 원점만(최종시험 없음, PROTOCOL_R3 §3)

import r2lib as R                          # noqa: E402  (R2/cache 는 읽기만: loaded_r2dev.pkl, feat_*.parquet …)
if MODE3 == 'dev':
    assert R.MODE == 'dev' and str(R.LAP_CUT) == '2022-01-01'
else:
    assert R.MODE == 'future'

import numpy as np, pandas as pd           # noqa: E402

RUNS3 = os.path.join(R3, 'runs' if MODE3 == 'dev' else 'runs_future')
R2RUNS = 'runs' if MODE3 == 'dev' else 'runs_future'
FRAME_FILES = ['ev.parquet', 'ev_haz.parquet', 'frame_base.parquet', 'frame_cls.parquet', 'frame_cls.json',
               'frame_haz.parquet', 'frame_haz.json']


def odir(o):
    p = os.path.join(RUNS3, o)
    os.makedirs(os.path.join(p, 'pred'), exist_ok=True)
    return p


R.odir = odir                              # r2lib 내부 함수들이 호출 시점에 전역 odir 를 찾으므로 여기서 교체


def mdir(o):
    p = os.path.join(odir(o), 'models'); os.makedirs(p, exist_ok=True)
    return p


R.mdir = mdir


def seed_frames(o, extra_x=()):
    """R2 원점 프레임을 R3 로 복사(없을 때만). extra_x: 재사용할 R2 추가열(x/*.npy) 이름"""
    src = os.path.join(R2DIR, R2RUNS, o); dst = odir(o)
    for f in FRAME_FILES:
        if not os.path.exists(os.path.join(dst, f)):
            shutil.copyfile(os.path.join(src, f), os.path.join(dst, f))
    os.makedirs(os.path.join(dst, 'x'), exist_ok=True)
    for c in extra_x:
        for stem in ('frame_cls', 'frame_haz', 'ev', 'ev_haz'):
            s = os.path.join(src, 'x', f'{stem}__{c}.npy'); d = os.path.join(dst, 'x', f'{stem}__{c}.npy')
            if os.path.exists(s) and not os.path.exists(d):
                shutil.copyfile(s, d)


def seed_preds(o, spec, comps=None):
    """R2 에 이미 있는 같은 해시의 구성요소 예측을 R3 로 복사(재학습 없이 재채점). 해시는 spec 내용으로 결정되므로 같은 모형이다."""
    import hashlib
    src = os.path.join(R2DIR, R2RUNS, o, 'pred'); dst = os.path.join(odir(o), 'pred')
    keys = [R.comp_key(spec, c) for c in (comps or spec['ensemble']['weights'])]
    keys.append('base_' + hashlib.md5(json.dumps(spec['baseline'], sort_keys=True).encode()).hexdigest()[:10])
    n = 0
    for f in os.listdir(src):
        if any(f == k + '.npy' or f.startswith(k + '_b') for k in keys):
            if not os.path.exists(os.path.join(dst, f)):
                shutil.copyfile(os.path.join(src, f), os.path.join(dst, f)); n += 1
    return n


def build_extra(o, stem, cols):
    """frame 에 없는 열을 r3ext(새 피처) / r2ext(2라운드 피처) 로 계산해 행 순서대로 npy 저장"""
    import r3ext
    idc = ['pid', 'L', 'ksic', 'next_annl', 'k', 'acd', 'age', 'atype']
    d = R._read_frame(o, stem + '.parquet', idc)
    d = r3ext.add(d, cols)
    for c in cols:
        np.save(R.xpath(o, stem, c), d[c].values.astype('float32'))
    print(o, stem, 'extra', cols, 'nan', {c: round(float(np.isnan(d[c].values.astype('float32')).mean()), 3) for c in cols}, flush=True)


R.build_extra = build_extra


# ---------------------------------------------------------------- (3라운드 재시도) 앙상블 풀: 이미 학습된 다른 실험의 같은 구성요소를 로짓 평균
_comp_pred0 = R.comp_pred


def comp_pred(o, spec, comp):
    pool = spec.get('ensemble', {}).get('pool', {}).get(comp)
    if not pool:
        return _comp_pred0(o, spec, comp)
    zs = []
    for e in pool:
        se = json.load(open(os.path.join(R3, 'specs', f'{e}.json'), encoding='utf-8'))['spec']
        p = _comp_pred0(o, se, comp)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        zs.append(np.log(p / (1 - p)))
    z = np.mean(zs, 0)
    return 1 / (1 + np.exp(-z))


R.comp_pred = comp_pred


# ---------------------------------------------------------------- (재시도) 출원인 가중 상한: spec['train']['acd_cap']=c 이면 학습행 가중치에
# min(1, c*N_L / n_{a,L}) 를 곱한다(기준일 L 마다 한 출원인의 가중 합이 전체의 c 를 넘지 않게). 'NA'(코드 없음)는 1.
_filter0, _recw0 = R._filter_window, R.recency_w
_CUR = {}


def _filter_window(d, spec):
    d = _filter0(d, spec)
    _CUR.clear()
    if spec['train'].get('acd_cap') and 'acd' in d.columns:
        c = spec['train']['acd_cap']
        a = d.acd.astype(str).values; Ls = d.L.astype(str).values
        key = pd.Series(Ls + '|' + a, index=d.index)
        n_a = key.map(key.value_counts()).values.astype(float)
        N = pd.Series(Ls, index=d.index).map(pd.Series(Ls).value_counts()).values.astype(float)
        pw = spec['train'].get('acd_cap_pow', 1.0)          # (B단계) 1.0 = 하드 상한(X21), 0.5 = 제곱근 완화 상한
        aw = np.minimum(1.0, (c * N / n_a) ** pw); aw[a == 'NA'] = 1.0
        _CUR['aw'] = pd.Series(aw, index=d.index); _CUR['L'] = pd.Series(Ls, index=d.index)
    return d


def recency_w(Lvals, spec):
    w = _recw0(Lvals, spec)
    if spec['train'].get('acd_cap') and _CUR:
        aw, Ls = _CUR['aw'], _CUR['L']
        if len(Lvals) != len(aw):                      # sklearn 구성요소: d.sample(n, random_state=0) 과 같은 위치 재현
            aw = aw.sample(n=len(Lvals), random_state=0); Ls = Ls.sample(n=len(Lvals), random_state=0)
        assert np.array_equal(np.asarray(Lvals).astype(str), Ls.values), '가중치 행 정렬 불일치'
        w = w * aw.values
        print('  acd_cap 가중: 평균', round(float(aw.mean()), 3), '최소', round(float(aw.min()), 3), flush=True)
    return w


R._filter_window = _filter_window
R.recency_w = recency_w
