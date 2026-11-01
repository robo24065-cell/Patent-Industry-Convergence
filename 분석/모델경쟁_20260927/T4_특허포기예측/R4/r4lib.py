# -*- coding: utf-8 -*-
"""T4 4라운드 하네스 — R3 하네스(r3lib: R2 r2lib dev 모드 + 출원인 가중 상한)를 **읽기 전용으로 import** 하고, 쓰기 경로만 R4/ 로 돌린다.
- 개발 원점 O12·O14·O16·O18(기준일 L ≤ 2018-07-01, 결과 창 끝 ≤ 2021-07 < F=2022). 소멸 이벤트는 dev 모드(E < 2022-01-01)만 메모리에 올린다.
- 원점 프레임은 R3/runs/<O>/ 에서 R4/runs/<O>/ 로 복사한다(R3 파일은 읽기만). 챔피언 X26 구성요소 예측도 해시 일치 파일만 복사(재학습 없이 재채점).
- 새 피처 gv_(NTIS 정부R&D 기원, NTIS Open API 활용)는 r4ext.add 로 계산해 R4/runs/<O>/x/ 에 저장한다.
"""
import os, sys, shutil, json, hashlib
R4 = os.path.dirname(os.path.abspath(__file__))
T4 = os.path.dirname(R4)
R3DIR = os.path.join(T4, 'R3')
for p in (R4, R3DIR):
    if p not in sys.path:
        sys.path.insert(1, p)
os.environ.setdefault('T4R3_MODE', 'dev')
assert os.environ['T4R3_MODE'] == 'dev', '4라운드 T4 는 개발 원점만(최종시험 없음)'
import r3lib                                   # noqa: E402  (R2 r2lib 를 dev 모드로 불러오고 가중 상한 패치를 건다)
R = r3lib.R
assert R.MODE == 'dev' and str(R.LAP_CUT) == '2022-01-01'
import numpy as np, pandas as pd               # noqa: E402

RUNS4 = os.path.join(R4, 'runs')


def odir(o):
    p = os.path.join(RUNS4, o)
    os.makedirs(os.path.join(p, 'pred'), exist_ok=True)
    return p


def mdir(o):
    p = os.path.join(odir(o), 'models'); os.makedirs(p, exist_ok=True)
    return p


R.odir = odir
R.mdir = mdir
r3lib.odir = odir


def seed_frames(o):
    src = os.path.join(R3DIR, 'runs', o); dst = odir(o)
    for f in r3lib.FRAME_FILES:
        if not os.path.exists(os.path.join(dst, f)):
            shutil.copyfile(os.path.join(src, f), os.path.join(dst, f))
    os.makedirs(os.path.join(dst, 'x'), exist_ok=True)


def seed_preds(o, spec):
    """R3 에 있는 같은 해시의 구성요소·기준선 예측을 복사(같은 spec·같은 입력 → 같은 모형)."""
    src = os.path.join(R3DIR, 'runs', o, 'pred'); dst = os.path.join(odir(o), 'pred')
    keys = [R.comp_key(spec, c) for c in spec['ensemble']['weights']]
    keys.append('base_' + hashlib.md5(json.dumps(spec['baseline'], sort_keys=True).encode()).hexdigest()[:10])
    n = 0
    for f in os.listdir(src):
        if any(f == k + '.npy' or f.startswith(k + '_b') for k in keys):
            if not os.path.exists(os.path.join(dst, f)):
                shutil.copyfile(os.path.join(src, f), os.path.join(dst, f)); n += 1
    return n


def build_extra(o, stem, cols):
    import r4ext
    idc = ['pid', 'L', 'ksic', 'next_annl', 'k', 'acd', 'age', 'atype']
    d = R._read_frame(o, stem + '.parquet', idc)
    d = r4ext.add(d, cols)
    for c in cols:
        np.save(R.xpath(o, stem, c), d[c].values.astype('float32'))
    print(o, stem, 'extra', cols, 'nan', {c: round(float(np.isnan(d[c].values.astype('float32')).mean()), 3) for c in cols},
          'mean', {c: round(float(np.nanmean(d[c].values.astype('float32'))), 4) for c in cols}, flush=True)


R.build_extra = build_extra
