# -*- coding: utf-8 -*-
"""T3 R4 — NTIS(국가R&D, NTIS Open API 활용) 셀 피처 챔피언/도전자. 공개연도 변형, 개발 원점 2010–2018(9개), 채택 = 평균 skill +0.003 이상 AND 6/9 이상 우세.
R3 코어(r3lib·r3core)를 import 해서 모형·지표·학습 규칙(학습 t0 ≤ v−3)을 그대로 쓰고, 쓰기 경로(성분 캐시·실험 기록·챔피언)만 R4/ 로 돌린다.
R3 성분 캐시(R3/comp_pub)는 같은 키일 때 **읽기만** 한다(챔피언 재채점 재학습 없음).
사용: python t3_exps.py START | N1 N2 ...
"""
import os, sys, copy, json, pathlib
R4 = pathlib.Path(__file__).resolve().parent; R3 = R4.parent / 'R3'
os.environ.update(T3R3_VAR='pub', T3R3_CACHE='cache_pub', T3R3_MAXYEAR='2021', T3R3_LABEL_MAX='2021', T3R3_EXT_SUFFIX='')
for k in ('T3R3_FINAL_UNLOCK', 'T3R3_TRAIN_MAX_T0', 'T3R3_HIER'):
    os.environ.pop(k, None)
sys.path.insert(0, str(R3))
import numpy as np, pandas as pd
import r3lib as L
import r3core as C
assert L.VAR == 'pub' and L.Y1 == 2021 and L.LABEL_MAX_YEAR == 2021 and max(C.DEV_T0) == 2018
C.COMP_DIR = str(R4 / 'comp_pub'); os.makedirs(C.COMP_DIR, exist_ok=True)
C.EXP_CSV = str(R4 / 'experiments.csv'); C.CHAMP_JSON = str(R4 / 'champion_pub.json')
R3COMP = str(R3 / 'comp_pub')
_np_load = np.load
_exists = os.path.exists


def _load_fallback(fn, *a, **k):
    """R4 성분 캐시에 없으면 R3 성분 캐시(같은 키 = 같은 설정·같은 데이터 서명)를 읽기만 한다."""
    if isinstance(fn, str) and fn.startswith(C.COMP_DIR) and not _exists(fn):
        alt = os.path.join(R3COMP, os.path.basename(fn))
        if _exists(alt):
            return _np_load(alt, *a, **k)
    return _np_load(fn, *a, **k)


_exists = os.path.exists
def _exists_fb(p):
    if isinstance(p, str) and p.startswith(C.COMP_DIR) and p.endswith('.npy') and not _exists(p):
        return _exists(os.path.join(R3COMP, os.path.basename(p)))
    return _exists(p)


C.np.load = _load_fallback
C.os.path.exists = _exists_fb      # r3core 안의 os.path.exists 는 모듈 공유 → 캐시 존재 확인에만 영향(다른 경로는 원래 동작)

NT = pd.read_parquet(R4 / 'ntis_cell_pub.parquet')
RANKC = ['gsh_pub', 'gsh_any', 'f_grow', 'f_int']


def prep(D):
    if 'ros_sci' not in D:
        cf = pd.read_parquet(R3 / 'extB_cell_pub.parquet'); kf = pd.read_parquet(R3 / 'extB_ksic_pub.parquet')
        n0 = len(D); D = D.merge(cf, on=['cell', 't0'], how='left').merge(kf, on=['ksic', 't0'], how='left'); assert len(D) == n0
        D['oa_mg_rel'] = D.oa_mg_g - D.oa_all_g if 'oa_all_g' in D else np.nan
        D['ix_sci_fx'] = D.ros_sci_rel * D.fx_mom; D['ix_sci_kr'] = D.ros_sci_rel * D.kr_mom; D['ix_oa_gap'] = D.oa_mg_g - D.kr_mom
    if 'gsh_pub' not in D:
        n0 = len(D); D = D.merge(NT, on=['cell', 't0'], how='left'); assert len(D) == n0
        for c in RANKC:      # 원점 내 순위(커버리지가 해마다 느는 문제 → 수준 대신 순위, 결측은 결측 유지)
            D[c + '_rk'] = D.groupby('t0')[c].rank(pct=True)
        # (INFO 위약 대조) gsh_pub 순위를 같은 원점·같은 KSIC 안에서 무작위로 섞은 것 — KSIC 수준 정보는 남고 셀 수준 정보만 사라진다
        rng = np.random.default_rng(0); v = D['gsh_pub_rk'].values.copy()
        for _, ix in D.groupby(['t0', 'ksic']).indices.items():
            v[ix] = v[rng.permutation(ix)]
        D['gsh_pub_plc_rk'] = v
    return D


_gw = C.get_wide
def gw(Dx, mt):
    w = _gw(Dx, mt)
    if 'gsh_pub' not in w:
        w = prep(w); C._WIDE[mt] = w
    return w
C.get_wide = gw

SPEC = json.load(open(R3 / 'final_spec_r3.json', encoding='utf-8'))
CFG0 = SPEC['cfg']
TREE = ('xgbc', 'nb')
EXPS = {
    'N1': ('[NTIS] 셀 공공(대학·출연연) 수행 정부 기원 비중 gsh_pub(공개연도 t0−3..t0−1, 레코드 Year ≤ t0, 출원 ≥2007 분모, KSIC 수축) 원점 내 순위 → 트리 2성분. ntis_TESTS T3-N1',
           lambda cfg: C.with_feats(cfg, ['gsh_pub_rk'], comps=TREE)),
    'N2': ('[NTIS] 셀 배분 정부연구비 성장 f_grow(연구비 y ≤ t0−1, 배분표 성과 Year ≤ t0−2, t0 < 2009 NaN) 원점 내 순위 → 트리 2성분. ntis_TESTS T3-N2',
           lambda cfg: C.with_feats(cfg, ['f_grow_rk'], comps=TREE)),
    'N3': ('[NTIS] 정부 기원 비중(전체 수행주체) gsh_any 순위 + 배분 연구비 집약도 f_int 순위 → 트리 2성분',
           lambda cfg: C.with_feats(cfg, ['gsh_any_rk', 'f_int_rk'], comps=TREE)),
    'N4': ('[NTIS] gsh_pub 순위를 로지스틱에도(3성분)', lambda cfg: C.with_feats(cfg, ['gsh_pub_rk'], comps=('logit', 'xgbc', 'nb'))),
    'P1': ('(INFO 위약) N1의 gsh_pub 순위 대신 같은 원점·KSIC 안에서 섞은 값 → 트리 2성분(E40-auditD 기준 비교용)',
           lambda cfg: C.with_feats(C.with_feats(cfg, drop=['gsh_pub_rk'], comps=TREE), ['gsh_pub_plc_rk'], comps=TREE)),
}

if __name__ == '__main__':
    D = prep(C.get_D())
    args = sys.argv[1:]
    if args and args[0] == 'START':
        res = C.rescore('E40-auditD', '출발 챔피언: R3 동결 E40-auditD(final_spec_r3.json) 재채점 — R3 성분 캐시 읽기(재학습 없음)', CFG0, D=D, parent='R3')
        s = C.summarize(res)
        print('재현', round(s['mean_skill'], 4), 'vs 동결', round(SPEC['dev']['mean_skill'], 4))
        args = args[1:]
    if args and args[0] == 'ROBUST':     # INFO: 트리 2성분을 2시드 배깅으로 바꿔 E40-auditD 대 N1 비교(R3 R40s2 와 같은 강건성 점검)
        def s2(cfg):
            c2 = copy.deepcopy(cfg)
            for c in c2['components']:
                if c['type'] in TREE: c['seeds'] = 2
            return c2
        cN1 = EXPS['N1'][1](copy.deepcopy(CFG0))
        r0 = C.evaluate(s2(CFG0), D); r1 = C.evaluate(s2(cN1), D)
        s0, s1 = C.summarize(r0), C.summarize(r1)
        d = s1['mean_skill'] - s0['mean_skill']; w = float(np.mean([r1[v]['skill'] > r0[v]['skill'] for v in r1]))
        C.record('R0s2', 'E40-auditD', '(INFO 강건성) E40-auditD 트리 2성분 2시드 배깅', r0, 'INFO')
        C.record('N1s2', 'R0s2', '(INFO 강건성) N1 트리 2성분 2시드 배깅 — R0s2 대비', r1, 'INFO', delta=d, win=w)
        print('ROBUST E40-auditD s2', round(s0['mean_skill'], 4), 'N1 s2', round(s1['mean_skill'], 4), 'Δ', round(d, 4), '승률', round(w, 2),
              [round(r1[v]['skill'] - r0[v]['skill'], 4) for v in r1], flush=True)
        args = args[1:]
    for e in args:
        ch = C.load_champ()
        change, fn = EXPS[e]
        cfg = fn(copy.deepcopy(ch['cfg']))
        C.challenge(e, change, cfg, D=D, decide=not e.startswith('P'))
