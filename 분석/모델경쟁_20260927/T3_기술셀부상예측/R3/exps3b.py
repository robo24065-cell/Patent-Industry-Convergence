"""T3 3라운드 B단계 실험 (공개연도 변형, 개발 원점 2010–2018). 각 실험 = 현재 챔피언(champion_pub.json)에 변경 1개.
신규 수집 외부데이터: RoS 셀 과학 의존도(r3ros), OpenAlex 셀 논문(r3oa), KOSIS 품목편·생산능력·국내공급(r3kosis). → build_ext_r3b.py
사용: python exps3b.py E32 E33 ...   (--force 재실행, --info 판정 없이 기록)
"""
import os, sys, copy, argparse
ap = argparse.ArgumentParser(); ap.add_argument('--force', action='store_true'); ap.add_argument('--info', action='store_true')
ap.add_argument('exps', nargs='+'); a = ap.parse_args()
os.environ['T3R3_VAR'] = 'pub'
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import r3core as C

EXTB_SUFFIX = os.environ.get('T3R3_EXTB_SUFFIX', '')
def prep(D):
    """B단계 외부 피처 결합(셀·KSIC) + 파생."""
    if 'ros_sci' in D: return D
    cf = pd.read_parquet(os.path.join(HERE, f'extB_cell_pub{EXTB_SUFFIX}.parquet'))
    kf = pd.read_parquet(os.path.join(HERE, f'extB_ksic_pub{EXTB_SUFFIX}.parquet'))
    n0 = len(D); D = D.merge(cf, on=['cell', 't0'], how='left').merge(kf, on=['ksic', 't0'], how='left'); assert len(D) == n0
    D['oa_mg_rel'] = D.oa_mg_g - D.oa_all_g if 'oa_all_g' in D else np.nan       # 셀(메인그룹) 논문 성장 − KSIC 논문 성장
    D['ix_sci_fx'] = D.ros_sci_rel * D.fx_mom                                   # 과학 의존 셀 × 해외 특허 모멘텀
    D['ix_sci_kr'] = D.ros_sci_rel * D.kr_mom
    D['ix_oa_gap'] = D.oa_mg_g - D.kr_mom                                        # 과학 성장 대비 KR 특허 지체
    return D

def comp(cfg, t): return [c for c in cfg['components'] if c['type'] == t][0]
def set_param(t, **kw):
    def f(cfg):
        c2 = copy.deepcopy(cfg); c = comp(c2, t)
        for k, v in kw.items():
            if k == 'params': c['params'].update(v)
            else: c[k] = v
        return c2
    return f
def set_top(**kw):
    def f(cfg):
        c2 = copy.deepcopy(cfg); c2.update(kw); return c2
    return f
def feats(add=(), drop=(), comps=('logit', 'xgbc', 'nb')):
    return lambda cfg: C.with_feats(cfg, add, drop, comps)
def chain(*fs):
    def f(cfg):
        for g in fs: cfg = g(cfg)
        return cfg
    return f

ROS = ['ros_lnus', 'ros_sci', 'ros_sci_rel', 'ros_ppp', 'ros_sci_chg']
OAC = ['oa_mg_g', 'oa_mg_lvl', 'oa_mg_int', 'oa_mg_rel']
KI = ['ki_gprod', 'ki_gprod2', 'ki_dinv', 'ki_invship']
CU = ['cu_cap_g1', 'cu_util_g1', 'cu_cap_g3', 'ds_imp_g1', 'ds_impdom_g3']

EXPS = {
 'E32': ('RoS 셀 과학 의존도 5피처(연결 US 특허 수, 논문인용 특허 비중[KSIC 축소], KSIC 대비 편차, 특허당 논문 인용, 최근5년−이전5년 비중 변화; US 발행 t0-9..t0, 연결 레코드 출원 ≤t0-2) → 3성분',
         feats(ROS), '조합 A 과학 고리를 실측 인용으로. 선행검정 9/9 양'),
 'E33': ('RoS 셀 과학 의존도 5피처 → 트리 2성분(XGB 분류·음이항)만', feats(ROS, comps=('xgbc', 'nb')), ''),
 'E34': ('OpenAlex 셀(메인그룹 460개) 논문 4피처(전세계 점유 3년 성장, 논문 규모, 과학 집약도=논문/셀 특허, 메인그룹−KSIC 논문 성장 차; 발행 ≤t0, 미수집 NaN) → 3성분',
         feats(OAC), '셀 수준 과학→특허 선행. 적격 셀 연결 49–58%'),
 'E35': ('OpenAlex 셀 논문 4피처 → 트리 2성분만(결측을 트리가 처리)', feats(OAC, comps=('xgbc', 'nb')), ''),
 'E36': ('KOSIS 연간 품목편 4피처(연쇄 생산액 성장 1년·2년 평균, 재고/출하 변화, 재고/출하 수준; 자료연도 ≤t0-2) → 3성분', feats(KI), '수요 견인(재고 소진) 가설'),
 'E37': ('KOSIS 생산능력·가동률·국내공급 5피처(생산능력 1·3년 성장, 가동률 1년 변화, 수입 공급 1년 성장, 수입−국산 3년 성장차; 연도 ≤t0-1) → 3성분', feats(CU), '설비투자·수입대체 압력'),
}

def load_more():
    fn = os.path.join(HERE, 'exps3b_more.py')
    if os.path.exists(fn):
        ns = dict(C=C, feats=feats, set_param=set_param, set_top=set_top, chain=chain, comp=comp, copy=copy, pd=pd, np=np,
                  ROS=ROS, OAC=OAC, KI=KI, CU=CU, os=os, HERE=HERE)
        exec(open(fn, encoding='utf-8').read(), ns); EXPS.update(ns.get('MORE', {}))

if __name__ == '__main__':
    load_more()
    done = set()
    if os.path.exists(C.EXP_CSV):
        df = pd.read_csv(C.EXP_CSV, encoding='utf-8-sig'); done = set(df[df.variant == 'pub'].exp_id)
    D = prep(C.get_D())
    _gw = C.get_wide
    def gw(Dx, mt):
        w = _gw(Dx, mt)
        if 'ros_sci' not in w: w = prep(w); C._WIDE[mt] = w
        return w
    C.get_wide = gw
    for e in a.exps:
        if e in done and not a.force: print('건너뜀', e); continue
        change, f, note = EXPS[e]
        cfg = f(C.load_champ()['cfg'])
        C.challenge(e, change, cfg, note, decide=not a.info, D=D)
